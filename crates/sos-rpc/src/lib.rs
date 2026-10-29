//! Provider-neutral RPC contracts. Concrete JSON-RPC clients belong here later.

use std::{fmt, io::Read, thread, time::Duration};

use base64::Engine;
use serde_json::{json, Value};
use sos_types::{AccountSnapshot, SignatureInfo, TransactionEvidence};

pub trait ChainProvider {
    fn get_account(&self, address: &str, slot: Option<u64>) -> Result<AccountSnapshot, RpcError>;

    fn get_signatures(&self, program: &str, limit: usize) -> Result<Vec<SignatureInfo>, RpcError>;

    fn get_transaction(
        &self,
        signature: &str,
        slot: Option<u64>,
    ) -> Result<TransactionEvidence, RpcError>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RpcError {
    InvalidAddress(String),
    NotFound(String),
    RateLimited,
    HistoricalStateUnavailable { address: String, slot: u64 },
    Transport(String),
    Response(String),
    ResponseTooLarge { limit: usize },
}

impl fmt::Display for RpcError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidAddress(address) => write!(f, "invalid address: {address}"),
            Self::NotFound(value) => write!(f, "not found: {value}"),
            Self::RateLimited => write!(f, "RPC rate limited"),
            Self::HistoricalStateUnavailable { address, slot } => {
                write!(
                    f,
                    "historical state unavailable for {address} at slot {slot}"
                )
            }
            Self::Transport(error) => write!(f, "RPC transport error: {error}"),
            Self::Response(error) => write!(f, "RPC response error: {error}"),
            Self::ResponseTooLarge { limit } => {
                write!(f, "RPC response exceeds the {limit}-byte limit")
            }
        }
    }
}

impl std::error::Error for RpcError {}

pub struct JsonRpcProvider {
    endpoint: String,
    commitment: String,
    agent: ureq::Agent,
    max_attempts: usize,
    max_response_bytes: usize,
    retry_base_delay: Duration,
}

impl JsonRpcProvider {
    pub fn new(endpoint: impl Into<String>) -> Self {
        let config = ureq::AgentBuilder::new()
            .timeout(Duration::from_secs(30))
            .build();
        Self {
            endpoint: endpoint.into(),
            commitment: "finalized".into(),
            agent: config,
            max_attempts: 3,
            max_response_bytes: 16 * 1024 * 1024,
            retry_base_delay: Duration::from_millis(250),
        }
    }

    pub fn with_commitment(mut self, commitment: impl Into<String>) -> Self {
        self.commitment = commitment.into();
        self
    }

    pub fn with_max_attempts(mut self, max_attempts: usize) -> Self {
        self.max_attempts = max_attempts.max(1);
        self
    }

    pub fn with_max_response_bytes(mut self, max_response_bytes: usize) -> Self {
        self.max_response_bytes = max_response_bytes.max(1);
        self
    }

    fn call(&self, method: &str, params: Value) -> Result<Value, RpcError> {
        let request = json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params,
        });
        let request_body = request.to_string();
        let mut last_rate_limit = false;
        for attempt in 0..self.max_attempts {
            match self
                .agent
                .post(&self.endpoint)
                .set("Content-Type", "application/json")
                .send_string(&request_body)
            {
                Ok(response) => {
                    let mut bytes = Vec::new();
                    response
                        .into_reader()
                        .take((self.max_response_bytes as u64).saturating_add(1))
                        .read_to_end(&mut bytes)
                        .map_err(|error| RpcError::Transport(error.to_string()))?;
                    if bytes.len() > self.max_response_bytes {
                        return Err(RpcError::ResponseTooLarge {
                            limit: self.max_response_bytes,
                        });
                    }
                    let body: Value = serde_json::from_slice(&bytes)
                        .map_err(|error| RpcError::Response(error.to_string()))?;
                    if let Some(error) = body.get("error") {
                        let message = error
                            .get("message")
                            .and_then(Value::as_str)
                            .unwrap_or("unknown RPC error");
                        if is_rate_limited_message(message) {
                            last_rate_limit = true;
                        } else {
                            return Err(RpcError::Response(message.into()));
                        }
                    } else {
                        return body
                            .get("result")
                            .cloned()
                            .ok_or_else(|| RpcError::Response("missing JSON-RPC result".into()));
                    }
                }
                Err(error) => {
                    let rate_limited = matches!(&error, ureq::Error::Status(429, _))
                        || is_rate_limited_message(&error.to_string());
                    if !rate_limited {
                        return Err(RpcError::Transport(error.to_string()));
                    }
                    last_rate_limit = true;
                }
            }

            if attempt + 1 < self.max_attempts {
                let multiplier = 1u32 << attempt.min(10);
                thread::sleep(self.retry_base_delay.saturating_mul(multiplier));
            }
        }

        if last_rate_limit {
            Err(RpcError::RateLimited)
        } else {
            Err(RpcError::Response(
                "RPC call failed without a response".into(),
            ))
        }
    }

    fn validate_address(address: &str) -> Result<(), RpcError> {
        let decoded = bs58::decode(address)
            .into_vec()
            .map_err(|_| RpcError::InvalidAddress(address.into()))?;
        if decoded.len() != 32 {
            return Err(RpcError::InvalidAddress(address.into()));
        }
        Ok(())
    }
}

fn is_rate_limited_message(message: &str) -> bool {
    let message = message.to_ascii_lowercase();
    message.contains("rate limit")
        || message.contains("too many requests")
        || message.contains("429")
}

impl ChainProvider for JsonRpcProvider {
    fn get_account(&self, address: &str, slot: Option<u64>) -> Result<AccountSnapshot, RpcError> {
        Self::validate_address(address)?;
        let mut config = json!({
            "encoding": "base64",
            "commitment": self.commitment,
        });
        if let Some(slot) = slot {
            config["minContextSlot"] = json!(slot);
        }
        let result = self.call("getAccountInfo", json!([address, config]))?;
        let context_slot = result
            .get("context")
            .and_then(|value| value.get("slot"))
            .and_then(Value::as_u64);
        if let Some(requested_slot) = slot {
            if context_slot != Some(requested_slot) {
                return Err(RpcError::HistoricalStateUnavailable {
                    address: address.into(),
                    slot: requested_slot,
                });
            }
        }
        let value = result
            .get("value")
            .ok_or_else(|| RpcError::Response("missing account value".into()))?;
        let Some(value) = value.as_object() else {
            return Err(RpcError::NotFound(address.into()));
        };
        let data = value
            .get("data")
            .and_then(Value::as_array)
            .ok_or_else(|| RpcError::Response("account data is not an encoded tuple".into()))?;
        let encoded = data
            .first()
            .and_then(Value::as_str)
            .ok_or_else(|| RpcError::Response("missing base64 account data".into()))?;
        let data = base64::engine::general_purpose::STANDARD
            .decode(encoded)
            .map_err(|error| RpcError::Response(format!("invalid base64 account data: {error}")))?;
        let owner = value
            .get("owner")
            .and_then(Value::as_str)
            .ok_or_else(|| RpcError::Response("missing account owner".into()))?;
        let executable = value
            .get("executable")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let lamports = value
            .get("lamports")
            .and_then(Value::as_u64)
            .unwrap_or_default();
        Ok(AccountSnapshot {
            address: address.into(),
            owner: owner.into(),
            executable,
            lamports,
            data,
            context_slot,
        })
    }

    fn get_signatures(&self, program: &str, limit: usize) -> Result<Vec<SignatureInfo>, RpcError> {
        Self::validate_address(program)?;
        let result = self.call(
            "getSignaturesForAddress",
            json!([program, {"limit": limit}]),
        )?;
        let entries = result
            .as_array()
            .ok_or_else(|| RpcError::Response("signature response is not an array".into()))?;
        entries
            .iter()
            .map(|entry| {
                Ok(SignatureInfo {
                    signature: entry
                        .get("signature")
                        .and_then(Value::as_str)
                        .ok_or_else(|| RpcError::Response("missing signature".into()))?
                        .into(),
                    slot: entry.get("slot").and_then(Value::as_u64),
                    err: entry
                        .get("err")
                        .filter(|value| !value.is_null())
                        .map(Value::to_string),
                })
            })
            .collect()
    }

    fn get_transaction(
        &self,
        signature: &str,
        slot: Option<u64>,
    ) -> Result<TransactionEvidence, RpcError> {
        let result = self.call(
            "getTransaction",
            json!([signature, {
                "encoding": "jsonParsed",
                "maxSupportedTransactionVersion": 0,
                "commitment": self.commitment,
            }]),
        )?;
        let actual_slot = result.get("slot").and_then(Value::as_u64);
        if let Some(expected) = slot {
            if actual_slot != Some(expected) {
                return Err(RpcError::HistoricalStateUnavailable {
                    address: signature.into(),
                    slot: expected,
                });
            }
        }
        let raw_transaction = self
            .call(
                "getTransaction",
                json!([signature, {
                    "encoding": "base64",
                    "maxSupportedTransactionVersion": 0,
                    "commitment": self.commitment,
                }]),
            )
            .ok();
        let mut evidence_json = result;
        if let Some(raw_transaction) = raw_transaction {
            evidence_json["raw_transaction"] = raw_transaction;
        }
        Ok(TransactionEvidence {
            signature: signature.into(),
            slot: actual_slot,
            raw_json: evidence_json,
        })
    }
}
