//! Replay contracts and conservative unavailable-state handling.
//!
//! A real backend must implement the same interface in an isolated SVM/LiteSVM
//! worker. The default engine never executes untrusted program bytes.

use std::fmt;

use serde::{Deserialize, Serialize};
use sos_solana::SemanticAnalysis;
use sos_types::{
    AccountSnapshot, CandidateObservation, EvidenceStatus, ProgramArtifact, TransactionEvidence,
};

#[cfg(feature = "litesvm")]
use {
    base64::Engine,
    litesvm::LiteSVM,
    solana_account::{Account, ReadableAccount},
    solana_address::Address,
    solana_pubkey::Pubkey,
    solana_transaction::versioned::VersionedTransaction,
    std::str::FromStr,
};

pub const TRACE_SCHEMA_VERSION: u32 = 2;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TraceStatus {
    Success,
    Failure,
    NotAvailable,
    Unverified,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AccountDiff {
    pub address: String,
    pub lamports_before: Option<u64>,
    pub lamports_after: Option<u64>,
    pub owner_before: Option<String>,
    pub owner_after: Option<String>,
    pub data_changed: Option<bool>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CpiTrace {
    pub target_program: Option<String>,
    pub instruction_data_base64: Option<String>,
    pub status: TraceStatus,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AccountStateProvenance {
    pub address: String,
    pub context_slot: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecutionTrace {
    pub schema_version: u32,
    pub transaction_signature: String,
    pub status: TraceStatus,
    pub error: Option<String>,
    pub logs: Vec<String>,
    pub compute_units: Option<u64>,
    pub cpis: Vec<CpiTrace>,
    pub account_diffs: Vec<AccountDiff>,
    pub return_data_base64: Option<String>,
    pub path_coverage: Vec<usize>,
    #[serde(default)]
    pub transaction_slot: Option<u64>,
    #[serde(default)]
    pub account_state: Vec<AccountStateProvenance>,
    pub limitations: Vec<String>,
    #[serde(default)]
    pub behavioral_comparison: Option<BehaviorComparison>,
}

impl ExecutionTrace {
    pub fn not_available(signature: impl Into<String>, reason: impl Into<String>) -> Self {
        Self {
            schema_version: TRACE_SCHEMA_VERSION,
            transaction_signature: signature.into(),
            status: TraceStatus::NotAvailable,
            error: None,
            logs: Vec::new(),
            compute_units: None,
            cpis: Vec::new(),
            account_diffs: Vec::new(),
            return_data_base64: None,
            path_coverage: Vec::new(),
            transaction_slot: None,
            account_state: Vec::new(),
            limitations: vec![reason.into()],
            behavioral_comparison: None,
        }
    }

    pub fn not_available_for_transaction(
        transaction: &TransactionEvidence,
        accounts: &[AccountSnapshot],
        reason: impl Into<String>,
    ) -> Self {
        let mut trace = Self::not_available(transaction.signature.clone(), reason);
        trace.transaction_slot = transaction.slot;
        trace.account_state = accounts
            .iter()
            .map(|snapshot| AccountStateProvenance {
                address: snapshot.address.clone(),
                context_slot: snapshot.context_slot,
            })
            .collect();
        trace
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BehaviorComparison {
    pub schema_version: u32,
    pub status: EvidenceStatus,
    pub matched_evidence: Vec<String>,
    pub mismatches: Vec<String>,
    pub limitations: Vec<String>,
}

pub const COMPARISON_SCHEMA_VERSION: u32 = 1;

/// Compare only observations available in the trace. This is intentionally not
/// a symbolic proof: an unexecuted branch remains unverified, and a successful
/// trace validates only the observed path.
pub fn compare_semantics(
    semantics: &SemanticAnalysis,
    trace: &ExecutionTrace,
) -> BehaviorComparison {
    if trace.status == TraceStatus::NotAvailable {
        return BehaviorComparison {
            schema_version: COMPARISON_SCHEMA_VERSION,
            status: EvidenceStatus::NotAvailable,
            matched_evidence: Vec::new(),
            mismatches: Vec::new(),
            limitations: trace.limitations.clone(),
        };
    }

    let mut matched_evidence = Vec::new();
    let mut mismatches = Vec::new();
    for candidate in &semantics.log_candidates {
        if trace.logs.iter().any(|log| log.contains(&candidate.value)) {
            matched_evidence.push(format!("log:0x{:x}", candidate.offset));
        }
    }
    if !semantics.cpi_candidates.is_empty() && trace.cpis.is_empty() {
        mismatches.push("static CPI candidates were not observed in this execution path".into());
    }
    if semantics.cpi_candidates.is_empty() && !trace.cpis.is_empty() {
        mismatches.push("trace contained CPI edges without a matching static CPI candidate".into());
    }
    if !semantics.cpi_candidates.is_empty() && !trace.cpis.is_empty() {
        matched_evidence.push(format!("cpi_count:{}", trace.cpis.len()));
    }

    let status = if trace.status == TraceStatus::Success && mismatches.is_empty() {
        EvidenceStatus::BehaviorallyValidated
    } else {
        EvidenceStatus::Unverified
    };
    BehaviorComparison {
        schema_version: COMPARISON_SCHEMA_VERSION,
        status,
        matched_evidence,
        mismatches,
        limitations: vec![
            "Validation is limited to the supplied transaction and account state.".into(),
            "Unexecuted branches and alternate account states remain unverified.".into(),
        ],
    }
}

/// Compare externally supplied behavior observations from one refinement
/// candidate with the supplied trace. This is deliberately a small contract:
/// it validates only observations that can be matched directly to trace data,
/// and never executes generated source code.
pub fn compare_candidate_observations(
    observations: &[CandidateObservation],
    trace: &ExecutionTrace,
) -> BehaviorComparison {
    if trace.status == TraceStatus::NotAvailable {
        return BehaviorComparison {
            schema_version: COMPARISON_SCHEMA_VERSION,
            status: EvidenceStatus::NotAvailable,
            matched_evidence: Vec::new(),
            mismatches: Vec::new(),
            limitations: trace.limitations.clone(),
        };
    }

    let mut matched_evidence = Vec::new();
    let mut mismatches = Vec::new();
    for observation in observations {
        let matched = match observation.kind.as_str() {
            "log_contains" => trace
                .logs
                .iter()
                .any(|log| log.contains(&observation.value)),
            "cpi_target" => trace
                .cpis
                .iter()
                .any(|cpi| cpi.target_program.as_deref() == Some(observation.value.as_str())),
            "status" => trace_status_name(&trace.status) == observation.value,
            "return_data_base64" => {
                trace.return_data_base64.as_deref() == Some(observation.value.as_str())
            }
            _ => false,
        };
        if matched {
            matched_evidence.push(format!(
                "candidate:{}:{}",
                observation.kind, observation.value
            ));
        } else {
            mismatches.push(format!(
                "candidate observation {}={:?} was not observed in the supplied trace",
                observation.kind, observation.value
            ));
        }
    }

    let status = if observations.is_empty() {
        EvidenceStatus::Unverified
    } else if mismatches.is_empty() {
        EvidenceStatus::BehaviorallyValidated
    } else {
        EvidenceStatus::Unverified
    };
    let mut limitations = vec![
        "candidate observations are compared with the supplied trace; generated source is not executed".into(),
        "unobserved branches and alternate account states remain unverified".into(),
    ];
    if observations.is_empty() {
        limitations
            .push("the candidate supplied no directly comparable behavior observations".into());
    }
    BehaviorComparison {
        schema_version: COMPARISON_SCHEMA_VERSION,
        status,
        matched_evidence,
        mismatches,
        limitations,
    }
}

fn trace_status_name(status: &TraceStatus) -> &'static str {
    match status {
        TraceStatus::Success => "success",
        TraceStatus::Failure => "failure",
        TraceStatus::NotAvailable => "not_available",
        TraceStatus::Unverified => "unverified",
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReplayError {
    Backend(String),
    InvalidInput(String),
}

impl fmt::Display for ReplayError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Backend(error) => write!(f, "replay backend error: {error}"),
            Self::InvalidInput(error) => write!(f, "invalid replay input: {error}"),
        }
    }
}

impl std::error::Error for ReplayError {}

pub trait ReplayEngine {
    fn replay(
        &self,
        artifact: &ProgramArtifact,
        transaction: &TransactionEvidence,
        accounts: &[AccountSnapshot],
    ) -> Result<ExecutionTrace, ReplayError>;
}

#[derive(Debug, Default, Clone, Copy)]
pub struct UnavailableReplayEngine;

impl ReplayEngine for UnavailableReplayEngine {
    fn replay(
        &self,
        artifact: &ProgramArtifact,
        transaction: &TransactionEvidence,
        accounts: &[AccountSnapshot],
    ) -> Result<ExecutionTrace, ReplayError> {
        Ok(ExecutionTrace::not_available_for_transaction(
            transaction,
            accounts,
            format!(
                "isolated SVM backend is not configured for artifact {}",
                artifact.program_address
            ),
        ))
    }
}

#[cfg(feature = "litesvm")]
#[derive(Debug, Default, Clone, Copy)]
pub struct LiteSvmReplayEngine;

#[cfg(feature = "litesvm")]
impl ReplayEngine for LiteSvmReplayEngine {
    fn replay(
        &self,
        artifact: &ProgramArtifact,
        transaction: &TransactionEvidence,
        accounts: &[AccountSnapshot],
    ) -> Result<ExecutionTrace, ReplayError> {
        let program_id = Address::from_str(&artifact.program_address).map_err(|error| {
            ReplayError::InvalidInput(format!("invalid program address: {error}"))
        })?;
        let transaction_bytes = transaction_bytes(transaction)?;
        let versioned_transaction: VersionedTransaction = bincode::deserialize(&transaction_bytes)
            .map_err(|error| {
                ReplayError::InvalidInput(format!("invalid serialized transaction: {error}"))
            })?;

        let mut svm = LiteSVM::new().with_sigverify(false);
        svm.add_program(program_id, &artifact.elf_bytes)
            .map_err(|error| ReplayError::Backend(error.to_string()))?;
        for snapshot in accounts {
            let address = Address::from_str(&snapshot.address).map_err(|error| {
                ReplayError::InvalidInput(format!("invalid account address: {error}"))
            })?;
            if address == program_id {
                continue;
            }
            if snapshot.executable && is_elf(&snapshot.data) {
                svm.add_program(address, &snapshot.data)
                    .map_err(|error| ReplayError::Backend(error.to_string()))?;
                continue;
            }
            let owner = Pubkey::from_str(&snapshot.owner).map_err(|error| {
                ReplayError::InvalidInput(format!("invalid account owner: {error}"))
            })?;
            svm.set_account(
                address,
                Account {
                    lamports: snapshot.lamports,
                    data: snapshot.data.clone(),
                    owner,
                    executable: snapshot.executable,
                    rent_epoch: 0,
                },
            )
            .map_err(|error| ReplayError::Backend(error.to_string()))?;
        }

        match svm.simulate_transaction(versioned_transaction) {
            Ok(result) => Ok(success_trace(
                transaction,
                accounts,
                result.meta.logs,
                result.meta.compute_units_consumed,
                result.meta.return_data.data,
                result.post_accounts,
                extract_cpis(&transaction.raw_json),
            )),
            Err(result) => Ok(failure_trace(
                transaction,
                accounts,
                format!("{:?}", result.err),
                result.meta.logs,
                result.meta.compute_units_consumed,
                result.meta.return_data.data,
                extract_cpis(&transaction.raw_json),
            )),
        }
    }
}

#[cfg(feature = "litesvm")]
fn is_elf(data: &[u8]) -> bool {
    data.len() >= 4 && data[..4] == [0x7f, b'E', b'L', b'F']
}

#[cfg(feature = "litesvm")]
fn transaction_bytes(transaction: &TransactionEvidence) -> Result<Vec<u8>, ReplayError> {
    let encoded = transaction
        .raw_json
        .get("raw_transaction")
        .and_then(|value| value.get("transaction"))
        .or_else(|| transaction.raw_json.get("transaction"))
        .and_then(|value| value.as_array())
        .and_then(|values| values.first())
        .and_then(|value| value.as_str())
        .ok_or_else(|| {
            ReplayError::InvalidInput(
                "RPC transaction evidence does not contain a base64 serialized transaction".into(),
            )
        })?;
    base64::engine::general_purpose::STANDARD
        .decode(encoded)
        .map_err(|error| ReplayError::InvalidInput(format!("invalid transaction base64: {error}")))
}

#[cfg(feature = "litesvm")]
fn success_trace(
    transaction: &TransactionEvidence,
    before: &[AccountSnapshot],
    logs: Vec<String>,
    compute_units: u64,
    return_data: Vec<u8>,
    post_accounts: Vec<(Address, solana_account::AccountSharedData)>,
    cpis: Vec<CpiTrace>,
) -> ExecutionTrace {
    ExecutionTrace {
        schema_version: TRACE_SCHEMA_VERSION,
        transaction_signature: transaction.signature.clone(),
        status: TraceStatus::Success,
        error: None,
        logs,
        compute_units: Some(compute_units),
        cpis,
        account_diffs: account_diffs(before, &post_accounts),
        return_data_base64: encode_optional(&return_data),
        path_coverage: Vec::new(),
        transaction_slot: transaction.slot,
        account_state: account_state_provenance(before),
        limitations: vec![
            "LiteSVM replay captures the supplied account state only.".into(),
            "This backend does not expose instruction-PC path coverage.".into(),
        ],
        behavioral_comparison: None,
    }
}

#[cfg(feature = "litesvm")]
fn failure_trace(
    transaction: &TransactionEvidence,
    before: &[AccountSnapshot],
    error: String,
    logs: Vec<String>,
    compute_units: u64,
    return_data: Vec<u8>,
    cpis: Vec<CpiTrace>,
) -> ExecutionTrace {
    ExecutionTrace {
        schema_version: TRACE_SCHEMA_VERSION,
        transaction_signature: transaction.signature.clone(),
        status: TraceStatus::Failure,
        error: Some(error),
        logs,
        compute_units: Some(compute_units),
        cpis,
        account_diffs: account_diffs(before, &[]),
        return_data_base64: encode_optional(&return_data),
        path_coverage: Vec::new(),
        transaction_slot: transaction.slot,
        account_state: account_state_provenance(before),
        limitations: vec![
            "Failed replay does not expose post-state from LiteSVM.".into(),
            "The trace is limited to the supplied account state.".into(),
            "This backend does not expose instruction-PC path coverage.".into(),
        ],
        behavioral_comparison: None,
    }
}

#[cfg(feature = "litesvm")]
fn account_state_provenance(before: &[AccountSnapshot]) -> Vec<AccountStateProvenance> {
    before
        .iter()
        .map(|snapshot| AccountStateProvenance {
            address: snapshot.address.clone(),
            context_slot: snapshot.context_slot,
        })
        .collect()
}

#[cfg(feature = "litesvm")]
fn account_diffs(
    before: &[AccountSnapshot],
    after: &[(Address, solana_account::AccountSharedData)],
) -> Vec<AccountDiff> {
    before
        .iter()
        .filter_map(|snapshot| {
            let address = Address::from_str(&snapshot.address).ok()?;
            let (_, account) = after.iter().find(|(candidate, _)| *candidate == address)?;
            let data_changed = snapshot.data != account.data();
            let owner_after = account.owner().to_string();
            let changed = data_changed
                || snapshot.lamports != account.lamports()
                || snapshot.owner != owner_after;
            changed.then(|| AccountDiff {
                address: snapshot.address.clone(),
                lamports_before: Some(snapshot.lamports),
                lamports_after: Some(account.lamports()),
                owner_before: Some(snapshot.owner.clone()),
                owner_after: Some(owner_after),
                data_changed: Some(data_changed),
            })
        })
        .collect()
}

#[cfg(feature = "litesvm")]
fn encode_optional(data: &[u8]) -> Option<String> {
    (!data.is_empty()).then(|| base64::engine::general_purpose::STANDARD.encode(data))
}

#[cfg(feature = "litesvm")]
fn extract_cpis(raw_json: &serde_json::Value) -> Vec<CpiTrace> {
    raw_json
        .get("meta")
        .and_then(|meta| meta.get("innerInstructions"))
        .and_then(|value| value.as_array())
        .into_iter()
        .flatten()
        .flat_map(|group| group.get("instructions"))
        .filter_map(|value| value.as_array())
        .flatten()
        .map(|instruction| CpiTrace {
            target_program: instruction
                .get("programId")
                .and_then(|value| value.as_str())
                .map(str::to_owned),
            instruction_data_base64: instruction
                .get("data")
                .and_then(|value| value.as_str())
                .map(str::to_owned),
            status: TraceStatus::Success,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use sos_types::{CandidateObservation, EvidenceRef, LoaderKind};

    #[test]
    fn unavailable_engine_preserves_trace_provenance() {
        let artifact = ProgramArtifact {
            program_address: "program".into(),
            loader: LoaderKind::Unknown,
            programdata_address: None,
            deployment_slot: None,
            upgrade_authority: None,
            raw_account_data: Vec::new(),
            elf_bytes: Vec::new(),
            elf_offset: 0,
        };
        let transaction = TransactionEvidence {
            signature: "signature".into(),
            slot: Some(7),
            raw_json: serde_json::json!({"slot": 7}),
        };
        let account = AccountSnapshot {
            address: "account".into(),
            owner: "owner".into(),
            executable: false,
            lamports: 1,
            data: Vec::new(),
            context_slot: Some(7),
        };
        let trace = UnavailableReplayEngine
            .replay(&artifact, &transaction, &[account])
            .unwrap();
        assert_eq!(trace.status, TraceStatus::NotAvailable);
        assert_eq!(trace.transaction_signature, "signature");
        assert_eq!(trace.transaction_slot, Some(7));
        assert_eq!(trace.account_state[0].address, "account");
        assert_eq!(trace.account_state[0].context_slot, Some(7));
        assert!(!trace.limitations.is_empty());
    }

    #[test]
    fn candidate_observations_match_only_direct_trace_evidence() {
        let trace = ExecutionTrace {
            schema_version: TRACE_SCHEMA_VERSION,
            transaction_signature: "signature".into(),
            status: TraceStatus::Success,
            error: None,
            logs: vec!["Program log: SOS hello world".into()],
            compute_units: Some(42),
            cpis: Vec::new(),
            account_diffs: Vec::new(),
            return_data_base64: None,
            path_coverage: vec![0],
            transaction_slot: Some(7),
            account_state: Vec::new(),
            limitations: Vec::new(),
            behavioral_comparison: None,
        };
        let observations = vec![CandidateObservation {
            kind: "log_contains".into(),
            value: "SOS hello world".into(),
            evidence: vec![EvidenceRef {
                source: "ir".into(),
                location: "function:fn_0".into(),
                detail: None,
            }],
        }];
        let comparison = compare_candidate_observations(&observations, &trace);
        assert_eq!(comparison.status, EvidenceStatus::BehaviorallyValidated);
        assert!(comparison.mismatches.is_empty());
        assert_eq!(comparison.matched_evidence.len(), 1);
    }

    #[test]
    fn candidate_observation_mismatch_is_not_validation() {
        let trace = ExecutionTrace {
            schema_version: TRACE_SCHEMA_VERSION,
            transaction_signature: "signature".into(),
            status: TraceStatus::Success,
            error: None,
            logs: vec!["Program log: observed".into()],
            compute_units: None,
            cpis: Vec::new(),
            account_diffs: Vec::new(),
            return_data_base64: None,
            path_coverage: Vec::new(),
            transaction_slot: None,
            account_state: Vec::new(),
            limitations: Vec::new(),
            behavioral_comparison: None,
        };
        let observations = vec![CandidateObservation {
            kind: "log_contains".into(),
            value: "missing".into(),
            evidence: vec![],
        }];
        let comparison = compare_candidate_observations(&observations, &trace);
        assert_eq!(comparison.status, EvidenceStatus::Unverified);
        assert_eq!(comparison.mismatches.len(), 1);
    }
}
