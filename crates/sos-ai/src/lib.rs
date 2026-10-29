//! Provider-independent, evidence-constrained AI gateway contracts.
//!
//! This crate keeps provider-specific transport behind `ModelProvider`. Local
//! subprocess inference and a bounded generic HTTP provider are available; the
//! gateway remains responsible for schema, evidence, and resource validation.

use std::{
    collections::BTreeSet,
    fmt, fs,
    io::{Read, Write},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use sos_analysis::{build_canonical_ir, CanonicalIr, StaticAnalysis};
use sos_solana::SemanticAnalysis;
use sos_types::{EvidenceItem, EvidenceStatus};

mod ollama;

pub use ollama::{trusted_open_model, OllamaModelProvider, OpenModelProfile, TRUSTED_OPEN_MODELS};

pub const AI_SCHEMA_VERSION: u32 = 1;
pub const AI_BUNDLE_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelTask {
    StructureRecovery,
    SemanticLabeling,
    Explanation,
    Refinement,
}

impl ModelTask {
    pub const ALL: [Self; 4] = [
        Self::StructureRecovery,
        Self::SemanticLabeling,
        Self::Explanation,
        Self::Refinement,
    ];
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AiStageStatus {
    Accepted,
    NotAvailable,
    Rejected,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AiStageResult {
    pub task: ModelTask,
    pub status: AiStageStatus,
    pub model: String,
    pub prompt_version: String,
    pub input_sha256: String,
    pub output_sha256: Option<String>,
    pub response: Option<ModelResponse>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AiPipelineBundle {
    pub schema_version: u32,
    pub artifact_sha256: Option<String>,
    pub provider: String,
    pub model: String,
    pub stages: Vec<AiStageResult>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AiPipelineConfig {
    pub provider_name: String,
    pub model: String,
    pub prompt_version: String,
    pub artifact_sha256: Option<String>,
    pub limits: EvidencePackLimits,
    pub max_attempts: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StructureRecoveryOutput {
    pub functions: Vec<RecoveredFunction>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RecoveredFunction {
    pub function_id: String,
    pub pseudocode: String,
    #[serde(default)]
    pub evidence: Vec<sos_types::EvidenceRef>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SemanticLabelingOutput {
    pub labels: Vec<SemanticLabel>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SemanticLabel {
    pub kind: String,
    pub value: serde_json::Value,
    pub confidence: Option<f32>,
    pub status: EvidenceStatus,
    #[serde(default)]
    pub evidence: Vec<sos_types::EvidenceRef>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExplanationOutput {
    pub summary: String,
    pub observations: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RefinementOutput {
    #[serde(default = "default_refinement_schema_version")]
    pub schema_version: u32,
    pub patches: Vec<RefinementPatch>,
    #[serde(default)]
    pub observations: Vec<sos_types::CandidateObservation>,
    pub validation: RefinementValidation,
}

pub const REFINEMENT_SCHEMA_VERSION: u32 = 1;

fn default_refinement_schema_version() -> u32 {
    REFINEMENT_SCHEMA_VERSION
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RefinementPatch {
    pub function_id: String,
    pub replacement: String,
    #[serde(default)]
    pub evidence: Vec<sos_types::EvidenceRef>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RefinementValidation {
    pub status: String,
    pub notes: Vec<String>,
}

pub const VALIDATION_SCHEMA_VERSION: u32 = 1;
pub const VALIDATION_BUNDLE_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompilationStatus {
    Passed,
    Failed,
    TimedOut,
    NotAvailable,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompilationResult {
    pub schema_version: u32,
    pub status: CompilationStatus,
    pub compiler: Option<String>,
    pub source_sha256: String,
    pub diagnostics: Vec<String>,
    pub limitations: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UntrustedEvidence {
    pub source: String,
    pub content: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvidencePack {
    pub schema_version: u32,
    pub task: ModelTask,
    pub artifact_sha256: Option<String>,
    pub analysis: StaticAnalysis,
    pub ir: CanonicalIr,
    pub semantics: SemanticAnalysis,
    pub untrusted_evidence: Vec<UntrustedEvidence>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidencePackLimits {
    pub max_instructions: usize,
    pub max_functions: usize,
    pub max_basic_blocks: usize,
    pub max_memory_accesses: usize,
    pub max_stack_accesses: usize,
    pub max_register_uses: usize,
    pub max_strings: usize,
    pub max_public_keys: usize,
    pub max_constants: usize,
    pub max_relocations: usize,
    pub max_sccs: usize,
    pub max_dataflow_edges: usize,
    pub max_ir_operations: usize,
    pub max_semantic_items: usize,
    pub max_evidence_items: usize,
}

impl Default for EvidencePackLimits {
    fn default() -> Self {
        Self {
            max_instructions: 512,
            max_functions: 128,
            max_basic_blocks: 256,
            max_memory_accesses: 512,
            max_stack_accesses: 512,
            max_register_uses: 512,
            max_strings: 128,
            max_public_keys: 128,
            max_constants: 256,
            max_relocations: 512,
            max_sccs: 256,
            max_dataflow_edges: 1_024,
            max_ir_operations: 2_048,
            max_semantic_items: 256,
            max_evidence_items: 512,
        }
    }
}

impl EvidencePack {
    pub fn from_analysis(
        task: ModelTask,
        artifact_sha256: Option<String>,
        analysis: StaticAnalysis,
        semantics: SemanticAnalysis,
    ) -> Self {
        Self {
            schema_version: AI_SCHEMA_VERSION,
            task,
            artifact_sha256,
            ir: build_canonical_ir(&analysis),
            analysis,
            semantics,
            untrusted_evidence: Vec::new(),
        }
    }

    pub fn with_untrusted_evidence(mut self, evidence: Vec<UntrustedEvidence>) -> Self {
        self.untrusted_evidence = evidence;
        self
    }

    /// Build a bounded model view while preserving the full deterministic analysis elsewhere.
    /// Instruction PCs mentioned by semantic candidates are prioritized; remaining capacity is
    /// filled in bytecode order so the model receives both relevant evidence and context.
    pub fn bounded_from_analysis(
        task: ModelTask,
        artifact_sha256: Option<String>,
        analysis: &StaticAnalysis,
        semantics: &SemanticAnalysis,
        limits: EvidencePackLimits,
    ) -> Self {
        let mut interesting_pcs = BTreeSet::new();
        for pc in semantics
            .syscalls
            .iter()
            .map(|item| item.instruction_pc)
            .chain(
                semantics
                    .dispatch_candidates
                    .iter()
                    .map(|item| item.instruction_pc),
            )
            .chain(
                semantics
                    .cpi_candidates
                    .iter()
                    .map(|item| item.instruction_pc),
            )
            .chain(
                semantics
                    .pda_candidates
                    .iter()
                    .map(|item| item.instruction_pc),
            )
        {
            interesting_pcs.insert(pc);
            for neighbor in pc.saturating_sub(2)..=pc.saturating_add(2) {
                interesting_pcs.insert(neighbor);
            }
        }

        let mut selected_pcs = BTreeSet::new();
        for instruction in &analysis.instructions {
            if interesting_pcs.contains(&instruction.pc)
                && selected_pcs.len() < limits.max_instructions
            {
                selected_pcs.insert(instruction.pc);
            }
        }
        for instruction in &analysis.instructions {
            if selected_pcs.len() >= limits.max_instructions {
                break;
            }
            selected_pcs.insert(instruction.pc);
        }

        let mut bounded_analysis = analysis.clone();
        bounded_analysis.instructions = analysis
            .instructions
            .iter()
            .filter(|item| selected_pcs.contains(&item.pc))
            .take(limits.max_instructions)
            .cloned()
            .collect();
        bounded_analysis.functions = analysis
            .functions
            .iter()
            .take(limits.max_functions)
            .cloned()
            .collect();
        bounded_analysis.basic_blocks = analysis
            .basic_blocks
            .iter()
            .take(limits.max_basic_blocks)
            .cloned()
            .collect();
        bounded_analysis.memory_accesses = analysis
            .memory_accesses
            .iter()
            .filter(|item| selected_pcs.contains(&item.pc))
            .take(limits.max_memory_accesses)
            .cloned()
            .collect();
        bounded_analysis.stack_accesses = analysis
            .stack_accesses
            .iter()
            .filter(|item| selected_pcs.contains(&item.pc))
            .take(limits.max_stack_accesses)
            .cloned()
            .collect();
        bounded_analysis.register_uses = analysis
            .register_uses
            .iter()
            .filter(|item| selected_pcs.contains(&item.pc))
            .take(limits.max_register_uses)
            .cloned()
            .collect();
        bounded_analysis.calls = analysis
            .calls
            .iter()
            .filter(|item| selected_pcs.contains(&item.call_pc))
            .take(limits.max_semantic_items)
            .cloned()
            .collect();
        bounded_analysis.strings = analysis
            .strings
            .iter()
            .take(limits.max_strings)
            .cloned()
            .collect();
        bounded_analysis.public_key_candidates = analysis
            .public_key_candidates
            .iter()
            .take(limits.max_public_keys)
            .cloned()
            .collect();
        bounded_analysis.immediate_constants = analysis
            .immediate_constants
            .iter()
            .take(limits.max_constants)
            .cloned()
            .collect();
        bounded_analysis.relocations = analysis
            .relocations
            .iter()
            .take(limits.max_relocations)
            .cloned()
            .collect();
        bounded_analysis.sccs = analysis
            .sccs
            .iter()
            .take(limits.max_sccs)
            .cloned()
            .collect();
        bounded_analysis.dataflow_edges = analysis
            .dataflow_edges
            .iter()
            .take(limits.max_dataflow_edges)
            .cloned()
            .collect();

        let mut bounded_semantics = semantics.clone();
        bounded_semantics.syscalls = semantics
            .syscalls
            .iter()
            .take(limits.max_semantic_items)
            .cloned()
            .collect();
        bounded_semantics.dispatch_candidates = semantics
            .dispatch_candidates
            .iter()
            .take(limits.max_semantic_items)
            .cloned()
            .collect();
        bounded_semantics.instruction_candidates = semantics
            .instruction_candidates
            .iter()
            .take(limits.max_semantic_items)
            .cloned()
            .collect();
        bounded_semantics.argument_candidates = semantics
            .argument_candidates
            .iter()
            .take(limits.max_semantic_items)
            .cloned()
            .collect();
        bounded_semantics.cpi_candidates = semantics
            .cpi_candidates
            .iter()
            .take(limits.max_semantic_items)
            .cloned()
            .collect();
        bounded_semantics.pda_candidates = semantics
            .pda_candidates
            .iter()
            .take(limits.max_semantic_items)
            .cloned()
            .collect();
        bounded_semantics.framework_candidates = semantics
            .framework_candidates
            .iter()
            .take(limits.max_semantic_items)
            .cloned()
            .collect();
        bounded_semantics.input_field_candidates = semantics
            .input_field_candidates
            .iter()
            .take(limits.max_semantic_items)
            .cloned()
            .collect();
        bounded_semantics.account_candidates = semantics
            .account_candidates
            .iter()
            .take(limits.max_semantic_items)
            .cloned()
            .collect();
        bounded_semantics.account_field_candidates = semantics
            .account_field_candidates
            .iter()
            .take(limits.max_semantic_items)
            .cloned()
            .collect();
        bounded_semantics.string_candidates = semantics
            .string_candidates
            .iter()
            .take(limits.max_strings)
            .cloned()
            .collect();
        bounded_semantics.constant_candidates = semantics
            .constant_candidates
            .iter()
            .take(limits.max_constants)
            .cloned()
            .collect();
        bounded_semantics.protocol_fingerprints = semantics
            .protocol_fingerprints
            .iter()
            .take(limits.max_semantic_items)
            .cloned()
            .collect();
        bounded_semantics.validation_checks = semantics
            .validation_checks
            .iter()
            .take(limits.max_semantic_items)
            .cloned()
            .collect();
        bounded_semantics.error_candidates = semantics
            .error_candidates
            .iter()
            .take(limits.max_strings)
            .cloned()
            .collect();
        bounded_semantics.log_candidates = semantics
            .log_candidates
            .iter()
            .take(limits.max_strings)
            .cloned()
            .collect();
        bounded_semantics.resolved_log_calls = semantics
            .resolved_log_calls
            .iter()
            .take(limits.max_semantic_items)
            .cloned()
            .collect();
        bounded_semantics.evidence = semantics
            .evidence
            .iter()
            .take(limits.max_evidence_items)
            .cloned()
            .collect();

        let mut pack =
            Self::from_analysis(task, artifact_sha256, bounded_analysis, bounded_semantics);
        // Malformed or heavily overlapping CFG ranges can cause the canonical
        // projection to repeat an instruction in multiple blocks. Keep the
        // model-facing IR explicitly bounded even when deterministic storage
        // retains the complete projection.
        pack.ir.operations.truncate(limits.max_ir_operations);
        pack.ir.blocks.truncate(limits.max_basic_blocks);
        pack.ir.functions.truncate(limits.max_functions);
        pack.ir.sccs.truncate(limits.max_sccs);
        pack
    }
}

/// Produce a deterministic one-function analysis slice. Global read-only data
/// remains available because strings and public keys are commonly referenced
/// through relocations, while executable facts are restricted to the target's
/// half-open PC range. This is the unit used by local semantic lifting.
pub fn scope_analysis_to_function(
    analysis: &StaticAnalysis,
    function_pc: usize,
) -> Option<StaticAnalysis> {
    let mut ordered = analysis.functions.clone();
    ordered.sort_by_key(|function| function.pc);
    let index = ordered
        .iter()
        .position(|function| function.pc == function_pc)?;
    let next_pc = ordered.get(index + 1).map(|function| function.pc);
    let in_range = |pc: usize| pc >= function_pc && next_pc.is_none_or(|next| pc < next);

    let mut scoped = analysis.clone();
    scoped.entrypoint = function_pc;
    scoped.functions = vec![ordered[index].clone()];
    scoped.instructions.retain(|item| in_range(item.pc));
    scoped.basic_blocks.retain(|item| in_range(item.start_pc));
    scoped.memory_accesses.retain(|item| in_range(item.pc));
    scoped.stack_accesses.retain(|item| in_range(item.pc));
    scoped.immediate_constants.retain(|item| in_range(item.pc));
    scoped.register_uses.retain(|item| in_range(item.pc));
    scoped.calls.retain(|item| in_range(item.call_pc));
    scoped
        .dataflow_edges
        .retain(|item| in_range(item.source_pc) && in_range(item.destination_pc));
    scoped.sccs.retain_mut(|item| {
        item.block_starts.retain(|pc| in_range(*pc));
        !item.block_starts.is_empty()
    });
    Some(scoped)
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelRequest {
    pub task: ModelTask,
    pub prompt_version: String,
    pub model: String,
    pub evidence_pack: EvidencePack,
    pub max_output_bytes: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelResponse {
    pub task: ModelTask,
    pub model: String,
    pub prompt_version: String,
    pub structured_output: serde_json::Value,
    pub evidence: Vec<EvidenceItem>,
}

pub trait ModelProvider {
    fn generate_structured(&self, request: ModelRequest) -> Result<ModelResponse, AiError>;
}

/// Provider-neutral HTTP adapter for a service implementing the SOS JSON
/// `ModelRequest`/`ModelResponse` protocol. It intentionally does not encode a
/// vendor API shape: vendor-specific clients can translate into this protocol
/// outside the deterministic gateway.
pub struct HttpModelProvider {
    endpoint: String,
    bearer_token: Option<String>,
    agent: ureq::Agent,
    max_response_bytes: usize,
}

impl HttpModelProvider {
    pub fn new(endpoint: impl Into<String>) -> Self {
        let agent = ureq::AgentBuilder::new()
            .timeout(Duration::from_secs(60))
            .build();
        Self {
            endpoint: endpoint.into(),
            bearer_token: None,
            agent,
            max_response_bytes: 1024 * 1024,
        }
    }

    pub fn with_bearer_token(mut self, token: impl Into<String>) -> Self {
        self.bearer_token = Some(token.into());
        self
    }

    pub fn with_max_response_bytes(mut self, limit: usize) -> Self {
        self.max_response_bytes = limit.max(1);
        self
    }
}

impl ModelProvider for HttpModelProvider {
    fn generate_structured(&self, request: ModelRequest) -> Result<ModelResponse, AiError> {
        let body = serde_json::to_string(&request)
            .map_err(|error| AiError::Provider(format!("request serialization failed: {error}")))?;
        let mut builder = self
            .agent
            .post(&self.endpoint)
            .set("Content-Type", "application/json")
            .set("Accept", "application/json");
        if let Some(token) = &self.bearer_token {
            builder = builder.set("Authorization", &format!("Bearer {token}"));
        }
        let response = builder
            .send_string(&body)
            .map_err(|error| AiError::Provider(format!("HTTP model request failed: {error}")))?;
        let mut output = Vec::new();
        response
            .into_reader()
            .take((self.max_response_bytes as u64).saturating_add(1))
            .read_to_end(&mut output)
            .map_err(|error| AiError::Provider(format!("HTTP response read failed: {error}")))?;
        if output.len() > self.max_response_bytes {
            return Err(AiError::OutputTooLarge {
                bytes: output.len(),
                limit: self.max_response_bytes,
            });
        }
        serde_json::from_slice(&output).map_err(|error| {
            AiError::InvalidResponse(format!("HTTP model JSON parse failed: {error}"))
        })
    }
}

/// Runs an explicitly configured local model adapter as a subprocess.
///
/// The adapter receives one JSON `ModelRequest` on stdin and must emit one JSON
/// `ModelResponse` on stdout. It is intentionally provider-neutral: llama.cpp,
/// Ollama wrappers, vLLM adapters, or a project-specific local runner can all
/// implement this small protocol without entering the deterministic core.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommandModelProvider {
    command: PathBuf,
    args: Vec<String>,
    timeout: Duration,
}

impl CommandModelProvider {
    pub fn new(command: impl AsRef<Path>) -> Self {
        Self {
            command: command.as_ref().to_owned(),
            args: Vec::new(),
            timeout: Duration::from_secs(60),
        }
    }

    pub fn with_args(mut self, args: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.args = args.into_iter().map(Into::into).collect();
        self
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout.max(Duration::from_millis(1));
        self
    }
}

impl ModelProvider for CommandModelProvider {
    fn generate_structured(&self, request: ModelRequest) -> Result<ModelResponse, AiError> {
        let input = serde_json::to_vec(&request)
            .map_err(|error| AiError::Provider(format!("request serialization failed: {error}")))?;
        let mut child = Command::new(&self.command)
            .args(&self.args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|error| AiError::Provider(format!("spawn failed: {error}")))?;
        let output_limit = request.max_output_bytes.saturating_add(1);
        let mut stdout = child
            .stdout
            .take()
            .ok_or_else(|| AiError::Provider("model stdout was unavailable".into()))?;
        let mut reader = Some(std::thread::spawn(move || {
            let mut output = Vec::new();
            let result = stdout
                .by_ref()
                .take(output_limit as u64)
                .read_to_end(&mut output);
            (output, result)
        }));
        child
            .stdin
            .take()
            .ok_or_else(|| AiError::Provider("model stdin was unavailable".into()))?
            .write_all(&input)
            .map_err(|error| {
                let _ = child.kill();
                let _ = child.wait();
                if let Some(reader) = reader.take() {
                    let _ = reader.join();
                }
                AiError::Provider(format!("stdin write failed: {error}"))
            })?;
        let deadline = Instant::now() + self.timeout;
        let status = loop {
            match child.try_wait() {
                Ok(Some(status)) => break status,
                Ok(None) if Instant::now() < deadline => thread::sleep(Duration::from_millis(10)),
                Ok(None) => {
                    let _ = child.kill();
                    let _ = child.wait();
                    if let Some(reader) = reader.take() {
                        let _ = reader.join();
                    }
                    return Err(AiError::Provider(format!(
                        "model process timed out after {} ms",
                        self.timeout.as_millis()
                    )));
                }
                Err(error) => {
                    let _ = child.kill();
                    let _ = child.wait();
                    if let Some(reader) = reader.take() {
                        let _ = reader.join();
                    }
                    return Err(AiError::Provider(format!(
                        "model process wait failed: {error}"
                    )));
                }
            }
        };
        let (output, read_result) = reader
            .take()
            .expect("stdout reader is present before completion")
            .join()
            .map_err(|_| AiError::Provider("model stdout reader panicked".into()))?;
        read_result.map_err(|error| AiError::Provider(format!("stdout read failed: {error}")))?;
        if output.len() > request.max_output_bytes {
            let _ = child.kill();
            let _ = child.wait();
            return Err(AiError::OutputTooLarge {
                bytes: output.len(),
                limit: request.max_output_bytes,
            });
        }
        if !status.success() {
            return Err(AiError::Provider(format!(
                "model exited with {}; stderr: {}",
                status, "stderr unavailable"
            )));
        }
        serde_json::from_slice(&output)
            .map_err(|error| AiError::InvalidResponse(format!("model JSON parse failed: {error}")))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AiError {
    EvidencePackTooLarge { bytes: usize, limit: usize },
    OutputTooLarge { bytes: usize, limit: usize },
    Provider(String),
    InvalidResponse(String),
}

impl fmt::Display for AiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EvidencePackTooLarge { bytes, limit } => {
                write!(f, "evidence pack is {bytes} bytes, exceeds limit {limit}")
            }
            Self::OutputTooLarge { bytes, limit } => {
                write!(f, "model output is {bytes} bytes, exceeds limit {limit}")
            }
            Self::Provider(error) => write!(f, "model provider error: {error}"),
            Self::InvalidResponse(error) => write!(f, "invalid model response: {error}"),
        }
    }
}

impl std::error::Error for AiError {}

pub struct ModelGateway<P> {
    provider: P,
    max_input_bytes: usize,
    max_output_bytes: usize,
}

impl<P> ModelGateway<P>
where
    P: ModelProvider,
{
    pub fn new(provider: P, max_input_bytes: usize, max_output_bytes: usize) -> Self {
        Self {
            provider,
            max_input_bytes,
            max_output_bytes,
        }
    }

    pub fn submit(
        &self,
        task: ModelTask,
        model: impl Into<String>,
        prompt_version: impl Into<String>,
        pack: EvidencePack,
    ) -> Result<ModelResponse, AiError> {
        let input_bytes = serde_json::to_vec(&pack)
            .map_err(|error| AiError::InvalidResponse(error.to_string()))?;
        if input_bytes.len() > self.max_input_bytes {
            return Err(AiError::EvidencePackTooLarge {
                bytes: input_bytes.len(),
                limit: self.max_input_bytes,
            });
        }
        let request = ModelRequest {
            task,
            prompt_version: prompt_version.into(),
            model: model.into(),
            evidence_pack: pack,
            max_output_bytes: self.max_output_bytes,
        };
        let expected_request = request.clone();
        let response = self.provider.generate_structured(request)?;
        let output_bytes = serde_json::to_vec(&response)
            .map_err(|error| AiError::InvalidResponse(error.to_string()))?;
        if output_bytes.len() > self.max_output_bytes {
            return Err(AiError::OutputTooLarge {
                bytes: output_bytes.len(),
                limit: self.max_output_bytes,
            });
        }
        validate_response_for_request(&expected_request, &response)?;
        Ok(response)
    }

    /// Retry provider and validation failures a bounded number of times. The
    /// evidence pack is cloned for each attempt so retries cannot mutate the
    /// request context or relax deterministic gateway checks.
    pub fn submit_with_retry(
        &self,
        task: ModelTask,
        model: impl Into<String>,
        prompt_version: impl Into<String>,
        pack: EvidencePack,
        max_attempts: usize,
    ) -> Result<ModelResponse, AiError> {
        let attempts = max_attempts.max(1);
        let model = model.into();
        let prompt_version = prompt_version.into();
        let mut last_error = None;
        for _ in 0..attempts {
            match self.submit(task, model.clone(), prompt_version.clone(), pack.clone()) {
                Ok(response) => return Ok(response),
                Err(error @ (AiError::Provider(_) | AiError::InvalidResponse(_))) => {
                    last_error = Some(error);
                }
                Err(error) => return Err(error),
            }
        }
        Err(last_error.expect("at least one retry attempt was executed"))
    }
}

/// Run the four bounded AI stages against one immutable deterministic evidence
/// snapshot. A provider or validation failure is recorded for that stage and
/// does not discard accepted results from earlier stages or break the
/// deterministic analysis bundle.
pub fn run_pipeline<P: ModelProvider>(
    provider: P,
    config: AiPipelineConfig,
    analysis: &StaticAnalysis,
    semantics: &SemanticAnalysis,
) -> AiPipelineBundle {
    let AiPipelineConfig {
        provider_name,
        model,
        prompt_version,
        artifact_sha256,
        limits,
        max_attempts,
    } = config;
    let gateway = ModelGateway::new(provider, 4 * 1024 * 1024, 1024 * 1024);
    let mut stages = Vec::with_capacity(ModelTask::ALL.len());
    let mut accepted_prior_stages = Vec::<UntrustedEvidence>::new();

    for task in ModelTask::ALL {
        let mut pack = EvidencePack::bounded_from_analysis(
            task,
            artifact_sha256.clone(),
            analysis,
            semantics,
            limits,
        );
        // Patina/Goonfi-style staged lifting: later stages receive accepted
        // earlier output as hypotheses while deterministic facts remain
        // immutable. The gateway still validates all evidence references.
        pack.untrusted_evidence
            .extend(accepted_prior_stages.iter().cloned());
        let input_sha256 = sha256_json(&pack);
        match gateway.submit_with_retry(
            task,
            model.clone(),
            prompt_version.clone(),
            pack,
            max_attempts,
        ) {
            Ok(response) => {
                let prior = serde_json::to_string(&response.structured_output)
                    .unwrap_or_else(|_| "{}".into())
                    .chars()
                    .take(128 * 1024)
                    .collect::<String>();
                accepted_prior_stages.push(UntrustedEvidence {
                    source: format!("accepted_model_stage:{task:?}"),
                    content: prior,
                });
                stages.push(AiStageResult {
                    task,
                    status: AiStageStatus::Accepted,
                    model: model.clone(),
                    prompt_version: prompt_version.clone(),
                    input_sha256,
                    output_sha256: Some(sha256_json(&response)),
                    response: Some(response),
                    error: None,
                })
            }
            Err(error) => {
                let status = if matches!(&error, AiError::Provider(_)) {
                    AiStageStatus::NotAvailable
                } else {
                    AiStageStatus::Rejected
                };
                stages.push(AiStageResult {
                    task,
                    status,
                    model: model.clone(),
                    prompt_version: prompt_version.clone(),
                    input_sha256,
                    output_sha256: None,
                    response: None,
                    error: Some(error.to_string()),
                });
            }
        }
    }

    AiPipelineBundle {
        schema_version: AI_BUNDLE_SCHEMA_VERSION,
        artifact_sha256,
        provider: provider_name,
        model,
        stages,
    }
}

/// Build the same auditable bundle when no provider is configured. This keeps
/// the deterministic inspect contract stable while making AI unavailability
/// explicit at every stage.
pub fn unavailable_pipeline_bundle(
    config: AiPipelineConfig,
    analysis: &StaticAnalysis,
    semantics: &SemanticAnalysis,
    reason: impl Into<String>,
) -> AiPipelineBundle {
    let AiPipelineConfig {
        provider_name,
        model,
        prompt_version,
        artifact_sha256,
        limits,
        ..
    } = config;
    let reason = reason.into();
    let stages = ModelTask::ALL
        .into_iter()
        .map(|task| {
            let pack = EvidencePack::bounded_from_analysis(
                task,
                artifact_sha256.clone(),
                analysis,
                semantics,
                limits,
            );
            AiStageResult {
                task,
                status: AiStageStatus::NotAvailable,
                model: model.clone(),
                prompt_version: prompt_version.clone(),
                input_sha256: sha256_json(&pack),
                output_sha256: None,
                response: None,
                error: Some(reason.clone()),
            }
        })
        .collect();
    AiPipelineBundle {
        schema_version: AI_BUNDLE_SCHEMA_VERSION,
        artifact_sha256,
        provider: provider_name,
        model,
        stages,
    }
}

fn sha256_json<T: Serialize>(value: &T) -> String {
    let bytes = serde_json::to_vec(value).unwrap_or_default();
    sha256_bytes(&bytes)
}

fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// Compile only the bounded refinement snippets supplied by a validated model
/// response. This is a syntax/type check, never execution of the candidate.
/// The compiler is invoked directly, with a temporary source/output path,
/// bounded diagnostics, and a wall-clock timeout.
pub fn compile_refinement_output(
    refinement: &RefinementOutput,
    compiler: Option<&Path>,
    timeout: Duration,
) -> CompilationResult {
    let replacements = refinement
        .patches
        .iter()
        .filter(|patch| !patch.replacement.trim().is_empty())
        .map(|patch| patch.replacement.as_str())
        .collect::<Vec<_>>();
    if replacements.is_empty() {
        return CompilationResult {
            schema_version: VALIDATION_SCHEMA_VERSION,
            status: CompilationStatus::NotAvailable,
            compiler: compiler.map(|path| path.display().to_string()),
            source_sha256: sha256_bytes(b""),
            diagnostics: Vec::new(),
            limitations: vec!["the refinement response contained no source patches".into()],
        };
    }

    let mut source = String::from(
        "#![allow(dead_code, unused_variables, unused_mut, non_snake_case)]\n\
         macro_rules! sos_raw_op { ($($arg:tt)*) => {}; }\n\
         fn unknown_return() {}\n\
         fn sos_unknown_return() {}\n\n",
    );
    for replacement in replacements {
        source.push_str(replacement);
        source.push_str("\n\n");
    }
    let source_sha256 = sha256_bytes(source.as_bytes());

    let Some(compiler) = compiler else {
        return CompilationResult {
            schema_version: VALIDATION_SCHEMA_VERSION,
            status: CompilationStatus::NotAvailable,
            compiler: None,
            source_sha256,
            diagnostics: Vec::new(),
            limitations: vec!["no Rust compiler was configured".into()],
        };
    };

    for forbidden in [
        "include!(",
        "include_str!(",
        "include_bytes!(",
        "env!(",
        "option_env!(",
        "proc_macro",
        "extern crate",
        "#[path",
    ] {
        if source.contains(forbidden) {
            return CompilationResult {
                schema_version: VALIDATION_SCHEMA_VERSION,
                status: CompilationStatus::Failed,
                compiler: Some(compiler.display().to_string()),
                source_sha256,
                diagnostics: vec![format!(
                    "candidate contains forbidden compilation directive `{forbidden}`"
                )],
                limitations: vec![
                    "candidate compilation rejects file inclusion, environment access, and external crates".into(),
                ],
            };
        }
    }

    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or_default();
    let base = std::env::temp_dir().join(format!("sos-refinement-{}-{stamp}", std::process::id()));
    let source_path = base.with_extension("rs");
    let output_path = base.with_extension("rmeta");
    if let Err(error) = fs::write(&source_path, source.as_bytes()) {
        return CompilationResult {
            schema_version: VALIDATION_SCHEMA_VERSION,
            status: CompilationStatus::Failed,
            compiler: Some(compiler.display().to_string()),
            source_sha256,
            diagnostics: vec![format!("unable to create compiler input: {error}")],
            limitations: Vec::new(),
        };
    }

    let mut child = match Command::new(compiler)
        .args(["--crate-type", "lib", "--emit", "metadata"])
        .arg("-o")
        .arg(&output_path)
        .arg(&source_path)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(child) => child,
        Err(error) => {
            let _ = fs::remove_file(&source_path);
            return CompilationResult {
                schema_version: VALIDATION_SCHEMA_VERSION,
                status: CompilationStatus::Failed,
                compiler: Some(compiler.display().to_string()),
                source_sha256,
                diagnostics: vec![format!("unable to start compiler: {error}")],
                limitations: Vec::new(),
            };
        }
    };
    let stdout_reader = child
        .stdout
        .take()
        .map(|reader| spawn_limited_reader(reader, 64 * 1024));
    let stderr_reader = child
        .stderr
        .take()
        .map(|reader| spawn_limited_reader(reader, 64 * 1024));
    let deadline = Instant::now() + timeout.max(Duration::from_millis(1));
    let status: Result<std::process::ExitStatus, String> = loop {
        match child.try_wait() {
            Ok(Some(status)) => break Ok(status),
            Ok(None) if Instant::now() < deadline => thread::sleep(Duration::from_millis(10)),
            Ok(None) => {
                let _ = child.kill();
                let _ = child.wait();
                let _ = join_reader(stdout_reader);
                let _ = join_reader(stderr_reader);
                let _ = fs::remove_file(&source_path);
                let _ = fs::remove_file(&output_path);
                return CompilationResult {
                    schema_version: VALIDATION_SCHEMA_VERSION,
                    status: CompilationStatus::TimedOut,
                    compiler: Some(compiler.display().to_string()),
                    source_sha256,
                    diagnostics: vec![format!(
                        "compiler timed out after {} ms",
                        timeout.as_millis()
                    )],
                    limitations: vec!["candidate compilation was terminated".into()],
                };
            }
            Err(error) => {
                let _ = child.kill();
                let _ = child.wait();
                let _ = join_reader(stdout_reader);
                let _ = join_reader(stderr_reader);
                let _ = fs::remove_file(&source_path);
                let _ = fs::remove_file(&output_path);
                return CompilationResult {
                    schema_version: VALIDATION_SCHEMA_VERSION,
                    status: CompilationStatus::Failed,
                    compiler: Some(compiler.display().to_string()),
                    source_sha256,
                    diagnostics: vec![format!("compiler wait failed: {error}")],
                    limitations: Vec::new(),
                };
            }
        }
    };
    let mut diagnostics = Vec::new();
    for reader in [stdout_reader, stderr_reader].into_iter().flatten() {
        match reader.join() {
            Ok(Ok(bytes)) => {
                let text = String::from_utf8_lossy(&bytes);
                diagnostics.extend(text.lines().map(str::to_owned));
            }
            Ok(Err(error)) => diagnostics.push(error),
            Err(_) => diagnostics.push("compiler output reader panicked".into()),
        }
    }
    let _ = fs::remove_file(&source_path);
    let _ = fs::remove_file(&output_path);
    let status = match status {
        Ok(status) if status.success() => CompilationStatus::Passed,
        Ok(_) => CompilationStatus::Failed,
        Err(error) => {
            diagnostics.push(error.to_string());
            CompilationStatus::Failed
        }
    };
    CompilationResult {
        schema_version: VALIDATION_SCHEMA_VERSION,
        status,
        compiler: Some(compiler.display().to_string()),
        source_sha256,
        diagnostics,
        limitations: vec![
            "compilation checks syntax and types only; it does not execute the candidate".into(),
        ],
    }
}

pub const REFINEMENT_LOOP_SCHEMA_VERSION: u32 = 1;

/// External validation facts supplied to a bounded repair pass. Compiler and
/// replay data are intentionally carried as tagged evidence, never as model
/// instructions or as a model-controlled validation status.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RefinementFeedback {
    pub compilation: CompilationStatus,
    pub compiler_diagnostics: Vec<String>,
    pub behavioral_status: String,
    pub mismatches: Vec<String>,
    pub limitations: Vec<String>,
}

impl RefinementFeedback {
    pub fn no_validation() -> Self {
        Self {
            compilation: CompilationStatus::NotAvailable,
            compiler_diagnostics: Vec::new(),
            behavioral_status: "not_requested".into(),
            mismatches: Vec::new(),
            limitations: vec!["no external validator was configured".into()],
        }
    }

    fn is_behaviorally_validated(&self) -> bool {
        self.compilation == CompilationStatus::Passed
            && self.behavioral_status == "behaviorally_validated"
            && self.mismatches.is_empty()
    }

    fn is_compiled_unverified(&self) -> bool {
        self.compilation == CompilationStatus::Passed
            && !self.is_behaviorally_validated()
            && self.mismatches.is_empty()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RefinementLoopStatus {
    BehaviorallyValidated,
    CompiledUnverified,
    Exhausted,
    NotAvailable,
    Rejected,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RefinementRound {
    pub round: usize,
    pub input_sha256: String,
    pub output_sha256: Option<String>,
    pub response: Option<RefinementOutput>,
    pub feedback: Option<RefinementFeedback>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RefinementLoopResult {
    pub schema_version: u32,
    pub model: String,
    pub prompt_version: String,
    pub status: RefinementLoopStatus,
    pub rounds: Vec<RefinementRound>,
    pub final_output: Option<RefinementOutput>,
    pub limitations: Vec<String>,
}

/// Run a bounded external-validation/repair loop for one refinement task.
///
/// The validator callback is owned by the caller so it can combine compiler
/// diagnostics and replay comparison without allowing generated source to run
/// inside this process. Only bounded feedback is returned to the model as
/// tagged untrusted evidence, and the loop never exceeds `max_rounds`.
pub fn run_refinement_loop<P, F>(
    provider: P,
    base_pack: EvidencePack,
    model: impl Into<String>,
    prompt_version: impl Into<String>,
    max_attempts: usize,
    max_rounds: usize,
    validate: F,
) -> RefinementLoopResult
where
    P: ModelProvider,
    F: Fn(&RefinementOutput) -> RefinementFeedback,
{
    let gateway = ModelGateway::new(provider, 4 * 1024 * 1024, 1024 * 1024);
    let model = model.into();
    let prompt_version = prompt_version.into();
    let round_limit = max_rounds.clamp(1, 8);
    let mut feedback = None;
    let mut rounds = Vec::with_capacity(round_limit);
    let mut final_output = None;

    for round in 1..=round_limit {
        let pack = refinement_pack_with_feedback(&base_pack, feedback.as_ref());
        let input_sha256 = sha256_json(&pack);
        let result = gateway.submit_with_retry(
            ModelTask::Refinement,
            model.clone(),
            prompt_version.clone(),
            pack,
            max_attempts,
        );
        let response = match result {
            Ok(response) => response,
            Err(error) => {
                let provider_error = matches!(error, AiError::Provider(_));
                rounds.push(RefinementRound {
                    round,
                    input_sha256,
                    output_sha256: None,
                    response: None,
                    feedback: feedback.clone(),
                    error: Some(error.to_string()),
                });
                return RefinementLoopResult {
                    schema_version: REFINEMENT_LOOP_SCHEMA_VERSION,
                    model: model.clone(),
                    prompt_version: prompt_version.clone(),
                    status: if provider_error {
                        RefinementLoopStatus::NotAvailable
                    } else {
                        RefinementLoopStatus::Rejected
                    },
                    rounds,
                    final_output,
                    limitations: vec![
                        "the bounded refinement loop stopped after a provider or validation error"
                            .into(),
                    ],
                };
            }
        };
        let output_sha256 = sha256_json(&response);
        let output =
            match serde_json::from_value::<RefinementOutput>(response.structured_output.clone()) {
                Ok(output) => output,
                Err(error) => {
                    rounds.push(RefinementRound {
                        round,
                        input_sha256,
                        output_sha256: Some(output_sha256),
                        response: None,
                        feedback: feedback.clone(),
                        error: Some(format!("refinement output parse failed: {error}")),
                    });
                    return RefinementLoopResult {
                        schema_version: REFINEMENT_LOOP_SCHEMA_VERSION,
                        model: model.clone(),
                        prompt_version: prompt_version.clone(),
                        status: RefinementLoopStatus::Rejected,
                        rounds,
                        final_output,
                        limitations: vec![
                            "the gateway response did not decode as refinement output".into(),
                        ],
                    };
                }
            };
        let current_feedback = validate(&output);
        let terminal_status = if current_feedback.is_behaviorally_validated() {
            Some(RefinementLoopStatus::BehaviorallyValidated)
        } else if current_feedback.is_compiled_unverified() {
            Some(RefinementLoopStatus::CompiledUnverified)
        } else if current_feedback.compilation == CompilationStatus::NotAvailable {
            Some(RefinementLoopStatus::NotAvailable)
        } else {
            None
        };
        rounds.push(RefinementRound {
            round,
            input_sha256,
            output_sha256: Some(output_sha256),
            response: Some(output.clone()),
            feedback: Some(current_feedback.clone()),
            error: None,
        });
        final_output = Some(output);
        if let Some(status) = terminal_status {
            return RefinementLoopResult {
                schema_version: REFINEMENT_LOOP_SCHEMA_VERSION,
                model: model.clone(),
                prompt_version: prompt_version.clone(),
                status,
                rounds,
                final_output,
                limitations: vec![
                    "behavioral agreement is limited to the supplied trace and observations".into(),
                ],
            };
        }
        feedback = Some(current_feedback);
    }

    RefinementLoopResult {
        schema_version: REFINEMENT_LOOP_SCHEMA_VERSION,
        model,
        prompt_version,
        status: RefinementLoopStatus::Exhausted,
        rounds,
        final_output,
        limitations: vec![
            "repair loop reached its configured round limit without passing external validation"
                .into(),
        ],
    }
}

fn refinement_pack_with_feedback(
    base_pack: &EvidencePack,
    feedback: Option<&RefinementFeedback>,
) -> EvidencePack {
    let mut pack = base_pack.clone();
    pack.task = ModelTask::Refinement;
    if let Some(feedback) = feedback {
        let content = serde_json::to_string(feedback).unwrap_or_else(|_| "{}".into());
        let bounded = content.chars().take(16 * 1024).collect::<String>();
        pack.untrusted_evidence.push(UntrustedEvidence {
            source: "external_validation".into(),
            content: bounded,
        });
    }
    pack
}

fn spawn_limited_reader<R>(
    mut reader: R,
    limit: usize,
) -> thread::JoinHandle<Result<Vec<u8>, String>>
where
    R: Read + Send + 'static,
{
    thread::spawn(move || {
        let mut bytes = Vec::new();
        reader
            .by_ref()
            .take(limit.saturating_add(1) as u64)
            .read_to_end(&mut bytes)
            .map_err(|error| format!("compiler output read failed: {error}"))?;
        if bytes.len() > limit {
            bytes.truncate(limit);
            bytes.extend_from_slice(b"\n[diagnostics truncated]");
        }
        Ok(bytes)
    })
}

fn join_reader(
    reader: Option<thread::JoinHandle<Result<Vec<u8>, String>>>,
) -> Result<Vec<u8>, String> {
    reader
        .map(|reader| {
            reader
                .join()
                .map_err(|_| String::from("compiler output reader panicked"))?
        })
        .unwrap_or_else(|| Ok(Vec::new()))
}

fn validate_response_for_request(
    request: &ModelRequest,
    response: &ModelResponse,
) -> Result<(), AiError> {
    if response.task != request.task {
        return Err(AiError::InvalidResponse(format!(
            "model response task {:?} does not match request task {:?}",
            response.task, request.task
        )));
    }
    if response.prompt_version != request.prompt_version {
        return Err(AiError::InvalidResponse(format!(
            "model response prompt version {:?} does not match request {:?}",
            response.prompt_version, request.prompt_version
        )));
    }
    validate_structured_output(request.task, &response.structured_output)?;
    validate_structured_consistency(request, &response.structured_output)?;
    validate_response(response)?;
    for item in &response.evidence {
        for evidence in &item.evidence {
            validate_evidence_ref(request, evidence)?;
        }
    }
    for evidence in structured_evidence_refs(request.task, &response.structured_output)? {
        validate_evidence_ref(request, &evidence)?;
    }
    Ok(())
}

/// Enforce that model edits address only functions present in the bounded IR.
/// This prevents a syntactically valid response from introducing a new function
/// identity that cannot be traced back to deterministic analysis evidence.
fn validate_structured_consistency(
    request: &ModelRequest,
    output: &serde_json::Value,
) -> Result<(), AiError> {
    let known_functions = request
        .evidence_pack
        .ir
        .functions
        .iter()
        .map(|function| function.id.as_str())
        .collect::<BTreeSet<_>>();
    match request.task {
        ModelTask::StructureRecovery => {
            let parsed: StructureRecoveryOutput = serde_json::from_value(output.clone())
                .map_err(|error| AiError::InvalidResponse(error.to_string()))?;
            for function in parsed.functions {
                if !known_functions.contains(function.function_id.as_str()) {
                    return Err(AiError::InvalidResponse(format!(
                        "structure recovery references unknown function {}",
                        function.function_id
                    )));
                }
            }
        }
        ModelTask::Refinement => {
            let parsed: RefinementOutput = serde_json::from_value(output.clone())
                .map_err(|error| AiError::InvalidResponse(error.to_string()))?;
            if parsed.schema_version != REFINEMENT_SCHEMA_VERSION {
                return Err(AiError::InvalidResponse(format!(
                    "unsupported refinement schema version {}",
                    parsed.schema_version
                )));
            }
            if !matches!(
                parsed.validation.status.as_str(),
                "validated" | "unverified" | "rejected" | "not_available"
            ) {
                return Err(AiError::InvalidResponse(format!(
                    "unsupported refinement validation status {}",
                    parsed.validation.status
                )));
            }
            if parsed.validation.status == "validated" {
                return Err(AiError::InvalidResponse(
                    "a model cannot self-assert validated refinement; external compiler or replay evidence is required"
                        .into(),
                ));
            }
            for patch in parsed.patches {
                if !known_functions.contains(patch.function_id.as_str()) {
                    return Err(AiError::InvalidResponse(format!(
                        "refinement references unknown function {}",
                        patch.function_id
                    )));
                }
            }
            if parsed.observations.len() > 64 {
                return Err(AiError::InvalidResponse(
                    "refinement contains more than 64 behavior observations".into(),
                ));
            }
            for observation in parsed.observations {
                if !matches!(
                    observation.kind.as_str(),
                    "log_contains" | "cpi_target" | "status" | "return_data_base64"
                ) {
                    return Err(AiError::InvalidResponse(format!(
                        "unsupported refinement observation kind {}",
                        observation.kind
                    )));
                }
                if observation.value.len() > 512 {
                    return Err(AiError::InvalidResponse(
                        "refinement observation value exceeds 512 bytes".into(),
                    ));
                }
            }
        }
        ModelTask::SemanticLabeling | ModelTask::Explanation => {}
    }
    Ok(())
}

fn validate_structured_output(task: ModelTask, output: &serde_json::Value) -> Result<(), AiError> {
    let parsed = match task {
        ModelTask::StructureRecovery => {
            serde_json::from_value::<StructureRecoveryOutput>(output.clone())
                .map(|_| "structure_recovery")
        }
        ModelTask::SemanticLabeling => {
            serde_json::from_value::<SemanticLabelingOutput>(output.clone())
                .map(|_| "semantic_labeling")
        }
        ModelTask::Explanation => {
            serde_json::from_value::<ExplanationOutput>(output.clone()).map(|_| "explanation")
        }
        ModelTask::Refinement => {
            serde_json::from_value::<RefinementOutput>(output.clone()).map(|_| "refinement")
        }
    };
    parsed.map(|_| ()).map_err(|error| {
        AiError::InvalidResponse(format!(
            "structured output does not match {:?} schema: {error}",
            task
        ))
    })
}

fn structured_evidence_refs(
    task: ModelTask,
    output: &serde_json::Value,
) -> Result<Vec<sos_types::EvidenceRef>, AiError> {
    let mut references = Vec::new();
    match task {
        ModelTask::StructureRecovery => {
            let parsed: StructureRecoveryOutput = serde_json::from_value(output.clone())
                .map_err(|error| AiError::InvalidResponse(error.to_string()))?;
            for function in parsed.functions {
                if !function.pseudocode.trim().is_empty() && function.evidence.is_empty() {
                    return Err(AiError::InvalidResponse(format!(
                        "recovered function {} has pseudocode but no evidence",
                        function.function_id
                    )));
                }
                validate_readable_candidate(&function.pseudocode, &function.function_id)?;
                references.extend(function.evidence);
            }
        }
        ModelTask::SemanticLabeling => {
            let parsed: SemanticLabelingOutput = serde_json::from_value(output.clone())
                .map_err(|error| AiError::InvalidResponse(error.to_string()))?;
            for label in parsed.labels {
                if !matches!(label.status, EvidenceStatus::NotAvailable)
                    && label.evidence.is_empty()
                {
                    return Err(AiError::InvalidResponse(format!(
                        "semantic label {} has status {:?} but no evidence",
                        label.kind, label.status
                    )));
                }
                if label
                    .confidence
                    .is_some_and(|value| !(0.0..=1.0).contains(&value))
                {
                    return Err(AiError::InvalidResponse(format!(
                        "semantic label {} has confidence outside [0, 1]",
                        label.kind
                    )));
                }
                references.extend(label.evidence);
            }
        }
        ModelTask::Refinement => {
            let parsed: RefinementOutput = serde_json::from_value(output.clone())
                .map_err(|error| AiError::InvalidResponse(error.to_string()))?;
            for patch in parsed.patches {
                if !patch.replacement.trim().is_empty() && patch.evidence.is_empty() {
                    return Err(AiError::InvalidResponse(format!(
                        "refinement patch for {} has no evidence",
                        patch.function_id
                    )));
                }
                validate_readable_candidate(&patch.replacement, &patch.function_id)?;
                references.extend(patch.evidence);
            }
            for observation in parsed.observations {
                if observation.evidence.is_empty() {
                    return Err(AiError::InvalidResponse(format!(
                        "refinement observation {} has no evidence",
                        observation.kind
                    )));
                }
                references.extend(observation.evidence);
            }
        }
        ModelTask::Explanation => {}
    }
    Ok(references)
}

fn validate_readable_candidate(source: &str, function_id: &str) -> Result<(), AiError> {
    if source.len() > 256 * 1024 {
        return Err(AiError::InvalidResponse(format!(
            "candidate for {function_id} exceeds the 256 KiB source limit"
        )));
    }
    for forbidden in [
        "asm!(",
        "asm! (",
        "global_asm!(",
        "include!(",
        "include_str!(",
        "include_bytes!(",
        "sos_raw_op",
        "core::arch::asm",
        "std::arch::asm",
        "transmute(",
        "write_volatile(",
        "black_box(",
        "#[no_mangle]",
    ] {
        if source.contains(forbidden) {
            return Err(AiError::InvalidResponse(format!(
                "candidate for {function_id} contains forbidden reconstruction shortcut `{forbidden}`"
            )));
        }
    }
    Ok(())
}

fn validate_evidence_ref(
    request: &ModelRequest,
    evidence: &sos_types::EvidenceRef,
) -> Result<(), AiError> {
    if evidence.source.trim().is_empty() || evidence.location.trim().is_empty() {
        return Err(AiError::InvalidResponse(
            "evidence references must include a source and location".into(),
        ));
    }
    let supported_source = matches!(evidence.source.as_str(), "analysis" | "ir" | "ro_section");
    if !supported_source {
        return Err(AiError::InvalidResponse(format!(
            "unsupported evidence source: {}",
            evidence.source
        )));
    }
    if let Some(pc_text) = evidence.location.strip_prefix("instruction:0x") {
        if evidence.source != "analysis" {
            return Err(AiError::InvalidResponse(format!(
                "instruction evidence must use the analysis source: {}",
                evidence.location
            )));
        }
        let pc = usize::from_str_radix(pc_text, 16).map_err(|_| {
            AiError::InvalidResponse(format!(
                "invalid instruction evidence location: {}",
                evidence.location
            ))
        })?;
        if !request
            .evidence_pack
            .analysis
            .instructions
            .iter()
            .any(|instruction| instruction.pc == pc)
        {
            return Err(AiError::InvalidResponse(format!(
                "instruction evidence location does not exist in the bounded pack: {}",
                evidence.location
            )));
        }
    }
    if evidence.source == "ir" {
        let referenced_id = evidence
            .location
            .strip_prefix("operation:")
            .or_else(|| evidence.location.strip_prefix("block:"))
            .or_else(|| evidence.location.strip_prefix("function:"))
            .unwrap_or(&evidence.location);
        let exists = request
            .evidence_pack
            .ir
            .operations
            .iter()
            .any(|item| item.id == referenced_id)
            || request
                .evidence_pack
                .ir
                .blocks
                .iter()
                .any(|item| item.id == referenced_id)
            || request
                .evidence_pack
                .ir
                .functions
                .iter()
                .any(|item| item.id == referenced_id);
        if !exists {
            return Err(AiError::InvalidResponse(format!(
                "IR evidence location does not exist in the bounded pack: {}",
                evidence.location
            )));
        }
    }
    if evidence.source == "ro_section" {
        let offset_text = evidence.location.strip_prefix("offset:0x").ok_or_else(|| {
            AiError::InvalidResponse(format!(
                "invalid read-only section evidence location: {}",
                evidence.location
            ))
        })?;
        let offset = usize::from_str_radix(offset_text, 16).map_err(|_| {
            AiError::InvalidResponse(format!(
                "invalid read-only section evidence location: {}",
                evidence.location
            ))
        })?;
        let exists = request
            .evidence_pack
            .analysis
            .strings
            .iter()
            .any(|item| item.offset == offset)
            || request
                .evidence_pack
                .analysis
                .public_key_candidates
                .iter()
                .any(|item| item.offset == offset);
        if !exists {
            return Err(AiError::InvalidResponse(format!(
                "read-only section evidence location does not exist in the bounded pack: {}",
                evidence.location
            )));
        }
    }
    if let Some(memory) = evidence.location.strip_prefix("memory:r") {
        let (register_text, offset_text) = memory.split_once('+').ok_or_else(|| {
            AiError::InvalidResponse(format!(
                "invalid memory evidence location: {}",
                evidence.location
            ))
        })?;
        let register = register_text.parse::<u8>().map_err(|_| {
            AiError::InvalidResponse(format!(
                "invalid memory evidence location: {}",
                evidence.location
            ))
        })?;
        let offset = offset_text.parse::<i16>().map_err(|_| {
            AiError::InvalidResponse(format!(
                "invalid memory evidence location: {}",
                evidence.location
            ))
        })?;
        if !request
            .evidence_pack
            .analysis
            .memory_accesses
            .iter()
            .any(|item| item.base_register == register && item.offset == offset)
        {
            return Err(AiError::InvalidResponse(format!(
                "memory evidence location does not exist in the bounded pack: {}",
                evidence.location
            )));
        }
    }
    let recognized_location = evidence.location.starts_with("instruction:0x")
        || evidence.location.starts_with("memory:r")
        || evidence.location.starts_with("offset:0x")
        || evidence.location.starts_with("operation:")
        || evidence.location.starts_with("block:")
        || evidence.location.starts_with("function:");
    if !recognized_location {
        return Err(AiError::InvalidResponse(format!(
            "unsupported evidence location: {}",
            evidence.location
        )));
    }
    if evidence.source == "analysis"
        && !evidence.location.starts_with("instruction:0x")
        && !evidence.location.starts_with("memory:r")
    {
        return Err(AiError::InvalidResponse(format!(
            "analysis evidence has an unsupported location: {}",
            evidence.location
        )));
    }
    if evidence.source == "ro_section" && !evidence.location.starts_with("offset:0x") {
        return Err(AiError::InvalidResponse(format!(
            "read-only section evidence has an unsupported location: {}",
            evidence.location
        )));
    }
    if evidence.source == "ir"
        && !evidence.location.starts_with("operation:")
        && !evidence.location.starts_with("block:")
        && !evidence.location.starts_with("function:")
    {
        return Err(AiError::InvalidResponse(format!(
            "IR evidence has an unsupported location: {}",
            evidence.location
        )));
    }
    Ok(())
}

pub fn validate_response(response: &ModelResponse) -> Result<(), AiError> {
    validate_structured_output(response.task, &response.structured_output)?;
    let _ = structured_evidence_refs(response.task, &response.structured_output)?;
    for item in &response.evidence {
        if !matches!(item.status, EvidenceStatus::NotAvailable) && item.evidence.is_empty() {
            return Err(AiError::InvalidResponse(format!(
                "{} item has status {:?} but no evidence references",
                item.kind, item.status
            )));
        }
        if item
            .confidence
            .is_some_and(|value| !(0.0..=1.0).contains(&value))
        {
            return Err(AiError::InvalidResponse(format!(
                "{} item has confidence outside [0, 1]",
                item.kind
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex,
    };
    #[cfg(unix)]
    use std::time::Duration;

    use super::*;
    use sos_analysis::{BasicBlockFact, FunctionFact, InstructionFact, StaticAnalysis};

    struct EchoProvider;

    impl ModelProvider for EchoProvider {
        fn generate_structured(&self, request: ModelRequest) -> Result<ModelResponse, AiError> {
            let structured_output = match request.task {
                ModelTask::StructureRecovery => serde_json::json!({"functions": []}),
                ModelTask::SemanticLabeling => serde_json::json!({"labels": []}),
                ModelTask::Explanation => serde_json::json!({
                    "summary": "No additional explanation was generated.",
                    "observations": []
                }),
                ModelTask::Refinement => serde_json::json!({
                    "patches": [],
                    "validation": {"status": "not_available", "notes": []}
                }),
            };
            Ok(ModelResponse {
                task: request.task,
                model: request.model,
                prompt_version: request.prompt_version,
                structured_output,
                evidence: vec![],
            })
        }
    }

    struct RecordingProvider {
        prior_counts: Arc<Mutex<Vec<usize>>>,
    }

    impl ModelProvider for RecordingProvider {
        fn generate_structured(&self, request: ModelRequest) -> Result<ModelResponse, AiError> {
            self.prior_counts
                .lock()
                .unwrap()
                .push(request.evidence_pack.untrusted_evidence.len());
            EchoProvider.generate_structured(request)
        }
    }

    struct FlakyProvider {
        attempts: Arc<AtomicUsize>,
    }

    impl ModelProvider for FlakyProvider {
        fn generate_structured(&self, request: ModelRequest) -> Result<ModelResponse, AiError> {
            if self.attempts.fetch_add(1, Ordering::SeqCst) == 0 {
                return Err(AiError::Provider("temporary provider failure".into()));
            }
            EchoProvider.generate_structured(request)
        }
    }

    struct RepairProvider;

    impl ModelProvider for RepairProvider {
        fn generate_structured(&self, request: ModelRequest) -> Result<ModelResponse, AiError> {
            let repairing = request
                .evidence_pack
                .untrusted_evidence
                .iter()
                .any(|item| item.source == "external_validation");
            let replacement = if repairing {
                "fn fn_0() { let recovered_value = 1_u64; let _ = recovered_value; }"
            } else {
                "fn broken( {"
            };
            Ok(ModelResponse {
                task: request.task,
                model: request.model,
                prompt_version: request.prompt_version,
                structured_output: serde_json::json!({
                    "patches": [{
                        "function_id": "fn_0",
                        "replacement": replacement,
                        "evidence": [{"source": "ir", "location": "function:fn_0"}]
                    }],
                    "observations": [],
                    "validation": {"status": "unverified", "notes": []}
                }),
                evidence: vec![],
            })
        }
    }

    struct WrongTaskProvider;

    impl ModelProvider for WrongTaskProvider {
        fn generate_structured(&self, request: ModelRequest) -> Result<ModelResponse, AiError> {
            Ok(ModelResponse {
                task: match request.task {
                    ModelTask::Explanation => ModelTask::StructureRecovery,
                    task => task,
                },
                model: request.model,
                prompt_version: request.prompt_version,
                structured_output: serde_json::json!({"functions": []}),
                evidence: vec![],
            })
        }
    }

    struct LargeResponseProvider;

    impl ModelProvider for LargeResponseProvider {
        fn generate_structured(&self, request: ModelRequest) -> Result<ModelResponse, AiError> {
            Ok(ModelResponse {
                task: request.task,
                model: request.model,
                prompt_version: request.prompt_version,
                structured_output: serde_json::json!({"payload": "0123456789"}),
                evidence: vec![],
            })
        }
    }

    struct BadReferenceProvider;

    impl ModelProvider for BadReferenceProvider {
        fn generate_structured(&self, request: ModelRequest) -> Result<ModelResponse, AiError> {
            Ok(ModelResponse {
                task: request.task,
                model: request.model,
                prompt_version: request.prompt_version,
                structured_output: serde_json::json!({
                    "summary": "A deliberately invalid IR reference follows.",
                    "observations": []
                }),
                evidence: vec![EvidenceItem {
                    kind: "bad_ir_claim".into(),
                    value: serde_json::json!("unknown"),
                    confidence: Some(0.1),
                    status: EvidenceStatus::Fact,
                    evidence: vec![sos_types::EvidenceRef {
                        source: "ir".into(),
                        location: "operation:op_missing".into(),
                        detail: None,
                    }],
                }],
            })
        }
    }

    fn empty_pack() -> EvidencePack {
        let analysis = StaticAnalysis {
            schema_version: sos_types::ANALYSIS_SCHEMA_VERSION,
            sbpf_version: "V0".into(),
            entrypoint: 0,
            instructions: vec![InstructionFact {
                pc: 0,
                opcode: 0,
                dst: 0,
                src: 0,
                offset: 0,
                immediate: 0,
                assembly: "ja 0".into(),
            }],
            functions: vec![FunctionFact {
                pc: 0,
                symbol_hash: 0,
                name: "entrypoint".into(),
            }],
            basic_blocks: vec![BasicBlockFact {
                start_pc: 0,
                end_instruction_index: 1,
                destinations: vec![],
                label: "entrypoint".into(),
            }],
            memory_accesses: vec![],
            stack_accesses: vec![],
            strings: vec![],
            public_key_candidates: vec![],
            immediate_constants: vec![],
            register_uses: vec![],
            calls: vec![],
            relocations: vec![],
            sccs: vec![],
            dataflow_edges: vec![],
            verifier_diagnostics: vec![],
        };
        let semantics = sos_solana::analyze(&analysis);
        EvidencePack::from_analysis(ModelTask::Explanation, None, analysis, semantics)
    }

    #[test]
    fn gateway_accepts_schema_valid_provider_output() {
        let gateway = ModelGateway::new(EchoProvider, 10_000, 10_000);
        let result = gateway.submit(ModelTask::Explanation, "test", "v1", empty_pack());
        assert!(result.is_ok());
    }

    #[test]
    fn gateway_retries_bounded_provider_failures() {
        let attempts = Arc::new(AtomicUsize::new(0));
        let gateway = ModelGateway::new(
            FlakyProvider {
                attempts: attempts.clone(),
            },
            10_000,
            10_000,
        );
        let result =
            gateway.submit_with_retry(ModelTask::Explanation, "test", "v1", empty_pack(), 2);
        assert!(result.is_ok());
        assert_eq!(attempts.load(Ordering::SeqCst), 2);
    }

    #[cfg(unix)]
    #[test]
    fn command_provider_times_out_stalled_local_models() {
        let provider = CommandModelProvider::new("/bin/sh")
            .with_args(["-c", "sleep 1"])
            .with_timeout(Duration::from_millis(20));
        let pack = empty_pack();
        let result = provider.generate_structured(ModelRequest {
            task: ModelTask::Explanation,
            prompt_version: "v1".into(),
            model: "test".into(),
            evidence_pack: pack,
            max_output_bytes: 10_000,
        });
        assert!(matches!(result, Err(AiError::Provider(error)) if error.contains("timed out")));
    }

    #[test]
    fn refinement_compiler_distinguishes_pass_and_fail() {
        let valid = RefinementOutput {
            schema_version: REFINEMENT_SCHEMA_VERSION,
            patches: vec![RefinementPatch {
                function_id: "fn_0".into(),
                replacement: "fn fn_0() { sos_raw_op!(0, 1, 2, 3, 4); unknown_return(); }".into(),
                evidence: vec![],
            }],
            observations: Vec::new(),
            validation: RefinementValidation {
                status: "unverified".into(),
                notes: vec![],
            },
        };
        let valid_result =
            compile_refinement_output(&valid, Some(Path::new("rustc")), Duration::from_secs(10));
        assert_eq!(valid_result.status, CompilationStatus::Passed);

        let invalid = RefinementOutput {
            schema_version: REFINEMENT_SCHEMA_VERSION,
            patches: vec![RefinementPatch {
                function_id: "fn_0".into(),
                replacement: "fn broken( {".into(),
                evidence: vec![],
            }],
            observations: Vec::new(),
            validation: RefinementValidation {
                status: "unverified".into(),
                notes: vec![],
            },
        };
        let invalid_result =
            compile_refinement_output(&invalid, Some(Path::new("rustc")), Duration::from_secs(10));
        assert_eq!(invalid_result.status, CompilationStatus::Failed);
        assert!(!invalid_result.diagnostics.is_empty());
    }

    #[test]
    fn refinement_loop_feeds_bounded_mismatch_back_to_repair_pass() {
        let result = run_refinement_loop(
            RepairProvider,
            empty_pack(),
            "test-model",
            "sos-v1",
            1,
            3,
            |output| {
                let replacement = output
                    .patches
                    .first()
                    .map(|patch| patch.replacement.as_str())
                    .unwrap_or_default();
                if replacement.contains("broken") {
                    RefinementFeedback {
                        compilation: CompilationStatus::Failed,
                        compiler_diagnostics: vec!["expected item, found `{`".into()],
                        behavioral_status: "unverified".into(),
                        mismatches: vec!["candidate did not compile".into()],
                        limitations: Vec::new(),
                    }
                } else {
                    RefinementFeedback {
                        compilation: CompilationStatus::Passed,
                        compiler_diagnostics: Vec::new(),
                        behavioral_status: "behaviorally_validated".into(),
                        mismatches: Vec::new(),
                        limitations: Vec::new(),
                    }
                }
            },
        );
        assert_eq!(result.status, RefinementLoopStatus::BehaviorallyValidated);
        assert_eq!(result.rounds.len(), 2);
        assert!(result.rounds[1]
            .input_sha256
            .as_bytes()
            .iter()
            .all(|byte| byte.is_ascii_hexdigit()));
    }

    #[test]
    fn pipeline_runs_all_stages_and_records_audit_hashes() {
        let pack = empty_pack();
        let bundle = run_pipeline(
            EchoProvider,
            AiPipelineConfig {
                provider_name: "test-provider".into(),
                model: "test-model".into(),
                prompt_version: "sos-v1".into(),
                artifact_sha256: Some("artifact".into()),
                limits: EvidencePackLimits::default(),
                max_attempts: 2,
            },
            &pack.analysis,
            &pack.semantics,
        );
        assert_eq!(bundle.schema_version, AI_BUNDLE_SCHEMA_VERSION);
        assert_eq!(bundle.stages.len(), ModelTask::ALL.len());
        assert!(bundle
            .stages
            .iter()
            .all(|stage| stage.status == AiStageStatus::Accepted));
        assert!(bundle
            .stages
            .iter()
            .all(|stage| stage.input_sha256.len() == 64));
        assert!(bundle.stages.iter().all(|stage| stage
            .output_sha256
            .as_ref()
            .is_some_and(|hash| hash.len() == 64)));
    }

    #[test]
    fn pipeline_passes_accepted_stage_output_forward_as_untrusted_context() {
        let pack = empty_pack();
        let prior_counts = Arc::new(Mutex::new(Vec::new()));
        let _ = run_pipeline(
            RecordingProvider {
                prior_counts: prior_counts.clone(),
            },
            AiPipelineConfig {
                provider_name: "test-provider".into(),
                model: "test-model".into(),
                prompt_version: "sos-v2".into(),
                artifact_sha256: None,
                limits: EvidencePackLimits::default(),
                max_attempts: 1,
            },
            &pack.analysis,
            &pack.semantics,
        );
        assert_eq!(*prior_counts.lock().unwrap(), vec![0, 1, 2, 3]);
    }

    #[test]
    fn gateway_rejects_oversized_evidence_pack() {
        let gateway = ModelGateway::new(EchoProvider, 1, 10_000);
        let result = gateway.submit(ModelTask::Explanation, "test", "v1", empty_pack());
        assert!(matches!(result, Err(AiError::EvidencePackTooLarge { .. })));
    }

    #[test]
    fn gateway_rejects_response_for_wrong_task() {
        let gateway = ModelGateway::new(WrongTaskProvider, 10_000, 10_000);
        let result = gateway.submit(ModelTask::Explanation, "test", "v1", empty_pack());
        assert!(matches!(result, Err(AiError::InvalidResponse(error)) if error.contains("task")));
    }

    #[test]
    fn task_schema_accepts_explanation_output() {
        let response = ModelResponse {
            task: ModelTask::Explanation,
            model: "test".into(),
            prompt_version: "v1".into(),
            structured_output: serde_json::json!({
                "summary": "The entrypoint logs a message.",
                "observations": ["A logging syscall was identified."]
            }),
            evidence: vec![],
        };
        assert!(validate_response(&response).is_ok());
    }

    #[test]
    fn task_schema_rejects_arbitrary_structured_output() {
        let response = ModelResponse {
            task: ModelTask::Explanation,
            model: "test".into(),
            prompt_version: "v1".into(),
            structured_output: serde_json::json!({"ok": true}),
            evidence: vec![],
        };
        assert!(validate_structured_output(response.task, &response.structured_output).is_err());
    }

    #[test]
    fn readable_candidates_reject_inline_assembly_and_raw_op_passthroughs() {
        for source in [
            "fn fn_0() { core::arch::asm!(\"exit\"); }",
            "fn fn_0() { sos_raw_op(0, 0, 0, 0, 0); }",
        ] {
            let response = ModelResponse {
                task: ModelTask::StructureRecovery,
                model: "test".into(),
                prompt_version: "v1".into(),
                structured_output: serde_json::json!({
                    "functions": [{
                        "function_id": "fn_0",
                        "pseudocode": source,
                        "evidence": [{"source": "ir", "location": "function:fn_0"}]
                    }]
                }),
                evidence: vec![],
            };
            assert!(validate_response(&response).is_err());
        }
    }

    #[test]
    fn nested_semantic_claims_require_evidence() {
        let response = ModelResponse {
            task: ModelTask::SemanticLabeling,
            model: "test".into(),
            prompt_version: "v1".into(),
            structured_output: serde_json::json!({
                "labels": [{
                    "kind": "function_name",
                    "value": "swap",
                    "confidence": 0.8,
                    "status": "hypothesis",
                    "evidence": []
                }]
            }),
            evidence: vec![],
        };
        assert!(validate_response(&response).is_err());
    }

    #[test]
    fn evidence_locations_must_exist_for_rodata_and_memory() {
        let mut pack = empty_pack();
        pack.analysis.strings.push(sos_analysis::StringFact {
            offset: 0x20,
            value: "log".into(),
        });
        pack.analysis
            .memory_accesses
            .push(sos_analysis::MemoryAccessFact {
                pc: 0,
                kind: "load".into(),
                width: 8,
                base_register: 1,
                value_register: Some(2),
                offset: 8,
                immediate: None,
                assembly: "ldxdw r2, [r1+8]".into(),
            });
        let request = ModelRequest {
            task: ModelTask::Explanation,
            prompt_version: "v1".into(),
            model: "test".into(),
            evidence_pack: pack,
            max_output_bytes: 10_000,
        };
        assert!(validate_evidence_ref(
            &request,
            &sos_types::EvidenceRef {
                source: "ro_section".into(),
                location: "offset:0x20".into(),
                detail: None,
            }
        )
        .is_ok());
        assert!(validate_evidence_ref(
            &request,
            &sos_types::EvidenceRef {
                source: "analysis".into(),
                location: "memory:r1+8".into(),
                detail: None,
            }
        )
        .is_ok());
        assert!(validate_evidence_ref(
            &request,
            &sos_types::EvidenceRef {
                source: "ro_section".into(),
                location: "offset:0x99".into(),
                detail: None,
            }
        )
        .is_err());
    }

    #[test]
    fn evidence_validator_rejects_unknown_sources_and_locations() {
        let request = ModelRequest {
            task: ModelTask::Explanation,
            prompt_version: "v1".into(),
            model: "test".into(),
            evidence_pack: empty_pack(),
            max_output_bytes: 10_000,
        };
        for evidence in [
            sos_types::EvidenceRef {
                source: "untrusted_model_claim".into(),
                location: "anything".into(),
                detail: None,
            },
            sos_types::EvidenceRef {
                source: "analysis".into(),
                location: "arbitrary:0x10".into(),
                detail: None,
            },
            sos_types::EvidenceRef {
                source: "ir".into(),
                location: "offset:0x10".into(),
                detail: None,
            },
        ] {
            assert!(validate_evidence_ref(&request, &evidence).is_err());
        }
    }

    #[test]
    fn http_provider_decodes_the_provider_neutral_protocol() {
        use std::{
            io::{Read, Write},
            net::{Shutdown, TcpListener},
            thread,
        };

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = Vec::new();
            let mut chunk = [0_u8; 1024];
            loop {
                let read = stream.read(&mut chunk).unwrap();
                if read == 0 {
                    break;
                }
                request.extend_from_slice(&chunk[..read]);
                let Some(headers_end) = request.windows(4).position(|window| window == b"\r\n\r\n")
                else {
                    continue;
                };
                let content_length = request[..headers_end]
                    .split(|byte| *byte == b'\n')
                    .find_map(|line| {
                        let line = std::str::from_utf8(line).ok()?;
                        let (name, value) = line.split_once(':')?;
                        (name.eq_ignore_ascii_case("content-length"))
                            .then(|| value.trim().parse::<usize>().ok())
                            .flatten()
                    })
                    .unwrap_or(0);
                if request.len() >= headers_end + 4 + content_length {
                    break;
                }
            }
            let response = serde_json::json!({
                "task": "explanation",
                "model": "hosted-test",
                "prompt_version": "v1",
                "structured_output": {
                    "summary": "server response",
                    "observations": []
                },
                "evidence": []
            })
            .to_string();
            let header = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                response.len()
            );
            stream.write_all(header.as_bytes()).unwrap();
            stream.write_all(response.as_bytes()).unwrap();
            stream.flush().unwrap();
            stream.shutdown(Shutdown::Write).unwrap();
        });

        let provider = HttpModelProvider::new(endpoint);
        let request = ModelRequest {
            task: ModelTask::Explanation,
            prompt_version: "v1".into(),
            model: "hosted-test".into(),
            evidence_pack: empty_pack(),
            max_output_bytes: 10_000,
        };
        let response = provider.generate_structured(request).unwrap();
        assert_eq!(response.model, "hosted-test");
        assert_eq!(response.task, ModelTask::Explanation);
        server.join().unwrap();
    }

    #[test]
    fn gateway_rejects_response_over_output_budget() {
        let gateway = ModelGateway::new(LargeResponseProvider, 10_000, 32);
        let result = gateway.submit(ModelTask::Explanation, "test", "v1", empty_pack());
        assert!(matches!(result, Err(AiError::OutputTooLarge { .. })));
    }

    #[test]
    fn gateway_rejects_nonexistent_ir_evidence() {
        let gateway = ModelGateway::new(BadReferenceProvider, 10_000, 10_000);
        let result = gateway.submit(ModelTask::Explanation, "test", "v1", empty_pack());
        assert!(
            matches!(result, Err(AiError::InvalidResponse(error)) if error.contains("IR evidence"))
        );
    }

    #[test]
    fn bounded_pack_applies_analysis_limits() {
        let pack = empty_pack();
        let bounded = EvidencePack::bounded_from_analysis(
            ModelTask::Explanation,
            None,
            &pack.analysis,
            &pack.semantics,
            EvidencePackLimits {
                max_instructions: 0,
                ..EvidencePackLimits::default()
            },
        );
        assert!(bounded.analysis.instructions.is_empty());
        assert!(bounded.analysis.register_uses.is_empty());
    }

    #[test]
    fn function_scope_keeps_only_the_target_executable_facts() {
        let mut analysis = empty_pack().analysis;
        analysis.functions.push(FunctionFact {
            pc: 8,
            symbol_hash: 8,
            name: "helper".into(),
        });
        analysis.instructions.push(InstructionFact {
            pc: 8,
            opcode: 0,
            dst: 0,
            src: 0,
            offset: 0,
            immediate: 0,
            assembly: "exit".into(),
        });
        analysis.basic_blocks.push(BasicBlockFact {
            start_pc: 8,
            end_instruction_index: 2,
            destinations: vec![],
            label: "helper".into(),
        });
        analysis.strings.push(sos_analysis::StringFact {
            offset: 4,
            value: "shared rodata".into(),
        });

        let first = scope_analysis_to_function(&analysis, 0).unwrap();
        assert_eq!(first.functions.len(), 1);
        assert!(first.instructions.iter().all(|item| item.pc < 8));
        assert_eq!(first.strings.len(), 1);

        let second = scope_analysis_to_function(&analysis, 8).unwrap();
        assert_eq!(second.functions[0].name, "helper");
        assert_eq!(second.instructions.len(), 1);
        assert_eq!(second.instructions[0].pc, 8);
    }

    #[test]
    fn refinement_must_target_a_known_ir_function() {
        let request = ModelRequest {
            task: ModelTask::Refinement,
            prompt_version: "v1".into(),
            model: "test".into(),
            evidence_pack: empty_pack(),
            max_output_bytes: 10_000,
        };
        let output = serde_json::json!({
            "patches": [{
                "function_id": "fn_missing",
                "replacement": "fn recovered() { }",
                "evidence": [{
                    "source": "ir",
                    "location": "function:fn_0"
                }]
            }],
            "validation": {"status": "unverified", "notes": []}
        });
        assert!(validate_structured_consistency(&request, &output).is_err());
    }

    #[test]
    fn refinement_status_is_bounded_to_known_validation_states() {
        let request = ModelRequest {
            task: ModelTask::Refinement,
            prompt_version: "v1".into(),
            model: "test".into(),
            evidence_pack: empty_pack(),
            max_output_bytes: 10_000,
        };
        let output = serde_json::json!({
            "patches": [],
            "validation": {"status": "certain", "notes": []}
        });
        assert!(validate_structured_consistency(&request, &output).is_err());
    }

    #[test]
    fn refinement_observations_are_bounded_and_evidence_linked() {
        let request = ModelRequest {
            task: ModelTask::Refinement,
            prompt_version: "v1".into(),
            model: "test".into(),
            evidence_pack: empty_pack(),
            max_output_bytes: 10_000,
        };
        let unsupported = serde_json::json!({
            "schema_version": 1,
            "patches": [],
            "observations": [{"kind": "host_call", "value": "exec", "evidence": []}],
            "validation": {"status": "unverified", "notes": []}
        });
        assert!(validate_structured_consistency(&request, &unsupported).is_err());

        let missing_evidence = serde_json::json!({
            "schema_version": 1,
            "patches": [],
            "observations": [{"kind": "log_contains", "value": "log", "evidence": []}],
            "validation": {"status": "unverified", "notes": []}
        });
        assert!(validate_response_for_request(
            &request,
            &ModelResponse {
                task: ModelTask::Refinement,
                model: "test".into(),
                prompt_version: "v1".into(),
                structured_output: missing_evidence,
                evidence: vec![],
            }
        )
        .is_err());
    }

    #[test]
    fn model_cannot_self_assert_validated_refinement() {
        let request = ModelRequest {
            task: ModelTask::Refinement,
            prompt_version: "v1".into(),
            model: "test".into(),
            evidence_pack: empty_pack(),
            max_output_bytes: 10_000,
        };
        let output = serde_json::json!({
            "patches": [],
            "validation": {"status": "validated", "notes": ["model claim"]}
        });
        let error = validate_structured_consistency(&request, &output).unwrap_err();
        assert!(error.to_string().contains("external compiler or replay"));
    }

    #[test]
    fn hypotheses_require_evidence() {
        let response = ModelResponse {
            task: ModelTask::Explanation,
            model: "test".into(),
            prompt_version: "v1".into(),
            structured_output: serde_json::Value::Null,
            evidence: vec![EvidenceItem {
                kind: "name".into(),
                value: serde_json::json!("swap"),
                confidence: Some(0.8),
                status: EvidenceStatus::Hypothesis,
                evidence: vec![],
            }],
        };
        assert!(validate_response(&response).is_err());
    }
}
