//! Conservative Solana-specific semantic recovery over deterministic SBPF facts.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use solana_sbpf::ebpf;
use sos_analysis::{
    ImmediateConstantFact, InstructionFact, MemoryAccessFact, StaticAnalysis, StringFact,
};
use sos_types::{EvidenceItem, EvidenceRef, EvidenceStatus, SEMANTIC_SCHEMA_VERSION};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SyscallDefinition {
    pub name: &'static str,
    pub category: &'static str,
    pub argument_count: u8,
    pub memory_effect: &'static str,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SyscallFact {
    pub instruction_pc: usize,
    pub hash: u32,
    pub name: Option<String>,
    pub category: Option<String>,
    pub argument_count: Option<u8>,
    pub memory_effect: Option<String>,
    pub status: EvidenceStatus,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DispatchCandidate {
    pub instruction_pc: usize,
    pub opcode: u8,
    pub immediate: i64,
    pub target_pc: usize,
    pub interpretations: Vec<String>,
    pub confidence_basis: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InstructionCandidate {
    pub instruction_pc: usize,
    pub tag_value: i64,
    pub target_pc: usize,
    pub interpretations: Vec<String>,
    pub status: EvidenceStatus,
    pub evidence: Vec<EvidenceRef>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArgumentCandidate {
    pub instruction_pc: usize,
    pub register: u8,
    pub value: Option<i64>,
    pub source_pc: Option<usize>,
    pub interpretations: Vec<String>,
    pub status: EvidenceStatus,
    pub evidence: Vec<EvidenceRef>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AccountCandidate {
    pub base_register: u8,
    pub interpretations: Vec<String>,
    pub status: EvidenceStatus,
    pub evidence: Vec<EvidenceRef>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AccountFieldCandidate {
    pub base_register: u8,
    pub offset: i16,
    pub widths: Vec<u8>,
    pub interpretations: Vec<String>,
    pub status: EvidenceStatus,
    pub evidence: Vec<EvidenceRef>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StackAccountFieldCandidate {
    pub account_index: usize,
    pub field: String,
    pub frame_offset: i16,
    pub width: u8,
    pub access_pcs: Vec<usize>,
    pub status: EvidenceStatus,
    pub evidence: Vec<EvidenceRef>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ErrorCandidate {
    pub offset: usize,
    pub value: String,
    pub status: EvidenceStatus,
    pub evidence: Vec<EvidenceRef>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LogCandidate {
    pub offset: usize,
    pub value: String,
    pub status: EvidenceStatus,
    pub evidence: Vec<EvidenceRef>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResolvedLogCall {
    pub instruction_pc: usize,
    pub syscall: String,
    pub pointer: u64,
    pub length: usize,
    pub string_offset: usize,
    pub value: String,
    pub pointer_source_pc: usize,
    pub length_source_pc: usize,
    pub status: EvidenceStatus,
    pub evidence: Vec<EvidenceRef>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CpiCandidate {
    pub instruction_pc: usize,
    pub syscall: String,
    pub target_program: Option<String>,
    pub argument_registers: Vec<u8>,
    #[serde(default)]
    pub argument_values: Vec<Option<i64>>,
    #[serde(default)]
    pub argument_source_pcs: Vec<Option<usize>>,
    pub status: EvidenceStatus,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PdaCandidate {
    pub instruction_pc: usize,
    pub syscall: String,
    pub seed_argument_registers: Vec<u8>,
    #[serde(default)]
    pub seed_argument_values: Vec<Option<i64>>,
    #[serde(default)]
    pub seed_argument_source_pcs: Vec<Option<usize>>,
    pub bump_output_register: Option<u8>,
    pub interpretations: Vec<String>,
    pub status: EvidenceStatus,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FrameworkCandidate {
    pub framework: String,
    pub instruction_pc: Option<usize>,
    pub confidence: f32,
    pub interpretations: Vec<String>,
    pub status: EvidenceStatus,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InputFieldCandidate {
    pub base_register: u8,
    pub offset: i16,
    pub widths: Vec<u8>,
    pub load_count: usize,
    pub store_count: usize,
    pub interpretations: Vec<String>,
    pub status: EvidenceStatus,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeAbiField {
    pub name: String,
    pub scope: String,
    pub width: Option<u32>,
    pub variable_length: bool,
    pub description: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StringCandidate {
    pub offset: usize,
    pub value: String,
    pub interpretations: Vec<String>,
    pub status: EvidenceStatus,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConstantCandidate {
    pub pc: usize,
    pub value: i64,
    pub kind: String,
    pub interpretations: Vec<String>,
    pub status: EvidenceStatus,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeErrorCandidate {
    pub instruction_pc: usize,
    pub value: u64,
    pub name: String,
    pub status: EvidenceStatus,
    pub evidence: Vec<EvidenceRef>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProtocolFingerprint {
    pub program_id: String,
    pub name: String,
    pub category: String,
    pub byte_offset: usize,
    pub interpretations: Vec<String>,
    pub status: EvidenceStatus,
    pub evidence: Vec<EvidenceRef>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProtocolDefinition {
    pub program_id: String,
    pub name: String,
    pub category: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProtocolRegistry {
    definitions: Vec<ProtocolDefinition>,
}

impl ProtocolRegistry {
    pub fn new() -> Self {
        Self {
            definitions: Vec::new(),
        }
    }

    pub fn with_builtins() -> Self {
        let mut registry = Self::new();
        for (program_id, name, category) in builtin_protocols() {
            registry.register(ProtocolDefinition {
                program_id: (*program_id).into(),
                name: (*name).into(),
                category: (*category).into(),
            });
        }
        registry
    }

    pub fn from_definitions(definitions: impl IntoIterator<Item = ProtocolDefinition>) -> Self {
        let mut registry = Self::new();
        for definition in definitions {
            registry.register(definition);
        }
        registry
    }

    pub fn register(&mut self, definition: ProtocolDefinition) {
        if !self
            .definitions
            .iter()
            .any(|item| item.program_id == definition.program_id)
        {
            self.definitions.push(definition);
        }
    }

    pub fn definitions(&self) -> &[ProtocolDefinition] {
        &self.definitions
    }
}

impl Default for ProtocolRegistry {
    fn default() -> Self {
        Self::with_builtins()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ValidationCheckCandidate {
    pub load_pc: usize,
    pub branch_pc: usize,
    pub base_register: u8,
    pub offset: i16,
    pub width: u8,
    pub comparison: String,
    #[serde(default)]
    pub compared_value: Option<i64>,
    #[serde(default)]
    pub nearby_runtime_error: Option<String>,
    #[serde(default)]
    pub account_index: Option<usize>,
    #[serde(default)]
    pub account_field: Option<String>,
    pub interpretations: Vec<String>,
    pub status: EvidenceStatus,
    pub evidence: Vec<EvidenceRef>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SemanticAnalysis {
    pub schema_version: u32,
    pub syscalls: Vec<SyscallFact>,
    pub dispatch_candidates: Vec<DispatchCandidate>,
    #[serde(default)]
    pub instruction_candidates: Vec<InstructionCandidate>,
    #[serde(default)]
    pub argument_candidates: Vec<ArgumentCandidate>,
    pub cpi_candidates: Vec<CpiCandidate>,
    pub pda_candidates: Vec<PdaCandidate>,
    pub framework_candidates: Vec<FrameworkCandidate>,
    pub input_field_candidates: Vec<InputFieldCandidate>,
    #[serde(default)]
    pub account_candidates: Vec<AccountCandidate>,
    #[serde(default)]
    pub account_field_candidates: Vec<AccountFieldCandidate>,
    #[serde(default)]
    pub stack_account_field_candidates: Vec<StackAccountFieldCandidate>,
    pub runtime_abi: Vec<RuntimeAbiField>,
    pub string_candidates: Vec<StringCandidate>,
    pub constant_candidates: Vec<ConstantCandidate>,
    #[serde(default)]
    pub runtime_error_candidates: Vec<RuntimeErrorCandidate>,
    #[serde(default)]
    pub protocol_fingerprints: Vec<ProtocolFingerprint>,
    #[serde(default)]
    pub validation_checks: Vec<ValidationCheckCandidate>,
    #[serde(default)]
    pub error_candidates: Vec<ErrorCandidate>,
    #[serde(default)]
    pub log_candidates: Vec<LogCandidate>,
    #[serde(default)]
    pub resolved_log_calls: Vec<ResolvedLogCall>,
    pub evidence: Vec<EvidenceItem>,
}

pub const SECURITY_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SecurityFinding {
    pub detector_id: String,
    pub severity: String,
    pub confidence: f32,
    pub title: String,
    pub description: String,
    pub instruction_pcs: Vec<usize>,
    pub status: EvidenceStatus,
    pub evidence: Vec<EvidenceRef>,
    pub limitations: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SecurityReport {
    pub schema_version: u32,
    pub findings: Vec<SecurityFinding>,
}

/// Run conservative review detectors. These findings identify code that deserves
/// review; they do not claim exploitability without a validated execution trace.
pub fn detect(analysis: &StaticAnalysis, semantics: &SemanticAnalysis) -> SecurityReport {
    let mut findings = Vec::new();

    for syscall in semantics
        .syscalls
        .iter()
        .filter(|syscall| syscall.name.is_none())
    {
        findings.push(SecurityFinding {
            detector_id: "unknown_syscall".into(),
            severity: "low".into(),
            confidence: 1.0,
            title: "Unknown syscall hash requires review".into(),
            description: format!(
                "The call at 0x{:x} does not match the built-in runtime syscall registry.",
                syscall.instruction_pc
            ),
            instruction_pcs: vec![syscall.instruction_pc],
            status: EvidenceStatus::NotAvailable,
            evidence: vec![EvidenceRef {
                source: "analysis".into(),
                location: format!("instruction:0x{:x}", syscall.instruction_pc),
                detail: Some(format!("syscall hash {}", syscall.hash)),
            }],
            limitations: vec![
                "The hash may belong to a runtime version not in the local registry.".into(),
                "No exploitability claim is made.".into(),
            ],
        });
    }

    for candidate in semantics
        .cpi_candidates
        .iter()
        .filter(|candidate| candidate.target_program.is_none())
    {
        findings.push(SecurityFinding {
            detector_id: "unresolved_cpi_target".into(),
            severity: "low".into(),
            confidence: 0.8,
            title: "CPI target was not recovered".into(),
            description: format!(
                "The {} call at 0x{:x} has no statically recovered target program.",
                candidate.syscall, candidate.instruction_pc
            ),
            instruction_pcs: vec![candidate.instruction_pc],
            status: EvidenceStatus::Hypothesis,
            evidence: vec![EvidenceRef {
                source: "analysis".into(),
                location: format!("instruction:0x{:x}", candidate.instruction_pc),
                detail: Some("CPI target register/dataflow remains unresolved".into()),
            }],
            limitations: vec![
                "A missing target is an analysis limitation, not proof of an unsafe CPI.".into(),
                "Transaction traces or deeper dataflow are required for validation.".into(),
            ],
        });
    }

    for candidate in semantics
        .pda_candidates
        .iter()
        .filter(|candidate| candidate.seed_argument_values.iter().any(Option::is_none))
    {
        findings.push(SecurityFinding {
            detector_id: "unresolved_pda_seeds".into(),
            severity: "informational".into(),
            confidence: 0.75,
            title: "PDA seed material is partially unresolved".into(),
            description: format!(
                "The {} call at 0x{:x} has seed arguments that are not constant-resolved.",
                candidate.syscall, candidate.instruction_pc
            ),
            instruction_pcs: vec![candidate.instruction_pc],
            status: EvidenceStatus::Hypothesis,
            evidence: vec![EvidenceRef {
                source: "analysis".into(),
                location: format!("instruction:0x{:x}", candidate.instruction_pc),
                detail: Some("seed argument values contain unresolved registers".into()),
            }],
            limitations: vec![
                "The detector does not determine whether the PDA derivation is incorrect.".into(),
            ],
        });
    }

    // These are deliberately absence-of-proof observations. The current
    // deterministic pass does not yet prove a complete account-validation
    // path, so it reports the review obligation as unverified rather than
    // claiming that a check is exploitable or definitely missing.
    if !semantics.account_candidates.is_empty() {
        let evidence = semantics
            .account_candidates
            .iter()
            .flat_map(|candidate| candidate.evidence.iter().cloned())
            .take(8)
            .collect::<Vec<_>>();
        for (detector_id, title, field, interpretation) in [
            (
                "missing_signer_check",
                "Signer validation was not proven",
                "signer",
                "signer_flag_candidate",
            ),
            (
                "missing_owner_check",
                "Owner validation was not proven",
                "owner",
                "owner_field_candidate",
            ),
            (
                "missing_key_equality_check",
                "Account-key equality validation was not proven",
                "key equality",
                "key_equality_candidate",
            ),
            (
                "unchecked_account_data_length",
                "Account data-length validation was not proven",
                "data length",
                "account_length_candidate",
            ),
        ] {
            if semantics.validation_checks.iter().any(|check| {
                check
                    .interpretations
                    .iter()
                    .any(|item| item == interpretation)
            }) {
                continue;
            }
            findings.push(SecurityFinding {
                detector_id: detector_id.into(),
                severity: "informational".into(),
                confidence: 0.2,
                title: title.into(),
                description: format!(
                    "The current static pattern pass recovered account/input accesses but did not prove a {field} check on every reachable path."
                ),
                instruction_pcs: Vec::new(),
                status: EvidenceStatus::Unverified,
                evidence: evidence.clone(),
                limitations: vec![
                    "Absence of a recovered pattern is not proof that the check is missing.".into(),
                    "Control-flow joins, aliases, and helper functions require deeper dataflow.".into(),
                    "No exploitability claim is made.".into(),
                ],
            });
        }
    }

    let input_writes = analysis
        .memory_accesses
        .iter()
        .filter(|access| access.kind == "store" && access.base_register == 1)
        .take(64)
        .collect::<Vec<_>>();
    if !input_writes.is_empty() {
        findings.push(SecurityFinding {
            detector_id: "writable_input_write".into(),
            severity: "low".into(),
            confidence: 0.55,
            title: "Entrypoint input-root write requires writable-account review".into(),
            description: format!(
                "{} store(s) write through the conventional entrypoint input root; writable-account and aliasing behavior requires review.",
                input_writes.len()
            ),
            instruction_pcs: input_writes.iter().map(|access| access.pc).collect(),
            status: EvidenceStatus::Hypothesis,
            evidence: input_writes.iter().map(|access| memory_ref(access)).collect(),
            limitations: vec![
                "The register may have been repurposed by local dataflow.".into(),
                "A store is not itself an authorization or writable-account bug.".into(),
            ],
        });
    }

    let arithmetic = analysis
        .instructions
        .iter()
        .filter(|instruction| {
            let assembly = instruction.assembly.to_ascii_lowercase();
            ["add64", "sub64", "mul64", "div64", "mod64"]
                .iter()
                .any(|operator| assembly.starts_with(operator))
        })
        .take(64)
        .collect::<Vec<_>>();
    if !arithmetic.is_empty() {
        findings.push(SecurityFinding {
            detector_id: "unchecked_arithmetic".into(),
            severity: "low".into(),
            confidence: 0.3,
            title: "Arithmetic operation requires overflow/underflow review".into(),
            description: format!(
                "{} arithmetic operation(s) were recovered, but checkedness could not be established from the current local dataflow.",
                arithmetic.len()
            ),
            instruction_pcs: arithmetic.iter().map(|instruction| instruction.pc).collect(),
            status: EvidenceStatus::Unverified,
            evidence: arithmetic
                .iter()
                .map(|instruction| instruction_ref(instruction))
                .collect(),
            limitations: vec![
                "The detector does not determine operand ranges or runtime reachability.".into(),
                "No exploitability claim is made.".into(),
            ],
        });
    }

    SecurityReport {
        schema_version: SECURITY_SCHEMA_VERSION,
        findings,
    }
}

/// Logical layout of the byte buffer passed to the standard Solana BPF entrypoint.
/// Per-account offsets are intentionally described by scope instead of flattened
/// addresses because account data and padding make the record size variable.
pub fn runtime_abi() -> Vec<RuntimeAbiField> {
    vec![
        abi_field(
            "account_count",
            "input_prefix",
            Some(8),
            false,
            "number of account records",
        ),
        abi_field(
            "duplicate_marker",
            "account_record",
            Some(1),
            false,
            "u8 marker or duplicate account index",
        ),
        abi_field("is_signer", "account_record", Some(1), false, "signer flag"),
        abi_field(
            "is_writable",
            "account_record",
            Some(1),
            false,
            "writable flag",
        ),
        abi_field(
            "executable",
            "account_record",
            Some(1),
            false,
            "executable flag",
        ),
        abi_field(
            "original_data_len",
            "account_record",
            Some(4),
            false,
            "runtime-maintained original data length",
        ),
        abi_field(
            "key",
            "account_record",
            Some(32),
            false,
            "account owner public key",
        ),
        abi_field(
            "owner",
            "account_record",
            Some(32),
            false,
            "account owner public key",
        ),
        abi_field(
            "lamports",
            "account_record",
            Some(8),
            false,
            "account lamports",
        ),
        abi_field(
            "data_len",
            "account_record",
            Some(8),
            false,
            "account data length",
        ),
        abi_field(
            "data",
            "account_record",
            None,
            true,
            "account data followed by permitted-growth space and alignment",
        ),
        abi_field(
            "rent_epoch",
            "account_record",
            Some(8),
            false,
            "account rent epoch",
        ),
        abi_field(
            "instruction_data_len",
            "instruction_suffix",
            Some(8),
            false,
            "instruction data length",
        ),
        abi_field(
            "instruction_data",
            "instruction_suffix",
            None,
            true,
            "serialized instruction data",
        ),
        abi_field(
            "program_id",
            "input_suffix",
            Some(32),
            false,
            "currently executing program public key",
        ),
    ]
}

fn abi_field(
    name: &str,
    scope: &str,
    width: Option<u32>,
    variable_length: bool,
    description: &str,
) -> RuntimeAbiField {
    RuntimeAbiField {
        name: name.into(),
        scope: scope.into(),
        width,
        variable_length,
        description: description.into(),
    }
}

/// Auditable subset of the runtime/SDK syscall names. Unknown hashes remain visible.
pub fn syscall_definitions() -> &'static [SyscallDefinition] {
    &[
        SyscallDefinition {
            name: "sol_log_",
            category: "logging",
            argument_count: 2,
            memory_effect: "reads_buffer",
        },
        SyscallDefinition {
            name: "sol_log_64_",
            category: "logging",
            argument_count: 5,
            memory_effect: "none",
        },
        SyscallDefinition {
            name: "sol_log_compute_units_",
            category: "logging",
            argument_count: 0,
            memory_effect: "none",
        },
        SyscallDefinition {
            name: "sol_log_data",
            category: "logging",
            argument_count: 2,
            memory_effect: "reads_buffer",
        },
        SyscallDefinition {
            name: "sol_memcpy_",
            category: "memory",
            argument_count: 3,
            memory_effect: "reads_and_writes_buffers",
        },
        SyscallDefinition {
            name: "sol_memcmp_",
            category: "memory",
            argument_count: 4,
            memory_effect: "reads_buffers",
        },
        SyscallDefinition {
            name: "sol_memmove_",
            category: "memory",
            argument_count: 3,
            memory_effect: "reads_and_writes_buffers",
        },
        SyscallDefinition {
            name: "sol_memset_",
            category: "memory",
            argument_count: 3,
            memory_effect: "writes_buffer",
        },
        SyscallDefinition {
            name: "sol_invoke_signed_c",
            category: "cpi",
            argument_count: 4,
            memory_effect: "reads_and_writes_runtime_state",
        },
        SyscallDefinition {
            name: "sol_invoke_signed_rust",
            category: "cpi",
            argument_count: 4,
            memory_effect: "reads_and_writes_runtime_state",
        },
        SyscallDefinition {
            name: "sol_get_return_data",
            category: "cpi",
            argument_count: 3,
            memory_effect: "writes_buffer",
        },
        SyscallDefinition {
            name: "sol_set_return_data",
            category: "cpi",
            argument_count: 2,
            memory_effect: "reads_buffer",
        },
        SyscallDefinition {
            name: "sol_create_program_address",
            category: "pda",
            argument_count: 4,
            memory_effect: "reads_buffers",
        },
        SyscallDefinition {
            name: "sol_try_find_program_address",
            category: "pda",
            argument_count: 5,
            memory_effect: "reads_and_writes_buffers",
        },
        SyscallDefinition {
            name: "sol_get_clock_sysvar",
            category: "sysvar",
            argument_count: 1,
            memory_effect: "writes_buffer",
        },
        SyscallDefinition {
            name: "sol_get_rent_sysvar",
            category: "sysvar",
            argument_count: 1,
            memory_effect: "writes_buffer",
        },
        SyscallDefinition {
            name: "sol_keccak256",
            category: "hash",
            argument_count: 3,
            memory_effect: "reads_and_writes_buffers",
        },
        SyscallDefinition {
            name: "sol_sha256",
            category: "hash",
            argument_count: 3,
            memory_effect: "reads_and_writes_buffers",
        },
        SyscallDefinition {
            name: "sol_blake3",
            category: "hash",
            argument_count: 3,
            memory_effect: "reads_and_writes_buffers",
        },
        SyscallDefinition {
            name: "sol_secp256k1_recover",
            category: "crypto",
            argument_count: 5,
            memory_effect: "writes_buffer",
        },
        SyscallDefinition {
            name: "sol_remaining_compute_units",
            category: "runtime",
            argument_count: 0,
            memory_effect: "none",
        },
    ]
}

pub fn analyze(analysis: &StaticAnalysis) -> SemanticAnalysis {
    analyze_with_registry(analysis, &ProtocolRegistry::default())
}

pub fn analyze_with_registry(
    analysis: &StaticAnalysis,
    protocols: &ProtocolRegistry,
) -> SemanticAnalysis {
    let mut syscalls = Vec::new();
    let mut dispatch_candidates = Vec::new();
    let mut cpi_candidates = Vec::new();
    let mut pda_candidates = Vec::new();
    let mut evidence = Vec::new();
    let input_field_candidates = input_fields(analysis, &mut evidence);
    let string_candidates = string_candidates(analysis, &mut evidence);
    let constant_candidates = constant_candidates(analysis, &mut evidence);
    let runtime_error_candidates = runtime_error_candidates(analysis, &mut evidence);
    let protocol_fingerprints = protocol_fingerprints(analysis, protocols, &mut evidence);
    let stack_account_field_candidates = stack_account_fields(analysis, &mut evidence);
    let validation_checks =
        validation_checks(analysis, &stack_account_field_candidates, &mut evidence);

    for (instruction_index, instruction) in analysis.instructions.iter().enumerate() {
        if is_syscall(instruction) {
            let hash = instruction.immediate as u32;
            let definition = syscall_definitions()
                .iter()
                .find(|definition| ebpf::hash_symbol_name(definition.name.as_bytes()) == hash);
            let fact = SyscallFact {
                instruction_pc: instruction.pc,
                hash,
                name: definition.map(|value| value.name.to_owned()),
                category: definition.map(|value| value.category.to_owned()),
                argument_count: definition.map(|value| value.argument_count),
                memory_effect: definition.map(|value| value.memory_effect.to_owned()),
                status: if definition.is_some() {
                    EvidenceStatus::Fact
                } else {
                    EvidenceStatus::NotAvailable
                },
            };
            if let Some(definition) = definition.filter(|value| value.category == "cpi") {
                let (argument_registers, argument_values, argument_source_pcs) =
                    register_arguments(analysis, instruction.pc, definition.argument_count.min(5));
                cpi_candidates.push(CpiCandidate {
                    instruction_pc: instruction.pc,
                    syscall: definition.name.to_owned(),
                    target_program: None,
                    argument_registers,
                    argument_values,
                    argument_source_pcs,
                    status: EvidenceStatus::Hypothesis,
                });
            }
            if let Some(definition) = definition.filter(|value| value.category == "pda") {
                let (seed_argument_registers, seed_argument_values, seed_argument_source_pcs) =
                    register_arguments(analysis, instruction.pc, definition.argument_count.min(3));
                pda_candidates.push(PdaCandidate {
                    instruction_pc: instruction.pc,
                    syscall: definition.name.to_owned(),
                    seed_argument_registers,
                    seed_argument_values,
                    seed_argument_source_pcs,
                    bump_output_register: Some(0),
                    interpretations: vec![
                        "runtime_pda_syscall_arguments".into(),
                        "seed_buffers_require_dataflow_recovery".into(),
                        "bump_output_requires_return_value_tracking".into(),
                    ],
                    status: EvidenceStatus::Hypothesis,
                });
            }
            evidence.push(EvidenceItem {
                kind: "syscall_call".into(),
                value: serde_json::json!({
                    "instruction_pc": instruction.pc,
                    "hash": hash,
                    "name": fact.name,
                }),
                confidence: definition.map(|_| 1.0),
                status: fact.status.clone(),
                evidence: vec![instruction_ref(instruction)],
            });
            syscalls.push(fact);
        }

        if is_dispatch_compare(instruction) {
            let Some((dispatch_immediate, constant_pc)) =
                resolved_dispatch_constant(analysis, instruction_index, instruction)
            else {
                continue;
            };
            let target_pc = branch_target(instruction);
            let mut interpretations = vec!["instruction_tag".into(), "numeric_constant".into()];
            if let Some(name) = anchor_global_discriminator_name(dispatch_immediate) {
                interpretations.push(format!("anchor_global_discriminator:{name}"));
            }
            dispatch_candidates.push(DispatchCandidate {
                instruction_pc: instruction.pc,
                opcode: instruction.opcode,
                immediate: dispatch_immediate,
                target_pc,
                interpretations,
                confidence_basis: match constant_pc {
                    Some(constant_pc) => format!(
                        "conditional branch compares a register against a same-block immediate loaded at 0x{constant_pc:x}; no framework-specific label is asserted"
                    ),
                    None => "conditional branch compares an immediate; no framework-specific label is asserted".into(),
                },
            });
            evidence.push(EvidenceItem {
                kind: "dispatch_compare".into(),
                value: serde_json::json!({
                    "instruction_pc": instruction.pc,
                    "immediate": dispatch_immediate,
                    "constant_pc": constant_pc,
                    "target_pc": target_pc,
                    "interpretations": ["instruction_tag", "numeric_constant"],
                }),
                confidence: Some(0.25),
                status: EvidenceStatus::Hypothesis,
                evidence: vec![instruction_ref(instruction)],
            });
        }
    }

    let framework_candidates = framework_candidates(analysis, &dispatch_candidates, &mut evidence);
    let instruction_candidates = instruction_candidates(&dispatch_candidates);
    let argument_candidates = argument_candidates(analysis);
    let account_candidates = account_candidates(&input_field_candidates);
    let account_field_candidates = account_field_candidates(&input_field_candidates);
    let (error_candidates, log_candidates) = string_role_candidates(analysis);
    let resolved_log_calls =
        resolved_log_calls(analysis, &syscalls, &argument_candidates, &mut evidence);
    SemanticAnalysis {
        schema_version: SEMANTIC_SCHEMA_VERSION,
        syscalls,
        dispatch_candidates,
        instruction_candidates,
        argument_candidates,
        cpi_candidates,
        pda_candidates,
        framework_candidates,
        input_field_candidates,
        account_candidates,
        account_field_candidates,
        stack_account_field_candidates,
        runtime_abi: runtime_abi(),
        string_candidates,
        constant_candidates,
        runtime_error_candidates,
        protocol_fingerprints,
        validation_checks,
        error_candidates,
        log_candidates,
        resolved_log_calls,
        evidence,
    }
}

fn runtime_error_candidates(
    analysis: &StaticAnalysis,
    evidence: &mut Vec<EvidenceItem>,
) -> Vec<RuntimeErrorCandidate> {
    analysis
        .immediate_constants
        .iter()
        .filter_map(|constant| {
            let value = u64::try_from(constant.value).ok()?;
            let name = solana_builtin_error_name(value)?;
            let evidence_refs = vec![constant_ref(constant)];
            let candidate = RuntimeErrorCandidate {
                instruction_pc: constant.pc,
                value,
                name: name.into(),
                status: EvidenceStatus::Recovered,
                evidence: evidence_refs.clone(),
            };
            evidence.push(EvidenceItem {
                kind: "solana_builtin_error".into(),
                value: serde_json::json!({
                    "instruction_pc": candidate.instruction_pc,
                    "value": candidate.value,
                    "name": candidate.name,
                }),
                confidence: Some(1.0),
                status: EvidenceStatus::Recovered,
                evidence: evidence_refs,
            });
            Some(candidate)
        })
        .collect()
}

fn solana_builtin_error_name(value: u64) -> Option<&'static str> {
    Some(match value {
        0x1_0000_0000 => "ERROR_CUSTOM_ZERO",
        0x2_0000_0000 => "ERROR_INVALID_ARGUMENT",
        0x3_0000_0000 => "ERROR_INVALID_INSTRUCTION_DATA",
        0x4_0000_0000 => "ERROR_INVALID_ACCOUNT_DATA",
        0x5_0000_0000 => "ERROR_ACCOUNT_DATA_TOO_SMALL",
        0x6_0000_0000 => "ERROR_INSUFFICIENT_FUNDS",
        0x7_0000_0000 => "ERROR_INCORRECT_PROGRAM_ID",
        0x8_0000_0000 => "ERROR_MISSING_REQUIRED_SIGNATURES",
        0x9_0000_0000 => "ERROR_ACCOUNT_ALREADY_INITIALIZED",
        0xa_0000_0000 => "ERROR_UNINITIALIZED_ACCOUNT",
        0xb_0000_0000 => "ERROR_NOT_ENOUGH_ACCOUNT_KEYS",
        0xc_0000_0000 => "ERROR_ACCOUNT_BORROW_FAILED",
        0xd_0000_0000 => "ERROR_MAX_SEED_LENGTH_EXCEEDED",
        0xe_0000_0000 => "ERROR_INVALID_SEEDS",
        0xf_0000_0000 => "ERROR_BORSH_IO_ERROR",
        0x10_0000_0000 => "ERROR_ACCOUNT_NOT_RENT_EXEMPT",
        0x11_0000_0000 => "ERROR_UNSUPPORTED_SYSVAR",
        0x12_0000_0000 => "ERROR_ILLEGAL_OWNER",
        0x13_0000_0000 => "ERROR_MAX_ACCOUNTS_DATA_ALLOCATIONS_EXCEEDED",
        0x14_0000_0000 => "ERROR_INVALID_ACCOUNT_DATA_REALLOC",
        0x15_0000_0000 => "ERROR_MAX_INSTRUCTION_TRACE_LENGTH_EXCEEDED",
        0x16_0000_0000 => "ERROR_BUILTIN_PROGRAMS_MUST_CONSUME_COMPUTE_UNITS",
        0x17_0000_0000 => "ERROR_INVALID_ACCOUNT_OWNER",
        0x18_0000_0000 => "ERROR_ARITHMETIC_OVERFLOW",
        0x19_0000_0000 => "ERROR_IMMUTABLE",
        0x1a_0000_0000 => "ERROR_INCORRECT_AUTHORITY",
        _ => return None,
    })
}

fn resolved_log_calls(
    analysis: &StaticAnalysis,
    syscalls: &[SyscallFact],
    arguments: &[ArgumentCandidate],
    evidence: &mut Vec<EvidenceItem>,
) -> Vec<ResolvedLogCall> {
    let mut calls = Vec::new();
    for syscall in syscalls
        .iter()
        .filter(|syscall| syscall.name.as_deref() == Some("sol_log_"))
    {
        let pointer = arguments.iter().find(|argument| {
            argument.instruction_pc == syscall.instruction_pc && argument.register == 1
        });
        let length = arguments.iter().find(|argument| {
            argument.instruction_pc == syscall.instruction_pc && argument.register == 2
        });
        let (Some(pointer), Some(length)) = (pointer, length) else {
            continue;
        };
        let (Some(pointer_value), Some(length_value)) = (pointer.value, length.value) else {
            continue;
        };
        let (Ok(pointer_value), Ok(length_value)) =
            (u64::try_from(pointer_value), usize::try_from(length_value))
        else {
            continue;
        };
        // Modern SBPF maps the loaded ELF image (text plus read-only data) at
        // MM_BYTECODE_START. StringFact offsets are relative to that image.
        let Some(string_offset) = pointer_value.checked_sub(ebpf::MM_BYTECODE_START) else {
            continue;
        };
        let Ok(string_offset) = usize::try_from(string_offset) else {
            continue;
        };
        let Some(string) = analysis
            .strings
            .iter()
            .find(|string| string.offset == string_offset && string.value.len() == length_value)
        else {
            continue;
        };
        let (Some(pointer_source_pc), Some(length_source_pc)) =
            (pointer.source_pc, length.source_pc)
        else {
            continue;
        };
        let mut evidence_refs = vec![string_ref(string)];
        for pc in [pointer_source_pc, length_source_pc, syscall.instruction_pc] {
            if let Some(instruction) = analysis
                .instructions
                .iter()
                .find(|instruction| instruction.pc == pc)
            {
                evidence_refs.push(instruction_ref(instruction));
            }
        }
        let call = ResolvedLogCall {
            instruction_pc: syscall.instruction_pc,
            syscall: "sol_log_".into(),
            pointer: pointer_value,
            length: length_value,
            string_offset,
            value: string.value.clone(),
            pointer_source_pc,
            length_source_pc,
            status: EvidenceStatus::Recovered,
            evidence: evidence_refs.clone(),
        };
        evidence.push(EvidenceItem {
            kind: "resolved_log_call".into(),
            value: serde_json::json!({
                "instruction_pc": call.instruction_pc,
                "syscall": call.syscall,
                "pointer": call.pointer,
                "length": call.length,
                "string_offset": call.string_offset,
                "value": call.value,
                "pointer_source_pc": call.pointer_source_pc,
                "length_source_pc": call.length_source_pc,
            }),
            confidence: Some(1.0),
            status: EvidenceStatus::Recovered,
            evidence: evidence_refs,
        });
        calls.push(call);
    }
    calls
}

fn protocol_fingerprints(
    analysis: &StaticAnalysis,
    protocols: &ProtocolRegistry,
    evidence: &mut Vec<EvidenceItem>,
) -> Vec<ProtocolFingerprint> {
    let mut fingerprints = Vec::new();
    for candidate in &analysis.public_key_candidates {
        for definition in protocols.definitions() {
            if candidate.value != definition.program_id {
                continue;
            }
            let reference = EvidenceRef {
                source: "ro_section".into(),
                location: format!("offset:0x{:x}", candidate.offset),
                detail: Some(
                    "exact 32-byte read-only ELF value matched a known Solana program ID".into(),
                ),
            };
            let fingerprint = ProtocolFingerprint {
                program_id: definition.program_id.clone(),
                name: definition.name.clone(),
                category: definition.category.clone(),
                byte_offset: candidate.offset,
                interpretations: vec![
                    "embedded_cpi_target_candidate".into(),
                    "embedded_account_key_candidate".into(),
                    "known_program_id_match".into(),
                ],
                status: EvidenceStatus::Recovered,
                evidence: vec![reference.clone()],
            };
            evidence.push(EvidenceItem {
                kind: "known_program_id".into(),
                value: serde_json::json!({
                    "program_id": definition.program_id,
                    "name": definition.name,
                    "category": definition.category,
                    "byte_offset": candidate.offset,
                }),
                confidence: Some(1.0),
                status: EvidenceStatus::Recovered,
                evidence: vec![reference],
            });
            if !fingerprints.iter().any(|item: &ProtocolFingerprint| {
                item.program_id == fingerprint.program_id
                    && item.byte_offset == fingerprint.byte_offset
            }) {
                fingerprints.push(fingerprint);
            }
        }
    }
    fingerprints
}

fn builtin_protocols() -> &'static [(&'static str, &'static str, &'static str)] {
    &[
        (
            "11111111111111111111111111111111",
            "System Program",
            "system",
        ),
        (
            "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA",
            "SPL Token Program",
            "token",
        ),
        (
            "TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb",
            "SPL Token-2022 Program",
            "token",
        ),
        (
            "ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL",
            "Associated Token Account Program",
            "token",
        ),
        (
            "ComputeBudget111111111111111111111111111111",
            "Compute Budget Program",
            "runtime",
        ),
        (
            "MemoSq4gqABAXKb96qnH8TysNcWxMyWCqXgDLGmfcHr",
            "SPL Memo Program",
            "utility",
        ),
        (
            "metaqbxxUerdq28cj1RbAWkYQm3ybzjb6a8bt518x1s",
            "Metaplex Token Metadata Program",
            "metadata",
        ),
    ]
}

fn validation_checks(
    analysis: &StaticAnalysis,
    stack_fields: &[StackAccountFieldCandidate],
    evidence: &mut Vec<EvidenceItem>,
) -> Vec<ValidationCheckCandidate> {
    let instruction_indices = analysis
        .instructions
        .iter()
        .enumerate()
        .map(|(index, instruction)| (instruction.pc, index))
        .collect::<BTreeMap<_, _>>();
    let mut checks = Vec::new();
    for load in analysis.memory_accesses.iter().filter(|access| {
        access.kind == "load"
            && (access.base_register == 1 || access.base_register == ebpf::FRAME_PTR_REG as u8)
    }) {
        let Some(value_register) = load.value_register else {
            continue;
        };
        let Some(&load_index) = instruction_indices.get(&load.pc) else {
            continue;
        };
        let Some(load_block) = block_for_instruction(analysis, load_index) else {
            continue;
        };
        for branch in analysis.instructions.iter().skip(load_index + 1).take(4) {
            let Some(&branch_index) = instruction_indices.get(&branch.pc) else {
                continue;
            };
            if block_for_instruction(analysis, branch_index).map(|block| block.start_pc)
                != Some(load_block.start_pc)
            {
                break;
            }
            if !is_dispatch_compare(branch) || branch.dst != value_register {
                continue;
            }
            let comparison = comparison_name(branch);
            let mut interpretations = vec!["conditional_guard_candidate".into()];
            if load.base_register == 1 {
                interpretations.push("runtime_input_validation_candidate".into());
            } else {
                interpretations.push("stack_materialized_runtime_field_candidate".into());
            }
            if load.width == 1 {
                interpretations.extend([
                    "duplicate_marker_candidate".into(),
                    "signer_flag_candidate".into(),
                    "writable_flag_candidate".into(),
                    "executable_flag_candidate".into(),
                ]);
            }
            if load.width == 4 || load.width == 8 {
                interpretations.extend([
                    "account_length_candidate".into(),
                    "lamports_or_runtime_scalar_candidate".into(),
                ]);
            }
            let compared_value = is_dispatch_immediate_compare(branch).then_some(branch.immediate);
            let nearby_error = analysis
                .immediate_constants
                .iter()
                .filter(|constant| constant.pc >= load_block.start_pc && constant.pc <= branch.pc)
                .filter_map(|constant| {
                    let value = u64::try_from(constant.value).ok()?;
                    Some((constant, solana_builtin_error_name(value)?))
                })
                .max_by_key(|(constant, _)| constant.pc);
            let nearby_runtime_error = nearby_error.map(|(_, name)| name.to_owned());
            if let Some(name) = nearby_runtime_error.as_deref() {
                interpretations.push(format!("nearby_return_candidate:{name}"));
            }
            let stack_field = (load.base_register == ebpf::FRAME_PTR_REG as u8)
                .then(|| {
                    stack_fields
                        .iter()
                        .find(|field| field.frame_offset == load.offset)
                })
                .flatten();
            let account_index = stack_field.map(|field| field.account_index);
            let account_field = stack_field.map(|field| field.field.clone());
            if let Some(field) = stack_field {
                interpretations.push(format!(
                    "recovered_account_field:account[{}].{}",
                    field.account_index, field.field
                ));
            }
            let mut evidence_refs = vec![memory_ref(load), instruction_ref(branch)];
            if let Some((constant, _)) = nearby_error {
                evidence_refs.push(constant_ref(constant));
            }
            let check = ValidationCheckCandidate {
                load_pc: load.pc,
                branch_pc: branch.pc,
                base_register: load.base_register,
                offset: load.offset,
                width: load.width,
                comparison,
                compared_value,
                nearby_runtime_error,
                account_index,
                account_field,
                interpretations,
                status: EvidenceStatus::Hypothesis,
                evidence: evidence_refs.clone(),
            };
            evidence.push(EvidenceItem {
                kind: "account_validation_check_candidate".into(),
                value: serde_json::json!({
                    "load_pc": load.pc,
                    "branch_pc": branch.pc,
                    "base_register": load.base_register,
                    "offset": load.offset,
                    "width": load.width,
                    "comparison": check.comparison,
                    "compared_value": check.compared_value,
                    "nearby_runtime_error": check.nearby_runtime_error,
                    "account_index": check.account_index,
                    "account_field": check.account_field,
                    "interpretations": check.interpretations,
                }),
                confidence: Some(0.45),
                status: EvidenceStatus::Hypothesis,
                evidence: evidence_refs,
            });
            checks.push(check);
            break;
        }
    }
    checks
}

fn stack_account_fields(
    analysis: &StaticAnalysis,
    evidence: &mut Vec<EvidenceItem>,
) -> Vec<StackAccountFieldCandidate> {
    const LAYOUT: [(&str, i16, u8); 9] = [
        ("key", 0, 8),
        ("lamports", 8, 8),
        ("data_len", 16, 8),
        ("data", 24, 8),
        ("owner", 32, 8),
        ("rent_epoch", 40, 8),
        ("is_signer", 48, 1),
        ("is_writable", 49, 1),
        ("executable", 50, 1),
    ];
    let stores = analysis
        .stack_accesses
        .iter()
        .filter(|access| access.kind == "store")
        .collect::<Vec<_>>();
    let mut bases = BTreeMap::<i16, usize>::new();
    for access in &stores {
        for (_, field_offset, _) in LAYOUT {
            let Some(base) = access.offset.checked_sub(field_offset) else {
                continue;
            };
            let complete = LAYOUT.iter().all(|(_, offset, width)| {
                base.checked_add(*offset).is_some_and(|expected| {
                    stores
                        .iter()
                        .any(|store| store.offset == expected && store.width == *width)
                })
            });
            if complete {
                bases.entry(base).or_default();
            }
        }
    }
    for (index, account_index) in bases.values_mut().enumerate() {
        *account_index = index;
    }

    let mut candidates = Vec::new();
    for (base, account_index) in bases {
        for (field, field_offset, width) in LAYOUT {
            let Some(frame_offset) = base.checked_add(field_offset) else {
                continue;
            };
            let matching = analysis
                .stack_accesses
                .iter()
                .filter(|access| access.offset == frame_offset && access.width == width)
                .collect::<Vec<_>>();
            if matching.is_empty() {
                continue;
            }
            let evidence_refs = matching
                .iter()
                .filter_map(|access| {
                    analysis
                        .instructions
                        .iter()
                        .find(|instruction| instruction.pc == access.pc)
                        .map(instruction_ref)
                })
                .collect::<Vec<_>>();
            let candidate = StackAccountFieldCandidate {
                account_index,
                field: field.into(),
                frame_offset,
                width,
                access_pcs: matching.iter().map(|access| access.pc).collect(),
                status: EvidenceStatus::Recovered,
                evidence: evidence_refs.clone(),
            };
            evidence.push(EvidenceItem {
                kind: "stack_account_field".into(),
                value: serde_json::json!({
                    "account_index": candidate.account_index,
                    "field": candidate.field,
                    "frame_offset": candidate.frame_offset,
                    "width": candidate.width,
                    "access_pcs": candidate.access_pcs,
                }),
                confidence: Some(0.95),
                status: EvidenceStatus::Recovered,
                evidence: evidence_refs,
            });
            candidates.push(candidate);
        }
    }
    candidates
}

fn block_for_instruction(
    analysis: &StaticAnalysis,
    instruction_index: usize,
) -> Option<&sos_analysis::BasicBlockFact> {
    analysis.basic_blocks.iter().find(|block| {
        instruction_index < block.end_instruction_index
            && analysis
                .instructions
                .get(instruction_index)
                .is_some_and(|instruction| instruction.pc >= block.start_pc)
    })
}

fn comparison_name(instruction: &InstructionFact) -> String {
    instruction
        .assembly
        .split_whitespace()
        .next()
        .unwrap_or("conditional_branch")
        .into()
}

fn instruction_candidates(dispatches: &[DispatchCandidate]) -> Vec<InstructionCandidate> {
    dispatches
        .iter()
        .map(|dispatch| InstructionCandidate {
            instruction_pc: dispatch.instruction_pc,
            tag_value: dispatch.immediate,
            target_pc: dispatch.target_pc,
            interpretations: dispatch.interpretations.clone(),
            status: EvidenceStatus::Hypothesis,
            evidence: vec![EvidenceRef {
                source: "analysis".into(),
                location: format!("instruction:0x{:x}", dispatch.instruction_pc),
                detail: Some(dispatch.confidence_basis.clone()),
            }],
        })
        .collect()
}

fn argument_candidates(analysis: &StaticAnalysis) -> Vec<ArgumentCandidate> {
    analysis
        .register_uses
        .iter()
        .take(1_024)
        .map(|use_fact| ArgumentCandidate {
            instruction_pc: use_fact.pc,
            register: use_fact.register,
            value: use_fact.value,
            source_pc: use_fact.source_pc,
            interpretations: vec![
                "call_argument_candidate".into(),
                "syscall_or_internal_call_argument".into(),
            ],
            status: if use_fact.value.is_some() {
                EvidenceStatus::Recovered
            } else {
                EvidenceStatus::Hypothesis
            },
            evidence: vec![EvidenceRef {
                source: "analysis".into(),
                location: format!("instruction:0x{:x}", use_fact.pc),
                detail: Some(format!("r{} at {}", use_fact.register, use_fact.use_kind)),
            }],
        })
        .collect()
}

fn account_candidates(fields: &[InputFieldCandidate]) -> Vec<AccountCandidate> {
    let mut bases = BTreeMap::<u8, Vec<EvidenceRef>>::new();
    for field in fields {
        bases
            .entry(field.base_register)
            .or_default()
            .push(EvidenceRef {
                source: "analysis".into(),
                location: format!("memory:r{}+{}", field.base_register, field.offset),
                detail: Some("direct entrypoint-input access".into()),
            });
    }
    bases
        .into_iter()
        .map(|(base_register, evidence)| AccountCandidate {
            base_register,
            interpretations: vec![
                "account_array_or_runtime_input_base".into(),
                "raw_entrypoint_pointer".into(),
            ],
            status: EvidenceStatus::Hypothesis,
            evidence,
        })
        .collect()
}

fn account_field_candidates(fields: &[InputFieldCandidate]) -> Vec<AccountFieldCandidate> {
    fields
        .iter()
        .map(|field| AccountFieldCandidate {
            base_register: field.base_register,
            offset: field.offset,
            widths: field.widths.clone(),
            interpretations: field.interpretations.clone(),
            status: field.status.clone(),
            evidence: vec![EvidenceRef {
                source: "analysis".into(),
                location: format!("memory:r{}+{}", field.base_register, field.offset),
                detail: Some(format!(
                    "{} load/store observations",
                    field.load_count + field.store_count
                )),
            }],
        })
        .collect()
}

fn string_role_candidates(analysis: &StaticAnalysis) -> (Vec<ErrorCandidate>, Vec<LogCandidate>) {
    let mut errors = Vec::new();
    let mut logs = Vec::new();
    for string in &analysis.strings {
        let lower = string.value.to_ascii_lowercase();
        let looks_like_error = ["error", "failed", "invalid", "panic", "assert"]
            .iter()
            .any(|marker| lower.contains(marker));
        let evidence = vec![string_ref(string)];
        if looks_like_error {
            errors.push(ErrorCandidate {
                offset: string.offset,
                value: string.value.clone(),
                status: EvidenceStatus::Hypothesis,
                evidence,
            });
        } else {
            logs.push(LogCandidate {
                offset: string.offset,
                value: string.value.clone(),
                status: EvidenceStatus::Hypothesis,
                evidence,
            });
        }
    }
    (errors, logs)
}

fn register_arguments(
    analysis: &StaticAnalysis,
    instruction_pc: usize,
    count: u8,
) -> (Vec<u8>, Vec<Option<i64>>, Vec<Option<usize>>) {
    let registers: Vec<u8> = (1..=count).collect();
    let mut values = Vec::with_capacity(registers.len());
    let mut source_pcs = Vec::with_capacity(registers.len());
    for register in &registers {
        let fact = analysis
            .register_uses
            .iter()
            .find(|fact| fact.pc == instruction_pc && fact.register == *register);
        values.push(fact.and_then(|fact| fact.value));
        source_pcs.push(fact.and_then(|fact| fact.source_pc));
    }
    (registers, values, source_pcs)
}

fn framework_candidates(
    analysis: &StaticAnalysis,
    dispatch_candidates: &[DispatchCandidate],
    evidence: &mut Vec<EvidenceItem>,
) -> Vec<FrameworkCandidate> {
    let mut candidates = Vec::new();
    for dispatch in dispatch_candidates {
        let early = dispatch.instruction_pc <= analysis.entrypoint.saturating_add(256);
        let eight_byte_compare = matches!(
            dispatch.opcode,
            ebpf::JEQ64_IMM
                | ebpf::JGT64_IMM
                | ebpf::JGE64_IMM
                | ebpf::JLT64_IMM
                | ebpf::JLE64_IMM
                | ebpf::JSET64_IMM
                | ebpf::JNE64_IMM
                | ebpf::JSGT64_IMM
                | ebpf::JSGE64_IMM
                | ebpf::JSLT64_IMM
                | ebpf::JSLE64_IMM
                | ebpf::JEQ64_REG
                | ebpf::JGT64_REG
                | ebpf::JGE64_REG
                | ebpf::JLT64_REG
                | ebpf::JLE64_REG
                | ebpf::JSET64_REG
                | ebpf::JNE64_REG
                | ebpf::JSGT64_REG
                | ebpf::JSGE64_REG
                | ebpf::JSLT64_REG
                | ebpf::JSLE64_REG
        );
        let anchor_name = anchor_global_discriminator_name(dispatch.immediate);
        if (early || anchor_name.is_some()) && eight_byte_compare {
            let (framework, confidence, detail) = match anchor_name {
                Some(name) => (
                    "Anchor-like discriminator dispatch",
                    0.78,
                    format!("exact SHA-256 global discriminator match for `{name}`"),
                ),
                None => (
                    "Anchor-like discriminator dispatch",
                    0.15,
                    "early 64-bit immediate conditional comparison".into(),
                ),
            };
            let mut interpretations = vec![
                "Anchor_instruction_discriminator_candidate".into(),
                "native_instruction_tag".into(),
                "ordinary_numeric_comparison".into(),
            ];
            if let Some(name) = anchor_name {
                interpretations.push(format!("matched_anchor_global:{name}"));
            }
            let candidate = FrameworkCandidate {
                framework: framework.into(),
                instruction_pc: Some(dispatch.instruction_pc),
                confidence,
                interpretations,
                status: EvidenceStatus::Hypothesis,
            };
            evidence.push(EvidenceItem {
                kind: "framework_candidate".into(),
                value: serde_json::json!({
                    "framework": candidate.framework,
                    "instruction_pc": dispatch.instruction_pc,
                    "confidence": candidate.confidence,
                    "interpretations": &candidate.interpretations,
                }),
                confidence: Some(candidate.confidence),
                status: EvidenceStatus::Hypothesis,
                evidence: vec![EvidenceRef {
                    source: "analysis".into(),
                    location: format!("instruction:0x{:x}", dispatch.instruction_pc),
                    detail: Some(detail),
                }],
            });
            candidates.push(candidate);
        }
    }
    candidates
}

/// Match only a small, explicit set of common Anchor global instruction
/// discriminators. A match is still a hypothesis because the same eight bytes
/// can be intentionally reused by native code; all alternative interpretations
/// remain attached to the dispatch candidate.
fn anchor_global_discriminator_name(immediate: i64) -> Option<&'static str> {
    [
        "initialize",
        "swap",
        "deposit",
        "withdraw",
        "transfer",
        "create",
        "set_value",
    ]
    .into_iter()
    .find(|name| anchor_global_discriminator(name) == immediate)
}

fn anchor_global_discriminator(name: &str) -> i64 {
    let mut hasher = Sha256::new();
    hasher.update(format!("global:{name}").as_bytes());
    let digest = hasher.finalize();
    i64::from_le_bytes(digest[..8].try_into().expect("SHA-256 has eight bytes"))
}

/// Match only a bounded set of common Anchor account names. This is a
/// candidate label, not proof that the bytes are an account discriminator;
/// numeric and pointer interpretations remain attached to the same constant.
fn anchor_account_discriminator_name(immediate: i64) -> Option<&'static str> {
    ["State", "Config", "Pool", "Vault", "User", "Account"]
        .into_iter()
        .find(|name| anchor_account_discriminator(name) == immediate)
}

fn anchor_account_discriminator(name: &str) -> i64 {
    let mut hasher = Sha256::new();
    hasher.update(format!("account:{name}").as_bytes());
    let digest = hasher.finalize();
    i64::from_le_bytes(digest[..8].try_into().expect("SHA-256 has eight bytes"))
}

fn string_candidates(
    analysis: &StaticAnalysis,
    evidence: &mut Vec<EvidenceItem>,
) -> Vec<StringCandidate> {
    analysis
        .strings
        .iter()
        .map(|string| {
            let candidate = StringCandidate {
                offset: string.offset,
                value: string.value.clone(),
                interpretations: vec![
                    "log_message".into(),
                    "error_text".into(),
                    "serialized_or_embedded_data".into(),
                ],
                status: EvidenceStatus::Fact,
            };
            evidence.push(EvidenceItem {
                kind: "readonly_string".into(),
                value: serde_json::json!({
                    "offset": string.offset,
                    "value": string.value,
                }),
                confidence: Some(1.0),
                status: EvidenceStatus::Fact,
                evidence: vec![string_ref(string)],
            });
            candidate
        })
        .collect()
}

fn constant_candidates(
    analysis: &StaticAnalysis,
    evidence: &mut Vec<EvidenceItem>,
) -> Vec<ConstantCandidate> {
    analysis
        .immediate_constants
        .iter()
        .map(|constant| {
            let account_name = anchor_account_discriminator_name(constant.value);
            let mut interpretations = vec![
                "numeric_constant".into(),
                "instruction_tag_or_discriminator_candidate".into(),
                "pointer_or_address_candidate".into(),
            ];
            if let Some(name) = account_name {
                interpretations.push("anchor_account_discriminator_candidate".into());
                interpretations.push(format!("matched_anchor_account:{name}"));
            }
            let candidate = ConstantCandidate {
                pc: constant.pc,
                value: constant.value,
                kind: constant.kind.clone(),
                interpretations,
                status: EvidenceStatus::Hypothesis,
            };
            evidence.push(EvidenceItem {
                kind: "immediate_constant".into(),
                value: serde_json::json!({
                    "pc": constant.pc,
                    "value": constant.value,
                    "kind": constant.kind,
                    "interpretations": candidate.interpretations,
                }),
                confidence: Some(1.0),
                status: EvidenceStatus::Fact,
                evidence: vec![constant_ref(constant)],
            });
            candidate
        })
        .collect()
}

fn input_fields(
    analysis: &StaticAnalysis,
    evidence: &mut Vec<EvidenceItem>,
) -> Vec<InputFieldCandidate> {
    let mut grouped: BTreeMap<(u8, i16), Vec<&MemoryAccessFact>> = BTreeMap::new();
    for access in &analysis.memory_accesses {
        // r1 is the conventional Solana entrypoint input pointer. We only label
        // it as an input root; exact account offsets require a runtime ABI model.
        if access.base_register == 1 {
            grouped
                .entry((access.base_register, access.offset))
                .or_default()
                .push(access);
        }
    }
    grouped
        .into_iter()
        .map(|((base_register, offset), accesses)| {
            let mut widths = accesses
                .iter()
                .map(|access| access.width)
                .collect::<Vec<_>>();
            widths.sort_unstable();
            widths.dedup();
            let load_count = accesses
                .iter()
                .filter(|access| access.kind == "load")
                .count();
            let store_count = accesses
                .iter()
                .filter(|access| access.kind == "store")
                .count();
            let candidate = InputFieldCandidate {
                base_register,
                offset,
                widths: widths.clone(),
                load_count,
                store_count,
                interpretations: interpretations_for(offset, &widths),
                status: EvidenceStatus::Hypothesis,
            };
            evidence.push(EvidenceItem {
                kind: "solana_input_field_candidate".into(),
                value: serde_json::json!({
                    "base_register": base_register,
                    "offset": offset,
                    "widths": &candidate.widths,
                    "load_count": load_count,
                    "store_count": store_count,
                    "interpretations": &candidate.interpretations,
                }),
                confidence: Some(0.25),
                status: EvidenceStatus::Hypothesis,
                evidence: accesses.iter().map(|access| memory_ref(access)).collect(),
            });
            candidate
        })
        .collect()
}

fn interpretations_for(offset: i16, widths: &[u8]) -> Vec<String> {
    if offset == 0 && widths.contains(&8) {
        vec![
            "account_count".into(),
            "runtime_input_field".into(),
            "numeric_constant_or_pointer".into(),
        ]
    } else {
        vec![
            "runtime_input_field".into(),
            "account_record_or_variable_length_region".into(),
            "instruction_data_or_program_id_region".into(),
        ]
    }
}

fn is_syscall(instruction: &InstructionFact) -> bool {
    // The static-syscall marker differs across SBPF versions. The disassembler
    // already applies the version-specific rule, so use its explicit external
    // call classification instead of assuming one source-register value.
    instruction.opcode == ebpf::CALL_IMM && instruction.assembly.starts_with("syscall ")
}

fn is_dispatch_compare(instruction: &InstructionFact) -> bool {
    is_dispatch_immediate_compare(instruction) || is_dispatch_register_compare(instruction)
}

fn is_dispatch_immediate_compare(instruction: &InstructionFact) -> bool {
    matches!(
        instruction.opcode,
        ebpf::JEQ32_IMM
            | ebpf::JGT32_IMM
            | ebpf::JGE32_IMM
            | ebpf::JLT32_IMM
            | ebpf::JLE32_IMM
            | ebpf::JSET32_IMM
            | ebpf::JNE32_IMM
            | ebpf::JSGT32_IMM
            | ebpf::JSGE32_IMM
            | ebpf::JSLT32_IMM
            | ebpf::JSLE32_IMM
            | ebpf::JEQ64_IMM
            | ebpf::JGT64_IMM
            | ebpf::JGE64_IMM
            | ebpf::JLT64_IMM
            | ebpf::JLE64_IMM
            | ebpf::JSET64_IMM
            | ebpf::JNE64_IMM
            | ebpf::JSGT64_IMM
            | ebpf::JSGE64_IMM
            | ebpf::JSLT64_IMM
            | ebpf::JSLE64_IMM
    )
}

fn is_dispatch_register_compare(instruction: &InstructionFact) -> bool {
    matches!(
        instruction.opcode,
        ebpf::JEQ32_REG
            | ebpf::JGT32_REG
            | ebpf::JGE32_REG
            | ebpf::JLT32_REG
            | ebpf::JLE32_REG
            | ebpf::JSET32_REG
            | ebpf::JNE32_REG
            | ebpf::JSGT32_REG
            | ebpf::JSGE32_REG
            | ebpf::JSLT32_REG
            | ebpf::JSLE32_REG
            | ebpf::JEQ64_REG
            | ebpf::JGT64_REG
            | ebpf::JGE64_REG
            | ebpf::JLT64_REG
            | ebpf::JLE64_REG
            | ebpf::JSET64_REG
            | ebpf::JNE64_REG
            | ebpf::JSGT64_REG
            | ebpf::JSGE64_REG
            | ebpf::JSLT64_REG
            | ebpf::JSLE64_REG
    )
}

fn resolved_dispatch_constant(
    analysis: &StaticAnalysis,
    instruction_index: usize,
    instruction: &InstructionFact,
) -> Option<(i64, Option<usize>)> {
    if is_dispatch_immediate_compare(instruction) {
        return Some((instruction.immediate, None));
    }
    if !is_dispatch_register_compare(instruction) {
        return None;
    }
    let block = block_for_instruction(analysis, instruction_index)?;
    for (candidate_index, candidate) in analysis.instructions[..instruction_index]
        .iter()
        .enumerate()
        .rev()
        .take(32)
    {
        if block_for_instruction(analysis, candidate_index).map(|value| value.start_pc)
            != Some(block.start_pc)
        {
            break;
        }
        if candidate.opcode == ebpf::LD_DW_IMM && candidate.dst == instruction.src {
            return Some((candidate.immediate, Some(candidate.pc)));
        }
        if candidate.dst == instruction.src && writes_register(candidate) {
            break;
        }
    }
    None
}

fn writes_register(instruction: &InstructionFact) -> bool {
    let class = instruction.opcode & 0x07;
    !matches!(
        instruction.opcode,
        ebpf::CALL_IMM | ebpf::CALL_REG | ebpf::EXIT
    ) && !matches!(
        class,
        ebpf::BPF_ST | ebpf::BPF_STX | ebpf::BPF_JMP64 | ebpf::BPF_JMP32
    )
}

fn branch_target(instruction: &InstructionFact) -> usize {
    instruction
        .pc
        .saturating_add(1)
        .saturating_add_signed(instruction.offset as isize)
}

fn instruction_ref(instruction: &InstructionFact) -> EvidenceRef {
    EvidenceRef {
        source: "analysis".into(),
        location: format!("instruction:0x{:x}", instruction.pc),
        detail: Some(instruction.assembly.clone()),
    }
}

fn memory_ref(access: &MemoryAccessFact) -> EvidenceRef {
    EvidenceRef {
        source: "analysis".into(),
        location: format!("instruction:0x{:x}", access.pc),
        detail: Some(access.assembly.clone()),
    }
}

fn string_ref(string: &StringFact) -> EvidenceRef {
    EvidenceRef {
        source: "ro_section".into(),
        location: format!("offset:0x{:x}", string.offset),
        detail: Some(string.value.clone()),
    }
}

fn constant_ref(constant: &ImmediateConstantFact) -> EvidenceRef {
    EvidenceRef {
        source: "analysis".into(),
        location: format!("instruction:0x{:x}", constant.pc),
        detail: Some(constant.assembly.clone()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn syscall_hashes_are_runtime_compatible() {
        let log_hash = ebpf::hash_symbol_name(b"sol_log_");
        assert!(syscall_definitions()
            .iter()
            .any(|definition| ebpf::hash_symbol_name(definition.name.as_bytes()) == log_hash));
    }

    #[test]
    fn unknown_syscalls_are_not_silently_named() {
        let instruction = InstructionFact {
            pc: 4,
            opcode: ebpf::CALL_IMM,
            dst: 0,
            src: 0,
            offset: 0,
            immediate: 123456,
            assembly: "syscall 123456".into(),
        };
        let analysis = StaticAnalysis {
            schema_version: sos_types::ANALYSIS_SCHEMA_VERSION,
            sbpf_version: "V0".into(),
            entrypoint: 0,
            instructions: vec![instruction],
            functions: Vec::new(),
            basic_blocks: Vec::new(),
            memory_accesses: Vec::new(),
            stack_accesses: Vec::new(),
            strings: Vec::new(),
            public_key_candidates: Vec::new(),
            immediate_constants: Vec::new(),
            register_uses: Vec::new(),
            calls: Vec::new(),
            relocations: Vec::new(),
            sccs: Vec::new(),
            dataflow_edges: Vec::new(),
            verifier_diagnostics: Vec::new(),
        };
        let result = analyze(&analysis);
        assert_eq!(result.syscalls[0].name, None);
        assert_eq!(result.syscalls[0].status, EvidenceStatus::NotAvailable);
    }

    #[test]
    fn recognizes_runtime_disassembled_syscalls_with_v0_marker() {
        let instruction = InstructionFact {
            pc: 340,
            opcode: ebpf::CALL_IMM,
            dst: 0,
            src: 1,
            offset: 0,
            immediate: ebpf::hash_symbol_name(b"sol_log_") as i32 as i64,
            assembly: "syscall 544561597".into(),
        };
        let analysis = StaticAnalysis {
            schema_version: sos_types::ANALYSIS_SCHEMA_VERSION,
            sbpf_version: "V0".into(),
            entrypoint: 0,
            instructions: vec![instruction],
            functions: Vec::new(),
            basic_blocks: Vec::new(),
            memory_accesses: Vec::new(),
            stack_accesses: Vec::new(),
            strings: Vec::new(),
            public_key_candidates: Vec::new(),
            immediate_constants: Vec::new(),
            register_uses: Vec::new(),
            calls: Vec::new(),
            relocations: Vec::new(),
            sccs: Vec::new(),
            dataflow_edges: Vec::new(),
            verifier_diagnostics: Vec::new(),
        };
        let result = analyze(&analysis);
        assert_eq!(result.syscalls[0].name.as_deref(), Some("sol_log_"));
        assert_eq!(result.syscalls[0].status, EvidenceStatus::Fact);
    }

    #[test]
    fn classifies_cpi_pda_and_early_dispatch_as_candidates() {
        let cpi_hash = ebpf::hash_symbol_name(b"sol_invoke_signed_rust") as i32 as i64;
        let pda_hash = ebpf::hash_symbol_name(b"sol_create_program_address") as i32 as i64;
        let analysis = StaticAnalysis {
            schema_version: sos_types::ANALYSIS_SCHEMA_VERSION,
            sbpf_version: "V0".into(),
            entrypoint: 65,
            instructions: vec![
                InstructionFact {
                    pc: 80,
                    opcode: ebpf::CALL_IMM,
                    dst: 0,
                    src: 1,
                    offset: 0,
                    immediate: cpi_hash,
                    assembly: format!("syscall {cpi_hash}"),
                },
                InstructionFact {
                    pc: 90,
                    opcode: ebpf::CALL_IMM,
                    dst: 0,
                    src: 1,
                    offset: 0,
                    immediate: pda_hash,
                    assembly: format!("syscall {pda_hash}"),
                },
                InstructionFact {
                    pc: 100,
                    opcode: ebpf::JEQ64_IMM,
                    dst: 1,
                    src: 0,
                    offset: 4,
                    immediate: 0x1122334455667788u64 as i64,
                    assembly: "jeq r1, 0x1122334455667788, +4".into(),
                },
            ],
            functions: Vec::new(),
            basic_blocks: Vec::new(),
            memory_accesses: Vec::new(),
            stack_accesses: Vec::new(),
            strings: Vec::new(),
            public_key_candidates: Vec::new(),
            immediate_constants: Vec::new(),
            register_uses: vec![sos_analysis::RegisterUseFact {
                pc: 80,
                register: 1,
                value: Some(123),
                source_pc: Some(77),
                use_kind: "call_argument".into(),
            }],
            calls: Vec::new(),
            relocations: Vec::new(),
            sccs: Vec::new(),
            dataflow_edges: Vec::new(),
            verifier_diagnostics: Vec::new(),
        };
        let result = analyze(&analysis);
        assert_eq!(result.cpi_candidates.len(), 1);
        assert_eq!(result.pda_candidates.len(), 1);
        assert_eq!(result.framework_candidates.len(), 1);
        assert_eq!(
            result.cpi_candidates[0].argument_registers,
            vec![1, 2, 3, 4]
        );
        assert_eq!(
            result.cpi_candidates[0].argument_values,
            vec![Some(123), None, None, None]
        );
        assert_eq!(
            result.cpi_candidates[0].argument_source_pcs,
            vec![Some(77), None, None, None]
        );
        let security = detect(&analysis, &result);
        assert!(security
            .findings
            .iter()
            .any(|finding| finding.detector_id == "unresolved_cpi_target"));
        assert!(security
            .findings
            .iter()
            .any(|finding| finding.detector_id == "unresolved_pda_seeds"));
    }

    #[test]
    fn recognizes_known_anchor_global_discriminator_without_dropping_alternatives() {
        let analysis = StaticAnalysis {
            schema_version: sos_types::ANALYSIS_SCHEMA_VERSION,
            sbpf_version: "V0".into(),
            entrypoint: 0,
            instructions: vec![InstructionFact {
                pc: 16,
                opcode: ebpf::JEQ64_IMM,
                dst: 1,
                src: 0,
                offset: 4,
                immediate: anchor_global_discriminator("swap"),
                assembly: "jeq r1, anchor_swap, +4".into(),
            }],
            functions: Vec::new(),
            basic_blocks: Vec::new(),
            memory_accesses: Vec::new(),
            stack_accesses: Vec::new(),
            strings: Vec::new(),
            public_key_candidates: Vec::new(),
            immediate_constants: Vec::new(),
            register_uses: Vec::new(),
            calls: Vec::new(),
            relocations: Vec::new(),
            sccs: Vec::new(),
            dataflow_edges: Vec::new(),
            verifier_diagnostics: Vec::new(),
        };

        let result = analyze(&analysis);
        assert!(result.dispatch_candidates[0]
            .interpretations
            .iter()
            .any(|value| value == "anchor_global_discriminator:swap"));
        let candidate = &result.framework_candidates[0];
        assert_eq!(candidate.confidence, 0.78);
        assert!(candidate
            .interpretations
            .iter()
            .any(|value| value == "native_instruction_tag"));
        assert!(candidate
            .interpretations
            .iter()
            .any(|value| value == "matched_anchor_global:swap"));
    }

    #[test]
    fn resolves_register_loaded_anchor_discriminator() {
        let analysis = StaticAnalysis {
            schema_version: sos_types::ANALYSIS_SCHEMA_VERSION,
            sbpf_version: "V3".into(),
            entrypoint: 0,
            instructions: vec![
                InstructionFact {
                    pc: 12,
                    opcode: ebpf::LD_DW_IMM,
                    dst: 5,
                    src: 0,
                    offset: 0,
                    immediate: anchor_global_discriminator("initialize"),
                    assembly: "lddw r5, anchor_initialize".into(),
                },
                InstructionFact {
                    pc: 14,
                    opcode: ebpf::JEQ64_REG,
                    dst: 4,
                    src: 5,
                    offset: 4,
                    immediate: 0,
                    assembly: "jeq r4, r5, +4".into(),
                },
            ],
            functions: Vec::new(),
            basic_blocks: vec![sos_analysis::BasicBlockFact {
                start_pc: 12,
                end_instruction_index: 2,
                destinations: Vec::new(),
                label: "lbb_12".into(),
            }],
            memory_accesses: Vec::new(),
            stack_accesses: Vec::new(),
            strings: Vec::new(),
            public_key_candidates: Vec::new(),
            immediate_constants: Vec::new(),
            register_uses: Vec::new(),
            calls: Vec::new(),
            relocations: Vec::new(),
            sccs: Vec::new(),
            dataflow_edges: Vec::new(),
            verifier_diagnostics: Vec::new(),
        };

        let result = analyze(&analysis);
        assert_eq!(result.dispatch_candidates.len(), 1);
        assert_eq!(
            result.dispatch_candidates[0].immediate,
            anchor_global_discriminator("initialize")
        );
        assert!(result.dispatch_candidates[0]
            .interpretations
            .iter()
            .any(|value| value == "anchor_global_discriminator:initialize"));
        assert!(result.framework_candidates.iter().any(|candidate| {
            candidate.confidence == 0.78
                && candidate
                    .interpretations
                    .iter()
                    .any(|value| value == "matched_anchor_global:initialize")
        }));
    }

    #[test]
    fn recognizes_bounded_anchor_account_discriminator_candidate() {
        let analysis = StaticAnalysis {
            schema_version: sos_types::ANALYSIS_SCHEMA_VERSION,
            sbpf_version: "V3".into(),
            entrypoint: 0,
            instructions: Vec::new(),
            functions: Vec::new(),
            basic_blocks: Vec::new(),
            memory_accesses: Vec::new(),
            stack_accesses: Vec::new(),
            strings: Vec::new(),
            public_key_candidates: Vec::new(),
            immediate_constants: vec![ImmediateConstantFact {
                pc: 44,
                value: anchor_account_discriminator("State"),
                kind: "lddw".into(),
                assembly: "lddw r4, anchor_account_state".into(),
            }],
            register_uses: Vec::new(),
            calls: Vec::new(),
            relocations: Vec::new(),
            sccs: Vec::new(),
            dataflow_edges: Vec::new(),
            verifier_diagnostics: Vec::new(),
        };

        let result = analyze(&analysis);
        assert!(result.constant_candidates[0]
            .interpretations
            .iter()
            .any(|value| value == "anchor_account_discriminator_candidate"));
        assert!(result.constant_candidates[0]
            .interpretations
            .iter()
            .any(|value| value == "matched_anchor_account:State"));
    }

    #[test]
    fn fingerprints_known_program_ids_without_claiming_usage() {
        let analysis = StaticAnalysis {
            schema_version: sos_types::ANALYSIS_SCHEMA_VERSION,
            sbpf_version: "V0".into(),
            entrypoint: 0,
            instructions: Vec::new(),
            functions: Vec::new(),
            basic_blocks: Vec::new(),
            memory_accesses: Vec::new(),
            stack_accesses: Vec::new(),
            strings: Vec::new(),
            public_key_candidates: vec![
                sos_analysis::PublicKeyCandidate {
                    offset: 0x80,
                    value: "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA".into(),
                },
                sos_analysis::PublicKeyCandidate {
                    offset: 0xa0,
                    value: "TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb".into(),
                },
                sos_analysis::PublicKeyCandidate {
                    offset: 0xc0,
                    value: "metaqbxxUerdq28cj1RbAWkYQm3ybzjb6a8bt518x1s".into(),
                },
            ],
            immediate_constants: Vec::new(),
            register_uses: Vec::new(),
            calls: Vec::new(),
            relocations: Vec::new(),
            sccs: Vec::new(),
            dataflow_edges: Vec::new(),
            verifier_diagnostics: Vec::new(),
        };
        let result = analyze(&analysis);
        assert_eq!(result.protocol_fingerprints.len(), 3);
        assert_eq!(result.protocol_fingerprints[0].name, "SPL Token Program");
        assert_eq!(result.protocol_fingerprints[0].byte_offset, 0x80);
        assert_eq!(
            result.protocol_fingerprints[0].status,
            EvidenceStatus::Recovered
        );
        assert!(result.protocol_fingerprints[0]
            .interpretations
            .contains(&"embedded_cpi_target_candidate".into()));
        assert_eq!(
            result.protocol_fingerprints[1].name,
            "SPL Token-2022 Program"
        );
        assert_eq!(
            result.protocol_fingerprints[2].name,
            "Metaplex Token Metadata Program"
        );
    }

    #[test]
    fn custom_protocol_registry_adds_domain_specific_fingerprint() {
        let analysis = StaticAnalysis {
            schema_version: sos_types::ANALYSIS_SCHEMA_VERSION,
            sbpf_version: "V0".into(),
            entrypoint: 0,
            instructions: Vec::new(),
            functions: Vec::new(),
            basic_blocks: Vec::new(),
            memory_accesses: Vec::new(),
            stack_accesses: Vec::new(),
            strings: Vec::new(),
            public_key_candidates: vec![sos_analysis::PublicKeyCandidate {
                offset: 0x120,
                value: "Custom111111111111111111111111111111111111".into(),
            }],
            immediate_constants: Vec::new(),
            register_uses: Vec::new(),
            calls: Vec::new(),
            relocations: Vec::new(),
            sccs: Vec::new(),
            dataflow_edges: Vec::new(),
            verifier_diagnostics: Vec::new(),
        };
        let mut registry = ProtocolRegistry::new();
        registry.register(ProtocolDefinition {
            program_id: "Custom111111111111111111111111111111111111".into(),
            name: "Custom Protocol".into(),
            category: "domain_specific".into(),
        });

        let result = analyze_with_registry(&analysis, &registry);
        assert_eq!(result.protocol_fingerprints.len(), 1);
        assert_eq!(result.protocol_fingerprints[0].name, "Custom Protocol");
        assert_eq!(result.protocol_fingerprints[0].category, "domain_specific");
    }

    #[test]
    fn protocol_registry_round_trips_as_an_external_extension() {
        let registry = ProtocolRegistry::from_definitions([ProtocolDefinition {
            program_id: "Custom111111111111111111111111111111111111".into(),
            name: "Custom Protocol".into(),
            category: "domain_specific".into(),
        }]);
        let encoded = serde_json::to_vec(&registry).expect("registry serializes");
        let decoded: ProtocolRegistry =
            serde_json::from_slice(&encoded).expect("registry deserializes");
        assert_eq!(decoded, registry);
        assert_eq!(decoded.definitions()[0].name, "Custom Protocol");
    }

    #[test]
    fn recovers_local_input_load_compare_guard_as_a_candidate() {
        let analysis = StaticAnalysis {
            schema_version: sos_types::ANALYSIS_SCHEMA_VERSION,
            sbpf_version: "V0".into(),
            entrypoint: 0,
            instructions: vec![
                InstructionFact {
                    pc: 0,
                    opcode: ebpf::LD_B_REG,
                    dst: 2,
                    src: 1,
                    offset: 1,
                    immediate: 0,
                    assembly: "ldxb r2, [r1+1]".into(),
                },
                InstructionFact {
                    pc: 1,
                    opcode: ebpf::JNE64_IMM,
                    dst: 2,
                    src: 0,
                    offset: 3,
                    immediate: 0,
                    assembly: "jne r2, 0, +3".into(),
                },
            ],
            functions: Vec::new(),
            basic_blocks: vec![sos_analysis::BasicBlockFact {
                start_pc: 0,
                end_instruction_index: 2,
                destinations: vec![5],
                label: "entry".into(),
            }],
            memory_accesses: vec![MemoryAccessFact {
                pc: 0,
                kind: "load".into(),
                width: 1,
                base_register: 1,
                value_register: Some(2),
                offset: 1,
                immediate: None,
                assembly: "ldxb r2, [r1+1]".into(),
            }],
            stack_accesses: Vec::new(),
            strings: Vec::new(),
            public_key_candidates: Vec::new(),
            immediate_constants: Vec::new(),
            register_uses: Vec::new(),
            calls: Vec::new(),
            relocations: Vec::new(),
            sccs: Vec::new(),
            dataflow_edges: Vec::new(),
            verifier_diagnostics: Vec::new(),
        };
        let result = analyze(&analysis);
        assert_eq!(result.validation_checks.len(), 1);
        assert_eq!(result.validation_checks[0].load_pc, 0);
        assert_eq!(result.validation_checks[0].branch_pc, 1);
        assert!(result.validation_checks[0]
            .interpretations
            .contains(&"signer_flag_candidate".into()));
    }

    #[test]
    fn recovers_stack_materialized_flag_guard_with_nearby_runtime_error() {
        let analysis = StaticAnalysis {
            schema_version: sos_types::ANALYSIS_SCHEMA_VERSION,
            sbpf_version: "V0".into(),
            entrypoint: 0,
            instructions: vec![
                InstructionFact {
                    pc: 0,
                    opcode: ebpf::LD_DW_IMM,
                    dst: 0,
                    src: 0,
                    offset: 0,
                    immediate: 0x8_0000_0000,
                    assembly: "lddw r0, 0x800000000".into(),
                },
                InstructionFact {
                    pc: 2,
                    opcode: ebpf::LD_B_REG,
                    dst: 1,
                    src: ebpf::FRAME_PTR_REG as u8,
                    offset: -64,
                    immediate: 0,
                    assembly: "ldxb r1, [r10-64]".into(),
                },
                InstructionFact {
                    pc: 3,
                    opcode: ebpf::JNE64_IMM,
                    dst: 1,
                    src: 0,
                    offset: 1,
                    immediate: 1,
                    assembly: "jne r1, 1, +1".into(),
                },
            ],
            functions: Vec::new(),
            basic_blocks: vec![sos_analysis::BasicBlockFact {
                start_pc: 0,
                end_instruction_index: 3,
                destinations: vec![4],
                label: "entry".into(),
            }],
            memory_accesses: vec![MemoryAccessFact {
                pc: 2,
                kind: "load".into(),
                width: 1,
                base_register: ebpf::FRAME_PTR_REG as u8,
                value_register: Some(1),
                offset: -64,
                immediate: None,
                assembly: "ldxb r1, [r10-64]".into(),
            }],
            stack_accesses: Vec::new(),
            strings: Vec::new(),
            public_key_candidates: Vec::new(),
            immediate_constants: vec![ImmediateConstantFact {
                pc: 0,
                value: 0x8_0000_0000,
                kind: "lddw".into(),
                assembly: "lddw r0, 0x800000000".into(),
            }],
            register_uses: Vec::new(),
            calls: Vec::new(),
            relocations: Vec::new(),
            sccs: Vec::new(),
            dataflow_edges: Vec::new(),
            verifier_diagnostics: Vec::new(),
        };

        let result = analyze(&analysis);

        assert_eq!(result.validation_checks.len(), 1);
        assert_eq!(result.validation_checks[0].compared_value, Some(1));
        assert_eq!(
            result.validation_checks[0].nearby_runtime_error.as_deref(),
            Some("ERROR_MISSING_REQUIRED_SIGNATURES")
        );
        assert!(result.validation_checks[0]
            .interpretations
            .contains(&"stack_materialized_runtime_field_candidate".into()));
    }

    #[test]
    fn recovers_complete_sol_account_info_layout_from_stack_alias_facts() {
        let layout = [
            (0, 8),
            (8, 8),
            (16, 8),
            (24, 8),
            (32, 8),
            (40, 8),
            (48, 1),
            (49, 1),
            (50, 1),
        ];
        let stack_accesses = layout
            .into_iter()
            .enumerate()
            .map(
                |(pc, (field_offset, width))| sos_analysis::StackAccessFact {
                    pc,
                    kind: "store".into(),
                    width,
                    offset: -112 + field_offset,
                    value_register: Some(0),
                    immediate: None,
                    frame_register: ebpf::FRAME_PTR_REG as u8,
                    assembly: "stack store".into(),
                },
            )
            .collect();
        let analysis = StaticAnalysis {
            schema_version: sos_types::ANALYSIS_SCHEMA_VERSION,
            sbpf_version: "V0".into(),
            entrypoint: 0,
            instructions: Vec::new(),
            functions: Vec::new(),
            basic_blocks: Vec::new(),
            memory_accesses: Vec::new(),
            stack_accesses,
            strings: Vec::new(),
            public_key_candidates: Vec::new(),
            immediate_constants: Vec::new(),
            register_uses: Vec::new(),
            calls: Vec::new(),
            relocations: Vec::new(),
            sccs: Vec::new(),
            dataflow_edges: Vec::new(),
            verifier_diagnostics: Vec::new(),
        };
        let mut evidence = Vec::new();

        let fields = stack_account_fields(&analysis, &mut evidence);

        assert_eq!(fields.len(), 9);
        assert!(fields.iter().any(|field| {
            field.account_index == 0 && field.field == "is_signer" && field.frame_offset == -64
        }));
        assert!(fields.iter().any(|field| {
            field.account_index == 0 && field.field == "is_writable" && field.frame_offset == -63
        }));
    }

    #[test]
    fn resolves_log_syscall_pointer_and_length_to_exact_rodata_string() {
        let pointer_pc = 9;
        let length_pc = 11;
        let call_pc = 12;
        let string_offset = 0x6b0;
        let value = "SOS C fixture";
        let analysis = StaticAnalysis {
            schema_version: sos_types::ANALYSIS_SCHEMA_VERSION,
            sbpf_version: "V0".into(),
            entrypoint: 0,
            instructions: vec![
                InstructionFact {
                    pc: pointer_pc,
                    opcode: ebpf::LD_DW_IMM,
                    dst: 1,
                    src: 0,
                    offset: 0,
                    immediate: (ebpf::MM_BYTECODE_START + string_offset) as i64,
                    assembly: "lddw r1, rodata".into(),
                },
                InstructionFact {
                    pc: length_pc,
                    opcode: ebpf::MOV64_IMM,
                    dst: 2,
                    src: 0,
                    offset: 0,
                    immediate: value.len() as i64,
                    assembly: "mov64 r2, 13".into(),
                },
                InstructionFact {
                    pc: call_pc,
                    opcode: ebpf::CALL_IMM,
                    dst: 0,
                    src: 1,
                    offset: 0,
                    immediate: ebpf::hash_symbol_name(b"sol_log_") as i64,
                    assembly: "syscall sol_log_".into(),
                },
            ],
            functions: Vec::new(),
            basic_blocks: Vec::new(),
            memory_accesses: Vec::new(),
            stack_accesses: Vec::new(),
            strings: vec![StringFact {
                offset: string_offset as usize,
                value: value.into(),
            }],
            public_key_candidates: Vec::new(),
            immediate_constants: Vec::new(),
            register_uses: Vec::new(),
            calls: Vec::new(),
            relocations: Vec::new(),
            sccs: Vec::new(),
            dataflow_edges: Vec::new(),
            verifier_diagnostics: Vec::new(),
        };
        let syscalls = [SyscallFact {
            instruction_pc: call_pc,
            hash: ebpf::hash_symbol_name(b"sol_log_"),
            name: Some("sol_log_".into()),
            category: Some("logging".into()),
            argument_count: Some(2),
            memory_effect: Some("reads_buffer".into()),
            status: EvidenceStatus::Fact,
        }];
        let arguments = [
            ArgumentCandidate {
                instruction_pc: call_pc,
                register: 1,
                value: Some((ebpf::MM_BYTECODE_START + string_offset) as i64),
                source_pc: Some(pointer_pc),
                interpretations: Vec::new(),
                status: EvidenceStatus::Recovered,
                evidence: Vec::new(),
            },
            ArgumentCandidate {
                instruction_pc: call_pc,
                register: 2,
                value: Some(value.len() as i64),
                source_pc: Some(length_pc),
                interpretations: Vec::new(),
                status: EvidenceStatus::Recovered,
                evidence: Vec::new(),
            },
        ];
        let mut evidence = Vec::new();

        let calls = resolved_log_calls(&analysis, &syscalls, &arguments, &mut evidence);

        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].instruction_pc, call_pc);
        assert_eq!(calls[0].string_offset, string_offset as usize);
        assert_eq!(calls[0].value, value);
        assert_eq!(calls[0].status, EvidenceStatus::Recovered);
        assert!(evidence.iter().any(|item| item.kind == "resolved_log_call"));
    }

    #[test]
    fn names_exact_solana_builtin_return_codes_without_guessing_custom_values() {
        assert_eq!(
            solana_builtin_error_name(0x2_0000_0000),
            Some("ERROR_INVALID_ARGUMENT")
        );
        assert_eq!(
            solana_builtin_error_name(0x8_0000_0000),
            Some("ERROR_MISSING_REQUIRED_SIGNATURES")
        );
        assert_eq!(solana_builtin_error_name(42), None);
    }
}
