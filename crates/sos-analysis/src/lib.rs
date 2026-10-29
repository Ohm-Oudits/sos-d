//! Deterministic analysis pipeline contracts.

use std::{
    collections::{BTreeMap, HashSet},
    fmt,
    ptr::NonNull,
    sync::Arc,
};

use object::{Object, ObjectSection, ObjectSymbol};
use serde::{Deserialize, Serialize};
use solana_sbpf::{
    elf::Executable, memory_region::MemoryMapping, program::BuiltinProgram,
    static_analysis::Analysis, verifier::RequisiteVerifier, vm::ContextObject,
};
use sos_types::{EvidenceItem, ProgramArtifact, ANALYSIS_SCHEMA_VERSION};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnalysisRequest {
    pub include_disassembly: bool,
    pub include_cfg: bool,
    pub include_dataflow: bool,
}

impl Default for AnalysisRequest {
    fn default() -> Self {
        Self {
            include_disassembly: true,
            include_cfg: true,
            include_dataflow: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct AnalysisOutput {
    pub artifact: ProgramArtifact,
    pub evidence: Vec<EvidenceItem>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnalysisError {
    pub stage: String,
    pub message: String,
}

impl fmt::Display for AnalysisError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "analysis stage {} failed: {}", self.stage, self.message)
    }
}

impl std::error::Error for AnalysisError {}

pub trait AnalysisStage {
    fn name(&self) -> &'static str;
    fn run(
        &self,
        artifact: &ProgramArtifact,
        request: &AnalysisRequest,
    ) -> Result<Vec<EvidenceItem>, AnalysisError>;
}

pub struct AnalysisPipeline {
    stages: Vec<Box<dyn AnalysisStage>>,
}

impl AnalysisPipeline {
    pub fn new() -> Self {
        Self { stages: Vec::new() }
    }

    pub fn add_stage(&mut self, stage: Box<dyn AnalysisStage>) {
        self.stages.push(stage);
    }

    pub fn run(
        &self,
        artifact: &ProgramArtifact,
        request: &AnalysisRequest,
    ) -> Result<AnalysisOutput, AnalysisError> {
        let mut evidence = Vec::new();
        for stage in &self.stages {
            evidence.extend(stage.run(artifact, request)?);
        }
        Ok(AnalysisOutput {
            artifact: artifact.clone(),
            evidence,
        })
    }
}

impl Default for AnalysisPipeline {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Default)]
struct AnalysisContext {
    remaining: u64,
}

impl ContextObject for AnalysisContext {
    fn consume(&mut self, amount: u64) {
        self.remaining = self.remaining.saturating_sub(amount);
    }

    fn get_remaining(&self) -> u64 {
        self.remaining
    }

    fn active_mapping_ptr(&mut self) -> NonNull<MemoryMapping> {
        NonNull::dangling()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InstructionFact {
    pub pc: usize,
    pub opcode: u8,
    pub dst: u8,
    pub src: u8,
    pub offset: i16,
    pub immediate: i64,
    pub assembly: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FunctionFact {
    pub pc: usize,
    pub symbol_hash: u32,
    pub name: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BasicBlockFact {
    pub start_pc: usize,
    pub end_instruction_index: usize,
    pub destinations: Vec<usize>,
    pub label: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryAccessFact {
    pub pc: usize,
    pub kind: String,
    pub width: u8,
    pub base_register: u8,
    pub value_register: Option<u8>,
    pub offset: i16,
    pub immediate: Option<i64>,
    pub assembly: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StackAccessFact {
    pub pc: usize,
    pub kind: String,
    pub width: u8,
    pub offset: i16,
    pub value_register: Option<u8>,
    pub immediate: Option<i64>,
    pub frame_register: u8,
    pub assembly: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StringFact {
    pub offset: usize,
    pub value: String,
}

/// A 32-byte value found in a read-only ELF section and rendered as a Solana
/// base58 address. This is a byte-level candidate only; semantic layers decide
/// whether it is a program ID, account key, seed material, or unrelated data.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PublicKeyCandidate {
    pub offset: usize,
    pub value: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImmediateConstantFact {
    pub pc: usize,
    pub value: i64,
    pub kind: String,
    pub assembly: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RegisterUseFact {
    pub pc: usize,
    pub register: u8,
    pub value: Option<i64>,
    pub source_pc: Option<usize>,
    pub use_kind: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CallFact {
    pub call_pc: usize,
    pub caller_pc: Option<usize>,
    pub kind: String,
    pub target_pc: Option<usize>,
    pub target_symbol_hash: Option<u32>,
    pub status: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RelocationFact {
    pub section: String,
    pub offset: u64,
    pub kind: String,
    pub encoding: String,
    pub size: u8,
    pub target: String,
    pub addend: i64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SccFact {
    pub id: usize,
    pub block_starts: Vec<usize>,
    pub recursive: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DataflowEdgeFact {
    pub source_pc: usize,
    pub destination_pc: usize,
    pub kind: String,
    pub resource: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StaticAnalysis {
    pub schema_version: u32,
    pub sbpf_version: String,
    pub entrypoint: usize,
    pub instructions: Vec<InstructionFact>,
    pub functions: Vec<FunctionFact>,
    pub basic_blocks: Vec<BasicBlockFact>,
    pub memory_accesses: Vec<MemoryAccessFact>,
    #[serde(default)]
    pub stack_accesses: Vec<StackAccessFact>,
    pub strings: Vec<StringFact>,
    #[serde(default)]
    pub public_key_candidates: Vec<PublicKeyCandidate>,
    pub immediate_constants: Vec<ImmediateConstantFact>,
    #[serde(default)]
    pub register_uses: Vec<RegisterUseFact>,
    #[serde(default)]
    pub calls: Vec<CallFact>,
    #[serde(default)]
    pub relocations: Vec<RelocationFact>,
    #[serde(default)]
    pub sccs: Vec<SccFact>,
    #[serde(default)]
    pub dataflow_edges: Vec<DataflowEdgeFact>,
    #[serde(default)]
    pub verifier_diagnostics: Vec<String>,
}

pub const IR_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IrFunction {
    pub id: String,
    pub entry_pc: usize,
    pub name: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IrBlock {
    pub id: String,
    pub function_id: String,
    pub start_pc: usize,
    pub end_instruction_index: usize,
    pub successors: Vec<String>,
    pub scc_id: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IrOperation {
    pub id: String,
    pub block_id: String,
    pub pc: usize,
    pub kind: String,
    pub assembly: String,
    pub dst: u8,
    pub src: u8,
    pub offset: i16,
    pub immediate: i64,
    pub memory_effect: Option<String>,
    pub call_target_pc: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CanonicalIr {
    pub schema_version: u32,
    pub functions: Vec<IrFunction>,
    pub blocks: Vec<IrBlock>,
    pub operations: Vec<IrOperation>,
    pub sccs: Vec<SccFact>,
}

/// Build a deterministic, GUI-independent IR projection from the static facts.
/// IDs are derived only from bytecode PCs and sorted analysis order, so repeated
/// analysis of identical ELF bytes produces byte-identical IR JSON.
pub fn build_canonical_ir(analysis: &StaticAnalysis) -> CanonicalIr {
    let mut functions = analysis.functions.clone();
    functions.sort_by_key(|function| function.pc);
    let ir_functions = functions
        .iter()
        .map(|function| IrFunction {
            id: format!("fn_{:x}", function.pc),
            entry_pc: function.pc,
            name: function.name.clone(),
        })
        .collect::<Vec<_>>();

    let mut blocks = analysis.basic_blocks.clone();
    blocks.sort_by_key(|block| block.start_pc);
    let ir_blocks = blocks
        .iter()
        .map(|block| {
            let function_id = functions
                .iter()
                .filter(|function| function.pc <= block.start_pc)
                .max_by_key(|function| function.pc)
                .map(|function| format!("fn_{:x}", function.pc))
                .unwrap_or_else(|| "fn_unknown".into());
            IrBlock {
                id: format!("bb_{:x}", block.start_pc),
                function_id,
                start_pc: block.start_pc,
                end_instruction_index: block.end_instruction_index,
                successors: block
                    .destinations
                    .iter()
                    .map(|destination| format!("bb_{destination:x}"))
                    .collect(),
                scc_id: analysis
                    .sccs
                    .iter()
                    .find(|scc| scc.block_starts.contains(&block.start_pc))
                    .map(|scc| scc.id),
            }
        })
        .collect::<Vec<_>>();

    let mut operations = Vec::new();
    for block in &blocks {
        let Some(start_index) = analysis
            .instructions
            .iter()
            .position(|instruction| instruction.pc == block.start_pc)
        else {
            continue;
        };
        let end_index = block.end_instruction_index.min(analysis.instructions.len());
        if end_index <= start_index {
            continue;
        }
        let last_pc = analysis.instructions[end_index - 1].pc;
        for instruction in &analysis.instructions[start_index..end_index] {
            let memory = analysis
                .memory_accesses
                .iter()
                .find(|access| access.pc == instruction.pc);
            let call = analysis
                .calls
                .iter()
                .find(|call| call.call_pc == instruction.pc);
            let kind = if let Some(call) = call {
                call.kind.clone()
            } else if let Some(memory) = memory {
                memory.kind.clone()
            } else if !block.destinations.is_empty() && instruction.pc == last_pc {
                "control_flow".into()
            } else {
                "raw".into()
            };
            operations.push(IrOperation {
                id: format!("op_{:x}", instruction.pc),
                block_id: format!("bb_{:x}", block.start_pc),
                pc: instruction.pc,
                kind,
                assembly: instruction.assembly.clone(),
                dst: instruction.dst,
                src: instruction.src,
                offset: instruction.offset,
                immediate: instruction.immediate,
                memory_effect: memory.map(|memory| {
                    format!(
                        "{}:{}:{}+{}",
                        memory.kind, memory.width, memory.base_register, memory.offset
                    )
                }),
                call_target_pc: call.and_then(|call| call.target_pc),
            });
        }
    }

    CanonicalIr {
        schema_version: IR_SCHEMA_VERSION,
        functions: ir_functions,
        blocks: ir_blocks,
        operations,
        sccs: analysis.sccs.clone(),
    }
}

pub fn analyze_elf(bytes: &[u8]) -> Result<StaticAnalysis, AnalysisError> {
    let executable =
        Executable::<AnalysisContext>::from_elf(bytes, Arc::new(BuiltinProgram::new_mock()))
            .map_err(|error| AnalysisError {
                stage: "elf-load".into(),
                message: format!("{error:?}"),
            })?;
    let analysis = Analysis::from_executable(&executable).map_err(|error| AnalysisError {
        stage: "static-analysis".into(),
        message: format!("{error:?}"),
    })?;
    let verifier_diagnostics = executable
        .verify::<RequisiteVerifier>()
        .err()
        .map(|error| format!("{error:?}"))
        .into_iter()
        .collect();
    Ok(convert_analysis(
        &analysis,
        executable.get_sbpf_version(),
        bytes,
        verifier_diagnostics,
    ))
}

fn convert_analysis(
    analysis: &Analysis<'_>,
    sbpf_version: solana_sbpf::program::SBPFVersion,
    elf_bytes: &[u8],
    verifier_diagnostics: Vec<String>,
) -> StaticAnalysis {
    let instructions: Vec<InstructionFact> = analysis
        .instructions
        .iter()
        .map(|instruction| InstructionFact {
            pc: instruction.ptr,
            opcode: instruction.opc,
            dst: instruction.dst,
            src: instruction.src,
            offset: instruction.off,
            immediate: instruction.imm,
            assembly: analysis.disassemble_instruction(instruction, instruction.ptr),
        })
        .collect();
    let functions: Vec<FunctionFact> = analysis
        .functions
        .iter()
        .map(|(pc, (symbol_hash, name))| FunctionFact {
            pc: *pc,
            symbol_hash: *symbol_hash,
            name: name.clone(),
        })
        .collect();
    let basic_blocks: Vec<BasicBlockFact> = analysis
        .cfg_nodes
        .iter()
        .filter(|(pc, _)| **pc != analysis.super_root)
        .map(|(pc, block)| BasicBlockFact {
            start_pc: *pc,
            end_instruction_index: block.instructions.end,
            destinations: block.destinations.clone(),
            label: block.label.clone(),
        })
        .collect();
    let memory_accesses = instructions.iter().filter_map(memory_access).collect();
    let stack_accesses = collect_stack_accesses(&instructions, &functions);
    let immediate_constants = instructions
        .iter()
        .filter(|instruction| instruction.opcode == solana_sbpf::ebpf::LD_DW_IMM)
        .map(|instruction| ImmediateConstantFact {
            pc: instruction.pc,
            value: instruction.immediate,
            kind: "lddw".into(),
            assembly: instruction.assembly.clone(),
        })
        .collect();
    let strings = extract_elf_strings(elf_bytes);
    let public_key_candidates = extract_elf_public_keys(elf_bytes);
    let register_uses = collect_register_uses(&instructions, &basic_blocks);
    let calls = collect_calls(&instructions, &functions);
    let relocations = extract_relocations(elf_bytes);
    let (sccs, dataflow_edges) = convert_graph_facts(analysis);
    StaticAnalysis {
        schema_version: ANALYSIS_SCHEMA_VERSION,
        sbpf_version: format!("{sbpf_version:?}"),
        entrypoint: analysis.entrypoint,
        instructions,
        functions,
        basic_blocks,
        memory_accesses,
        stack_accesses,
        strings,
        public_key_candidates,
        immediate_constants,
        register_uses,
        calls,
        relocations,
        sccs,
        dataflow_edges,
        verifier_diagnostics,
    }
}

fn convert_graph_facts(analysis: &Analysis<'_>) -> (Vec<SccFact>, Vec<DataflowEdgeFact>) {
    let mut components = BTreeMap::<usize, Vec<usize>>::new();
    for (start_pc, block) in &analysis.cfg_nodes {
        if *start_pc == analysis.super_root || block.topo_index.scc_id == usize::MAX {
            continue;
        }
        components
            .entry(block.topo_index.scc_id)
            .or_default()
            .push(*start_pc);
    }
    let sccs = components
        .into_iter()
        .map(|(id, mut block_starts)| {
            block_starts.sort_unstable();
            let recursive = block_starts.len() > 1
                || block_starts.iter().any(|start_pc| {
                    analysis
                        .cfg_nodes
                        .get(start_pc)
                        .is_some_and(|block| block.destinations.contains(start_pc))
                });
            SccFact {
                id,
                block_starts,
                recursive,
            }
        })
        .collect();

    let node_pc = |node: &solana_sbpf::static_analysis::DfgNode| -> usize {
        match node {
            solana_sbpf::static_analysis::DfgNode::InstructionNode(pc)
            | solana_sbpf::static_analysis::DfgNode::PhiNode(pc) => *pc,
        }
    };
    let mut dataflow_edges = analysis
        .dfg_forward_edges
        .values()
        .flat_map(|edges| edges.iter())
        .map(|edge| DataflowEdgeFact {
            source_pc: node_pc(&edge.source),
            destination_pc: node_pc(&edge.destination),
            kind: format!("{:?}", edge.kind),
            resource: format!("{:?}", edge.resource),
        })
        .collect::<Vec<_>>();
    dataflow_edges.sort_by_key(|edge| (edge.source_pc, edge.destination_pc));
    dataflow_edges.dedup();
    (sccs, dataflow_edges)
}

fn extract_relocations(bytes: &[u8]) -> Vec<RelocationFact> {
    let Ok(file) = object::File::parse(bytes) else {
        return Vec::new();
    };
    let mut relocations = Vec::new();
    for section in file.sections() {
        let section_name = section.name().unwrap_or_default().to_owned();
        for (offset, relocation) in section.relocations() {
            let target = match relocation.target() {
                object::RelocationTarget::Symbol(index) => file
                    .symbol_by_index(index)
                    .ok()
                    .and_then(|symbol| symbol.name().ok())
                    .map(|name| format!("symbol:{name}"))
                    .unwrap_or_else(|| format!("symbol-index:{}", index.0)),
                object::RelocationTarget::Section(index) => file
                    .section_by_index(index)
                    .ok()
                    .and_then(|section| section.name().ok())
                    .map(|name| format!("section:{name}"))
                    .unwrap_or_else(|| format!("section-index:{}", index.0)),
                object::RelocationTarget::Absolute => "absolute".into(),
                _ => "unknown".into(),
            };
            relocations.push(RelocationFact {
                section: section_name.clone(),
                offset,
                kind: format!("{:?}", relocation.kind()),
                encoding: format!("{:?}", relocation.encoding()),
                size: relocation.size(),
                target,
                addend: relocation.addend(),
            });
            if relocations.len() >= 16_384 {
                return relocations;
            }
        }
    }
    relocations
}

fn collect_calls(instructions: &[InstructionFact], functions: &[FunctionFact]) -> Vec<CallFact> {
    use solana_sbpf::ebpf;

    instructions
        .iter()
        .filter_map(|instruction| {
            let (kind, target_pc, target_symbol_hash, status) = match instruction.opcode {
                ebpf::CALL_REG => ("indirect", None, None, "unresolved"),
                ebpf::CALL_IMM if instruction.assembly.starts_with("syscall ") => (
                    "syscall",
                    None,
                    Some(instruction.immediate as u32),
                    "resolved",
                ),
                ebpf::CALL_IMM => (
                    "direct_or_relocated",
                    None,
                    Some(instruction.immediate as u32),
                    "unresolved",
                ),
                _ => return None,
            };
            let caller_pc = functions
                .iter()
                .filter(|function| function.pc <= instruction.pc)
                .max_by_key(|function| function.pc)
                .map(|function| function.pc);
            Some(CallFact {
                call_pc: instruction.pc,
                caller_pc,
                kind: kind.into(),
                target_pc,
                target_symbol_hash,
                status: status.into(),
            })
        })
        .collect()
}

#[derive(Debug, Clone, Copy)]
struct KnownRegisterValue {
    value: i64,
    source_pc: usize,
}

/// Performs a deliberately conservative, intra-basic-block register propagation pass.
///
/// This is not a symbolic executor: control-flow joins, memory aliases, and runtime
/// input values remain unknown. The useful result is a bounded set of facts such as
/// "r1 at this call is the immediate 7 introduced at pc 42".
fn collect_register_uses(
    instructions: &[InstructionFact],
    basic_blocks: &[BasicBlockFact],
) -> Vec<RegisterUseFact> {
    use solana_sbpf::ebpf;

    let block_starts: HashSet<usize> = basic_blocks.iter().map(|block| block.start_pc).collect();
    let mut registers: [Option<KnownRegisterValue>; 11] = [None; 11];
    let mut uses = Vec::new();
    let first_pc = instructions.first().map(|instruction| instruction.pc);

    for instruction in instructions {
        if block_starts.contains(&instruction.pc) && Some(instruction.pc) != first_pc {
            registers = [None; 11];
        }

        if matches!(instruction.opcode, ebpf::CALL_IMM | ebpf::CALL_REG) {
            for register in 1..=5u8 {
                let known = registers[register as usize];
                uses.push(RegisterUseFact {
                    pc: instruction.pc,
                    register,
                    value: known.map(|value| value.value),
                    source_pc: known.map(|value| value.source_pc),
                    use_kind: "call_argument".into(),
                });
            }
        }

        match instruction.opcode {
            ebpf::LD_DW_IMM | ebpf::MOV32_IMM | ebpf::MOV64_IMM => {
                update_immediate(
                    &mut registers,
                    instruction.dst,
                    instruction.immediate,
                    instruction.pc,
                );
            }
            ebpf::MOV32_REG | ebpf::MOV64_REG => {
                let source = registers.get(instruction.src as usize).copied().flatten();
                update_register(&mut registers, instruction.dst, source);
            }
            ebpf::ADD32_IMM | ebpf::ADD64_IMM => {
                update_arithmetic(
                    &mut registers,
                    instruction.dst,
                    Some(instruction.immediate),
                    false,
                );
            }
            ebpf::SUB32_IMM | ebpf::SUB64_IMM => {
                update_arithmetic(
                    &mut registers,
                    instruction.dst,
                    Some(instruction.immediate),
                    true,
                );
            }
            ebpf::ADD32_REG | ebpf::ADD64_REG => {
                let operand = registers
                    .get(instruction.src as usize)
                    .copied()
                    .flatten()
                    .map(|value| value.value);
                update_arithmetic(&mut registers, instruction.dst, operand, false);
            }
            ebpf::SUB32_REG | ebpf::SUB64_REG => {
                let operand = registers
                    .get(instruction.src as usize)
                    .copied()
                    .flatten()
                    .map(|value| value.value);
                update_arithmetic(&mut registers, instruction.dst, operand, true);
            }
            ebpf::CALL_IMM | ebpf::CALL_REG => registers[0] = None,
            _ if writes_register(instruction.opcode) => {
                registers[instruction.dst as usize] = None;
            }
            _ => {}
        }
    }
    uses
}

fn writes_register(opcode: u8) -> bool {
    matches!(opcode & 0x07, 0x01 | 0x04 | 0x06 | 0x07)
}

fn update_immediate(
    registers: &mut [Option<KnownRegisterValue>; 11],
    destination: u8,
    value: i64,
    source_pc: usize,
) {
    if let Some(slot) = registers.get_mut(destination as usize) {
        *slot = Some(KnownRegisterValue { value, source_pc });
    }
}

fn update_register(
    registers: &mut [Option<KnownRegisterValue>; 11],
    destination: u8,
    value: Option<KnownRegisterValue>,
) {
    if let Some(slot) = registers.get_mut(destination as usize) {
        *slot = value;
    }
}

fn update_arithmetic(
    registers: &mut [Option<KnownRegisterValue>; 11],
    destination: u8,
    operand: Option<i64>,
    subtract: bool,
) {
    let Some(slot) = registers.get_mut(destination as usize) else {
        return;
    };
    *slot = match (*slot, operand) {
        (Some(mut current), Some(operand)) => {
            current.value = if subtract {
                current.value.wrapping_sub(operand)
            } else {
                current.value.wrapping_add(operand)
            };
            Some(current)
        }
        _ => None,
    };
}

fn extract_elf_strings(bytes: &[u8]) -> Vec<StringFact> {
    let Ok(file) = object::File::parse(bytes) else {
        return Vec::new();
    };
    let mut strings = Vec::new();
    for section in file.sections() {
        let name = section.name().unwrap_or_default();
        if !name.contains("rodata") && !name.starts_with(".data.rel.ro") {
            continue;
        }
        let Ok(data) = section.data() else {
            continue;
        };
        let base_offset = section
            .file_range()
            .map(|(offset, _)| offset as usize)
            .unwrap_or_default();
        strings.extend(extract_strings(data, base_offset));
        if strings.len() >= 4_096 {
            break;
        }
    }
    strings.truncate(4_096);
    strings
}

fn extract_elf_public_keys(bytes: &[u8]) -> Vec<PublicKeyCandidate> {
    let Ok(file) = object::File::parse(bytes) else {
        return Vec::new();
    };
    let mut candidates = Vec::new();
    for section in file.sections() {
        let name = section.name().unwrap_or_default().to_ascii_lowercase();
        if !(name.contains("rodata") || name.contains("const")) {
            continue;
        }
        let Ok(data) = section.data() else {
            continue;
        };
        let base_offset = section
            .file_range()
            .map(|(offset, _)| offset as usize)
            .unwrap_or_default();
        append_public_key_candidates(data, base_offset, &mut candidates);
    }
    candidates
}

fn append_public_key_candidates(
    data: &[u8],
    base_offset: usize,
    candidates: &mut Vec<PublicKeyCandidate>,
) {
    if data.len() < 32 {
        return;
    }
    // Scan aligned candidates first. Program IDs emitted by common Rust
    // SDKs are normally represented as aligned 32-byte constants. A small
    // unaligned fallback is retained for C/hand-authored binaries, but the
    // total output is bounded to keep downstream evidence packs finite.
    let mut offsets = (0..=data.len().saturating_sub(32))
        .step_by(8)
        .collect::<Vec<_>>();
    offsets.extend((0..=data.len().saturating_sub(32)).step_by(1));
    for offset in offsets {
        if candidates.len() >= 4_096 {
            return;
        }
        let bytes = &data[offset..offset + 32];
        if bytes.iter().all(|byte| *byte == 0) {
            continue;
        }
        let value = bs58::encode(bytes).into_string();
        let fact = PublicKeyCandidate {
            offset: base_offset.saturating_add(offset),
            value,
        };
        if !candidates
            .iter()
            .any(|item: &PublicKeyCandidate| item == &fact)
        {
            candidates.push(fact);
        }
    }
}

fn extract_strings(bytes: &[u8], base_offset: usize) -> Vec<StringFact> {
    const MIN_STRING_LENGTH: usize = 4;
    let mut strings = Vec::new();
    let mut start = None;
    for (offset, byte) in bytes.iter().copied().enumerate() {
        let printable = byte.is_ascii_graphic() || byte == b' ' || byte == b'\t';
        if printable {
            start.get_or_insert(offset);
        } else if let Some(begin) = start.take() {
            if offset.saturating_sub(begin) >= MIN_STRING_LENGTH {
                strings.push(StringFact {
                    offset: base_offset.saturating_add(begin),
                    value: String::from_utf8_lossy(&bytes[begin..offset]).into_owned(),
                });
            }
        }
    }
    if let Some(begin) = start {
        if bytes.len().saturating_sub(begin) >= MIN_STRING_LENGTH {
            strings.push(StringFact {
                offset: base_offset.saturating_add(begin),
                value: String::from_utf8_lossy(&bytes[begin..]).into_owned(),
            });
        }
    }
    strings
}

fn memory_access(instruction: &InstructionFact) -> Option<MemoryAccessFact> {
    use solana_sbpf::ebpf;

    let (kind, width, base_register, value_register, immediate) = match instruction.opcode {
        ebpf::LD_B_REG | ebpf::LD_1B_REG => {
            ("load", 1, instruction.src, Some(instruction.dst), None)
        }
        ebpf::LD_H_REG | ebpf::LD_2B_REG => {
            ("load", 2, instruction.src, Some(instruction.dst), None)
        }
        ebpf::LD_W_REG | ebpf::LD_4B_REG => {
            ("load", 4, instruction.src, Some(instruction.dst), None)
        }
        ebpf::LD_DW_REG | ebpf::LD_8B_REG => {
            ("load", 8, instruction.src, Some(instruction.dst), None)
        }
        ebpf::ST_B_IMM | ebpf::ST_1B_IMM => (
            "store",
            1,
            instruction.dst,
            None,
            Some(instruction.immediate),
        ),
        ebpf::ST_H_IMM | ebpf::ST_2B_IMM => (
            "store",
            2,
            instruction.dst,
            None,
            Some(instruction.immediate),
        ),
        ebpf::ST_W_IMM | ebpf::ST_4B_IMM => (
            "store",
            4,
            instruction.dst,
            None,
            Some(instruction.immediate),
        ),
        ebpf::ST_DW_IMM | ebpf::ST_8B_IMM => (
            "store",
            8,
            instruction.dst,
            None,
            Some(instruction.immediate),
        ),
        ebpf::ST_B_REG | ebpf::ST_1B_REG => {
            ("store", 1, instruction.dst, Some(instruction.src), None)
        }
        ebpf::ST_H_REG | ebpf::ST_2B_REG => {
            ("store", 2, instruction.dst, Some(instruction.src), None)
        }
        ebpf::ST_W_REG | ebpf::ST_4B_REG => {
            ("store", 4, instruction.dst, Some(instruction.src), None)
        }
        ebpf::ST_DW_REG | ebpf::ST_8B_REG => {
            ("store", 8, instruction.dst, Some(instruction.src), None)
        }
        _ => return None,
    };
    Some(MemoryAccessFact {
        pc: instruction.pc,
        kind: kind.into(),
        width,
        base_register,
        value_register,
        offset: instruction.offset,
        immediate,
        assembly: instruction.assembly.clone(),
    })
}

/// Recover stack accesses made through registers that are simple aliases of
/// r10. This deliberately tracks only MOV64 plus immediate ADD/SUB chains and
/// invalidates aliases on calls or any unsupported register write.
fn collect_stack_accesses(
    instructions: &[InstructionFact],
    functions: &[FunctionFact],
) -> Vec<StackAccessFact> {
    use solana_sbpf::ebpf;

    let function_starts = functions
        .iter()
        .map(|function| function.pc)
        .collect::<HashSet<_>>();
    let mut aliases = [None::<i64>; 11];
    aliases[ebpf::FRAME_PTR_REG] = Some(0);
    let mut accesses = Vec::new();

    for instruction in instructions {
        if function_starts.contains(&instruction.pc) {
            aliases = [None; 11];
            aliases[ebpf::FRAME_PTR_REG] = Some(0);
        }

        if let Some(access) = memory_access(instruction) {
            if let Some(base_offset) = aliases
                .get(access.base_register as usize)
                .copied()
                .flatten()
            {
                let effective_offset = base_offset.saturating_add(i64::from(access.offset));
                if let Ok(effective_offset) = i16::try_from(effective_offset) {
                    accesses.push(StackAccessFact {
                        pc: access.pc,
                        kind: access.kind,
                        width: access.width,
                        offset: effective_offset,
                        value_register: access.value_register,
                        immediate: access.immediate,
                        frame_register: ebpf::FRAME_PTR_REG as u8,
                        assembly: access.assembly,
                    });
                }
            }
        }

        match instruction.opcode {
            ebpf::MOV64_REG => {
                aliases[instruction.dst as usize] =
                    aliases.get(instruction.src as usize).copied().flatten();
            }
            ebpf::ADD64_IMM => {
                if let Some(alias) = aliases[instruction.dst as usize].as_mut() {
                    *alias = alias.saturating_add(instruction.immediate);
                }
            }
            ebpf::SUB64_IMM => {
                if let Some(alias) = aliases[instruction.dst as usize].as_mut() {
                    *alias = alias.saturating_sub(instruction.immediate);
                }
            }
            ebpf::CALL_IMM | ebpf::CALL_REG => {
                for alias in aliases.iter_mut().take(6) {
                    *alias = None;
                }
            }
            _ if writes_register(instruction.opcode) => {
                aliases[instruction.dst as usize] = None;
            }
            _ => {}
        }
        aliases[ebpf::FRAME_PTR_REG] = Some(0);
    }
    accesses
}

#[cfg(test)]
mod static_tests {
    use super::*;

    #[test]
    fn rejects_non_elf_bytes_without_panicking() {
        let error = analyze_elf(&[0u8; 64]).unwrap_err();
        assert_eq!(error.stage, "elf-load");
    }

    #[test]
    fn extracts_load_and_store_access_facts() {
        let load = InstructionFact {
            pc: 1,
            opcode: solana_sbpf::ebpf::LD_DW_REG,
            dst: 2,
            src: 1,
            offset: 24,
            immediate: 0,
            assembly: "ldxdw r2, [r1+0x18]".into(),
        };
        let store = InstructionFact {
            pc: 2,
            opcode: solana_sbpf::ebpf::ST_W_IMM,
            dst: 10,
            src: 0,
            offset: -4,
            immediate: 7,
            assembly: "stw [r10-0x4], 0x7".into(),
        };
        let facts = [
            memory_access(&load).unwrap(),
            memory_access(&store).unwrap(),
        ];
        assert_eq!(facts[0].kind, "load");
        assert_eq!(facts[0].base_register, 1);
        assert_eq!(facts[0].width, 8);
        assert_eq!(facts[1].kind, "store");
        assert_eq!(facts[1].immediate, Some(7));
    }

    #[test]
    fn recovers_stack_accesses_through_frame_pointer_aliases() {
        use solana_sbpf::ebpf;

        let instructions = vec![
            InstructionFact {
                pc: 0,
                opcode: ebpf::MOV64_REG,
                dst: 2,
                src: ebpf::FRAME_PTR_REG as u8,
                offset: 0,
                immediate: 0,
                assembly: "mov64 r2, r10".into(),
            },
            InstructionFact {
                pc: 1,
                opcode: ebpf::ADD64_IMM,
                dst: 2,
                src: 0,
                offset: 0,
                immediate: -62,
                assembly: "add64 r2, -62".into(),
            },
            InstructionFact {
                pc: 2,
                opcode: ebpf::ST_B_REG,
                dst: 2,
                src: 0,
                offset: -2,
                immediate: 0,
                assembly: "stxb [r2-2], r0".into(),
            },
        ];
        let functions = vec![FunctionFact {
            pc: 0,
            symbol_hash: 0,
            name: "entrypoint".into(),
        }];

        let accesses = collect_stack_accesses(&instructions, &functions);

        assert_eq!(accesses.len(), 1);
        assert_eq!(accesses[0].offset, -64);
        assert_eq!(accesses[0].kind, "store");
        assert_eq!(accesses[0].frame_register, ebpf::FRAME_PTR_REG as u8);
    }

    #[test]
    fn extracts_printable_runs_with_source_offsets() {
        let facts = extract_strings(b"\0hello\0no\0world\0", 100);
        assert_eq!(facts.len(), 2);
        assert_eq!(facts[0].offset, 101);
        assert_eq!(facts[0].value, "hello");
        assert_eq!(facts[1].value, "world");
    }

    #[test]
    fn ignores_read_only_sections_shorter_than_a_public_key() {
        let mut candidates = Vec::new();
        append_public_key_candidates(&[7u8; 29], 10, &mut candidates);
        assert!(candidates.is_empty());
    }

    #[test]
    fn propagates_bounded_immediates_into_call_arguments() {
        use solana_sbpf::ebpf;

        let instructions = vec![
            InstructionFact {
                pc: 0,
                opcode: ebpf::MOV64_IMM,
                dst: 1,
                src: 0,
                offset: 0,
                immediate: 7,
                assembly: "mov r1, 0x7".into(),
            },
            InstructionFact {
                pc: 1,
                opcode: ebpf::ADD64_IMM,
                dst: 1,
                src: 0,
                offset: 0,
                immediate: 5,
                assembly: "add64 r1, 0x5".into(),
            },
            InstructionFact {
                pc: 2,
                opcode: ebpf::CALL_IMM,
                dst: 0,
                src: 1,
                offset: 0,
                immediate: 99,
                assembly: "call 99".into(),
            },
        ];
        let uses = collect_register_uses(
            &instructions,
            &[BasicBlockFact {
                start_pc: 0,
                end_instruction_index: 3,
                destinations: vec![],
                label: "entry".into(),
            }],
        );
        let r1 = uses.iter().find(|fact| fact.register == 1).unwrap();
        assert_eq!(r1.pc, 2);
        assert_eq!(r1.value, Some(12));
        assert_eq!(r1.source_pc, Some(0));
        assert_eq!(r1.use_kind, "call_argument");
    }

    #[test]
    fn records_direct_indirect_and_syscall_call_sites_without_guessing_targets() {
        use solana_sbpf::ebpf;

        let calls = collect_calls(
            &[
                InstructionFact {
                    pc: 4,
                    opcode: ebpf::CALL_IMM,
                    dst: 0,
                    src: 0,
                    offset: 0,
                    immediate: 123,
                    assembly: "call ".into(),
                },
                InstructionFact {
                    pc: 8,
                    opcode: ebpf::CALL_REG,
                    dst: 0,
                    src: 9,
                    offset: 0,
                    immediate: 0,
                    assembly: "callx r9".into(),
                },
                InstructionFact {
                    pc: 12,
                    opcode: ebpf::CALL_IMM,
                    dst: 0,
                    src: 1,
                    offset: 0,
                    immediate: 456,
                    assembly: "syscall 456".into(),
                },
            ],
            &[FunctionFact {
                pc: 0,
                symbol_hash: 0,
                name: "entry".into(),
            }],
        );
        assert_eq!(calls.len(), 3);
        assert_eq!(calls[0].kind, "direct_or_relocated");
        assert_eq!(calls[0].target_pc, None);
        assert_eq!(calls[1].kind, "indirect");
        assert_eq!(calls[2].kind, "syscall");
        assert_eq!(calls[2].status, "resolved");
    }

    #[test]
    fn canonical_ir_uses_stable_ids_and_bounds_malformed_blocks() {
        let analysis = StaticAnalysis {
            schema_version: sos_types::ANALYSIS_SCHEMA_VERSION,
            sbpf_version: "V0".into(),
            entrypoint: 0,
            instructions: vec![InstructionFact {
                pc: 0,
                opcode: solana_sbpf::ebpf::MOV64_IMM,
                dst: 1,
                src: 0,
                offset: 0,
                immediate: 7,
                assembly: "mov r1, 0x7".into(),
            }],
            functions: vec![FunctionFact {
                pc: 0,
                symbol_hash: 0,
                name: "entry".into(),
            }],
            basic_blocks: vec![BasicBlockFact {
                start_pc: 0,
                end_instruction_index: usize::MAX,
                destinations: vec![0],
                label: "entry".into(),
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
        let ir = build_canonical_ir(&analysis);
        assert_eq!(ir.schema_version, IR_SCHEMA_VERSION);
        assert_eq!(ir.functions[0].id, "fn_0");
        assert_eq!(ir.blocks[0].id, "bb_0");
        assert_eq!(ir.blocks[0].successors, vec!["bb_0"]);
        assert_eq!(ir.operations[0].id, "op_0");
    }
}
