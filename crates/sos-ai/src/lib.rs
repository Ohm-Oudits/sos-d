//! Provider-independent, evidence-constrained AI gateway contracts.
//!
//! This crate keeps provider-specific transport behind `ModelProvider`. Local
//! subprocess inference and a bounded generic HTTP provider are available; the
//! gateway remains responsible for schema, evidence, and resource validation.

use std::{
    collections::{BTreeMap, BTreeSet},
    fmt, fs,
    io::{Read, Write},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use sos_analysis::{
    build_canonical_ir, BasicBlockFact, CanonicalIr, FunctionFact, InstructionFact, StaticAnalysis,
};
use sos_solana::SemanticAnalysis;
use sos_types::{EvidenceItem, EvidenceStatus, ANALYSIS_SCHEMA_VERSION};

mod anthropic;
mod cloudflare;
mod groq;
mod huggingface;
mod ollama;
mod openai;

pub use anthropic::{AnthropicModelProvider, ANTHROPIC_API_VERSION};
pub use cloudflare::{CloudflareModelProvider, CLOUDFLARE_GPT_OSS_120B};
pub use groq::{GroqModelProvider, GROQ_GPT_OSS_120B};
pub use huggingface::{HuggingFaceModelProvider, HUGGING_FACE_GPT_OSS_120B};
pub use ollama::{trusted_open_model, OllamaModelProvider, OpenModelProfile, TRUSTED_OPEN_MODELS};
pub use openai::OpenAiModelProvider;

pub const AI_SCHEMA_VERSION: u32 = 6;
pub const AI_BUNDLE_SCHEMA_VERSION: u32 = 8;
pub const MIN_STAGE_OUTPUT_TOKENS: usize = 128;
pub const MAX_STAGE_OUTPUT_TOKENS: usize = 32_768;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelTask {
    StructureRecovery,
    SignatureRecovery,
    SemanticLabeling,
    Explanation,
    Refinement,
    PseudocodeHumanization,
}

impl ModelTask {
    pub const ALL: [Self; 5] = [
        Self::SignatureRecovery,
        Self::StructureRecovery,
        Self::SemanticLabeling,
        Self::Explanation,
        Self::Refinement,
    ];
}

pub fn default_stage_output_tokens(task: ModelTask) -> usize {
    match task {
        ModelTask::Explanation => 640,
        ModelTask::StructureRecovery
        | ModelTask::SignatureRecovery
        | ModelTask::SemanticLabeling
        | ModelTask::Refinement
        | ModelTask::PseudocodeHumanization => 1_200,
    }
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
    #[serde(default)]
    pub max_output_tokens: usize,
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
    #[serde(default)]
    pub stage_models: AiStageModels,
    #[serde(default)]
    pub stage_token_budgets: AiStageTokenBudgets,
    pub prompt_version: String,
    pub artifact_sha256: Option<String>,
    pub limits: EvidencePackLimits,
    pub max_attempts: usize,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AiStageTokenBudgets {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub structure_recovery: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub signature_recovery: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub semantic_labeling: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub explanation: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub refinement: Option<usize>,
}

impl AiStageTokenBudgets {
    pub fn tokens_for(&self, task: ModelTask) -> usize {
        match task {
            ModelTask::StructureRecovery => self.structure_recovery,
            ModelTask::SignatureRecovery => self.signature_recovery,
            ModelTask::SemanticLabeling => self.semantic_labeling,
            ModelTask::Explanation => self.explanation,
            ModelTask::Refinement => self.refinement,
            ModelTask::PseudocodeHumanization => None,
        }
        .unwrap_or_else(|| default_stage_output_tokens(task))
    }

    pub fn configured(&self) -> impl Iterator<Item = (ModelTask, usize)> {
        [
            (ModelTask::StructureRecovery, self.structure_recovery),
            (ModelTask::SignatureRecovery, self.signature_recovery),
            (ModelTask::SemanticLabeling, self.semantic_labeling),
            (ModelTask::Explanation, self.explanation),
            (ModelTask::Refinement, self.refinement),
        ]
        .into_iter()
        .filter_map(|(task, tokens)| tokens.map(|tokens| (task, tokens)))
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AiStageModels {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub structure_recovery: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub signature_recovery: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub semantic_labeling: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub explanation: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub refinement: Option<String>,
}

impl AiStageModels {
    pub fn model_for<'a>(&'a self, task: ModelTask, fallback: &'a str) -> &'a str {
        match task {
            ModelTask::StructureRecovery => self.structure_recovery.as_deref(),
            ModelTask::SignatureRecovery => self.signature_recovery.as_deref(),
            ModelTask::SemanticLabeling => self.semantic_labeling.as_deref(),
            ModelTask::Explanation => self.explanation.as_deref(),
            ModelTask::Refinement => self.refinement.as_deref(),
            ModelTask::PseudocodeHumanization => None,
        }
        .unwrap_or(fallback)
    }

    pub fn configured(&self) -> impl Iterator<Item = (ModelTask, &str)> {
        [
            (
                ModelTask::StructureRecovery,
                self.structure_recovery.as_deref(),
            ),
            (
                ModelTask::SignatureRecovery,
                self.signature_recovery.as_deref(),
            ),
            (
                ModelTask::SemanticLabeling,
                self.semantic_labeling.as_deref(),
            ),
            (ModelTask::Explanation, self.explanation.as_deref()),
            (ModelTask::Refinement, self.refinement.as_deref()),
        ]
        .into_iter()
        .filter_map(|(task, model)| model.map(|model| (task, model)))
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StructureRecoveryOutput {
    pub functions: Vec<RecoveredFunction>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RecoveredFunction {
    pub function_id: String,
    pub pseudocode: String,
    /// Required only for textual pseudocode humanization. Each entry binds one
    /// deterministic source-line obligation to a concrete fragment retained in
    /// the readable output.
    #[serde(default)]
    pub source_mappings: Vec<RecoveredSourceMapping>,
    /// One concrete source header for every deterministic branch/loop region
    /// in the scoped CFG. This prevents a candidate from satisfying several
    /// distinct regions with one generic `if` or `loop` token.
    #[serde(default)]
    pub control_flow_mappings: Vec<RecoveredControlFlowMapping>,
    /// One evidence-backed source expression for every deterministic fixed
    /// memory field used by this function. Known SolAccountInfo fields retain
    /// their recovered names; unresolved offsets use stable unknown names.
    #[serde(default)]
    pub memory_field_mappings: Vec<RecoveredMemoryFieldMapping>,
    /// One exact source call for every deterministically resolved internal
    /// SBPF call edge made by this function.
    #[serde(default)]
    pub direct_call_mappings: Vec<RecoveredDirectCallMapping>,
    #[serde(default)]
    pub effects: Vec<RecoveredEffect>,
    #[serde(default)]
    pub evidence: Vec<sos_types::EvidenceRef>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecoveredSourceMapping {
    pub obligation_id: String,
    pub output_fragment: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecoveredControlFlowMapping {
    pub region_id: String,
    pub output_fragment: String,
    #[serde(default)]
    pub evidence: Vec<sos_types::EvidenceRef>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecoveredMemoryFieldMapping {
    pub contract_id: String,
    pub field_path: String,
    pub type_name: String,
    pub offset: i32,
    pub layout_type: String,
    pub layout_field: String,
    pub output_fragment: String,
    #[serde(default)]
    pub evidence: Vec<sos_types::EvidenceRef>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecoveredDirectCallMapping {
    pub contract_id: String,
    pub target_function_id: String,
    pub output_fragment: String,
    #[serde(default)]
    pub evidence: Vec<sos_types::EvidenceRef>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ControlFlowRegionKind {
    Branch,
    Loop,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ControlFlowArmContract {
    pub entry_block_id: String,
    pub member_blocks: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ControlFlowEdgeContract {
    pub source_block_id: String,
    pub destination_block_id: String,
}

/// Deterministic structured-control-flow evidence derived from the bounded CFG.
/// It is intentionally source-language neutral: the model chooses `if` versus
/// `match` and `while` versus `loop`, but it cannot invent or merge regions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ControlFlowRegionContract {
    pub id: String,
    pub function_id: String,
    pub kind: ControlFlowRegionKind,
    pub header_block_id: String,
    pub member_blocks: Vec<String>,
    pub entry_blocks: Vec<String>,
    pub exit_blocks: Vec<String>,
    pub join_block_id: Option<String>,
    pub arms: Vec<ControlFlowArmContract>,
    pub cycle_edges: Vec<ControlFlowEdgeContract>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryFieldContractKind {
    RegisterOffset,
    SolAccountInfo,
}

/// Source-neutral layout fact derived from fixed-width machine memory accesses.
/// Names are deterministic: only fully recovered SolAccountInfo fields receive
/// semantic names; all other offsets remain explicit `unknown_*` paths.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryFieldContract {
    pub id: String,
    pub function_id: String,
    pub kind: MemoryFieldContractKind,
    pub base: String,
    /// Logical byte offset in the recovered source-level record.
    pub offset: i32,
    /// Concrete machine offsets that established this field. For stack-copied
    /// SolAccountInfo records these differ from the logical record offset.
    pub machine_offsets: Vec<i32>,
    pub widths: Vec<u8>,
    pub access_kinds: Vec<String>,
    pub access_pcs: Vec<usize>,
    pub field_path: String,
    pub type_name: String,
    pub layout_type: String,
    pub layout_field: String,
    pub semantic_name_recovered: bool,
    pub evidence: Vec<sos_types::EvidenceRef>,
}

/// A resolved internal SBPF call edge. The historical `DirectCallContract`
/// name covers both immediate calls and indirect `callx` sites whose target
/// register is deterministically resolved. Unlike a generic call effect, this
/// contract preserves the exact caller, call site, and callee identity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DirectCallContract {
    pub id: String,
    pub caller_function_id: String,
    pub target_function_id: String,
    pub call_pc: usize,
    pub operation_id: String,
    pub evidence: Vec<sos_types::EvidenceRef>,
}

pub fn required_control_flow_regions(
    pack: &EvidencePack,
    function_id: &str,
) -> Vec<ControlFlowRegionContract> {
    pack.control_flow_regions
        .iter()
        .filter(|region| region.function_id == function_id)
        .cloned()
        .collect()
}

pub fn required_memory_field_contracts(
    pack: &EvidencePack,
    function_id: &str,
) -> Vec<MemoryFieldContract> {
    pack.memory_field_contracts
        .iter()
        .filter(|contract| contract.function_id == function_id)
        .cloned()
        .collect()
}

pub fn required_direct_call_contracts(
    pack: &EvidencePack,
    function_id: &str,
) -> Vec<DirectCallContract> {
    pack.direct_call_contracts
        .iter()
        .filter(|contract| contract.caller_function_id == function_id)
        .cloned()
        .collect()
}

fn derive_control_flow_regions(ir: &CanonicalIr) -> Vec<ControlFlowRegionContract> {
    let mut regions = Vec::new();
    for function in &ir.functions {
        let mut blocks = ir
            .blocks
            .iter()
            .filter(|block| block.function_id == function.id)
            .collect::<Vec<_>>();
        blocks.sort_by_key(|block| block.start_pc);
        if blocks.is_empty() {
            continue;
        }
        let order = blocks
            .iter()
            .enumerate()
            .map(|(index, block)| (block.id.clone(), index))
            .collect::<BTreeMap<_, _>>();
        let block_ids = order.keys().cloned().collect::<BTreeSet<_>>();
        let entry = blocks
            .iter()
            .find(|block| block.start_pc == function.entry_pc)
            .unwrap_or(&blocks[0])
            .id
            .clone();
        let successors = blocks
            .iter()
            .map(|block| {
                let mut next = block
                    .successors
                    .iter()
                    .filter(|successor| block_ids.contains(*successor))
                    .cloned()
                    .collect::<Vec<_>>();
                sort_block_ids(&mut next, &order);
                next.dedup();
                (block.id.clone(), next)
            })
            .collect::<BTreeMap<_, _>>();
        let mut predecessors = block_ids
            .iter()
            .map(|id| (id.clone(), Vec::new()))
            .collect::<BTreeMap<_, Vec<String>>>();
        for (source, destinations) in &successors {
            for destination in destinations {
                predecessors
                    .entry(destination.clone())
                    .or_default()
                    .push(source.clone());
            }
        }
        for incoming in predecessors.values_mut() {
            sort_block_ids(incoming, &order);
            incoming.dedup();
        }
        let reachable = reachable_blocks(&entry, &successors);
        let dominators = dominator_sets(&entry, &reachable, &predecessors);

        let mut loop_members = BTreeMap::<String, BTreeSet<String>>::new();
        let mut loop_edges = BTreeMap::<String, BTreeSet<(String, String)>>::new();
        for source in &reachable {
            for destination in successors.get(source).into_iter().flatten() {
                if !dominators
                    .get(source)
                    .is_some_and(|set| set.contains(destination))
                {
                    continue;
                }
                let members = loop_members.entry(destination.clone()).or_default();
                members.insert(destination.clone());
                members.insert(source.clone());
                let mut pending = vec![source.clone()];
                while let Some(block) = pending.pop() {
                    for predecessor in predecessors.get(&block).into_iter().flatten() {
                        if reachable.contains(predecessor)
                            && members.insert(predecessor.clone())
                            && predecessor != destination
                        {
                            pending.push(predecessor.clone());
                        }
                    }
                }
                loop_edges
                    .entry(destination.clone())
                    .or_default()
                    .insert((source.clone(), destination.clone()));
            }
        }

        let mut function_regions = Vec::new();
        for (header, members) in &loop_members {
            function_regions.push(loop_region_contract(
                &function.id,
                header,
                members,
                loop_edges.get(header).cloned().unwrap_or_default(),
                &successors,
                &order,
            ));
        }

        for scc in ir.sccs.iter().filter(|scc| scc.recursive) {
            let mut members = scc
                .block_starts
                .iter()
                .map(|pc| format!("bb_{pc:x}"))
                .filter(|block| reachable.contains(block))
                .collect::<BTreeSet<_>>();
            if members.is_empty()
                || function_regions.iter().any(|region| {
                    region.kind == ControlFlowRegionKind::Loop
                        && members
                            .iter()
                            .all(|block| region.member_blocks.contains(block))
                })
            {
                continue;
            }
            let mut entries = members
                .iter()
                .filter(|block| {
                    predecessors
                        .get(*block)
                        .into_iter()
                        .flatten()
                        .any(|predecessor| !members.contains(predecessor))
                })
                .cloned()
                .collect::<Vec<_>>();
            sort_block_ids(&mut entries, &order);
            let header = entries
                .first()
                .cloned()
                .or_else(|| {
                    members
                        .iter()
                        .min_by_key(|block| order.get(*block))
                        .cloned()
                })
                .expect("a non-empty SCC has a header");
            let mut cycle_edges = BTreeSet::new();
            for source in &members {
                for destination in successors.get(source).into_iter().flatten() {
                    if members.contains(destination) {
                        cycle_edges.insert((source.clone(), destination.clone()));
                    }
                }
            }
            members.insert(header.clone());
            function_regions.push(loop_region_contract(
                &function.id,
                &header,
                &members,
                cycle_edges,
                &successors,
                &order,
            ));
        }

        let loop_control_blocks = function_regions
            .iter()
            .filter(|region| region.kind == ControlFlowRegionKind::Loop)
            .flat_map(|region| {
                std::iter::once(region.header_block_id.clone()).chain(
                    region
                        .cycle_edges
                        .iter()
                        .map(|edge| edge.source_block_id.clone()),
                )
            })
            .collect::<BTreeSet<_>>();
        let postdominators = postdominator_sets(&reachable, &successors);
        for block in &blocks {
            if !reachable.contains(&block.id) || loop_control_blocks.contains(&block.id) {
                continue;
            }
            let entries = successors.get(&block.id).cloned().unwrap_or_default();
            if entries.len() < 2 {
                continue;
            }
            let join = immediate_postdominator(&block.id, &postdominators, &order);
            let mut arms = Vec::new();
            let mut members = BTreeSet::from([block.id.clone()]);
            for arm_entry in &entries {
                let arm_members = reachable_before_join(
                    arm_entry,
                    join.as_deref(),
                    &block.id,
                    &reachable,
                    &successors,
                );
                members.extend(arm_members.iter().cloned());
                let mut arm_members = arm_members.into_iter().collect::<Vec<_>>();
                sort_block_ids(&mut arm_members, &order);
                arms.push(ControlFlowArmContract {
                    entry_block_id: arm_entry.clone(),
                    member_blocks: arm_members,
                });
            }
            let mut member_blocks = members.iter().cloned().collect::<Vec<_>>();
            sort_block_ids(&mut member_blocks, &order);
            let mut exits = if let Some(join) = join.clone() {
                vec![join]
            } else {
                members
                    .iter()
                    .flat_map(|source| successors.get(source).into_iter().flatten())
                    .filter(|destination| !members.contains(*destination))
                    .cloned()
                    .collect::<Vec<_>>()
            };
            sort_block_ids(&mut exits, &order);
            exits.dedup();
            function_regions.push(ControlFlowRegionContract {
                id: format!("region_branch_{}", block.id.trim_start_matches("bb_")),
                function_id: function.id.clone(),
                kind: ControlFlowRegionKind::Branch,
                header_block_id: block.id.clone(),
                member_blocks,
                entry_blocks: entries,
                exit_blocks: exits,
                join_block_id: join,
                arms,
                cycle_edges: Vec::new(),
            });
        }
        function_regions.sort_by_key(|region| {
            (
                order
                    .get(&region.header_block_id)
                    .copied()
                    .unwrap_or(usize::MAX),
                region.kind,
            )
        });
        regions.extend(function_regions);
    }
    regions
}

fn sort_block_ids(blocks: &mut [String], order: &BTreeMap<String, usize>) {
    blocks.sort_by_key(|block| order.get(block).copied().unwrap_or(usize::MAX));
}

fn reachable_blocks(entry: &str, successors: &BTreeMap<String, Vec<String>>) -> BTreeSet<String> {
    let mut reachable = BTreeSet::new();
    let mut pending = vec![entry.to_owned()];
    while let Some(block) = pending.pop() {
        if !reachable.insert(block.clone()) {
            continue;
        }
        pending.extend(successors.get(&block).into_iter().flatten().cloned());
    }
    reachable
}

fn dominator_sets(
    entry: &str,
    reachable: &BTreeSet<String>,
    predecessors: &BTreeMap<String, Vec<String>>,
) -> BTreeMap<String, BTreeSet<String>> {
    let mut dominators = reachable
        .iter()
        .map(|block| {
            let initial = if block == entry {
                BTreeSet::from([block.clone()])
            } else {
                reachable.clone()
            };
            (block.clone(), initial)
        })
        .collect::<BTreeMap<_, _>>();
    loop {
        let mut changed = false;
        for block in reachable.iter().filter(|block| block.as_str() != entry) {
            let incoming = predecessors
                .get(block)
                .into_iter()
                .flatten()
                .filter(|predecessor| reachable.contains(*predecessor))
                .collect::<Vec<_>>();
            let mut next = if let Some(first) = incoming.first() {
                dominators.get(*first).cloned().unwrap_or_default()
            } else {
                BTreeSet::new()
            };
            for predecessor in incoming.iter().skip(1) {
                if let Some(set) = dominators.get(*predecessor) {
                    next.retain(|candidate| set.contains(candidate));
                }
            }
            next.insert(block.clone());
            if dominators.get(block) != Some(&next) {
                dominators.insert(block.clone(), next);
                changed = true;
            }
        }
        if !changed {
            return dominators;
        }
    }
}

fn postdominator_sets(
    reachable: &BTreeSet<String>,
    successors: &BTreeMap<String, Vec<String>>,
) -> BTreeMap<String, BTreeSet<String>> {
    let exits = reachable
        .iter()
        .filter(|block| {
            successors
                .get(*block)
                .is_none_or(|next| next.iter().all(|successor| !reachable.contains(successor)))
        })
        .cloned()
        .collect::<BTreeSet<_>>();
    if exits.is_empty() {
        return BTreeMap::new();
    }
    let mut postdominators = reachable
        .iter()
        .map(|block| {
            let initial = if exits.contains(block) {
                BTreeSet::from([block.clone()])
            } else {
                reachable.clone()
            };
            (block.clone(), initial)
        })
        .collect::<BTreeMap<_, _>>();
    loop {
        let mut changed = false;
        for block in reachable.iter().filter(|block| !exits.contains(*block)) {
            let next_blocks = successors
                .get(block)
                .into_iter()
                .flatten()
                .filter(|successor| reachable.contains(*successor))
                .collect::<Vec<_>>();
            let mut next = if let Some(first) = next_blocks.first() {
                postdominators.get(*first).cloned().unwrap_or_default()
            } else {
                BTreeSet::new()
            };
            for successor in next_blocks.iter().skip(1) {
                if let Some(set) = postdominators.get(*successor) {
                    next.retain(|candidate| set.contains(candidate));
                }
            }
            next.insert(block.clone());
            if postdominators.get(block) != Some(&next) {
                postdominators.insert(block.clone(), next);
                changed = true;
            }
        }
        if !changed {
            return postdominators;
        }
    }
}

fn immediate_postdominator(
    block: &str,
    postdominators: &BTreeMap<String, BTreeSet<String>>,
    order: &BTreeMap<String, usize>,
) -> Option<String> {
    let mut strict = postdominators
        .get(block)?
        .iter()
        .filter(|candidate| candidate.as_str() != block)
        .cloned()
        .collect::<Vec<_>>();
    sort_block_ids(&mut strict, order);
    strict.iter().find_map(|candidate| {
        (!strict.iter().any(|other| {
            other != candidate
                && postdominators
                    .get(other)
                    .is_some_and(|set| set.contains(candidate))
        }))
        .then(|| candidate.clone())
    })
}

fn reachable_before_join(
    entry: &str,
    join: Option<&str>,
    header: &str,
    reachable: &BTreeSet<String>,
    successors: &BTreeMap<String, Vec<String>>,
) -> BTreeSet<String> {
    let mut members = BTreeSet::new();
    let mut pending = vec![entry.to_owned()];
    while let Some(block) = pending.pop() {
        if block == header || join == Some(block.as_str()) || !reachable.contains(&block) {
            continue;
        }
        if !members.insert(block.clone()) {
            continue;
        }
        pending.extend(successors.get(&block).into_iter().flatten().cloned());
    }
    members
}

fn loop_region_contract(
    function_id: &str,
    header: &str,
    members: &BTreeSet<String>,
    edges: BTreeSet<(String, String)>,
    successors: &BTreeMap<String, Vec<String>>,
    order: &BTreeMap<String, usize>,
) -> ControlFlowRegionContract {
    let mut member_blocks = members.iter().cloned().collect::<Vec<_>>();
    sort_block_ids(&mut member_blocks, order);
    let mut exits = members
        .iter()
        .flat_map(|source| successors.get(source).into_iter().flatten())
        .filter(|destination| !members.contains(*destination))
        .cloned()
        .collect::<Vec<_>>();
    sort_block_ids(&mut exits, order);
    exits.dedup();
    let mut cycle_edges = edges
        .into_iter()
        .map(
            |(source_block_id, destination_block_id)| ControlFlowEdgeContract {
                source_block_id,
                destination_block_id,
            },
        )
        .collect::<Vec<_>>();
    cycle_edges.sort_by_key(|edge| {
        (
            order
                .get(&edge.source_block_id)
                .copied()
                .unwrap_or(usize::MAX),
            order
                .get(&edge.destination_block_id)
                .copied()
                .unwrap_or(usize::MAX),
        )
    });
    ControlFlowRegionContract {
        id: format!("region_loop_{}", header.trim_start_matches("bb_")),
        function_id: function_id.to_owned(),
        kind: ControlFlowRegionKind::Loop,
        header_block_id: header.to_owned(),
        member_blocks,
        entry_blocks: vec![header.to_owned()],
        exit_blocks: exits,
        join_block_id: None,
        arms: Vec::new(),
        cycle_edges,
    }
}

fn derive_memory_field_contracts(
    analysis: &StaticAnalysis,
    ir: &CanonicalIr,
    semantics: &SemanticAnalysis,
) -> Vec<MemoryFieldContract> {
    #[derive(Default)]
    struct FieldBuilder {
        widths: BTreeSet<u8>,
        access_kinds: BTreeSet<String>,
        access_pcs: BTreeSet<usize>,
    }

    let function_for_pc = |pc: usize| {
        ir.operations
            .iter()
            .find(|operation| operation.pc == pc)
            .and_then(|operation| {
                ir.blocks
                    .iter()
                    .find(|block| block.id == operation.block_id)
            })
            .map(|block| block.function_id.clone())
    };
    let recovered_stack_pcs = semantics
        .stack_account_field_candidates
        .iter()
        .flat_map(|field| field.access_pcs.iter().copied())
        .collect::<BTreeSet<_>>();
    let mut contracts = Vec::new();

    for field in &semantics.stack_account_field_candidates {
        let mut by_function = BTreeMap::<String, Vec<usize>>::new();
        for pc in &field.access_pcs {
            if let Some(function_id) = function_for_pc(*pc) {
                by_function.entry(function_id).or_default().push(*pc);
            }
        }
        for (function_id, mut access_pcs) in by_function {
            access_pcs.sort_unstable();
            access_pcs.dedup();
            let field_path = format!("account_{}.{}", field.account_index, field.field);
            let layout_type =
                layout_type_name(&function_id, &format!("account_{}", field.account_index));
            let evidence = access_pcs
                .iter()
                .map(|pc| sos_types::EvidenceRef {
                    source: "analysis".into(),
                    location: format!("instruction:0x{pc:x}"),
                    detail: None,
                })
                .collect();
            contracts.push(MemoryFieldContract {
                id: format!(
                    "layout_{}_account_{}_{}",
                    sanitize_contract_component(&function_id),
                    field.account_index,
                    sanitize_contract_component(&field.field)
                ),
                function_id,
                kind: MemoryFieldContractKind::SolAccountInfo,
                base: format!("account_{}", field.account_index),
                offset: sol_account_info_field_offset(&field.field).unwrap_or_default(),
                machine_offsets: vec![i32::from(field.frame_offset)],
                widths: vec![field.width],
                access_kinds: analysis
                    .stack_accesses
                    .iter()
                    .filter(|access| access_pcs.contains(&access.pc))
                    .map(|access| access.kind.clone())
                    .collect::<BTreeSet<_>>()
                    .into_iter()
                    .collect(),
                access_pcs,
                field_path,
                type_name: fixed_width_type(&[field.width]),
                layout_type,
                layout_field: field.field.clone(),
                semantic_name_recovered: true,
                evidence,
            });
        }
    }

    let mut generic = BTreeMap::<(String, u8, i16), FieldBuilder>::new();
    for access in &analysis.memory_accesses {
        if recovered_stack_pcs.contains(&access.pc) {
            continue;
        }
        let Some(function_id) = function_for_pc(access.pc) else {
            continue;
        };
        let field = generic
            .entry((function_id, access.base_register, access.offset))
            .or_default();
        field.widths.insert(access.width);
        field.access_kinds.insert(access.kind.clone());
        field.access_pcs.insert(access.pc);
    }
    for ((function_id, base_register, offset), field) in generic {
        let layout_type = layout_type_name(&function_id, &format!("r{base_register}"));
        let widths = field.widths.into_iter().collect::<Vec<_>>();
        let access_pcs = field.access_pcs.into_iter().collect::<Vec<_>>();
        let offset_name = if offset < 0 {
            format!("neg_0x{:x}", offset.unsigned_abs())
        } else {
            format!("pos_0x{:x}", offset as u16)
        };
        let field_path = format!("unknown_memory_r{base_register}.field_{offset_name}");
        contracts.push(MemoryFieldContract {
            id: format!(
                "layout_{}_r{}_{}",
                sanitize_contract_component(&function_id),
                base_register,
                offset_name
            ),
            function_id,
            kind: MemoryFieldContractKind::RegisterOffset,
            base: format!("r{base_register}"),
            offset: i32::from(offset),
            machine_offsets: vec![i32::from(offset)],
            type_name: fixed_width_type(&widths),
            widths,
            access_kinds: field.access_kinds.into_iter().collect(),
            evidence: access_pcs
                .iter()
                .map(|pc| sos_types::EvidenceRef {
                    source: "analysis".into(),
                    location: format!("instruction:0x{pc:x}"),
                    detail: None,
                })
                .collect(),
            access_pcs,
            field_path,
            layout_type,
            layout_field: format!("field_{offset_name}"),
            semantic_name_recovered: false,
        });
    }
    contracts.sort_by(|left, right| {
        (&left.function_id, left.access_pcs.first(), &left.id).cmp(&(
            &right.function_id,
            right.access_pcs.first(),
            &right.id,
        ))
    });
    contracts
}

fn derive_direct_call_contracts(
    analysis: &StaticAnalysis,
    ir: &CanonicalIr,
) -> Vec<DirectCallContract> {
    let function_for_block = ir
        .blocks
        .iter()
        .map(|block| (block.id.as_str(), block.function_id.as_str()))
        .collect::<BTreeMap<_, _>>();
    let resolved_calls = analysis
        .calls
        .iter()
        .filter(|call| {
            matches!(
                call.kind.as_str(),
                "direct_internal_call" | "indirect_internal_call"
            ) && call.status == "resolved"
        })
        .filter_map(|call| call.target_pc.map(|target| ((call.call_pc, target), call)))
        .collect::<BTreeMap<_, _>>();
    let mut contracts = BTreeMap::new();
    for operation in &ir.operations {
        let Some(target_pc) = operation.call_target_pc else {
            continue;
        };
        if !resolved_calls.contains_key(&(operation.pc, target_pc)) {
            continue;
        }
        let Some(caller_function_id) = function_for_block.get(operation.block_id.as_str()) else {
            continue;
        };
        let target_function_id = format!("fn_{target_pc:x}");
        let id = format!(
            "direct_call_{}_0x{:x}_{}",
            sanitize_contract_component(caller_function_id),
            operation.pc,
            sanitize_contract_component(&target_function_id)
        );
        contracts
            .entry(id.clone())
            .or_insert_with(|| DirectCallContract {
                id,
                caller_function_id: (*caller_function_id).to_owned(),
                target_function_id,
                call_pc: operation.pc,
                operation_id: operation.id.clone(),
                evidence: vec![sos_types::EvidenceRef {
                    source: "ir".into(),
                    location: format!("operation:{}", operation.id),
                    detail: None,
                }],
            });
    }
    contracts.into_values().collect()
}

fn fixed_width_type(widths: &[u8]) -> String {
    match widths {
        [1] => "u8".into(),
        [2] => "u16".into(),
        [4] => "u32".into(),
        [8] => "u64".into(),
        _ => "unknown_bytes".into(),
    }
}

fn sol_account_info_field_offset(field: &str) -> Option<i32> {
    match field {
        "key" => Some(0),
        "lamports" => Some(8),
        "data_len" => Some(16),
        "data" => Some(24),
        "owner" => Some(32),
        "rent_epoch" => Some(40),
        "is_signer" => Some(48),
        "is_writable" => Some(49),
        "executable" => Some(50),
        _ => None,
    }
}

fn sanitize_contract_component(value: &str) -> String {
    value
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || character == '_' {
                character
            } else {
                '_'
            }
        })
        .collect()
}

fn layout_type_name(function_id: &str, base: &str) -> String {
    format!(
        "SosLayout{}{}",
        rust_type_component(function_id),
        rust_type_component(base)
    )
}

fn rust_type_component(value: &str) -> String {
    value
        .split(|character: char| !character.is_ascii_alphanumeric())
        .filter(|part| !part.is_empty())
        .map(|part| {
            let mut characters = part.chars();
            let first = characters
                .next()
                .map(|character| character.to_ascii_uppercase())
                .unwrap_or_default();
            format!("{first}{}", characters.as_str())
        })
        .collect()
}

/// Machine-checkable coverage ledger linking a readable reconstruction back
/// to deterministic binary effects. This is intentionally coarser than a full
/// MIR/dataflow proof, but unlike prose citations it lets the gateway reject a
/// candidate that silently drops a branch, write, call, loop, or return.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecoveredEffect {
    pub kind: RecoveredEffectKind,
    #[serde(default)]
    pub input_registers: Vec<u8>,
    #[serde(default)]
    pub target_input_registers: Vec<u8>,
    #[serde(default)]
    pub value_input_registers: Vec<u8>,
    #[serde(default)]
    pub evidence: Vec<sos_types::EvidenceRef>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RecoveredEffectKind {
    MemoryWrite,
    Branch,
    Loop,
    Call,
    Return,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub struct RequiredEffect {
    pub kind: RecoveredEffectKind,
    pub location: String,
    pub input_registers: Vec<u8>,
    pub target_input_registers: Vec<u8>,
    pub value_input_registers: Vec<u8>,
}

pub fn required_structure_effects(pack: &EvidencePack, function_id: &str) -> Vec<RequiredEffect> {
    let function_entry_pc = pack
        .ir
        .functions
        .iter()
        .find(|function| function.id == function_id)
        .map(|function| function.entry_pc);
    let dependency_facts = function_entry_pc
        .map(|pc| sos_analysis::abi_effect_dependencies(&pack.analysis, pc))
        .unwrap_or_default();
    let dependency_map = dependency_facts
        .iter()
        .filter_map(|fact| {
            recovered_effect_kind(&fact.kind).map(|kind| {
                (
                    (fact.pc, kind),
                    (
                        fact.input_registers.clone(),
                        fact.target_input_registers.clone(),
                        fact.value_input_registers.clone(),
                    ),
                )
            })
        })
        .collect::<BTreeMap<_, _>>();
    let function_blocks = pack
        .ir
        .blocks
        .iter()
        .filter(|block| block.function_id == function_id)
        .map(|block| block.id.as_str())
        .collect::<BTreeSet<_>>();
    let mut required = BTreeSet::new();
    for operation in pack
        .ir
        .operations
        .iter()
        .filter(|operation| function_blocks.contains(operation.block_id.as_str()))
    {
        let kind = if operation
            .memory_effect
            .as_deref()
            .is_some_and(|effect| effect.starts_with("store:"))
        {
            Some(RecoveredEffectKind::MemoryWrite)
        } else if matches!(operation.assembly.as_str(), "exit" | "return") {
            Some(RecoveredEffectKind::Return)
        } else if operation.kind.contains("call")
            || operation.kind == "syscall"
            || operation.assembly.starts_with("call ")
            || operation.assembly.starts_with("callx ")
            || operation.assembly.starts_with("syscall ")
        {
            Some(RecoveredEffectKind::Call)
        } else if operation.kind == "control_flow" && operation.assembly.starts_with('j') {
            Some(RecoveredEffectKind::Branch)
        } else {
            None
        };
        if let Some(kind) = kind {
            required.insert(RequiredEffect {
                kind,
                location: format!("operation:{}", operation.id),
                input_registers: dependency_map
                    .get(&(operation.pc, kind))
                    .map(|dependencies| dependencies.0.clone())
                    .unwrap_or_default(),
                target_input_registers: dependency_map
                    .get(&(operation.pc, kind))
                    .map(|dependencies| dependencies.1.clone())
                    .unwrap_or_default(),
                value_input_registers: dependency_map
                    .get(&(operation.pc, kind))
                    .map(|dependencies| dependencies.2.clone())
                    .unwrap_or_default(),
            });
        }
    }
    for scc in pack.ir.sccs.iter().filter(|scc| scc.recursive) {
        let scc_blocks = scc
            .block_starts
            .iter()
            .map(|pc| format!("bb_{pc:x}"))
            .collect::<BTreeSet<_>>();
        let loop_inputs = pack
            .ir
            .operations
            .iter()
            .filter(|operation| scc_blocks.contains(&operation.block_id))
            .filter_map(|operation| {
                dependency_map
                    .get(&(operation.pc, RecoveredEffectKind::Branch))
                    .map(|dependencies| &dependencies.0)
            })
            .flatten()
            .copied()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        for pc in &scc.block_starts {
            let location = format!("block:bb_{pc:x}");
            if function_blocks.contains(location.trim_start_matches("block:")) {
                required.insert(RequiredEffect {
                    kind: RecoveredEffectKind::Loop,
                    location,
                    input_registers: loop_inputs.clone(),
                    target_input_registers: Vec::new(),
                    value_input_registers: Vec::new(),
                });
            }
        }
    }
    required.into_iter().collect()
}

fn recovered_effect_kind(kind: &str) -> Option<RecoveredEffectKind> {
    match kind {
        "memory_write" => Some(RecoveredEffectKind::MemoryWrite),
        "branch" => Some(RecoveredEffectKind::Branch),
        "call" => Some(RecoveredEffectKind::Call),
        "return" => Some(RecoveredEffectKind::Return),
        _ => None,
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SignatureRecoveryOutput {
    pub signatures: Vec<RecoveredSignature>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RecoveredSignature {
    pub function_id: String,
    pub function_name: String,
    pub return_type: String,
    pub parameters: Vec<RecoveredParameter>,
    pub confidence: Option<f32>,
    #[serde(default)]
    pub evidence: Vec<sos_types::EvidenceRef>,
}

/// A signature-stage hypothesis that later source-producing stages must carry
/// forward exactly. The source response hash binds the contract to the
/// accepted, gateway-validated model response that introduced it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SignatureContract {
    pub source_output_sha256: String,
    pub signature: RecoveredSignature,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecoveredParameter {
    pub name: String,
    pub type_name: String,
    pub role: Option<String>,
    /// SBPF ABI argument register when this parameter maps to a directly
    /// observed machine-level input. `None` preserves uncertainty for
    /// source-level parameters that cannot be assigned to an ABI slot.
    #[serde(default)]
    pub abi_register: Option<u8>,
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

pub const REFINEMENT_SCHEMA_VERSION: u32 = 3;

fn default_refinement_schema_version() -> u32 {
    REFINEMENT_SCHEMA_VERSION
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RefinementPatch {
    pub function_id: String,
    pub replacement: String,
    #[serde(default)]
    pub memory_field_mappings: Vec<RecoveredMemoryFieldMapping>,
    #[serde(default)]
    pub direct_call_mappings: Vec<RecoveredDirectCallMapping>,
    #[serde(default)]
    pub evidence: Vec<sos_types::EvidenceRef>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RefinementValidation {
    pub status: String,
    pub notes: Vec<String>,
}

pub const VALIDATION_SCHEMA_VERSION: u32 = 2;
pub const VALIDATION_BUNDLE_SCHEMA_VERSION: u32 = 2;

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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mir_summary: Option<MirProgramSummary>,
    pub limitations: Vec<String>,
}

pub const MIR_SUMMARY_SCHEMA_VERSION: u32 = 3;
pub const MIR_EFFECT_COMPARISON_SCHEMA_VERSION: u32 = 3;

/// Bounded, compiler-produced control/effect summary of a reconstructed Rust
/// candidate. The raw textual MIR is intentionally not persisted: rustc marks
/// that representation unstable, while this projection is versioned by SOS.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MirProgramSummary {
    pub schema_version: u32,
    pub functions: Vec<MirFunctionSummary>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MirFunctionSummary {
    pub name: String,
    pub parameter_count: usize,
    pub basic_block_count: usize,
    pub edges: Vec<MirEdge>,
    pub branch_terminators: usize,
    pub calls: usize,
    pub memory_writes: usize,
    pub returns: usize,
    pub has_loop: bool,
    pub effect_dependencies: Vec<MirEffectDependency>,
    #[serde(default)]
    pub memory_write_dependencies: Vec<MirMemoryWriteDependency>,
    /// Direct call operands recovered from rustc MIR. Local-number operands
    /// remain unresolved; named recovered functions retain their exact ID.
    #[serde(default)]
    pub direct_calls: Vec<MirDirectCallSummary>,
    /// Structured regions recovered from rustc's compiler-produced CFG. These
    /// are compared with the binary-derived contracts without requiring block
    /// numbers or compiler-inserted safety paths to match exactly.
    #[serde(default)]
    pub control_flow_regions: Vec<MirControlFlowRegionSummary>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MirControlFlowArmSummary {
    pub entry_block: usize,
    pub member_blocks: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MirControlFlowRegionSummary {
    pub kind: ControlFlowRegionKind,
    pub header_block: usize,
    pub member_blocks: Vec<usize>,
    pub entry_blocks: Vec<usize>,
    pub exit_blocks: Vec<usize>,
    pub join_block: Option<usize>,
    pub arms: Vec<MirControlFlowArmSummary>,
    pub cycle_edges: Vec<MirEdge>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct MirEffectDependency {
    pub kind: RecoveredEffectKind,
    pub input_parameters: Vec<u8>,
    #[serde(default)]
    pub target_input_parameters: Vec<u8>,
    #[serde(default)]
    pub value_input_parameters: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct MirMemoryWriteDependency {
    pub block: usize,
    pub statement_index: usize,
    pub target_input_parameters: Vec<u8>,
    pub value_input_parameters: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct MirDirectCallSummary {
    pub block: usize,
    pub statement_index: usize,
    pub target: String,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct MirEdge {
    pub source: usize,
    pub destination: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MirEffectComparisonStatus {
    Covered,
    Mismatch,
    NotAvailable,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MirFunctionEffectComparison {
    pub function_id: String,
    pub required: Vec<RecoveredEffectKind>,
    pub observed: Vec<RecoveredEffectKind>,
    pub missing: Vec<RecoveredEffectKind>,
    pub dependency_mismatches: Vec<MirDependencyMismatch>,
    pub memory_write_mismatches: Vec<MirMemoryWriteMismatch>,
    #[serde(default)]
    pub region_mismatches: Vec<MirRegionMismatch>,
    #[serde(default)]
    pub direct_call_mismatches: Vec<MirDirectCallMismatch>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MirRegionMismatch {
    pub required_region_id: String,
    pub kind: ControlFlowRegionKind,
    pub required_entry_count: usize,
    pub required_has_join: bool,
    pub required_has_exit: bool,
    pub observed_candidate_headers: Vec<usize>,
    pub reasons: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MirMemoryWriteMismatch {
    pub required_location: String,
    pub required_target_input_registers: Vec<u8>,
    pub required_value_input_registers: Vec<u8>,
    pub observed_role_signatures: Vec<MirMemoryWriteDependency>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MirDirectCallMismatch {
    pub required_contract_id: String,
    pub required_location: String,
    pub required_target_function_id: String,
    pub observed_targets: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MirDependencyMismatch {
    pub kind: RecoveredEffectKind,
    pub required_input_registers: Vec<u8>,
    pub observed_input_parameters: Vec<u8>,
    pub missing_input_dependencies: Vec<u8>,
    pub required_target_input_registers: Vec<u8>,
    pub observed_target_input_parameters: Vec<u8>,
    pub missing_target_input_dependencies: Vec<u8>,
    pub required_value_input_registers: Vec<u8>,
    pub observed_value_input_parameters: Vec<u8>,
    pub missing_value_input_dependencies: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MirEffectComparison {
    pub schema_version: u32,
    pub status: MirEffectComparisonStatus,
    pub functions: Vec<MirFunctionEffectComparison>,
    pub limitations: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UntrustedEvidence {
    pub source: String,
    pub content: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PseudocodeUnit {
    pub id: String,
    pub start_line: usize,
    pub end_line: usize,
    pub source: String,
    pub function_name: Option<String>,
    pub referenced_functions: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PseudocodeConstruct {
    FunctionDeclaration,
    Branch,
    Loop,
    Call,
    MemoryWrite,
    Return,
    Statement,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PseudocodeObligation {
    pub id: String,
    pub source_line: usize,
    pub source_sha256: String,
    pub operation_ids: Vec<String>,
    pub constructs: Vec<PseudocodeConstruct>,
    /// Exact source-level function identity introduced by this line. Body
    /// reconstruction may improve local names, but it may not silently rename
    /// a known function and break whole-program call consistency.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub declared_function: Option<String>,
    /// Exact known function identities called by this source line. These are
    /// deterministic symbol contracts, not semantic guesses about the callee.
    #[serde(default)]
    pub called_functions: Vec<String>,
}

/// Evidence-preserving syntax normalization used when semantic model inference
/// is unavailable or rejected. This deliberately does not claim recovered
/// types or behavior: it only rewrites recognized decompiler syntax and keeps
/// every unrecognized source line visible.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeterministicHumanization {
    pub schema_version: u32,
    pub pseudocode: String,
    pub transformed_lines: usize,
    pub preserved_lines: usize,
    pub line_mappings: Vec<DeterministicLineMapping>,
    pub limitations: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeterministicLineMapping {
    pub source_line: usize,
    pub kind: String,
    pub source: String,
    pub output: String,
}

pub const MODEL_QUALIFICATION_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelQualificationCorpus {
    pub schema_version: u32,
    pub name: String,
    pub cases: Vec<ModelQualificationCase>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelQualificationCase {
    pub id: String,
    pub category: String,
    pub pseudocode: String,
    pub required_concepts: Vec<Vec<String>>,
    #[serde(default)]
    pub forbidden_fragments: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelQualificationResult {
    pub schema_version: u32,
    pub corpus_name: String,
    pub corpus_sha256: String,
    pub provider: String,
    pub model: String,
    pub prompt_version: String,
    pub minimum_score: f64,
    pub overall_score: f64,
    pub qualified: bool,
    pub cases: Vec<ModelQualificationCaseResult>,
    pub limitations: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelQualificationCaseResult {
    pub id: String,
    pub category: String,
    pub input_sha256: String,
    pub schema_and_evidence_accepted: bool,
    pub unit_count: usize,
    pub matched_concepts: usize,
    pub required_concepts: usize,
    pub missing_concepts: Vec<Vec<String>>,
    pub forbidden_matches: Vec<String>,
    pub score: f64,
    pub output: String,
    pub errors: Vec<String>,
}

pub fn qualify_humanization_model<P: ModelProvider>(
    provider: P,
    provider_name: &str,
    model: &str,
    prompt_version: &str,
    corpus: &ModelQualificationCorpus,
    minimum_score: f64,
) -> Result<ModelQualificationResult, AiError> {
    if corpus.schema_version != MODEL_QUALIFICATION_SCHEMA_VERSION {
        return Err(AiError::InvalidResponse(format!(
            "unsupported model qualification corpus schema {}",
            corpus.schema_version
        )));
    }
    if corpus.name.trim().is_empty()
        || corpus.name.len() > 256
        || corpus.cases.is_empty()
        || corpus.cases.len() > 64
    {
        return Err(AiError::InvalidResponse(
            "model qualification corpus must contain 1..=64 cases".into(),
        ));
    }
    if !minimum_score.is_finite() || !(0.0..=1.0).contains(&minimum_score) {
        return Err(AiError::InvalidResponse(
            "model qualification minimum score must be between 0 and 1".into(),
        ));
    }
    let gateway = ModelGateway::new(provider, 2 * 1024 * 1024, 1024 * 1024);
    let mut results = Vec::with_capacity(corpus.cases.len());
    let mut case_ids = BTreeSet::new();
    for case in &corpus.cases {
        if case.id.trim().is_empty()
            || !case_ids.insert(case.id.clone())
            || case.category.trim().is_empty()
            || case.id.len() > 256
            || case.category.len() > 256
            || case.pseudocode.is_empty()
            || case.pseudocode.len() > 256 * 1024
            || case.required_concepts.len() > 64
            || case.forbidden_fragments.len() > 64
            || case.required_concepts.iter().any(|alternatives| {
                alternatives.is_empty()
                    || alternatives.len() > 16
                    || alternatives
                        .iter()
                        .any(|term| term.is_empty() || term.len() > 256)
            })
            || case
                .forbidden_fragments
                .iter()
                .any(|fragment| fragment.is_empty() || fragment.len() > 256)
        {
            return Err(AiError::InvalidResponse(format!(
                "qualification case {:?} exceeds corpus bounds",
                case.id
            )));
        }
        let units = partition_pseudocode(&case.pseudocode)?;
        if units.len() > 64 {
            return Err(AiError::InvalidResponse(format!(
                "qualification case {:?} requires more than 64 bounded units",
                case.id
            )));
        }
        let mut output = Vec::new();
        let mut output_bytes = 0_usize;
        let mut errors = Vec::new();
        for unit in &units {
            let pack = evidence_pack_from_pseudocode_unit(unit)?;
            match gateway.submit_with_token_budget(
                ModelTask::PseudocodeHumanization,
                model,
                prompt_version,
                pack,
                2_400,
            ) {
                Ok(response) => {
                    let parsed = serde_json::from_value::<StructureRecoveryOutput>(
                        response.structured_output,
                    )
                    .map_err(|error| AiError::InvalidResponse(error.to_string()))?;
                    for function in parsed.functions {
                        output_bytes = output_bytes.saturating_add(function.pseudocode.len());
                        if output_bytes > 1024 * 1024 {
                            return Err(AiError::OutputTooLarge {
                                bytes: output_bytes,
                                limit: 1024 * 1024,
                            });
                        }
                        output.push(function.pseudocode);
                    }
                }
                Err(error) => errors.push(error.to_string()),
            }
        }
        let output = output.join("\n\n");
        let lower = output.to_ascii_lowercase();
        let missing_concepts = case
            .required_concepts
            .iter()
            .filter(|alternatives| {
                !alternatives
                    .iter()
                    .any(|term| lower.contains(&term.to_ascii_lowercase()))
            })
            .cloned()
            .collect::<Vec<_>>();
        let forbidden_matches = case
            .forbidden_fragments
            .iter()
            .filter(|fragment| lower.contains(&fragment.to_ascii_lowercase()))
            .cloned()
            .collect::<Vec<_>>();
        let matched_concepts = case
            .required_concepts
            .len()
            .saturating_sub(missing_concepts.len());
        let concept_score = if case.required_concepts.is_empty() {
            1.0
        } else {
            matched_concepts as f64 / case.required_concepts.len() as f64
        };
        let accepted = errors.is_empty() && !output.trim().is_empty();
        let score = if accepted && forbidden_matches.is_empty() {
            concept_score
        } else {
            0.0
        };
        results.push(ModelQualificationCaseResult {
            id: case.id.clone(),
            category: case.category.clone(),
            input_sha256: sha256_bytes(case.pseudocode.as_bytes()),
            schema_and_evidence_accepted: accepted,
            unit_count: units.len(),
            matched_concepts,
            required_concepts: case.required_concepts.len(),
            missing_concepts,
            forbidden_matches,
            score,
            output,
            errors,
        });
    }
    let overall_score = results.iter().map(|case| case.score).sum::<f64>() / results.len() as f64;
    let qualified = overall_score >= minimum_score
        && results
            .iter()
            .all(|case| case.schema_and_evidence_accepted && case.forbidden_matches.is_empty());
    Ok(ModelQualificationResult {
        schema_version: MODEL_QUALIFICATION_SCHEMA_VERSION,
        corpus_name: corpus.name.clone(),
        corpus_sha256: sha256_json(corpus),
        provider: provider_name.into(),
        model: model.into(),
        prompt_version: prompt_version.into(),
        minimum_score,
        overall_score,
        qualified,
        cases: results,
        limitations: vec![
            "qualification measures this bounded corpus and provider response, not universal model correctness".into(),
            "semantic concept matching is case-insensitive substring matching after schema, evidence, and anti-cheese validation".into(),
            "model outputs remain untrusted hypotheses and still require compiler, replay, and binary gates".into(),
        ],
    })
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvidencePack {
    pub schema_version: u32,
    pub task: ModelTask,
    pub artifact_sha256: Option<String>,
    pub analysis: StaticAnalysis,
    pub ir: CanonicalIr,
    pub semantics: SemanticAnalysis,
    /// Validated output from the earlier signature stage. Unlike
    /// `untrusted_evidence`, these entries are populated by the orchestrator
    /// only after the gateway accepts the signature response, and are enforced
    /// as exact interfaces on later source-producing stages.
    #[serde(default)]
    pub signature_contracts: Vec<SignatureContract>,
    /// Line-level preservation obligations derived deterministically from
    /// textual decompiler output. Empty for ELF/SBPF analysis tasks.
    #[serde(default)]
    pub pseudocode_obligations: Vec<PseudocodeObligation>,
    /// Deterministic branch and loop regions recovered from the bounded CFG.
    /// Structure-stage output must map every region to one concrete source
    /// header; other tasks receive the same immutable contracts for context.
    #[serde(default)]
    pub control_flow_regions: Vec<ControlFlowRegionContract>,
    /// Deterministic fixed-offset memory layout obligations. Structure-stage
    /// output must retain every field path and width instead of replacing raw
    /// offsets with unsupported semantic names.
    #[serde(default)]
    pub memory_field_contracts: Vec<MemoryFieldContract>,
    /// Resolved internal call edges that source-producing stages and rustc MIR
    /// must preserve with the exact target function identity.
    #[serde(default)]
    pub direct_call_contracts: Vec<DirectCallContract>,
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
        let ir = build_canonical_ir(&analysis);
        let control_flow_regions = derive_control_flow_regions(&ir);
        let memory_field_contracts = derive_memory_field_contracts(&analysis, &ir, &semantics);
        let direct_call_contracts = derive_direct_call_contracts(&analysis, &ir);
        Self {
            schema_version: AI_SCHEMA_VERSION,
            task,
            artifact_sha256,
            ir,
            analysis,
            semantics,
            signature_contracts: Vec::new(),
            pseudocode_obligations: Vec::new(),
            control_flow_regions,
            memory_field_contracts,
            direct_call_contracts,
            untrusted_evidence: Vec::new(),
        }
    }

    pub fn with_signature_contracts(mut self, contracts: Vec<SignatureContract>) -> Self {
        self.signature_contracts = contracts;
        self
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
        bounded_semantics.stack_account_field_candidates = semantics
            .stack_account_field_candidates
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
        bounded_semantics.runtime_error_candidates = semantics
            .runtime_error_candidates
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
        bounded_semantics.deserialization_candidates = semantics
            .deserialization_candidates
            .iter()
            .take(limits.max_semantic_items)
            .cloned()
            .collect();
        bounded_semantics.instruction_data_guard_candidates = semantics
            .instruction_data_guard_candidates
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
        pack.control_flow_regions = derive_control_flow_regions(&pack.ir);
        pack.memory_field_contracts =
            derive_memory_field_contracts(&pack.analysis, &pack.ir, &pack.semantics);
        pack.direct_call_contracts = derive_direct_call_contracts(&pack.analysis, &pack.ir);
        pack
    }
}

/// Convert arbitrary textual decompiler output into a deterministic synthetic
/// IR function. Each original line becomes one or more evidence-addressable
/// operations, allowing a hosted model to improve readability without receiving
/// an ELF or inventing citations absent from the submitted pseudocode.
const MAX_PSEUDOCODE_BYTES: usize = 8 * 1024 * 1024;
const MAX_PSEUDOCODE_UNITS: usize = 512;
const MAX_PSEUDOCODE_UNIT_LINES: usize = 128;
const MAX_PSEUDOCODE_LINE_BYTES: usize = 4_096;
const MAX_PSEUDOCODE_IR_OPERATIONS: usize = 128;
const PSEUDOCODE_OPERATION_TEXT_CHARS: usize = 96;

/// Partition decompiler output by detected function boundaries and then by the
/// hosted operation budget. Units preserve global line numbers, so independently
/// recovered regions can be merged and audited without losing source identity.
pub fn partition_pseudocode(source: &str) -> Result<Vec<PseudocodeUnit>, AiError> {
    if source.len() > MAX_PSEUDOCODE_BYTES {
        return Err(AiError::EvidencePackTooLarge {
            bytes: source.len(),
            limit: MAX_PSEUDOCODE_BYTES,
        });
    }
    if source.trim().is_empty() {
        return Err(AiError::InvalidResponse(
            "pseudocode input must not be empty".into(),
        ));
    }
    let lines = source
        .split('\n')
        .map(|line| line.strip_suffix('\r').unwrap_or(line))
        .collect::<Vec<_>>();
    if let Some((index, _)) = lines
        .iter()
        .enumerate()
        .find(|(_, line)| line.len() > MAX_PSEUDOCODE_LINE_BYTES)
    {
        return Err(AiError::InvalidResponse(format!(
            "pseudocode line {} exceeds {MAX_PSEUDOCODE_LINE_BYTES} bytes",
            index + 1
        )));
    }

    let mut units = Vec::new();
    let mut start = 0;
    let mut operation_count = 0_usize;
    for (index, line) in lines.iter().enumerate() {
        let line_operations = line
            .chars()
            .count()
            .div_ceil(PSEUDOCODE_OPERATION_TEXT_CHARS)
            .max(1);
        let function_boundary = index > start && looks_like_function_start(&lines, index);
        let budget_boundary = index > start
            && (index - start >= MAX_PSEUDOCODE_UNIT_LINES
                || operation_count.saturating_add(line_operations) > MAX_PSEUDOCODE_IR_OPERATIONS);
        if function_boundary || budget_boundary {
            push_pseudocode_unit(&mut units, &lines, start, index)?;
            start = index;
            operation_count = 0;
        }
        operation_count = operation_count.saturating_add(line_operations);
    }
    push_pseudocode_unit(&mut units, &lines, start, lines.len())?;
    annotate_pseudocode_units(&mut units, &lines);
    Ok(units)
}

fn looks_like_function_start(lines: &[&str], index: usize) -> bool {
    let trimmed = lines[index].trim();
    if trimmed.is_empty() {
        return false;
    }
    let lower = trimmed.to_ascii_lowercase();
    if [
        "if ", "if(", "for ", "for(", "while ", "while(", "switch ", "switch(", "match ", "loop ",
        "else", "catch ", "catch(",
    ]
    .iter()
    .any(|prefix| lower.starts_with(prefix))
    {
        return false;
    }
    if lower.starts_with("fn ") && trimmed.contains('(') {
        return true;
    }
    let next_is_open_brace = lines.get(index + 1).is_some_and(|next| next.trim() == "{");
    trimmed.contains('(')
        && trimmed.contains(')')
        && !trimmed.ends_with(';')
        && !trimmed.contains('=')
        && (trimmed.ends_with('{') || next_is_open_brace)
}

fn push_pseudocode_unit(
    units: &mut Vec<PseudocodeUnit>,
    lines: &[&str],
    start: usize,
    end: usize,
) -> Result<(), AiError> {
    if start >= end {
        return Ok(());
    }
    if units.len() >= MAX_PSEUDOCODE_UNITS {
        return Err(AiError::EvidencePackTooLarge {
            bytes: units.len().saturating_add(1),
            limit: MAX_PSEUDOCODE_UNITS,
        });
    }
    units.push(PseudocodeUnit {
        id: format!("unit_{:04}", units.len() + 1),
        start_line: start + 1,
        end_line: end,
        source: lines[start..end].join("\n"),
        function_name: None,
        referenced_functions: Vec::new(),
    });
    Ok(())
}

fn annotate_pseudocode_units(units: &mut [PseudocodeUnit], lines: &[&str]) {
    let declarations = lines
        .iter()
        .enumerate()
        .filter(|(index, _)| looks_like_function_start(lines, *index))
        .filter_map(|(index, line)| pseudocode_function_name(line).map(|name| (index + 1, name)))
        .collect::<Vec<_>>();
    let known_names = declarations
        .iter()
        .map(|(_, name)| name.clone())
        .collect::<BTreeSet<_>>();
    for unit in units {
        unit.function_name = declarations
            .iter()
            .rfind(|(line, _)| *line <= unit.start_line)
            .map(|(_, name)| name.clone());
        unit.referenced_functions = known_names
            .iter()
            .filter(|name| unit.function_name.as_ref() != Some(*name))
            .filter(|name| contains_pseudocode_call(&unit.source, name))
            .cloned()
            .collect();
    }
}

fn pseudocode_function_name(line: &str) -> Option<String> {
    let prefix = line.trim().split_once('(')?.0.trim();
    let candidate = prefix
        .strip_prefix("fn ")
        .unwrap_or(prefix)
        .split_whitespace()
        .last()?
        .trim_matches(|character: char| matches!(character, '*' | '&'));
    if candidate.is_empty()
        || !candidate
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || character == '_')
    {
        return None;
    }
    Some(candidate.into())
}

fn contains_pseudocode_call(source: &str, function_name: &str) -> bool {
    pseudocode_named_call_occurrences(source, function_name) > 0
}

fn pseudocode_named_call_occurrences(source: &str, function_name: &str) -> usize {
    source
        .match_indices(function_name)
        .filter(|(index, _)| {
            let before = source[..*index].chars().next_back();
            let after_index = index + function_name.len();
            let after = source[after_index..].chars().next();
            let identifier =
                |character: char| character.is_ascii_alphanumeric() || character == '_';
            !before.is_some_and(identifier)
                && !after.is_some_and(identifier)
                && source[after_index..].trim_start().starts_with('(')
        })
        .count()
}

/// Produce a deterministic pseudo-Rust baseline without invoking a model.
/// Recognized syntax is normalized, while unknown syntax is retained as an
/// explicit comment. Every emitted line carries its original global line
/// number and the audit mapping retains the exact source text.
pub fn deterministic_humanize_pseudocode_unit(
    unit: &PseudocodeUnit,
) -> Result<DeterministicHumanization, AiError> {
    if unit.start_line == 0 || unit.end_line < unit.start_line {
        return Err(AiError::InvalidResponse(
            "pseudocode unit has an invalid line range".into(),
        ));
    }
    let lines = unit
        .source
        .split('\n')
        .map(|line| line.strip_suffix('\r').unwrap_or(line))
        .collect::<Vec<_>>();
    if lines.len() != unit.end_line - unit.start_line + 1 {
        return Err(AiError::InvalidResponse(
            "pseudocode unit line range does not match its source".into(),
        ));
    }

    let mut output = vec![
        "// SOS deterministic syntax normalization; not original source.".to_owned(),
        "// Types and pointer helpers are unverified readability aids; inspect source-line evidence.".to_owned(),
    ];
    let mut line_mappings = Vec::with_capacity(lines.len());
    let mut transformed_lines = 0_usize;
    let mut preserved_lines = 0_usize;
    for (index, source) in lines.iter().enumerate() {
        let source_line = unit.start_line + index;
        let normalized = normalize_pseudocode_line(source);
        if normalized.transformed {
            transformed_lines += 1;
        } else if normalized.kind == "untranslated" {
            preserved_lines += 1;
        }
        if normalized.output.is_empty() {
            output.push(String::new());
        } else {
            output.push(format!(
                "{} // source line {}",
                normalized.output, source_line
            ));
        }
        line_mappings.push(DeterministicLineMapping {
            source_line,
            kind: normalized.kind.into(),
            source: (*source).into(),
            output: normalized.output,
        });
    }

    Ok(DeterministicHumanization {
        schema_version: 1,
        pseudocode: output.join("\n"),
        transformed_lines,
        preserved_lines,
        line_mappings,
        limitations: vec![
            "syntax normalization does not prove recovered types, names, control flow, or behavior"
                .into(),
            "read_*/write_* helpers describe decompiler pointer syntax and are not runtime implementations"
                .into(),
            "untranslated lines remain comments so source evidence is never silently discarded".into(),
        ],
    })
}

struct LineNormalization {
    kind: &'static str,
    output: String,
    transformed: bool,
}

fn normalize_pseudocode_line(source: &str) -> LineNormalization {
    let indentation = source
        .chars()
        .take_while(|character| character.is_whitespace())
        .map(|character| if character == '\t' { "    " } else { " " })
        .collect::<String>();
    let trimmed = source.trim();
    if trimmed.is_empty() {
        return normalized_line("blank", String::new(), false);
    }
    if trimmed == "{" || trimmed == "}" || trimmed == "};" {
        return normalized_line(
            "brace",
            format!("{indentation}{}", trimmed.trim_end_matches(';')),
            true,
        );
    }
    if trimmed.starts_with("//") || trimmed.starts_with("/*") || trimmed.starts_with("* ") {
        return normalized_line(
            "comment",
            format!("{indentation}// {}", trimmed.trim_start_matches('/').trim()),
            true,
        );
    }
    if let Some(signature) = normalize_function_signature(trimmed) {
        return normalized_line(
            "function_signature",
            format!("{indentation}{signature}"),
            true,
        );
    }
    if let Some(declaration) = normalize_local_declaration(trimmed) {
        return normalized_line(
            "local_declaration",
            format!("{indentation}{declaration}"),
            true,
        );
    }
    if let Some(control) = normalize_parenthesized_control(trimmed, "if") {
        return normalized_line("guard", format!("{indentation}{control}"), true);
    }
    if let Some(control) = normalize_parenthesized_control(trimmed, "while") {
        return normalized_line("loop_guard", format!("{indentation}{control}"), true);
    }
    if trimmed == "else" || trimmed == "else {" || trimmed == "} else {" {
        return normalized_line("control", format!("{indentation}{trimmed}"), true);
    }
    if let Some(value) = trimmed.strip_prefix("return") {
        let value = value.trim();
        if value.is_empty() || value == ";" {
            return normalized_line("return", format!("{indentation}return;"), true);
        }
        return normalized_line(
            "return",
            format!(
                "{indentation}return {};",
                normalize_pseudocode_expression(value.trim_end_matches(';').trim())
            ),
            true,
        );
    }
    if trimmed.starts_with("goto ")
        || trimmed.starts_with("switch")
        || (trimmed.ends_with(':') && !trimmed.contains("::"))
    {
        return normalized_line(
            "untranslated",
            format!("{indentation}// untranslated control flow: {trimmed}"),
            false,
        );
    }
    if let Some(statement) = normalize_assignment_or_call(trimmed) {
        return normalized_line("statement", format!("{indentation}{statement}"), true);
    }
    normalized_line(
        "untranslated",
        format!("{indentation}// untranslated: {trimmed}"),
        false,
    )
}

fn normalized_line(kind: &'static str, output: String, transformed: bool) -> LineNormalization {
    LineNormalization {
        kind,
        output,
        transformed,
    }
}

fn normalize_function_signature(source: &str) -> Option<String> {
    let open = source.find('(')?;
    let close = matching_parenthesis(source, open)?;
    let trailing = source[close + 1..].trim();
    if !trailing.is_empty() && trailing != "{" {
        return None;
    }
    let prefix = source[..open].trim();
    if prefix.contains('=') {
        return None;
    }
    let name_start = trailing_identifier_start(prefix)?;
    let name = prefix[name_start..].trim();
    if !is_identifier(name)
        || matches!(
            name.to_ascii_lowercase().as_str(),
            "if" | "for" | "while" | "switch" | "match" | "catch"
        )
    {
        return None;
    }
    let return_type = prefix[..name_start].trim();
    if return_type.is_empty() {
        return None;
    }
    let arguments = split_top_level(&source[open + 1..close], ',')
        .into_iter()
        .filter(|argument| !argument.trim().is_empty() && argument.trim() != "void")
        .map(normalize_function_argument)
        .collect::<Option<Vec<_>>>()?;
    let result = normalize_declared_type(return_type)
        .unwrap_or_else(|| sanitize_type_identifier(return_type));
    let result = if result == "void" {
        String::new()
    } else {
        format!(" -> {result}")
    };
    Some(format!(
        "pub unsafe fn {name}({}){result}{}",
        arguments.join(", "),
        if trailing == "{" { " {" } else { "" }
    ))
}

fn normalize_function_argument(argument: &str) -> Option<String> {
    let argument = argument.trim();
    let name_start = trailing_identifier_start(argument)?;
    let name = argument[name_start..].trim();
    if !is_identifier(name) {
        return None;
    }
    let raw_type = argument[..name_start].trim();
    let pointer_depth = raw_type.matches('*').count();
    let base = normalize_declared_type(raw_type)
        .unwrap_or_else(|| sanitize_type_identifier(raw_type.trim_matches('*').trim()));
    let normalized_type = (0..pointer_depth).fold(base, |inner, _| format!("*mut {inner}"));
    Some(format!("{name}: {normalized_type}"))
}

fn normalize_local_declaration(source: &str) -> Option<String> {
    let body = source.strip_suffix(';')?.trim();
    let assignment = top_level_assignment(body);
    let declaration = assignment.map_or(body, |index| body[..index].trim());
    let name_start = trailing_identifier_start(declaration)?;
    let name = declaration[name_start..].trim();
    if !is_identifier(name) {
        return None;
    }
    let raw_type = declaration[..name_start].trim();
    let pointer_depth = raw_type.matches('*').count();
    let base = normalize_declared_type(raw_type)?;
    let normalized_type = (0..pointer_depth).fold(base, |inner, _| format!("*mut {inner}"));
    let initializer = assignment.map(|index| body[index + 1..].trim());
    Some(match initializer {
        Some(value) => format!(
            "let mut {name}: {normalized_type} = {};",
            normalize_pseudocode_expression(value)
        ),
        None => format!("let mut {name}: {normalized_type};"),
    })
}

fn normalize_parenthesized_control(source: &str, keyword: &str) -> Option<String> {
    let rest = source.strip_prefix(keyword)?.trim_start();
    if !rest.starts_with('(') {
        return None;
    }
    let close = matching_parenthesis(rest, 0)?;
    let trailing = rest[close + 1..].trim();
    if !trailing.is_empty() && trailing != "{" {
        return None;
    }
    Some(format!(
        "{keyword} {}{}",
        normalize_pseudocode_expression(rest[1..close].trim()),
        if trailing == "{" { " {" } else { "" }
    ))
}

fn normalize_assignment_or_call(source: &str) -> Option<String> {
    let body = source.strip_suffix(';')?.trim();
    if let Some(index) = top_level_assignment(body) {
        let left = body[..index].trim();
        let right = body[index + 1..].trim();
        if let Some(dereference) = parse_cast_dereference(left, 0) {
            if dereference.end == left.len() {
                return Some(format!(
                    "write_{}({}, {});",
                    dereference.type_name,
                    normalize_pseudocode_expression(&dereference.address),
                    normalize_pseudocode_expression(right)
                ));
            }
        }
        return Some(format!(
            "{} = {};",
            normalize_pseudocode_expression(left),
            normalize_pseudocode_expression(right)
        ));
    }
    if body.contains('(') && body.ends_with(')') {
        return Some(format!("{};", normalize_pseudocode_expression(body)));
    }
    None
}

fn normalize_pseudocode_expression(source: &str) -> String {
    let mut output = String::with_capacity(source.len());
    let mut cursor = 0_usize;
    while let Some(start) = find_unquoted(source, cursor, "*(") {
        output.push_str(&source[cursor..start]);
        if let Some(dereference) = parse_cast_dereference(source, start) {
            output.push_str(&format!(
                "read_{}({})",
                dereference.type_name,
                normalize_pseudocode_expression(&dereference.address)
            ));
            cursor = dereference.end;
        } else {
            output.push_str("*(");
            cursor = start + 2;
        }
    }
    output.push_str(&source[cursor..]);
    replace_simple_pointer_reads(&output).replace("sol_log_(", "sol_log(")
}

struct ParsedDereference {
    end: usize,
    type_name: String,
    address: String,
}

fn parse_cast_dereference(source: &str, start: usize) -> Option<ParsedDereference> {
    if source.get(start..start + 2)? != "*(" {
        return None;
    }
    let cast_close = matching_parenthesis(source, start + 1)?;
    let cast = source[start + 2..cast_close].trim();
    let raw_type = cast.strip_suffix('*')?.trim();
    let mut address_open = cast_close + 1;
    while source[address_open..]
        .chars()
        .next()
        .is_some_and(char::is_whitespace)
    {
        address_open += source[address_open..].chars().next()?.len_utf8();
    }
    if source.get(address_open..address_open + 1)? != "(" {
        return None;
    }
    let address_close = matching_parenthesis(source, address_open)?;
    let type_name =
        normalize_declared_type(raw_type).unwrap_or_else(|| sanitize_type_identifier(raw_type));
    Some(ParsedDereference {
        end: address_close + 1,
        type_name,
        address: source[address_open + 1..address_close].trim().into(),
    })
}

fn replace_simple_pointer_reads(source: &str) -> String {
    let mut output = String::with_capacity(source.len());
    let mut cursor = 0_usize;
    while let Some(star) = find_unquoted(source, cursor, "*") {
        output.push_str(&source[cursor..star]);
        let next = star + 1;
        let Some(first) = source[next..].chars().next() else {
            output.push('*');
            cursor = next;
            break;
        };
        let previous = source[..star]
            .chars()
            .rev()
            .find(|character| !character.is_whitespace());
        let unary_context = previous.is_none_or(|character| {
            matches!(
                character,
                '(' | '=' | '!' | '<' | '>' | ',' | '+' | '-' | ':' | '['
            )
        });
        if unary_context && (first.is_ascii_alphabetic() || first == '_') {
            let identifier_start = next;
            let scan_start = next + first.len_utf8();
            let mut end = scan_start;
            for (offset, character) in source[scan_start..].char_indices() {
                if character.is_ascii_alphanumeric() || character == '_' {
                    end = scan_start + offset + character.len_utf8();
                } else {
                    break;
                }
            }
            output.push_str(&format!("read_value({})", &source[identifier_start..end]));
            cursor = end;
        } else {
            output.push('*');
            cursor = next;
        }
    }
    output.push_str(&source[cursor..]);
    output
}

fn find_unquoted(source: &str, from: usize, needle: &str) -> Option<usize> {
    let mut quote = None;
    let mut escaped = false;
    for (relative, character) in source.get(from..)?.char_indices() {
        let index = from + relative;
        if let Some(active_quote) = quote {
            if escaped {
                escaped = false;
            } else if character == '\\' {
                escaped = true;
            } else if character == active_quote {
                quote = None;
            }
            continue;
        }
        if character == '"' || character == '\'' {
            quote = Some(character);
        } else if source[index..].starts_with(needle) {
            return Some(index);
        }
    }
    None
}

fn matching_parenthesis(source: &str, open: usize) -> Option<usize> {
    if source.get(open..)?.chars().next()? != '(' {
        return None;
    }
    let mut depth = 0_usize;
    let mut quote = None;
    let mut escaped = false;
    for (relative, character) in source[open..].char_indices() {
        if let Some(active_quote) = quote {
            if escaped {
                escaped = false;
            } else if character == '\\' {
                escaped = true;
            } else if character == active_quote {
                quote = None;
            }
            continue;
        }
        if character == '"' || character == '\'' {
            quote = Some(character);
        } else if character == '(' {
            depth += 1;
        } else if character == ')' {
            depth = depth.checked_sub(1)?;
            if depth == 0 {
                return Some(open + relative);
            }
        }
    }
    None
}

fn split_top_level(source: &str, delimiter: char) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut start = 0_usize;
    let mut depth = 0_usize;
    for (index, character) in source.char_indices() {
        match character {
            '(' | '[' | '{' => depth += 1,
            ')' | ']' | '}' => depth = depth.saturating_sub(1),
            _ if character == delimiter && depth == 0 => {
                parts.push(&source[start..index]);
                start = index + character.len_utf8();
            }
            _ => {}
        }
    }
    parts.push(&source[start..]);
    parts
}

fn top_level_assignment(source: &str) -> Option<usize> {
    let mut depth = 0_usize;
    let bytes = source.as_bytes();
    for (index, character) in source.char_indices() {
        match character {
            '(' | '[' | '{' => depth += 1,
            ')' | ']' | '}' => depth = depth.saturating_sub(1),
            '=' if depth == 0 => {
                let previous = index
                    .checked_sub(1)
                    .and_then(|prior| bytes.get(prior))
                    .copied();
                let next = bytes.get(index + 1).copied();
                if !matches!(previous, Some(b'=' | b'!' | b'<' | b'>')) && next != Some(b'=') {
                    return Some(index);
                }
            }
            _ => {}
        }
    }
    None
}

fn trailing_identifier_start(source: &str) -> Option<usize> {
    let trimmed = source.trim_end();
    let mut start = trimmed.len();
    for (index, character) in trimmed.char_indices().rev() {
        if character.is_ascii_alphanumeric() || character == '_' {
            start = index;
        } else {
            break;
        }
    }
    (start < trimmed.len()).then_some(start)
}

fn is_identifier(value: &str) -> bool {
    let mut characters = value.chars();
    characters
        .next()
        .is_some_and(|character| character.is_ascii_alphabetic() || character == '_')
        && characters.all(|character| character.is_ascii_alphanumeric() || character == '_')
}

fn normalize_declared_type(raw: &str) -> Option<String> {
    let normalized = raw
        .replace('*', " ")
        .split_whitespace()
        .filter(|part| !matches!(part.to_ascii_lowercase().as_str(), "const" | "volatile"))
        .collect::<Vec<_>>()
        .join(" ")
        .to_ascii_lowercase();
    Some(
        match normalized.as_str() {
            "void" => "void",
            "bool" | "boolean" => "bool",
            "byte" | "uchar" | "unsigned char" | "uint8_t" | "undefined1" => "u8",
            "char" | "signed char" | "int8_t" => "i8",
            "ushort" | "unsigned short" | "uint16_t" | "undefined2" => "u16",
            "short" | "signed short" | "int16_t" => "i16",
            "uint" | "unsigned int" | "uint32_t" | "undefined4" => "u32",
            "int" | "signed int" | "int32_t" => "i32",
            "ulong" | "unsigned long" | "unsigned long long" | "uint64_t" | "undefined8" => "u64",
            "long" | "signed long" | "long long" | "signed long long" | "int64_t" => "i64",
            "size_t" | "usize" => "usize",
            "ssize_t" | "isize" => "isize",
            _ => return None,
        }
        .into(),
    )
}

fn sanitize_type_identifier(raw: &str) -> String {
    let mut output = raw
        .trim_matches('*')
        .trim()
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || character == '_' {
                character
            } else {
                '_'
            }
        })
        .collect::<String>();
    while output.contains("__") {
        output = output.replace("__", "_");
    }
    let output = output.trim_matches('_');
    if output.is_empty() {
        "unknown_type".into()
    } else {
        output.into()
    }
}

pub fn evidence_pack_from_pseudocode(source: &str) -> Result<EvidencePack, AiError> {
    let units = partition_pseudocode(source)?;
    if units.len() != 1 {
        return Err(AiError::InvalidResponse(format!(
            "pseudocode requires {} bounded units; use partition_pseudocode and evidence_pack_from_pseudocode_unit",
            units.len()
        )));
    }
    evidence_pack_from_pseudocode_unit(&units[0])
}

pub fn evidence_pack_from_pseudocode_unit(unit: &PseudocodeUnit) -> Result<EvidencePack, AiError> {
    if unit.start_line == 0 || unit.end_line < unit.start_line {
        return Err(AiError::InvalidResponse(
            "pseudocode unit has an invalid line range".into(),
        ));
    }
    let lines = unit
        .source
        .split('\n')
        .map(|line| line.strip_suffix('\r').unwrap_or(line))
        .collect::<Vec<_>>();
    if lines.len() != unit.end_line - unit.start_line + 1 {
        return Err(AiError::InvalidResponse(
            "pseudocode unit line range does not match its source".into(),
        ));
    }

    let mut instructions = Vec::new();
    for (line_index, line) in lines.iter().enumerate() {
        let chars = line.chars().collect::<Vec<_>>();
        let parts = chars
            .chunks(PSEUDOCODE_OPERATION_TEXT_CHARS)
            .collect::<Vec<_>>();
        let part_count = parts.len().max(1);
        for part_index in 0..part_count {
            if instructions.len() >= MAX_PSEUDOCODE_IR_OPERATIONS {
                return Err(AiError::EvidencePackTooLarge {
                    bytes: instructions.len().saturating_add(1),
                    limit: MAX_PSEUDOCODE_IR_OPERATIONS,
                });
            }
            let text = parts
                .get(part_index)
                .map(|part| part.iter().collect::<String>())
                .unwrap_or_default();
            instructions.push(InstructionFact {
                pc: ((unit.start_line + line_index - 1) << 8) | part_index,
                opcode: 0,
                dst: 0,
                src: 0,
                offset: 0,
                immediate: 0,
                assembly: format!(
                    "line_{:04}_part_{:02}: {text}",
                    unit.start_line + line_index,
                    part_index + 1
                ),
            });
        }
    }
    let instruction_count = instructions.len();
    let function_pc = (unit.start_line - 1) << 8;
    let analysis = StaticAnalysis {
        schema_version: ANALYSIS_SCHEMA_VERSION,
        sbpf_version: "textual_pseudocode".into(),
        entrypoint: function_pc,
        instructions,
        functions: vec![FunctionFact {
            pc: function_pc,
            symbol_hash: 0,
            name: unit
                .function_name
                .clone()
                .unwrap_or_else(|| format!("humanize_{}", unit.id)),
        }],
        basic_blocks: vec![BasicBlockFact {
            start_pc: function_pc,
            end_instruction_index: instruction_count,
            destinations: Vec::new(),
            label: "pseudocode_input".into(),
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
    let semantics = sos_solana::analyze(&analysis);
    let context = serde_json::json!({
        "unit_id": unit.id,
        "global_line_range": [unit.start_line, unit.end_line],
        "declared_function": unit.function_name,
        "referenced_known_functions": unit.referenced_functions,
    });
    let mut pack = EvidencePack::from_analysis(
        ModelTask::PseudocodeHumanization,
        Some(sha256_bytes(unit.source.as_bytes())),
        analysis,
        semantics,
    )
    .with_untrusted_evidence(vec![UntrustedEvidence {
        source: "deterministic_pseudocode_index".into(),
        content: context.to_string(),
    }]);
    pack.pseudocode_obligations = build_pseudocode_obligations(unit, &lines, &pack.ir.operations);
    Ok(pack)
}

fn build_pseudocode_obligations(
    unit: &PseudocodeUnit,
    lines: &[&str],
    operations: &[sos_analysis::IrOperation],
) -> Vec<PseudocodeObligation> {
    let labels = lines
        .iter()
        .enumerate()
        .filter_map(|(index, line)| {
            pseudocode_label(line).map(|label| (label, unit.start_line + index))
        })
        .collect::<BTreeMap<_, _>>();
    let mut known_functions = unit.referenced_functions.clone();
    if let Some(function_name) = &unit.function_name {
        known_functions.push(function_name.clone());
    }
    known_functions.sort();
    known_functions.dedup();
    lines
        .iter()
        .enumerate()
        .filter_map(|(index, source)| {
            let source_line = unit.start_line + index;
            let constructs = pseudocode_line_constructs(source, source_line, &labels);
            if constructs.is_empty() {
                return None;
            }
            let declared_function = constructs
                .contains(&PseudocodeConstruct::FunctionDeclaration)
                .then(|| pseudocode_function_name(source))
                .flatten();
            let called_functions = known_functions
                .iter()
                .filter(|function_name| {
                    let declaration_occurrence =
                        usize::from(declared_function.as_deref() == Some(function_name.as_str()));
                    pseudocode_named_call_occurrences(source, function_name)
                        > declaration_occurrence
                })
                .cloned()
                .collect::<Vec<_>>();
            let operation_ids = operations
                .iter()
                .filter(|operation| (operation.pc >> 8) + 1 == source_line)
                .map(|operation| operation.id.clone())
                .collect::<Vec<_>>();
            (!operation_ids.is_empty()).then(|| PseudocodeObligation {
                id: format!("source_line_{source_line:04}"),
                source_line,
                source_sha256: sha256_bytes(source.as_bytes()),
                operation_ids,
                constructs,
                declared_function,
                called_functions,
            })
        })
        .collect()
}

fn pseudocode_label(source: &str) -> Option<String> {
    let label = source.trim().strip_suffix(':')?.trim();
    if label.is_empty()
        || label.contains("::")
        || !label
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || character == '_')
    {
        return None;
    }
    Some(label.to_ascii_lowercase())
}

fn pseudocode_line_constructs(
    source: &str,
    source_line: usize,
    labels: &BTreeMap<String, usize>,
) -> Vec<PseudocodeConstruct> {
    let trimmed = source.trim();
    if trimmed.is_empty()
        || matches!(trimmed, "{" | "}" | "};")
        || trimmed.starts_with("//")
        || trimmed.starts_with("/*")
        || trimmed.starts_with("* ")
        || pseudocode_label(trimmed).is_some()
    {
        return Vec::new();
    }
    let lower = trimmed.to_ascii_lowercase();
    let normalized = normalize_pseudocode_line(trimmed);
    let mut constructs = BTreeSet::new();
    let open_brace = trimmed.find('{');
    let declaration_like = normalized.kind == "function_signature"
        || open_brace.is_some_and(|brace| {
            let prefix = &trimmed[..brace];
            prefix.contains('(')
                && prefix.contains(')')
                && !["if", "for", "while", "switch", "match", "catch"]
                    .iter()
                    .any(|keyword| prefix.trim_start().starts_with(keyword))
        });
    if declaration_like {
        constructs.insert(PseudocodeConstruct::FunctionDeclaration);
    }
    if contains_pseudocode_keyword(&lower, "if")
        || contains_pseudocode_keyword(&lower, "switch")
        || contains_pseudocode_keyword(&lower, "match")
        || contains_pseudocode_keyword(&lower, "goto")
    {
        constructs.insert(PseudocodeConstruct::Branch);
    }
    if contains_pseudocode_keyword(&lower, "while")
        || contains_pseudocode_keyword(&lower, "for")
        || contains_pseudocode_keyword(&lower, "loop")
        || backward_goto_target(&lower, source_line, labels)
    {
        constructs.insert(PseudocodeConstruct::Loop);
    }
    if contains_pseudocode_keyword(&lower, "return") {
        constructs.insert(PseudocodeConstruct::Return);
    }
    if pseudocode_has_call(trimmed, declaration_like) {
        constructs.insert(PseudocodeConstruct::Call);
    }
    if pseudocode_has_memory_write(trimmed) {
        constructs.insert(PseudocodeConstruct::MemoryWrite);
    }
    if constructs.is_empty() {
        constructs.insert(PseudocodeConstruct::Statement);
    }
    constructs.into_iter().collect()
}

fn contains_pseudocode_keyword(source: &str, keyword: &str) -> bool {
    source.match_indices(keyword).any(|(index, _)| {
        let before = source[..index].chars().next_back();
        let after = source[index + keyword.len()..].chars().next();
        !before.is_some_and(|character| character.is_ascii_alphanumeric() || character == '_')
            && !after.is_some_and(|character| character.is_ascii_alphanumeric() || character == '_')
    })
}

fn backward_goto_target(lower: &str, source_line: usize, labels: &BTreeMap<String, usize>) -> bool {
    lower.match_indices("goto").any(|(goto_index, _)| {
        let before = lower[..goto_index].chars().next_back();
        let after_keyword = &lower[goto_index + "goto".len()..];
        let after = after_keyword.chars().next();
        if before.is_some_and(|character| character.is_ascii_alphanumeric() || character == '_')
            || after.is_some_and(|character| character.is_ascii_alphanumeric() || character == '_')
        {
            return false;
        }
        let target = after_keyword
            .trim_start()
            .chars()
            .take_while(|character| character.is_ascii_alphanumeric() || *character == '_')
            .collect::<String>();
        labels
            .get(&target)
            .is_some_and(|target_line| *target_line < source_line)
            || (!target.is_empty() && lower[..goto_index].contains(&format!("{target}:")))
    })
}

fn pseudocode_has_call(source: &str, declaration_like: bool) -> bool {
    let search = if declaration_like {
        source
            .find('{')
            .map(|brace| &source[brace + 1..])
            .unwrap_or_default()
    } else {
        source
    };
    search.match_indices('(').any(|(open, _)| {
        let prefix = search[..open].trim_end();
        let name = prefix
            .chars()
            .rev()
            .take_while(|character| character.is_ascii_alphanumeric() || *character == '_')
            .collect::<String>()
            .chars()
            .rev()
            .collect::<String>()
            .to_ascii_lowercase();
        !name.is_empty()
            && !matches!(
                name.as_str(),
                "fn" | "if" | "for" | "while" | "switch" | "match" | "sizeof" | "return"
            )
    })
}

fn pseudocode_has_memory_write(source: &str) -> bool {
    let bytes = source.as_bytes();
    source.char_indices().any(|(index, character)| {
        if character != '=' {
            return false;
        }
        let previous = index
            .checked_sub(1)
            .and_then(|prior| bytes.get(prior))
            .copied();
        let next = bytes.get(index + 1).copied();
        if matches!(previous, Some(b'=' | b'!' | b'<' | b'>')) || next == Some(b'=') {
            return false;
        }
        let statement_start = source[..index]
            .rfind([';', '{', '}'])
            .map_or(0, |delimiter| delimiter + 1);
        let left = source[statement_start..index].trim();
        left.starts_with('*')
            || left.contains("*(")
            || left.contains('[')
            || left.contains("->")
            || left.contains('.')
    })
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
    let instruction_index_base = analysis
        .instructions
        .iter()
        .position(|instruction| instruction.pc >= function_pc)?;
    let scoped_instruction_count = analysis
        .instructions
        .iter()
        .filter(|instruction| in_range(instruction.pc))
        .count();

    let mut scoped = analysis.clone();
    scoped.entrypoint = function_pc;
    scoped.functions = vec![ordered[index].clone()];
    scoped.instructions.retain(|item| in_range(item.pc));
    scoped.basic_blocks.retain_mut(|item| {
        if !in_range(item.start_pc) {
            return false;
        }
        // Static analysis stores an exclusive index into the whole-program
        // instruction vector. Rebase it when producing a one-function slice;
        // otherwise canonical IR construction can make each block consume the
        // remainder of the function and duplicate operations across blocks.
        item.end_instruction_index = item
            .end_instruction_index
            .saturating_sub(instruction_index_base)
            .min(scoped_instruction_count);
        true
    });
    scoped.memory_accesses.retain(|item| in_range(item.pc));
    scoped.stack_accesses.retain(|item| in_range(item.pc));
    scoped.immediate_constants.retain(|item| in_range(item.pc));
    scoped.register_uses.retain(|item| in_range(item.pc));
    scoped.calls.retain(|item| in_range(item.call_pc));
    // Read-only data is global in the ELF, but exposing every string and public
    // key to a one-function lift gives a model unrelated semantic bait. Keep
    // only values whose section offsets are materialized by this function.
    // Solana's loader maps read-only data at a fixed high base, so the low
    // 32 bits of a pointer immediate are the section offset used by our facts.
    let referenced_ro_offsets = scoped
        .immediate_constants
        .iter()
        .filter_map(|item| (item.value >= 0).then_some(item.value as usize & 0xffff_ffff))
        .collect::<BTreeSet<_>>();
    scoped.strings.retain(|item| {
        referenced_ro_offsets
            .iter()
            .any(|offset| *offset >= item.offset && *offset < item.offset + item.value.len())
    });
    scoped
        .public_key_candidates
        .retain(|item| referenced_ro_offsets.contains(&item.offset));
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
    pub max_output_tokens: usize,
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

impl fmt::Debug for HttpModelProvider {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("HttpModelProvider")
            .field("endpoint", &self.endpoint)
            .field(
                "bearer_token",
                &self.bearer_token.as_ref().map(|_| "<redacted>"),
            )
            .field("max_response_bytes", &self.max_response_bytes)
            .finish_non_exhaustive()
    }
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
        self.submit_with_token_budget(
            task,
            model,
            prompt_version,
            pack,
            default_stage_output_tokens(task),
        )
    }

    pub fn submit_with_token_budget(
        &self,
        task: ModelTask,
        model: impl Into<String>,
        prompt_version: impl Into<String>,
        mut pack: EvidencePack,
        max_output_tokens: usize,
    ) -> Result<ModelResponse, AiError> {
        if !(MIN_STAGE_OUTPUT_TOKENS..=MAX_STAGE_OUTPUT_TOKENS).contains(&max_output_tokens) {
            return Err(AiError::InvalidResponse(format!(
                "model output token budget {max_output_tokens} is outside {MIN_STAGE_OUTPUT_TOKENS}..={MAX_STAGE_OUTPUT_TOKENS}"
            )));
        }
        // Region and memory-layout contracts are deterministic projections.
        // Recompute at the gateway boundary so a stale cache or external caller
        // cannot suppress obligations by editing convenience fields before
        // model inference.
        pack.control_flow_regions = derive_control_flow_regions(&pack.ir);
        pack.memory_field_contracts =
            derive_memory_field_contracts(&pack.analysis, &pack.ir, &pack.semantics);
        pack.direct_call_contracts = derive_direct_call_contracts(&pack.analysis, &pack.ir);
        if task == ModelTask::StructureRecovery {
            let maximum_regions = pack
                .ir
                .functions
                .iter()
                .map(|function| {
                    pack.control_flow_regions
                        .iter()
                        .filter(|region| region.function_id == function.id)
                        .count()
                })
                .max()
                .unwrap_or(0);
            if maximum_regions > 128 {
                return Err(AiError::InvalidResponse(format!(
                    "bounded structure request contains {maximum_regions} control-flow regions for one function; the supported maximum is 128"
                )));
            }
            let maximum_fields = pack
                .ir
                .functions
                .iter()
                .map(|function| {
                    pack.memory_field_contracts
                        .iter()
                        .filter(|contract| contract.function_id == function.id)
                        .count()
                })
                .max()
                .unwrap_or(0);
            if maximum_fields > 1_024 {
                return Err(AiError::InvalidResponse(format!(
                    "bounded structure request contains {maximum_fields} memory fields for one function; the supported maximum is 1024"
                )));
            }
            let maximum_calls = pack
                .ir
                .functions
                .iter()
                .map(|function| {
                    pack.direct_call_contracts
                        .iter()
                        .filter(|contract| contract.caller_function_id == function.id)
                        .count()
                })
                .max()
                .unwrap_or(0);
            if maximum_calls > 1_024 {
                return Err(AiError::InvalidResponse(format!(
                    "bounded structure request contains {maximum_calls} resolved direct calls for one function; the supported maximum is 1024"
                )));
            }
        }
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
            max_output_tokens,
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
        self.submit_with_retry_and_token_budget(
            task,
            model,
            prompt_version,
            pack,
            max_attempts,
            default_stage_output_tokens(task),
        )
    }

    pub fn submit_with_retry_and_token_budget(
        &self,
        task: ModelTask,
        model: impl Into<String>,
        prompt_version: impl Into<String>,
        pack: EvidencePack,
        max_attempts: usize,
        max_output_tokens: usize,
    ) -> Result<ModelResponse, AiError> {
        let attempts = max_attempts.max(1);
        let model = model.into();
        let prompt_version = prompt_version.into();
        let mut last_error = None;
        for _ in 0..attempts {
            match self.submit_with_token_budget(
                task,
                model.clone(),
                prompt_version.clone(),
                pack.clone(),
                max_output_tokens,
            ) {
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

/// Run the five bounded AI stages against one immutable deterministic evidence
/// snapshot. A provider or validation failure is recorded for that stage and
/// does not discard accepted results from earlier stages or break the
/// deterministic analysis bundle.
pub fn run_pipeline<P: ModelProvider>(
    provider: P,
    config: AiPipelineConfig,
    analysis: &StaticAnalysis,
    semantics: &SemanticAnalysis,
) -> AiPipelineBundle {
    run_pipeline_with_context(provider, config, analysis, semantics, &[])
}

/// Run the five bounded stages with caller-supplied, explicitly untrusted
/// context from already accepted dependency functions. Deterministic facts in
/// `analysis` and `semantics` remain authoritative and every response must cite
/// evidence from the current function's bounded pack.
pub fn run_pipeline_with_context<P: ModelProvider>(
    provider: P,
    config: AiPipelineConfig,
    analysis: &StaticAnalysis,
    semantics: &SemanticAnalysis,
    initial_context: &[UntrustedEvidence],
) -> AiPipelineBundle {
    let AiPipelineConfig {
        provider_name,
        model,
        stage_models,
        stage_token_budgets,
        prompt_version,
        artifact_sha256,
        limits,
        max_attempts,
    } = config;
    let gateway = ModelGateway::new(provider, 4 * 1024 * 1024, 1024 * 1024);
    let mut stages = Vec::with_capacity(ModelTask::ALL.len());
    let mut accepted_prior_stages = bounded_initial_context(initial_context);
    let mut signature_contracts = Vec::new();

    for task in ModelTask::ALL {
        let stage_model = stage_models.model_for(task, &model).to_owned();
        let max_output_tokens = stage_token_budgets.tokens_for(task);
        let mut pack = EvidencePack::bounded_from_analysis(
            task,
            artifact_sha256.clone(),
            analysis,
            semantics,
            limits,
        );
        pack.signature_contracts = signature_contracts.clone();
        // Patina/Goonfi-style staged lifting: later stages receive accepted
        // earlier output as hypotheses while deterministic facts remain
        // immutable. The gateway still validates all evidence references.
        pack.untrusted_evidence
            .extend(accepted_prior_stages.iter().cloned());
        let input_sha256 = sha256_json(&pack);
        match gateway.submit_with_retry_and_token_budget(
            task,
            stage_model.clone(),
            prompt_version.clone(),
            pack,
            max_attempts,
            max_output_tokens,
        ) {
            Ok(response) => {
                let output_sha256 = sha256_json(&response);
                if task == ModelTask::SignatureRecovery {
                    let parsed: SignatureRecoveryOutput =
                        serde_json::from_value(response.structured_output.clone())
                            .expect("gateway-validated signature output must deserialize");
                    signature_contracts = parsed
                        .signatures
                        .into_iter()
                        .map(|signature| SignatureContract {
                            source_output_sha256: output_sha256.clone(),
                            signature,
                        })
                        .collect();
                }
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
                    model: stage_model.clone(),
                    max_output_tokens,
                    prompt_version: prompt_version.clone(),
                    input_sha256,
                    output_sha256: Some(output_sha256),
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
                    model: stage_model,
                    max_output_tokens,
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

fn bounded_initial_context(initial_context: &[UntrustedEvidence]) -> Vec<UntrustedEvidence> {
    let mut bounded = initial_context
        .iter()
        .rev()
        .take(4)
        .cloned()
        .collect::<Vec<_>>();
    bounded.reverse();
    for item in &mut bounded {
        item.content = item.content.chars().take(16 * 1024).collect();
    }
    bounded
}

/// Rebuild and verify a persisted five-stage function bundle before reuse.
/// Only fully accepted base pipelines are checkpointable; transient failures
/// and later compiler-repair rounds must be rerun instead of becoming trusted
/// cache state.
pub fn validate_pipeline_bundle_with_context(
    config: &AiPipelineConfig,
    analysis: &StaticAnalysis,
    semantics: &SemanticAnalysis,
    initial_context: &[UntrustedEvidence],
    bundle: &AiPipelineBundle,
) -> Result<(), AiError> {
    if bundle.schema_version != AI_BUNDLE_SCHEMA_VERSION
        || bundle.artifact_sha256 != config.artifact_sha256
        || bundle.provider != config.provider_name
        || bundle.model != config.model
        || bundle.stages.len() != ModelTask::ALL.len()
    {
        return Err(AiError::InvalidResponse(
            "checkpointed pipeline bundle identity or stage count does not match the current request"
                .into(),
        ));
    }

    let mut accepted_prior_stages = bounded_initial_context(initial_context);
    let mut signature_contracts = Vec::new();
    for (stage, expected_task) in bundle.stages.iter().zip(ModelTask::ALL) {
        let expected_model = config.stage_models.model_for(expected_task, &config.model);
        let expected_tokens = config.stage_token_budgets.tokens_for(expected_task);
        if stage.task != expected_task
            || stage.status != AiStageStatus::Accepted
            || stage.model != expected_model
            || stage.max_output_tokens != expected_tokens
            || stage.prompt_version != config.prompt_version
            || stage.error.is_some()
        {
            return Err(AiError::InvalidResponse(format!(
                "checkpointed {expected_task:?} stage is not a matching accepted result"
            )));
        }
        let mut pack = EvidencePack::bounded_from_analysis(
            expected_task,
            config.artifact_sha256.clone(),
            analysis,
            semantics,
            config.limits,
        );
        pack.signature_contracts = signature_contracts.clone();
        pack.untrusted_evidence
            .extend(accepted_prior_stages.iter().cloned());
        if stage.input_sha256 != sha256_json(&pack) {
            return Err(AiError::InvalidResponse(format!(
                "checkpointed {expected_task:?} input hash does not match current evidence"
            )));
        }
        let response = stage.response.as_ref().ok_or_else(|| {
            AiError::InvalidResponse(format!(
                "checkpointed {expected_task:?} stage has no response"
            ))
        })?;
        validate_model_response_for_pack(expected_model, &config.prompt_version, pack, response)?;
        let output_sha256 = sha256_json(response);
        if stage.output_sha256.as_deref() != Some(output_sha256.as_str()) {
            return Err(AiError::InvalidResponse(format!(
                "checkpointed {expected_task:?} output hash does not match its response"
            )));
        }
        if expected_task == ModelTask::SignatureRecovery {
            let parsed: SignatureRecoveryOutput =
                serde_json::from_value(response.structured_output.clone()).map_err(|error| {
                    AiError::InvalidResponse(format!(
                        "checkpointed signature output cannot populate contracts: {error}"
                    ))
                })?;
            signature_contracts = parsed
                .signatures
                .into_iter()
                .map(|signature| SignatureContract {
                    source_output_sha256: output_sha256.clone(),
                    signature,
                })
                .collect();
        }
        let prior = serde_json::to_string(&response.structured_output)
            .unwrap_or_else(|_| "{}".into())
            .chars()
            .take(128 * 1024)
            .collect::<String>();
        accepted_prior_stages.push(UntrustedEvidence {
            source: format!("accepted_model_stage:{expected_task:?}"),
            content: prior,
        });
    }
    Ok(())
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
        stage_models,
        stage_token_budgets,
        prompt_version,
        artifact_sha256,
        limits,
        ..
    } = config;
    let reason = reason.into();
    let stages = ModelTask::ALL
        .into_iter()
        .map(|task| {
            let stage_model = stage_models.model_for(task, &model).to_owned();
            let max_output_tokens = stage_token_budgets.tokens_for(task);
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
                model: stage_model,
                max_output_tokens,
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

/// Parse stable facts from rustc's human-readable MIR output. SOS intentionally
/// recognizes only function headers, basic-block labels, CFG targets, returns,
/// calls, and projected-place writes. Unknown lines remain ignored rather than
/// being guessed into semantic facts.
pub fn summarize_rustc_mir(source: &str) -> MirProgramSummary {
    #[derive(Default)]
    struct Builder {
        name: String,
        parameter_count: usize,
        blocks: BTreeSet<usize>,
        edges: BTreeSet<MirEdge>,
        branch_terminators: usize,
        calls: usize,
        memory_writes: usize,
        returns: usize,
        current_block: Option<usize>,
        block_lines: BTreeMap<usize, Vec<String>>,
        direct_calls: BTreeSet<MirDirectCallSummary>,
    }

    fn finish(builder: Builder) -> MirFunctionSummary {
        let edges = builder.edges.into_iter().collect::<Vec<_>>();
        let has_loop = edges.iter().any(|edge| {
            edge.source == edge.destination
                || mir_path_exists(&edges, edge.destination, edge.source)
        });
        let (effect_dependencies, memory_write_dependencies) = summarize_mir_effect_dependencies(
            &builder.block_lines,
            &edges,
            builder.parameter_count,
            has_loop,
        );
        let control_flow_regions = derive_mir_control_flow_regions(&builder.blocks, &edges);
        MirFunctionSummary {
            name: builder.name,
            parameter_count: builder.parameter_count,
            basic_block_count: builder.blocks.len(),
            edges,
            branch_terminators: builder.branch_terminators,
            calls: builder.calls,
            memory_writes: builder.memory_writes,
            returns: builder.returns,
            has_loop,
            effect_dependencies,
            memory_write_dependencies,
            direct_calls: builder.direct_calls.into_iter().collect(),
            control_flow_regions,
        }
    }

    let mut functions = Vec::new();
    let mut current: Option<Builder> = None;
    for line in source.lines() {
        if !line.chars().next().is_some_and(char::is_whitespace) && line.starts_with("fn ") {
            if let Some(builder) = current.take() {
                functions.push(finish(builder));
            }
            let signature = line.trim_start_matches("fn ");
            let Some((name, remainder)) = signature.split_once('(') else {
                continue;
            };
            let parameters = remainder
                .split_once(") ->")
                .map(|(parameters, _)| parameters)
                .unwrap_or_default();
            current = Some(Builder {
                name: name.trim().to_owned(),
                parameter_count: count_mir_parameters(parameters),
                ..Builder::default()
            });
            continue;
        }
        let Some(builder) = current.as_mut() else {
            continue;
        };
        if line == "}" {
            functions.push(finish(current.take().expect("MIR builder exists")));
            continue;
        }
        let trimmed = line.trim();
        if let Some(block) = parse_mir_block_label(trimmed) {
            builder.blocks.insert(block);
            builder.current_block = Some(block);
            continue;
        }
        let Some(source_block) = builder.current_block else {
            continue;
        };
        builder
            .block_lines
            .entry(source_block)
            .or_default()
            .push(trimmed.to_owned());
        let targets = mir_block_targets(trimmed);
        for destination in &targets {
            builder.edges.insert(MirEdge {
                source: source_block,
                destination: *destination,
            });
        }
        let is_call = trimmed.contains(" -> [return: bb")
            && trimmed
                .split_once(" = ")
                .is_some_and(|(_, right)| right.contains('('));
        if is_call {
            builder.calls = builder.calls.saturating_add(1);
            if let Some(target) = mir_call_target(trimmed) {
                let statement_index = builder
                    .block_lines
                    .get(&source_block)
                    .map_or(0, |lines| lines.len().saturating_sub(1));
                builder.direct_calls.insert(MirDirectCallSummary {
                    block: source_block,
                    statement_index,
                    target,
                });
            }
        } else if !targets.is_empty()
            && (trimmed.starts_with("goto ->")
                || trimmed.starts_with("switchInt(")
                || trimmed.starts_with("assert(")
                || trimmed.starts_with("drop(")
                || trimmed.starts_with("falseEdge ->")
                || trimmed.starts_with("falseUnwind ->"))
        {
            builder.branch_terminators = builder.branch_terminators.saturating_add(1);
        }
        if trimmed == "return;" {
            builder.returns = builder.returns.saturating_add(1);
        }
        if let Some((left, _)) = trimmed.split_once(" = ") {
            let left = left.trim();
            if left.contains("(*") || left.contains('[') {
                builder.memory_writes = builder.memory_writes.saturating_add(1);
            }
        }
    }
    if let Some(builder) = current {
        functions.push(finish(builder));
    }
    functions.sort_by(|left, right| left.name.cmp(&right.name));
    MirProgramSummary {
        schema_version: MIR_SUMMARY_SCHEMA_VERSION,
        functions,
    }
}

fn parse_mir_block_label(line: &str) -> Option<usize> {
    let digits = line.strip_prefix("bb")?.strip_suffix(": {")?;
    digits.parse().ok()
}

fn mir_call_target(line: &str) -> Option<String> {
    if !line.contains(" -> [return: bb") {
        return None;
    }
    let (_, right) = line.split_once(" = ")?;
    let (target, _) = right.split_once('(')?;
    let target = target.trim();
    (!target.is_empty()).then(|| target.to_owned())
}

fn mir_block_targets(line: &str) -> Vec<usize> {
    let bytes = line.as_bytes();
    let mut targets = BTreeSet::new();
    let mut index = 0;
    while index + 2 <= bytes.len() {
        if bytes[index] == b'b' && bytes[index + 1] == b'b' {
            let begin = index + 2;
            let mut end = begin;
            while end < bytes.len() && bytes[end].is_ascii_digit() {
                end += 1;
            }
            if end > begin {
                if let Ok(value) = line[begin..end].parse::<usize>() {
                    targets.insert(value);
                }
                index = end;
                continue;
            }
        }
        index += 1;
    }
    targets.into_iter().collect()
}

fn count_mir_parameters(parameters: &str) -> usize {
    let mut depth = 0_i32;
    let mut count = usize::from(!parameters.trim().is_empty());
    for character in parameters.chars() {
        match character {
            '(' | '[' | '<' | '{' => depth += 1,
            ')' | ']' | '>' | '}' => depth = depth.saturating_sub(1),
            ',' if depth == 0 => count = count.saturating_add(1),
            _ => {}
        }
    }
    count
}

fn summarize_mir_effect_dependencies(
    blocks: &BTreeMap<usize, Vec<String>>,
    edges: &[MirEdge],
    parameter_count: usize,
    has_loop: bool,
) -> (Vec<MirEffectDependency>, Vec<MirMemoryWriteDependency>) {
    #[derive(Clone, Copy, Default)]
    struct DependencyMasks {
        all: u8,
        target: u8,
        value: u8,
    }

    let Some(entry_block) = blocks.keys().next().copied() else {
        return (Vec::new(), Vec::new());
    };
    let mut entry = BTreeMap::<usize, u8>::new();
    for parameter in 1..=parameter_count.min(5) {
        entry.insert(parameter, 1 << parameter);
    }
    let mut incoming = BTreeMap::<usize, BTreeMap<usize, u8>>::new();
    incoming.insert(entry_block, entry);
    let mut queue = std::collections::VecDeque::from([entry_block]);
    let mut effects = BTreeMap::<RecoveredEffectKind, DependencyMasks>::new();
    let mut memory_writes = BTreeMap::<(usize, usize), (u8, u8)>::new();
    while let Some(block) = queue.pop_front() {
        let mut origins = incoming.get(&block).cloned().unwrap_or_default();
        for (statement_index, line) in blocks.get(&block).into_iter().flatten().enumerate() {
            let is_call = line.contains(" -> [return: bb")
                && line
                    .split_once(" = ")
                    .is_some_and(|(_, right)| right.contains('('));
            if let Some((left, right)) = line.split_once(" = ") {
                let right_dependencies = mir_local_dependencies(right, &origins);
                if mir_is_projected_place(left) {
                    let target_dependencies = mir_local_dependencies(left, &origins);
                    let effect = effects.entry(RecoveredEffectKind::MemoryWrite).or_default();
                    effect.all |= right_dependencies | target_dependencies;
                    effect.target |= target_dependencies;
                    effect.value |= right_dependencies;
                    let write = memory_writes.entry((block, statement_index)).or_default();
                    write.0 |= target_dependencies;
                    write.1 |= right_dependencies;
                } else if let Some(local) = parse_plain_mir_local(left) {
                    origins.insert(local, right_dependencies);
                }
            }
            if is_call {
                effects.entry(RecoveredEffectKind::Call).or_default().all |=
                    mir_local_dependencies(line, &origins);
            } else if line.starts_with("switchInt(") || line.starts_with("assert(") {
                effects.entry(RecoveredEffectKind::Branch).or_default().all |=
                    mir_local_dependencies(line, &origins);
            } else if line.starts_with("goto ->") {
                effects.entry(RecoveredEffectKind::Branch).or_default();
            }
            if line == "return;" {
                effects.entry(RecoveredEffectKind::Return).or_default().all |=
                    origins.get(&0).copied().unwrap_or_default();
            }
        }
        for successor in edges
            .iter()
            .filter(|edge| edge.source == block)
            .map(|edge| edge.destination)
        {
            let state = incoming.entry(successor).or_default();
            let mut changed = false;
            for (local, propagated) in &origins {
                let slot = state.entry(*local).or_default();
                let merged = *slot | *propagated;
                changed |= merged != *slot;
                *slot = merged;
            }
            if changed {
                queue.push_back(successor);
            }
        }
    }
    if has_loop {
        let branch_dependencies = effects
            .get(&RecoveredEffectKind::Branch)
            .map(|dependencies| dependencies.all)
            .unwrap_or_default();
        effects.insert(
            RecoveredEffectKind::Loop,
            DependencyMasks {
                all: branch_dependencies,
                ..DependencyMasks::default()
            },
        );
    }
    let effect_dependencies = effects
        .into_iter()
        .map(|(kind, dependencies)| MirEffectDependency {
            kind,
            input_parameters: dependency_registers_from_mask(dependencies.all),
            target_input_parameters: dependency_registers_from_mask(dependencies.target),
            value_input_parameters: dependency_registers_from_mask(dependencies.value),
        })
        .collect();
    let memory_write_dependencies = memory_writes
        .into_iter()
        .map(
            |((block, statement_index), (target_dependencies, value_dependencies))| {
                MirMemoryWriteDependency {
                    block,
                    statement_index,
                    target_input_parameters: dependency_registers_from_mask(target_dependencies),
                    value_input_parameters: dependency_registers_from_mask(value_dependencies),
                }
            },
        )
        .collect();
    (effect_dependencies, memory_write_dependencies)
}

fn mir_local_dependencies(line: &str, origins: &BTreeMap<usize, u8>) -> u8 {
    mir_locals(line).into_iter().fold(0, |dependencies, local| {
        dependencies | origins.get(&local).copied().unwrap_or_default()
    })
}

fn mir_locals(line: &str) -> Vec<usize> {
    let bytes = line.as_bytes();
    let mut locals = BTreeSet::new();
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'_' {
            let begin = index + 1;
            let mut end = begin;
            while end < bytes.len() && bytes[end].is_ascii_digit() {
                end += 1;
            }
            if end > begin {
                if let Ok(local) = line[begin..end].parse::<usize>() {
                    locals.insert(local);
                }
                index = end;
                continue;
            }
        }
        index += 1;
    }
    locals.into_iter().collect()
}

fn parse_plain_mir_local(place: &str) -> Option<usize> {
    let place = place.trim();
    let digits = place.strip_prefix('_')?;
    (!digits.is_empty() && digits.bytes().all(|byte| byte.is_ascii_digit()))
        .then(|| digits.parse().ok())
        .flatten()
}

fn mir_is_projected_place(place: &str) -> bool {
    let place = place.trim();
    place.contains("(*") || place.contains('[')
}

fn dependency_registers_from_mask(mask: u8) -> Vec<u8> {
    (1..=5)
        .filter(|parameter| mask & (1 << parameter) != 0)
        .collect()
}

fn mir_path_exists(edges: &[MirEdge], start: usize, goal: usize) -> bool {
    let mut pending = vec![start];
    let mut visited = BTreeSet::new();
    while let Some(node) = pending.pop() {
        if node == goal {
            return true;
        }
        if !visited.insert(node) {
            continue;
        }
        pending.extend(
            edges
                .iter()
                .filter(|edge| edge.source == node)
                .map(|edge| edge.destination),
        );
    }
    false
}

fn derive_mir_control_flow_regions(
    blocks: &BTreeSet<usize>,
    edges: &[MirEdge],
) -> Vec<MirControlFlowRegionSummary> {
    let Some(entry) = blocks.first().copied() else {
        return Vec::new();
    };
    let successors = blocks
        .iter()
        .map(|block| {
            let next = edges
                .iter()
                .filter(|edge| edge.source == *block && blocks.contains(&edge.destination))
                .map(|edge| edge.destination)
                .collect::<BTreeSet<_>>();
            (*block, next)
        })
        .collect::<BTreeMap<_, _>>();
    let mut predecessors = blocks
        .iter()
        .map(|block| (*block, BTreeSet::new()))
        .collect::<BTreeMap<_, _>>();
    for (source, destinations) in &successors {
        for destination in destinations {
            predecessors
                .entry(*destination)
                .or_default()
                .insert(*source);
        }
    }
    let reachable = mir_reachable_blocks(entry, &successors);
    let dominators = mir_dominator_sets(entry, &reachable, &predecessors);

    let mut loop_members = BTreeMap::<usize, BTreeSet<usize>>::new();
    let mut loop_edges = BTreeMap::<usize, BTreeSet<MirEdge>>::new();
    for source in &reachable {
        for destination in successors.get(source).into_iter().flatten() {
            if !dominators
                .get(source)
                .is_some_and(|set| set.contains(destination))
            {
                continue;
            }
            let members = loop_members.entry(*destination).or_default();
            members.insert(*destination);
            members.insert(*source);
            let mut pending = vec![*source];
            while let Some(block) = pending.pop() {
                for predecessor in predecessors.get(&block).into_iter().flatten() {
                    if reachable.contains(predecessor)
                        && members.insert(*predecessor)
                        && predecessor != destination
                    {
                        pending.push(*predecessor);
                    }
                }
            }
            loop_edges.entry(*destination).or_default().insert(MirEdge {
                source: *source,
                destination: *destination,
            });
        }
    }

    let mut regions = loop_members
        .iter()
        .map(|(header, members)| {
            mir_loop_region(
                *header,
                members,
                loop_edges.get(header).cloned().unwrap_or_default(),
                &successors,
            )
        })
        .collect::<Vec<_>>();

    // Rust source normally yields reducible loops. Keep a deterministic SCC
    // fallback so unusual compiler CFGs are still represented rather than
    // silently disappearing from the topology summary.
    let mut assigned = BTreeSet::new();
    for seed in &reachable {
        if assigned.contains(seed) {
            continue;
        }
        let forward = mir_reachable_blocks(*seed, &successors);
        let members = forward
            .into_iter()
            .filter(|candidate| mir_path_exists(edges, *candidate, *seed))
            .collect::<BTreeSet<_>>();
        assigned.extend(members.iter().copied());
        let recursive =
            members.len() > 1 || successors.get(seed).is_some_and(|next| next.contains(seed));
        if !recursive
            || regions.iter().any(|region| {
                region.kind == ControlFlowRegionKind::Loop
                    && members
                        .iter()
                        .all(|block| region.member_blocks.contains(block))
            })
        {
            continue;
        }
        let entries = members
            .iter()
            .filter(|block| {
                predecessors
                    .get(block)
                    .into_iter()
                    .flatten()
                    .any(|predecessor| !members.contains(predecessor))
            })
            .copied()
            .collect::<Vec<_>>();
        let header = entries.first().copied().unwrap_or(*seed);
        let cycle_edges = edges
            .iter()
            .filter(|edge| members.contains(&edge.source) && members.contains(&edge.destination))
            .cloned()
            .collect::<BTreeSet<_>>();
        regions.push(mir_loop_region(header, &members, cycle_edges, &successors));
    }

    let loop_control_blocks = regions
        .iter()
        .filter(|region| region.kind == ControlFlowRegionKind::Loop)
        .flat_map(|region| {
            std::iter::once(region.header_block)
                .chain(region.cycle_edges.iter().map(|edge| edge.source))
        })
        .collect::<BTreeSet<_>>();
    let postdominators = mir_postdominator_sets(&reachable, &successors);
    for header in &reachable {
        if loop_control_blocks.contains(header) {
            continue;
        }
        let entries = successors.get(header).cloned().unwrap_or_default();
        if entries.len() < 2 {
            continue;
        }
        let join = mir_immediate_postdominator(*header, &postdominators);
        let mut members = BTreeSet::from([*header]);
        let arms = entries
            .iter()
            .map(|entry| {
                let arm_members =
                    mir_reachable_before_join(*entry, join, *header, &reachable, &successors);
                members.extend(arm_members.iter().copied());
                MirControlFlowArmSummary {
                    entry_block: *entry,
                    member_blocks: arm_members.into_iter().collect(),
                }
            })
            .collect::<Vec<_>>();
        let exit_blocks = if let Some(join) = join {
            vec![join]
        } else {
            members
                .iter()
                .flat_map(|source| successors.get(source).into_iter().flatten())
                .filter(|destination| !members.contains(destination))
                .copied()
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect()
        };
        regions.push(MirControlFlowRegionSummary {
            kind: ControlFlowRegionKind::Branch,
            header_block: *header,
            member_blocks: members.into_iter().collect(),
            entry_blocks: entries.into_iter().collect(),
            exit_blocks,
            join_block: join,
            arms,
            cycle_edges: Vec::new(),
        });
    }
    regions.sort_by_key(|region| (region.header_block, region.kind));
    regions
}

fn mir_reachable_blocks(
    entry: usize,
    successors: &BTreeMap<usize, BTreeSet<usize>>,
) -> BTreeSet<usize> {
    let mut reachable = BTreeSet::new();
    let mut pending = vec![entry];
    while let Some(block) = pending.pop() {
        if !reachable.insert(block) {
            continue;
        }
        pending.extend(successors.get(&block).into_iter().flatten().copied());
    }
    reachable
}

fn mir_dominator_sets(
    entry: usize,
    reachable: &BTreeSet<usize>,
    predecessors: &BTreeMap<usize, BTreeSet<usize>>,
) -> BTreeMap<usize, BTreeSet<usize>> {
    let mut dominators = reachable
        .iter()
        .map(|block| {
            let initial = if *block == entry {
                BTreeSet::from([*block])
            } else {
                reachable.clone()
            };
            (*block, initial)
        })
        .collect::<BTreeMap<_, _>>();
    loop {
        let mut changed = false;
        for block in reachable.iter().filter(|block| **block != entry) {
            let incoming = predecessors
                .get(block)
                .into_iter()
                .flatten()
                .filter(|predecessor| reachable.contains(predecessor))
                .collect::<Vec<_>>();
            let mut next = incoming
                .first()
                .and_then(|first| dominators.get(*first))
                .cloned()
                .unwrap_or_default();
            for predecessor in incoming.iter().skip(1) {
                if let Some(set) = dominators.get(*predecessor) {
                    next.retain(|candidate| set.contains(candidate));
                }
            }
            next.insert(*block);
            if dominators.get(block) != Some(&next) {
                dominators.insert(*block, next);
                changed = true;
            }
        }
        if !changed {
            return dominators;
        }
    }
}

fn mir_postdominator_sets(
    reachable: &BTreeSet<usize>,
    successors: &BTreeMap<usize, BTreeSet<usize>>,
) -> BTreeMap<usize, BTreeSet<usize>> {
    let exits = reachable
        .iter()
        .filter(|block| {
            successors
                .get(block)
                .is_none_or(|next| next.iter().all(|successor| !reachable.contains(successor)))
        })
        .copied()
        .collect::<BTreeSet<_>>();
    if exits.is_empty() {
        return BTreeMap::new();
    }
    let mut postdominators = reachable
        .iter()
        .map(|block| {
            let initial = if exits.contains(block) {
                BTreeSet::from([*block])
            } else {
                reachable.clone()
            };
            (*block, initial)
        })
        .collect::<BTreeMap<_, _>>();
    loop {
        let mut changed = false;
        for block in reachable.iter().filter(|block| !exits.contains(block)) {
            let next_blocks = successors
                .get(block)
                .into_iter()
                .flatten()
                .filter(|successor| reachable.contains(successor))
                .collect::<Vec<_>>();
            let mut next = next_blocks
                .first()
                .and_then(|first| postdominators.get(*first))
                .cloned()
                .unwrap_or_default();
            for successor in next_blocks.iter().skip(1) {
                if let Some(set) = postdominators.get(*successor) {
                    next.retain(|candidate| set.contains(candidate));
                }
            }
            next.insert(*block);
            if postdominators.get(block) != Some(&next) {
                postdominators.insert(*block, next);
                changed = true;
            }
        }
        if !changed {
            return postdominators;
        }
    }
}

fn mir_immediate_postdominator(
    block: usize,
    postdominators: &BTreeMap<usize, BTreeSet<usize>>,
) -> Option<usize> {
    let strict = postdominators
        .get(&block)?
        .iter()
        .filter(|candidate| **candidate != block)
        .copied()
        .collect::<Vec<_>>();
    strict.iter().find_map(|candidate| {
        (!strict.iter().any(|other| {
            other != candidate
                && postdominators
                    .get(other)
                    .is_some_and(|set| set.contains(candidate))
        }))
        .then_some(*candidate)
    })
}

fn mir_reachable_before_join(
    entry: usize,
    join: Option<usize>,
    header: usize,
    reachable: &BTreeSet<usize>,
    successors: &BTreeMap<usize, BTreeSet<usize>>,
) -> BTreeSet<usize> {
    let mut members = BTreeSet::new();
    let mut pending = vec![entry];
    while let Some(block) = pending.pop() {
        if block == header || join == Some(block) || !reachable.contains(&block) {
            continue;
        }
        if !members.insert(block) {
            continue;
        }
        pending.extend(successors.get(&block).into_iter().flatten().copied());
    }
    members
}

fn mir_loop_region(
    header: usize,
    members: &BTreeSet<usize>,
    cycle_edges: BTreeSet<MirEdge>,
    successors: &BTreeMap<usize, BTreeSet<usize>>,
) -> MirControlFlowRegionSummary {
    let exit_blocks = members
        .iter()
        .flat_map(|source| successors.get(source).into_iter().flatten())
        .filter(|destination| !members.contains(destination))
        .copied()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    MirControlFlowRegionSummary {
        kind: ControlFlowRegionKind::Loop,
        header_block: header,
        member_blocks: members.iter().copied().collect(),
        entry_blocks: vec![header],
        exit_blocks,
        join_block: None,
        arms: Vec::new(),
        cycle_edges: cycle_edges.into_iter().collect(),
    }
}

/// Compare compiler-produced MIR effects and structured regions with
/// deterministic SBPF obligations. Instruction counts and block identifiers
/// are deliberately not compared because one source operation can lower to
/// several compiler operations, but every recovered branch/loop region must
/// have its own shape-compatible MIR region.
pub fn compare_mir_effects(
    pack: &EvidencePack,
    summary: Option<&MirProgramSummary>,
) -> MirEffectComparison {
    #[derive(Default)]
    struct RequiredDependencies {
        all: BTreeSet<u8>,
        target: BTreeSet<u8>,
        value: BTreeSet<u8>,
    }

    let Some(summary) = summary else {
        return MirEffectComparison {
            schema_version: MIR_EFFECT_COMPARISON_SCHEMA_VERSION,
            status: MirEffectComparisonStatus::NotAvailable,
            functions: Vec::new(),
            limitations: vec!["rustc MIR output was not available for comparison".into()],
        };
    };
    let mut functions = Vec::new();
    let mut has_mismatch = false;
    for function in &pack.ir.functions {
        let required = required_structure_effects(pack, &function.id)
            .into_iter()
            .map(|effect| effect.kind)
            .collect::<BTreeSet<_>>();
        let observed = summary
            .functions
            .iter()
            .find(|candidate| candidate.name == function.id)
            .map(mir_observed_effects)
            .unwrap_or_default();
        let missing = required.difference(&observed).copied().collect::<Vec<_>>();
        let candidate = summary
            .functions
            .iter()
            .find(|candidate| candidate.name == function.id);
        let dependency_mismatches = required_structure_effects(pack, &function.id)
            .into_iter()
            .fold(
                BTreeMap::<RecoveredEffectKind, RequiredDependencies>::new(),
                |mut dependencies, effect| {
                    let required = dependencies.entry(effect.kind).or_default();
                    required.all.extend(effect.input_registers);
                    required.target.extend(effect.target_input_registers);
                    required.value.extend(effect.value_input_registers);
                    dependencies
                },
            )
            .into_iter()
            .filter_map(|(kind, required)| {
                let observed_dependency = candidate.and_then(|candidate| {
                    candidate
                        .effect_dependencies
                        .iter()
                        .find(|dependency| dependency.kind == kind)
                });
                let observed_inputs = observed_dependency
                    .map(|dependency| {
                        dependency
                            .input_parameters
                            .iter()
                            .copied()
                            .collect::<BTreeSet<_>>()
                    })
                    .unwrap_or_default();
                let observed_targets = observed_dependency
                    .map(|dependency| {
                        dependency
                            .target_input_parameters
                            .iter()
                            .copied()
                            .collect::<BTreeSet<_>>()
                    })
                    .unwrap_or_default();
                let observed_values = observed_dependency
                    .map(|dependency| {
                        dependency
                            .value_input_parameters
                            .iter()
                            .copied()
                            .collect::<BTreeSet<_>>()
                    })
                    .unwrap_or_default();
                let missing_inputs = required
                    .all
                    .difference(&observed_inputs)
                    .copied()
                    .collect::<Vec<_>>();
                let missing_targets = required
                    .target
                    .difference(&observed_targets)
                    .copied()
                    .collect::<Vec<_>>();
                let missing_values = required
                    .value
                    .difference(&observed_values)
                    .copied()
                    .collect::<Vec<_>>();
                (!missing_inputs.is_empty()
                    || !missing_targets.is_empty()
                    || !missing_values.is_empty())
                .then(|| MirDependencyMismatch {
                    kind,
                    required_input_registers: required.all.into_iter().collect(),
                    observed_input_parameters: observed_inputs.into_iter().collect(),
                    missing_input_dependencies: missing_inputs,
                    required_target_input_registers: required.target.into_iter().collect(),
                    observed_target_input_parameters: observed_targets.into_iter().collect(),
                    missing_target_input_dependencies: missing_targets,
                    required_value_input_registers: required.value.into_iter().collect(),
                    observed_value_input_parameters: observed_values.into_iter().collect(),
                    missing_value_input_dependencies: missing_values,
                })
            })
            .collect::<Vec<_>>();
        let memory_write_mismatches = required_structure_effects(pack, &function.id)
            .into_iter()
            .filter(|effect| effect.kind == RecoveredEffectKind::MemoryWrite)
            .filter(|effect| {
                !candidate.is_some_and(|candidate| {
                    candidate.memory_write_dependencies.iter().any(|observed| {
                        (effect.target_input_registers.is_empty()
                            || observed.target_input_parameters == effect.target_input_registers)
                            && (effect.value_input_registers.is_empty()
                                || observed.value_input_parameters == effect.value_input_registers)
                    })
                })
            })
            .map(|effect| MirMemoryWriteMismatch {
                required_location: effect.location,
                required_target_input_registers: effect.target_input_registers,
                required_value_input_registers: effect.value_input_registers,
                observed_role_signatures: candidate
                    .map(|candidate| candidate.memory_write_dependencies.clone())
                    .unwrap_or_default(),
            })
            .collect::<Vec<_>>();
        let region_mismatches = compare_mir_regions(
            &required_control_flow_regions(pack, &function.id),
            candidate
                .map(|candidate| candidate.control_flow_regions.as_slice())
                .unwrap_or_default(),
        );
        let observed_call_targets = candidate
            .map(|candidate| {
                candidate
                    .direct_calls
                    .iter()
                    .map(|call| call.target.clone())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let observed_target_counts = observed_call_targets.iter().fold(
            BTreeMap::<String, usize>::new(),
            |mut counts, target| {
                *counts.entry(target.clone()).or_default() += 1;
                counts
            },
        );
        let mut matched_target_counts = BTreeMap::<String, usize>::new();
        let direct_call_mismatches = required_direct_call_contracts(pack, &function.id)
            .into_iter()
            .filter_map(|contract| {
                let matched = matched_target_counts
                    .entry(contract.target_function_id.clone())
                    .or_default();
                if *matched
                    < observed_target_counts
                        .get(&contract.target_function_id)
                        .copied()
                        .unwrap_or_default()
                {
                    *matched += 1;
                    return None;
                }
                Some(MirDirectCallMismatch {
                    required_contract_id: contract.id,
                    required_location: format!("operation:{}", contract.operation_id),
                    required_target_function_id: contract.target_function_id,
                    observed_targets: observed_call_targets.clone(),
                })
            })
            .collect::<Vec<_>>();
        has_mismatch |= !missing.is_empty()
            || !dependency_mismatches.is_empty()
            || !memory_write_mismatches.is_empty()
            || !region_mismatches.is_empty()
            || !direct_call_mismatches.is_empty()
            || !summary
                .functions
                .iter()
                .any(|candidate| candidate.name == function.id);
        functions.push(MirFunctionEffectComparison {
            function_id: function.id.clone(),
            required: required.into_iter().collect(),
            observed: observed.into_iter().collect(),
            missing,
            dependency_mismatches,
            memory_write_mismatches,
            region_mismatches,
            direct_call_mismatches,
        });
    }
    MirEffectComparison {
        schema_version: MIR_EFFECT_COMPARISON_SCHEMA_VERSION,
        status: if has_mismatch {
            MirEffectComparisonStatus::Mismatch
        } else {
            MirEffectComparisonStatus::Covered
        },
        functions,
        limitations: vec![
            "MIR validation checks effect/dependency coverage, resolved direct-call targets, and one-to-one branch/loop region topology, not full semantic equivalence or one-to-one instruction lowering".into(),
            "extra rustc regions are permitted for bounds checks, cleanup, and other compiler-introduced control flow, but cannot satisfy more than one binary-derived region".into(),
            "region matching preserves branch arity, join/exit presence, and recovered region nesting while allowing compiler block identifiers and member counts to differ".into(),
            "memory-write target/value role signatures are checked per required SBPF store, but identical signatures may reuse one MIR assignment because one source assignment can lower to multiple stores".into(),
            "rustc textual MIR is parsed into a versioned SOS summary because the upstream text format is not stable".into(),
        ],
    }
}

fn compare_mir_regions(
    required: &[ControlFlowRegionContract],
    observed: &[MirControlFlowRegionSummary],
) -> Vec<MirRegionMismatch> {
    let mut required_order = (0..required.len()).collect::<Vec<_>>();
    required_order.sort_by_key(|index| {
        (
            std::cmp::Reverse(required[*index].member_blocks.len()),
            required[*index].kind,
            required[*index].header_block_id.clone(),
        )
    });
    let mut used = BTreeSet::new();
    let mut matched = BTreeMap::<usize, usize>::new();
    let mut mismatches = BTreeMap::<usize, MirRegionMismatch>::new();

    for required_index in required_order {
        let contract = &required[required_index];
        let mut compatible = observed
            .iter()
            .enumerate()
            .filter(|(index, candidate)| {
                !used.contains(index)
                    && candidate.kind == contract.kind
                    && mir_region_shape_reasons(contract, candidate).is_empty()
                    && matched.iter().all(|(parent_required, parent_observed)| {
                        !required[*parent_required]
                            .member_blocks
                            .contains(&contract.header_block_id)
                            || observed[*parent_observed]
                                .member_blocks
                                .contains(&candidate.header_block)
                    })
            })
            .map(|(index, candidate)| {
                (
                    contract
                        .member_blocks
                        .len()
                        .abs_diff(candidate.member_blocks.len()),
                    candidate.header_block,
                    index,
                )
            })
            .collect::<Vec<_>>();
        compatible.sort_unstable();
        if let Some((_, _, observed_index)) = compatible.first().copied() {
            used.insert(observed_index);
            matched.insert(required_index, observed_index);
            continue;
        }

        let same_kind = observed
            .iter()
            .filter(|candidate| candidate.kind == contract.kind)
            .collect::<Vec<_>>();
        let mut reasons = same_kind
            .iter()
            .flat_map(|candidate| mir_region_shape_reasons(contract, candidate))
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        if same_kind.is_empty() {
            reasons.push(format!("no {:?} MIR region was recovered", contract.kind));
        } else if reasons.is_empty() {
            reasons.push(
                "all shape-compatible MIR regions were already used or violate recovered nesting"
                    .into(),
            );
        }
        mismatches.insert(
            required_index,
            mir_region_mismatch(contract, observed, reasons),
        );
    }

    // Verify all pairwise containment relationships after matching. This
    // catches flattened nested control flow as well as invented nesting among
    // regions that were siblings in the binary CFG.
    for (left_required, left_observed) in &matched {
        for (right_required, right_observed) in &matched {
            if left_required == right_required {
                continue;
            }
            let required_contains = required[*left_required]
                .member_blocks
                .contains(&required[*right_required].header_block_id);
            let observed_contains = observed[*left_observed]
                .member_blocks
                .contains(&observed[*right_observed].header_block);
            if required_contains == observed_contains {
                continue;
            }
            let reason = if required_contains {
                format!(
                    "MIR region does not preserve containment inside {}",
                    required[*left_required].id
                )
            } else {
                format!(
                    "MIR region is unexpectedly nested inside {}",
                    required[*left_required].id
                )
            };
            mismatches
                .entry(*right_required)
                .or_insert_with(|| {
                    mir_region_mismatch(&required[*right_required], observed, Vec::new())
                })
                .reasons
                .push(reason);
        }
    }

    mismatches.into_values().collect()
}

fn mir_region_shape_reasons(
    required: &ControlFlowRegionContract,
    observed: &MirControlFlowRegionSummary,
) -> Vec<String> {
    let mut reasons = Vec::new();
    if required.kind != observed.kind {
        reasons.push(format!(
            "required {:?}, observed {:?}",
            required.kind, observed.kind
        ));
        return reasons;
    }
    if required.entry_blocks.len() != observed.entry_blocks.len() {
        reasons.push(format!(
            "required {} entries/arms, observed {}",
            required.entry_blocks.len(),
            observed.entry_blocks.len()
        ));
    }
    if required.join_block_id.is_some() != observed.join_block.is_some() {
        reasons.push(format!(
            "required join={}, observed join={}",
            required.join_block_id.is_some(),
            observed.join_block.is_some()
        ));
    }
    if required.exit_blocks.is_empty() != observed.exit_blocks.is_empty() {
        reasons.push(format!(
            "required exit={}, observed exit={}",
            !required.exit_blocks.is_empty(),
            !observed.exit_blocks.is_empty()
        ));
    }
    if required.kind == ControlFlowRegionKind::Loop && observed.cycle_edges.is_empty() {
        reasons.push("observed loop has no cycle edge".into());
    }
    reasons
}

fn mir_region_mismatch(
    required: &ControlFlowRegionContract,
    observed: &[MirControlFlowRegionSummary],
    reasons: Vec<String>,
) -> MirRegionMismatch {
    MirRegionMismatch {
        required_region_id: required.id.clone(),
        kind: required.kind,
        required_entry_count: required.entry_blocks.len(),
        required_has_join: required.join_block_id.is_some(),
        required_has_exit: !required.exit_blocks.is_empty(),
        observed_candidate_headers: observed
            .iter()
            .filter(|candidate| candidate.kind == required.kind)
            .map(|candidate| candidate.header_block)
            .collect(),
        reasons,
    }
}

fn mir_observed_effects(function: &MirFunctionSummary) -> BTreeSet<RecoveredEffectKind> {
    let mut observed = BTreeSet::new();
    if function.memory_writes > 0 {
        observed.insert(RecoveredEffectKind::MemoryWrite);
    }
    if function.branch_terminators > 0 {
        observed.insert(RecoveredEffectKind::Branch);
    }
    if function.has_loop {
        observed.insert(RecoveredEffectKind::Loop);
    }
    if function.calls > 0 {
        observed.insert(RecoveredEffectKind::Call);
    }
    if function.returns > 0 {
        observed.insert(RecoveredEffectKind::Return);
    }
    observed
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
            mir_summary: None,
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
    let (layout_assertions, mut layout_limitations) = match compile_layout_assertions(refinement) {
        Ok(assertions) => assertions,
        Err(error) => {
            let source_sha256 = sha256_bytes(source.as_bytes());
            return CompilationResult {
                schema_version: VALIDATION_SCHEMA_VERSION,
                status: CompilationStatus::Failed,
                compiler: compiler.map(|path| path.display().to_string()),
                source_sha256,
                diagnostics: vec![error],
                mir_summary: None,
                limitations: vec![
                    "memory-layout assertions reject malformed or conflicting model mappings"
                        .into(),
                ],
            };
        }
    };
    source.push_str(&layout_assertions);
    let source_sha256 = sha256_bytes(source.as_bytes());

    let Some(compiler) = compiler else {
        return CompilationResult {
            schema_version: VALIDATION_SCHEMA_VERSION,
            status: CompilationStatus::NotAvailable,
            compiler: None,
            source_sha256,
            diagnostics: Vec::new(),
            mir_summary: None,
            limitations: {
                layout_limitations.push("no Rust compiler was configured".into());
                layout_limitations
            },
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
                mir_summary: None,
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
    let mir_path = base.with_extension("mir");
    if let Err(error) = fs::write(&source_path, source.as_bytes()) {
        return CompilationResult {
            schema_version: VALIDATION_SCHEMA_VERSION,
            status: CompilationStatus::Failed,
            compiler: Some(compiler.display().to_string()),
            source_sha256,
            diagnostics: vec![format!("unable to create compiler input: {error}")],
            mir_summary: None,
            limitations: Vec::new(),
        };
    }

    let mut child = match Command::new(compiler)
        .args(["--crate-type", "lib", "--emit"])
        .arg(format!(
            "metadata={},mir={}",
            output_path.display(),
            mir_path.display()
        ))
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
                mir_summary: None,
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
                let _ = fs::remove_file(&mir_path);
                return CompilationResult {
                    schema_version: VALIDATION_SCHEMA_VERSION,
                    status: CompilationStatus::TimedOut,
                    compiler: Some(compiler.display().to_string()),
                    source_sha256,
                    diagnostics: vec![format!(
                        "compiler timed out after {} ms",
                        timeout.as_millis()
                    )],
                    mir_summary: None,
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
                let _ = fs::remove_file(&mir_path);
                return CompilationResult {
                    schema_version: VALIDATION_SCHEMA_VERSION,
                    status: CompilationStatus::Failed,
                    compiler: Some(compiler.display().to_string()),
                    source_sha256,
                    diagnostics: vec![format!("compiler wait failed: {error}")],
                    mir_summary: None,
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
    let status = match status {
        Ok(status) if status.success() => CompilationStatus::Passed,
        Ok(_) => CompilationStatus::Failed,
        Err(error) => {
            diagnostics.push(error.to_string());
            CompilationStatus::Failed
        }
    };
    let mir_summary = if status == CompilationStatus::Passed {
        fs::metadata(&mir_path)
            .ok()
            .filter(|metadata| metadata.len() <= 4 * 1024 * 1024)
            .and_then(|_| fs::read_to_string(&mir_path).ok())
            .map(|mir| summarize_rustc_mir(&mir))
    } else {
        None
    };
    let _ = fs::remove_file(&source_path);
    let _ = fs::remove_file(&output_path);
    let _ = fs::remove_file(&mir_path);
    CompilationResult {
        schema_version: VALIDATION_SCHEMA_VERSION,
        status,
        compiler: Some(compiler.display().to_string()),
        source_sha256,
        diagnostics,
        mir_summary,
        limitations: {
            layout_limitations.push(
                "compilation checks syntax, types, and bounded MIR structure only; it does not execute the candidate".into(),
            );
            layout_limitations
        },
    }
}

fn compile_layout_assertions(
    refinement: &RefinementOutput,
) -> Result<(String, Vec<String>), String> {
    let mut fields = BTreeMap::<(String, String), (i32, String)>::new();
    let mut limitations = Vec::new();
    for mapping in refinement
        .patches
        .iter()
        .flat_map(|patch| patch.memory_field_mappings.iter())
    {
        if !is_rust_identifier(&mapping.layout_type) || !is_rust_identifier(&mapping.layout_field) {
            return Err(format!(
                "memory field mapping {} contains an invalid Rust layout identifier",
                mapping.contract_id
            ));
        }
        if !matches!(
            mapping.type_name.as_str(),
            "u8" | "u16" | "u32" | "u64" | "unknown_bytes"
        ) {
            return Err(format!(
                "memory field mapping {} contains unsupported layout type {}",
                mapping.contract_id, mapping.type_name
            ));
        }
        let key = (mapping.layout_type.clone(), mapping.layout_field.clone());
        let expected = (mapping.offset, mapping.type_name.clone());
        if let Some(previous) = fields.insert(key.clone(), expected.clone()) {
            if previous != expected {
                return Err(format!(
                    "conflicting memory layout expectations for {}.{}",
                    key.0, key.1
                ));
            }
        }
    }

    let mut source = String::from("\n// SOS compiler-generated memory layout assertions.\n");
    for (index, ((layout_type, layout_field), (offset, type_name))) in
        fields.into_iter().enumerate()
    {
        if offset < 0 {
            limitations.push(format!(
                "negative machine-relative offset {offset} for {layout_type}.{layout_field} cannot be represented by Rust offset_of!"
            ));
            continue;
        }
        source.push_str(&format!(
            "const _: () = assert!(std::mem::offset_of!({layout_type}, {layout_field}) == {offset}usize);\n"
        ));
        if type_name != "unknown_bytes" {
            source.push_str(&format!(
                "#[allow(dead_code)] fn __sos_layout_type_assert_{index}(value: &{layout_type}) {{ let _: &{type_name} = &value.{layout_field}; }}\n"
            ));
        }
    }
    Ok((source, limitations))
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
    pub max_output_tokens: usize,
    pub prompt_version: String,
    pub status: RefinementLoopStatus,
    pub rounds: Vec<RefinementRound>,
    pub final_output: Option<RefinementOutput>,
    pub limitations: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RefinementLoopLimits {
    pub max_output_tokens: usize,
    pub max_attempts: usize,
    pub max_rounds: usize,
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
    run_refinement_loop_with_limits(
        provider,
        base_pack,
        model,
        prompt_version,
        RefinementLoopLimits {
            max_output_tokens: default_stage_output_tokens(ModelTask::Refinement),
            max_attempts,
            max_rounds,
        },
        validate,
    )
}

pub fn run_refinement_loop_with_limits<P, F>(
    provider: P,
    base_pack: EvidencePack,
    model: impl Into<String>,
    prompt_version: impl Into<String>,
    limits: RefinementLoopLimits,
    validate: F,
) -> RefinementLoopResult
where
    P: ModelProvider,
    F: Fn(&RefinementOutput) -> RefinementFeedback,
{
    let gateway = ModelGateway::new(provider, 4 * 1024 * 1024, 1024 * 1024);
    let model = model.into();
    let prompt_version = prompt_version.into();
    let round_limit = limits.max_rounds.clamp(1, 8);
    let max_output_tokens = limits.max_output_tokens;
    let mut feedback = None;
    let mut rounds = Vec::with_capacity(round_limit);
    let mut final_output = None;

    for round in 1..=round_limit {
        let pack = refinement_pack_with_feedback(&base_pack, feedback.as_ref());
        let input_sha256 = sha256_json(&pack);
        let result = gateway.submit_with_retry_and_token_budget(
            ModelTask::Refinement,
            model.clone(),
            prompt_version.clone(),
            pack,
            limits.max_attempts,
            max_output_tokens,
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
                    max_output_tokens,
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
                        max_output_tokens,
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
                max_output_tokens,
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
        max_output_tokens,
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
    if response.model != request.model {
        return Err(AiError::InvalidResponse(format!(
            "model response identity {:?} does not match request {:?}",
            response.model, request.model
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

/// Revalidate a persisted provider response against a fresh immutable evidence
/// pack before cache reuse. This applies the same task, model, prompt, schema,
/// anti-cheese, and evidence-reference checks as a live gateway submission.
pub fn validate_model_response_for_pack(
    model: &str,
    prompt_version: &str,
    mut pack: EvidencePack,
    response: &ModelResponse,
) -> Result<(), AiError> {
    const MAX_CACHED_INPUT_BYTES: usize = 4 * 1024 * 1024;
    const MAX_CACHED_OUTPUT_BYTES: usize = 1024 * 1024;
    pack.control_flow_regions = derive_control_flow_regions(&pack.ir);
    pack.memory_field_contracts =
        derive_memory_field_contracts(&pack.analysis, &pack.ir, &pack.semantics);
    pack.direct_call_contracts = derive_direct_call_contracts(&pack.analysis, &pack.ir);
    let input_bytes = serde_json::to_vec(&pack)
        .map_err(|error| AiError::InvalidResponse(error.to_string()))?
        .len();
    if input_bytes > MAX_CACHED_INPUT_BYTES {
        return Err(AiError::EvidencePackTooLarge {
            bytes: input_bytes,
            limit: MAX_CACHED_INPUT_BYTES,
        });
    }
    let output_bytes = serde_json::to_vec(response)
        .map_err(|error| AiError::InvalidResponse(error.to_string()))?
        .len();
    if output_bytes > MAX_CACHED_OUTPUT_BYTES {
        return Err(AiError::OutputTooLarge {
            bytes: output_bytes,
            limit: MAX_CACHED_OUTPUT_BYTES,
        });
    }
    let request = ModelRequest {
        task: pack.task,
        prompt_version: prompt_version.into(),
        model: model.into(),
        evidence_pack: pack,
        max_output_bytes: MAX_CACHED_OUTPUT_BYTES,
        max_output_tokens: default_stage_output_tokens(response.task),
    };
    validate_response_for_request(&request, response)
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
        ModelTask::StructureRecovery | ModelTask::PseudocodeHumanization => {
            let parsed: StructureRecoveryOutput = serde_json::from_value(output.clone())
                .map_err(|error| AiError::InvalidResponse(error.to_string()))?;
            if !known_functions.is_empty() && parsed.functions.is_empty() {
                return Err(AiError::InvalidResponse(
                    "structure recovery returned no candidate for a known IR function".into(),
                ));
            }
            for function in parsed.functions {
                if !known_functions.contains(function.function_id.as_str()) {
                    return Err(AiError::InvalidResponse(format!(
                        "structure recovery references unknown function {}",
                        function.function_id
                    )));
                }
                if request.task == ModelTask::StructureRecovery {
                    validate_structure_candidate(request, &function)?;
                } else {
                    validate_pseudocode_humanization_candidate(request, &function)?;
                }
            }
        }
        ModelTask::SignatureRecovery => {
            let parsed: SignatureRecoveryOutput = serde_json::from_value(output.clone())
                .map_err(|error| AiError::InvalidResponse(error.to_string()))?;
            if !known_functions.is_empty() && parsed.signatures.is_empty() {
                return Err(AiError::InvalidResponse(
                    "signature recovery returned no candidate for a known IR function".into(),
                ));
            }
            if parsed.signatures.len() > 128 {
                return Err(AiError::InvalidResponse(
                    "signature recovery contains more than 128 signatures".into(),
                ));
            }
            let mut recovered_functions = BTreeSet::new();
            for signature in parsed.signatures {
                if !known_functions.contains(signature.function_id.as_str()) {
                    return Err(AiError::InvalidResponse(format!(
                        "signature recovery references unknown function {}",
                        signature.function_id
                    )));
                }
                if !recovered_functions.insert(signature.function_id.clone()) {
                    return Err(AiError::InvalidResponse(format!(
                        "signature recovery contains duplicate function {}",
                        signature.function_id
                    )));
                }
                validate_signature_identifier(&signature.function_name, "function name")?;
                validate_signature_type(&signature.return_type, "return type")?;
                if signature.parameters.len() > 32 {
                    return Err(AiError::InvalidResponse(format!(
                        "signature for {} contains more than 32 parameters",
                        signature.function_id
                    )));
                }
                let function_pc = request
                    .evidence_pack
                    .ir
                    .functions
                    .iter()
                    .find(|function| function.id == signature.function_id)
                    .map(|function| function.entry_pc)
                    .ok_or_else(|| {
                        AiError::InvalidResponse(format!(
                            "signature recovery cannot resolve {} entry PC",
                            signature.function_id
                        ))
                    })?;
                let observed_parameters = sos_analysis::observed_abi_parameters(
                    &request.evidence_pack.analysis,
                    function_pc,
                );
                let mut mapped_registers = BTreeSet::new();
                for parameter in &signature.parameters {
                    validate_signature_identifier(&parameter.name, "parameter name")?;
                    validate_signature_type(&parameter.type_name, "parameter type")?;
                    if let Some(register) = parameter.abi_register {
                        if !(1..=5).contains(&register) {
                            return Err(AiError::InvalidResponse(format!(
                                "signature parameter {} maps to invalid SBPF ABI register r{}",
                                parameter.name, register
                            )));
                        }
                        if !mapped_registers.insert(register) {
                            return Err(AiError::InvalidResponse(format!(
                                "signature for {} maps multiple parameters to r{}",
                                signature.function_id, register
                            )));
                        }
                    }
                    if let Some(role) = &parameter.role {
                        if role.len() > 128 || role.chars().any(|character| character.is_control())
                        {
                            return Err(AiError::InvalidResponse(
                                "signature parameter role is not bounded plain text".into(),
                            ));
                        }
                    }
                }
                for observed in observed_parameters {
                    if !mapped_registers.contains(&observed.register) {
                        return Err(AiError::InvalidResponse(format!(
                            "signature for {} omits observed SBPF ABI input r{}",
                            signature.function_id, observed.register
                        )));
                    }
                    let expected_location = format!("instruction:0x{:x}", observed.first_use_pc);
                    if !signature
                        .evidence
                        .iter()
                        .any(|evidence| evidence.location == expected_location)
                    {
                        return Err(AiError::InvalidResponse(format!(
                            "signature for {} must cite {} for observed SBPF ABI input r{}",
                            signature.function_id, expected_location, observed.register
                        )));
                    }
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
                validate_signature_contract(
                    &request.evidence_pack,
                    &patch.function_id,
                    &patch.replacement,
                    "refinement patch",
                )?;
                validate_memory_field_mappings(
                    &request.evidence_pack,
                    &patch.function_id,
                    &patch.replacement,
                    &patch.memory_field_mappings,
                    "refinement patch",
                )?;
                validate_direct_call_mappings(
                    &request.evidence_pack,
                    &patch.function_id,
                    &patch.replacement,
                    &patch.direct_call_mappings,
                    "refinement patch",
                )?;
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

fn validate_structure_candidate(
    request: &ModelRequest,
    function: &RecoveredFunction,
) -> Result<(), AiError> {
    let declaration = format!("fn {}(", function.function_id);
    if !function.pseudocode.contains(&declaration) {
        return Err(AiError::InvalidResponse(format!(
            "structure candidate for {} must preserve the IR identity with `{declaration}`",
            function.function_id
        )));
    }
    validate_signature_contract(
        &request.evidence_pack,
        &function.function_id,
        &function.pseudocode,
        "structure candidate",
    )?;
    let required_effects =
        required_structure_effects(&request.evidence_pack, &function.function_id);
    let expected_effects = required_effects
        .iter()
        .map(|required| {
            (
                required.kind,
                required.location.clone(),
                required.input_registers.clone(),
                required.target_input_registers.clone(),
                required.value_input_registers.clone(),
            )
        })
        .collect::<BTreeSet<_>>();
    let mut covered_effects = BTreeSet::new();
    for effect in &function.effects {
        if effect.evidence.len() != 1 || effect.evidence[0].source != "ir" {
            return Err(AiError::InvalidResponse(format!(
                "structure candidate for {} must cite exactly one IR location per effect",
                function.function_id
            )));
        }
        let mut normalized_inputs = effect.input_registers.clone();
        normalized_inputs.sort_unstable();
        normalized_inputs.dedup();
        let mut normalized_targets = effect.target_input_registers.clone();
        normalized_targets.sort_unstable();
        normalized_targets.dedup();
        let mut normalized_values = effect.value_input_registers.clone();
        normalized_values.sort_unstable();
        normalized_values.dedup();
        let role_union = normalized_targets
            .iter()
            .chain(&normalized_values)
            .copied()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        if [
            (&normalized_inputs, &effect.input_registers),
            (&normalized_targets, &effect.target_input_registers),
            (&normalized_values, &effect.value_input_registers),
        ]
        .iter()
        .any(|(normalized, supplied)| {
            normalized != supplied
                || normalized
                    .iter()
                    .any(|register| !(1..=5).contains(register))
        }) || effect.kind != RecoveredEffectKind::MemoryWrite
            && (!normalized_targets.is_empty() || !normalized_values.is_empty())
            || effect.kind == RecoveredEffectKind::MemoryWrite && role_union != normalized_inputs
        {
            return Err(AiError::InvalidResponse(format!(
                "structure candidate for {} has invalid, non-canonical, or inapplicable effect dependency roles",
                function.function_id
            )));
        }
        let key = (
            effect.kind,
            effect.evidence[0].location.clone(),
            normalized_inputs,
            normalized_targets,
            normalized_values,
        );
        if !expected_effects.contains(&key) {
            return Err(AiError::InvalidResponse(format!(
                "structure candidate for {} invents a binary effect not present in the deterministic contract",
                function.function_id
            )));
        }
        if !covered_effects.insert(key) {
            return Err(AiError::InvalidResponse(format!(
                "structure candidate for {} duplicates one deterministic binary effect",
                function.function_id
            )));
        }
    }
    let missing_effects = required_effects
        .iter()
        .filter(|required| {
            !covered_effects.contains(&(
                required.kind,
                required.location.clone(),
                required.input_registers.clone(),
                required.target_input_registers.clone(),
                required.value_input_registers.clone(),
            ))
        })
        .map(|required| {
            format!(
                "{:?}:{}<-all:{:?},target:{:?},value:{:?}",
                required.kind,
                required.location,
                required.input_registers,
                required.target_input_registers,
                required.value_input_registers
            )
        })
        .collect::<Vec<_>>();
    if !missing_effects.is_empty() {
        return Err(AiError::InvalidResponse(format!(
            "structure candidate for {} omits required binary effects: {}",
            function.function_id,
            missing_effects.join(", ")
        )));
    }
    validate_control_flow_mappings(request, function)?;
    validate_memory_field_mappings(
        &request.evidence_pack,
        &function.function_id,
        &function.pseudocode,
        &function.memory_field_mappings,
        "structure candidate",
    )?;
    validate_direct_call_mappings(
        &request.evidence_pack,
        &function.function_id,
        &function.pseudocode,
        &function.direct_call_mappings,
        "structure candidate",
    )?;
    let function_blocks = request
        .evidence_pack
        .ir
        .blocks
        .iter()
        .filter(|block| block.function_id == function.function_id)
        .map(|block| block.id.as_str())
        .collect::<BTreeSet<_>>();
    let operations = request
        .evidence_pack
        .ir
        .operations
        .iter()
        .filter(|operation| function_blocks.contains(operation.block_id.as_str()))
        .collect::<Vec<_>>();
    let has_loop = request.evidence_pack.ir.sccs.iter().any(|scc| {
        scc.recursive
            && scc
                .block_starts
                .iter()
                .any(|pc| function_blocks.contains(format!("bb_{pc:x}").as_str()))
    });
    let lower = function.pseudocode.to_ascii_lowercase();
    if has_loop
        && !["loop", "while", "for ", "unknown_loop"]
            .iter()
            .any(|token| lower.contains(token))
    {
        return Err(AiError::InvalidResponse(format!(
            "structure candidate for {} omits a deterministic CFG loop",
            function.function_id
        )));
    }
    let has_write = operations.iter().any(|operation| {
        operation.kind == "store"
            || operation
                .memory_effect
                .as_deref()
                .is_some_and(|effect| effect.starts_with("store:"))
    });
    if has_write
        && ![
            "] =",
            ".fill(",
            ".copy",
            "copy_",
            "memset",
            "write_",
            "store_",
            "unknown_memory_write",
        ]
        .iter()
        .any(|token| lower.contains(token))
    {
        return Err(AiError::InvalidResponse(format!(
            "structure candidate for {} omits deterministic memory writes",
            function.function_id
        )));
    }
    Ok(())
}

fn validate_memory_field_mappings(
    pack: &EvidencePack,
    function_id: &str,
    source: &str,
    mappings: &[RecoveredMemoryFieldMapping],
    candidate_kind: &str,
) -> Result<(), AiError> {
    let contracts = required_memory_field_contracts(pack, function_id)
        .into_iter()
        .map(|contract| (contract.id.clone(), contract))
        .collect::<BTreeMap<_, _>>();
    if mappings.len() != contracts.len() {
        return Err(AiError::InvalidResponse(format!(
            "{candidate_kind} for {} maps {} memory fields but {} are required",
            function_id,
            mappings.len(),
            contracts.len()
        )));
    }
    let compact_output = compact_signature(source);
    let mut covered = BTreeSet::new();
    let mut fragments = BTreeSet::new();
    for mapping in mappings {
        let Some(contract) = contracts.get(&mapping.contract_id) else {
            return Err(AiError::InvalidResponse(format!(
                "{candidate_kind} for {} maps unknown memory field contract {}",
                function_id, mapping.contract_id
            )));
        };
        if !covered.insert(mapping.contract_id.as_str()) {
            return Err(AiError::InvalidResponse(format!(
                "{candidate_kind} for {} maps memory field {} more than once",
                function_id, mapping.contract_id
            )));
        }
        if mapping.field_path != contract.field_path
            || mapping.type_name != contract.type_name
            || mapping.offset != contract.offset
            || mapping.layout_type != contract.layout_type
            || mapping.layout_field != contract.layout_field
        {
            return Err(AiError::InvalidResponse(format!(
                "memory field mapping {} must preserve deterministic path {}, type {}, offset {}, layout type {}, and layout field {}",
                mapping.contract_id,
                contract.field_path,
                contract.type_name,
                contract.offset,
                contract.layout_type,
                contract.layout_field
            )));
        }
        if candidate_kind == "refinement patch"
            && contract.offset >= 0
            && !source_declares_repr_c_layout(source, &contract.layout_type)
        {
            return Err(AiError::InvalidResponse(format!(
                "refinement patch for {} must declare #[repr(C)] struct {} for memory field contract {}",
                function_id, contract.layout_type, contract.id
            )));
        }
        let fragment = mapping.output_fragment.trim();
        let compact_fragment = compact_signature(fragment);
        if fragment.chars().count() < 2
            || fragment.chars().count() > 512
            || fragment.contains(['\n', '\r'])
            || fragment.starts_with("//")
            || fragment.starts_with("/*")
            || compact_fragment.is_empty()
            || !compact_output.contains(&compact_fragment)
            || !compact_fragment.contains(&compact_signature(&mapping.field_path))
        {
            return Err(AiError::InvalidResponse(format!(
                "memory field mapping {} must name a bounded one-line non-comment expression present in the pseudocode and containing {}",
                mapping.contract_id, mapping.field_path
            )));
        }
        if !fragments.insert(compact_fragment) {
            return Err(AiError::InvalidResponse(format!(
                "{candidate_kind} for {} reuses one source fragment for multiple memory fields",
                function_id
            )));
        }
        let supported_evidence = mapping.evidence.len() == 1
            && contract.evidence.iter().any(|evidence| {
                evidence.source == mapping.evidence[0].source
                    && evidence.location == mapping.evidence[0].location
            });
        if !supported_evidence {
            return Err(AiError::InvalidResponse(format!(
                "memory field mapping {} must cite exactly one instruction from its deterministic contract",
                mapping.contract_id
            )));
        }
    }
    Ok(())
}

fn validate_direct_call_mappings(
    pack: &EvidencePack,
    function_id: &str,
    source: &str,
    mappings: &[RecoveredDirectCallMapping],
    candidate_kind: &str,
) -> Result<(), AiError> {
    let contracts = required_direct_call_contracts(pack, function_id)
        .into_iter()
        .map(|contract| (contract.id.clone(), contract))
        .collect::<BTreeMap<_, _>>();
    if mappings.len() != contracts.len() {
        return Err(AiError::InvalidResponse(format!(
            "{candidate_kind} for {function_id} maps {} resolved direct calls but {} are required",
            mappings.len(),
            contracts.len()
        )));
    }
    let compact_source = rust_code_without_comments_and_literals(source)
        .chars()
        .filter(|character| !character.is_whitespace())
        .collect::<String>();
    let mut covered = BTreeSet::new();
    let mut fragments = BTreeSet::new();
    for mapping in mappings {
        let Some(contract) = contracts.get(&mapping.contract_id) else {
            return Err(AiError::InvalidResponse(format!(
                "{candidate_kind} for {function_id} maps unknown direct-call contract {}",
                mapping.contract_id
            )));
        };
        if !covered.insert(mapping.contract_id.as_str()) {
            return Err(AiError::InvalidResponse(format!(
                "{candidate_kind} for {function_id} maps direct call {} more than once",
                mapping.contract_id
            )));
        }
        if mapping.target_function_id != contract.target_function_id {
            return Err(AiError::InvalidResponse(format!(
                "direct-call mapping {} must preserve deterministic target {}",
                mapping.contract_id, contract.target_function_id
            )));
        }
        let fragment = mapping.output_fragment.trim();
        let compact_fragment = rust_code_without_comments_and_literals(fragment)
            .chars()
            .filter(|character| !character.is_whitespace())
            .collect::<String>();
        let expected_call = format!("{}(", contract.target_function_id);
        if fragment.chars().count() < 2
            || fragment.chars().count() > 512
            || fragment.contains(['\n', '\r'])
            || compact_fragment.is_empty()
            || !compact_source.contains(&compact_fragment)
            || !compact_fragment.contains(&expected_call)
        {
            return Err(AiError::InvalidResponse(format!(
                "direct-call mapping {} must name a bounded one-line source call to {} present in the candidate",
                mapping.contract_id, contract.target_function_id
            )));
        }
        if !fragments.insert(compact_fragment) {
            return Err(AiError::InvalidResponse(format!(
                "{candidate_kind} for {function_id} reuses one source fragment for multiple direct calls"
            )));
        }
        let supported_evidence = mapping.evidence.len() == 1
            && contract.evidence.iter().any(|evidence| {
                evidence.source == mapping.evidence[0].source
                    && evidence.location == mapping.evidence[0].location
            });
        if !supported_evidence {
            return Err(AiError::InvalidResponse(format!(
                "direct-call mapping {} must cite exactly its deterministic call operation",
                mapping.contract_id
            )));
        }
    }
    Ok(())
}

fn source_declares_repr_c_layout(source: &str, layout_type: &str) -> bool {
    if !is_rust_identifier(layout_type) {
        return false;
    }
    let compact = rust_code_without_comments_and_literals(source)
        .chars()
        .filter(|character| !character.is_whitespace())
        .collect::<String>();
    [
        format!("#[repr(C)]struct{layout_type}{{"),
        format!("#[repr(C)]pubstruct{layout_type}{{"),
        format!("#[repr(C)]pub(crate)struct{layout_type}{{"),
        format!("#[repr(C)]pub(super)struct{layout_type}{{"),
    ]
    .iter()
    .any(|declaration| compact.contains(declaration))
}

fn rust_code_without_comments_and_literals(source: &str) -> String {
    let bytes = source.as_bytes();
    let mut output = bytes.to_vec();
    let mut index = 0usize;
    while index < bytes.len() {
        if bytes[index..].starts_with(b"//") {
            let start = index;
            index += 2;
            while index < bytes.len() && bytes[index] != b'\n' {
                index += 1;
            }
            output[start..index].fill(b' ');
            continue;
        }
        if bytes[index..].starts_with(b"/*") {
            let start = index;
            index += 2;
            let mut depth = 1usize;
            while index < bytes.len() && depth > 0 {
                if bytes[index..].starts_with(b"/*") {
                    depth += 1;
                    index += 2;
                } else if bytes[index..].starts_with(b"*/") {
                    depth -= 1;
                    index += 2;
                } else {
                    index += 1;
                }
            }
            output[start..index].fill(b' ');
            continue;
        }
        let raw_prefix = if bytes[index..].starts_with(b"br") || bytes[index..].starts_with(b"cr") {
            Some(2usize)
        } else if bytes[index] == b'r' {
            Some(1usize)
        } else {
            None
        };
        if let Some(prefix_len) = raw_prefix {
            let mut marker = index + prefix_len;
            while marker < bytes.len() && bytes[marker] == b'#' {
                marker += 1;
            }
            if marker < bytes.len() && bytes[marker] == b'\"' {
                let hashes = marker - (index + prefix_len);
                let start = index;
                index = marker + 1;
                while index < bytes.len() {
                    if bytes[index] == b'\"'
                        && index + 1 + hashes <= bytes.len()
                        && bytes[index + 1..index + 1 + hashes]
                            .iter()
                            .all(|byte| *byte == b'#')
                    {
                        index += 1 + hashes;
                        break;
                    }
                    index += 1;
                }
                output[start..index].fill(b' ');
                continue;
            }
        }
        let quote = if bytes[index] == b'\"' {
            Some(b'\"')
        } else if bytes[index..].starts_with(b"b\"") || bytes[index..].starts_with(b"c\"") {
            index += 1;
            Some(b'\"')
        } else {
            None
        };
        if let Some(quote) = quote {
            let start = if index > 0 && matches!(bytes[index - 1], b'b' | b'c') {
                index - 1
            } else {
                index
            };
            index += 1;
            while index < bytes.len() {
                if bytes[index] == b'\\' {
                    index = (index + 2).min(bytes.len());
                } else if bytes[index] == quote {
                    index += 1;
                    break;
                } else {
                    index += 1;
                }
            }
            output[start..index].fill(b' ');
            continue;
        }
        index += 1;
    }
    String::from_utf8(output).unwrap_or_default()
}

fn is_rust_identifier(value: &str) -> bool {
    let mut characters = value.chars();
    matches!(characters.next(), Some(first) if first == '_' || first.is_ascii_alphabetic())
        && characters.all(|character| character == '_' || character.is_ascii_alphanumeric())
}

fn validate_control_flow_mappings(
    request: &ModelRequest,
    function: &RecoveredFunction,
) -> Result<(), AiError> {
    let contracts = required_control_flow_regions(&request.evidence_pack, &function.function_id)
        .into_iter()
        .map(|region| (region.id.clone(), region))
        .collect::<BTreeMap<_, _>>();
    if contracts.len() > 128 {
        return Err(AiError::InvalidResponse(format!(
            "function {} has {} deterministic control-flow regions, exceeding the supported mapping bound of 128",
            function.function_id,
            contracts.len()
        )));
    }
    if function.control_flow_mappings.len() != contracts.len() {
        return Err(AiError::InvalidResponse(format!(
            "structure candidate for {} maps {} control-flow regions but {} are required",
            function.function_id,
            function.control_flow_mappings.len(),
            contracts.len()
        )));
    }
    let compact_output = compact_signature(&function.pseudocode);
    let mut covered = BTreeSet::new();
    let mut fragments = BTreeSet::new();
    for mapping in &function.control_flow_mappings {
        let Some(contract) = contracts.get(&mapping.region_id) else {
            return Err(AiError::InvalidResponse(format!(
                "structure candidate for {} maps unknown control-flow region {}",
                function.function_id, mapping.region_id
            )));
        };
        if !covered.insert(mapping.region_id.as_str()) {
            return Err(AiError::InvalidResponse(format!(
                "structure candidate for {} maps control-flow region {} more than once",
                function.function_id, mapping.region_id
            )));
        }
        let fragment = mapping.output_fragment.trim();
        let compact_fragment = compact_signature(fragment);
        if fragment.chars().count() < 2
            || fragment.chars().count() > 256
            || fragment.contains(['\n', '\r'])
            || fragment.starts_with("//")
            || fragment.starts_with("/*")
            || compact_fragment.is_empty()
            || !compact_output.contains(&compact_fragment)
        {
            return Err(AiError::InvalidResponse(format!(
                "control-flow mapping {} must name a bounded one-line non-comment header present in the pseudocode",
                mapping.region_id
            )));
        }
        if !fragments.insert(compact_fragment.clone()) {
            return Err(AiError::InvalidResponse(format!(
                "structure candidate for {} reuses one source header for multiple control-flow regions",
                function.function_id
            )));
        }
        if fragments
            .iter()
            .filter(|candidate| **candidate != compact_fragment)
            .any(|candidate| {
                candidate.contains(&compact_fragment)
                    || compact_fragment.contains(candidate.as_str())
            })
        {
            return Err(AiError::InvalidResponse(format!(
                "structure candidate for {} uses overlapping source headers for multiple control-flow regions",
                function.function_id
            )));
        }
        let realizes_kind = match contract.kind {
            ControlFlowRegionKind::Branch => ["if", "match", "switch"].iter().any(|keyword| {
                contains_pseudocode_keyword(&fragment.to_ascii_lowercase(), keyword)
            }),
            ControlFlowRegionKind::Loop => ["while", "for", "loop"].iter().any(|keyword| {
                contains_pseudocode_keyword(&fragment.to_ascii_lowercase(), keyword)
            }),
        };
        if !realizes_kind {
            return Err(AiError::InvalidResponse(format!(
                "control-flow mapping {} does not visibly preserve {:?}",
                mapping.region_id, contract.kind
            )));
        }
        let expected_location = format!("block:{}", contract.header_block_id);
        if mapping.evidence.len() != 1
            || mapping.evidence[0].source != "ir"
            || mapping.evidence[0].location != expected_location
        {
            return Err(AiError::InvalidResponse(format!(
                "control-flow mapping {} must cite its deterministic header {}",
                mapping.region_id, expected_location
            )));
        }
    }
    let missing = contracts
        .keys()
        .filter(|region| !covered.contains(region.as_str()))
        .cloned()
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        return Err(AiError::InvalidResponse(format!(
            "structure candidate for {} omits control-flow regions: {}",
            function.function_id,
            missing.join(", ")
        )));
    }
    Ok(())
}

fn validate_pseudocode_humanization_candidate(
    request: &ModelRequest,
    function: &RecoveredFunction,
) -> Result<(), AiError> {
    let obligations = request
        .evidence_pack
        .pseudocode_obligations
        .iter()
        .map(|obligation| (obligation.id.as_str(), obligation))
        .collect::<BTreeMap<_, _>>();
    if function.source_mappings.len() != obligations.len() {
        return Err(AiError::InvalidResponse(format!(
            "humanization candidate for {} maps {} source obligations but {} are required",
            function.function_id,
            function.source_mappings.len(),
            obligations.len()
        )));
    }
    let compact_output = compact_signature(&function.pseudocode);
    let mut covered = BTreeSet::new();
    let mut fragments = BTreeSet::new();
    for mapping in &function.source_mappings {
        let Some(obligation) = obligations.get(mapping.obligation_id.as_str()) else {
            return Err(AiError::InvalidResponse(format!(
                "humanization candidate for {} maps unknown source obligation {}",
                function.function_id, mapping.obligation_id
            )));
        };
        if !covered.insert(mapping.obligation_id.as_str()) {
            return Err(AiError::InvalidResponse(format!(
                "humanization candidate for {} maps source obligation {} more than once",
                function.function_id, mapping.obligation_id
            )));
        }
        let fragment = mapping.output_fragment.trim();
        let compact_fragment = compact_signature(fragment);
        if fragment.chars().count() < 2
            || fragment.chars().count() > 1_024
            || fragment.starts_with("//")
            || fragment.starts_with("/*")
            || compact_fragment.is_empty()
            || !compact_output.contains(&compact_fragment)
        {
            return Err(AiError::InvalidResponse(format!(
                "humanization mapping {} must name a bounded non-comment code fragment present in the output",
                mapping.obligation_id
            )));
        }
        if !fragments.insert(compact_fragment.clone()) {
            return Err(AiError::InvalidResponse(format!(
                "humanization candidate for {} reuses one output fragment for multiple source obligations",
                function.function_id
            )));
        }
        if fragments
            .iter()
            .filter(|candidate| **candidate != compact_fragment)
            .any(|candidate| {
                candidate.contains(&compact_fragment)
                    || compact_fragment.contains(candidate.as_str())
            })
        {
            return Err(AiError::InvalidResponse(format!(
                "humanization candidate for {} uses overlapping output fragments for multiple source obligations",
                function.function_id
            )));
        }
        for construct in &obligation.constructs {
            if !fragment_realizes_construct(fragment, *construct) {
                return Err(AiError::InvalidResponse(format!(
                    "humanization mapping {} does not visibly preserve {:?}",
                    mapping.obligation_id, construct
                )));
            }
        }
        if let Some(declared_function) = &obligation.declared_function {
            if !fragment_declares_function(fragment, declared_function) {
                return Err(AiError::InvalidResponse(format!(
                    "humanization mapping {} renames or omits declared function {}; exact symbol identity is required",
                    mapping.obligation_id, declared_function
                )));
            }
        }
        for called_function in &obligation.called_functions {
            if !fragment_calls_function(fragment, called_function) {
                return Err(AiError::InvalidResponse(format!(
                    "humanization mapping {} redirects or omits call to known function {}; exact callee identity is required",
                    mapping.obligation_id, called_function
                )));
            }
        }
    }
    let missing = obligations
        .keys()
        .filter(|obligation| !covered.contains(**obligation))
        .copied()
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        return Err(AiError::InvalidResponse(format!(
            "humanization candidate for {} omits source obligations: {}",
            function.function_id,
            missing.join(", ")
        )));
    }
    Ok(())
}

fn fragment_declares_function(fragment: &str, function_name: &str) -> bool {
    fragment.lines().any(|line| {
        contains_pseudocode_keyword(&line.to_ascii_lowercase(), "fn")
            && pseudocode_function_name(line).as_deref() == Some(function_name)
    })
}

fn fragment_calls_function(fragment: &str, function_name: &str) -> bool {
    let declaration_occurrence = usize::from(fragment_declares_function(fragment, function_name));
    pseudocode_named_call_occurrences(fragment, function_name) > declaration_occurrence
}

fn fragment_realizes_construct(fragment: &str, construct: PseudocodeConstruct) -> bool {
    let lower = fragment.to_ascii_lowercase();
    match construct {
        PseudocodeConstruct::FunctionDeclaration => {
            contains_pseudocode_keyword(&lower, "fn") && fragment.contains('(')
        }
        PseudocodeConstruct::Branch => [
            "if", "match", "switch", "while", "for", "loop", "break", "continue",
        ]
        .iter()
        .any(|keyword| contains_pseudocode_keyword(&lower, keyword)),
        PseudocodeConstruct::Loop => ["while", "for", "loop"]
            .iter()
            .any(|keyword| contains_pseudocode_keyword(&lower, keyword)),
        PseudocodeConstruct::Call => {
            pseudocode_has_call(fragment, lower.trim_start().starts_with("fn "))
        }
        PseudocodeConstruct::MemoryWrite => {
            pseudocode_has_memory_write(fragment)
                || [".copy", ".fill", ".write", ".insert", ".push", "write_"]
                    .iter()
                    .any(|token| lower.contains(token))
        }
        PseudocodeConstruct::Return => contains_pseudocode_keyword(&lower, "return"),
        PseudocodeConstruct::Statement => true,
        PseudocodeConstruct::Unknown => {
            lower.contains("unknown_") || lower.contains("untranslated")
        }
    }
}

fn validate_signature_contract(
    pack: &EvidencePack,
    function_id: &str,
    source: &str,
    candidate_kind: &str,
) -> Result<(), AiError> {
    let contracts = pack
        .signature_contracts
        .iter()
        .filter(|contract| contract.signature.function_id == function_id)
        .collect::<Vec<_>>();
    if contracts.len() > 1 {
        return Err(AiError::InvalidResponse(format!(
            "{candidate_kind} for {function_id} has multiple accepted signature contracts"
        )));
    }
    let Some(contract) = contracts.first() else {
        return Ok(());
    };
    if contract.source_output_sha256.len() != 64
        || !contract
            .source_output_sha256
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err(AiError::InvalidResponse(format!(
            "{candidate_kind} for {function_id} has an invalid signature-contract source hash"
        )));
    }
    let expected = signature_contract_declaration(&contract.signature);
    let observed = candidate_function_declaration(source, function_id).ok_or_else(|| {
        AiError::InvalidResponse(format!(
            "{candidate_kind} for {function_id} does not contain a complete function declaration"
        ))
    })?;
    if compact_signature(&observed) != compact_signature(&expected) {
        return Err(AiError::InvalidResponse(format!(
            "{candidate_kind} for {function_id} changes the accepted signature contract; expected `{expected}`, observed `{observed}`"
        )));
    }
    Ok(())
}

fn signature_contract_declaration(signature: &RecoveredSignature) -> String {
    let parameters = signature
        .parameters
        .iter()
        .map(|parameter| format!("{}: {}", parameter.name, parameter.type_name))
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "fn {}({parameters}) -> {}",
        signature.function_id, signature.return_type
    )
}

fn candidate_function_declaration(source: &str, function_id: &str) -> Option<String> {
    let marker = format!("fn {function_id}");
    let start = source.find(&marker)?;
    let remainder = &source[start..];
    let body = remainder.find('{')?;
    let declaration = remainder[..body].trim();
    (!declaration.is_empty()).then(|| declaration.to_owned())
}

fn compact_signature(source: &str) -> String {
    source
        .chars()
        .filter(|character| !character.is_whitespace())
        .collect()
}

fn validate_structured_output(task: ModelTask, output: &serde_json::Value) -> Result<(), AiError> {
    let parsed = match task {
        ModelTask::StructureRecovery | ModelTask::PseudocodeHumanization => {
            serde_json::from_value::<StructureRecoveryOutput>(output.clone())
                .map(|_| "structure_recovery")
        }
        ModelTask::SemanticLabeling => {
            serde_json::from_value::<SemanticLabelingOutput>(output.clone())
                .map(|_| "semantic_labeling")
        }
        ModelTask::SignatureRecovery => {
            serde_json::from_value::<SignatureRecoveryOutput>(output.clone())
                .map(|_| "signature_recovery")
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
        ModelTask::StructureRecovery | ModelTask::PseudocodeHumanization => {
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
                for effect in function.effects {
                    if effect.evidence.is_empty() {
                        return Err(AiError::InvalidResponse(format!(
                            "recovered effect for {} has no evidence",
                            function.function_id
                        )));
                    }
                    references.extend(effect.evidence);
                }
                for mapping in function.control_flow_mappings {
                    if mapping.evidence.is_empty() {
                        return Err(AiError::InvalidResponse(format!(
                            "control-flow mapping {} for {} has no evidence",
                            mapping.region_id, function.function_id
                        )));
                    }
                    references.extend(mapping.evidence);
                }
                for mapping in function.memory_field_mappings {
                    if mapping.evidence.is_empty() {
                        return Err(AiError::InvalidResponse(format!(
                            "memory field mapping {} for {} has no evidence",
                            mapping.contract_id, function.function_id
                        )));
                    }
                    references.extend(mapping.evidence);
                }
                for mapping in function.direct_call_mappings {
                    if mapping.evidence.is_empty() {
                        return Err(AiError::InvalidResponse(format!(
                            "direct-call mapping {} for {} has no evidence",
                            mapping.contract_id, function.function_id
                        )));
                    }
                    references.extend(mapping.evidence);
                }
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
        ModelTask::SignatureRecovery => {
            let parsed: SignatureRecoveryOutput = serde_json::from_value(output.clone())
                .map_err(|error| AiError::InvalidResponse(error.to_string()))?;
            for signature in parsed.signatures {
                if signature.evidence.is_empty() {
                    return Err(AiError::InvalidResponse(format!(
                        "recovered signature for {} has no evidence",
                        signature.function_id
                    )));
                }
                if signature
                    .confidence
                    .is_some_and(|value| !(0.0..=1.0).contains(&value))
                {
                    return Err(AiError::InvalidResponse(format!(
                        "recovered signature for {} has confidence outside [0, 1]",
                        signature.function_id
                    )));
                }
                references.extend(signature.evidence);
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
                for mapping in patch.memory_field_mappings {
                    if mapping.evidence.is_empty() {
                        return Err(AiError::InvalidResponse(format!(
                            "memory field mapping {} for refinement {} has no evidence",
                            mapping.contract_id, patch.function_id
                        )));
                    }
                    references.extend(mapping.evidence);
                }
                for mapping in patch.direct_call_mappings {
                    if mapping.evidence.is_empty() {
                        return Err(AiError::InvalidResponse(format!(
                            "direct-call mapping {} for refinement {} has no evidence",
                            mapping.contract_id, patch.function_id
                        )));
                    }
                    references.extend(mapping.evidence);
                }
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

fn validate_signature_identifier(value: &str, field: &str) -> Result<(), AiError> {
    let valid = !value.is_empty()
        && value.len() <= 128
        && value.bytes().enumerate().all(|(index, byte)| {
            byte == b'_' || byte.is_ascii_alphanumeric() && (index > 0 || !byte.is_ascii_digit())
        });
    if !valid {
        return Err(AiError::InvalidResponse(format!(
            "signature {field} is not a bounded Rust-like identifier"
        )));
    }
    Ok(())
}

fn validate_signature_type(value: &str, field: &str) -> Result<(), AiError> {
    if value.is_empty()
        || value.len() > 128
        || !value.bytes().all(|byte| {
            byte.is_ascii_alphanumeric()
                || matches!(
                    byte,
                    b'_' | b':'
                        | b'<'
                        | b'>'
                        | b'['
                        | b']'
                        | b'('
                        | b')'
                        | b','
                        | b'&'
                        | b'*'
                        | b'\''
                        | b' '
                )
        })
    {
        return Err(AiError::InvalidResponse(format!(
            "signature {field} contains unsafe or unsupported syntax"
        )));
    }
    Ok(())
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
    if evidence.detail.is_some() {
        return Err(AiError::InvalidResponse(
            "model evidence detail must be null; SOS resolves immutable evidence text by location"
                .into(),
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

    fn test_humanization_output(
        pack: &EvidencePack,
        function_id: &str,
        evidence: serde_json::Value,
    ) -> serde_json::Value {
        let mut fragments = Vec::new();
        let mut mappings = Vec::new();
        let mut open_declaration = false;
        for obligation in &pack.pseudocode_obligations {
            let suffix = obligation.id.replace('-', "_");
            let has = |construct| obligation.constructs.contains(&construct);
            let mut statements = Vec::new();
            if has(PseudocodeConstruct::Loop) {
                statements.push(format!(
                    "while unknown_condition_{suffix}() {{ unknown_loop_{suffix}(); break; }}"
                ));
            } else if has(PseudocodeConstruct::Branch) {
                statements.push(format!(
                    "if unknown_condition_{suffix}() {{ unknown_branch_{suffix}(); }}"
                ));
            }
            for called_function in &obligation.called_functions {
                statements.push(format!("{called_function}();"));
            }
            if has(PseudocodeConstruct::Call)
                && obligation.called_functions.is_empty()
                && !statements.iter().any(|statement| statement.contains('('))
            {
                statements.push(format!("unknown_call_{suffix}();"));
            }
            if has(PseudocodeConstruct::MemoryWrite) {
                statements.push(format!(
                    "write_unknown_target_{suffix}(unknown_value_{suffix});"
                ));
            }
            if has(PseudocodeConstruct::Return) {
                statements.push(format!("return unknown_result_{suffix}();"));
            }
            if has(PseudocodeConstruct::Unknown) {
                statements.push(format!("unknown_source_{suffix}();"));
            }
            if statements.is_empty() && !has(PseudocodeConstruct::FunctionDeclaration) {
                statements.push(format!("let recovered_{suffix} = ();"));
            }
            let fragment = if has(PseudocodeConstruct::FunctionDeclaration) {
                let declaration_name = obligation
                    .declared_function
                    .clone()
                    .unwrap_or_else(|| format!("recovered_{function_id}"));
                if statements.is_empty() {
                    open_declaration = true;
                    format!("fn {declaration_name}() {{")
                } else {
                    format!("fn {declaration_name}() {{ {} }}", statements.join(" "))
                }
            } else {
                statements.join(" ")
            };
            fragments.push(fragment.clone());
            mappings.push(serde_json::json!({
                "obligation_id": obligation.id,
                "output_fragment": fragment,
            }));
        }
        if open_declaration {
            fragments.push("}".into());
        }
        if !pack.pseudocode_obligations.iter().any(|obligation| {
            obligation
                .constructs
                .contains(&PseudocodeConstruct::FunctionDeclaration)
        }) {
            fragments.insert(0, format!("fn recovered_{function_id}() {{"));
            fragments.push("}".into());
        }
        serde_json::json!({
            "functions": [{
                "function_id": function_id,
                "pseudocode": fragments.join("\n"),
                "source_mappings": mappings,
                "evidence": [evidence]
            }]
        })
    }

    fn test_structure_output(
        pack: &EvidencePack,
        function_id: &str,
        declaration: &str,
        evidence: serde_json::Value,
    ) -> serde_json::Value {
        let effects = required_structure_effects(pack, function_id)
            .into_iter()
            .map(|effect| {
                serde_json::json!({
                    "kind": effect.kind,
                    "input_registers": effect.input_registers,
                    "target_input_registers": effect.target_input_registers,
                    "value_input_registers": effect.value_input_registers,
                    "evidence": [{
                        "source": "ir",
                        "location": effect.location,
                        "detail": null
                    }]
                })
            })
            .collect::<Vec<_>>();
        let regions = required_control_flow_regions(pack, function_id);
        let mut body = Vec::new();
        let mut mappings = Vec::new();
        for region in &regions {
            let suffix = region.id.replace('-', "_");
            let header = match region.kind {
                ControlFlowRegionKind::Branch => {
                    format!("if unknown_{suffix}() {{")
                }
                ControlFlowRegionKind::Loop => {
                    format!("while unknown_{suffix}() {{")
                }
            };
            body.push(header.clone());
            body.push("unknown_region_body();".into());
            body.push("}".into());
            mappings.push(serde_json::json!({
                "region_id": region.id,
                "output_fragment": header,
                "evidence": [{
                    "source": "ir",
                    "location": format!("block:{}", region.header_block_id),
                    "detail": null
                }]
            }));
        }
        let memory_fields = required_memory_field_contracts(pack, function_id);
        let mut memory_field_mappings = Vec::new();
        for contract in &memory_fields {
            let variable = sanitize_contract_component(&contract.id);
            let fragment = format!(
                "let recovered_{variable}: {} = {};",
                contract.type_name, contract.field_path
            );
            body.push(fragment.clone());
            memory_field_mappings.push(serde_json::json!({
                "contract_id": contract.id,
                "field_path": contract.field_path,
                "type_name": contract.type_name,
                "offset": contract.offset,
                "layout_type": contract.layout_type,
                "layout_field": contract.layout_field,
                "output_fragment": fragment,
                "evidence": contract.evidence.first().into_iter().collect::<Vec<_>>(),
            }));
        }
        let direct_calls = required_direct_call_contracts(pack, function_id);
        let mut direct_call_mappings = Vec::new();
        for contract in &direct_calls {
            let fragment = format!("{}();", contract.target_function_id);
            body.push(fragment.clone());
            direct_call_mappings.push(serde_json::json!({
                "contract_id": contract.id,
                "target_function_id": contract.target_function_id,
                "output_fragment": fragment,
                "evidence": contract.evidence.first().into_iter().collect::<Vec<_>>(),
            }));
        }
        let required = required_structure_effects(pack, function_id);
        if required
            .iter()
            .any(|effect| effect.kind == RecoveredEffectKind::MemoryWrite)
        {
            body.push("unknown_memory_write();".into());
        }
        if required
            .iter()
            .any(|effect| effect.kind == RecoveredEffectKind::Call)
        {
            body.push("unknown_call();".into());
        }
        if required
            .iter()
            .any(|effect| effect.kind == RecoveredEffectKind::Return)
        {
            body.push("return unknown_result();".into());
        }
        if body.is_empty() {
            body.push("unknown_semantics();".into());
        }
        serde_json::json!({
            "functions": [{
                "function_id": function_id,
                "pseudocode": format!("{declaration} {{\n{}\n}}", body.join("\n")),
                "control_flow_mappings": mappings,
                "memory_field_mappings": memory_field_mappings,
                "direct_call_mappings": direct_call_mappings,
                "effects": effects,
                "evidence": [evidence]
            }]
        })
    }

    struct EchoProvider;

    impl ModelProvider for EchoProvider {
        fn generate_structured(&self, request: ModelRequest) -> Result<ModelResponse, AiError> {
            let function_id = request
                .evidence_pack
                .ir
                .functions
                .first()
                .map(|function| function.id.clone())
                .unwrap_or_else(|| "fn_0".into());
            let evidence_location = request
                .evidence_pack
                .ir
                .operations
                .first()
                .map(|operation| format!("operation:{}", operation.id))
                .unwrap_or_else(|| format!("function:{function_id}"));
            let function_evidence = serde_json::json!({
                "source": "ir",
                "location": evidence_location,
                "detail": null
            });
            let structured_output = match request.task {
                ModelTask::StructureRecovery => {
                    let declaration = request
                        .evidence_pack
                        .signature_contracts
                        .iter()
                        .find(|contract| contract.signature.function_id == function_id)
                        .map(|contract| signature_contract_declaration(&contract.signature))
                        .unwrap_or_else(|| format!("fn {function_id}()"));
                    test_structure_output(
                        &request.evidence_pack,
                        &function_id,
                        &declaration,
                        function_evidence,
                    )
                }
                ModelTask::PseudocodeHumanization => test_humanization_output(
                    &request.evidence_pack,
                    &function_id,
                    function_evidence,
                ),
                ModelTask::SignatureRecovery => serde_json::json!({
                    "signatures": [{
                        "function_id": function_id,
                        "function_name": "recovered_function",
                        "return_type": "unknown",
                        "parameters": [],
                        "confidence": 0.1,
                        "evidence": [function_evidence]
                    }]
                }),
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

    struct ContractRecordingProvider {
        contract_counts: Arc<Mutex<Vec<usize>>>,
    }

    impl ModelProvider for ContractRecordingProvider {
        fn generate_structured(&self, request: ModelRequest) -> Result<ModelResponse, AiError> {
            self.contract_counts
                .lock()
                .unwrap()
                .push(request.evidence_pack.signature_contracts.len());
            EchoProvider.generate_structured(request)
        }
    }

    struct ModelRecordingProvider {
        requests: Arc<Mutex<Vec<(ModelTask, String, usize)>>>,
    }

    impl ModelProvider for ModelRecordingProvider {
        fn generate_structured(&self, request: ModelRequest) -> Result<ModelResponse, AiError> {
            self.requests.lock().unwrap().push((
                request.task,
                request.model.clone(),
                request.max_output_tokens,
            ));
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
                        "memory_field_mappings": [],
                        "direct_call_mappings": [],
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

    struct WrongModelProvider;

    impl ModelProvider for WrongModelProvider {
        fn generate_structured(&self, request: ModelRequest) -> Result<ModelResponse, AiError> {
            Ok(ModelResponse {
                task: request.task,
                model: "different-model".into(),
                prompt_version: request.prompt_version,
                structured_output: serde_json::json!({
                    "summary": "bounded explanation",
                    "observations": []
                }),
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
            max_output_tokens: default_stage_output_tokens(ModelTask::Explanation),
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
                memory_field_mappings: Vec::new(),
                direct_call_mappings: Vec::new(),
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
        assert!(valid_result
            .mir_summary
            .as_ref()
            .is_some_and(|summary| summary.functions.iter().any(|item| item.name == "fn_0")));

        let invalid = RefinementOutput {
            schema_version: REFINEMENT_SCHEMA_VERSION,
            patches: vec![RefinementPatch {
                function_id: "fn_0".into(),
                replacement: "fn broken( {".into(),
                memory_field_mappings: Vec::new(),
                direct_call_mappings: Vec::new(),
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

    fn refinement_with_layout(replacement: &str, type_name: &str) -> RefinementOutput {
        RefinementOutput {
            schema_version: REFINEMENT_SCHEMA_VERSION,
            patches: vec![RefinementPatch {
                function_id: "fn_0".into(),
                replacement: replacement.into(),
                memory_field_mappings: vec![RecoveredMemoryFieldMapping {
                    contract_id: "layout_fn_0_r1_pos_0x8".into(),
                    field_path: "unknown_memory_r1.field_pos_0x8".into(),
                    type_name: type_name.into(),
                    offset: 8,
                    layout_type: "SosLayoutFn0R1".into(),
                    layout_field: "field_pos_0x8".into(),
                    output_fragment: "unknown_memory_r1.field_pos_0x8".into(),
                    evidence: vec![],
                }],
                direct_call_mappings: Vec::new(),
                evidence: vec![],
            }],
            observations: Vec::new(),
            validation: RefinementValidation {
                status: "unverified".into(),
                notes: vec![],
            },
        }
    }

    #[test]
    fn refinement_compiler_proves_repr_c_field_offset_and_type() {
        let valid = refinement_with_layout(
            "#[repr(C)] struct SosLayoutFn0R1 { _pad_0: [u8; 8], field_pos_0x8: u64 } fn fn_0() {}",
            "u64",
        );
        let valid_result =
            compile_refinement_output(&valid, Some(Path::new("rustc")), Duration::from_secs(10));
        assert_eq!(valid_result.status, CompilationStatus::Passed);

        let wrong_offset = refinement_with_layout(
            "#[repr(C)] struct SosLayoutFn0R1 { field_pos_0x8: u64 } fn fn_0() {}",
            "u64",
        );
        let wrong_offset_result = compile_refinement_output(
            &wrong_offset,
            Some(Path::new("rustc")),
            Duration::from_secs(10),
        );
        assert_eq!(wrong_offset_result.status, CompilationStatus::Failed);

        let wrong_type = refinement_with_layout(
            "#[repr(C)] struct SosLayoutFn0R1 { _pad_0: [u8; 8], field_pos_0x8: u32 } fn fn_0() {}",
            "u64",
        );
        let wrong_type_result = compile_refinement_output(
            &wrong_type,
            Some(Path::new("rustc")),
            Duration::from_secs(10),
        );
        assert_eq!(wrong_type_result.status, CompilationStatus::Failed);
    }

    #[test]
    fn refinement_compiler_rejects_malformed_layout_mapping() {
        let mut refinement = refinement_with_layout(
            "#[repr(C)] struct SosLayoutFn0R1 { _pad_0: [u8; 8], field_pos_0x8: u64 } fn fn_0() {}",
            "u64",
        );
        refinement.patches[0].memory_field_mappings[0].layout_field = "bad.field".into();
        let result = compile_refinement_output(
            &refinement,
            Some(Path::new("rustc")),
            Duration::from_secs(10),
        );
        assert_eq!(result.status, CompilationStatus::Failed);
        assert!(result.diagnostics[0].contains("invalid Rust layout identifier"));
    }

    #[test]
    fn refinement_compiler_reports_negative_layout_offsets_as_unproved() {
        let mut refinement = refinement_with_layout("fn fn_0() {}", "u64");
        let mapping = &mut refinement.patches[0].memory_field_mappings[0];
        mapping.offset = -8;
        mapping.layout_field = "field_neg_0x8".into();
        let result = compile_refinement_output(
            &refinement,
            Some(Path::new("rustc")),
            Duration::from_secs(10),
        );
        assert_eq!(result.status, CompilationStatus::Passed);
        assert!(result
            .limitations
            .iter()
            .any(|limitation| limitation.contains("cannot be represented by Rust offset_of!")));
    }

    #[test]
    fn rustc_mir_summary_recovers_cfg_calls_writes_and_loops() {
        let mir = r#"
fn fn_80(_1: &mut [u8], _2: u8, _3: usize) -> () {
    let mut _0: ();

    bb0: {
        goto -> bb1;
    }

    bb1: {
        switchInt(copy _3) -> [0: bb3, otherwise: bb2];
    }

    bb2: {
        (*_1)[_3] = copy _2;
        _3 = helper(copy _3) -> [return: bb1, unwind continue];
    }

    bb3: {
        return;
    }
}
"#;
        let summary = summarize_rustc_mir(mir);
        let function = &summary.functions[0];
        assert_eq!(function.name, "fn_80");
        assert_eq!(function.parameter_count, 3);
        assert_eq!(function.basic_block_count, 4);
        assert_eq!(function.calls, 1);
        assert_eq!(
            function.direct_calls,
            vec![MirDirectCallSummary {
                block: 2,
                statement_index: 1,
                target: "helper".into(),
            }]
        );
        assert_eq!(function.memory_writes, 1);
        assert_eq!(function.returns, 1);
        assert!(function.branch_terminators >= 2);
        assert!(function.has_loop);
        assert_eq!(function.control_flow_regions.len(), 1);
        assert_eq!(
            function.control_flow_regions[0].kind,
            ControlFlowRegionKind::Loop
        );
        assert_eq!(function.control_flow_regions[0].header_block, 1);
        assert_eq!(function.control_flow_regions[0].member_blocks, vec![1, 2]);
        assert_eq!(function.control_flow_regions[0].exit_blocks, vec![3]);
        assert_eq!(
            function.control_flow_regions[0].cycle_edges,
            vec![MirEdge {
                source: 2,
                destination: 1
            }]
        );
        assert!(function.effect_dependencies.iter().any(|dependency| {
            dependency.kind == RecoveredEffectKind::MemoryWrite
                && dependency.input_parameters == vec![1, 2, 3]
                && dependency.target_input_parameters == vec![1, 3]
                && dependency.value_input_parameters == vec![2]
        }));
        assert_eq!(function.memory_write_dependencies.len(), 1);
        assert_eq!(
            function.memory_write_dependencies[0].target_input_parameters,
            vec![1, 3]
        );
        assert_eq!(
            function.memory_write_dependencies[0].value_input_parameters,
            vec![2]
        );

        let mut pack = empty_pack();
        pack.task = ModelTask::Refinement;
        pack.ir.functions[0].id = "fn_80".into();
        pack.ir.blocks[0].function_id = "fn_80".into();
        pack.ir.operations[0].kind = "store".into();
        pack.ir.operations[0].assembly = "stxb [r1], r2".into();
        pack.ir.operations[0].memory_effect = Some("store:1:1+0".into());
        pack.ir.operations.push(sos_analysis::IrOperation {
            id: "op_1".into(),
            block_id: "bb_0".into(),
            pc: 1,
            kind: "control_flow".into(),
            assembly: "jne r3, 0, lbb_0".into(),
            dst: 3,
            src: 0,
            offset: -1,
            immediate: 0,
            memory_effect: None,
            call_target_pc: None,
            call_target_pcs: Vec::new(),
        });
        pack.ir.operations.push(sos_analysis::IrOperation {
            id: "op_2".into(),
            block_id: "bb_0".into(),
            pc: 2,
            kind: "raw".into(),
            assembly: "exit".into(),
            dst: 0,
            src: 0,
            offset: 0,
            immediate: 0,
            memory_effect: None,
            call_target_pc: None,
            call_target_pcs: Vec::new(),
        });
        pack.ir.sccs = vec![sos_analysis::SccFact {
            id: 0,
            block_starts: vec![0],
            recursive: true,
        }];
        let comparison = compare_mir_effects(&pack, Some(&summary));
        assert_eq!(comparison.status, MirEffectComparisonStatus::Covered);
        assert!(comparison.functions[0].missing.is_empty());
    }

    #[test]
    fn resolved_internal_calls_require_exact_source_and_mir_targets() {
        let mut analysis = empty_pack().analysis;
        analysis.instructions = vec![
            InstructionFact {
                pc: 0,
                opcode: 0,
                dst: 0,
                src: 0,
                offset: 0,
                immediate: 8,
                assembly: "call 8".into(),
            },
            InstructionFact {
                pc: 8,
                opcode: 0,
                dst: 0,
                src: 0,
                offset: 0,
                immediate: 0,
                assembly: "exit".into(),
            },
        ];
        analysis.functions = vec![
            FunctionFact {
                pc: 0,
                symbol_hash: 1,
                name: "caller".into(),
            },
            FunctionFact {
                pc: 8,
                symbol_hash: 8,
                name: "callee".into(),
            },
        ];
        analysis.basic_blocks = vec![
            BasicBlockFact {
                start_pc: 0,
                end_instruction_index: 1,
                destinations: Vec::new(),
                label: "caller".into(),
            },
            BasicBlockFact {
                start_pc: 8,
                end_instruction_index: 2,
                destinations: Vec::new(),
                label: "callee".into(),
            },
        ];
        analysis.calls = vec![sos_analysis::CallFact {
            call_pc: 0,
            caller_pc: Some(0),
            kind: "direct_internal_call".into(),
            target_pc: Some(8),
            target_pcs: vec![8],
            target_symbol_hash: Some(8),
            status: "resolved".into(),
        }];
        let semantics = sos_solana::analyze(&analysis);
        let mut pack =
            EvidencePack::from_analysis(ModelTask::StructureRecovery, None, analysis, semantics);
        let contracts = required_direct_call_contracts(&pack, "fn_0");
        assert_eq!(contracts.len(), 1);
        assert_eq!(contracts[0].target_function_id, "fn_8");
        assert_eq!(contracts[0].evidence[0].location, "operation:op_0");
        let hosted = crate::groq::hosted_model_evidence(&pack);
        assert_eq!(
            hosted["program"]["direct_call_contracts"]
                .as_array()
                .map(Vec::len),
            Some(1)
        );
        let mut indirect_analysis = pack.analysis.clone();
        indirect_analysis.calls[0].kind = "indirect_internal_call".into();
        indirect_analysis.instructions[0].assembly = "callx r9".into();
        let indirect_semantics = sos_solana::analyze(&indirect_analysis);
        let indirect_pack = EvidencePack::from_analysis(
            ModelTask::StructureRecovery,
            None,
            indirect_analysis,
            indirect_semantics,
        );
        assert_eq!(
            required_direct_call_contracts(&indirect_pack, "fn_0"),
            contracts
        );
        assert_eq!(
            indirect_pack.ir.operations[0].kind,
            "indirect_internal_call"
        );
        let mut candidate_analysis = pack.analysis.clone();
        candidate_analysis.calls[0].kind = "indirect_internal_call_set".into();
        candidate_analysis.calls[0].target_pc = None;
        candidate_analysis.calls[0].target_pcs = vec![8, 10];
        candidate_analysis.calls[0].target_symbol_hash = None;
        candidate_analysis.calls[0].status = "candidates".into();
        candidate_analysis.instructions[0].assembly = "callx r9".into();
        let candidate_semantics = sos_solana::analyze(&candidate_analysis);
        let candidate_pack = EvidencePack::from_analysis(
            ModelTask::StructureRecovery,
            None,
            candidate_analysis,
            candidate_semantics,
        );
        assert!(required_direct_call_contracts(&candidate_pack, "fn_0").is_empty());
        let candidate_hosted = crate::groq::hosted_model_evidence(&candidate_pack);
        assert_eq!(
            candidate_hosted["program"]["operations"][0]["call_candidates"],
            serde_json::json!([8, 10])
        );

        let output = test_structure_output(
            &pack,
            "fn_0",
            "fn fn_0()",
            serde_json::json!({
                "source": "ir",
                "location": "function:fn_0",
                "detail": null
            }),
        );
        let request = ModelRequest {
            task: ModelTask::StructureRecovery,
            prompt_version: "v1".into(),
            model: "test".into(),
            evidence_pack: pack.clone(),
            max_output_bytes: 100_000,
            max_output_tokens: 2_400,
        };
        validate_structured_consistency(&request, &output).unwrap();
        let mut renamed = output;
        renamed["functions"][0]["direct_call_mappings"][0]["target_function_id"] =
            serde_json::json!("fn_10");
        assert!(validate_structured_consistency(&request, &renamed)
            .unwrap_err()
            .to_string()
            .contains("deterministic target fn_8"));

        let mut suppressed = pack.clone();
        suppressed.direct_call_contracts.clear();
        let restored = ModelGateway::new(EchoProvider, 1024 * 1024, 1024 * 1024)
            .submit(ModelTask::StructureRecovery, "test", "v1", suppressed)
            .unwrap();
        assert_eq!(
            restored.structured_output["functions"][0]["direct_call_mappings"]
                .as_array()
                .map(Vec::len),
            Some(1)
        );

        let refinement = RefinementOutput {
            schema_version: REFINEMENT_SCHEMA_VERSION,
            patches: vec![RefinementPatch {
                function_id: "fn_0".into(),
                replacement: "#[inline(never)] fn fn_8(value: u64) -> u64 { value + 1 } fn fn_0() -> u64 { fn_8(1) }".into(),
                memory_field_mappings: Vec::new(),
                direct_call_mappings: vec![RecoveredDirectCallMapping {
                    contract_id: contracts[0].id.clone(),
                    target_function_id: "fn_8".into(),
                    output_fragment: "fn_8(1)".into(),
                    evidence: contracts[0].evidence.clone(),
                }],
                evidence: Vec::new(),
            }],
            observations: Vec::new(),
            validation: RefinementValidation {
                status: "unverified".into(),
                notes: Vec::new(),
            },
        };
        let compilation = compile_refinement_output(
            &refinement,
            Some(Path::new("rustc")),
            Duration::from_secs(10),
        );
        assert_eq!(compilation.status, CompilationStatus::Passed);
        pack.task = ModelTask::Refinement;
        let comparison = compare_mir_effects(&pack, compilation.mir_summary.as_ref());
        assert!(comparison.functions[0].direct_call_mismatches.is_empty());

        let mut wrong = refinement;
        wrong.patches[0].replacement = "#[inline(never)] fn fn_8(value: u64) -> u64 { value + 1 } #[inline(never)] fn fn_10(value: u64) -> u64 { value + 2 } fn mapped_only() { let _ = fn_8(1); } fn fn_0() -> u64 { fn_10(1) }".into();
        validate_direct_call_mappings(
            &pack,
            "fn_0",
            &wrong.patches[0].replacement,
            &wrong.patches[0].direct_call_mappings,
            "refinement patch",
        )
        .unwrap();
        let wrong_compilation =
            compile_refinement_output(&wrong, Some(Path::new("rustc")), Duration::from_secs(10));
        assert_eq!(wrong_compilation.status, CompilationStatus::Passed);
        let wrong_comparison = compare_mir_effects(&pack, wrong_compilation.mir_summary.as_ref());
        assert_eq!(wrong_comparison.status, MirEffectComparisonStatus::Mismatch);
        assert_eq!(
            wrong_comparison.functions[0].direct_call_mismatches[0].required_target_function_id,
            "fn_8"
        );
        assert!(wrong_comparison.functions[0].direct_call_mismatches[0]
            .observed_targets
            .contains(&"fn_10".into()));
    }

    #[test]
    fn mir_region_comparison_preserves_nested_topology_and_allows_extra_regions() {
        let nested_mir = r#"
fn fn_80(_1: bool, _2: bool, _3: bool) -> () {
    let mut _0: ();

    bb0: {
        switchInt(copy _1) -> [0: bb7, otherwise: bb1];
    }

    bb1: {
        switchInt(copy _2) -> [0: bb8, otherwise: bb2];
    }

    bb2: {
        switchInt(copy _3) -> [0: bb4, otherwise: bb3];
    }

    bb3: {
        goto -> bb2;
    }

    bb4: {
        goto -> bb9;
    }

    bb7: {
        goto -> bb9;
    }

    bb8: {
        goto -> bb2;
    }

    bb9: {
        return;
    }
}
"#;
        let summary = summarize_rustc_mir(nested_mir);
        let function = &summary.functions[0];
        assert_eq!(
            function
                .control_flow_regions
                .iter()
                .filter(|region| region.kind == ControlFlowRegionKind::Branch)
                .count(),
            2
        );
        assert_eq!(
            function
                .control_flow_regions
                .iter()
                .filter(|region| region.kind == ControlFlowRegionKind::Loop)
                .count(),
            1
        );

        let mut pack = empty_pack();
        pack.task = ModelTask::Refinement;
        pack.ir.functions[0].id = "fn_80".into();
        pack.ir.operations.clear();
        pack.control_flow_regions = vec![
            ControlFlowRegionContract {
                id: "region_branch_0".into(),
                function_id: "fn_80".into(),
                kind: ControlFlowRegionKind::Branch,
                header_block_id: "bb_0".into(),
                member_blocks: ["bb_0", "bb_1", "bb_2", "bb_3", "bb_4", "bb_7"]
                    .into_iter()
                    .map(str::to_owned)
                    .collect(),
                entry_blocks: vec!["bb_1".into(), "bb_7".into()],
                exit_blocks: vec!["bb_9".into()],
                join_block_id: Some("bb_9".into()),
                arms: vec![
                    ControlFlowArmContract {
                        entry_block_id: "bb_1".into(),
                        member_blocks: vec!["bb_1".into(), "bb_2".into(), "bb_3".into()],
                    },
                    ControlFlowArmContract {
                        entry_block_id: "bb_7".into(),
                        member_blocks: vec!["bb_7".into()],
                    },
                ],
                cycle_edges: Vec::new(),
            },
            ControlFlowRegionContract {
                id: "region_loop_2".into(),
                function_id: "fn_80".into(),
                kind: ControlFlowRegionKind::Loop,
                header_block_id: "bb_2".into(),
                member_blocks: vec!["bb_2".into(), "bb_3".into()],
                entry_blocks: vec!["bb_2".into()],
                exit_blocks: vec!["bb_4".into()],
                join_block_id: None,
                arms: Vec::new(),
                cycle_edges: vec![ControlFlowEdgeContract {
                    source_block_id: "bb_3".into(),
                    destination_block_id: "bb_2".into(),
                }],
            },
        ];

        let comparison = compare_mir_effects(&pack, Some(&summary));
        assert_eq!(comparison.status, MirEffectComparisonStatus::Covered);
        assert!(comparison.functions[0].region_mismatches.is_empty());

        let mut missing_loop_exit = summary.clone();
        missing_loop_exit.functions[0]
            .control_flow_regions
            .iter_mut()
            .find(|region| region.kind == ControlFlowRegionKind::Loop)
            .unwrap()
            .exit_blocks
            .clear();
        let comparison = compare_mir_effects(&pack, Some(&missing_loop_exit));
        assert_eq!(comparison.status, MirEffectComparisonStatus::Mismatch);
        assert!(comparison.functions[0]
            .region_mismatches
            .iter()
            .flat_map(|mismatch| &mismatch.reasons)
            .any(|reason| reason.contains("required exit=true")));

        let flattened_mir = r#"
fn fn_80(_1: bool, _2: bool) -> () {
    let mut _0: ();

    bb0: {
        switchInt(copy _1) -> [0: bb2, otherwise: bb1];
    }

    bb1: {
        goto -> bb3;
    }

    bb2: {
        goto -> bb3;
    }

    bb3: {
        goto -> bb4;
    }

    bb4: {
        switchInt(copy _2) -> [0: bb6, otherwise: bb5];
    }

    bb5: {
        goto -> bb4;
    }

    bb6: {
        return;
    }
}
"#;
        let flattened = summarize_rustc_mir(flattened_mir);
        let comparison = compare_mir_effects(&pack, Some(&flattened));
        assert_eq!(comparison.status, MirEffectComparisonStatus::Mismatch);
        assert!(
            comparison.functions[0]
                .region_mismatches
                .iter()
                .flat_map(|mismatch| &mismatch.reasons)
                .any(|reason| { reason.contains("containment") || reason.contains("nesting") }),
            "{:?}",
            comparison.functions[0].region_mismatches
        );
    }

    #[test]
    fn mir_effect_comparison_rejects_ledger_only_memory_writes() {
        let mut pack = empty_pack();
        pack.task = ModelTask::Refinement;
        pack.ir.operations[0].kind = "store".into();
        pack.ir.operations[0].assembly = "stxdw [r1], r2".into();
        pack.ir.operations[0].memory_effect = Some("store:8:1+0".into());
        let summary = MirProgramSummary {
            schema_version: MIR_SUMMARY_SCHEMA_VERSION,
            functions: vec![MirFunctionSummary {
                name: "fn_0".into(),
                parameter_count: 0,
                basic_block_count: 1,
                edges: Vec::new(),
                branch_terminators: 0,
                calls: 0,
                memory_writes: 0,
                returns: 1,
                has_loop: false,
                effect_dependencies: vec![MirEffectDependency {
                    kind: RecoveredEffectKind::Return,
                    input_parameters: Vec::new(),
                    target_input_parameters: Vec::new(),
                    value_input_parameters: Vec::new(),
                }],
                memory_write_dependencies: Vec::new(),
                direct_calls: Vec::new(),
                control_flow_regions: Vec::new(),
            }],
        };
        let comparison = compare_mir_effects(&pack, Some(&summary));
        assert_eq!(comparison.status, MirEffectComparisonStatus::Mismatch);
        assert_eq!(
            comparison.functions[0].missing,
            vec![RecoveredEffectKind::MemoryWrite]
        );
    }

    #[test]
    fn mir_effect_comparison_rejects_swapped_memory_dependency_roles() {
        let mut analysis = empty_pack().analysis;
        analysis.instructions[0] = InstructionFact {
            pc: 0,
            opcode: 0x7b,
            dst: 1,
            src: 2,
            offset: 0,
            immediate: 0,
            assembly: "stxdw [r1], r2".into(),
        };
        analysis.memory_accesses = vec![sos_analysis::MemoryAccessFact {
            pc: 0,
            kind: "store".into(),
            width: 8,
            base_register: 1,
            value_register: Some(2),
            offset: 0,
            immediate: None,
            assembly: "stxdw [r1], r2".into(),
        }];
        let semantics = sos_solana::analyze(&analysis);
        let pack = EvidencePack::from_analysis(ModelTask::Refinement, None, analysis, semantics);
        let summary = MirProgramSummary {
            schema_version: MIR_SUMMARY_SCHEMA_VERSION,
            functions: vec![MirFunctionSummary {
                name: "fn_0".into(),
                parameter_count: 2,
                basic_block_count: 1,
                edges: Vec::new(),
                branch_terminators: 0,
                calls: 0,
                memory_writes: 1,
                returns: 0,
                has_loop: false,
                effect_dependencies: vec![MirEffectDependency {
                    kind: RecoveredEffectKind::MemoryWrite,
                    input_parameters: vec![1, 2],
                    target_input_parameters: vec![2],
                    value_input_parameters: vec![1],
                }],
                memory_write_dependencies: vec![MirMemoryWriteDependency {
                    block: 0,
                    statement_index: 0,
                    target_input_parameters: vec![2],
                    value_input_parameters: vec![1],
                }],
                direct_calls: Vec::new(),
                control_flow_regions: Vec::new(),
            }],
        };

        let comparison = compare_mir_effects(&pack, Some(&summary));
        assert_eq!(comparison.status, MirEffectComparisonStatus::Mismatch);
        assert!(comparison.functions[0].missing.is_empty());
        assert!(comparison.functions[0].dependency_mismatches[0]
            .missing_input_dependencies
            .is_empty());
        assert_eq!(
            comparison.functions[0].dependency_mismatches[0].missing_target_input_dependencies,
            vec![1]
        );
        assert_eq!(
            comparison.functions[0].dependency_mismatches[0].missing_value_input_dependencies,
            vec![2]
        );
    }

    #[test]
    fn mir_effect_comparison_requires_each_distinct_memory_role_signature() {
        let mut analysis = empty_pack().analysis;
        analysis.instructions = vec![
            InstructionFact {
                pc: 0,
                opcode: 0x7b,
                dst: 1,
                src: 2,
                offset: 0,
                immediate: 0,
                assembly: "stxdw [r1], r2".into(),
            },
            InstructionFact {
                pc: 1,
                opcode: 0x7b,
                dst: 3,
                src: 4,
                offset: 8,
                immediate: 0,
                assembly: "stxdw [r3+0x8], r4".into(),
            },
            InstructionFact {
                pc: 2,
                opcode: 0x95,
                dst: 0,
                src: 0,
                offset: 0,
                immediate: 0,
                assembly: "exit".into(),
            },
        ];
        analysis.basic_blocks[0].end_instruction_index = 3;
        analysis.memory_accesses = vec![
            sos_analysis::MemoryAccessFact {
                pc: 0,
                kind: "store".into(),
                width: 8,
                base_register: 1,
                value_register: Some(2),
                offset: 0,
                immediate: None,
                assembly: "stxdw [r1], r2".into(),
            },
            sos_analysis::MemoryAccessFact {
                pc: 1,
                kind: "store".into(),
                width: 8,
                base_register: 3,
                value_register: Some(4),
                offset: 8,
                immediate: None,
                assembly: "stxdw [r3+0x8], r4".into(),
            },
        ];
        let semantics = sos_solana::analyze(&analysis);
        let pack = EvidencePack::from_analysis(ModelTask::Refinement, None, analysis, semantics);
        let summary = MirProgramSummary {
            schema_version: MIR_SUMMARY_SCHEMA_VERSION,
            functions: vec![MirFunctionSummary {
                name: "fn_0".into(),
                parameter_count: 4,
                basic_block_count: 1,
                edges: Vec::new(),
                branch_terminators: 0,
                calls: 0,
                memory_writes: 2,
                returns: 1,
                has_loop: false,
                effect_dependencies: vec![
                    MirEffectDependency {
                        kind: RecoveredEffectKind::MemoryWrite,
                        input_parameters: vec![1, 2, 3, 4],
                        target_input_parameters: vec![1, 3],
                        value_input_parameters: vec![2, 4],
                    },
                    MirEffectDependency {
                        kind: RecoveredEffectKind::Return,
                        input_parameters: Vec::new(),
                        target_input_parameters: Vec::new(),
                        value_input_parameters: Vec::new(),
                    },
                ],
                memory_write_dependencies: vec![
                    MirMemoryWriteDependency {
                        block: 0,
                        statement_index: 0,
                        target_input_parameters: vec![1],
                        value_input_parameters: vec![4],
                    },
                    MirMemoryWriteDependency {
                        block: 0,
                        statement_index: 1,
                        target_input_parameters: vec![3],
                        value_input_parameters: vec![2],
                    },
                ],
                direct_calls: Vec::new(),
                control_flow_regions: Vec::new(),
            }],
        };

        let comparison = compare_mir_effects(&pack, Some(&summary));
        assert_eq!(comparison.status, MirEffectComparisonStatus::Mismatch);
        assert!(comparison.functions[0].dependency_mismatches.is_empty());
        assert_eq!(comparison.functions[0].memory_write_mismatches.len(), 2);
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
        assert_eq!(
            result.max_output_tokens,
            default_stage_output_tokens(ModelTask::Refinement)
        );
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
                stage_models: AiStageModels::default(),
                stage_token_budgets: AiStageTokenBudgets::default(),
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
    fn pipeline_routes_and_revalidates_stage_specific_models() {
        let pack = empty_pack();
        let requests = Arc::new(Mutex::new(Vec::new()));
        let config = AiPipelineConfig {
            provider_name: "test-provider".into(),
            model: "default-model".into(),
            stage_models: AiStageModels {
                structure_recovery: Some("structure-model".into()),
                signature_recovery: Some("signature-model".into()),
                semantic_labeling: Some("semantic-model".into()),
                explanation: None,
                refinement: Some("refinement-model".into()),
            },
            stage_token_budgets: AiStageTokenBudgets {
                structure_recovery: Some(2_000),
                signature_recovery: Some(1_000),
                semantic_labeling: Some(800),
                explanation: Some(400),
                refinement: Some(2_400),
            },
            prompt_version: "sos-v3".into(),
            artifact_sha256: Some("artifact".into()),
            limits: EvidencePackLimits::default(),
            max_attempts: 1,
        };
        let bundle = run_pipeline(
            ModelRecordingProvider {
                requests: requests.clone(),
            },
            config.clone(),
            &pack.analysis,
            &pack.semantics,
        );
        assert_eq!(
            *requests.lock().unwrap(),
            vec![
                (
                    ModelTask::SignatureRecovery,
                    "signature-model".into(),
                    1_000
                ),
                (
                    ModelTask::StructureRecovery,
                    "structure-model".into(),
                    2_000
                ),
                (ModelTask::SemanticLabeling, "semantic-model".into(), 800),
                (ModelTask::Explanation, "default-model".into(), 400),
                (ModelTask::Refinement, "refinement-model".into(), 2_400),
            ]
        );
        assert!(validate_pipeline_bundle_with_context(
            &config,
            &pack.analysis,
            &pack.semantics,
            &[],
            &bundle,
        )
        .is_ok());

        let mut changed = config;
        changed.stage_models.refinement = Some("different-refinement-model".into());
        assert!(validate_pipeline_bundle_with_context(
            &changed,
            &pack.analysis,
            &pack.semantics,
            &[],
            &bundle,
        )
        .is_err());

        let mut changed_budget = changed;
        changed_budget.stage_models.refinement = Some("refinement-model".into());
        changed_budget.stage_token_budgets.refinement = Some(2_401);
        assert!(validate_pipeline_bundle_with_context(
            &changed_budget,
            &pack.analysis,
            &pack.semantics,
            &[],
            &bundle,
        )
        .is_err());
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
                stage_models: AiStageModels::default(),
                stage_token_budgets: AiStageTokenBudgets::default(),
                prompt_version: "sos-v2".into(),
                artifact_sha256: None,
                limits: EvidencePackLimits::default(),
                max_attempts: 1,
            },
            &pack.analysis,
            &pack.semantics,
        );
        assert_eq!(*prior_counts.lock().unwrap(), vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn pipeline_promotes_only_the_accepted_signature_stage_into_contracts() {
        let pack = empty_pack();
        let contract_counts = Arc::new(Mutex::new(Vec::new()));
        let bundle = run_pipeline(
            ContractRecordingProvider {
                contract_counts: contract_counts.clone(),
            },
            AiPipelineConfig {
                provider_name: "test-provider".into(),
                model: "test-model".into(),
                stage_models: AiStageModels::default(),
                stage_token_budgets: AiStageTokenBudgets::default(),
                prompt_version: "sos-v2".into(),
                artifact_sha256: None,
                limits: EvidencePackLimits::default(),
                max_attempts: 1,
            },
            &pack.analysis,
            &pack.semantics,
        );
        assert_eq!(*contract_counts.lock().unwrap(), vec![0, 1, 1, 1, 1]);
        assert!(bundle
            .stages
            .iter()
            .all(|stage| stage.status == AiStageStatus::Accepted));
    }

    #[test]
    fn pipeline_passes_bounded_dependency_context_into_the_first_stage() {
        let pack = empty_pack();
        let prior_counts = Arc::new(Mutex::new(Vec::new()));
        let context = (0..6)
            .map(|index| UntrustedEvidence {
                source: format!("accepted_callee:fn_{index:x}"),
                content: "x".repeat(20 * 1024),
            })
            .collect::<Vec<_>>();
        let _ = run_pipeline_with_context(
            RecordingProvider {
                prior_counts: prior_counts.clone(),
            },
            AiPipelineConfig {
                provider_name: "test-provider".into(),
                model: "test-model".into(),
                stage_models: AiStageModels::default(),
                stage_token_budgets: AiStageTokenBudgets::default(),
                prompt_version: "sos-v2".into(),
                artifact_sha256: None,
                limits: EvidencePackLimits::default(),
                max_attempts: 1,
            },
            &pack.analysis,
            &pack.semantics,
            &context,
        );
        assert_eq!(*prior_counts.lock().unwrap(), vec![4, 5, 6, 7, 8]);
    }

    #[test]
    fn persisted_pipeline_bundle_is_revalidated_with_dependency_context() {
        let pack = empty_pack();
        let config = AiPipelineConfig {
            provider_name: "test-provider".into(),
            model: "test-model".into(),
            stage_models: AiStageModels::default(),
            stage_token_budgets: AiStageTokenBudgets::default(),
            prompt_version: "sos-v2".into(),
            artifact_sha256: Some("artifact".into()),
            limits: EvidencePackLimits::default(),
            max_attempts: 1,
        };
        let context = vec![UntrustedEvidence {
            source: "accepted_callee:fn_8".into(),
            content: r#"{"functions":[]}"#.into(),
        }];
        let bundle = run_pipeline_with_context(
            EchoProvider,
            config.clone(),
            &pack.analysis,
            &pack.semantics,
            &context,
        );
        assert!(validate_pipeline_bundle_with_context(
            &config,
            &pack.analysis,
            &pack.semantics,
            &context,
            &bundle,
        )
        .is_ok());

        let mut tampered = bundle.clone();
        tampered.stages[0].output_sha256 = Some("00".repeat(32));
        assert!(validate_pipeline_bundle_with_context(
            &config,
            &pack.analysis,
            &pack.semantics,
            &context,
            &tampered,
        )
        .is_err());

        assert!(validate_pipeline_bundle_with_context(
            &config,
            &pack.analysis,
            &pack.semantics,
            &[],
            &bundle,
        )
        .is_err());
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
    fn gateway_rejects_response_for_wrong_model_identity() {
        let gateway = ModelGateway::new(WrongModelProvider, 10_000, 10_000);
        let result = gateway.submit(ModelTask::Explanation, "test", "v1", empty_pack());
        assert!(
            matches!(result, Err(AiError::InvalidResponse(error)) if error.contains("model response identity"))
        );
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
    fn signature_recovery_accepts_evidence_linked_bounded_hypotheses() {
        let mut pack = empty_pack();
        pack.task = ModelTask::SignatureRecovery;
        let request = ModelRequest {
            task: ModelTask::SignatureRecovery,
            prompt_version: "v1".into(),
            model: "test".into(),
            evidence_pack: pack,
            max_output_bytes: 10_000,
            max_output_tokens: 1_200,
        };
        let output = serde_json::json!({
            "signatures": [{
                "function_id": "fn_0",
                "function_name": "process_instruction",
                "return_type": "Result<(), ProgramError>",
                "parameters": [{
                    "name": "instruction_data",
                    "type_name": "&[u8]",
                    "role": "serialized instruction input",
                    "abi_register": null
                }],
                "confidence": 0.6,
                "evidence": [{"source": "ir", "location": "function:fn_0"}]
            }]
        });
        assert!(validate_structured_output(ModelTask::SignatureRecovery, &output).is_ok());
        assert!(validate_structured_consistency(&request, &output).is_ok());
        assert_eq!(
            structured_evidence_refs(ModelTask::SignatureRecovery, &output)
                .unwrap()
                .len(),
            1
        );
    }

    #[test]
    fn signature_recovery_must_map_observed_abi_inputs() {
        let mut analysis = empty_pack().analysis;
        analysis.instructions[0] = InstructionFact {
            pc: 0,
            opcode: 0x79,
            dst: 6,
            src: 1,
            offset: 0,
            immediate: 0,
            assembly: "ldxdw r6, [r1]".into(),
        };
        let semantics = sos_solana::analyze(&analysis);
        let pack =
            EvidencePack::from_analysis(ModelTask::SignatureRecovery, None, analysis, semantics);
        let request = ModelRequest {
            task: ModelTask::SignatureRecovery,
            prompt_version: "v1".into(),
            model: "test".into(),
            evidence_pack: pack,
            max_output_bytes: 10_000,
            max_output_tokens: 1_200,
        };
        let mut output = serde_json::json!({
            "signatures": [{
                "function_id": "fn_0",
                "function_name": "unknown_fn_0",
                "return_type": "unknown_return",
                "parameters": [{
                    "name": "arg_r1",
                    "type_name": "unknown_ptr",
                    "role": null,
                    "abi_register": 1
                }],
                "confidence": 0.4,
                "evidence": [{
                    "source": "analysis",
                    "location": "instruction:0x0",
                    "detail": null
                }]
            }]
        });
        assert!(validate_structured_consistency(&request, &output).is_ok());

        output["signatures"][0]["parameters"][0]["abi_register"] = serde_json::Value::Null;
        let error = validate_structured_consistency(&request, &output).unwrap_err();
        assert!(error
            .to_string()
            .contains("omits observed SBPF ABI input r1"));
    }

    #[test]
    fn signature_recovery_rejects_unknown_functions_unsafe_types_and_missing_evidence() {
        let mut pack = empty_pack();
        pack.task = ModelTask::SignatureRecovery;
        let request = ModelRequest {
            task: ModelTask::SignatureRecovery,
            prompt_version: "v1".into(),
            model: "test".into(),
            evidence_pack: pack,
            max_output_bytes: 10_000,
            max_output_tokens: 1_200,
        };
        let base = serde_json::json!({
            "signatures": [{
                "function_id": "fn_missing",
                "function_name": "process_instruction",
                "return_type": "unknown",
                "parameters": [],
                "confidence": 0.5,
                "evidence": [{"source": "ir", "location": "function:fn_0"}]
            }]
        });
        assert!(validate_structured_consistency(&request, &base).is_err());

        let mut unsafe_type = base.clone();
        unsafe_type["signatures"][0]["function_id"] = serde_json::json!("fn_0");
        unsafe_type["signatures"][0]["return_type"] =
            serde_json::json!("u64; include_bytes!(\"secret\")");
        assert!(validate_structured_consistency(&request, &unsafe_type).is_err());

        let mut no_evidence = base;
        no_evidence["signatures"][0]["function_id"] = serde_json::json!("fn_0");
        no_evidence["signatures"][0]["evidence"] = serde_json::json!([]);
        assert!(structured_evidence_refs(ModelTask::SignatureRecovery, &no_evidence).is_err());
    }

    #[test]
    fn source_stages_must_preserve_the_accepted_signature_contract() {
        let mut pack = empty_pack();
        pack.task = ModelTask::StructureRecovery;
        pack.signature_contracts = vec![SignatureContract {
            source_output_sha256: "ab".repeat(32),
            signature: RecoveredSignature {
                function_id: "fn_0".into(),
                function_name: "process_instruction".into(),
                return_type: "Result<(), ProgramError>".into(),
                parameters: vec![
                    RecoveredParameter {
                        name: "instruction_data".into(),
                        type_name: "&[u8]".into(),
                        role: Some("serialized instruction input".into()),
                        abi_register: Some(1),
                    },
                    RecoveredParameter {
                        name: "account_count".into(),
                        type_name: "u64".into(),
                        role: Some("number of supplied accounts".into()),
                        abi_register: Some(2),
                    },
                ],
                confidence: Some(0.6),
                evidence: vec![sos_types::EvidenceRef {
                    source: "ir".into(),
                    location: "function:fn_0".into(),
                    detail: None,
                }],
            },
        }];
        let valid_source = "fn fn_0(instruction_data: &[u8], account_count: u64) -> Result<(), ProgramError> { unknown_semantics(); }";
        let effects = required_structure_effects(&pack, "fn_0")
            .into_iter()
            .map(|required| RecoveredEffect {
                kind: required.kind,
                input_registers: required.input_registers,
                target_input_registers: required.target_input_registers,
                value_input_registers: required.value_input_registers,
                evidence: vec![sos_types::EvidenceRef {
                    source: "ir".into(),
                    location: required.location,
                    detail: None,
                }],
            })
            .collect();
        let candidate = RecoveredFunction {
            function_id: "fn_0".into(),
            pseudocode: valid_source.into(),
            source_mappings: Vec::new(),
            control_flow_mappings: Vec::new(),
            memory_field_mappings: Vec::new(),
            direct_call_mappings: Vec::new(),
            effects,
            evidence: vec![sos_types::EvidenceRef {
                source: "ir".into(),
                location: "function:fn_0".into(),
                detail: None,
            }],
        };
        let request = ModelRequest {
            task: ModelTask::StructureRecovery,
            prompt_version: "v1".into(),
            model: "test".into(),
            evidence_pack: pack.clone(),
            max_output_bytes: 10_000,
            max_output_tokens: 1_200,
        };
        assert!(validate_structure_candidate(&request, &candidate).is_ok());

        for changed in [
            "fn fn_0(data: &[u8], account_count: u64) -> Result<(), ProgramError> { unknown_semantics(); }",
            "fn fn_0(instruction_data: &[u8], account_count: usize) -> Result<(), ProgramError> { unknown_semantics(); }",
            "fn fn_0(account_count: u64, instruction_data: &[u8]) -> Result<(), ProgramError> { unknown_semantics(); }",
            "fn fn_0(instruction_data: &[u8], account_count: u64) -> u64 { unknown_semantics(); }",
        ] {
            let mut mutated = candidate.clone();
            mutated.pseudocode = changed.into();
            let error = validate_structure_candidate(&request, &mutated).unwrap_err();
            assert!(error.to_string().contains("changes the accepted signature contract"));
        }

        let mut refinement_pack = pack;
        refinement_pack.task = ModelTask::Refinement;
        let refinement_request = ModelRequest {
            task: ModelTask::Refinement,
            prompt_version: "v1".into(),
            model: "test".into(),
            evidence_pack: refinement_pack,
            max_output_bytes: 10_000,
            max_output_tokens: 1_200,
        };
        let valid_refinement = serde_json::json!({
            "schema_version": REFINEMENT_SCHEMA_VERSION,
            "patches": [{
                "function_id": "fn_0",
                "replacement": valid_source,
                "memory_field_mappings": [],
                "direct_call_mappings": [],
                "evidence": [{"source": "ir", "location": "function:fn_0", "detail": null}]
            }],
            "observations": [],
            "validation": {"status": "unverified", "notes": []}
        });
        assert!(validate_structured_consistency(&refinement_request, &valid_refinement).is_ok());
        let mut changed_refinement = valid_refinement;
        changed_refinement["patches"][0]["replacement"] = serde_json::json!(
            "fn fn_0(data: &[u8], account_count: u64) -> Result<(), ProgramError> {}"
        );
        assert!(validate_structured_consistency(&refinement_request, &changed_refinement).is_err());
    }

    #[test]
    fn scoped_structure_and_signature_recovery_reject_empty_candidates() {
        for (task, output) in [
            (
                ModelTask::StructureRecovery,
                serde_json::json!({"functions": []}),
            ),
            (
                ModelTask::SignatureRecovery,
                serde_json::json!({"signatures": []}),
            ),
        ] {
            let mut pack = empty_pack();
            pack.task = task;
            let request = ModelRequest {
                task,
                prompt_version: "v1".into(),
                model: "test".into(),
                evidence_pack: pack,
                max_output_bytes: 10_000,
                max_output_tokens: 1_200,
            };
            assert!(validate_structured_consistency(&request, &output).is_err());
        }
    }

    #[test]
    fn structure_recovery_must_cover_deterministic_effect_contract() {
        let mut pack = empty_pack();
        pack.task = ModelTask::StructureRecovery;
        pack.analysis.instructions[0] = InstructionFact {
            pc: 0,
            opcode: 0x7b,
            dst: 1,
            src: 2,
            offset: 0,
            immediate: 0,
            assembly: "stxdw [r1], r2".into(),
        };
        pack.analysis.memory_accesses = vec![sos_analysis::MemoryAccessFact {
            pc: 0,
            kind: "store".into(),
            width: 8,
            base_register: 1,
            value_register: Some(2),
            offset: 0,
            immediate: None,
            assembly: "stxdw [r1], r2".into(),
        }];
        pack.ir.operations[0].kind = "store".into();
        pack.ir.operations[0].assembly = "stxdw [r1+0x0], r2".into();
        pack.ir.operations[0].memory_effect = Some("store:8:1+0".into());
        pack.ir.operations.push(sos_analysis::IrOperation {
            id: "op_1".into(),
            block_id: "bb_0".into(),
            pc: 1,
            kind: "control_flow".into(),
            assembly: "jne r3, 0, lbb_0".into(),
            dst: 3,
            src: 0,
            offset: -1,
            immediate: 0,
            memory_effect: None,
            call_target_pc: None,
            call_target_pcs: Vec::new(),
        });
        pack.ir.operations.push(sos_analysis::IrOperation {
            id: "op_2".into(),
            block_id: "bb_0".into(),
            pc: 2,
            kind: "raw".into(),
            assembly: "exit".into(),
            dst: 0,
            src: 0,
            offset: 0,
            immediate: 0,
            memory_effect: None,
            call_target_pc: None,
            call_target_pcs: Vec::new(),
        });
        pack.ir.sccs = vec![sos_analysis::SccFact {
            id: 0,
            block_starts: vec![0],
            recursive: true,
        }];
        let request = ModelRequest {
            task: ModelTask::StructureRecovery,
            prompt_version: "v1".into(),
            model: "test".into(),
            evidence_pack: pack,
            max_output_bytes: 10_000,
            max_output_tokens: 1_200,
        };
        let effect = |kind: &str, location: &str| {
            let (inputs, targets, values) = if kind == "memory_write" {
                (vec![1, 2], vec![1], vec![2])
            } else {
                (Vec::new(), Vec::new(), Vec::new())
            };
            serde_json::json!({
                "kind": kind,
                "input_registers": inputs,
                "target_input_registers": targets,
                "value_input_registers": values,
                "evidence": [{"source": "ir", "location": location, "detail": null}]
            })
        };
        let valid = serde_json::json!({"functions": [{
            "function_id": "fn_0",
            "pseudocode": "fn fn_0() { loop { unknown_memory_write(); if unknown_condition() { break; } } }",
            "effects": [
                effect("memory_write", "operation:op_0"),
                effect("branch", "operation:op_1"),
                effect("loop", "block:bb_0"),
                effect("return", "operation:op_2")
            ],
            "evidence": [{"source": "ir", "location": "function:fn_0", "detail": null}]
        }]});
        assert!(validate_structured_consistency(&request, &valid).is_ok());

        let mut wrong_dependencies = valid.clone();
        wrong_dependencies["functions"][0]["effects"][0]["input_registers"] =
            serde_json::json!([1]);
        assert!(validate_structured_consistency(&request, &wrong_dependencies).is_err());

        let mut swapped_roles = valid.clone();
        swapped_roles["functions"][0]["effects"][0]["target_input_registers"] =
            serde_json::json!([2]);
        swapped_roles["functions"][0]["effects"][0]["value_input_registers"] =
            serde_json::json!([1]);
        assert!(validate_structured_consistency(&request, &swapped_roles).is_err());

        let mut invented = valid.clone();
        invented["functions"][0]["effects"]
            .as_array_mut()
            .unwrap()
            .push(effect("call", "operation:op_0"));
        assert!(validate_structured_consistency(&request, &invented)
            .unwrap_err()
            .to_string()
            .contains("invents a binary effect"));

        let mut duplicated = valid.clone();
        let duplicate = duplicated["functions"][0]["effects"][0].clone();
        duplicated["functions"][0]["effects"]
            .as_array_mut()
            .unwrap()
            .push(duplicate);
        assert!(validate_structured_consistency(&request, &duplicated)
            .unwrap_err()
            .to_string()
            .contains("duplicates one deterministic binary effect"));

        let mut missing_write = valid;
        missing_write["functions"][0]["effects"]
            .as_array_mut()
            .unwrap()
            .remove(0);
        let error = validate_structured_consistency(&request, &missing_write).unwrap_err();
        assert!(error.to_string().contains("MemoryWrite:operation:op_0"));
    }

    #[test]
    fn structure_recovery_must_map_each_deterministic_cfg_region() {
        let mut pack = empty_pack();
        pack.task = ModelTask::StructureRecovery;
        pack.ir.blocks = vec![
            sos_analysis::IrBlock {
                id: "bb_0".into(),
                function_id: "fn_0".into(),
                start_pc: 0,
                end_instruction_index: 1,
                successors: vec!["bb_1".into(), "bb_4".into()],
                scc_id: None,
            },
            sos_analysis::IrBlock {
                id: "bb_1".into(),
                function_id: "fn_0".into(),
                start_pc: 1,
                end_instruction_index: 2,
                successors: vec!["bb_2".into()],
                scc_id: Some(7),
            },
            sos_analysis::IrBlock {
                id: "bb_2".into(),
                function_id: "fn_0".into(),
                start_pc: 2,
                end_instruction_index: 3,
                successors: vec!["bb_1".into(), "bb_3".into()],
                scc_id: Some(7),
            },
            sos_analysis::IrBlock {
                id: "bb_3".into(),
                function_id: "fn_0".into(),
                start_pc: 3,
                end_instruction_index: 4,
                successors: vec![],
                scc_id: None,
            },
            sos_analysis::IrBlock {
                id: "bb_4".into(),
                function_id: "fn_0".into(),
                start_pc: 4,
                end_instruction_index: 5,
                successors: vec!["bb_3".into()],
                scc_id: None,
            },
        ];
        pack.ir.sccs = vec![sos_analysis::SccFact {
            id: 7,
            block_starts: vec![1, 2],
            recursive: true,
        }];
        pack.control_flow_regions = derive_control_flow_regions(&pack.ir);
        assert_eq!(pack.control_flow_regions.len(), 2);
        let branch = pack
            .control_flow_regions
            .iter()
            .find(|region| region.kind == ControlFlowRegionKind::Branch)
            .unwrap();
        assert_eq!(branch.header_block_id, "bb_0");
        assert_eq!(branch.join_block_id.as_deref(), Some("bb_3"));
        assert_eq!(branch.entry_blocks, vec!["bb_1", "bb_4"]);
        assert_eq!(branch.arms.len(), 2);
        let loop_region = pack
            .control_flow_regions
            .iter()
            .find(|region| region.kind == ControlFlowRegionKind::Loop)
            .unwrap();
        assert_eq!(loop_region.header_block_id, "bb_1");
        assert_eq!(loop_region.member_blocks, vec!["bb_1", "bb_2"]);
        assert_eq!(loop_region.exit_blocks, vec!["bb_3"]);
        assert_eq!(loop_region.cycle_edges.len(), 1);
        let hosted = crate::groq::hosted_model_evidence(&pack);
        assert_eq!(
            hosted["program"]["control_flow_regions"]
                .as_array()
                .unwrap()
                .len(),
            2
        );

        let request = ModelRequest {
            task: ModelTask::StructureRecovery,
            prompt_version: "v1".into(),
            model: "test".into(),
            evidence_pack: pack,
            max_output_bytes: 10_000,
            max_output_tokens: 1_200,
        };
        let valid = serde_json::json!({"functions": [{
            "function_id": "fn_0",
            "pseudocode": "fn fn_0() {\nif choose_path {\nloop {\nunknown_body();\n}\n}\n}",
            "control_flow_mappings": [
                {
                    "region_id": "region_branch_0",
                    "output_fragment": "if choose_path {",
                    "evidence": [{"source": "ir", "location": "block:bb_0", "detail": null}]
                },
                {
                    "region_id": "region_loop_1",
                    "output_fragment": "loop {",
                    "evidence": [{"source": "ir", "location": "block:bb_1", "detail": null}]
                }
            ],
            "effects": [
                {
                    "kind": "loop",
                    "input_registers": [],
                    "target_input_registers": [],
                    "value_input_registers": [],
                    "evidence": [{"source": "ir", "location": "block:bb_1", "detail": null}]
                },
                {
                    "kind": "loop",
                    "input_registers": [],
                    "target_input_registers": [],
                    "value_input_registers": [],
                    "evidence": [{"source": "ir", "location": "block:bb_2", "detail": null}]
                }
            ],
            "evidence": [{"source": "ir", "location": "function:fn_0", "detail": null}]
        }]});
        if let Err(error) = validate_structured_consistency(&request, &valid) {
            panic!("valid region mapping was rejected: {error}");
        }

        let mut omitted = valid.clone();
        omitted["functions"][0]["control_flow_mappings"]
            .as_array_mut()
            .unwrap()
            .pop();
        assert!(validate_structured_consistency(&request, &omitted)
            .unwrap_err()
            .to_string()
            .contains("control-flow regions"));

        let mut wrong_header = valid;
        wrong_header["functions"][0]["control_flow_mappings"][1]["evidence"][0]["location"] =
            serde_json::json!("block:bb_2");
        assert!(validate_structured_consistency(&request, &wrong_header)
            .unwrap_err()
            .to_string()
            .contains("must cite its deterministic header"));

        #[derive(Clone)]
        struct RegionEchoProvider;
        impl ModelProvider for RegionEchoProvider {
            fn generate_structured(&self, request: ModelRequest) -> Result<ModelResponse, AiError> {
                let output = test_structure_output(
                    &request.evidence_pack,
                    "fn_0",
                    "fn fn_0()",
                    serde_json::json!({
                        "source": "ir",
                        "location": "function:fn_0",
                        "detail": null
                    }),
                );
                Ok(ModelResponse {
                    task: request.task,
                    model: request.model,
                    prompt_version: request.prompt_version,
                    structured_output: output,
                    evidence: Vec::new(),
                })
            }
        }
        let mut stale_pack = request.evidence_pack.clone();
        stale_pack.control_flow_regions.clear();
        let response = ModelGateway::new(RegionEchoProvider, 1024 * 1024, 1024 * 1024)
            .submit(ModelTask::StructureRecovery, "test", "v1", stale_pack)
            .unwrap();
        assert_eq!(
            response.structured_output["functions"][0]["control_flow_mappings"]
                .as_array()
                .unwrap()
                .len(),
            2
        );
    }

    #[test]
    fn structure_recovery_must_preserve_deterministic_memory_layouts() {
        let mut analysis = empty_pack().analysis;
        analysis.instructions[0] = InstructionFact {
            pc: 0,
            opcode: 0x7b,
            dst: 1,
            src: 2,
            offset: 8,
            immediate: 0,
            assembly: "stxdw [r1+0x8], r2".into(),
        };
        analysis.memory_accesses = vec![sos_analysis::MemoryAccessFact {
            pc: 0,
            kind: "store".into(),
            width: 8,
            base_register: 1,
            value_register: Some(2),
            offset: 8,
            immediate: None,
            assembly: "stxdw [r1+0x8], r2".into(),
        }];
        let semantics = sos_solana::analyze(&analysis);
        let mut pack =
            EvidencePack::from_analysis(ModelTask::StructureRecovery, None, analysis, semantics);
        let contracts = required_memory_field_contracts(&pack, "fn_0");
        assert_eq!(contracts.len(), 1);
        assert_eq!(contracts[0].kind, MemoryFieldContractKind::RegisterOffset);
        assert_eq!(contracts[0].offset, 8);
        assert_eq!(contracts[0].machine_offsets, vec![8]);
        assert_eq!(contracts[0].widths, vec![8]);
        assert_eq!(contracts[0].access_kinds, vec!["store"]);
        assert_eq!(contracts[0].field_path, "unknown_memory_r1.field_pos_0x8");
        assert_eq!(contracts[0].type_name, "u64");
        assert_eq!(contracts[0].layout_type, "SosLayoutFn0R1");
        assert_eq!(contracts[0].layout_field, "field_pos_0x8");
        assert!(!contracts[0].semantic_name_recovered);
        let hosted = crate::groq::hosted_model_evidence(&pack);
        assert_eq!(
            hosted["program"]["memory_field_contracts"]
                .as_array()
                .unwrap()
                .len(),
            1
        );

        let output = test_structure_output(
            &pack,
            "fn_0",
            "fn fn_0()",
            serde_json::json!({
                "source": "ir",
                "location": "function:fn_0",
                "detail": null
            }),
        );
        let request = ModelRequest {
            task: ModelTask::StructureRecovery,
            prompt_version: "v1".into(),
            model: "test".into(),
            evidence_pack: pack.clone(),
            max_output_bytes: 100_000,
            max_output_tokens: 2_400,
        };
        validate_structured_consistency(&request, &output).unwrap();

        let mut renamed = output.clone();
        renamed["functions"][0]["memory_field_mappings"][0]["field_path"] =
            serde_json::json!("account.balance");
        assert!(validate_structured_consistency(&request, &renamed)
            .unwrap_err()
            .to_string()
            .contains("deterministic path"));

        let mut omitted = output;
        omitted["functions"][0]["memory_field_mappings"] = serde_json::json!([]);
        assert!(validate_structured_consistency(&request, &omitted)
            .unwrap_err()
            .to_string()
            .contains("memory fields"));

        let refinement_output = serde_json::json!({
            "schema_version": REFINEMENT_SCHEMA_VERSION,
            "patches": [{
                "function_id": "fn_0",
                "replacement": "#[repr(C)] struct SosLayoutFn0R1 { _pad_0: [u8; 8], field_pos_0x8: u64 } fn fn_0() { let unknown_memory_r1 = SosLayoutFn0R1 { _pad_0: [0; 8], field_pos_0x8: 0 }; let recovered: u64 = unknown_memory_r1.field_pos_0x8; unknown_memory_write(); }",
                "memory_field_mappings": [{
                    "contract_id": contracts[0].id.clone(),
                    "field_path": contracts[0].field_path.clone(),
                    "type_name": contracts[0].type_name.clone(),
                    "offset": contracts[0].offset,
                    "layout_type": contracts[0].layout_type.clone(),
                    "layout_field": contracts[0].layout_field.clone(),
                    "output_fragment": "let recovered: u64 = unknown_memory_r1.field_pos_0x8;",
                    "evidence": [contracts[0].evidence[0].clone()]
                }],
                "direct_call_mappings": [],
                "evidence": [{"source": "ir", "location": "function:fn_0", "detail": null}]
            }],
            "observations": [],
            "validation": {"status": "unverified", "notes": []}
        });
        let mut refinement_request = request.clone();
        refinement_request.task = ModelTask::Refinement;
        refinement_request.evidence_pack.task = ModelTask::Refinement;
        validate_structured_consistency(&refinement_request, &refinement_output).unwrap();
        let mut missing_layout = refinement_output.clone();
        missing_layout["patches"][0]["replacement"] = serde_json::json!(
            "// #[repr(C)] struct SosLayoutFn0R1 { field_pos_0x8: u64 }\nfn fn_0() { let unknown_memory_r1 = unknown(); let recovered: u64 = unknown_memory_r1.field_pos_0x8; unknown_memory_write(); }"
        );
        assert!(
            validate_structured_consistency(&refinement_request, &missing_layout)
                .unwrap_err()
                .to_string()
                .contains("must declare #[repr(C)] struct")
        );

        let mut string_only_layout = refinement_output.clone();
        string_only_layout["patches"][0]["replacement"] = serde_json::json!(
            "fn fn_0() { let fake = \"#[repr(C)] struct SosLayoutFn0R1 { field_pos_0x8: u64 }\"; let unknown_memory_r1 = unknown(); let recovered: u64 = unknown_memory_r1.field_pos_0x8; unknown_memory_write(); }"
        );
        assert!(validate_structured_consistency(&refinement_request, &string_only_layout).is_err());

        let mut renamed_refinement = refinement_output;
        renamed_refinement["patches"][0]["memory_field_mappings"][0]["field_path"] =
            serde_json::json!("account.balance");
        assert!(validate_structured_consistency(&refinement_request, &renamed_refinement).is_err());

        pack.memory_field_contracts.clear();
        let response = ModelGateway::new(EchoProvider, 1024 * 1024, 1024 * 1024)
            .submit(ModelTask::StructureRecovery, "test", "v1", pack)
            .unwrap();
        assert_eq!(
            response.structured_output["functions"][0]["memory_field_mappings"]
                .as_array()
                .unwrap()
                .len(),
            1
        );
    }

    #[test]
    fn recovered_sol_account_info_layout_uses_logical_fields_and_machine_offsets() {
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
        let mut analysis = empty_pack().analysis;
        analysis.instructions = layout
            .iter()
            .enumerate()
            .map(|(pc, _)| InstructionFact {
                pc,
                opcode: 0,
                dst: 0,
                src: 0,
                offset: 0,
                immediate: 0,
                assembly: "stack materialization".into(),
            })
            .collect();
        analysis.basic_blocks[0].end_instruction_index = layout.len();
        analysis.stack_accesses = layout
            .iter()
            .enumerate()
            .map(
                |(pc, (field_offset, width))| sos_analysis::StackAccessFact {
                    pc,
                    kind: "store".into(),
                    width: *width,
                    offset: -112 + *field_offset,
                    value_register: Some(0),
                    immediate: None,
                    frame_register: 10,
                    assembly: "stack store".into(),
                },
            )
            .collect();
        let semantics = sos_solana::analyze(&analysis);
        assert_eq!(semantics.stack_account_field_candidates.len(), 9);
        let pack =
            EvidencePack::from_analysis(ModelTask::StructureRecovery, None, analysis, semantics);
        let signer = required_memory_field_contracts(&pack, "fn_0")
            .into_iter()
            .find(|contract| contract.field_path == "account_0.is_signer")
            .unwrap();
        assert_eq!(signer.kind, MemoryFieldContractKind::SolAccountInfo);
        assert_eq!(signer.offset, 48);
        assert_eq!(signer.machine_offsets, vec![-64]);
        assert_eq!(signer.type_name, "u8");
        assert_eq!(signer.layout_type, "SosLayoutFn0Account0");
        assert_eq!(signer.layout_field, "is_signer");
        assert!(signer.semantic_name_recovered);
    }

    #[test]
    fn irreducible_recursive_scc_still_becomes_a_loop_region() {
        let block = |pc: usize, successors: &[usize], scc_id| sos_analysis::IrBlock {
            id: format!("bb_{pc:x}"),
            function_id: "fn_0".into(),
            start_pc: pc,
            end_instruction_index: pc + 1,
            successors: successors
                .iter()
                .map(|successor| format!("bb_{successor:x}"))
                .collect(),
            scc_id,
        };
        let ir = CanonicalIr {
            schema_version: sos_analysis::IR_SCHEMA_VERSION,
            functions: vec![sos_analysis::IrFunction {
                id: "fn_0".into(),
                entry_pc: 0,
                name: "entry".into(),
            }],
            blocks: vec![
                block(0, &[1, 2], None),
                block(1, &[2, 3], Some(9)),
                block(2, &[1, 3], Some(9)),
                block(3, &[], None),
            ],
            operations: Vec::new(),
            sccs: vec![sos_analysis::SccFact {
                id: 9,
                block_starts: vec![1, 2],
                recursive: true,
            }],
        };
        let regions = derive_control_flow_regions(&ir);
        let loop_region = regions
            .iter()
            .find(|region| region.kind == ControlFlowRegionKind::Loop)
            .unwrap();
        assert_eq!(loop_region.member_blocks, vec!["bb_1", "bb_2"]);
        assert_eq!(loop_region.exit_blocks, vec!["bb_3"]);
        assert_eq!(loop_region.cycle_edges.len(), 2);
        assert!(regions.iter().any(|region| {
            region.kind == ControlFlowRegionKind::Branch && region.header_block_id == "bb_0"
        }));
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
            max_output_tokens: default_stage_output_tokens(ModelTask::Explanation),
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
            max_output_tokens: default_stage_output_tokens(ModelTask::Explanation),
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
            max_output_tokens: default_stage_output_tokens(ModelTask::Explanation),
        };
        let response = provider.generate_structured(request).unwrap();
        assert_eq!(response.model, "hosted-test");
        assert_eq!(response.task, ModelTask::Explanation);
        server.join().unwrap();
    }

    #[test]
    fn http_provider_debug_redacts_bearer_token() {
        let debug = format!(
            "{:?}",
            HttpModelProvider::new("https://model.invalid")
                .with_bearer_token("provider-neutral-secret")
        );
        assert!(!debug.contains("provider-neutral-secret"));
        assert!(debug.contains("<redacted>"));
    }

    #[test]
    fn gateway_rejects_response_over_output_budget() {
        let gateway = ModelGateway::new(LargeResponseProvider, 10_000, 32);
        let result = gateway.submit(ModelTask::Explanation, "test", "v1", empty_pack());
        assert!(matches!(result, Err(AiError::OutputTooLarge { .. })));
    }

    #[test]
    fn gateway_rejects_out_of_range_stage_token_budgets() {
        let gateway = ModelGateway::new(EchoProvider, 10_000, 10_000);
        for invalid in [MIN_STAGE_OUTPUT_TOKENS - 1, MAX_STAGE_OUTPUT_TOKENS + 1] {
            let result = gateway.submit_with_token_budget(
                ModelTask::Explanation,
                "test",
                "v1",
                empty_pack(),
                invalid,
            );
            assert!(
                matches!(result, Err(AiError::InvalidResponse(error)) if error.contains("token budget"))
            );
        }
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
        assert!(first.strings.is_empty());

        analysis
            .immediate_constants
            .push(sos_analysis::ImmediateConstantFact {
                pc: 0,
                value: 0x1_0000_0004,
                kind: "lddw".into(),
                assembly: "lddw r1, 0x100000004".into(),
            });
        let referenced = scope_analysis_to_function(&analysis, 0).unwrap();
        assert_eq!(referenced.strings.len(), 1);

        let second = scope_analysis_to_function(&analysis, 8).unwrap();
        assert_eq!(second.functions[0].name, "helper");
        assert_eq!(second.instructions.len(), 1);
        assert_eq!(second.instructions[0].pc, 8);
        assert_eq!(second.basic_blocks[0].end_instruction_index, 1);
        let second_ir = build_canonical_ir(&second);
        assert_eq!(second_ir.operations.len(), 1);
        assert_eq!(second_ir.operations[0].pc, 8);
        assert_eq!(second_ir.operations[0].block_id, "bb_8");
    }

    #[test]
    fn refinement_must_target_a_known_ir_function() {
        let request = ModelRequest {
            task: ModelTask::Refinement,
            prompt_version: "v1".into(),
            model: "test".into(),
            evidence_pack: empty_pack(),
            max_output_bytes: 10_000,
            max_output_tokens: default_stage_output_tokens(ModelTask::Refinement),
        };
        let output = serde_json::json!({
            "patches": [{
                "function_id": "fn_missing",
                "replacement": "fn recovered() { }",
                "memory_field_mappings": [],
                "direct_call_mappings": [],
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
            max_output_tokens: default_stage_output_tokens(ModelTask::Refinement),
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
            max_output_tokens: default_stage_output_tokens(ModelTask::Refinement),
        };
        let unsupported = serde_json::json!({
            "schema_version": REFINEMENT_SCHEMA_VERSION,
            "patches": [],
            "observations": [{"kind": "host_call", "value": "exec", "evidence": []}],
            "validation": {"status": "unverified", "notes": []}
        });
        assert!(validate_structured_consistency(&request, &unsupported).is_err());

        let missing_evidence = serde_json::json!({
            "schema_version": REFINEMENT_SCHEMA_VERSION,
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
            max_output_tokens: default_stage_output_tokens(ModelTask::Refinement),
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

    #[test]
    fn textual_pseudocode_gets_stable_line_evidence_and_validated_output() {
        let pack = evidence_pack_from_pseudocode(
            "tmp = *(u64 *)(ctx + 8)\nif (tmp == 0) goto fail\nreturn tmp",
        )
        .unwrap();
        assert_eq!(pack.task, ModelTask::PseudocodeHumanization);
        assert_eq!(pack.ir.functions[0].id, "fn_0");
        assert_eq!(pack.ir.operations.len(), 3);
        assert_eq!(pack.ir.operations[1].id, "op_100");
        assert!(pack.ir.operations[1]
            .assembly
            .starts_with("line_0002_part_01:"));
        assert_eq!(pack.pseudocode_obligations.len(), 3);
        assert!(pack.pseudocode_obligations[1]
            .constructs
            .contains(&PseudocodeConstruct::Branch));
        assert!(pack.pseudocode_obligations[2]
            .constructs
            .contains(&PseudocodeConstruct::Return));

        let long_line = "x".repeat(150);
        let segmented = evidence_pack_from_pseudocode(&long_line).unwrap();
        assert_eq!(segmented.ir.operations.len(), 2);
        let recovered_text = segmented
            .ir
            .operations
            .iter()
            .map(|operation| operation.assembly.split_once(": ").unwrap().1)
            .collect::<String>();
        assert_eq!(recovered_text, long_line);

        struct HumanizeProvider;
        impl ModelProvider for HumanizeProvider {
            fn generate_structured(&self, request: ModelRequest) -> Result<ModelResponse, AiError> {
                Ok(ModelResponse {
                    task: request.task,
                    model: request.model,
                    prompt_version: request.prompt_version,
                    structured_output: serde_json::json!({
                        "functions": [{
                            "function_id": "fn_0",
                            "pseudocode": "fn read_value(ctx: &Context) -> Result<u64, Error> {\nlet tmp = read_u64(ctx, 8);\nif tmp == 0 { return Err(Error::Invalid); }\nreturn Ok(tmp);\n}",
                            "source_mappings": [
                                {"obligation_id": "source_line_0001", "output_fragment": "let tmp = read_u64(ctx, 8);"},
                                {"obligation_id": "source_line_0002", "output_fragment": "if tmp == 0 { return Err(Error::Invalid); }"},
                                {"obligation_id": "source_line_0003", "output_fragment": "return Ok(tmp);"}
                            ],
                            "evidence": [{
                                "source": "ir",
                                "location": "operation:op_100",
                                "detail": null
                            }]
                        }]
                    }),
                    evidence: Vec::new(),
                })
            }
        }

        let response = ModelGateway::new(HumanizeProvider, 1024 * 1024, 64 * 1024)
            .submit(
                ModelTask::PseudocodeHumanization,
                "test",
                "humanize-v1",
                pack,
            )
            .unwrap();
        assert!(response.structured_output["functions"][0]["pseudocode"]
            .as_str()
            .unwrap()
            .contains("read_value"));
    }

    #[test]
    fn one_line_decompiler_input_recovers_loop_write_and_control_obligations() {
        let pack = evidence_pack_from_pseudocode(
            "void clear(u8 *dst) { LAB_1: if (i >= n) goto LAB_2; dst[i] = 0; i = i + 1; goto LAB_1; LAB_2: return; }",
        )
        .unwrap();
        assert_eq!(pack.pseudocode_obligations.len(), 1);
        let constructs = &pack.pseudocode_obligations[0].constructs;
        for expected in [
            PseudocodeConstruct::FunctionDeclaration,
            PseudocodeConstruct::Branch,
            PseudocodeConstruct::Loop,
            PseudocodeConstruct::MemoryWrite,
            PseudocodeConstruct::Return,
        ] {
            assert!(constructs.contains(&expected), "missing {expected:?}");
        }
    }

    #[test]
    fn humanization_rejects_omitted_reused_and_semantically_wrong_source_mappings() {
        let pack =
            evidence_pack_from_pseudocode("if (flag) return 1;\nstate.value = 2;\nreturn 0;")
                .unwrap();
        let request = ModelRequest {
            task: ModelTask::PseudocodeHumanization,
            prompt_version: "sos-v2".into(),
            model: "test".into(),
            evidence_pack: pack,
            max_output_bytes: 10_000,
            max_output_tokens: default_stage_output_tokens(ModelTask::PseudocodeHumanization),
        };
        let valid = test_humanization_output(
            &request.evidence_pack,
            "fn_0",
            serde_json::json!({
                "source": "ir",
                "location": "function:fn_0",
                "detail": null
            }),
        );
        assert!(validate_structured_consistency(&request, &valid).is_ok());

        let mut omitted = valid.clone();
        omitted["functions"][0]["source_mappings"]
            .as_array_mut()
            .unwrap()
            .pop();
        assert!(validate_structured_consistency(&request, &omitted)
            .unwrap_err()
            .to_string()
            .contains("source obligations"));

        let statement_pack = evidence_pack_from_pseudocode("x = 1;\ny = 2;").unwrap();
        let statement_request = ModelRequest {
            evidence_pack: statement_pack,
            ..request.clone()
        };
        let reused = serde_json::json!({
            "functions": [{
                "function_id": "fn_0",
                "pseudocode": "fn recovered() { let same = (); }",
                "source_mappings": [
                    {"obligation_id": "source_line_0001", "output_fragment": "let same = ();"},
                    {"obligation_id": "source_line_0002", "output_fragment": "let same = ();"}
                ],
                "evidence": []
            }]
        });
        assert!(validate_structured_consistency(&statement_request, &reused)
            .unwrap_err()
            .to_string()
            .contains("reuses one output fragment"));

        let wrong_construct = serde_json::json!({
            "functions": [{
                "function_id": "fn_0",
                "pseudocode": "fn recovered() { let branch_missing = (); state.value = 2; return 0; }",
                "source_mappings": [
                    {"obligation_id": "source_line_0001", "output_fragment": "let branch_missing = ();"},
                    {"obligation_id": "source_line_0002", "output_fragment": "state.value = 2;"},
                    {"obligation_id": "source_line_0003", "output_fragment": "return 0;"}
                ],
                "evidence": []
            }]
        });
        assert!(validate_structured_consistency(&request, &wrong_construct)
            .unwrap_err()
            .to_string()
            .contains("does not visibly preserve Branch"));
    }

    #[test]
    fn humanization_preserves_known_declarations_and_cross_function_calls() {
        let source =
            "undefined8 helper()\n{\n  return 7;\n}\nundefined8 caller()\n{\n  return helper();\n}";
        let units = partition_pseudocode(source).unwrap();
        let pack = evidence_pack_from_pseudocode_unit(&units[1]).unwrap();
        assert_eq!(
            pack.pseudocode_obligations[0].declared_function.as_deref(),
            Some("caller")
        );
        assert_eq!(
            pack.pseudocode_obligations[1].called_functions,
            vec!["helper"]
        );
        let function_id = pack.ir.functions[0].id.clone();
        let request = ModelRequest {
            task: ModelTask::PseudocodeHumanization,
            prompt_version: "sos-v3".into(),
            model: "test".into(),
            evidence_pack: pack,
            max_output_bytes: 10_000,
            max_output_tokens: default_stage_output_tokens(ModelTask::PseudocodeHumanization),
        };
        let valid = test_humanization_output(
            &request.evidence_pack,
            &function_id,
            serde_json::json!({
                "source": "ir",
                "location": format!("function:{function_id}"),
                "detail": null
            }),
        );
        assert!(validate_structured_consistency(&request, &valid).is_ok());

        let mut redirected_call = valid.clone();
        for field in ["pseudocode", "source_mappings"] {
            let value = &mut redirected_call["functions"][0][field];
            if let Some(source) = value.as_str() {
                *value = serde_json::json!(source.replace("helper()", "other_helper()"));
            } else if let Some(mappings) = value.as_array_mut() {
                for mapping in mappings {
                    if let Some(fragment) = mapping["output_fragment"].as_str() {
                        mapping["output_fragment"] =
                            serde_json::json!(fragment.replace("helper()", "other_helper()"));
                    }
                }
            }
        }
        assert!(validate_structured_consistency(&request, &redirected_call)
            .unwrap_err()
            .to_string()
            .contains("exact callee identity"));

        let mut renamed_declaration = valid;
        let pseudocode = renamed_declaration["functions"][0]["pseudocode"]
            .as_str()
            .unwrap()
            .replace("fn caller", "fn renamed_caller");
        renamed_declaration["functions"][0]["pseudocode"] = serde_json::json!(pseudocode);
        let declaration_mapping = renamed_declaration["functions"][0]["source_mappings"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .find(|mapping| mapping["obligation_id"] == "source_line_0005")
            .unwrap();
        let fragment = declaration_mapping["output_fragment"]
            .as_str()
            .unwrap()
            .replace("fn caller", "fn renamed_caller");
        declaration_mapping["output_fragment"] = serde_json::json!(fragment);
        assert!(
            validate_structured_consistency(&request, &renamed_declaration)
                .unwrap_err()
                .to_string()
                .contains("exact symbol identity")
        );
    }

    #[test]
    fn persisted_humanization_response_is_revalidated_against_fresh_evidence() {
        let pack = evidence_pack_from_pseudocode("let value = 7\nreturn value").unwrap();
        let response = ModelResponse {
            task: ModelTask::PseudocodeHumanization,
            model: "test".into(),
            prompt_version: "sos-v1".into(),
            structured_output: serde_json::json!({
                "functions": [{
                    "function_id": "fn_0",
                    "pseudocode": "fn recovered() -> u64 { let value = 7; return value; }",
                    "source_mappings": [
                        {"obligation_id": "source_line_0001", "output_fragment": "let value = 7;"},
                        {"obligation_id": "source_line_0002", "output_fragment": "return value;"}
                    ],
                    "evidence": [{
                        "source": "ir",
                        "location": "operation:op_100",
                        "detail": null
                    }]
                }]
            }),
            evidence: Vec::new(),
        };
        assert!(
            validate_model_response_for_pack("test", "sos-v1", pack.clone(), &response).is_ok()
        );

        let mut wrong_model = response.clone();
        wrong_model.model = "other".into();
        assert!(
            validate_model_response_for_pack("test", "sos-v1", pack.clone(), &wrong_model).is_err()
        );

        let different_pack = evidence_pack_from_pseudocode("return 8").unwrap();
        assert!(
            validate_model_response_for_pack("test", "sos-v1", different_pack, &response).is_err()
        );
    }

    #[test]
    fn pseudocode_partitioning_prefers_functions_then_bounds_oversized_regions() {
        let functions =
            "undefined8 first()\n{\n  return 1;\n}\nundefined8 second()\n{\n  return first();\n}";
        let units = partition_pseudocode(functions).unwrap();
        assert_eq!(units.len(), 2);
        assert_eq!((units[0].start_line, units[0].end_line), (1, 4));
        assert_eq!((units[1].start_line, units[1].end_line), (5, 8));
        assert_eq!(units[0].function_name.as_deref(), Some("first"));
        assert_eq!(units[1].function_name.as_deref(), Some("second"));
        assert_eq!(units[1].referenced_functions, vec!["first"]);

        let large = (1..=300)
            .map(|line| format!("value_{line} = value_{line} + 1;"))
            .collect::<Vec<_>>()
            .join("\n");
        let units = partition_pseudocode(&large).unwrap();
        assert_eq!(units.len(), 3);
        assert_eq!((units[0].start_line, units[0].end_line), (1, 128));
        assert_eq!((units[1].start_line, units[1].end_line), (129, 256));
        assert_eq!((units[2].start_line, units[2].end_line), (257, 300));
        assert_eq!(
            units
                .iter()
                .map(|unit| unit.source.lines().count())
                .sum::<usize>(),
            300
        );

        let second = evidence_pack_from_pseudocode_unit(&units[1]).unwrap();
        assert_eq!(second.ir.functions[0].id, "fn_8000");
        assert_eq!(second.ir.operations[0].id, "op_8000");
        assert!(second.ir.operations[0]
            .assembly
            .starts_with("line_0129_part_01:"));
    }

    #[test]
    fn deterministic_humanization_normalizes_common_ghidra_constructs() {
        let source = "undefined8 process_account(long ctx, byte *instruction_data, ulong instruction_len)\n{\n  long account;\n  ulong amount;\n  if (instruction_len < 9) {\n    return 0x100000002;\n  }\n  account = *(long *)(ctx + 0x20);\n  if (*(byte *)(account + 0x08) == 0) {\n    return 5;\n  }\n  if (*instruction_data != 0x2a) {\n    return 0x100000003;\n  }\n  amount = *(ulong *)(instruction_data + 1);\n  *(ulong *)(account + 0x50) = *(ulong *)(account + 0x50) + amount;\n  sol_log_(\"account updated\", 15);\n  return 0;\n}";
        let unit = partition_pseudocode(source).unwrap().remove(0);
        let fallback = deterministic_humanize_pseudocode_unit(&unit).unwrap();

        assert!(fallback.pseudocode.contains(
            "pub unsafe fn process_account(ctx: i64, instruction_data: *mut u8, instruction_len: u64) -> u64"
        ));
        assert!(fallback
            .pseudocode
            .contains("let mut account: i64; // source line 3"));
        assert!(fallback
            .pseudocode
            .contains("account = read_i64(ctx + 0x20);"));
        assert!(fallback
            .pseudocode
            .contains("if read_u8(account + 0x08) == 0 {"));
        assert!(fallback
            .pseudocode
            .contains("if read_value(instruction_data) != 0x2a {"));
        assert!(fallback
            .pseudocode
            .contains("write_u64(account + 0x50, read_u64(account + 0x50) + amount);"));
        assert!(fallback
            .pseudocode
            .contains("sol_log(\"account updated\", 15);"));
        assert_eq!(fallback.line_mappings.len(), source.lines().count());
        assert_eq!(fallback.preserved_lines, 0);
    }

    #[test]
    fn deterministic_humanization_preserves_unknown_lines_as_evidence() {
        let source = "mystery opcode syntax ???";
        let unit = partition_pseudocode(source).unwrap().remove(0);
        let fallback = deterministic_humanize_pseudocode_unit(&unit).unwrap();
        assert!(fallback
            .pseudocode
            .contains("// untranslated: mystery opcode syntax ??? // source line 1"));
        assert_eq!(fallback.transformed_lines, 0);
        assert_eq!(fallback.preserved_lines, 1);
        assert_eq!(fallback.line_mappings[0].source, source);
    }

    #[test]
    fn deterministic_humanization_matches_checked_account_guard_golden() {
        let source = include_str!("../../../fixtures/pseudocode/account_guard.txt");
        let unit = partition_pseudocode(source).unwrap().remove(0);
        let fallback = deterministic_humanize_pseudocode_unit(&unit).unwrap();
        assert_eq!(
            fallback.pseudocode,
            include_str!("../../../fixtures/pseudocode/account_guard.deterministic.rs")
        );
    }

    #[test]
    fn deterministic_humanization_is_total_over_malformed_textual_units() {
        let corpus = [
            "*(u64 *)",
            "if (unterminated",
            "λ_value = *(ulong *)(ptr + 1);",
            "*",
            "switch(value) {\ncase 1:\n  goto LAB_001;\n}",
            "void FUN_001(char *message) {\n  mystery<syntax> ???;\n}",
            "quoted = \"not *(u64 *)(an expression)\";",
        ];
        for source in corpus {
            let units = partition_pseudocode(source).unwrap();
            for unit in units {
                let fallback = deterministic_humanize_pseudocode_unit(&unit).unwrap();
                assert_eq!(
                    fallback.line_mappings.len(),
                    unit.end_line - unit.start_line + 1
                );
                assert_eq!(
                    fallback
                        .line_mappings
                        .iter()
                        .map(|mapping| mapping.source.as_str())
                        .collect::<Vec<_>>(),
                    unit.source.split('\n').collect::<Vec<_>>()
                );
            }
        }
        let quoted = partition_pseudocode("quoted = \"not *(u64 *)(an expression)\";")
            .unwrap()
            .remove(0);
        let fallback = deterministic_humanize_pseudocode_unit(&quoted).unwrap();
        assert!(fallback
            .pseudocode
            .contains("quoted = \"not *(u64 *)(an expression)\";"));
    }

    #[derive(Clone)]
    struct QualificationProvider;

    impl ModelProvider for QualificationProvider {
        fn generate_structured(&self, request: ModelRequest) -> Result<ModelResponse, AiError> {
            let function_id = request.evidence_pack.ir.functions[0].id.clone();
            let mut structured_output = test_humanization_output(
                &request.evidence_pack,
                &function_id,
                serde_json::json!({
                    "source": "ir",
                    "location": format!("function:{function_id}"),
                    "detail": null
                }),
            );
            let function = structured_output["functions"][0]["pseudocode"]
                .as_str()
                .unwrap()
                .to_owned();
            structured_output["functions"][0]["pseudocode"] = serde_json::json!(format!(
                "{function}\n// require signer; verify owner; reject invalid instruction tag; match instruction; invoke token program CPI; checked arithmetic;"
            ));
            Ok(ModelResponse {
                task: request.task,
                model: request.model,
                prompt_version: request.prompt_version,
                structured_output,
                evidence: Vec::new(),
            })
        }
    }

    #[test]
    fn model_qualification_requires_schema_evidence_and_semantic_concepts() {
        let corpus = ModelQualificationCorpus {
            schema_version: MODEL_QUALIFICATION_SCHEMA_VERSION,
            name: "test-solana-security".into(),
            cases: vec![
                ModelQualificationCase {
                    id: "account-guard".into(),
                    category: "solana-security".into(),
                    pseudocode: "void guard(Account *a) { if (!a->signer) return 1; if (a->owner != expected) return 2; }".into(),
                    required_concepts: vec![
                        vec!["signer".into()],
                        vec!["owner".into()],
                        vec!["reject".into(), "error".into()],
                    ],
                    forbidden_fragments: vec!["sos_raw_op".into()],
                },
                ModelQualificationCase {
                    id: "dispatch-cpi".into(),
                    category: "solana-blockchain".into(),
                    pseudocode: "void dispatch(u8 tag) { if (tag == 1) token_cpi(); else abort(); }".into(),
                    required_concepts: vec![
                        vec!["instruction tag".into(), "discriminator".into()],
                        vec!["match".into(), "branch".into()],
                        vec!["token program".into(), "cpi".into()],
                    ],
                    forbidden_fragments: vec!["goto".into()],
                },
            ],
        };
        let result = qualify_humanization_model(
            QualificationProvider,
            "test_provider",
            "test-model",
            "qualification-v1",
            &corpus,
            1.0,
        )
        .unwrap();
        assert!(result.qualified);
        assert_eq!(result.overall_score, 1.0);
        assert!(result
            .cases
            .iter()
            .all(|case| case.schema_and_evidence_accepted));

        let mut impossible = corpus;
        impossible.cases[0]
            .required_concepts
            .push(vec!["rent epoch".into()]);
        let result = qualify_humanization_model(
            QualificationProvider,
            "test_provider",
            "test-model",
            "qualification-v1",
            &impossible,
            1.0,
        )
        .unwrap();
        assert!(!result.qualified);
        assert_eq!(
            result.cases[0].missing_concepts,
            vec![vec![String::from("rent epoch")]]
        );
    }
}
