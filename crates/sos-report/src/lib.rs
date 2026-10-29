//! Deterministic, evidence-preserving report projections.

use sos_ai::{
    AiPipelineBundle, AiStageStatus, ExplanationOutput, RefinementOutput, SemanticLabelingOutput,
    StructureRecoveryOutput,
};
use sos_analysis::{FunctionFact, StaticAnalysis};
use sos_solana::{SecurityReport, SemanticAnalysis};
use std::collections::{BTreeMap, BTreeSet};

pub fn render_sarif(report: &SecurityReport) -> String {
    let results = report
        .findings
        .iter()
        .map(|finding| {
            serde_json::json!({
                "ruleId": finding.detector_id,
                "level": sarif_level(&finding.severity),
                "message": {"text": finding.description},
                "locations": finding.instruction_pcs.iter().map(|pc| serde_json::json!({
                    "physicalLocation": {
                        "artifactLocation": {"uri": "analysis.json"},
                        "region": {"message": {"text": format!("instruction:0x{pc:x}")}}
                    }
                })).collect::<Vec<_>>(),
                "properties": {
                    "title": finding.title,
                    "confidence": finding.confidence,
                    "status": format!("{:?}", finding.status),
                    "limitations": finding.limitations,
                    "evidence": finding.evidence,
                }
            })
        })
        .collect::<Vec<_>>();
    serde_json::to_string_pretty(&serde_json::json!({
        "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
        "version": "2.1.0",
        "runs": [{
            "tool": {"driver": {"name": "SOS Decompiler", "version": env!("CARGO_PKG_VERSION")}},
            "results": results
        }]
    }))
    .unwrap_or_else(|error| format!("{{\"error\":\"{error}\"}}"))
}

fn sarif_level(severity: &str) -> &'static str {
    match severity {
        "critical" | "high" => "error",
        "medium" => "warning",
        _ => "note",
    }
}

pub fn render_markdown(analysis: &StaticAnalysis, semantics: &SemanticAnalysis) -> String {
    let mut output = String::new();
    output.push_str("# SOS Decompiler Report\n\n");
    output.push_str("This report is generated from deterministic analysis. Names and semantic interpretations are hypotheses unless explicitly marked as facts.\n\n");
    output.push_str("## Binary facts\n\n");
    output.push_str(&format!("- SBPF version: `{}`\n- Entrypoint: `{}`\n- Instructions: `{}`\n- Functions: `{}`\n- Basic blocks: `{}`\n- Stack accesses: `{}`\n- SCCs: `{}`\n- Dataflow edges: `{}`\n\n", analysis.sbpf_version, analysis.entrypoint, analysis.instructions.len(), analysis.functions.len(), analysis.basic_blocks.len(), analysis.stack_accesses.len(), analysis.sccs.len(), analysis.dataflow_edges.len()));
    if analysis.verifier_diagnostics.is_empty() {
        output.push_str("Verifier diagnostics: none reported by the runtime-aligned verifier.\n\n");
    } else {
        output.push_str("Verifier diagnostics:\n");
        for diagnostic in analysis.verifier_diagnostics.iter().take(32) {
            output.push_str(&format!("- `{diagnostic}`\n"));
        }
        output.push('\n');
    }
    output.push_str(&format!("Runtime ABI model: `{}` logical fields; account-record offsets remain variable-length.\n\n", semantics.runtime_abi.len()));
    output.push_str("## Relocations\n\n");
    if analysis.relocations.is_empty() {
        output.push_str("No ELF relocations were exposed by the object parser.\n\n");
    } else {
        output.push_str("| Section | Offset | Kind | Encoding | Size | Target | Addend |\n|---|---:|---|---|---:|---|---:|\n");
        for relocation in analysis.relocations.iter().take(256) {
            output.push_str(&format!(
                "| `{}` | `0x{:x}` | `{}` | `{}` | `{}` | `{}` | `{}` |\n",
                relocation.section,
                relocation.offset,
                relocation.kind,
                relocation.encoding,
                relocation.size,
                relocation.target,
                relocation.addend
            ));
        }
        output.push('\n');
    }
    output.push_str("## Read-only strings\n\n");
    for string in semantics.string_candidates.iter().take(64) {
        output.push_str(&format!(
            "- `ro+0x{:x}`: `{}` (possible log/error/data string)\n",
            string.offset, string.value
        ));
    }
    if semantics.string_candidates.is_empty() {
        output.push_str("No printable read-only strings were identified.\n");
    }
    output.push('\n');
    output.push_str("## Resolved log calls\n\n");
    if semantics.resolved_log_calls.is_empty() {
        output.push_str("No logging syscall arguments resolved to exact read-only strings.\n");
    } else {
        for call in semantics.resolved_log_calls.iter().take(64) {
            output.push_str(&format!(
                "- `0x{:x}`: `{}`(`ro+0x{:x}`, `{}` bytes) -> `{}` (`{:?}`).\n",
                call.instruction_pc,
                call.syscall,
                call.string_offset,
                call.length,
                call.value,
                call.status
            ));
        }
    }
    output.push('\n');
    output.push_str("## Immediate constants\n\n");
    for constant in semantics.constant_candidates.iter().take(64) {
        output.push_str(&format!(
            "- `0x{:x}`: `{}`; interpretations: `{}`.\n",
            constant.pc,
            constant.value,
            constant.interpretations.join("`, `")
        ));
    }
    if semantics.constant_candidates.is_empty() {
        output.push_str("No LDDW immediate constants were identified.\n");
    }
    output.push('\n');
    output.push_str("## Recovered stack account fields\n\n");
    if semantics.stack_account_field_candidates.is_empty() {
        output.push_str("No complete SolAccountInfo stack layouts were recovered.\n");
    } else {
        for field in semantics.stack_account_field_candidates.iter().take(128) {
            output.push_str(&format!(
                "- `r10{:+}`: `accounts[{}].{}` width `{}`; accesses `{}` (`{:?}`).\n",
                field.frame_offset,
                field.account_index,
                field.field,
                field.width,
                field
                    .access_pcs
                    .iter()
                    .map(|pc| format!("0x{pc:x}"))
                    .collect::<Vec<_>>()
                    .join("`, `"),
                field.status
            ));
        }
    }
    output.push('\n');
    output.push_str("## Recovered Solana return codes\n\n");
    if semantics.runtime_error_candidates.is_empty() {
        output.push_str("No exact Solana built-in error values were recovered.\n");
    } else {
        for candidate in semantics.runtime_error_candidates.iter().take(64) {
            output.push_str(&format!(
                "- `0x{:x}`: `0x{:x}` -> `{}` (`{:?}`).\n",
                candidate.instruction_pc, candidate.value, candidate.name, candidate.status
            ));
        }
    }
    output.push('\n');
    output.push_str("## Known Solana program fingerprints\n\n");
    if semantics.protocol_fingerprints.is_empty() {
        output.push_str(
            "No exact known-program ID byte sequences were recovered from read-only sections.\n\n",
        );
    } else {
        for fingerprint in &semantics.protocol_fingerprints {
            output.push_str(&format!(
                "- `{}` `{}` ({}) at `ro+0x{:x}`; interpretations: `{}` (`{:?}`).\n",
                fingerprint.name,
                fingerprint.program_id,
                fingerprint.category,
                fingerprint.byte_offset,
                fingerprint.interpretations.join("`, `"),
                fingerprint.status
            ));
        }
        output.push('\n');
    }
    output.push_str("## Syscalls\n\n");
    if semantics.syscalls.is_empty() {
        output.push_str("No syscall call sites were identified.\n\n");
    } else {
        output.push_str("| PC | Hash | Name | Category | Status |\n|---:|---:|---|---|---|\n");
        for syscall in &semantics.syscalls {
            output.push_str(&format!(
                "| `0x{:x}` | `{}` | `{}` | `{}` | `{:?}` |\n",
                syscall.instruction_pc,
                syscall.hash,
                syscall.name.as_deref().unwrap_or("unknown"),
                syscall.category.as_deref().unwrap_or("unknown"),
                syscall.status
            ));
        }
        output.push('\n');
    }
    output.push_str("## Calls\n\n");
    if analysis.calls.is_empty() {
        output.push_str("No call sites were identified.\n\n");
    } else {
        output.push_str("| Call PC | Caller | Kind | Target | Symbol hash | Status |\n|---:|---:|---|---:|---:|---|\n");
        for call in &analysis.calls {
            output.push_str(&format!(
                "| `0x{:x}` | `{}` | `{}` | `{}` | `{}` | `{}` |\n",
                call.call_pc,
                call.caller_pc
                    .map(|pc| format!("0x{pc:x}"))
                    .unwrap_or_else(|| "unknown".into()),
                call.kind,
                call.target_pc
                    .map(|pc| format!("0x{pc:x}"))
                    .unwrap_or_else(|| "unresolved".into()),
                call.target_symbol_hash
                    .map(|hash| hash.to_string())
                    .unwrap_or_else(|| "unknown".into()),
                call.status
            ));
        }
        output.push('\n');
    }
    output.push_str("## CPI candidates\n\n");
    if semantics.cpi_candidates.is_empty() {
        output.push_str("No CPI syscall candidates were identified.\n\n");
    } else {
        for candidate in &semantics.cpi_candidates {
            output.push_str(&format!(
                "- `0x{:x}` `{}` with argument registers `{:?}` and recovered values `{:?}`; target program: `{}` (`{:?}`).\n",
                candidate.instruction_pc,
                candidate.syscall,
                candidate.argument_registers,
                candidate.argument_values,
                candidate
                    .target_program
                    .as_deref()
                    .unwrap_or("not recovered"),
                candidate.status
            ));
        }
        output.push('\n');
    }
    output.push_str("## PDA candidates\n\n");
    if semantics.pda_candidates.is_empty() {
        output.push_str("No PDA syscall candidates were identified.\n\n");
    } else {
        for candidate in &semantics.pda_candidates {
            output.push_str(&format!(
                "- `0x{:x}` `{}`; possible seed registers `{:?}`, recovered values `{:?}`, bump result `{:?}` (`{:?}`).\n",
                candidate.instruction_pc,
                candidate.syscall,
                candidate.seed_argument_registers,
                candidate.seed_argument_values,
                candidate.bump_output_register,
                candidate.status
            ));
        }
        output.push('\n');
    }
    output.push_str("## Framework candidates\n\n");
    if semantics.framework_candidates.is_empty() {
        output.push_str("No framework-specific candidate was identified.\n\n");
    } else {
        for candidate in &semantics.framework_candidates {
            output.push_str(&format!(
                "- `{}` at `{}` confidence `{:.2}`; alternatives: `{}`.\n",
                candidate.framework,
                candidate
                    .instruction_pc
                    .map(|pc| format!("0x{pc:x}"))
                    .unwrap_or_else(|| "unknown".into()),
                candidate.confidence,
                candidate.interpretations.join("`, `")
            ));
        }
        output.push('\n');
    }
    output.push_str("## Solana input-field candidates\n\n");
    if semantics.input_field_candidates.is_empty() {
        output.push_str("No direct accesses rooted at the conventional entrypoint input register were identified.\n\n");
    } else {
        for candidate in &semantics.input_field_candidates {
            output.push_str(&format!(
                "- `r{}` + `{}`: widths `{:?}`, loads `{}`, stores `{}`; interpretations: `{}` (`{:?}`).\n",
                candidate.base_register,
                candidate.offset,
                candidate.widths,
                candidate.load_count,
                candidate.store_count,
                candidate.interpretations.join("`, `"),
                candidate.status
            ));
        }
        output.push('\n');
    }
    output.push_str("## Dispatch candidates\n\n");
    if semantics.dispatch_candidates.is_empty() {
        output.push_str("No immediate conditional dispatch candidates were identified.\n\n");
    } else {
        for candidate in &semantics.dispatch_candidates {
            output.push_str(&format!(
                "- `0x{:x}` compares `{}` and branches to `0x{:x}`; interpretations: `{}`.\n",
                candidate.instruction_pc,
                candidate.immediate,
                candidate.target_pc,
                candidate.interpretations.join("`, `")
            ));
        }
        output.push('\n');
    }
    output.push_str("## Semantic graph candidates\n\n");
    output.push_str(&format!(
        "Instruction candidates: `{}`; argument candidates: `{}`; account bases: `{}`; account fields: `{}`; validation checks: `{}`; C SDK deserializers: `{}`; instruction-data guards: `{}`; log candidates: `{}`; error candidates: `{}`.\n\n",
        semantics.instruction_candidates.len(),
        semantics.argument_candidates.len(),
        semantics.account_candidates.len(),
        semantics.account_field_candidates.len(),
        semantics.validation_checks.len(),
        semantics.deserialization_candidates.len(),
        semantics.instruction_data_guard_candidates.len(),
        semantics.log_candidates.len(),
        semantics.error_candidates.len()
    ));
    output.push_str("## Functions\n\n");
    for function in &analysis.functions {
        output.push_str(&format!(
            "- `fn_{:04x}` at `0x{:x}`{}\n",
            function.pc,
            function.pc,
            function_suffix(function)
        ));
    }
    output
}

pub fn render_pseudocode(analysis: &StaticAnalysis, semantics: &SemanticAnalysis) -> String {
    render_source_like(analysis, semantics, false)
}

pub fn render_c_pseudocode(analysis: &StaticAnalysis, semantics: &SemanticAnalysis) -> String {
    render_source_like(analysis, semantics, true)
}

/// Render accepted model output separately from deterministic facts. Model text
/// is always marked as untrusted hypothesis material and remains accompanied by
/// the stage status and audit hashes from the pipeline bundle.
pub fn render_ai_markdown(bundle: &AiPipelineBundle) -> String {
    let mut output = String::new();
    output.push_str("# SOS AI Pipeline Report\n\n");
    output.push_str("This document contains untrusted model output. It is not original source, ");
    output.push_str(
        "does not change deterministic facts, and must not be treated as behaviorally validated.\n\n",
    );
    output.push_str(&format!(
        "Provider: `{}`  \nModel: `{}`  \nArtifact SHA-256: `{}`\n\n",
        escape_inline(&bundle.provider),
        escape_inline(&bundle.model),
        bundle.artifact_sha256.as_deref().unwrap_or("not_available")
    ));
    output.push_str("## Stage audit\n\n");
    output.push_str("| Stage | Status | Model | Output tokens | Prompt | Input hash | Output hash | Error |\n|---|---|---|---:|---|---|---|---|\n");
    for stage in &bundle.stages {
        output.push_str(&format!(
            "| `{:?}` | `{:?}` | `{}` | `{}` | `{}` | `{}` | `{}` | {} |\n",
            stage.task,
            stage.status,
            escape_inline(&stage.model),
            stage.max_output_tokens,
            escape_inline(&stage.prompt_version),
            stage.input_sha256,
            stage.output_sha256.as_deref().unwrap_or("—"),
            stage
                .error
                .as_deref()
                .map(escape_inline)
                .unwrap_or_else(|| "—".into())
        ));
    }
    output.push('\n');

    for stage in &bundle.stages {
        let Some(response) = stage.response.as_ref() else {
            continue;
        };
        if stage.status != AiStageStatus::Accepted {
            continue;
        }
        match stage.task {
            sos_ai::ModelTask::StructureRecovery | sos_ai::ModelTask::PseudocodeHumanization => {
                output.push_str("## Structure recovery / humanization (hypothesis)\n\n");
                if let Ok(parsed) = serde_json::from_value::<StructureRecoveryOutput>(
                    response.structured_output.clone(),
                ) {
                    for function in parsed.functions {
                        output.push_str(&format!(
                            "### `{}`\n\n```text\n{}\n```\n\nEvidence: {}\n\n",
                            escape_inline(&function.function_id),
                            sanitize_code_block(&function.pseudocode),
                            render_evidence_refs(&function.evidence)
                        ));
                    }
                }
            }
            sos_ai::ModelTask::SemanticLabeling => {
                output.push_str("## Semantic labeling (hypotheses)\n\n");
                if let Ok(parsed) = serde_json::from_value::<SemanticLabelingOutput>(
                    response.structured_output.clone(),
                ) {
                    for label in parsed.labels {
                        let value = serde_json::to_string(&label.value)
                            .unwrap_or_else(|_| "<unserializable>".into());
                        output.push_str(&format!(
                            "- `{}` = `{}`; status `{:?}`, confidence `{:?}`; evidence: {}\n",
                            escape_inline(&label.kind),
                            escape_inline(&value),
                            label.status,
                            label.confidence,
                            render_evidence_refs(&label.evidence)
                        ));
                    }
                }
                output.push('\n');
            }
            sos_ai::ModelTask::Explanation => {
                output.push_str("## Explanation (hypothesis)\n\n");
                if let Ok(parsed) =
                    serde_json::from_value::<ExplanationOutput>(response.structured_output.clone())
                {
                    output.push_str(&format!("{}\n\n", escape_inline(&parsed.summary)));
                    for observation in parsed.observations {
                        output.push_str(&format!("- {}\n", escape_inline(&observation)));
                    }
                    output.push('\n');
                }
            }
            sos_ai::ModelTask::Refinement => {
                output.push_str("## Refinement (bounded, unverified)\n\n");
                if let Ok(parsed) =
                    serde_json::from_value::<RefinementOutput>(response.structured_output.clone())
                {
                    output.push_str(&format!(
                        "Validation status: `{}`; notes: {}\n\n",
                        escape_inline(&parsed.validation.status),
                        parsed
                            .validation
                            .notes
                            .iter()
                            .map(|note| escape_inline(note))
                            .collect::<Vec<_>>()
                            .join("; ")
                    ));
                    for patch in parsed.patches {
                        output.push_str(&format!(
                            "### Patch for `{}`\n\n```text\n{}\n```\n\nEvidence: {}\n\n",
                            escape_inline(&patch.function_id),
                            sanitize_code_block(&patch.replacement),
                            render_evidence_refs(&patch.evidence)
                        ));
                    }
                }
            }
        }
    }
    output
}

/// Render the best accepted semantic-lifting candidate for each deterministic
/// IR function. Refinement output wins over structure recovery, but neither is
/// promoted beyond a hypothesis. Missing functions remain explicit instead of
/// being silently invented or omitted.
pub fn render_readable_rust(analysis: &StaticAnalysis, bundle: &AiPipelineBundle) -> String {
    #[derive(Clone)]
    struct Candidate {
        source: String,
        stage: &'static str,
        evidence: Vec<sos_types::EvidenceRef>,
    }

    let valid_ids = analysis
        .functions
        .iter()
        .map(|function| format!("fn_{:x}", function.pc))
        .collect::<BTreeSet<_>>();
    let mut candidates = BTreeMap::<String, Candidate>::new();

    for stage in &bundle.stages {
        if stage.status != AiStageStatus::Accepted {
            continue;
        }
        let Some(response) = stage.response.as_ref() else {
            continue;
        };
        match stage.task {
            sos_ai::ModelTask::StructureRecovery => {
                if let Ok(recovered) = serde_json::from_value::<StructureRecoveryOutput>(
                    response.structured_output.clone(),
                ) {
                    for function in recovered.functions {
                        if valid_ids.contains(&function.function_id)
                            && !function.pseudocode.trim().is_empty()
                        {
                            candidates.entry(function.function_id).or_insert(Candidate {
                                source: strip_markdown_fence(&function.pseudocode),
                                stage: "structure_recovery",
                                evidence: function.evidence,
                            });
                        }
                    }
                }
            }
            sos_ai::ModelTask::Refinement => {
                if let Ok(refinement) =
                    serde_json::from_value::<RefinementOutput>(response.structured_output.clone())
                {
                    for patch in refinement.patches {
                        if valid_ids.contains(&patch.function_id)
                            && !patch.replacement.trim().is_empty()
                        {
                            candidates.insert(
                                patch.function_id,
                                Candidate {
                                    source: strip_markdown_fence(&patch.replacement),
                                    stage: "refinement",
                                    evidence: patch.evidence,
                                },
                            );
                        }
                    }
                }
            }
            _ => {}
        }
    }

    let mut output = String::new();
    output.push_str("// SOS AI-assisted readable Rust-like reconstruction.\n");
    output
        .push_str("// This is evidence-linked hypothesis output, not recovered original source.\n");
    output.push_str(&format!(
        "// Provider: {}; model: {}.\n\n",
        bundle.provider, bundle.model
    ));

    let mut functions = analysis.functions.clone();
    functions.sort_by_key(|function| function.pc);
    for function in functions {
        let function_id = format!("fn_{:x}", function.pc);
        if let Some(candidate) = candidates.get(&function_id) {
            output.push_str(&format!(
                "// candidate: {function_id}; stage: {}; evidence: {}\n",
                candidate.stage,
                candidate
                    .evidence
                    .iter()
                    .map(|evidence| format!("{}:{}", evidence.source, evidence.location))
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
            output.push_str(candidate.source.trim());
            output.push_str("\n\n");
        } else {
            output.push_str(&format!(
                "// unresolved: {function_id} at SBPF PC 0x{:x}; no accepted readable candidate.\n",
                function.pc
            ));
            output.push_str(&format!(
                "fn {function_id}() {{\n    unknown_semantics();\n}}\n\n"
            ));
        }
    }
    output
}

fn strip_markdown_fence(source: &str) -> String {
    let trimmed = source.trim();
    if !trimmed.starts_with("```") {
        return trimmed.to_owned();
    }
    let mut lines = trimmed.lines();
    let _ = lines.next();
    let mut body = lines.collect::<Vec<_>>();
    if body.last().is_some_and(|line| line.trim() == "```") {
        body.pop();
    }
    body.join("\n")
}

fn render_evidence_refs(evidence: &[sos_types::EvidenceRef]) -> String {
    if evidence.is_empty() {
        return "none".into();
    }
    evidence
        .iter()
        .map(|item| {
            format!(
                "`{}:{}`",
                escape_inline(&item.source),
                escape_inline(&item.location)
            )
        })
        .collect::<Vec<_>>()
        .join(", ")
}

fn escape_inline(value: &str) -> String {
    value.replace('`', "'").replace(['\n', '\r'], " ")
}

fn sanitize_code_block(value: &str) -> String {
    value.replace("```", "`` `")
}

/// Render the recovered CFG without inventing source-level control flow. Every
/// emitted operation retains its bytecode PC and block successors, making the
/// projection useful even when type or function recovery is incomplete.
fn render_source_like(
    analysis: &StaticAnalysis,
    semantics: &SemanticAnalysis,
    c_like: bool,
) -> String {
    let mut output = String::new();
    if c_like {
        output.push_str("/* SOS C-like pseudocode; this is not recovered original source. */\n");
    } else {
        output
            .push_str("// SOS deterministic pseudocode; this is not recovered original source.\n");
    }
    output.push_str(
        "// Control flow and raw operations are preserved; unknown semantics remain explicit.\n\n",
    );

    let mut functions = analysis.functions.clone();
    functions.sort_by_key(|function| function.pc);
    for (function_index, function) in functions.iter().enumerate() {
        let next_function_pc = functions.get(function_index + 1).map(|next| next.pc);
        if c_like {
            output.push_str(&format!("uint64_t fn_{:04x}(void) {{\n", function.pc));
        } else {
            output.push_str(&format!("fn fn_{:04x}() {{\n", function.pc));
        }
        if let Some(candidate) = semantics
            .deserialization_candidates
            .iter()
            .find(|candidate| candidate.function_pc == function.pc)
        {
            if c_like {
                output.push_str(&format!(
                    "    /* recovered inlined Solana C SDK deserializer: {} materialized account record(s), {}-byte records; evidence remains in semantics.json */\n",
                    candidate.account_count, candidate.account_record_size
                ));
                output.push_str(&format!(
                    "    SolParameters params = sol_deserialize(input, {});\n",
                    candidate.account_count
                ));
                output.push_str("    SolAccountInfo *accounts = params.ka;\n");
                output.push_str("    uint8_t *instruction_data = params.data;\n");
                output.push_str("    uint64_t instruction_data_len = params.data_len;\n");
            } else {
                output.push_str(&format!(
                    "    // Recovered inlined Solana C SDK deserializer: {} materialized account record(s), {}-byte records; evidence remains in semantics.json.\n",
                    candidate.account_count, candidate.account_record_size
                ));
                output.push_str(&format!(
                    "    let (accounts, instruction_data, program_id) = sol_deserialize(input, {});\n",
                    candidate.account_count
                ));
                output.push_str("    let instruction_data_len = instruction_data.len();\n");
            }
        }

        let blocks = analysis
            .basic_blocks
            .iter()
            .filter(|block| {
                block.start_pc >= function.pc
                    && next_function_pc.is_none_or(|next| block.start_pc < next)
            })
            .collect::<Vec<_>>();
        if blocks.is_empty() {
            for instruction in analysis.instructions.iter().filter(|instruction| {
                instruction.pc >= function.pc
                    && next_function_pc.is_none_or(|next| instruction.pc < next)
            }) {
                render_instruction(&mut output, instruction, analysis, semantics, c_like, 1);
            }
        } else {
            for block in blocks {
                output.push_str(&format!(
                    "    block_{:x}: // {}\n",
                    block.start_pc, block.label
                ));
                let Some(start_index) = analysis
                    .instructions
                    .iter()
                    .position(|instruction| instruction.pc == block.start_pc)
                else {
                    output.push_str("        // block start instruction was not decoded\n");
                    continue;
                };
                let end_index = block.end_instruction_index.min(analysis.instructions.len());
                for instruction in analysis.instructions[start_index..end_index].iter() {
                    render_instruction(&mut output, instruction, analysis, semantics, c_like, 2);
                }
                if !block.destinations.is_empty() {
                    let destinations = block
                        .destinations
                        .iter()
                        .map(|destination| format!("block_{destination:x}"))
                        .collect::<Vec<_>>()
                        .join(", ");
                    if c_like {
                        output.push_str(&format!("        /* successors: {destinations} */\n"));
                    } else {
                        output.push_str(&format!("        // successors: {destinations}\n"));
                    }
                }
            }
        }
        if c_like {
            output.push_str("        sos_unknown_return();\n}\n\n");
        } else {
            output.push_str("    unknown_return();\n}\n\n");
        }
    }
    output
}

fn render_instruction(
    output: &mut String,
    instruction: &sos_analysis::InstructionFact,
    analysis: &StaticAnalysis,
    semantics: &SemanticAnalysis,
    c_like: bool,
    indent: usize,
) {
    let padding = "    ".repeat(indent);
    if let Some(guard) = semantics
        .instruction_data_guard_candidates
        .iter()
        .find(|guard| guard.length_load_pc == instruction.pc)
    {
        let comment_prefix = if c_like { "/*" } else { "//" };
        let comment_suffix = if c_like { " */" } else { "" };
        output.push_str(&format!(
            "{padding}{comment_prefix} 0x{:x}: recovered instruction-data length load from r{}+{}; evidence: analysis/instruction:0x{:x}{comment_suffix}\n",
            instruction.pc, guard.base_register, guard.length_offset, instruction.pc
        ));
        output.push_str(&format!(
            "{padding}instruction_data_len = instruction_data.len();\n"
        ));
        return;
    }
    if let Some(guard) = semantics
        .instruction_data_guard_candidates
        .iter()
        .find(|guard| guard.length_branch_pc == instruction.pc)
    {
        let condition = if instruction.assembly.starts_with("jeq") {
            "=="
        } else {
            "!="
        };
        output.push_str(&format!(
            "{padding}if instruction_data_len {condition} 0 {{ goto block_{:x}; }}\n",
            guard.empty_target_pc
        ));
        return;
    }
    if let Some(guard) = semantics
        .instruction_data_guard_candidates
        .iter()
        .find(|guard| guard.byte_load_pc == instruction.pc)
    {
        output.push_str(&format!(
            "{padding}let instruction_tag = instruction_data[0]; // recovered from r{}+{} at 0x{:x}\n",
            guard.base_register, guard.byte_offset, instruction.pc
        ));
        return;
    }
    if let Some(guard) = semantics
        .instruction_data_guard_candidates
        .iter()
        .find(|guard| guard.tag_compare_pc == instruction.pc)
    {
        let condition = if instruction.assembly.starts_with("jeq") {
            "=="
        } else {
            "!="
        };
        output.push_str(&format!(
            "{padding}if instruction_tag {condition} 0x{:02x} {{ goto block_{:x}; }}\n",
            guard.tag_value, guard.mismatch_target_pc
        ));
        return;
    }
    let resolved_log = semantics
        .resolved_log_calls
        .iter()
        .find(|call| call.instruction_pc == instruction.pc);
    if let Some(call) = resolved_log {
        let rendered_value =
            serde_json::to_string(&call.value).unwrap_or_else(|_| "\"<invalid string>\"".into());
        if c_like {
            output.push_str(&format!(
                "{padding}/* 0x{:x}: {} — recovered {} pointer/length; evidence: ro_section/offset:0x{:x}, analysis/instruction:0x{:x} */\n",
                instruction.pc,
                instruction.assembly,
                call.syscall,
                call.string_offset,
                instruction.pc
            ));
        } else {
            output.push_str(&format!(
                "{padding}// 0x{:x}: {} — recovered {} pointer/length; evidence: ro_section/offset:0x{:x}, analysis/instruction:0x{:x}\n",
                instruction.pc,
                instruction.assembly,
                call.syscall,
                call.string_offset,
                instruction.pc
            ));
        }
        output.push_str(&format!("{padding}sol_log({rendered_value});\n"));
        return;
    }
    let runtime_error = semantics
        .runtime_error_candidates
        .iter()
        .find(|candidate| candidate.instruction_pc == instruction.pc);
    if let Some(candidate) = runtime_error {
        let comment_prefix = if c_like { "/*" } else { "//" };
        let comment_suffix = if c_like { " */" } else { "" };
        output.push_str(&format!(
            "{padding}{comment_prefix} 0x{:x}: {} — exact Solana built-in return code; evidence: analysis/instruction:0x{:x}{comment_suffix}\n",
            instruction.pc, instruction.assembly, instruction.pc
        ));
        output.push_str(&format!(
            "{padding}r{} = {};\n",
            instruction.dst, candidate.name
        ));
        return;
    }
    let validation_load = semantics
        .validation_checks
        .iter()
        .find(|check| check.load_pc == instruction.pc && check.account_field.is_some());
    if let Some(check) = validation_load {
        let account_index = check.account_index.unwrap_or_default();
        let field = check.account_field.as_deref().unwrap_or("unknown_flag");
        let variable = format!("account_{account_index}_{field}");
        let comment_prefix = if c_like { "/*" } else { "//" };
        let comment_suffix = if c_like { " */" } else { "" };
        output.push_str(&format!(
            "{padding}{comment_prefix} 0x{:x}: recovered SolAccountInfo stack field; evidence: analysis/instruction:0x{:x}{comment_suffix}\n",
            instruction.pc, instruction.pc
        ));
        if c_like {
            output.push_str(&format!(
                "{padding}bool {variable} = accounts[{account_index}].{field};\n"
            ));
        } else {
            output.push_str(&format!(
                "{padding}let {variable} = accounts[{account_index}].{field};\n"
            ));
        }
        return;
    }
    let validation_branch = semantics.validation_checks.iter().find(|check| {
        check.branch_pc == instruction.pc
            && check.account_field.is_some()
            && check.compared_value == Some(1)
    });
    if let Some(check) = validation_branch {
        let account_index = check.account_index.unwrap_or_default();
        let field = check.account_field.as_deref().unwrap_or("unknown_flag");
        let variable = format!("account_{account_index}_{field}");
        let failure = check
            .nearby_runtime_error
            .as_deref()
            .unwrap_or("UNKNOWN_GUARD_FAILURE");
        let comment_prefix = if c_like { "/*" } else { "//" };
        let comment_suffix = if c_like { " */" } else { "" };
        output.push_str(&format!(
            "{padding}{comment_prefix} 0x{:x}: recovered boolean account guard; evidence: analysis/instruction:0x{:x}{comment_suffix}\n",
            instruction.pc, instruction.pc
        ));
        output.push_str(&format!(
            "{padding}if !{variable} {{ return {failure}; }}\n"
        ));
        return;
    }
    let syscall = semantics
        .syscalls
        .iter()
        .find(|syscall| syscall.instruction_pc == instruction.pc);
    let memory = analysis
        .memory_accesses
        .iter()
        .find(|access| access.pc == instruction.pc);
    let comment = if let Some(syscall) = syscall {
        format!(
            "syscall {} — evidence: analysis/instruction:0x{:x}",
            syscall.name.as_deref().unwrap_or("unknown"),
            instruction.pc
        )
    } else if let Some(memory) = memory {
        format!(
            "{}{} width={} [r{} {:+}] — evidence: analysis/memory:r{}+{}",
            memory.kind,
            memory
                .value_register
                .map(|register| format!(" r{register}"))
                .unwrap_or_default(),
            memory.width,
            memory.base_register,
            memory.offset,
            memory.base_register,
            memory.offset
        )
    } else {
        format!(
            "raw SBPF operation — evidence: analysis/instruction:0x{:x}",
            instruction.pc
        )
    };
    if c_like {
        output.push_str(&format!(
            "{padding}/* 0x{:x}: {} — {} */\n",
            instruction.pc, instruction.assembly, comment
        ));
        output.push_str(&format!(
            "{padding}sos_raw_op(0x{:x}, {}, {}, {}, {});\n",
            instruction.pc,
            instruction.dst,
            instruction.src,
            instruction.offset,
            instruction.immediate
        ));
    } else {
        output.push_str(&format!(
            "{padding}// 0x{:x}: {} — {}\n",
            instruction.pc, instruction.assembly, comment
        ));
        output.push_str(&format!(
            "{padding}sos_raw_op(0x{:x}, {}, {}, {}, {});\n",
            instruction.pc,
            instruction.dst,
            instruction.src,
            instruction.offset,
            instruction.immediate
        ));
    }
}

pub fn render_cfg_dot(analysis: &StaticAnalysis) -> String {
    let mut output = String::from("digraph sos_cfg {\n  graph [fontname=monospace];\n  node [shape=box,fontname=monospace];\n");
    for block in &analysis.basic_blocks {
        let label = format!("0x{:x}: {}", block.start_pc, block.label).replace('"', "\\\"");
        output.push_str(&format!(
            "  block_{:x} [label=\"{}\"];\n",
            block.start_pc, label
        ));
        for destination in &block.destinations {
            output.push_str(&format!(
                "  block_{:x} -> block_{:x};\n",
                block.start_pc, destination
            ));
        }
    }
    output.push_str("}\n");
    output
}

fn function_suffix(function: &FunctionFact) -> String {
    if function.name.is_empty() {
        String::new()
    } else {
        format!(" (symbol `{}`)", function.name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sos_ai::{AiStageResult, ModelResponse, ModelTask, AI_BUNDLE_SCHEMA_VERSION};
    use sos_types::{EvidenceRef, EvidenceStatus};

    #[test]
    fn sarif_preserves_finding_identity_and_evidence_metadata() {
        let report = SecurityReport {
            schema_version: AI_BUNDLE_SCHEMA_VERSION,
            findings: vec![sos_solana::SecurityFinding {
                detector_id: "test_detector".into(),
                severity: "high".into(),
                confidence: 0.8,
                title: "Test finding".into(),
                description: "Review this path".into(),
                instruction_pcs: vec![0x40],
                status: EvidenceStatus::Hypothesis,
                evidence: vec![EvidenceRef {
                    source: "analysis".into(),
                    location: "instruction:0x40".into(),
                    detail: None,
                }],
                limitations: vec!["static only".into()],
            }],
        };
        let value: serde_json::Value = serde_json::from_str(&render_sarif(&report)).unwrap();
        assert_eq!(value["version"], "2.1.0");
        assert_eq!(value["runs"][0]["results"][0]["ruleId"], "test_detector");
        let confidence = value["runs"][0]["results"][0]["properties"]["confidence"]
            .as_f64()
            .unwrap();
        assert!((confidence - 0.8).abs() < 0.001);
    }

    #[test]
    fn ai_report_keeps_model_output_labeled_and_audited() {
        let bundle = AiPipelineBundle {
            schema_version: 1,
            artifact_sha256: Some("artifact".into()),
            provider: "test".into(),
            model: "model".into(),
            stages: vec![AiStageResult {
                task: ModelTask::Explanation,
                status: AiStageStatus::Accepted,
                model: "model".into(),
                max_output_tokens: 640,
                prompt_version: "sos-v1".into(),
                input_sha256: "a".repeat(64),
                output_sha256: Some("b".repeat(64)),
                response: Some(ModelResponse {
                    task: ModelTask::Explanation,
                    model: "model".into(),
                    prompt_version: "sos-v1".into(),
                    structured_output: serde_json::json!({
                        "summary": "observed",
                        "observations": ["candidate"]
                    }),
                    evidence: vec![],
                }),
                error: None,
            }],
        };
        let report = render_ai_markdown(&bundle);
        assert!(report.contains("untrusted model output"));
        assert!(report.contains("Explanation (hypothesis)"));
        assert!(report.contains("Output tokens"));
        assert!(report.contains("`640`"));
        assert!(report.contains("observed"));
        assert!(report.contains(&"a".repeat(64)));
    }

    #[test]
    fn readable_rust_prefers_refinement_and_marks_missing_functions() {
        let analysis = StaticAnalysis {
            schema_version: AI_BUNDLE_SCHEMA_VERSION,
            sbpf_version: "V0".into(),
            entrypoint: 0,
            instructions: vec![],
            functions: vec![
                FunctionFact {
                    pc: 0,
                    symbol_hash: 0,
                    name: "entrypoint".into(),
                },
                FunctionFact {
                    pc: 8,
                    symbol_hash: 0,
                    name: "helper".into(),
                },
            ],
            basic_blocks: vec![],
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
        let evidence = vec![EvidenceRef {
            source: "ir".into(),
            location: "function:fn_0".into(),
            detail: None,
        }];
        let bundle = AiPipelineBundle {
            schema_version: 1,
            artifact_sha256: None,
            provider: "ollama".into(),
            model: "gpt-oss:20b".into(),
            stages: vec![
                AiStageResult {
                    task: ModelTask::StructureRecovery,
                    status: AiStageStatus::Accepted,
                    model: "gpt-oss:20b".into(),
                    max_output_tokens: 1_200,
                    prompt_version: "sos-v2".into(),
                    input_sha256: "a".repeat(64),
                    output_sha256: Some("b".repeat(64)),
                    response: Some(ModelResponse {
                        task: ModelTask::StructureRecovery,
                        model: "gpt-oss:20b".into(),
                        prompt_version: "sos-v2".into(),
                        structured_output: serde_json::json!({"functions": [{
                            "function_id": "fn_0", "pseudocode": "fn old_name() {}",
                            "evidence": evidence
                        }]}),
                        evidence: vec![],
                    }),
                    error: None,
                },
                AiStageResult {
                    task: ModelTask::Refinement,
                    status: AiStageStatus::Accepted,
                    model: "gpt-oss:20b".into(),
                    max_output_tokens: 1_200,
                    prompt_version: "sos-v2".into(),
                    input_sha256: "c".repeat(64),
                    output_sha256: Some("d".repeat(64)),
                    response: Some(ModelResponse {
                        task: ModelTask::Refinement,
                        model: "gpt-oss:20b".into(),
                        prompt_version: "sos-v2".into(),
                        structured_output: serde_json::json!({
                            "schema_version": 1,
                            "patches": [{"function_id": "fn_0", "replacement": "```rust\nfn process_instruction() {}\n```", "evidence": evidence}],
                            "observations": [],
                            "validation": {"status": "unverified", "notes": []}
                        }),
                        evidence: vec![],
                    }),
                    error: None,
                },
            ],
        };
        let source = render_readable_rust(&analysis, &bundle);
        assert!(source.contains("fn process_instruction() {}"));
        assert!(!source.contains("old_name"));
        assert!(source.contains("unresolved: fn_8"));
    }
}
