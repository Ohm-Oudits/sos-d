use std::{
    collections::BTreeMap,
    env, fs,
    path::{Path, PathBuf},
    process::ExitCode,
    time::Duration,
};

use sos_ai::{
    compile_refinement_output, evidence_pack_from_pseudocode_unit, partition_pseudocode,
    run_pipeline, run_refinement_loop, scope_analysis_to_function, trusted_open_model,
    unavailable_pipeline_bundle, AiPipelineBundle, AiPipelineConfig, AiStageResult, AiStageStatus,
    CommandModelProvider, CompilationStatus, EvidencePack, EvidencePackLimits, GroqModelProvider,
    HttpModelProvider, ModelGateway, ModelProvider, ModelResponse, ModelTask, OllamaModelProvider,
    PseudocodeUnit, RefinementFeedback, RefinementLoopResult, RefinementOutput,
    RefinementValidation, StructureRecoveryOutput, UntrustedEvidence,
    VALIDATION_BUNDLE_SCHEMA_VERSION,
};
use sos_analysis::{analyze_elf, build_canonical_ir};
use sos_artifact::{
    build_manifest, local_artifact, sha256_hex, FileArtifactStore, ManifestContext,
};
use sos_core::read_elf_facts;
use sos_loader::default_registry;
#[cfg(feature = "litesvm")]
use sos_replay::LiteSvmReplayEngine;
#[cfg(not(feature = "litesvm"))]
use sos_replay::UnavailableReplayEngine;
use sos_replay::{compare_candidate_observations, compare_semantics, ExecutionTrace, ReplayEngine};
use sos_report::{
    render_ai_markdown, render_c_pseudocode, render_cfg_dot, render_markdown, render_pseudocode,
    render_readable_rust, render_sarif,
};
use sos_rpc::{ChainProvider, JsonRpcProvider};
use sos_solana::analyze as analyze_semantics;
use sos_storage::MetadataStore;
use sos_types::{ArtifactManifest, ProgramArtifact};

fn main() -> ExitCode {
    let mut args = env::args().skip(1);
    let command = args.next().unwrap_or_default();

    match command.as_str() {
        "artifact-info" => artifact_info_command(args.collect()),
        "manifest" => match args.next() {
            Some(path) => match local_artifact(&path, "local-artifact") {
                Ok((_artifact, manifest)) => match serde_json::to_string_pretty(&manifest) {
                    Ok(json) => {
                        println!("{json}");
                        ExitCode::SUCCESS
                    }
                    Err(error) => {
                        eprintln!("error: {error}");
                        ExitCode::from(1)
                    }
                },
                Err(error) => {
                    eprintln!("error: {error}");
                    ExitCode::from(1)
                }
            },
            None => usage(),
        },
        "fetch" => fetch_command(args.collect()),
        "inspect" => inspect_command(args.collect()),
        "analyze" | "analyze-elf" => analyze_elf_command(args.collect()),
        "detect" => detect_command(args.collect()),
        "validate" => validate_command(args.collect()),
        "explain" => explain_command(args.collect()),
        "replay" => replay_command(args.collect()),
        "report" => report_command(args.collect()),
        "decompile" => decompile_command(args.collect()),
        "lift" => lift_command(args.collect()),
        "humanize" => humanize_command(args.collect()),
        "benchmark" => benchmark_command(args.collect()),
        _ => usage(),
    }
}

fn benchmark_command(args: Vec<String>) -> ExitCode {
    let options = match BenchmarkOptions::parse(args) {
        Ok(options) => options,
        Err(error) => {
            eprintln!("error: {error}");
            return ExitCode::from(2);
        }
    };
    let bytes = match fs::read(&options.artifact) {
        Ok(bytes) => bytes,
        Err(error) => {
            eprintln!("error: {}: {error}", options.artifact.display());
            return ExitCode::from(1);
        }
    };

    let mut baseline_json = None;
    let mut analysis = None;
    let mut deterministic = true;
    for _ in 0..options.repeats {
        let current = match analyze_elf(&bytes) {
            Ok(analysis) => analysis,
            Err(error) => {
                eprintln!("error: {error}");
                return ExitCode::from(1);
            }
        };
        let current_json = match serde_json::to_vec(&current) {
            Ok(json) => json,
            Err(error) => {
                eprintln!("error: {error}");
                return ExitCode::from(1);
            }
        };
        if let Some(baseline) = &baseline_json {
            deterministic &= baseline == &current_json;
        } else {
            baseline_json = Some(current_json);
        }
        analysis = Some(current);
    }
    let analysis = analysis.expect("benchmark repeats is at least one");
    let semantics = analyze_semantics(&analysis);
    let security = sos_solana::detect(&analysis, &semantics);
    let analysis_json = match serde_json::to_vec(&analysis) {
        Ok(json) => json,
        Err(error) => {
            eprintln!("error: {error}");
            return ExitCode::from(1);
        }
    };
    let result = serde_json::json!({
        "schema_version": 1,
        "tool_version": env!("CARGO_PKG_VERSION"),
        "artifact": options.artifact,
        "artifact_sha256": sha256_hex(&bytes),
        "analysis_sha256": sha256_hex(&analysis_json),
        "repeats": options.repeats,
        "deterministic": deterministic,
        "binary": {
            "sbpf_version": analysis.sbpf_version,
            "entrypoint": analysis.entrypoint,
            "instructions": analysis.instructions.len(),
            "functions": analysis.functions.len(),
            "basic_blocks": analysis.basic_blocks.len(),
            "calls": analysis.calls.len(),
            "memory_accesses": analysis.memory_accesses.len(),
            "stack_accesses": analysis.stack_accesses.len(),
            "strings": analysis.strings.len(),
            "public_key_candidates": analysis.public_key_candidates.len(),
            "relocations": analysis.relocations.len(),
            "sccs": analysis.sccs.len(),
            "dataflow_edges": analysis.dataflow_edges.len(),
            "verifier_diagnostics": analysis.verifier_diagnostics.len(),
        },
        "semantic": {
            "syscalls": semantics.syscalls.len(),
            "instruction_candidates": semantics.instruction_candidates.len(),
            "anchor_global_discriminator_candidates": semantics.dispatch_candidates.iter()
                .filter(|candidate| candidate.interpretations.iter().any(|value| value.starts_with("anchor_global_discriminator:")))
                .count(),
            "anchor_account_discriminator_candidates": semantics.constant_candidates.iter()
                .filter(|candidate| candidate.interpretations.iter().any(|value| value == "anchor_account_discriminator_candidate"))
                .count(),
            "argument_candidates": semantics.argument_candidates.len(),
            "account_candidates": semantics.account_candidates.len(),
            "account_field_candidates": semantics.account_field_candidates.len(),
            "cpi_candidates": semantics.cpi_candidates.len(),
            "pda_candidates": semantics.pda_candidates.len(),
            "protocol_fingerprints": semantics.protocol_fingerprints.len(),
            "validation_checks": semantics.validation_checks.len(),
            "log_candidates": semantics.log_candidates.len(),
            "error_candidates": semantics.error_candidates.len(),
        },
        "security": {
            "findings": security.findings.len(),
        },
    });
    let json = match serde_json::to_string_pretty(&result) {
        Ok(json) => json,
        Err(error) => {
            eprintln!("error: {error}");
            return ExitCode::from(1);
        }
    };
    if let Some(path) = options.output {
        if let Err(error) = fs::write(&path, json) {
            eprintln!("error: {}: {error}", path.display());
            return ExitCode::from(1);
        }
        println!("benchmark={}", path.display());
    } else {
        println!("{json}");
    }
    ExitCode::SUCCESS
}

fn analyze_elf_command(args: Vec<String>) -> ExitCode {
    let Some(path) = args.first() else {
        return usage();
    };
    let output = match parse_output_path(&args) {
        Ok(path) => path,
        Err(error) => {
            eprintln!("error: {error}");
            return ExitCode::from(2);
        }
    };
    let json_requested = args.iter().skip(1).any(|argument| argument == "--json");
    let bytes = match std::fs::read(path) {
        Ok(bytes) => bytes,
        Err(error) => {
            eprintln!("error: {error}");
            return ExitCode::from(1);
        }
    };
    match analyze_elf(&bytes) {
        Ok(analysis) => {
            let Some(output) = output else {
                return match serde_json::to_string_pretty(&analysis) {
                    Ok(json) => {
                        println!("{json}");
                        ExitCode::SUCCESS
                    }
                    Err(error) => {
                        eprintln!("error: {error}");
                        ExitCode::from(1)
                    }
                };
            };
            if let Err(error) = fs::create_dir_all(&output) {
                eprintln!("error: {}: {error}", output.display());
                return ExitCode::from(1);
            }
            let analysis_path = output.join("analysis.json");
            let semantics = analyze_semantics(&analysis);
            let ir = build_canonical_ir(&analysis);
            let security = sos_solana::detect(&analysis, &semantics);
            let writes = [
                (
                    analysis_path,
                    serde_json::to_vec_pretty(&analysis).map_err(|error| error.to_string()),
                ),
                (
                    output.join("semantics.json"),
                    serde_json::to_vec_pretty(&semantics).map_err(|error| error.to_string()),
                ),
                (
                    output.join("ir.json"),
                    serde_json::to_vec_pretty(&ir).map_err(|error| error.to_string()),
                ),
                (
                    output.join("security.json"),
                    serde_json::to_vec_pretty(&security).map_err(|error| error.to_string()),
                ),
            ];
            for (path, bytes) in writes {
                let bytes = match bytes {
                    Ok(bytes) => bytes,
                    Err(error) => {
                        eprintln!("error: {error}");
                        return ExitCode::from(1);
                    }
                };
                if let Err(error) = fs::write(&path, bytes) {
                    eprintln!("error: {}: {error}", path.display());
                    return ExitCode::from(1);
                }
            }
            if let Err(error) = fs::write(
                output.join("report.md"),
                render_markdown(&analysis, &semantics),
            ) {
                eprintln!("error: {error}");
                return ExitCode::from(1);
            }
            if let Err(error) = fs::write(
                output.join("pseudocode.rs"),
                render_pseudocode(&analysis, &semantics),
            ) {
                eprintln!("error: {error}");
                return ExitCode::from(1);
            }
            if let Err(error) = fs::write(
                output.join("pseudocode.c"),
                render_c_pseudocode(&analysis, &semantics),
            ) {
                eprintln!("error: {error}");
                return ExitCode::from(1);
            }
            if json_requested {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&serde_json::json!({
                        "analysis": output.join("analysis.json"),
                        "semantics": output.join("semantics.json"),
                        "ir": output.join("ir.json"),
                        "security": output.join("security.json"),
                        "report": output.join("report.md"),
                        "pseudocode": output.join("pseudocode.rs"),
                        "pseudocode_c": output.join("pseudocode.c"),
                    }))
                    .unwrap_or_else(|error| format!("{{\"error\":\"{error}\"}}"))
                );
            } else {
                println!("analysis={}", output.join("analysis.json").display());
                println!("semantics={}", output.join("semantics.json").display());
                println!("ir={}", output.join("ir.json").display());
                println!("security={}", output.join("security.json").display());
                println!("report={}", output.join("report.md").display());
                println!("pseudocode={}", output.join("pseudocode.rs").display());
                println!("pseudocode_c={}", output.join("pseudocode.c").display());
            }
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::from(1)
        }
    }
}

fn artifact_info_command(args: Vec<String>) -> ExitCode {
    let Some(path) = args.first() else {
        return usage();
    };
    let json = args.iter().skip(1).any(|argument| argument == "--json");
    if let Some(argument) = args
        .iter()
        .skip(1)
        .find(|argument| argument.as_str() != "--json")
    {
        eprintln!("error: unknown artifact-info option: {argument}");
        return ExitCode::from(2);
    }
    match read_elf_facts(path) {
        Ok(facts) if json => match serde_json::to_string_pretty(&serde_json::json!({
            "path": path,
            "machine": facts.machine,
            "sbpf": facts.is_sbpf(),
            "entrypoint": facts.entrypoint,
        })) {
            Ok(value) => {
                println!("{value}");
                ExitCode::SUCCESS
            }
            Err(error) => {
                eprintln!("error: {error}");
                ExitCode::from(1)
            }
        },
        Ok(facts) => {
            println!("path={path}");
            println!("machine=0x{:x}", facts.machine);
            println!("sbpf={}", facts.is_sbpf());
            println!("entrypoint=0x{:x}", facts.entrypoint);
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::from(1)
        }
    }
}

fn detect_command(args: Vec<String>) -> ExitCode {
    let Some(directory) = args.first() else {
        eprintln!("usage: sos detect <analysis-directory> [--format json|sarif] [--out PATH]");
        return ExitCode::from(2);
    };
    let (output, format) = match parse_detect_options(&args) {
        Ok(path) => path,
        Err(error) => {
            eprintln!("error: {error}");
            return ExitCode::from(2);
        }
    };
    let directory = PathBuf::from(directory);
    let analysis = match read_json::<sos_analysis::StaticAnalysis>(&directory.join("analysis.json"))
    {
        Ok(value) => value,
        Err(error) => {
            eprintln!("error: {error}");
            return ExitCode::from(1);
        }
    };
    let semantics = read_json::<sos_solana::SemanticAnalysis>(&directory.join("semantics.json"))
        .unwrap_or_else(|_| analyze_semantics(&analysis));
    let report = sos_solana::detect(&analysis, &semantics);
    let rendered = if format == "sarif" {
        render_sarif(&report)
    } else {
        match serde_json::to_string_pretty(&report) {
            Ok(json) => json,
            Err(error) => {
                eprintln!("error: {error}");
                return ExitCode::from(1);
            }
        }
    };
    if let Some(path) = output {
        if let Err(error) = fs::write(&path, rendered) {
            eprintln!("error: {}: {error}", path.display());
            return ExitCode::from(1);
        }
        println!("security={}", path.display());
    } else {
        println!("{rendered}");
    }
    ExitCode::SUCCESS
}

fn parse_detect_options(args: &[String]) -> Result<(Option<PathBuf>, String), String> {
    let mut index = 1;
    let mut output = None;
    let mut format = "json".to_owned();
    while index < args.len() {
        match args[index].as_str() {
            "--json" => {
                format = "json".into();
                index += 1;
            }
            "--format" => {
                let Some(value) = args.get(index + 1) else {
                    return Err("missing value for --format".into());
                };
                if !matches!(value.as_str(), "json" | "sarif") {
                    return Err(format!("unsupported detect format: {value}"));
                }
                format = value.clone();
                index += 2;
            }
            "--out" => {
                let Some(path) = args.get(index + 1) else {
                    return Err("missing value for --out".into());
                };
                output = Some(PathBuf::from(path));
                index += 2;
            }
            option => return Err(format!("unknown detect option: {option}")),
        }
    }
    Ok((output, format))
}

fn validate_command(args: Vec<String>) -> ExitCode {
    validate_command_inner(args, true)
}

fn validate_command_inner(args: Vec<String>, announce_output: bool) -> ExitCode {
    let Some(directory) = args.first() else {
        eprintln!(
            "usage: sos validate <analysis-directory> [--compiler PATH|--no-compile] [--trace PATH] [--repair --model-command PATH --repair-rounds N] [--out PATH]"
        );
        return ExitCode::from(2);
    };
    let options = match parse_validate_options(&args) {
        Ok(options) => options,
        Err(error) => {
            eprintln!("error: {error}");
            return ExitCode::from(2);
        }
    };
    let ValidateOptions {
        compiler,
        trace_path,
        output,
        repair,
        model_command,
        repair_rounds,
    } = options;
    let directory = PathBuf::from(directory);
    let bundle = match read_json::<AiPipelineBundle>(&directory.join("model.json")) {
        Ok(bundle) => bundle,
        Err(error) => {
            eprintln!("error: {error}");
            return ExitCode::from(1);
        }
    };
    let mut refinement = bundle
        .stages
        .iter()
        .find(|stage| stage.task == ModelTask::Refinement && stage.response.is_some())
        .and_then(|stage| stage.response.as_ref())
        .and_then(|response| {
            serde_json::from_value::<RefinementOutput>(response.structured_output.clone()).ok()
        });
    let mut repair_result: Option<RefinementLoopResult> = None;
    if repair {
        let Some(model_command) = model_command else {
            eprintln!("error: --repair requires --model-command PATH");
            return ExitCode::from(2);
        };
        let analysis =
            match read_json::<sos_analysis::StaticAnalysis>(&directory.join("analysis.json")) {
                Ok(analysis) => analysis,
                Err(error) => {
                    eprintln!("error: {error}");
                    return ExitCode::from(1);
                }
            };
        let semantics =
            match read_json::<sos_solana::SemanticAnalysis>(&directory.join("semantics.json")) {
                Ok(semantics) => semantics,
                Err(error) => {
                    eprintln!("error: {error}");
                    return ExitCode::from(1);
                }
            };
        let pack = EvidencePack::bounded_from_analysis(
            ModelTask::Refinement,
            bundle.artifact_sha256.clone(),
            &analysis,
            &semantics,
            EvidencePackLimits::default(),
        );
        let repair_trace = trace_path
            .as_deref()
            .and_then(|path| read_json::<ExecutionTrace>(path).ok());
        let repair_compiler = compiler.clone();
        let loop_result = run_refinement_loop(
            CommandModelProvider::new(model_command).with_timeout(Duration::from_secs(60)),
            pack,
            bundle.model.clone(),
            bundle
                .stages
                .iter()
                .find(|stage| stage.task == ModelTask::Refinement)
                .map(|stage| stage.prompt_version.clone())
                .unwrap_or_else(|| "sos-v1".into()),
            2,
            repair_rounds,
            move |candidate| {
                let compilation = compile_refinement_output(
                    candidate,
                    repair_compiler.as_deref(),
                    Duration::from_secs(30),
                );
                let (behavioral_status, mismatches, limitations) = match repair_trace.as_ref() {
                    Some(trace) if !candidate.observations.is_empty() => {
                        let comparison =
                            compare_candidate_observations(&candidate.observations, trace);
                        let status = match comparison.status {
                            sos_types::EvidenceStatus::BehaviorallyValidated => {
                                "behaviorally_validated"
                            }
                            sos_types::EvidenceStatus::NotAvailable => "not_available",
                            _ => "unverified",
                        };
                        (status.into(), comparison.mismatches, comparison.limitations)
                    }
                    Some(_) => (
                        "unverified".into(),
                        Vec::new(),
                        vec!["candidate supplied no directly comparable observations".into()],
                    ),
                    None => (
                        "not_requested".into(),
                        Vec::new(),
                        vec!["candidate replay comparison was not requested".into()],
                    ),
                };
                RefinementFeedback {
                    compilation: compilation.status,
                    compiler_diagnostics: compilation.diagnostics,
                    behavioral_status,
                    mismatches,
                    limitations,
                }
            },
        );
        if let Some(final_output) = loop_result.final_output.clone() {
            refinement = Some(final_output);
        }
        repair_result = Some(loop_result);
    }
    let empty_refinement = RefinementOutput {
        schema_version: sos_ai::REFINEMENT_SCHEMA_VERSION,
        patches: Vec::new(),
        observations: Vec::new(),
        validation: RefinementValidation {
            status: "not_available".into(),
            notes: Vec::new(),
        },
    };
    let compilation = compile_refinement_output(
        refinement.as_ref().unwrap_or(&empty_refinement),
        compiler.as_deref(),
        Duration::from_secs(30),
    );
    let (behavioral_status, behavioral_evidence, behavioral_mismatches, behavioral_limitations) =
        match trace_path.as_deref() {
            Some(path) => match read_json::<ExecutionTrace>(path) {
                Ok(trace) => {
                    let candidate_comparison = refinement.as_ref().and_then(|candidate| {
                        (!candidate.observations.is_empty()).then(|| {
                            compare_candidate_observations(&candidate.observations, &trace)
                        })
                    });
                    match candidate_comparison.or_else(|| trace.behavioral_comparison.clone()) {
                        Some(comparison) => (
                            match comparison.status {
                                sos_types::EvidenceStatus::BehaviorallyValidated => {
                                    "behaviorally_validated"
                                }
                                sos_types::EvidenceStatus::NotAvailable => "not_available",
                                _ if comparison.mismatches.is_empty() => "unverified",
                                _ => "mismatch",
                            },
                            comparison.matched_evidence,
                            comparison.mismatches,
                            comparison.limitations,
                        ),
                        None => (
                            "unverified",
                            Vec::new(),
                            Vec::new(),
                            vec!["trace did not contain a behavioral comparison".into()],
                        ),
                    }
                }
                Err(error) => (
                    "not_available",
                    Vec::new(),
                    Vec::new(),
                    vec![format!("unable to read replay trace: {error}")],
                ),
            },
            None => (
                "not_requested",
                Vec::new(),
                Vec::new(),
                vec!["candidate replay comparison was not requested".into()],
            ),
        };
    let refinement_output_sha256 = if repair {
        refinement.as_ref().and_then(|candidate| {
            serde_json::to_vec(candidate)
                .ok()
                .map(|bytes| sha256_hex(&bytes))
        })
    } else {
        bundle
            .stages
            .iter()
            .find(|stage| stage.task == ModelTask::Refinement)
            .and_then(|stage| stage.output_sha256.clone())
    };
    let value = serde_json::json!({
        "schema_version": VALIDATION_BUNDLE_SCHEMA_VERSION,
        "artifact_sha256": bundle.artifact_sha256,
        "refinement_output_sha256": refinement_output_sha256,
        "compilation": compilation,
        "behavioral_status": behavioral_status,
        "behavioral_evidence": behavioral_evidence,
        "behavioral_mismatches": behavioral_mismatches,
        "repair": repair_result,
        "limitations": behavioral_limitations.into_iter().chain([
            "successful compilation does not establish behavioral agreement".into(),
            "trace comparison describes observed deterministic semantics and does not execute the generated candidate".into()
        ]).collect::<Vec<_>>()
    });
    let json = match serde_json::to_string_pretty(&value) {
        Ok(json) => json,
        Err(error) => {
            eprintln!("error: {error}");
            return ExitCode::from(1);
        }
    };
    if let Some(path) = output {
        if let Err(error) = fs::write(&path, json) {
            eprintln!("error: {}: {error}", path.display());
            return ExitCode::from(1);
        }
        if announce_output {
            println!("validation={}", path.display());
        }
    } else {
        println!("{json}");
    }
    ExitCode::SUCCESS
}

struct ValidateOptions {
    compiler: Option<PathBuf>,
    trace_path: Option<PathBuf>,
    output: Option<PathBuf>,
    repair: bool,
    model_command: Option<PathBuf>,
    repair_rounds: usize,
}

fn parse_validate_options(args: &[String]) -> Result<ValidateOptions, String> {
    let mut compiler = Some(PathBuf::from("rustc"));
    let mut trace = None;
    let mut output = None;
    let mut repair = false;
    let mut model_command = None;
    let mut repair_rounds = 2;
    let mut index = 1;
    while index < args.len() {
        match args[index].as_str() {
            "--no-compile" => {
                compiler = None;
                index += 1;
            }
            "--compiler" => {
                let Some(path) = args.get(index + 1) else {
                    return Err("missing value for --compiler".into());
                };
                compiler = Some(PathBuf::from(path));
                index += 2;
            }
            "--trace" => {
                let Some(path) = args.get(index + 1) else {
                    return Err("missing value for --trace".into());
                };
                trace = Some(PathBuf::from(path));
                index += 2;
            }
            "--repair" => {
                repair = true;
                index += 1;
            }
            "--model-command" => {
                let Some(path) = args.get(index + 1) else {
                    return Err("missing value for --model-command".into());
                };
                model_command = Some(PathBuf::from(path));
                index += 2;
            }
            "--repair-rounds" => {
                let Some(value) = args.get(index + 1) else {
                    return Err("missing value for --repair-rounds".into());
                };
                repair_rounds = value
                    .parse::<usize>()
                    .map_err(|_| "--repair-rounds must be an integer".to_string())?
                    .clamp(1, 8);
                index += 2;
            }
            "--out" => {
                let Some(path) = args.get(index + 1) else {
                    return Err("missing value for --out".into());
                };
                output = Some(PathBuf::from(path));
                index += 2;
            }
            option => return Err(format!("unknown validate option: {option}")),
        }
    }
    Ok(ValidateOptions {
        compiler,
        trace_path: trace,
        output,
        repair,
        model_command,
        repair_rounds,
    })
}

fn explain_command(args: Vec<String>) -> ExitCode {
    let options = match ExplainOptions::parse(args) {
        Ok(options) => options,
        Err(error) => {
            eprintln!("error: {error}");
            return ExitCode::from(2);
        }
    };
    let response = match run_model(
        &options.directory,
        options.model_command.as_deref(),
        options.model_endpoint.as_deref(),
        &options.model,
        options.task,
    ) {
        Ok(response) => response,
        Err(error) => {
            eprintln!("error: {error}");
            return ExitCode::from(1);
        }
    };
    let json = match serde_json::to_string_pretty(&response) {
        Ok(json) => json,
        Err(error) => {
            eprintln!("error: {error}");
            return ExitCode::from(1);
        }
    };
    if let Some(path) = options.output {
        if let Err(error) = fs::write(&path, json) {
            eprintln!("error: {}: {error}", path.display());
            return ExitCode::from(1);
        }
        println!("explanation={}", path.display());
    } else {
        println!("{json}");
    }
    ExitCode::SUCCESS
}

fn replay_command(args: Vec<String>) -> ExitCode {
    let options = match ReplayOptions::parse(args) {
        Ok(options) => options,
        Err(error) => {
            eprintln!("error: {error}");
            return ExitCode::from(2);
        }
    };
    let provider = JsonRpcProvider::new(&options.rpc_url).with_commitment(&options.commitment);
    let account = match provider.get_account(&options.program_address, options.requested_slot) {
        Ok(account) => account,
        Err(error) => {
            eprintln!("error: {error}");
            return ExitCode::from(1);
        }
    };
    let registry = default_registry();
    let adapter = match registry.identify(&account) {
        Some(adapter) => adapter,
        None => {
            eprintln!("error: unsupported program owner {}", account.owner);
            return ExitCode::from(1);
        }
    };
    let artifact = match adapter.resolve_artifact(&account, &provider) {
        Ok(artifact) => artifact,
        Err(error) => {
            eprintln!("error: {error}");
            return ExitCode::from(1);
        }
    };
    let transaction = match provider.get_transaction(&options.signature, options.requested_slot) {
        Ok(transaction) => transaction,
        Err(error) => {
            eprintln!("error: {error}");
            return ExitCode::from(1);
        }
    };
    let addresses = transaction_account_addresses(&transaction.raw_json);
    let mut accounts = Vec::with_capacity(addresses.len());
    for address in addresses {
        match provider.get_account(&address, transaction.slot.or(options.requested_slot)) {
            Ok(account) => accounts.push(account),
            Err(error) => {
                let trace = sos_replay::ExecutionTrace::not_available_for_transaction(
                    &transaction,
                    &accounts,
                    format!(
                        "historical account state unavailable for {address}: {error}; \
                         replay was not attempted and current state was not substituted"
                    ),
                );
                return emit_trace(trace, options.output.as_deref());
            }
        }
    }
    let mut trace = match replay_transaction(&artifact, &transaction, &accounts) {
        Ok(trace) => trace,
        Err(error) => {
            eprintln!("error: {error}");
            return ExitCode::from(1);
        }
    };
    if let Ok(analysis) = analyze_elf(&artifact.elf_bytes) {
        let semantics = analyze_semantics(&analysis);
        trace.behavioral_comparison = Some(compare_semantics(&semantics, &trace));
    }
    emit_trace(trace, options.output.as_deref())
}

fn emit_trace(trace: sos_replay::ExecutionTrace, output: Option<&Path>) -> ExitCode {
    let json = match serde_json::to_string_pretty(&trace) {
        Ok(json) => json,
        Err(error) => {
            eprintln!("error: {error}");
            return ExitCode::from(1);
        }
    };
    if let Some(path) = output {
        if let Err(error) = fs::write(path, json) {
            eprintln!("error: {}: {error}", path.display());
            return ExitCode::from(1);
        }
        println!("trace={}", path.display());
    } else {
        println!("{json}");
    }
    ExitCode::SUCCESS
}

fn fetch_command(args: Vec<String>) -> ExitCode {
    let options = match RemoteOptions::parse(args, "fetch") {
        Ok(options) => options,
        Err(error) => {
            eprintln!("error: {error}");
            return ExitCode::from(2);
        }
    };
    let (artifact, manifest) = match resolve_remote(&options) {
        Ok(result) => result,
        Err(error) => {
            eprintln!("error: {error}");
            return ExitCode::from(1);
        }
    };
    let store = FileArtifactStore::new(&options.output);
    let artifact_path = match store.put(&artifact, &manifest) {
        Ok(path) => path,
        Err(error) => {
            eprintln!("error: {error}");
            return ExitCode::from(1);
        }
    };
    let (artifact_id, metadata_path) =
        match register_artifact_metadata(&options.output, &manifest, &artifact_path) {
            Ok(value) => value,
            Err(error) => {
                eprintln!("error: {error}");
                return ExitCode::from(1);
            }
        };
    let manifest_path = PathBuf::from(&options.output).join("manifest.json");
    if let Err(error) = fs::write(
        &manifest_path,
        serde_json::to_vec_pretty(&manifest).expect("artifact manifest is serializable"),
    ) {
        eprintln!("error: {error}");
        return ExitCode::from(1);
    }
    if options.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "artifact": artifact_path,
                "artifact_id": artifact_id,
                "metadata": metadata_path,
                "manifest": manifest_path,
                "elf_sha256": manifest.elf_sha256,
                "loader": manifest.loader,
                "context_slot": manifest.context_slot,
            }))
            .unwrap_or_else(|error| format!("{{\"error\":\"{error}\"}}"))
        );
    } else {
        println!("artifact={}", artifact_path.display());
        println!("artifact_id={artifact_id}");
        println!("metadata={}", metadata_path.display());
        println!("manifest={}", manifest_path.display());
        println!("elf_sha256={}", manifest.elf_sha256);
        println!("loader={:?}", manifest.loader);
        println!("context_slot={:?}", manifest.context_slot);
    }
    ExitCode::SUCCESS
}

fn inspect_command(args: Vec<String>) -> ExitCode {
    let options = match RemoteOptions::parse(args, "inspect") {
        Ok(options) => options,
        Err(error) => {
            eprintln!("error: {error}");
            return ExitCode::from(2);
        }
    };
    if (options.validate_requested || options.repair_requested) && options.model.is_none() {
        eprintln!("error: inspect --validate/--repair requires --model");
        return ExitCode::from(2);
    }
    let (artifact, manifest) = match resolve_remote(&options) {
        Ok(result) => result,
        Err(error) => {
            eprintln!("error: {error}");
            return ExitCode::from(1);
        }
    };
    let store = FileArtifactStore::new(&options.output);
    let artifact_path = match store.put(&artifact, &manifest) {
        Ok(path) => path,
        Err(error) => {
            eprintln!("error: {error}");
            return ExitCode::from(1);
        }
    };
    let (artifact_id, metadata_path) =
        match register_artifact_metadata(&options.output, &manifest, &artifact_path) {
            Ok(value) => value,
            Err(error) => {
                eprintln!("error: {error}");
                return ExitCode::from(1);
            }
        };
    let manifest_path = PathBuf::from(&options.output).join("manifest.json");
    if let Err(error) = fs::write(
        &manifest_path,
        serde_json::to_vec_pretty(&manifest).expect("artifact manifest is serializable"),
    ) {
        eprintln!("error: {error}");
        return ExitCode::from(1);
    }
    let analysis = match analyze_elf(&artifact.elf_bytes) {
        Ok(analysis) => analysis,
        Err(error) => {
            eprintln!("error: {error}");
            return ExitCode::from(1);
        }
    };
    let analysis_path = PathBuf::from(&options.output).join("analysis.json");
    let analysis_json = match serde_json::to_vec_pretty(&analysis) {
        Ok(json) => json,
        Err(error) => {
            eprintln!("error: {error}");
            return ExitCode::from(1);
        }
    };
    if let Err(error) = fs::write(&analysis_path, analysis_json) {
        eprintln!("error: {error}");
        return ExitCode::from(1);
    }
    let semantics = analyze_semantics(&analysis);
    let ir = build_canonical_ir(&analysis);
    let security = sos_solana::detect(&analysis, &semantics);
    let semantics_path = PathBuf::from(&options.output).join("semantics.json");
    let semantics_json = match serde_json::to_vec_pretty(&semantics) {
        Ok(json) => json,
        Err(error) => {
            eprintln!("error: {error}");
            return ExitCode::from(1);
        }
    };
    if let Err(error) = fs::write(&semantics_path, semantics_json) {
        eprintln!("error: {error}");
        return ExitCode::from(1);
    }
    let ir_path = PathBuf::from(&options.output).join("ir.json");
    let ir_json = match serde_json::to_vec_pretty(&ir) {
        Ok(json) => json,
        Err(error) => {
            eprintln!("error: {error}");
            return ExitCode::from(1);
        }
    };
    if let Err(error) = fs::write(&ir_path, ir_json) {
        eprintln!("error: {error}");
        return ExitCode::from(1);
    }
    let security_path = PathBuf::from(&options.output).join("security.json");
    let security_json = match serde_json::to_vec_pretty(&security) {
        Ok(json) => json,
        Err(error) => {
            eprintln!("error: {error}");
            return ExitCode::from(1);
        }
    };
    if let Err(error) = fs::write(&security_path, security_json) {
        eprintln!("error: {error}");
        return ExitCode::from(1);
    }
    let report_path = PathBuf::from(&options.output).join("report.md");
    if let Err(error) = fs::write(&report_path, render_markdown(&analysis, &semantics)) {
        eprintln!("error: {error}");
        return ExitCode::from(1);
    }
    let pseudocode_path = PathBuf::from(&options.output).join("pseudocode.rs");
    if let Err(error) = fs::write(&pseudocode_path, render_pseudocode(&analysis, &semantics)) {
        eprintln!("error: {error}");
        return ExitCode::from(1);
    }
    let pseudocode_c_path = PathBuf::from(&options.output).join("pseudocode.c");
    if let Err(error) = fs::write(
        &pseudocode_c_path,
        render_c_pseudocode(&analysis, &semantics),
    ) {
        eprintln!("error: {error}");
        return ExitCode::from(1);
    }
    let history_path = if options.history_limit > 0 {
        let path = PathBuf::from(&options.output).join("history.json");
        let history = collect_history(&options);
        match serde_json::to_vec_pretty(&history) {
            Ok(json) => {
                if let Err(error) = fs::write(&path, json) {
                    eprintln!("error: {error}");
                    return ExitCode::from(1);
                }
                Some(path)
            }
            Err(error) => {
                eprintln!("error: {error}");
                return ExitCode::from(1);
            }
        }
    } else {
        None
    };
    let replay_path = if options.replay_requested {
        let path = PathBuf::from(&options.output).join("replay.json");
        let mut trace = replay_for_inspect(&options, &artifact);
        trace.behavioral_comparison = Some(compare_semantics(&semantics, &trace));
        let value = serde_json::to_value(trace).expect("replay trace is serializable");
        if let Err(error) = fs::write(
            &path,
            serde_json::to_vec_pretty(&value).expect("replay status is serializable"),
        ) {
            eprintln!("error: {error}");
            return ExitCode::from(1);
        }
        Some(path)
    } else {
        None
    };
    let (model_path, ai_report_path, readable_path, lift_repair_path) = if let Some(model) =
        &options.model
    {
        let path = PathBuf::from(&options.output).join("model.json");
        let (bundle, lift_repair_records) = match (
            options.model_command.as_deref(),
            options.model_endpoint.as_deref(),
        ) {
            (Some(command), _) => (
                run_model_pipeline(
                    &analysis,
                    &semantics,
                    Some(&manifest.elf_sha256),
                    Some(Path::new(command)),
                    None,
                    model,
                ),
                None,
            ),
            (None, Some(endpoint)) => (
                run_model_pipeline(
                    &analysis,
                    &semantics,
                    Some(&manifest.elf_sha256),
                    None,
                    Some(endpoint),
                    model,
                ),
                None,
            ),
            (None, None) if trusted_open_model(model).is_some() => {
                match run_selected_lift_bundle(
                    &analysis,
                    &semantics,
                    SelectedLiftConfig {
                        artifact_sha256: Some(manifest.elf_sha256.clone()),
                        provider: options.lift_provider,
                        model,
                        ollama_url: &options.ollama_url,
                        context_tokens: options.context_tokens,
                        function_id: options.function_id.as_deref(),
                        repair_rounds: options.repair_rounds.min(4),
                    },
                ) {
                    Ok((bundle, records)) => (bundle, Some(records)),
                    Err(error) => {
                        eprintln!("error: {error}");
                        return ExitCode::from(2);
                    }
                }
            }
            (None, None) => (
                unavailable_pipeline_bundle(
                    AiPipelineConfig {
                        provider_name: "none".into(),
                        model: model.clone(),
                        prompt_version: "sos-v1".into(),
                        artifact_sha256: Some(manifest.elf_sha256.clone()),
                        limits: EvidencePackLimits::default(),
                        max_attempts: 2,
                    },
                    &analysis,
                    &semantics,
                    "provide --model-command PATH, --model-endpoint URL, or a reviewed open-weight model with --provider ollama|groq",
                ),
                None,
            ),
        };
        let value = match serde_json::to_value(&bundle) {
            Ok(value) => value,
            Err(error) => {
                eprintln!("error: {error}");
                return ExitCode::from(1);
            }
        };
        if let Err(error) = fs::write(
            &path,
            serde_json::to_vec_pretty(&value).expect("model status is serializable"),
        ) {
            eprintln!("error: {error}");
            return ExitCode::from(1);
        }
        let ai_report_path = PathBuf::from(&options.output).join("ai-report.md");
        if let Err(error) = fs::write(&ai_report_path, render_ai_markdown(&bundle)) {
            eprintln!("error: {error}");
            return ExitCode::from(1);
        }
        let (readable_path, lift_repair_path) = if let Some(records) = lift_repair_records {
            let readable_path = PathBuf::from(&options.output).join("readable.rs");
            if let Err(error) = fs::write(&readable_path, render_readable_rust(&analysis, &bundle))
            {
                eprintln!("error: {error}");
                return ExitCode::from(1);
            }
            let repair_path = PathBuf::from(&options.output).join("repair.json");
            if let Err(error) = fs::write(
                &repair_path,
                serde_json::to_vec_pretty(&records).expect("repair records are serializable"),
            ) {
                eprintln!("error: {error}");
                return ExitCode::from(1);
            }
            let has_candidate = bundle.stages.iter().any(|stage| {
                stage.status == AiStageStatus::Accepted
                    && matches!(
                        stage.task,
                        ModelTask::StructureRecovery | ModelTask::Refinement
                    )
            });
            if !has_candidate {
                eprintln!(
                    "error: the reviewed local model produced no accepted source candidate; inspect model.json for provider or validation errors"
                );
                return ExitCode::from(1);
            }
            (Some(readable_path), Some(repair_path))
        } else {
            (None, None)
        };
        (
            Some(path),
            Some(ai_report_path),
            readable_path,
            lift_repair_path,
        )
    } else {
        (None, None, None, None)
    };
    let validation_path = if options.validate_requested || options.repair_requested {
        let path = PathBuf::from(&options.output).join("validation.json");
        let mut validation_args = vec![options.output.clone()];
        if let Some(trace) = options
            .validation_trace
            .clone()
            .or_else(|| replay_path.clone())
        {
            validation_args.extend(["--trace".into(), trace.display().to_string()]);
        }
        if options.repair_requested {
            let Some(model_command) = options.model_command.clone() else {
                eprintln!("error: inspect --repair requires --model-command PATH");
                return ExitCode::from(2);
            };
            validation_args.extend([
                "--repair".into(),
                "--model-command".into(),
                model_command,
                "--repair-rounds".into(),
                options.repair_rounds.to_string(),
            ]);
        }
        validation_args.extend(["--out".into(), path.display().to_string()]);
        let status = validate_command_inner(validation_args, false);
        if status != ExitCode::SUCCESS {
            return status;
        }
        Some(path)
    } else {
        None
    };
    if options.json {
        let value = serde_json::json!({
            "artifact": artifact_path,
            "artifact_id": artifact_id,
            "metadata": metadata_path,
            "manifest": manifest_path,
            "analysis": analysis_path,
            "semantics": semantics_path,
            "ir": ir_path,
            "security": security_path,
            "report": report_path,
            "pseudocode": pseudocode_path,
            "pseudocode_c": pseudocode_c_path,
            "history": history_path,
            "replay": replay_path,
            "model": model_path,
            "ai_report": ai_report_path,
            "readable_source": readable_path,
            "lift_repair": lift_repair_path,
            "validation": validation_path,
            "elf_sha256": manifest.elf_sha256,
            "sbpf_version": analysis.sbpf_version,
            "instructions": analysis.instructions.len(),
            "functions": analysis.functions.len(),
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&value)
                .unwrap_or_else(|error| format!("{{\"error\":\"{error}\"}}"))
        );
    } else {
        println!("artifact={}", artifact_path.display());
        println!("artifact_id={artifact_id}");
        println!("metadata={}", metadata_path.display());
        println!("manifest={}", manifest_path.display());
        println!("analysis={}", analysis_path.display());
        println!("semantics={}", semantics_path.display());
        println!("ir={}", ir_path.display());
        println!("security={}", security_path.display());
        println!("report={}", report_path.display());
        println!("pseudocode={}", pseudocode_path.display());
        println!("pseudocode_c={}", pseudocode_c_path.display());
        if let Some(path) = history_path {
            println!("history={}", path.display());
        }
        if let Some(path) = replay_path {
            println!("replay={}", path.display());
        }
        if let Some(path) = model_path {
            println!("model={}", path.display());
        }
        if let Some(path) = ai_report_path {
            println!("ai_report={}", path.display());
        }
        if let Some(path) = readable_path {
            println!("readable_source={}", path.display());
        }
        if let Some(path) = lift_repair_path {
            println!("lift_repair={}", path.display());
        }
        if let Some(path) = validation_path {
            println!("validation={}", path.display());
        }
        println!("elf_sha256={}", manifest.elf_sha256);
        println!("sbpf_version={}", analysis.sbpf_version);
        println!("instructions={}", analysis.instructions.len());
        println!("functions={}", analysis.functions.len());
    }
    ExitCode::SUCCESS
}

fn report_command(args: Vec<String>) -> ExitCode {
    let Some(directory) = args.first() else {
        eprintln!("usage: sos report <analysis-directory> [--out PATH]");
        return ExitCode::from(2);
    };
    let (output, format) = match parse_report_options(&args) {
        Ok(options) => options,
        Err(error) => {
            eprintln!("error: {error}");
            return ExitCode::from(2);
        }
    };
    let directory = PathBuf::from(directory);
    let analysis = match read_json::<sos_analysis::StaticAnalysis>(&directory.join("analysis.json"))
    {
        Ok(value) => value,
        Err(error) => {
            eprintln!("error: {error}");
            return ExitCode::from(1);
        }
    };
    let semantics = read_json::<sos_solana::SemanticAnalysis>(&directory.join("semantics.json"))
        .unwrap_or_else(|_| analyze_semantics(&analysis));
    let report = match format.as_str() {
        "markdown" => render_markdown(&analysis, &semantics),
        "graphviz" => render_cfg_dot(&analysis),
        "json" => match serde_json::to_string_pretty(&serde_json::json!({
            "schema_version": 1,
            "analysis": analysis,
            "semantics": semantics,
            "security": sos_solana::detect(&analysis, &semantics),
        })) {
            Ok(json) => json,
            Err(error) => {
                eprintln!("error: {error}");
                return ExitCode::from(1);
            }
        },
        _ => unreachable!("format validated by parse_report_options"),
    };
    if let Some(path) = output {
        if let Err(error) = fs::write(&path, report) {
            eprintln!("error: {error}");
            return ExitCode::from(1);
        }
        println!("report={}", path.display());
    } else {
        print!("{report}");
    }
    ExitCode::SUCCESS
}

fn parse_report_options(args: &[String]) -> Result<(Option<PathBuf>, String), String> {
    let mut index = 1;
    let mut output = None;
    let mut format = "markdown".to_owned();
    while index < args.len() {
        match args[index].as_str() {
            "--out" => {
                let Some(path) = args.get(index + 1) else {
                    return Err("missing value for --out".into());
                };
                output = Some(PathBuf::from(path));
                index += 2;
            }
            "--format" => {
                let Some(value) = args.get(index + 1) else {
                    return Err("missing value for --format".into());
                };
                if !matches!(value.as_str(), "markdown" | "json" | "graphviz") {
                    return Err(format!("unsupported report format: {value}"));
                }
                format = value.clone();
                index += 2;
            }
            option => return Err(format!("unknown option: {option}")),
        }
    }
    Ok((output, format))
}

fn decompile_command(args: Vec<String>) -> ExitCode {
    let Some(directory) = args.first() else {
        eprintln!("usage: sos decompile <analysis-directory> [--out PATH]");
        return ExitCode::from(2);
    };
    let (output, format) = match parse_decompile_options(&args) {
        Ok(options) => options,
        Err(error) => {
            eprintln!("error: {error}");
            return ExitCode::from(2);
        }
    };
    let directory = PathBuf::from(directory);
    let analysis = match read_json::<sos_analysis::StaticAnalysis>(&directory.join("analysis.json"))
    {
        Ok(value) => value,
        Err(error) => {
            eprintln!("error: {error}");
            return ExitCode::from(1);
        }
    };
    let semantics = read_json::<sos_solana::SemanticAnalysis>(&directory.join("semantics.json"))
        .unwrap_or_else(|_| analyze_semantics(&analysis));
    let pseudocode = match format.as_str() {
        "rust-like" => render_pseudocode(&analysis, &semantics),
        "c-like" => render_c_pseudocode(&analysis, &semantics),
        "json" => match serde_json::to_string_pretty(&serde_json::json!({
            "schema_version": 1,
            "analysis": analysis,
            "semantics": semantics,
            "pseudocode": render_pseudocode(&analysis, &semantics),
        })) {
            Ok(json) => json,
            Err(error) => {
                eprintln!("error: {error}");
                return ExitCode::from(1);
            }
        },
        _ => unreachable!("format validated by parse_decompile_options"),
    };
    if let Some(path) = output {
        if let Err(error) = fs::write(&path, pseudocode) {
            eprintln!("error: {error}");
            return ExitCode::from(1);
        }
        println!("pseudocode={}", path.display());
    } else {
        print!("{pseudocode}");
    }
    ExitCode::SUCCESS
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LiftProvider {
    Ollama,
    Groq,
}

impl LiftProvider {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "ollama" => Ok(Self::Ollama),
            "groq" => Ok(Self::Groq),
            _ => Err(format!(
                "unsupported lift provider {value:?}; use ollama or groq"
            )),
        }
    }
}

#[derive(Debug)]
struct HumanizeOptions {
    input: PathBuf,
    output: PathBuf,
    audit_output: PathBuf,
    provider: LiftProvider,
    model: String,
    ollama_url: String,
    allow_unreviewed_model: bool,
    max_units: usize,
    plan_only: bool,
}

impl HumanizeOptions {
    fn parse(args: Vec<String>) -> Result<Self, String> {
        let Some(input) = args.first().map(PathBuf::from) else {
            return Err(
                "usage: sos humanize <pseudocode-file> [--plan] [--provider groq|ollama] [--model MODEL] [--max-units N] [--out PATH] [--audit-out PATH]".into(),
            );
        };
        let mut output = input.clone();
        output.set_extension("readable.rs");
        let mut audit_output = input.clone();
        audit_output.set_extension("humanize.json");
        let mut options = Self {
            input,
            output,
            audit_output,
            provider: LiftProvider::Groq,
            model: "openai/gpt-oss-120b".into(),
            ollama_url: "http://127.0.0.1:11434".into(),
            allow_unreviewed_model: false,
            max_units: 16,
            plan_only: false,
        };
        let mut model_explicit = false;
        let mut index = 1;
        while index < args.len() {
            if args[index] == "--allow-unreviewed-model" {
                options.allow_unreviewed_model = true;
                index += 1;
                continue;
            }
            if args[index] == "--plan" {
                options.plan_only = true;
                index += 1;
                continue;
            }
            let flag = &args[index];
            let Some(value) = args.get(index + 1) else {
                return Err(format!("missing value for {flag}"));
            };
            match flag.as_str() {
                "--provider" => options.provider = LiftProvider::parse(value)?,
                "--model" => {
                    options.model = value.clone();
                    model_explicit = true;
                }
                "--ollama-url" => options.ollama_url = value.clone(),
                "--out" => options.output = PathBuf::from(value),
                "--audit-out" => options.audit_output = PathBuf::from(value),
                "--max-units" => {
                    options.max_units = value
                        .parse::<usize>()
                        .map_err(|error| format!("invalid max unit count: {error}"))?;
                    if !(1..=512).contains(&options.max_units) {
                        return Err("--max-units must be between 1 and 512".into());
                    }
                }
                _ => return Err(format!("unknown humanize option: {flag}")),
            }
            index += 2;
        }
        if !model_explicit {
            options.model = match options.provider {
                LiftProvider::Groq => "openai/gpt-oss-120b",
                LiftProvider::Ollama => "gpt-oss:20b",
            }
            .into();
        }
        if trusted_open_model(&options.model).is_none() && !options.allow_unreviewed_model {
            return Err(format!(
                "model {:?} is not in SOS's reviewed open-weight registry; pass --allow-unreviewed-model to override",
                options.model
            ));
        }
        if options.output == options.input || options.audit_output == options.input {
            return Err("humanize output paths must not overwrite the input pseudocode".into());
        }
        if options.output == options.audit_output {
            return Err("--out and --audit-out must use different paths".into());
        }
        Ok(options)
    }
}

struct HumanizationRun {
    readable_fragments: Vec<String>,
    unit_audits: Vec<serde_json::Value>,
    failures: usize,
}

fn humanization_processing_order(units: &[PseudocodeUnit]) -> Vec<usize> {
    fn visit(
        index: usize,
        units: &[PseudocodeUnit],
        by_name: &BTreeMap<String, Vec<usize>>,
        state: &mut [u8],
        order: &mut Vec<usize>,
    ) {
        if state[index] == 2 {
            return;
        }
        if state[index] == 1 {
            // Preserve deterministic source order across recursive cycles.
            return;
        }
        state[index] = 1;
        for referenced in &units[index].referenced_functions {
            if let Some(dependencies) = by_name.get(referenced) {
                for dependency in dependencies {
                    if *dependency != index {
                        visit(*dependency, units, by_name, state, order);
                    }
                }
            }
        }
        state[index] = 2;
        order.push(index);
    }

    let mut by_name = BTreeMap::<String, Vec<usize>>::new();
    for (index, unit) in units.iter().enumerate() {
        if let Some(name) = &unit.function_name {
            by_name.entry(name.clone()).or_default().push(index);
        }
    }
    let mut state = vec![0_u8; units.len()];
    let mut order = Vec::with_capacity(units.len());
    for index in 0..units.len() {
        visit(index, units, &by_name, &mut state, &mut order);
    }
    order
}

fn build_humanization_plan(
    source: &str,
    units: &[PseudocodeUnit],
) -> Result<serde_json::Value, String> {
    let processing_order = humanization_processing_order(units);
    let processing_positions = processing_order
        .iter()
        .enumerate()
        .map(|(position, unit_index)| (*unit_index, position + 1))
        .collect::<BTreeMap<_, _>>();
    let mut planned_units = Vec::with_capacity(units.len());
    for (unit_index, unit) in units.iter().enumerate() {
        let pack = evidence_pack_from_pseudocode_unit(unit).map_err(|error| error.to_string())?;
        planned_units.push(serde_json::json!({
            "id": unit.id,
            "start_line": unit.start_line,
            "end_line": unit.end_line,
            "function_id": pack.ir.functions.first().map(|function| &function.id),
            "function_name": unit.function_name,
            "referenced_functions": unit.referenced_functions,
            "operation_count": pack.ir.operations.len(),
            "unit_sha256": pack.artifact_sha256,
            "processing_position": processing_positions.get(&unit_index),
        }));
    }
    Ok(serde_json::json!({
        "schema_version": 1,
        "mode": "plan_only_no_inference",
        "source_sha256": sha256_hex(source.as_bytes()),
        "unit_count": units.len(),
        "processing_sequence": processing_order
            .iter()
            .map(|index| &units[*index].id)
            .collect::<Vec<_>>(),
        "units": planned_units,
    }))
}

fn run_humanization_units<P>(provider: P, model: &str, units: &[PseudocodeUnit]) -> HumanizationRun
where
    P: ModelProvider + Clone,
{
    let mut run = HumanizationRun {
        readable_fragments: vec![String::new(); units.len()],
        unit_audits: vec![serde_json::Value::Null; units.len()],
        failures: 0,
    };
    let mut accepted_context = Vec::<UntrustedEvidence>::new();
    let processing_order = humanization_processing_order(units);
    for (processing_index, unit_index) in processing_order.iter().enumerate() {
        let unit = &units[*unit_index];
        eprintln!(
            "humanizing unit {}/{}: {} (lines {}-{})",
            processing_index + 1,
            units.len(),
            unit.id,
            unit.start_line,
            unit.end_line
        );
        let mut pack = match evidence_pack_from_pseudocode_unit(unit) {
            Ok(pack) => pack,
            Err(error) => {
                run.failures += 1;
                run.readable_fragments[*unit_index] = format!(
                    "// UNRESOLVED {} (lines {}-{}): {}",
                    unit.id, unit.start_line, unit.end_line, error
                );
                run.unit_audits[*unit_index] = serde_json::json!({
                    "unit": unit,
                    "processing_position": processing_index + 1,
                    "status": "rejected",
                    "error": error.to_string(),
                });
                continue;
            }
        };
        let context_start = accepted_context.len().saturating_sub(2);
        pack.untrusted_evidence
            .extend_from_slice(&accepted_context[context_start..]);
        let expected_function = pack
            .ir
            .functions
            .first()
            .map(|function| function.id.clone())
            .unwrap_or_else(|| "fn_unknown".into());
        let response = submit_model(
            provider.clone(),
            ModelTask::PseudocodeHumanization,
            model,
            pack.clone(),
        );
        match response {
            Ok(response) => {
                let candidate = serde_json::from_value::<StructureRecoveryOutput>(
                    response.structured_output.clone(),
                )
                .ok()
                .and_then(|recovered| {
                    recovered
                        .functions
                        .into_iter()
                        .find(|function| function.function_id == expected_function)
                });
                if let Some(function) = candidate {
                    let prior = serde_json::to_string(&response.structured_output)
                        .unwrap_or_else(|_| "{}".into())
                        .chars()
                        .take(4_096)
                        .collect::<String>();
                    accepted_context.push(UntrustedEvidence {
                        source: format!("accepted_humanization:{}", unit.id),
                        content: prior,
                    });
                    run.readable_fragments[*unit_index] = format!(
                        "// {}: original pseudocode lines {}-{}; evidence function {}\n{}",
                        unit.id,
                        unit.start_line,
                        unit.end_line,
                        expected_function,
                        function.pseudocode.trim()
                    );
                    run.unit_audits[*unit_index] = serde_json::json!({
                        "unit": unit,
                        "processing_position": processing_index + 1,
                        "status": "accepted_unverified",
                        "evidence_pack": pack,
                        "response": response,
                    });
                } else {
                    run.failures += 1;
                    let error = format!(
                        "model returned no candidate for expected function {expected_function}"
                    );
                    run.readable_fragments[*unit_index] = format!(
                        "// UNRESOLVED {} (lines {}-{}): {}",
                        unit.id, unit.start_line, unit.end_line, error
                    );
                    run.unit_audits[*unit_index] = serde_json::json!({
                        "unit": unit,
                        "processing_position": processing_index + 1,
                        "status": "rejected",
                        "error": error,
                        "evidence_pack": pack,
                        "response": response,
                    });
                }
            }
            Err(error) => {
                run.failures += 1;
                run.readable_fragments[*unit_index] = format!(
                    "// UNRESOLVED {} (lines {}-{}): {}",
                    unit.id, unit.start_line, unit.end_line, error
                );
                run.unit_audits[*unit_index] = serde_json::json!({
                    "unit": unit,
                    "processing_position": processing_index + 1,
                    "status": "not_available",
                    "error": error,
                    "evidence_pack": pack,
                });
            }
        }
    }
    run
}

fn humanize_command(args: Vec<String>) -> ExitCode {
    let options = match HumanizeOptions::parse(args) {
        Ok(options) => options,
        Err(error) => {
            eprintln!("error: {error}");
            return ExitCode::from(2);
        }
    };
    let source = match fs::read_to_string(&options.input) {
        Ok(source) => source,
        Err(error) => {
            eprintln!("error: {}: {error}", options.input.display());
            return ExitCode::from(1);
        }
    };
    let units = match partition_pseudocode(&source) {
        Ok(units) => units,
        Err(error) => {
            eprintln!("error: {error}");
            return ExitCode::from(2);
        }
    };
    if units.len() > options.max_units {
        eprintln!(
            "error: pseudocode requires {} hosted units, exceeding --max-units {}; review the partition or raise the limit explicitly",
            units.len(),
            options.max_units
        );
        return ExitCode::from(2);
    }
    if options.plan_only {
        let plan = match build_humanization_plan(&source, &units) {
            Ok(plan) => plan,
            Err(error) => {
                eprintln!("error: unable to build humanization plan: {error}");
                return ExitCode::from(1);
            }
        };
        if let Some(parent) = options
            .audit_output
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            if let Err(error) = fs::create_dir_all(parent) {
                eprintln!("error: {}: {error}", parent.display());
                return ExitCode::from(1);
            }
        }
        if let Err(error) = fs::write(
            &options.audit_output,
            serde_json::to_vec_pretty(&plan).expect("humanization plan is serializable"),
        ) {
            eprintln!("error: {}: {error}", options.audit_output.display());
            return ExitCode::from(1);
        }
        println!("humanization_plan={}", options.audit_output.display());
        return ExitCode::SUCCESS;
    }
    let run = match options.provider {
        LiftProvider::Groq => {
            let api_key = match env::var("GROQ_API_KEY") {
                Ok(api_key) => api_key,
                Err(_) => {
                    eprintln!("error: GROQ_API_KEY is required for --provider groq");
                    return ExitCode::from(2);
                }
            };
            let provider = match GroqModelProvider::new(api_key) {
                Ok(provider) => provider,
                Err(error) => {
                    eprintln!("error: {error}");
                    return ExitCode::from(2);
                }
            };
            run_humanization_units(provider, &options.model, &units)
        }
        LiftProvider::Ollama => run_humanization_units(
            OllamaModelProvider::new(&options.ollama_url),
            &options.model,
            &units,
        ),
    };
    if let Some(parent) = options
        .output
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        if let Err(error) = fs::create_dir_all(parent) {
            eprintln!("error: {}: {error}", parent.display());
            return ExitCode::from(1);
        }
    }
    let readable = format!(
        "// SOS model-assisted reconstruction. Unverified; not original source.\n\n{}\n",
        run.readable_fragments.join("\n\n")
    );
    if let Err(error) = fs::write(&options.output, readable) {
        eprintln!("error: {}: {error}", options.output.display());
        return ExitCode::from(1);
    }
    let audit = serde_json::json!({
        "schema_version": 1,
        "provider": match options.provider { LiftProvider::Groq => "groq_hosted_open_weights", LiftProvider::Ollama => "ollama_local_open_weights" },
        "model": options.model,
        "source_sha256": sha256_hex(source.as_bytes()),
        "unit_count": units.len(),
        "failed_units": run.failures,
        "units": run.unit_audits,
    });
    if let Some(parent) = options
        .audit_output
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        if let Err(error) = fs::create_dir_all(parent) {
            eprintln!("error: {}: {error}", parent.display());
            return ExitCode::from(1);
        }
    }
    if let Err(error) = fs::write(
        &options.audit_output,
        serde_json::to_vec_pretty(&audit).expect("humanization audit is serializable"),
    ) {
        eprintln!("error: {}: {error}", options.audit_output.display());
        return ExitCode::from(1);
    }
    println!("readable_source={}", options.output.display());
    println!("audit={}", options.audit_output.display());
    if run.failures == 0 {
        ExitCode::SUCCESS
    } else {
        eprintln!(
            "error: {} of {} pseudocode units were not humanized; partial output was preserved",
            run.failures,
            units.len()
        );
        ExitCode::from(1)
    }
}

#[derive(Debug)]
struct LiftOptions {
    directory: PathBuf,
    output: PathBuf,
    provider: LiftProvider,
    ollama_url: String,
    model: String,
    context_tokens: usize,
    allow_unreviewed_model: bool,
    function_id: Option<String>,
    repair_rounds: usize,
}

impl LiftOptions {
    fn parse(args: Vec<String>) -> Result<Self, String> {
        let Some(directory) = args.first() else {
            return Err(
                "usage: sos lift <analysis-directory> [--provider ollama|groq] [--function fn_ID] [--model MODEL] [--ollama-url URL] [--context-tokens N] [--repair-rounds N] [--out DIR]".into(),
            );
        };
        let mut options = Self {
            directory: PathBuf::from(directory),
            output: PathBuf::from(directory).join("lifted"),
            provider: LiftProvider::Ollama,
            ollama_url: "http://127.0.0.1:11434".into(),
            model: "gpt-oss:20b".into(),
            context_tokens: 32_768,
            allow_unreviewed_model: false,
            function_id: None,
            repair_rounds: 2,
        };
        let mut model_explicit = false;
        let mut index = 1;
        while index < args.len() {
            if args[index] == "--allow-unreviewed-model" {
                options.allow_unreviewed_model = true;
                index += 1;
                continue;
            }
            let flag = &args[index];
            let Some(value) = args.get(index + 1) else {
                return Err(format!("missing value for {flag}"));
            };
            match flag.as_str() {
                "--out" => options.output = PathBuf::from(value),
                "--provider" => options.provider = LiftProvider::parse(value)?,
                "--ollama-url" => options.ollama_url = value.clone(),
                "--model" => {
                    options.model = value.clone();
                    model_explicit = true;
                }
                "--function" => options.function_id = Some(value.clone()),
                "--context-tokens" => {
                    options.context_tokens = value
                        .parse::<usize>()
                        .map_err(|error| format!("invalid context token count: {error}"))?;
                    if !(4_096..=131_072).contains(&options.context_tokens) {
                        return Err("--context-tokens must be between 4096 and 131072".into());
                    }
                }
                "--repair-rounds" => {
                    options.repair_rounds = value
                        .parse::<usize>()
                        .map_err(|error| format!("invalid repair round count: {error}"))?;
                    if options.repair_rounds > 4 {
                        return Err("--repair-rounds must be between 0 and 4".into());
                    }
                }
                _ => return Err(format!("unknown lift option: {flag}")),
            }
            index += 2;
        }
        if options.provider == LiftProvider::Groq && !model_explicit {
            options.model = "openai/gpt-oss-120b".into();
        }
        if trusted_open_model(&options.model).is_none() && !options.allow_unreviewed_model {
            return Err(format!(
                "model {:?} is not in SOS's reviewed open-weight registry; use gpt-oss:20b, openai/gpt-oss-120b, or explicitly pass --allow-unreviewed-model",
                options.model
            ));
        }
        Ok(options)
    }
}

struct LiftRunConfig<'a> {
    artifact_sha256: Option<String>,
    model: &'a str,
    provider_name: &'a str,
    function_id: Option<&'a str>,
    repair_rounds: usize,
}

fn run_lift_bundle<P>(
    provider: P,
    analysis: &sos_analysis::StaticAnalysis,
    global_semantics: &sos_solana::SemanticAnalysis,
    options: LiftRunConfig<'_>,
) -> Result<(AiPipelineBundle, Vec<serde_json::Value>), String>
where
    P: ModelProvider + Clone,
{
    let config = AiPipelineConfig {
        provider_name: options.provider_name.into(),
        model: options.model.into(),
        prompt_version: "sos-semantic-lift-v3".into(),
        artifact_sha256: options.artifact_sha256.clone(),
        limits: EvidencePackLimits::default(),
        max_attempts: 2,
    };
    let mut bundle = AiPipelineBundle {
        schema_version: sos_ai::AI_BUNDLE_SCHEMA_VERSION,
        artifact_sha256: options.artifact_sha256,
        provider: config.provider_name.clone(),
        model: config.model.clone(),
        stages: Vec::new(),
    };
    let mut repair_records = Vec::<serde_json::Value>::new();
    let mut functions = analysis.functions.clone();
    functions.sort_by_key(|function| function.pc);
    if let Some(function_id) = options.function_id {
        functions.retain(|function| format!("fn_{:x}", function.pc) == function_id);
        if functions.is_empty() {
            return Err(format!(
                "function {function_id:?} does not exist in the analysis IR"
            ));
        }
    }
    if functions.is_empty() {
        bundle = run_pipeline(provider, config, analysis, global_semantics);
    } else {
        for (index, function) in functions.iter().enumerate() {
            eprintln!(
                "lifting function {}/{}: fn_{:x}",
                index + 1,
                functions.len(),
                function.pc
            );
            let Some(scoped_analysis) = scope_analysis_to_function(analysis, function.pc) else {
                continue;
            };
            let scoped_semantics = analyze_semantics(&scoped_analysis);
            let mut partial = run_pipeline(
                provider.clone(),
                config.clone(),
                &scoped_analysis,
                &scoped_semantics,
            );
            if options.repair_rounds > 0 {
                repair_failed_refinement(
                    &provider,
                    &config,
                    &scoped_analysis,
                    &scoped_semantics,
                    options.repair_rounds,
                    &mut partial,
                    &mut repair_records,
                );
            }
            bundle.stages.extend(partial.stages);
        }
    }
    Ok((bundle, repair_records))
}

struct SelectedLiftConfig<'a> {
    artifact_sha256: Option<String>,
    provider: LiftProvider,
    model: &'a str,
    ollama_url: &'a str,
    context_tokens: usize,
    function_id: Option<&'a str>,
    repair_rounds: usize,
}

fn run_selected_lift_bundle(
    analysis: &sos_analysis::StaticAnalysis,
    semantics: &sos_solana::SemanticAnalysis,
    options: SelectedLiftConfig<'_>,
) -> Result<(AiPipelineBundle, Vec<serde_json::Value>), String> {
    let run = |provider_name| LiftRunConfig {
        artifact_sha256: options.artifact_sha256.clone(),
        model: options.model,
        provider_name,
        function_id: options.function_id,
        repair_rounds: options.repair_rounds,
    };
    match options.provider {
        LiftProvider::Ollama => run_lift_bundle(
            OllamaModelProvider::new(options.ollama_url)
                .with_context_tokens(options.context_tokens),
            analysis,
            semantics,
            run("ollama_local_open_weights"),
        ),
        LiftProvider::Groq => {
            let api_key = env::var("GROQ_API_KEY").map_err(|_| {
                "GROQ_API_KEY is required for --provider groq; create a free Groq key and export it in this shell".to_owned()
            })?;
            let provider = GroqModelProvider::new(api_key).map_err(|error| error.to_string())?;
            run_lift_bundle(
                provider,
                analysis,
                semantics,
                run("groq_hosted_open_weights"),
            )
        }
    }
}

fn lift_command(args: Vec<String>) -> ExitCode {
    let options = match LiftOptions::parse(args) {
        Ok(options) => options,
        Err(error) => {
            eprintln!("error: {error}");
            return ExitCode::from(2);
        }
    };
    let analysis =
        match read_json::<sos_analysis::StaticAnalysis>(&options.directory.join("analysis.json")) {
            Ok(analysis) => analysis,
            Err(error) => {
                eprintln!("error: {error}");
                return ExitCode::from(1);
            }
        };
    let global_semantics =
        read_json::<sos_solana::SemanticAnalysis>(&options.directory.join("semantics.json"))
            .unwrap_or_else(|_| analyze_semantics(&analysis));
    let artifact_sha256 = read_json::<ArtifactManifest>(&options.directory.join("manifest.json"))
        .ok()
        .map(|manifest| manifest.elf_sha256);
    let (bundle, repair_records) = match run_selected_lift_bundle(
        &analysis,
        &global_semantics,
        SelectedLiftConfig {
            artifact_sha256,
            provider: options.provider,
            model: &options.model,
            ollama_url: &options.ollama_url,
            context_tokens: options.context_tokens,
            function_id: options.function_id.as_deref(),
            repair_rounds: options.repair_rounds,
        },
    ) {
        Ok(result) => result,
        Err(error) => {
            eprintln!("error: {error}");
            return ExitCode::from(2);
        }
    };
    if let Err(error) = fs::create_dir_all(&options.output) {
        eprintln!("error: {}: {error}", options.output.display());
        return ExitCode::from(1);
    }
    let model_path = options.output.join("model.json");
    let report_path = options.output.join("ai-report.md");
    let source_path = options.output.join("readable.rs");
    let repair_path = options.output.join("repair.json");
    let writes = [
        (
            &model_path,
            serde_json::to_vec_pretty(&bundle).map_err(|error| error.to_string()),
        ),
        (&report_path, Ok(render_ai_markdown(&bundle).into_bytes())),
        (
            &source_path,
            Ok(render_readable_rust(&analysis, &bundle).into_bytes()),
        ),
    ];
    for (path, bytes) in writes {
        let bytes = match bytes {
            Ok(bytes) => bytes,
            Err(error) => {
                eprintln!("error: {error}");
                return ExitCode::from(1);
            }
        };
        if let Err(error) = fs::write(path, bytes) {
            eprintln!("error: {}: {error}", path.display());
            return ExitCode::from(1);
        }
    }
    if let Err(error) = fs::write(
        &repair_path,
        serde_json::to_vec_pretty(&repair_records).expect("repair records are serializable"),
    ) {
        eprintln!("error: {}: {error}", repair_path.display());
        return ExitCode::from(1);
    }

    println!("model={}", model_path.display());
    println!("ai_report={}", report_path.display());
    println!("readable_source={}", source_path.display());
    println!("repair={}", repair_path.display());
    if let Some(profile) = trusted_open_model(&options.model) {
        println!(
            "open_model={} publisher={} license={} weights={}",
            profile.id, profile.publisher, profile.license, profile.weights_url
        );
    }

    let has_candidate = bundle.stages.iter().any(|stage| {
        stage.status == sos_ai::AiStageStatus::Accepted
            && matches!(
                stage.task,
                ModelTask::StructureRecovery | ModelTask::Refinement
            )
            && stage.response.as_ref().is_some_and(|response| {
                response
                    .structured_output
                    .get("functions")
                    .or_else(|| response.structured_output.get("patches"))
                    .and_then(serde_json::Value::as_array)
                    .is_some_and(|items| !items.is_empty())
            })
    });
    if has_candidate {
        ExitCode::SUCCESS
    } else {
        eprintln!("error: no readable candidate was accepted; inspect model.json for stage errors");
        ExitCode::from(1)
    }
}

fn repair_failed_refinement<P>(
    provider: &P,
    config: &AiPipelineConfig,
    analysis: &sos_analysis::StaticAnalysis,
    semantics: &sos_solana::SemanticAnalysis,
    repair_rounds: usize,
    bundle: &mut AiPipelineBundle,
    repair_records: &mut Vec<serde_json::Value>,
) where
    P: ModelProvider + Clone,
{
    let function_id = format!("fn_{:x}", analysis.entrypoint);
    let Some(initial) = bundle
        .stages
        .iter()
        .rev()
        .find(|stage| {
            stage.task == ModelTask::Refinement && stage.status == AiStageStatus::Accepted
        })
        .and_then(|stage| stage.response.as_ref())
        .and_then(|response| {
            serde_json::from_value::<RefinementOutput>(response.structured_output.clone()).ok()
        })
    else {
        repair_records.push(serde_json::json!({
            "function_id": function_id,
            "status": "not_available",
            "reason": "no accepted initial refinement"
        }));
        return;
    };
    let compiler = Path::new("rustc");
    let initial_compilation =
        compile_refinement_output(&initial, Some(compiler), Duration::from_secs(30));
    if matches!(
        initial_compilation.status,
        CompilationStatus::Passed | CompilationStatus::NotAvailable
    ) {
        repair_records.push(serde_json::json!({
            "function_id": function_id,
            "status": if initial_compilation.status == CompilationStatus::Passed { "compiled_unverified" } else { "not_available" },
            "initial_compilation": initial_compilation,
            "repair_loop": null
        }));
        return;
    }

    let mut pack = EvidencePack::bounded_from_analysis(
        ModelTask::Refinement,
        config.artifact_sha256.clone(),
        analysis,
        semantics,
        config.limits,
    );
    for stage in bundle
        .stages
        .iter()
        .filter(|stage| stage.status == AiStageStatus::Accepted)
    {
        if let Some(response) = stage.response.as_ref() {
            let content = serde_json::to_string(&response.structured_output)
                .unwrap_or_else(|_| "{}".into())
                .chars()
                .take(128 * 1024)
                .collect::<String>();
            pack.untrusted_evidence.push(UntrustedEvidence {
                source: format!("accepted_model_stage:{:?}", stage.task),
                content,
            });
        }
    }
    let initial_feedback = RefinementFeedback {
        compilation: initial_compilation.status,
        compiler_diagnostics: initial_compilation.diagnostics.clone(),
        behavioral_status: "not_requested".into(),
        mismatches: Vec::new(),
        limitations: initial_compilation.limitations.clone(),
    };
    pack.untrusted_evidence.push(UntrustedEvidence {
        source: "external_validation:initial_compilation".into(),
        content: serde_json::to_string(&initial_feedback).unwrap_or_else(|_| "{}".into()),
    });

    let loop_result = run_refinement_loop(
        provider.clone(),
        pack,
        config.model.clone(),
        config.prompt_version.clone(),
        config.max_attempts,
        repair_rounds,
        |candidate| {
            let compilation =
                compile_refinement_output(candidate, Some(compiler), Duration::from_secs(30));
            RefinementFeedback {
                compilation: compilation.status,
                compiler_diagnostics: compilation.diagnostics,
                behavioral_status: "not_requested".into(),
                mismatches: Vec::new(),
                limitations: compilation.limitations,
            }
        },
    );
    let compiled = loop_result
        .rounds
        .last()
        .and_then(|round| round.feedback.as_ref())
        .is_some_and(|feedback| feedback.compilation == CompilationStatus::Passed);
    if compiled {
        if let (Some(final_output), Some(round)) =
            (loop_result.final_output.clone(), loop_result.rounds.last())
        {
            bundle.stages.push(AiStageResult {
                task: ModelTask::Refinement,
                status: AiStageStatus::Accepted,
                model: config.model.clone(),
                prompt_version: config.prompt_version.clone(),
                input_sha256: round.input_sha256.clone(),
                output_sha256: round.output_sha256.clone(),
                response: Some(ModelResponse {
                    task: ModelTask::Refinement,
                    model: config.model.clone(),
                    prompt_version: config.prompt_version.clone(),
                    structured_output: serde_json::to_value(final_output)
                        .expect("refinement output is serializable"),
                    evidence: Vec::new(),
                }),
                error: None,
            });
        }
    }
    repair_records.push(serde_json::json!({
        "function_id": function_id,
        "status": if compiled { "compiled_unverified" } else { "repair_exhausted" },
        "initial_compilation": initial_compilation,
        "repair_loop": loop_result
    }));
}

fn parse_decompile_options(args: &[String]) -> Result<(Option<PathBuf>, String), String> {
    let mut index = 1;
    let mut output = None;
    let mut format = "rust-like".to_owned();
    while index < args.len() {
        match args[index].as_str() {
            "--out" => {
                let Some(path) = args.get(index + 1) else {
                    return Err("missing value for --out".into());
                };
                output = Some(PathBuf::from(path));
                index += 2;
            }
            "--format" => {
                let Some(value) = args.get(index + 1) else {
                    return Err("missing value for --format".into());
                };
                if !matches!(value.as_str(), "rust-like" | "c-like" | "json") {
                    return Err(format!("unsupported decompile format: {value}"));
                }
                format = value.clone();
                index += 2;
            }
            option => return Err(format!("unknown option: {option}")),
        }
    }
    Ok((output, format))
}

fn read_json<T: serde::de::DeserializeOwned>(path: &std::path::Path) -> Result<T, String> {
    let bytes = fs::read(path).map_err(|error| format!("{}: {error}", path.display()))?;
    serde_json::from_slice(&bytes).map_err(|error| format!("{}: {error}", path.display()))
}

fn run_model_pipeline(
    analysis: &sos_analysis::StaticAnalysis,
    semantics: &sos_solana::SemanticAnalysis,
    artifact_sha256: Option<&str>,
    command: Option<&Path>,
    endpoint: Option<&str>,
    model: &str,
) -> AiPipelineBundle {
    let artifact_sha256 = artifact_sha256.map(str::to_owned);
    match (command, endpoint) {
        (Some(command), _) => run_pipeline(
            CommandModelProvider::new(command),
            AiPipelineConfig {
                provider_name: "local_command".into(),
                model: model.into(),
                prompt_version: "sos-v1".into(),
                artifact_sha256,
                limits: EvidencePackLimits::default(),
                max_attempts: 2,
            },
            analysis,
            semantics,
        ),
        (None, Some(endpoint)) => {
            let mut provider = HttpModelProvider::new(endpoint);
            if let Ok(token) = env::var("SOS_MODEL_TOKEN") {
                provider = provider.with_bearer_token(token);
            }
            run_pipeline(
                provider,
                AiPipelineConfig {
                    provider_name: "http".into(),
                    model: model.into(),
                    prompt_version: "sos-v1".into(),
                    artifact_sha256,
                    limits: EvidencePackLimits::default(),
                    max_attempts: 2,
                },
                analysis,
                semantics,
            )
        }
        (None, None) => unreachable!("inspect handles an absent model provider explicitly"),
    }
}

fn run_model(
    directory: &Path,
    command: Option<&Path>,
    endpoint: Option<&str>,
    model: &str,
    task: ModelTask,
) -> Result<sos_ai::ModelResponse, String> {
    let analysis = read_json::<sos_analysis::StaticAnalysis>(&directory.join("analysis.json"))?;
    let semantics = read_json::<sos_solana::SemanticAnalysis>(&directory.join("semantics.json"))
        .unwrap_or_else(|_| analyze_semantics(&analysis));
    let artifact_sha256 =
        read_json::<sos_types::ArtifactManifest>(&directory.join("manifest.json"))
            .ok()
            .map(|manifest| manifest.elf_sha256);
    let pack = EvidencePack::bounded_from_analysis(
        task,
        artifact_sha256,
        &analysis,
        &semantics,
        EvidencePackLimits::default(),
    );
    if let Some(command) = command {
        return submit_model(CommandModelProvider::new(command), task, model, pack);
    }
    if let Some(endpoint) = endpoint {
        let mut provider = HttpModelProvider::new(endpoint);
        if let Ok(token) = env::var("SOS_MODEL_TOKEN") {
            provider = provider.with_bearer_token(token);
        }
        return submit_model(provider, task, model, pack);
    }
    Err("no model provider configured".into())
}

fn submit_model<P: ModelProvider>(
    provider: P,
    task: ModelTask,
    model: &str,
    pack: EvidencePack,
) -> Result<sos_ai::ModelResponse, String> {
    let gateway = ModelGateway::new(provider, 4 * 1024 * 1024, 1024 * 1024);
    gateway
        .submit_with_retry(task, model.to_owned(), "sos-v1", pack, 2)
        .map_err(|error| error.to_string())
}

fn parse_output_path(args: &[String]) -> Result<Option<PathBuf>, String> {
    let mut index = 1;
    let mut output = None;
    while index < args.len() {
        if args[index] == "--json" {
            index += 1;
            continue;
        }
        if args[index] != "--out" {
            return Err(format!("unknown option: {}", args[index]));
        }
        let Some(path) = args.get(index + 1) else {
            return Err("missing value for --out".into());
        };
        output = Some(PathBuf::from(path));
        index += 2;
    }
    Ok(output)
}

struct RemoteOptions {
    program_address: String,
    rpc_url: String,
    cluster: Option<String>,
    rpc_explicit: bool,
    output: String,
    commitment: String,
    requested_slot: Option<u64>,
    history_limit: usize,
    json: bool,
    replay_requested: bool,
    model: Option<String>,
    model_command: Option<String>,
    model_endpoint: Option<String>,
    lift_provider: LiftProvider,
    ollama_url: String,
    context_tokens: usize,
    function_id: Option<String>,
    signature: Option<String>,
    validate_requested: bool,
    repair_requested: bool,
    repair_rounds: usize,
    validation_trace: Option<PathBuf>,
}

struct BenchmarkOptions {
    artifact: PathBuf,
    output: Option<PathBuf>,
    repeats: usize,
}

impl BenchmarkOptions {
    fn parse(args: Vec<String>) -> Result<Self, String> {
        let Some(artifact) = args.first() else {
            return Err("usage: sos benchmark <artifact> [--repeats N] [--out PATH]".into());
        };
        let mut options = Self {
            artifact: PathBuf::from(artifact),
            output: None,
            repeats: 2,
        };
        let mut index = 1;
        while index < args.len() {
            let flag = &args[index];
            let Some(value) = args.get(index + 1) else {
                return Err(format!("missing value for {flag}"));
            };
            match flag.as_str() {
                "--out" => options.output = Some(PathBuf::from(value)),
                "--repeats" => {
                    options.repeats = value
                        .parse::<usize>()
                        .map_err(|error| format!("invalid repeat count: {error}"))?;
                    if !(1..=8).contains(&options.repeats) {
                        return Err("--repeats must be between 1 and 8".into());
                    }
                }
                _ => return Err(format!("unknown benchmark option: {flag}")),
            }
            index += 2;
        }
        Ok(options)
    }
}

struct ReplayOptions {
    program_address: String,
    signature: String,
    rpc_url: String,
    commitment: String,
    requested_slot: Option<u64>,
    output: Option<PathBuf>,
}

struct ExplainOptions {
    directory: PathBuf,
    model_command: Option<PathBuf>,
    model_endpoint: Option<String>,
    model: String,
    task: ModelTask,
    output: Option<PathBuf>,
}

impl ExplainOptions {
    fn parse(args: Vec<String>) -> Result<Self, String> {
        let Some(directory) = args.first() else {
            return Err("usage: sos explain <analysis-directory> --model-command PATH or --model-endpoint URL [options]".into());
        };
        let mut options = Self {
            directory: PathBuf::from(directory),
            model_command: None,
            model_endpoint: None,
            model: "local".into(),
            task: ModelTask::Explanation,
            output: None,
        };
        let mut index = 1;
        while index < args.len() {
            let flag = &args[index];
            let Some(value) = args.get(index + 1) else {
                return Err(format!("missing value for {flag}"));
            };
            match flag.as_str() {
                "--model-command" => options.model_command = Some(PathBuf::from(value)),
                "--model-endpoint" => options.model_endpoint = Some(value.clone()),
                "--model" => options.model = value.clone(),
                "--task" => {
                    options.task = match value.as_str() {
                        "structure" => ModelTask::StructureRecovery,
                        "semantic" => ModelTask::SemanticLabeling,
                        "explanation" => ModelTask::Explanation,
                        "refinement" => ModelTask::Refinement,
                        _ => return Err(format!("unsupported model task: {value}")),
                    }
                }
                "--out" => options.output = Some(PathBuf::from(value)),
                _ => return Err(format!("unknown explain option: {flag}")),
            }
            index += 2;
        }
        if options.model_command.is_none() && options.model_endpoint.is_none() {
            return Err("provide --model-command PATH or --model-endpoint URL".into());
        }
        Ok(options)
    }
}

impl ReplayOptions {
    fn parse(args: Vec<String>) -> Result<Self, String> {
        let Some(program_address) = args.first().cloned() else {
            return Err(
                "usage: sos replay <program-address> --signature SIGNATURE [options]".into(),
            );
        };
        let mut signature = None;
        let mut options = Self {
            program_address,
            signature: String::new(),
            rpc_url: "https://api.mainnet.solana.com".into(),
            commitment: "finalized".into(),
            requested_slot: None,
            output: None,
        };
        let mut index = 1;
        while index < args.len() {
            let flag = &args[index];
            let Some(value) = args.get(index + 1) else {
                return Err(format!("missing value for {flag}"));
            };
            match flag.as_str() {
                "--signature" => signature = Some(value.clone()),
                "--rpc-url" => options.rpc_url = value.clone(),
                "--commitment" => options.commitment = value.clone(),
                "--slot" => {
                    options.requested_slot = Some(
                        value
                            .parse::<u64>()
                            .map_err(|error| format!("invalid slot: {error}"))?,
                    )
                }
                "--out" => options.output = Some(PathBuf::from(value)),
                _ => return Err(format!("unknown replay option: {flag}")),
            }
            index += 2;
        }
        options.signature = signature.ok_or_else(|| "missing required --signature".to_owned())?;
        Ok(options)
    }
}

impl RemoteOptions {
    fn parse(args: Vec<String>, command: &str) -> Result<Self, String> {
        let Some(program_address) = args.first().cloned() else {
            return Err(format!("usage: sos {command} <program-address> [options]"));
        };
        let mut options = Self {
            program_address,
            rpc_url: "https://api.mainnet.solana.com".into(),
            cluster: None,
            rpc_explicit: false,
            output: "artifacts".into(),
            commitment: "finalized".into(),
            requested_slot: None,
            history_limit: 0,
            json: false,
            replay_requested: false,
            model: None,
            model_command: None,
            model_endpoint: None,
            lift_provider: LiftProvider::Ollama,
            ollama_url: "http://127.0.0.1:11434".into(),
            context_tokens: 32_768,
            function_id: None,
            signature: None,
            validate_requested: false,
            repair_requested: false,
            repair_rounds: 2,
            validation_trace: None,
        };
        let mut provider_explicit = false;
        let mut index = 1;
        while index < args.len() {
            let flag = &args[index];
            if flag == "--json" {
                options.json = true;
                index += 1;
                continue;
            }
            if flag == "--replay" {
                options.replay_requested = true;
                index += 1;
                continue;
            }
            if flag == "--validate" {
                if command != "inspect" {
                    return Err("--validate is only supported by inspect".into());
                }
                options.validate_requested = true;
                index += 1;
                continue;
            }
            if flag == "--repair" {
                if command != "inspect" {
                    return Err("--repair is only supported by inspect".into());
                }
                options.repair_requested = true;
                options.validate_requested = true;
                index += 1;
                continue;
            }
            let Some(value) = args.get(index + 1) else {
                return Err(format!("missing value for {flag}"));
            };
            match flag.as_str() {
                "--rpc-url" => {
                    options.rpc_url = value.clone();
                    options.rpc_explicit = true;
                }
                "--cluster" => options.cluster = Some(value.clone()),
                "--out" => options.output = value.clone(),
                "--commitment" => options.commitment = value.clone(),
                "--history" => {
                    options.history_limit = value
                        .parse::<usize>()
                        .map_err(|error| format!("invalid history limit: {error}"))?;
                }
                "--slot" => {
                    options.requested_slot = Some(
                        value
                            .parse::<u64>()
                            .map_err(|error| format!("invalid slot: {error}"))?,
                    );
                }
                "--model" => options.model = Some(value.clone()),
                "--model-command" => options.model_command = Some(value.clone()),
                "--model-endpoint" => options.model_endpoint = Some(value.clone()),
                "--provider" if command == "inspect" => {
                    options.lift_provider = LiftProvider::parse(value)?;
                    provider_explicit = true;
                }
                "--ollama-url" if command == "inspect" => options.ollama_url = value.clone(),
                "--function" if command == "inspect" => options.function_id = Some(value.clone()),
                "--context-tokens" if command == "inspect" => {
                    options.context_tokens = value
                        .parse::<usize>()
                        .map_err(|error| format!("invalid context token count: {error}"))?;
                    if !(4_096..=131_072).contains(&options.context_tokens) {
                        return Err("--context-tokens must be between 4096 and 131072".into());
                    }
                }
                "--signature" => options.signature = Some(value.clone()),
                "--repair-rounds" if command == "inspect" => {
                    options.repair_rounds = value
                        .parse::<usize>()
                        .map_err(|_| "--repair-rounds must be an integer".to_string())?
                        .clamp(1, 8);
                }
                "--validation-trace" if command == "inspect" => {
                    options.validation_trace = Some(PathBuf::from(value));
                }
                _ => return Err(format!("unknown {command} option: {flag}")),
            }
            index += 2;
        }
        if command == "inspect" && provider_explicit && options.model.is_none() {
            options.model = Some(
                match options.lift_provider {
                    LiftProvider::Ollama => "gpt-oss:20b",
                    LiftProvider::Groq => "openai/gpt-oss-120b",
                }
                .into(),
            );
        }
        if !options.rpc_explicit {
            if let Some(cluster) = options.cluster.as_deref() {
                options.rpc_url = match cluster {
                    "mainnet-beta" | "mainnet" => "https://api.mainnet.solana.com".into(),
                    "devnet" => "https://api.devnet.solana.com".into(),
                    "testnet" => "https://api.testnet.solana.com".into(),
                    "localnet" | "localhost" => "http://127.0.0.1:8899".into(),
                    value => return Err(format!("unsupported cluster: {value}")),
                };
            }
        }
        Ok(options)
    }
}

fn resolve_remote(options: &RemoteOptions) -> Result<(ProgramArtifact, ArtifactManifest), String> {
    let provider = JsonRpcProvider::new(&options.rpc_url).with_commitment(&options.commitment);
    let account = provider
        .get_account(&options.program_address, options.requested_slot)
        .map_err(|error| error.to_string())?;
    let registry = default_registry();
    let adapter = registry
        .identify(&account)
        .ok_or_else(|| format!("unsupported program owner {}", account.owner))?;
    let artifact = adapter
        .resolve_artifact(&account, &provider)
        .map_err(|error| error.to_string())?;
    let sbpf_version = analyze_elf(&artifact.elf_bytes)
        .map_err(|error| format!("unable to determine SBPF version: {error}"))?
        .sbpf_version;
    let context = ManifestContext {
        cluster: options.cluster.clone(),
        rpc_endpoint_id: Some(options.rpc_url.clone()),
        commitment: Some(options.commitment.clone()),
        context_slot: account.context_slot,
        requested_slot: options.requested_slot,
        sbpf_version: Some(sbpf_version),
        sos_version: env!("CARGO_PKG_VERSION").into(),
    };
    let manifest = build_manifest(&artifact, &context).map_err(|error| error.to_string())?;
    Ok((artifact, manifest))
}

fn register_artifact_metadata(
    output: &str,
    manifest: &ArtifactManifest,
    artifact_path: &Path,
) -> Result<(String, PathBuf), String> {
    let metadata_path = PathBuf::from(output).join("metadata.sqlite");
    let store = MetadataStore::open(&metadata_path).map_err(|error| error.to_string())?;
    let record = store
        .register_artifact(manifest, artifact_path)
        .map_err(|error| error.to_string())?;
    Ok((record.artifact_id, metadata_path))
}

fn replay_for_inspect(
    options: &RemoteOptions,
    artifact: &ProgramArtifact,
) -> sos_replay::ExecutionTrace {
    let provider = JsonRpcProvider::new(&options.rpc_url).with_commitment(&options.commitment);
    let signature = match options.signature.clone() {
        Some(signature) => signature,
        None => match provider.get_signatures(&options.program_address, 1) {
            Ok(signatures) => match signatures.first() {
                Some(signature) => signature.signature.clone(),
                None => {
                    return sos_replay::ExecutionTrace::not_available(
                        "inspect",
                        "no transaction signatures were found for the program",
                    )
                }
            },
            Err(error) => {
                return sos_replay::ExecutionTrace::not_available(
                    "inspect",
                    format!("transaction history unavailable: {error}"),
                )
            }
        },
    };
    let transaction = match provider.get_transaction(&signature, options.requested_slot) {
        Ok(transaction) => transaction,
        Err(error) => {
            return sos_replay::ExecutionTrace::not_available(
                signature,
                format!("transaction evidence unavailable: {error}"),
            )
        }
    };
    let slot = transaction.slot.or(options.requested_slot);
    let mut accounts = Vec::new();
    for address in transaction_account_addresses(&transaction.raw_json) {
        match provider.get_account(&address, slot) {
            Ok(account) => accounts.push(account),
            Err(error) => {
                return sos_replay::ExecutionTrace::not_available_for_transaction(
                    &transaction,
                    &accounts,
                    format!(
                        "historical account state unavailable for {address}: {error}; \
                         replay was not attempted and current state was not substituted"
                    ),
                )
            }
        }
    }
    match replay_transaction(artifact, &transaction, &accounts) {
        Ok(trace) => trace,
        Err(error) => sos_replay::ExecutionTrace::not_available_for_transaction(
            &transaction,
            &accounts,
            format!("replay backend unavailable: {error}"),
        ),
    }
}

fn collect_history(options: &RemoteOptions) -> serde_json::Value {
    let provider = JsonRpcProvider::new(&options.rpc_url).with_commitment(&options.commitment);
    let signatures = match provider.get_signatures(&options.program_address, options.history_limit)
    {
        Ok(signatures) => signatures,
        Err(error) => {
            return serde_json::json!({
                "program_address": options.program_address,
                "status": "not_available",
                "error": error.to_string(),
            });
        }
    };
    let transactions = signatures
        .iter()
        .map(
            |signature| match provider.get_transaction(&signature.signature, signature.slot) {
                Ok(transaction) => serde_json::json!({
                    "status": "available",
                    "transaction": transaction,
                }),
                Err(error) => serde_json::json!({
                    "status": "not_available",
                    "signature": signature.signature,
                    "slot": signature.slot,
                    "error": error.to_string(),
                }),
            },
        )
        .collect::<Vec<_>>();
    serde_json::json!({
        "program_address": options.program_address,
        "commitment": options.commitment,
        "limit": options.history_limit,
        "signatures": signatures,
        "transactions": transactions,
    })
}

fn replay_transaction(
    artifact: &ProgramArtifact,
    transaction: &sos_types::TransactionEvidence,
    accounts: &[sos_types::AccountSnapshot],
) -> Result<sos_replay::ExecutionTrace, sos_replay::ReplayError> {
    #[cfg(feature = "litesvm")]
    {
        LiteSvmReplayEngine.replay(artifact, transaction, accounts)
    }
    #[cfg(not(feature = "litesvm"))]
    {
        let _ = accounts;
        UnavailableReplayEngine.replay(artifact, transaction, accounts)
    }
}

fn transaction_account_addresses(raw_json: &serde_json::Value) -> Vec<String> {
    let mut addresses = std::collections::BTreeSet::new();
    let transaction = raw_json.get("transaction").unwrap_or(raw_json);
    let message = transaction.get("message");
    if let Some(account_keys) = message
        .and_then(|message| message.get("accountKeys"))
        .and_then(serde_json::Value::as_array)
    {
        for account in account_keys {
            if let Some(address) = account
                .get("pubkey")
                .and_then(serde_json::Value::as_str)
                .or_else(|| account.as_str())
            {
                addresses.insert(address.to_owned());
            }
        }
    }
    if let Some(loaded) = raw_json
        .get("meta")
        .and_then(|meta| meta.get("loadedAddresses"))
    {
        for field in ["writable", "readonly"] {
            if let Some(values) = loaded.get(field).and_then(serde_json::Value::as_array) {
                for address in values.iter().filter_map(serde_json::Value::as_str) {
                    addresses.insert(address.to_owned());
                }
            }
        }
    }
    addresses.into_iter().collect()
}

fn usage() -> ExitCode {
    eprintln!("usage: sos artifact-info <path-to-sbf-elf> [--json]");
    eprintln!("       sos manifest <path-to-sbf-elf>");
    eprintln!("       sos fetch <program-address> [--cluster NAME] [--rpc-url URL] [--out DIR] [--slot SLOT] [--commitment LEVEL] [--json]");
    eprintln!("       sos inspect <program-address> [--cluster NAME] [--rpc-url URL] [--out DIR] [--slot SLOT] [--history N] [--replay] [--signature SIGNATURE] [--provider ollama|groq] [--model MODEL] [--function fn_ID] [--ollama-url URL] [--context-tokens N] [--model-command PATH|--model-endpoint URL] [--validate] [--repair --repair-rounds N] [--validation-trace PATH] [--commitment LEVEL] [--json]");
    eprintln!("       sos analyze <path-to-sbf-elf> [--out DIR]");
    eprintln!("       sos detect <analysis-directory> [--format json|sarif] [--out PATH]");
    eprintln!("       sos validate <analysis-directory> [--compiler PATH|--no-compile] [--trace PATH] [--repair --model-command PATH --repair-rounds N] [--out PATH]");
    eprintln!("       sos explain <analysis-directory> --model-command PATH|--model-endpoint URL [--model NAME] [--task structure|semantic|explanation|refinement] [--out PATH]");
    eprintln!(
        "       sos replay <program-address> --signature SIGNATURE [--rpc-url URL] [--out PATH]"
    );
    eprintln!(
        "       sos report <analysis-directory> [--format markdown|json|graphviz] [--out PATH]"
    );
    eprintln!(
        "       sos decompile <analysis-directory> [--format rust-like|c-like|json] [--out PATH]"
    );
    eprintln!("       sos lift <analysis-directory> [--provider ollama|groq] [--function fn_ID] [--model MODEL] [--ollama-url URL] [--context-tokens N] [--repair-rounds N] [--out DIR]");
    eprintln!("       sos humanize <pseudocode-file> [--plan] [--provider groq|ollama] [--model MODEL] [--max-units N] [--out PATH] [--audit-out PATH]");
    eprintln!("       sos benchmark <artifact> [--repeats N] [--out PATH]");
    ExitCode::from(2)
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::{Arc, Mutex};

    use super::{
        run_humanization_units, transaction_account_addresses, BenchmarkOptions, ExplainOptions,
        HumanizeOptions, LiftOptions, LiftProvider, RemoteOptions,
    };
    use sos_ai::{AiError, ModelProvider, ModelRequest, ModelResponse};

    #[test]
    fn transaction_accounts_include_static_and_loaded_keys_without_duplicates() {
        let value = serde_json::json!({
            "transaction": {
                "message": {
                    "accountKeys": [
                        {"pubkey": "static-a"},
                        {"pubkey": "static-b"},
                        {"pubkey": "static-a"}
                    ]
                }
            },
            "meta": {
                "loadedAddresses": {
                    "writable": ["loaded-a"],
                    "readonly": ["static-b", "loaded-b"]
                }
            }
        });
        assert_eq!(
            transaction_account_addresses(&value),
            vec![
                "loaded-a".to_owned(),
                "loaded-b".to_owned(),
                "static-a".to_owned(),
                "static-b".to_owned()
            ]
        );
    }

    #[test]
    fn explain_accepts_a_hosted_model_endpoint_without_a_local_command() {
        let options = ExplainOptions::parse(vec![
            "./analysis".into(),
            "--model".into(),
            "hosted".into(),
            "--model-endpoint".into(),
            "http://127.0.0.1:8080/generate".into(),
        ])
        .unwrap();
        assert!(options.model_command.is_none());
        assert_eq!(
            options.model_endpoint.as_deref(),
            Some("http://127.0.0.1:8080/generate")
        );
    }

    #[test]
    fn lift_accepts_a_function_scope_and_rejects_unreviewed_models() {
        let options = LiftOptions::parse(vec![
            "./analysis".into(),
            "--function".into(),
            "fn_10".into(),
        ])
        .unwrap();
        assert_eq!(options.function_id.as_deref(), Some("fn_10"));
        assert_eq!(options.repair_rounds, 2);
        assert_eq!(
            LiftOptions::parse(vec![
                "./analysis".into(),
                "--repair-rounds".into(),
                "0".into(),
            ])
            .unwrap()
            .repair_rounds,
            0
        );
        assert!(LiftOptions::parse(vec![
            "./analysis".into(),
            "--model".into(),
            "unknown:latest".into(),
        ])
        .is_err());

        let hosted = LiftOptions::parse(vec![
            "./analysis".into(),
            "--provider".into(),
            "groq".into(),
        ])
        .unwrap();
        assert_eq!(hosted.provider, LiftProvider::Groq);
        assert_eq!(hosted.model, "openai/gpt-oss-120b");
    }

    #[test]
    fn inspect_accepts_reviewed_ollama_lift_options() {
        let options = RemoteOptions::parse(
            vec![
                "11111111111111111111111111111111".into(),
                "--model".into(),
                "gpt-oss:20b".into(),
                "--function".into(),
                "fn_0".into(),
                "--context-tokens".into(),
                "16384".into(),
                "--ollama-url".into(),
                "http://127.0.0.1:11434".into(),
            ],
            "inspect",
        )
        .unwrap();
        assert_eq!(options.model.as_deref(), Some("gpt-oss:20b"));
        assert_eq!(options.function_id.as_deref(), Some("fn_0"));
        assert_eq!(options.context_tokens, 16_384);
        assert!(options.model_command.is_none());
        assert!(options.model_endpoint.is_none());
    }

    #[test]
    fn inspect_provider_groq_selects_the_reviewed_hosted_default() {
        let options = RemoteOptions::parse(
            vec![
                "11111111111111111111111111111111".into(),
                "--provider".into(),
                "groq".into(),
                "--function".into(),
                "fn_0".into(),
            ],
            "inspect",
        )
        .unwrap();
        assert_eq!(options.lift_provider, LiftProvider::Groq);
        assert_eq!(options.model.as_deref(), Some("openai/gpt-oss-120b"));
    }

    #[test]
    fn humanize_defaults_to_hosted_gpt_oss_and_allows_explicit_local_mode() {
        let hosted = HumanizeOptions::parse(vec!["decompiler.txt".into()]).unwrap();
        assert_eq!(hosted.provider, LiftProvider::Groq);
        assert_eq!(hosted.model, "openai/gpt-oss-120b");
        assert_eq!(hosted.output, PathBuf::from("decompiler.readable.rs"));
        assert_eq!(hosted.max_units, 16);
        assert!(!hosted.plan_only);

        let plan = HumanizeOptions::parse(vec!["decompiler.txt".into(), "--plan".into()]).unwrap();
        assert!(plan.plan_only);

        let local = HumanizeOptions::parse(vec![
            "decompiler.txt".into(),
            "--provider".into(),
            "ollama".into(),
        ])
        .unwrap();
        assert_eq!(local.provider, LiftProvider::Ollama);
        assert_eq!(local.model, "gpt-oss:20b");

        assert!(HumanizeOptions::parse(vec![
            "decompiler.txt".into(),
            "--out".into(),
            "decompiler.txt".into(),
        ])
        .is_err());
        assert!(HumanizeOptions::parse(vec![
            "decompiler.txt".into(),
            "--max-units".into(),
            "513".into(),
        ])
        .is_err());
    }

    #[test]
    fn humanize_merges_every_automatically_partitioned_unit() {
        #[derive(Clone)]
        struct FakeHumanizer {
            contexts: Arc<Mutex<Vec<Vec<String>>>>,
        }
        impl ModelProvider for FakeHumanizer {
            fn generate_structured(&self, request: ModelRequest) -> Result<ModelResponse, AiError> {
                self.contexts.lock().unwrap().push(
                    request
                        .evidence_pack
                        .untrusted_evidence
                        .iter()
                        .map(|item| item.source.clone())
                        .collect(),
                );
                let function_id = request.evidence_pack.ir.functions[0].id.clone();
                let operation_id = request.evidence_pack.ir.operations[0].id.clone();
                Ok(ModelResponse {
                    task: request.task,
                    model: request.model,
                    prompt_version: request.prompt_version,
                    structured_output: serde_json::json!({
                        "functions": [{
                            "function_id": function_id,
                            "pseudocode": format!("fn recovered_{operation_id}() {{}}"),
                            "evidence": [{
                                "source": "ir",
                                "location": format!("operation:{operation_id}"),
                                "detail": "first operation in bounded unit"
                            }]
                        }]
                    }),
                    evidence: Vec::new(),
                })
            }
        }

        let source = (1..=300)
            .map(|line| format!("value_{line} = value_{line} + 1;"))
            .collect::<Vec<_>>()
            .join("\n");
        let units = sos_ai::partition_pseudocode(&source).unwrap();
        let contexts = Arc::new(Mutex::new(Vec::new()));
        let run = run_humanization_units(
            FakeHumanizer {
                contexts: contexts.clone(),
            },
            "test",
            &units,
        );
        assert_eq!(units.len(), 3);
        assert_eq!(run.failures, 0);
        assert_eq!(run.readable_fragments.len(), 3);
        assert_eq!(run.unit_audits.len(), 3);
        assert!(run.readable_fragments[1].contains("lines 129-256"));
        let contexts = contexts.lock().unwrap();
        assert_eq!(contexts.len(), 3);
        assert!(contexts[1]
            .iter()
            .any(|source| source == "accepted_humanization:unit_0001"));
    }

    #[test]
    fn humanize_processes_dependencies_first_but_merges_in_source_order() {
        type RecordedRequests = Arc<Mutex<Vec<(String, Vec<String>)>>>;

        #[derive(Clone)]
        struct RecordingHumanizer {
            requests: RecordedRequests,
        }
        impl ModelProvider for RecordingHumanizer {
            fn generate_structured(&self, request: ModelRequest) -> Result<ModelResponse, AiError> {
                let function = request.evidence_pack.ir.functions[0].clone();
                let operation_id = request.evidence_pack.ir.operations[0].id.clone();
                self.requests.lock().unwrap().push((
                    function.name.clone(),
                    request
                        .evidence_pack
                        .untrusted_evidence
                        .iter()
                        .map(|item| item.source.clone())
                        .collect(),
                ));
                Ok(ModelResponse {
                    task: request.task,
                    model: request.model,
                    prompt_version: request.prompt_version,
                    structured_output: serde_json::json!({
                        "functions": [{
                            "function_id": function.id,
                            "pseudocode": format!("fn {}() {{}}", function.name),
                            "evidence": [{
                                "source": "ir",
                                "location": format!("operation:{operation_id}"),
                                "detail": "bounded function evidence"
                            }]
                        }]
                    }),
                    evidence: Vec::new(),
                })
            }
        }

        let source =
            "undefined8 caller()\n{\n  return helper();\n}\nundefined8 helper()\n{\n  return 7;\n}";
        let units = sos_ai::partition_pseudocode(source).unwrap();
        let requests = Arc::new(Mutex::new(Vec::new()));
        let run = run_humanization_units(
            RecordingHumanizer {
                requests: requests.clone(),
            },
            "test",
            &units,
        );
        assert_eq!(run.failures, 0);
        let requests = requests.lock().unwrap();
        assert_eq!(requests[0].0, "helper");
        assert_eq!(requests[1].0, "caller");
        assert!(requests[1]
            .1
            .iter()
            .any(|source| source == "accepted_humanization:unit_0002"));
        assert!(run.readable_fragments[0].contains("unit_0001"));
        assert!(run.readable_fragments[1].contains("unit_0002"));
    }

    #[test]
    fn benchmark_options_bound_repeat_count() {
        let options =
            BenchmarkOptions::parse(vec!["program.so".into(), "--repeats".into(), "4".into()])
                .unwrap();
        assert_eq!(options.repeats, 4);
        assert!(BenchmarkOptions::parse(
            vec!["program.so".into(), "--repeats".into(), "9".into(),]
        )
        .is_err());
    }
}
