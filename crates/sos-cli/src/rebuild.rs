use std::{
    ffi::OsString,
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
    process::{Command, ExitCode, Stdio},
    sync::atomic::{AtomicU64, Ordering},
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use serde::{Deserialize, Serialize};
use sos_analysis::{analyze_elf, compare_binary_analyses, BinaryComparison};
use sos_artifact::sha256_hex;

const REBUILD_JUDGE_SCHEMA_VERSION: u32 = 1;
const DEFAULT_SOLANA_VERSION: &str = "3.0.0";
const DEFAULT_PLATFORM_TOOLS_VERSION: &str = "v1.51";
const DEFAULT_SBF_RUSTC_VERSION: &str = "1.84.1";
const DEFAULT_TIMEOUT_SECONDS: u64 = 900;
const MAX_TIMEOUT_SECONDS: u64 = 3_600;
const BUILD_LOG_LIMIT: usize = 256 * 1024;
const METADATA_OUTPUT_LIMIT: usize = 4 * 1024 * 1024;
const SOURCE_HASH_LIMIT: u64 = 128 * 1024 * 1024;
const SOURCE_FILE_LIMIT: usize = 25_000;
static TEMP_FILE_COUNTER: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone)]
struct RebuildJudgeOptions {
    original: PathBuf,
    manifest: PathBuf,
    output: Option<PathBuf>,
    build_directory: Option<PathBuf>,
    solana_command: PathBuf,
    cargo_build_sbf_command: PathBuf,
    cargo_command: PathBuf,
    expected_solana_version: String,
    expected_platform_tools_version: String,
    expected_sbf_rustc_version: String,
    timeout: Duration,
    minimum_score: f64,
    require_exact: bool,
}

impl RebuildJudgeOptions {
    fn parse(args: Vec<String>) -> Result<Self, String> {
        let (Some(original), Some(manifest)) = (args.first(), args.get(1)) else {
            return Err("usage: sos rebuild-judge <original-sbf-elf> <candidate-Cargo.toml> [--expected-solana-version VERSION] [--expected-platform-tools-version VERSION] [--expected-sbf-rustc-version VERSION] [--timeout-seconds N] [--build-dir DIR] [--solana PATH] [--cargo-build-sbf PATH] [--cargo PATH] [--minimum-score 0..1] [--require-exact] [--out PATH]".into());
        };
        let mut options = Self {
            original: PathBuf::from(original),
            manifest: PathBuf::from(manifest),
            output: None,
            build_directory: None,
            solana_command: PathBuf::from("solana"),
            cargo_build_sbf_command: PathBuf::from("cargo-build-sbf"),
            cargo_command: PathBuf::from("cargo"),
            expected_solana_version: DEFAULT_SOLANA_VERSION.into(),
            expected_platform_tools_version: DEFAULT_PLATFORM_TOOLS_VERSION.into(),
            expected_sbf_rustc_version: DEFAULT_SBF_RUSTC_VERSION.into(),
            timeout: Duration::from_secs(DEFAULT_TIMEOUT_SECONDS),
            minimum_score: 0.9,
            require_exact: false,
        };
        let mut index = 2;
        while index < args.len() {
            if args[index] == "--require-exact" {
                options.require_exact = true;
                index += 1;
                continue;
            }
            let flag = &args[index];
            let Some(value) = args.get(index + 1) else {
                return Err(format!("missing value for {flag}"));
            };
            match flag.as_str() {
                "--out" => options.output = Some(PathBuf::from(value)),
                "--build-dir" => options.build_directory = Some(PathBuf::from(value)),
                "--solana" => options.solana_command = PathBuf::from(value),
                "--cargo-build-sbf" => options.cargo_build_sbf_command = PathBuf::from(value),
                "--cargo" => options.cargo_command = PathBuf::from(value),
                "--expected-solana-version" => {
                    if value.trim().is_empty() {
                        return Err("--expected-solana-version cannot be empty".into());
                    }
                    options.expected_solana_version = value.clone();
                }
                "--expected-platform-tools-version" => {
                    if value.trim().is_empty() {
                        return Err("--expected-platform-tools-version cannot be empty".into());
                    }
                    options.expected_platform_tools_version = value.clone();
                }
                "--expected-sbf-rustc-version" => {
                    if value.trim().is_empty() {
                        return Err("--expected-sbf-rustc-version cannot be empty".into());
                    }
                    options.expected_sbf_rustc_version = value.clone();
                }
                "--timeout-seconds" => {
                    let seconds = value
                        .parse::<u64>()
                        .map_err(|error| format!("invalid timeout: {error}"))?;
                    if !(1..=MAX_TIMEOUT_SECONDS).contains(&seconds) {
                        return Err(format!(
                            "--timeout-seconds must be between 1 and {MAX_TIMEOUT_SECONDS}"
                        ));
                    }
                    options.timeout = Duration::from_secs(seconds);
                }
                "--minimum-score" => {
                    options.minimum_score = value
                        .parse::<f64>()
                        .map_err(|error| format!("invalid minimum score: {error}"))?;
                    if !options.minimum_score.is_finite()
                        || !(0.0..=1.0).contains(&options.minimum_score)
                    {
                        return Err("--minimum-score must be between 0 and 1".into());
                    }
                }
                _ => return Err(format!("unknown rebuild-judge option: {flag}")),
            }
            index += 2;
        }
        Ok(options)
    }
}

#[derive(Debug, Serialize)]
struct ArtifactSummary {
    path: PathBuf,
    sha256: String,
    sbpf_version: String,
}

#[derive(Debug, Serialize)]
struct BuildProvenance {
    status: &'static str,
    expected_solana_version: String,
    expected_platform_tools_version: String,
    expected_sbf_rustc_version: String,
    solana_command: PathBuf,
    solana_version: String,
    cargo_build_sbf_command: PathBuf,
    cargo_build_sbf_version: String,
    cargo_command: PathBuf,
    cargo_version: String,
    manifest_path: PathBuf,
    manifest_sha256: String,
    workspace_lock_path: PathBuf,
    workspace_lock_sha256: String,
    package_source_root: PathBuf,
    package_source_sha256: String,
    package_source_file_count: usize,
    output_directory: PathBuf,
    command_arguments: Vec<String>,
    duration_ms: u128,
    stdout: String,
    stderr: String,
    stdout_truncated: bool,
    stderr_truncated: bool,
}

#[derive(Debug, Serialize)]
struct RebuildJudgeResult {
    schema_version: u32,
    accepted: bool,
    minimum_score: f64,
    require_exact: bool,
    limitations: Vec<&'static str>,
    build: BuildProvenance,
    original: ArtifactSummary,
    candidate: ArtifactSummary,
    comparison: BinaryComparison,
}

#[derive(Debug, Deserialize)]
struct CargoMetadata {
    packages: Vec<CargoPackage>,
    target_directory: PathBuf,
    workspace_root: PathBuf,
}

#[derive(Debug, Deserialize)]
struct CargoPackage {
    manifest_path: PathBuf,
    targets: Vec<CargoTarget>,
}

#[derive(Debug, Deserialize)]
struct CargoTarget {
    name: String,
    kind: Vec<String>,
}

#[derive(Debug)]
struct CommandOutput {
    success: bool,
    status_code: Option<i32>,
    timed_out: bool,
    duration: Duration,
    stdout: String,
    stderr: String,
    stdout_truncated: bool,
    stderr_truncated: bool,
}

pub(super) fn rebuild_judge_command(args: Vec<String>) -> ExitCode {
    let options = match RebuildJudgeOptions::parse(args) {
        Ok(options) => options,
        Err(error) => {
            eprintln!("error: {error}");
            return ExitCode::from(2);
        }
    };
    let result = match run_rebuild_judge(&options) {
        Ok(result) => result,
        Err(error) => {
            eprintln!("error: {error}");
            return ExitCode::from(1);
        }
    };
    let accepted = result.accepted;
    let score = result.comparison.overall_score;
    let rendered = serde_json::to_vec_pretty(&result).expect("rebuild judge is serializable");
    if let Some(path) = &options.output {
        if let Err(error) = write_result(path, &rendered) {
            eprintln!("error: {error}");
            return ExitCode::from(1);
        }
        println!("rebuild_judge={}", path.display());
        println!("candidate={}", result.candidate.path.display());
    } else {
        println!("{}", String::from_utf8_lossy(&rendered));
    }
    if accepted {
        ExitCode::SUCCESS
    } else {
        eprintln!("error: rebuilt candidate score {score:.6} did not satisfy the judge threshold");
        ExitCode::from(1)
    }
}

fn run_rebuild_judge(options: &RebuildJudgeOptions) -> Result<RebuildJudgeResult, String> {
    let original_path = canonical_file(&options.original, "original SBF ELF")?;
    let manifest_path = canonical_file(&options.manifest, "candidate manifest")?;
    if manifest_path.file_name().and_then(|name| name.to_str()) != Some("Cargo.toml") {
        return Err("candidate manifest must be named Cargo.toml".into());
    }
    let manifest_bytes = fs::read(&manifest_path)
        .map_err(|error| format!("{}: {error}", manifest_path.display()))?;

    let solana_version = checked_tool_version(
        &options.solana_command,
        &[OsString::from("--version")],
        options.timeout.min(Duration::from_secs(30)),
    )?;
    require_tool_version(
        &solana_version,
        &options.expected_solana_version,
        "solana-cli",
    )?;
    let cargo_build_sbf_version = checked_tool_version(
        &options.cargo_build_sbf_command,
        &[OsString::from("--version")],
        options.timeout.min(Duration::from_secs(30)),
    )?;
    require_tool_version(
        &cargo_build_sbf_version,
        &options.expected_solana_version,
        "solana-cargo-build-sbf",
    )?;
    require_tool_line(
        &cargo_build_sbf_version,
        "platform-tools",
        &options.expected_platform_tools_version,
    )?;
    require_tool_line(
        &cargo_build_sbf_version,
        "rustc",
        &options.expected_sbf_rustc_version,
    )?;
    let cargo_version = checked_tool_version(
        &options.cargo_command,
        &[OsString::from("--version")],
        options.timeout.min(Duration::from_secs(30)),
    )?;

    let metadata_args = vec![
        OsString::from("metadata"),
        OsString::from("--locked"),
        OsString::from("--no-deps"),
        OsString::from("--format-version"),
        OsString::from("1"),
        OsString::from("--manifest-path"),
        manifest_path.as_os_str().to_owned(),
    ];
    let metadata_output = run_command(
        &options.cargo_command,
        &metadata_args,
        manifest_path.parent().expect("manifest has a parent"),
        options.timeout.min(Duration::from_secs(60)),
        METADATA_OUTPUT_LIMIT,
    )?;
    if !metadata_output.success {
        return Err(command_failure("cargo metadata", &metadata_output));
    }
    if metadata_output.stdout_truncated {
        return Err("cargo metadata output exceeded the 4 MiB safety limit".into());
    }
    let metadata: CargoMetadata = serde_json::from_str(&metadata_output.stdout)
        .map_err(|error| format!("unable to parse cargo metadata: {error}"))?;
    let package = metadata
        .packages
        .iter()
        .find(|package| same_file_path(&package.manifest_path, &manifest_path))
        .ok_or_else(|| "cargo metadata did not return the candidate package".to_string())?;
    let target = package
        .targets
        .iter()
        .find(|target| target.kind.iter().any(|kind| kind == "cdylib"))
        .ok_or_else(|| "candidate package must expose a cdylib target for SBF".to_string())?;
    let package_root = manifest_path
        .parent()
        .expect("manifest has a parent")
        .to_path_buf();
    let (package_source_sha256, package_source_file_count) = hash_source_tree(&package_root)?;
    let lock_path = metadata.workspace_root.join("Cargo.lock");
    let lock_bytes = fs::read(&lock_path).map_err(|error| {
        format!(
            "{}: {error}; a locked dependency graph is required",
            lock_path.display()
        )
    })?;

    let output_directory = match &options.build_directory {
        Some(directory) => absolute_path(directory)?,
        None => unique_build_directory(&metadata.target_directory),
    };
    fs::create_dir_all(&output_directory)
        .map_err(|error| format!("{}: {error}", output_directory.display()))?;
    let candidate_path = output_directory.join(format!("{}.so", target.name));
    if candidate_path.exists() {
        return Err(format!(
            "refusing to reuse existing candidate {}; select an empty --build-dir",
            candidate_path.display()
        ));
    }

    let build_args = vec![
        OsString::from("--manifest-path"),
        manifest_path.as_os_str().to_owned(),
        OsString::from("--sbf-out-dir"),
        output_directory.as_os_str().to_owned(),
        OsString::from("--arch"),
        OsString::from("v0"),
        OsString::from("--tools-version"),
        OsString::from(&options.expected_platform_tools_version),
        OsString::from("--"),
        OsString::from("--locked"),
    ];
    let build_output = run_command(
        &options.cargo_build_sbf_command,
        &build_args,
        &metadata.workspace_root,
        options.timeout,
        BUILD_LOG_LIMIT,
    )?;
    if !build_output.success {
        return Err(command_failure("cargo-build-sbf", &build_output));
    }
    if !candidate_path.is_file() {
        return Err(format!(
            "cargo-build-sbf succeeded but did not emit expected artifact {}",
            candidate_path.display()
        ));
    }

    let original_bytes = fs::read(&original_path)
        .map_err(|error| format!("{}: {error}", original_path.display()))?;
    let candidate_bytes = fs::read(&candidate_path)
        .map_err(|error| format!("{}: {error}", candidate_path.display()))?;
    let original_analysis =
        analyze_elf(&original_bytes).map_err(|error| format!("original artifact: {error}"))?;
    let candidate_analysis =
        analyze_elf(&candidate_bytes).map_err(|error| format!("candidate artifact: {error}"))?;
    let comparison = compare_binary_analyses(&original_analysis, &candidate_analysis);
    let accepted = comparison.overall_score >= options.minimum_score
        && (!options.require_exact || comparison.exact_normalized_match);

    Ok(RebuildJudgeResult {
        schema_version: REBUILD_JUDGE_SCHEMA_VERSION,
        accepted,
        minimum_score: options.minimum_score,
        require_exact: options.require_exact,
        limitations: vec![
            "structural binary similarity is a regression signal, not proof of semantic equivalence",
            "the package source hash excludes generated target/.git content and local path dependencies outside the package root",
            "a locked dependency graph and pinned Solana tools do not independently attest downloaded dependency contents",
        ],
        build: BuildProvenance {
            status: "succeeded",
            expected_solana_version: options.expected_solana_version.clone(),
            expected_platform_tools_version: options.expected_platform_tools_version.clone(),
            expected_sbf_rustc_version: options.expected_sbf_rustc_version.clone(),
            solana_command: options.solana_command.clone(),
            solana_version,
            cargo_build_sbf_command: options.cargo_build_sbf_command.clone(),
            cargo_build_sbf_version,
            cargo_command: options.cargo_command.clone(),
            cargo_version,
            manifest_path: manifest_path.clone(),
            manifest_sha256: sha256_hex(&manifest_bytes),
            workspace_lock_path: lock_path,
            workspace_lock_sha256: sha256_hex(&lock_bytes),
            package_source_root: package_root,
            package_source_sha256,
            package_source_file_count,
            output_directory,
            command_arguments: build_args
                .iter()
                .map(|argument| argument.to_string_lossy().into_owned())
                .collect(),
            duration_ms: build_output.duration.as_millis(),
            stdout: build_output.stdout,
            stderr: build_output.stderr,
            stdout_truncated: build_output.stdout_truncated,
            stderr_truncated: build_output.stderr_truncated,
        },
        original: ArtifactSummary {
            path: original_path,
            sha256: sha256_hex(&original_bytes),
            sbpf_version: original_analysis.sbpf_version,
        },
        candidate: ArtifactSummary {
            path: candidate_path,
            sha256: sha256_hex(&candidate_bytes),
            sbpf_version: candidate_analysis.sbpf_version,
        },
        comparison,
    })
}

fn canonical_file(path: &Path, label: &str) -> Result<PathBuf, String> {
    let canonical = path
        .canonicalize()
        .map_err(|error| format!("{label} {}: {error}", path.display()))?;
    if !canonical.is_file() {
        return Err(format!("{label} {} is not a file", canonical.display()));
    }
    Ok(canonical)
}

fn absolute_path(path: &Path) -> Result<PathBuf, String> {
    if path.is_absolute() {
        Ok(path.to_path_buf())
    } else {
        std::env::current_dir()
            .map(|directory| directory.join(path))
            .map_err(|error| format!("unable to resolve current directory: {error}"))
    }
}

fn same_file_path(left: &Path, right: &Path) -> bool {
    left.canonicalize().ok().as_deref() == Some(right)
}

fn unique_build_directory(target_directory: &Path) -> PathBuf {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    target_directory
        .join("sos-rebuild")
        .join(format!("{}-{timestamp}", std::process::id()))
}

fn checked_tool_version(
    command: &Path,
    args: &[OsString],
    timeout: Duration,
) -> Result<String, String> {
    let current = std::env::current_dir()
        .map_err(|error| format!("unable to resolve current directory: {error}"))?;
    let output = run_command(command, args, &current, timeout, BUILD_LOG_LIMIT)?;
    if !output.success {
        return Err(command_failure(&command.display().to_string(), &output));
    }
    let combined = if output.stderr.trim().is_empty() {
        output.stdout.trim().to_string()
    } else if output.stdout.trim().is_empty() {
        output.stderr.trim().to_string()
    } else {
        format!("{}\n{}", output.stdout.trim(), output.stderr.trim())
    };
    if combined.is_empty() {
        return Err(format!("{} returned an empty version", command.display()));
    }
    Ok(combined)
}

fn require_tool_version(output: &str, expected: &str, tool_name: &str) -> Result<(), String> {
    let first_line = output.lines().next().unwrap_or_default();
    let mut fields = first_line.split_whitespace();
    let observed_name = fields.next().unwrap_or_default();
    let observed_version = fields.next().unwrap_or_default();
    if observed_name != tool_name || observed_version != expected {
        return Err(format!(
            "pinned toolchain mismatch: expected {tool_name} {expected}, observed {first_line:?}"
        ));
    }
    Ok(())
}

fn require_tool_line(
    output: &str,
    expected_name: &str,
    expected_version: &str,
) -> Result<(), String> {
    let observed = output.lines().find_map(|line| {
        let mut fields = line.split_whitespace();
        let name = fields.next()?;
        let version = fields.next()?;
        (name == expected_name).then_some((line, version))
    });
    match observed {
        Some((_line, version)) if version == expected_version => Ok(()),
        Some((line, _version)) => Err(format!(
            "pinned toolchain mismatch: expected {expected_name} {expected_version}, observed {line:?}"
        )),
        None => Err(format!(
            "pinned toolchain mismatch: {expected_name} version was absent from cargo-build-sbf --version"
        )),
    }
}

fn command_failure(name: &str, output: &CommandOutput) -> String {
    if output.timed_out {
        return format!(
            "{name} timed out after {} seconds; stderr: {}",
            output.duration.as_secs(),
            output.stderr.trim()
        );
    }
    format!(
        "{name} failed with status {:?}; stderr: {}",
        output.status_code,
        output.stderr.trim()
    )
}

fn run_command(
    program: &Path,
    args: &[OsString],
    current_directory: &Path,
    timeout: Duration,
    output_limit: usize,
) -> Result<CommandOutput, String> {
    let nonce = TEMP_FILE_COUNTER.fetch_add(1, Ordering::Relaxed);
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let prefix = format!("sos-command-{}-{stamp}-{nonce}", std::process::id());
    let stdout_path = std::env::temp_dir().join(format!("{prefix}.stdout"));
    let stderr_path = std::env::temp_dir().join(format!("{prefix}.stderr"));
    let stdout_file = create_log_file(&stdout_path)?;
    let stderr_file = create_log_file(&stderr_path)?;
    let started = Instant::now();
    let spawn_result = Command::new(program)
        .args(args)
        .current_dir(current_directory)
        .stdin(Stdio::null())
        .stdout(Stdio::from(stdout_file))
        .stderr(Stdio::from(stderr_file))
        .spawn();
    let mut child = match spawn_result {
        Ok(child) => child,
        Err(error) => {
            let _ = fs::remove_file(&stdout_path);
            let _ = fs::remove_file(&stderr_path);
            return Err(format!("unable to start {}: {error}", program.display()));
        }
    };
    let (status, timed_out) = loop {
        match child.try_wait() {
            Ok(Some(status)) => break (status, false),
            Ok(None) if started.elapsed() < timeout => {
                thread::sleep(Duration::from_millis(25));
            }
            Ok(None) => {
                let _ = child.kill();
                let status = child
                    .wait()
                    .map_err(|error| format!("unable to reap {}: {error}", program.display()))?;
                break (status, true);
            }
            Err(error) => {
                let _ = child.kill();
                let _ = child.wait();
                let _ = fs::remove_file(&stdout_path);
                let _ = fs::remove_file(&stderr_path);
                return Err(format!("unable to poll {}: {error}", program.display()));
            }
        }
    };
    let (stdout, stdout_truncated) = read_bounded(&stdout_path, output_limit)?;
    let (stderr, stderr_truncated) = read_bounded(&stderr_path, output_limit)?;
    let _ = fs::remove_file(&stdout_path);
    let _ = fs::remove_file(&stderr_path);
    Ok(CommandOutput {
        success: status.success() && !timed_out,
        status_code: status.code(),
        timed_out,
        duration: started.elapsed(),
        stdout,
        stderr,
        stdout_truncated,
        stderr_truncated,
    })
}

fn create_log_file(path: &Path) -> Result<File, String> {
    OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| format!("{}: {error}", path.display()))
}

fn read_bounded(path: &Path, limit: usize) -> Result<(String, bool), String> {
    let length = fs::metadata(path)
        .map_err(|error| format!("{}: {error}", path.display()))?
        .len();
    let mut bytes = Vec::with_capacity(limit.min(length as usize));
    File::open(path)
        .map_err(|error| format!("{}: {error}", path.display()))?
        .take(limit as u64)
        .read_to_end(&mut bytes)
        .map_err(|error| format!("{}: {error}", path.display()))?;
    Ok((
        String::from_utf8_lossy(&bytes).into_owned(),
        length > limit as u64,
    ))
}

fn hash_source_tree(root: &Path) -> Result<(String, usize), String> {
    let mut files = Vec::new();
    collect_source_files(root, root, &mut files)?;
    files.sort();
    if files.len() > SOURCE_FILE_LIMIT {
        return Err(format!(
            "candidate source contains more than {SOURCE_FILE_LIMIT} files"
        ));
    }
    let mut record = Vec::new();
    let mut total = 0_u64;
    for relative in &files {
        let path = root.join(relative);
        let bytes = fs::read(&path).map_err(|error| format!("{}: {error}", path.display()))?;
        total = total.saturating_add(bytes.len() as u64);
        if total > SOURCE_HASH_LIMIT {
            return Err(format!(
                "candidate source exceeds the {} MiB hashing limit",
                SOURCE_HASH_LIMIT / (1024 * 1024)
            ));
        }
        let name = relative.to_string_lossy();
        record
            .write_all(&(name.len() as u64).to_le_bytes())
            .expect("writing to Vec cannot fail");
        record
            .write_all(name.as_bytes())
            .expect("writing to Vec cannot fail");
        record
            .write_all(&(bytes.len() as u64).to_le_bytes())
            .expect("writing to Vec cannot fail");
        record
            .write_all(&bytes)
            .expect("writing to Vec cannot fail");
    }
    Ok((sha256_hex(&record), files.len()))
}

fn collect_source_files(
    root: &Path,
    directory: &Path,
    files: &mut Vec<PathBuf>,
) -> Result<(), String> {
    let entries =
        fs::read_dir(directory).map_err(|error| format!("{}: {error}", directory.display()))?;
    for entry in entries {
        let entry = entry.map_err(|error| format!("{}: {error}", directory.display()))?;
        let path = entry.path();
        let file_type = entry
            .file_type()
            .map_err(|error| format!("{}: {error}", path.display()))?;
        if file_type.is_symlink() {
            return Err(format!(
                "candidate source contains unsupported symlink {}; reproducible source hashing requires regular files",
                path.display()
            ));
        }
        if file_type.is_dir() {
            let name = entry.file_name();
            if name == "target" || name == ".git" {
                continue;
            }
            collect_source_files(root, &path, files)?;
        } else if file_type.is_file() {
            let relative = path
                .strip_prefix(root)
                .map_err(|error| format!("{}: {error}", path.display()))?;
            files.push(relative.to_path_buf());
        }
    }
    Ok(())
}

fn write_result(path: &Path, bytes: &[u8]) -> Result<(), String> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent).map_err(|error| format!("{}: {error}", parent.display()))?;
    }
    fs::write(path, bytes).map_err(|error| format!("{}: {error}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temporary_directory(label: &str) -> PathBuf {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "sos-rebuild-{label}-{}-{stamp}",
            std::process::id()
        ));
        fs::create_dir_all(&path).unwrap();
        path
    }

    #[test]
    fn options_pin_versions_and_bound_timeouts() {
        let options = RebuildJudgeOptions::parse(vec![
            "original.so".into(),
            "Cargo.toml".into(),
            "--timeout-seconds".into(),
            "45".into(),
            "--require-exact".into(),
        ])
        .unwrap();
        assert_eq!(options.expected_solana_version, "3.0.0");
        assert_eq!(options.expected_platform_tools_version, "v1.51");
        assert_eq!(options.expected_sbf_rustc_version, "1.84.1");
        assert_eq!(options.timeout, Duration::from_secs(45));
        assert!(options.require_exact);
        assert!(RebuildJudgeOptions::parse(vec![
            "original.so".into(),
            "Cargo.toml".into(),
            "--timeout-seconds".into(),
            "0".into(),
        ])
        .is_err());
    }

    #[test]
    fn version_gate_rejects_unpinned_tools() {
        assert!(require_tool_version("solana-cli 3.0.0 (src:test)", "3.0.0", "solana-cli").is_ok());
        assert!(require_tool_version(
            "solana-cargo-build-sbf 3.1.0",
            "3.0.0",
            "solana-cargo-build-sbf"
        )
        .is_err());
        let output = "solana-cargo-build-sbf 3.0.0\nplatform-tools v1.51\nrustc 1.84.1";
        assert!(require_tool_line(output, "platform-tools", "v1.51").is_ok());
        assert!(require_tool_line(output, "rustc", "1.85.0").is_err());
    }

    #[test]
    fn source_hash_is_stable_and_excludes_target() {
        let directory = temporary_directory("source-hash");
        fs::write(directory.join("Cargo.toml"), "[package]\nname='x'\n").unwrap();
        fs::create_dir_all(directory.join("src")).unwrap();
        fs::write(directory.join("src/lib.rs"), "pub fn x() {}\n").unwrap();
        let first = hash_source_tree(&directory).unwrap();
        fs::create_dir_all(directory.join("target")).unwrap();
        fs::write(directory.join("target/stale.so"), b"stale").unwrap();
        let second = hash_source_tree(&directory).unwrap();
        assert_eq!(first, second);
        fs::write(directory.join("src/lib.rs"), "pub fn y() {}\n").unwrap();
        assert_ne!(first.0, hash_source_tree(&directory).unwrap().0);
        fs::remove_dir_all(directory).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn command_runner_enforces_timeout_and_captures_logs() {
        let directory = temporary_directory("runner");
        let output = run_command(
            Path::new("/bin/sh"),
            &[
                OsString::from("-c"),
                OsString::from("printf ready; printf warning >&2"),
            ],
            &directory,
            Duration::from_secs(1),
            1024,
        )
        .unwrap();
        assert!(output.success);
        assert_eq!(output.stdout, "ready");
        assert_eq!(output.stderr, "warning");

        let timed_out = run_command(
            Path::new("/bin/sh"),
            &[OsString::from("-c"), OsString::from("sleep 2")],
            &directory,
            Duration::from_millis(50),
            1024,
        )
        .unwrap();
        assert!(timed_out.timed_out);
        assert!(!timed_out.success);
        fs::remove_dir_all(directory).unwrap();
    }
}
