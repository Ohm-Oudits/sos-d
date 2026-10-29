# SOS Decompiler — Development Plan

This plan turns [SPEC.md](SPEC.md) and [ARCHITECTURE.md](ARCHITECTURE.md) into an
incremental build sequence. Each phase ends with a usable artifact and an objective
acceptance gate. Work should be merged only when the gate and regression tests pass.

## Current baseline

Already implemented in this workspace:

- Rust workspace with versioned common types.
- Content-addressed ELF storage and manifests.
- Artifact storage validates ELF class/encoding/machine before writing bytes,
  rejecting malformed or non-SBPF inputs with typed errors.
- Artifact storage verifies ELF/raw-account hashes and byte ranges against the
  supplied manifest before indexing or writing the content-addressed payload.
- Replaceable SQLite metadata index for provenance-aware artifact identity and
  derived bundle paths (`metadata.sqlite` beside each output root).
- JSON-RPC account acquisition.
- Legacy, upgradeable-v3, and loader-v4 adapter boundaries.
- Loader adapters reject non-executable program accounts with a typed error before
  resolving bytes.
- Native Rust/SBF hello-world fixture.
- Address-to-ELF fetch against a local validator.
- Runtime-aligned SBPF ELF parsing and deterministic instruction/function/CFG JSON.
- Versioned canonical IR JSON with stable function, basic-block, and operation IDs;
  analysis bundles and AI evidence packs include this projection.
- Checked-in Draft 2020-12 JSON contracts for artifact manifests, static analysis,
  canonical IR, semantic analysis, AI evidence packs, replay traces, security
  reports, AI pipeline bundles, and report bundles; contract tests parse every schema and assert its
  declared wire version.
- Checked-in golden hello-world manifest with a contract deserialization test.
- `StaticAnalysis` now carries `schema_version`, and report/decompile JSON bundles
  carry an explicit report schema version so consumers can reject incompatible data.
- Runtime-aligned SCC and dataflow-edge facts are exported in analysis JSON and
  attached to canonical IR blocks, including recursive-component markers.
- Runtime-aligned verifier diagnostics are captured as structured analysis facts
  and rendered in reports, including an explicit clean/empty state.
- Conservative direct, indirect, and syscall call-site facts with unresolved-target
  status, plus ELF relocation facts when exposed by the object parser.
- Explicit load/store memory facts and conservative entrypoint-input field candidates.
- Bounded read-only-section string extraction and immediate-constant facts.
- Bounded read-only-section 32-byte public-key candidates with base58 rendering,
  plus evidence-linked fingerprints for common System, SPL Token, Associated Token,
  Token-2022, Metaplex Token Metadata, Compute Budget, and Memo program IDs.
- Public protocol registry with built-in definitions and custom registration through
  `analyze_with_registry`, so domain-specific program IDs can extend fingerprints
  without changing deterministic analysis logic.
- Explicit stack-frame access facts rooted at the SBPF frame register, retained in
  analysis JSON, canonical IR generation, reports, and AI evidence packs.
- Conservative intra-basic-block register-use facts with immediate/copy/arithmetic
  propagation and source PCs for call arguments.
- Versioned logical model of the standard Solana BPF entrypoint ABI, without assuming
  fixed offsets for variable-length account records.
- Provider-independent AI evidence-pack and response-validation contracts.
- Bounded AI evidence-pack builder that prioritizes semantic call/dispatch windows
  while retaining deterministic full-bundle storage as the source of truth.
- Model-facing IR projections are independently bounded, including protection
  against overlapping CFG ranges duplicating operations in evidence packs.
- Explicit local command-model provider protocol and `sos explain` integration path.
- Provider-neutral bounded HTTP model provider with optional bearer-token support;
  local subprocess and hosted transports now share the same gateway/schema checks.
- Reviewed Hugging Face Inference Providers transport for the Apache-2.0
  `openai/gpt-oss-120b:fireworks-ai` route, including strict evidence validation,
  token isolation, bounded responses, resumable provider-bound checkpoints, and
  no local model load.
- Reviewed Cloudflare Workers AI transport pinned to
  `@cf/openai/gpt-oss-120b`, with environment-only credentials, native JSON mode,
  bounded hosted evidence, local schema/evidence revalidation, resumable
  provider-bound checkpoints, and access to the documented daily free allocation.
- Local model subprocesses have bounded wall-clock execution and convert stalls into
  explicit unavailable stage results.
- AI gateway exposes bounded retry for transient provider and validation failures;
  CLI model requests use two attempts without weakening deterministic checks.
- AI gateway validation now checks non-empty evidence for non-`not_available`
  claims and validates bounded instruction/IR reference locations before accepting
  model output.
- AI gateway responses now use task-specific typed schemas for structure recovery,
  semantic labeling, explanations, and refinement; arbitrary structured output is
  rejected before it reaches reports.
- Address-first `inspect --model` now orchestrates all four AI stages and writes a
  versioned audit bundle with per-stage acceptance, rejection/unavailability,
  prompt/model identity, input hashes, and output hashes.
- Function-level lifting now follows resolved call dependencies, processes callees
  before callers, and passes bounded accepted-callee output as untrusted context.
  `lift` and address-first `inspect` support validated `--resume` checkpoints;
  every cached four-stage bundle is rebuilt and revalidated against current scoped
  evidence and dependency context before reuse.
- Accepted AI output is also rendered to a separate `ai-report.md` projection;
  deterministic reports and pseudocode remain unchanged.
- A fresh-validator address-first E2E script deploys the hello fixture and verifies
  manifest, deterministic analysis, reports, and four accepted local-model stages.
- Nested semantic/function/refinement claims are also required to carry evidence
  and are checked against bounded instruction/IR locations.
- Evidence validation now covers instruction PCs, IR IDs, read-only offsets, and
  memory base/offset locations present in the bounded pack.
- AI structure-recovery and refinement responses are constrained to known bounded
  IR function IDs, and refinement validation states use a closed status vocabulary.
- The gateway rejects model self-assertions of `validated`; only an external compiler
  or replay validator may provide behavioral validation evidence.
- `sos validate` compiles accepted refinement patches in a bounded direct `rustc`
  subprocess and emits a versioned result that keeps replay status separate; an
  optional `--trace` records observed replay comparison without calling it a
  candidate execution proof.
- `sos judge` compares an original SBPF ELF with a recompiled candidate using
  deterministic name-independent function pairing, normalized instruction
  n-grams, instruction counts, CFG/call/memory shape, unmatched-function
  penalties, content hashes, and an optional normalized exact-match gate.
- Refinement candidates can now emit bounded, evidence-linked behavior observations;
  `sos validate` compares them directly with supplied logs, CPI targets, status, and
  return data, emits mismatch records, and supports a capped compiler/mismatch-driven
  repair loop through `--repair --model-command`.
- Evidence validation rejects unsupported sources and location namespaces instead of
  accepting unverifiable model-provided coordinates.
- Conservative CPI, PDA-syscall, and early-dispatch framework candidates with
  explicit alternatives and uncertainty.
- Explicit matching for a bounded set of common Anchor `global:<instruction>`
  discriminator hashes, while retaining native-tag and numeric alternatives.
- Bounded Anchor account-discriminator candidates are attached to immediate
  constants with the same competing numeric/pointer interpretations.
- Conservative load/compare/branch account-validation candidates preserve both
  observed local guards and unresolved field interpretations for security review.
- CPI/PDA candidates now preserve recovered argument values separately from the
  register identities, so unresolved pointers are not presented as resolved targets.
- Versioned semantic graph entities for instruction, argument, account, account-field,
  log, and error candidates, each retaining evidence and status.
- Solana syscall/hash recognition, conservative dispatch candidates, and CPI syscall
  candidates with evidence references.
- Deterministic Markdown report and Rust-like pseudocode projections.
- C-like and JSON decompile projections with explicit format selection.
- CFG-faithful C-like/Rust-like pseudocode that preserves decoded instructions,
  basic-block labels, successor edges, memory effects, and syscall PCs.
- Pseudocode comments now carry explicit analysis instruction or memory evidence
  locations alongside each emitted raw operation.
- Graphviz CFG export through `sos report --format graphviz`.
- Evidence-backed security review findings for unknown syscalls, unresolved CPI
  targets, unresolved PDA seed material, unproven account-validation checks,
  entrypoint-root writes, and unchecked-arithmetic review candidates. Findings
  preserve limitations and never assert exploitability from absence of proof.
- SARIF 2.1.0 projection for security findings, preserving detector status,
  confidence, limitations, and bytecode evidence in CI-compatible output.
- Versioned replay/trace contracts with explicit `not_available` output when an
  isolated SVM backend is not configured, plus a LiteSVM 0.13 backend behind an
  explicit CLI feature with transaction-account hydration and trace normalization.
- LiteSVM now installs directly hydrated executable ELF accounts as external CPI
  programs while treating non-ELF loader metadata as ordinary account state.
- Replay traces carry a conservative semantic comparison result, distinguishing
  observed evidence, mismatches, and unexecuted or unavailable behavior.
- A reproducible LiteSVM fixture example now constructs and signs a transaction
  against the built hello-world SBF, runs it through `LiteSvmReplayEngine`, and
  asserts successful status, logs, and compute-unit capture.
- Replay trace schema v2 records the transaction slot and every hydrated account's
  context slot, making state provenance explicit for both successful and unavailable
  replay outcomes.
- Replay CLI account-hydration failures now emit a typed `not_available` trace and
  explicitly state that current account state was not substituted.
- Optional cluster selection and transaction-history evidence bundles with explicit
  `not_available` handling.
- Requested account slots are now provenance-checked: a standard RPC response whose
  context slot does not equal the requested slot is rejected as historical state
  unavailable rather than being mislabeled as exact historical state.
- JSON-RPC acquisition now applies bounded retries for rate limits and a bounded
  response body size, with typed `RateLimited` and `ResponseTooLarge` failures.
- `sos fetch`, `sos analyze`, `sos detect`, `sos replay`, and `sos inspect` CLI
  commands.
- `sos benchmark` emits versioned machine-readable artifact/analysis hashes,
  structural and semantic metrics, security-finding counts, and repeated-run
  determinism for evaluation fixtures.
- The fixture corpus now includes a second native-Rust ABI program with explicit
  instruction-tag, signer, owner, and logging behavior, plus a manifest that keeps
  a reproducible C/SBF fixture entry built through the active Solana platform-tools SDK.
- The fixture corpus now includes a buildable, pinned Pinocchio 0.9.0 program with
  explicit tag dispatch, signer/owner validation, and logging targets.
- The fixture corpus now includes a buildable, pinned Anchor 0.31.1 program with
  global discriminators, account initialization, `has_one` authorization, and
  error/logging targets.
- `fetch` and address-based `inspect` register artifacts in an idempotent SQLite
  metadata index and report the provenance-aware artifact ID.
- Remote fetch manifests now include the detected SBPF version and are emitted both
  in content-addressed storage and at the requested output root for direct bundle
  consumption.
- `analyze` and address-based `inspect` compose security detection into the output
  bundle as `security.json`.
- Machine-readable JSON summaries for address-based fetch and inspect workflows.
- Apache-2.0 license text plus contributor and security-reporting guidance for
  open-source collaboration.
- Standalone cargo-fuzz targets for ELF/SBPF analysis, loader-state parsing,
  and model-response/evidence validation, with offline execution guidance.
- Least-privilege GitHub CI for workspace formatting, tests, Clippy, and fuzz
  target compilation, with an opt-in SBF fixture job for runners carrying the
  pinned Solana 3.0.0 toolchain.
- Explicit `--json` summaries for local `analyze`, `detect`, and `artifact-info`
  output-path workflows; report/decompile expose JSON through their format switch.
- Address-only `inspect` accepts `--replay` and `--model` requests, writes an
  explicit replay status when no signature is available, selects the latest
  transaction by default or an explicit `--signature`, and executes a configured
  local model command through the same bounded gateway used by `sos explain`.
- Manifest construction uses a typed provenance context, and the workspace passes
  strict Clippy checks with warnings denied.

The verified local fixture currently produces SBPF V0 analysis with 1,726 decoded
instructions, 39 function candidates, and 427 basic blocks. Address-first `inspect`
now composes deterministic analysis, semantic recovery, reports, AI stages, external
candidate validation, and a bounded repair path; generated source remains a
non-executing hypothesis unless a separately isolated runtime validator supplies
trace evidence.

## Phase 0 — Repository and safety foundation

Deliverables:

- schema migration policy and compatibility rules for manifests, static analysis,
  IR, evidence, traces, and reports;
- error taxonomy and stable CLI exit codes;
- fixture corpus metadata and reproducible build scripts;
- threat model covering ELF parsing, RPC data, prompt injection, replay, and model
  providers;
- resource-limit configuration for every potentially untrusted operation.

Tasks:

1. Add golden JSON fixtures and schema migration tests for every emitted bundle.
3. Add malformed ELF/account fixtures and fuzz targets for loader parsing.
4. Add CI for host tests, SBF fixture build, formatting, clippy, and reproducibility.

Gate: a clean checkout can build the fixture and validate every checked-in golden
manifest without network access.

## Phase 1 — Address-to-ELF acquisition MVP

Deliverables:

- `sos fetch ADDRESS` with custom RPC, commitment, output, and optional slot;
- immutable artifact plus provenance manifest;
- loader-v2/v3/v4 and direct-code fixture coverage;
- clear unsupported-loader and historical-provider errors.

Tasks:

1. Add provider-neutral history methods and archive-provider capability flags.
2. Replace minimum-context semantics with exact-slot adapters where supported.
3. Validate ELF bounds and machine before storing an artifact.
4. Add upgrade-history tests proving two deployments create two artifact hashes.
5. Add RPC retry/backoff, request limits, and response-size limits.

Gate: a deployed address can be fetched reproducibly, and its manifest explains
exactly which bytes were used or why exact history was unavailable.

## Phase 2 — Deterministic SBPF analysis

Deliverables:

- stable analysis JSON and IR;
- disassembly, CFG, call graph, stack/frame and register dataflow;
- constants, strings, relocations, syscall registry, and verifier diagnostics;
- Graphviz/JSON exports for tooling and UI.

Tasks:

1. Split `sos-analysis` into ELF normalization, decode, CFG, dataflow, and IR modules.
2. Define stable IDs for instructions, blocks, functions, values, and memory regions.
3. Add indirect branch and tail-call confidence classes.
4. Implement an SSA-like IR with explicit unknown values and memory effects.
5. Build a syscall database from runtime-aligned definitions and test call extraction.
6. Compare selected results with Anza SBPF and optional external tools.

Gate: deterministic facts match source-built fixtures across optimization levels and
remain byte-identical on repeated runs.

## Phase 3 — Solana semantic graph

Deliverables:

- instruction candidates;
- account-role and field-layout candidates;
- syscall/CPI graph;
- Anchor/native/Pinocchio fingerprints;
- known-program and error databases.

Tasks:

1. Model the Solana input buffer, account array, and runtime return conventions.
2. Detect discriminator/tag comparisons with alternative interpretations.
3. Recover account checks, load/store layouts, and serialization boundaries.
4. Detect CPI construction, target programs, PDA seeds, and bumps.
5. Associate logs, errors, and return data with graph nodes.
6. Add evidence-preserving protocol plugins instead of hardcoded report logic.

Gate: benchmark fixtures produce precision/recall reports for dispatch, account roles,
CPI targets, and common framework fingerprints; uncertain cases remain candidates.

## Phase 4 — Conservative pseudocode and reports

Deliverables:

- baseline non-AI pseudocode renderer;
- Markdown, JSON, Graphviz, SARIF, and Rust-like projections;
- evidence navigation from every claim to bytes/IR/trace.

Tasks:

1. Render CFG and typed IR with generic names first.
2. Add a report model that cannot lose status, confidence, or evidence references.
3. Render unknown operations explicitly rather than guessing source constructs.
4. Add report snapshots and golden tests.
5. Add `sos report` and make `sos inspect` compose acquisition, analysis, semantics,
   and reporting.

Gate: a report is useful without a model and never presents a hypothesis as a fact.

## Phase 5 — AI evidence gateway

Deliverables:

- local/remote model provider trait;
- bounded evidence-pack builder;
- structure, labeling, explanation, and validation agents;
- prompt/model audit records and strict JSON validation.

Tasks:

1. Define JSON schemas for each agent input/output.
2. Build compact function-level evidence packs with byte/IR references.
3. Enforce output budgets, timeouts, retries, and no-fact-invention rules.
4. Add local-model support and provider-neutral configuration.
5. Add prompt-injection corpus using hostile strings/logs/memos.
6. Store model ID, prompt version, temperature, input hash, and output hash.

Gate: invalid or adversarial model output is rejected or downgraded to an explicit
hypothesis; deterministic facts remain unchanged.

## Phase 6 — History, replay, and refinement

Deliverables:

- transaction history ingestion;
- account hydration and trace bundles;
- isolated local replay;
- automatic candidate builds under a pinned Solana/SBPF compiler and linker;
- type-layout and deeper call-graph checks around the deterministic binary judge;
- bounded candidate-vs-trace comparison and repair loop.

Tasks:

1. Add signatures, parsed transactions, logs, inner instructions, and account state.
2. Define a replay input bundle with deterministic slot/state provenance.
3. Run replay in a resource-limited worker process.
4. Normalize status, logs, compute units, CPIs, return data, and account diffs.
5. Compare candidate observations with observed traces and emit mismatch records.
6. Permit only bounded, auditable AI repair passes.
7. Rebuild whole-program candidates with a pinned toolchain and retain binary-judge
   results beside replay evidence.

Gate: a replay result is reproducible, read-only, and clearly scoped to its observed
state/path; mismatches cannot be silently hidden.

## Phase 7 — Product surface and ecosystem

Deliverables:

- web/API client with synchronized bytes, CFG, IR, report, and traces;
- plugin API for protocols and model providers;
- public benchmark and evaluation harness;
- optional Ghidra/Binary Ninja comparison adapters.

Tasks:

1. Expose versioned HTTP/JSON APIs around immutable analysis bundles.
2. Add browser views for evidence, confidence, and alternate hypotheses.
3. Publish corpus/build recipes and benchmark metrics.
4. Add cache/index service without making it a source of truth.
5. Document contribution and detector-plugin compatibility rules.

Gate: an external contributor can add a detector, run the corpus, and inspect its
claims without access to private infrastructure.

## Test and benchmark matrix

The checked-in corpus manifest is `fixtures/corpus.toml`. The available native-Rust,
Pinocchio, Anchor, and C fixtures can be built and benchmarked with
`scripts/benchmark-fixtures.sh`; planned entries are excluded from published scores
until their source and build toolchains are pinned.

Every release should include:

- legacy, upgradeable-v3, and loader-v4 acquisition fixtures;
- native Rust, Anchor, Pinocchio, and C programs;
- debug, release, LTO, stripped, and multiple compiler/SBPF versions;
- malformed/truncated ELF and loader account fixtures;
- programs with indirect branches, syscalls, CPIs, PDAs, errors, and custom data;
- hostile on-chain strings and prompt-injection payloads;
- replay cases with success, failure, account mutation, and missing-history states.

Track separately:

- instruction/function/CFG correctness;
- syscall, CPI, discriminator, account-role, and field-layout precision/recall;
- schema-valid AI output and evidence coverage;
- recompilation and behavioral agreement rates;
- latency, memory, RPC volume, cache hits, and resource-limit failures.

The headline quality metric is evidence-supported behavioral understanding, not visual
similarity to Rust source.

## Recommended implementation order after the current baseline

1. Add ELF validation and golden schemas.
2. Split deterministic analysis into stable IR modules.
3. Add syscall and Solana input-buffer semantics.
4. Implement the baseline pseudocode/report renderer.
5. Add transaction evidence and replay contracts.
6. Add the AI gateway only after deterministic evidence packs are stable.
7. Add web UX and protocol plugins after report schemas settle.
