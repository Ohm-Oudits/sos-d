# SOS Decompiler — End-to-End Product Specification

Status: proposed implementation contract

## 1. Purpose

SOS Decompiler is an open-source, read-only reverse-engineering system for Solana
programs. Given a deployed program address and an RPC endpoint, it produces a
reproducible analysis bundle that explains what can be established from the exact
on-chain executable, what was recovered by deterministic analysis, and what remains
a semantic hypothesis.

The product is a decompiler and evidence system, not a source-code recovery oracle.
Optimized or stripped binaries do not contain enough information to recover original
names, comments, types, or source structure reliably. Every reconstructed detail must
therefore carry a status and evidence references.

## 2. User contract

The primary workflow is:

```bash
sos inspect <PROGRAM_ADDRESS> \
  --rpc-url <RPC_URL> \
  --commitment finalized \
  --history 100 \
  --out ./reports/<PROGRAM_ADDRESS>
```

The command must be read-only with respect to the chain. It may create local files,
run a sandboxed local replay, and call an explicitly configured model provider. It must
never require a wallet or submit a transaction.

The minimum useful output is:

```text
report directory/
  manifest.json       exact artifact and acquisition provenance
  analysis.json       deterministic ELF/sBPF facts
  ir.json             stable intermediate representation
  semantics.json      Solana-specific candidates and evidence
  traces/             optional historical transactions and replay results
  report.md           human-readable conservative explanation
  pseudocode.rs       Rust-like output with uncertainty annotations
  pseudocode.c        C-like output with uncertainty annotations
  model.json          optional five-stage AI audit bundle
  ai-report.md        optional human-readable AI hypotheses and stage audit
  validation.json     optional compilation/replay validation status
  binary-judge.json   optional original-vs-recompiled structural comparison
```

## 3. Trust model and evidence statuses

All output objects use one of these statuses:

| Status | Meaning |
| --- | --- |
| `fact` | Directly observed in ELF bytes, RPC response, or a trace. |
| `recovered` | Deterministically derived from facts, such as a CFG edge. |
| `hypothesis` | Heuristic or AI interpretation that may be wrong. |
| `behaviorally_validated` | A hypothesis matched a selected replay or observed trace. |
| `not_available` | The required bytes, history, or provider capability was unavailable. |
| `unverified` | Produced but not yet checked by the relevant validator. |

Every non-trivial claim must include:

- a stable object identifier;
- confidence, when it is not a direct fact;
- one or more evidence references;
- the analysis/tool/model version that produced it;
- a reproducible byte range, instruction address, trace location, or RPC record.

AI output is never allowed to upgrade a hypothesis to a fact. Only deterministic
validation or observed behavior can do that. In particular, the model gateway
rejects a model-authored `validated` refinement status; compiler and replay
validators must supply that status independently.

## 4. Functional requirements

### 4.1 Address and artifact acquisition

The resolver must:

1. Validate the program address and RPC configuration.
2. Fetch the program account with base64 account data and record the RPC context slot.
3. Classify the owner as native loader, legacy BPF loader, upgradeable loader v3,
   loader v4, or unsupported.
4. Resolve ProgramData for upgradeable programs using the official loader state
   representation rather than a business-logic magic offset.
5. Extract the canonical ELF bytes and preserve the raw account bytes.
6. Validate ELF magic, class, machine, bounds, and SBPF compatibility.
7. Record program address, loader, ProgramData address, deployment slot, upgrade
   authority presence, commitment, requested slot, and SHA-256 hashes.
8. Store artifacts immutably under their ELF hash.

The MVP supports latest reads and provider-dependent historical reads. A requested
slot must never be represented as exact historical proof unless the provider confirms
that the account and transaction state came from that slot. If the provider can only
offer a minimum context slot, the manifest must say so.

### 4.2 Deterministic binary analysis

The deterministic core must use runtime-aligned SBPF decoding and provide:

- SBPF version and ELF segment/section facts;
- instruction records with address, opcode, operands, immediates, and assembly;
- entrypoint and function candidates with confidence;
- basic blocks, direct/indirect branch edges, and CFG;
- call graph, tail calls, recursion/SCCs, and unresolved calls;
- register liveness, stack slots, frame bounds, and memory-offset accesses;
- constants, strings, public-key-shaped values, format strings, and relocations;
- syscall calls, known signatures, and memory effects;
- verifier/runtime diagnostics.

The binary analysis output must be deterministic for the same artifact, tool version,
and options. It must be serializable as versioned JSON and usable without an AI model.

### 4.3 Solana semantic recovery

The semantic layer must build a graph containing:

- program entrypoint and instruction candidates;
- instruction discriminators/tags and argument layouts;
- account-role and account-field candidates;
- signer, writable, executable, owner, and key-equality checks;
- syscall and CPI edges, including target-program candidates;
- PDA seed and bump candidates;
- framework fingerprints for Anchor, native Rust, Pinocchio, and common protocols;
  known-program definitions must be versioned and extensible through a public
  registry rather than embedded as irreversible report logic;
- error-code, log-string, and return-data associations.

Anchor recognition is probabilistic. An eight-byte constant must not be labeled as an
Anchor discriminator without supporting control-flow, parsing, or transaction
evidence.

The initial ABI model follows Solana's standard BPF entrypoint deserializer: an
account count, variable-length account records, instruction-data length and bytes,
then the executing program ID. Account-record offsets must remain symbolic because
duplicate-account records, account-data lengths, growth space, and alignment make a
single flattened offset table unsound. The implementation reference is the official
[`entrypoint::deserialize`](https://github.com/solana-labs/solana/blob/master/sdk/program/src/entrypoint.rs)
source.

### 4.4 Historical evidence and replay

When requested and supported, SOS must ingest program-related signatures, transaction
instructions, logs, inner instructions, account keys, return data, and account
snapshots. Replay is optional and must run in an isolated local SVM/LiteSVM process.

Replay output must include status/error, logs, compute units, CPI targets and data,
account diffs, return data, available path coverage, the transaction slot, and the
context slot for every hydrated account. A historical transaction or single replay
is evidence for one path, not proof of complete behavior.

### 4.5 AI-assisted decompilation

The AI subsystem is a constrained evidence consumer with five stages:

1. Signature recovery over bounded ABI, call-site, memory-layout, and type evidence.
   Directly observed SBPF `r1`-`r5` live-ins must be mapped one-to-one to recovered
   parameters and cite the exact first-use instruction; inferred names and types
   remain hypotheses.
2. Structure recovery over bounded IR, constrained by the signature hypothesis and
   a required-effect coverage ledger. Deterministic CFG region contracts must
   separately describe branch arms/joins and loop bodies/exits/cycle edges, and
   every contract must map to a unique concrete source header citing its exact
   header block.
3. Semantic labeling using IR, strings, known fingerprints, and traces.
4. Explanation and report generation from verified graph data.
5. Bounded refinement using schema validation, compilation, and replay comparison.

Successful refinement compilation must request compiler-produced MIR and persist
a bounded, versioned CFG/effect summary. SOS compares the MIR function matching
each canonical IR function against the deterministic required-effect classes.
Missing writes, branches, loops, calls, or returns are validation mismatches and
must be supplied to a bounded repair round when repair is enabled. This gate does
not replace pinned SBPF recompilation or replay-based behavioral validation.
Incoming SBPF argument identities must be conservatively propagated through
register operations, loads, loops, and CFG joins. For each effect class, every
deterministically required input dependency must appear in the corresponding
candidate MIR effect; dropped dependencies are repairable mismatches.

Each stage returns strict JSON. Inputs are tagged as untrusted data, and model output
cannot alter facts, evidence references, or confidence values directly.

The model gateway must support local and remote providers, prompt/model versioning,
timeouts, token/size budgets, redaction, deterministic settings, and audit logs.

Reviewed model identity and demonstrated task competence are separate claims.
SOS must provide a versioned qualification corpus and command that runs the exact
model/runtime through the ordinary schema, evidence, and anti-cheese gateway,
then scores Solana, cybersecurity, and systems reconstruction concepts without
placing expected answers in the prompt. Qualification output must retain input
hashes, accepted/rejected state, missing concepts, forbidden shortcuts, per-case
scores, aggregate threshold, and limitations. Passing this bounded corpus must
not be represented as universal model trust or semantic correctness.

The local integration uses an explicitly configured subprocess protocol: one JSON
`ModelRequest` on stdin and one JSON `ModelResponse` on stdout. Hosted providers use
the same provider-neutral request/response contract over bounded HTTP. The gateway validates
task and prompt identity, output size, confidence ranges, and evidence references
before accepting the response. Address-first inspection persists one versioned
pipeline bundle with separate status and input/output hashes for signature recovery,
structure recovery, semantic labeling, explanation, and refinement; a failed stage cannot erase accepted
results or alter deterministic facts.

Complete-program lifting must use a deterministic call-graph dependency walk:
known callees are processed before callers, recursive cycles use stable address
order, and only accepted bounded callee output may enter a caller pack as
untrusted context. Resumable lifts may persist only fully accepted base-stage
function bundles. Before reuse, the implementation must rebuild the scoped
analysis, semantic facts, and dependency context and revalidate provider/model/
prompt identity, stage order, request and response hashes, schema constraints,
anti-cheese rules, and evidence references. Credentials must never enter cache
records.

Validation is an external trust boundary. `sos validate` may compile accepted
refinement snippets in a bounded, non-executing `rustc` subprocess and may compare
observed replay traces when a trace is supplied. It emits `validation.json` with
independent compilation and behavioral statuses. Successful compilation means only
that the candidate passed the configured syntax/type check; it is never behavioral
proof, and a model cannot self-assert a validated status. Refinement candidates may
also emit a bounded set of evidence-linked observations (`log_contains`, `cpi_target`,
`status`, or `return_data_base64`); only direct matches against the supplied trace
can promote those observations to `behaviorally_validated`. A repair pass receives
only capped compiler/mismatch feedback and is limited by `--repair-rounds`.

For every SBPF memory write, the deterministic dependency contract must keep
destination-address inputs separate from stored-value inputs. The structure
ledger and compiler-produced MIR summary must preserve both roles in addition
to their union. A candidate that uses the correct parameters but swaps pointer
and value roles is a mismatch and must be returned to bounded repair. MIR
validation must retain individual write role signatures and cover every distinct
required SBPF signature; aggregate dependency sets alone are insufficient.

`sos judge <original.so> <candidate.so>` is the deterministic recompilation
gate. Both inputs must pass the normal ELF/SBPF analyzer. The judge must pair
functions without trusting recovered source names and emit a versioned JSON
record containing both content hashes, SBPF versions, acceptance threshold,
exact-match requirement, per-function matches, unmatched functions, component
scores, and limitations. The score must include normalized instruction n-grams,
instruction count, CFG shape, call shape, and memory effects. Missing original
functions and unmatched candidate functions must reduce acceptance quality.

The judge must never represent structural similarity as semantic equivalence.
An exact normalized match means only that all normalized function shapes used by
this judge matched. Reproducible evaluation must pin the compiler, linker,
optimization flags, SDK, and SBPF toolchain. Behavioral claims still require
independent replay or observed execution evidence.

`sos rebuild-judge <original.so> <candidate-Cargo.toml>` must automate the
reproducible path. It rejects Solana CLI and `cargo-build-sbf` versions that do
not exactly match the configured pin, also verifies the configured platform-tools
and SBF Rust compiler versions, requires a workspace `Cargo.lock`, invokes
the builder without a shell using `--locked`, emits into a new output directory,
and enforces a bounded wall-clock time and bounded captured logs. Its versioned
record includes tool versions, exact arguments, manifest/source/lock hashes,
build duration, emitted ELF hash, and the ordinary binary-judge result. It must
never consume a pre-existing candidate artifact from the selected output path.

### 4.6 Output formats

Required formats are versioned JSON and Markdown. Rust-like pseudocode is a readable
projection of the IR and must preserve unknowns explicitly, for example:

```rust
fn fn_0042(arg_0: &[u8], accounts: &[AccountInfo]) -> Result<(), UnknownError> {
    // hypothesis: instruction tag matches candidate 0x12345678
    let field_0x18 = load_u64(arg_0, 0x18); // evidence: insn 0x2a0..0x2b2
    unknown_branch();
}
```

The report must distinguish generated pseudocode from compilable source. A report may
claim behavioral agreement only when the verification record says which traces were
matched and which checks passed.

## 5. Non-functional requirements

- Read-only by default; no signing or transaction submission dependency.
- Safe handling of hostile ELF and untrusted on-chain strings/logs.
- Resource limits for parsing, emulation, replay, and model calls.
- Reproducibility from immutable artifact hash plus manifest.
- Provider-neutral RPC and model interfaces.
- Stable schemas with explicit versioning and migrations.
- Deterministic baseline output when no AI or archive RPC is available.
- Cacheable content-addressed artifacts and analysis stages.
- Clear failure modes for unsupported loaders, unavailable history, malformed ELF,
  and incomplete replay state.

## 6. Security requirements

ELF parsing, optional third-party decompiler adapters, and replay run in separate
processes or containers with CPU, memory, file-size, and wall-clock limits. RPC
credentials never enter prompts or report bundles. On-chain strings, transaction
memos, logs, and account bytes are untrusted evidence and must never be interpreted
as instructions to the system or model.

## 7. Acceptance criteria for the complete product

The product is complete when it can, from a deployed address:

1. Fetch and hash the correct executable for legacy, upgradeable-v3, and loader-v4
   fixtures, with provenance and negative tests.
2. Reproduce deterministic instruction, function, CFG, call, syscall, constant, and
   memory facts from a pinned artifact.
3. Produce Solana semantic candidates with evidence and uncertainty.
4. Ingest representative transaction history and produce replay traces where the RPC
   and local runtime support them.
5. Generate evidence-linked Markdown and Rust-like pseudocode without inventing facts.
6. Demonstrate schema-valid AI output, prompt-injection resistance, and bounded retry.
7. Publish benchmark results for native Rust, Anchor, Pinocchio, C, optimized, and
   stripped programs across multiple compiler/SBPF versions.
8. Pass reproducibility, resource-limit, malformed-input, and regression suites.

## 8. Deliberate non-goals

SOS will not promise original source recovery, exact original names, complete type
reconstruction, universal historical availability, or proof of correctness from one
transaction. It will not execute arbitrary on-chain code with unrestricted host access
and will not depend on a proprietary GUI as its canonical analysis engine.
