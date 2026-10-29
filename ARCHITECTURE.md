# SOS Decompiler — Architecture Plan

## 1. Product definition

SOS Decompiler is an open-source, read-only Solana program reverse-engineering system.
Its primary input is:

```text
program address + cluster/RPC endpoint
```

Its primary output is a versioned analysis bundle containing:

- the exact on-chain executable bytes and provenance;
- ELF and sBPF metadata;
- disassembly, function boundaries, control-flow graphs, and call graphs;
- Solana syscall and cross-program-invocation identification;
- inferred instruction layouts, account layouts, and framework fingerprints;
- readable C/Rust-like pseudocode;
- historical transaction evidence and optional local replay results;
- confidence scores and bytecode/trace evidence for every semantic inference.

The system must distinguish three different products:

1. **Facts** — directly observed in the ELF, RPC response, transaction, or replay.
2. **Recovered structure** — derived by deterministic binary analysis.
3. **Semantic hypotheses** — inferred by heuristics or AI and always linked to evidence.

This distinction is the central trust boundary of the product.

## 2. Research conclusions

### Solana deployment model

Solana compiles Rust/C programs to ELF files containing Solana's sBPF bytecode. The
ELF is stored on-chain in an executable account, and sBPF is Solana-specific rather
than ordinary Linux eBPF. Programs use one runtime entrypoint; instruction dispatch,
account parsing, and CPI routing are generally implemented inside the program.

For the current upgradeable loader, the user-facing program address is a small
Program account that points to a separate ProgramData account. The executable ELF
bytes follow ProgramData metadata. Legacy loaders can store the ELF directly in the
program account, while newer loader variants require their own account adapter.

Therefore, `address -> getAccountInfo -> loader adapter -> canonical ELF` must be a
first-class pipeline, not a command-line convenience.

Sources:

- [Solana program execution](https://solana.com/docs/core/programs/program-execution)
- [Solana program deployment](https://solana.com/docs/core/programs/program-deployment)
- [Solana account types](https://solana.com/docs/core/accounts/account-types)
- [Solana reading from the network](https://solana.com/docs/intro/quick-start/reading-from-network)
- [UpgradeableLoaderState](https://docs.rs/solana-loader-v3-interface/latest/solana_loader_v3_interface/state/enum.UpgradeableLoaderState.html)

### Existing reverse-engineering work

The strongest existing ideas should be treated as references, not as the final
architecture:

- `solana-sbpf` provides the runtime-aligned interpreter, verifier, assembler,
  disassembler, ELF support, and SBPF version model.
- `sol-azy` demonstrates a useful modular split between fetching, disassembly,
  immediate-data tracking, CFG generation, and rule-based analysis.
- OtterSec's Binary Ninja plugin demonstrates the importance of Solana-specific
  memory maps, relocations, syscall signatures, and SDK types.
- `soldecomp` demonstrates the value of combining offline binary analysis with
  historical transaction capture, account hydration, replay, probing, and a labeled
  benchmark. It is explicitly released as a research workbench and is no longer
  maintained, so SOS should learn from its workflow without depending on it.

Sources:

- [Anza SBPF](https://github.com/anza-xyz/sbpf)
- [sol-azy reverse engineering](https://fuzzinglabs.github.io/sol-azy/reverse.html)
- [sol-azy architecture](https://fuzzinglabs.github.io/sol-azy/architecture.html)
- [Reverse engineering Solana with Binary Ninja](https://osec.io/blog/reverse-engineering-solana/)
- [soldecomp](https://github.com/aaronjmars/soldecomp)

### Solana-specific semantic evidence

Transaction history is not merely a UI feature. It is a semantic side channel:
logs expose instruction names and values, CPI traces expose target programs and
serialized instruction data, and account changes expose read/write behavior.

Anchor programs provide especially useful fingerprints. Anchor normally places an
instruction discriminator at the beginning of instruction data and account
discriminators at the beginning of account data. However, recent Anchor versions
allow custom discriminator lengths, and native/Pinocchio programs may use completely
different formats. The detector must treat Anchor recognition as probabilistic and
must never assume that every eight-byte prefix is an Anchor discriminator.

Sources:

- [Anchor IDL and discriminators](https://www.anchor-lang.com/docs/basics/idl)
- [Anchor program structure](https://www.anchor-lang.com/docs/basics/program-structure)
- [Solana RPC methods](https://solana.com/docs/rpc/http)

### AI-assisted decompilation research

Recent binary-decompilation research supports a staged architecture:

1. Recover a structurally valid intermediate representation.
2. Recover names and semantic labels separately.
3. Use compilation, execution, and I/O feedback to refine output.

Directly asking an LLM to turn a long disassembly into Rust is unreliable. Research
shows that conventional decompiler output can be readable while behaviorally wrong,
that identifiers and source-level types are often irretrievably lost, and that
obfuscation/optimization causes large accuracy drops. SOS should therefore use the
LLM as a constrained analyst over a compact, typed evidence graph rather than as the
source of truth.

Sources:

- [LLM4Decompile](https://arxiv.org/abs/2403.05286)
- [SK2Decompile](https://arxiv.org/abs/2509.22114)
- [ASAP assembly-source aligned refinement](https://arxiv.org/abs/2610.06900)
- [AutoDecompiler feedback-driven refinement](https://arxiv.org/abs/2606.16162)
- [FuzzDelSol binary-only Solana analysis](https://arxiv.org/abs/2309.03006)
- [SseRex symbolic execution for Solana](https://arxiv.org/abs/2603.16349)

## 3. Proposed system architecture

```text
                  +-----------------------+
  Program address |  CLI / Web / API       |
  cluster/RPC  ->|  request normalization  |
                  +-----------+-----------+
                              |
                              v
                  +-----------------------+
                  | Chain acquisition      |
                  | loader classification  |
                  | slot/version pinning   |
                  +-----------+-----------+
                              |
                              v
                  +-----------------------+
                  | Canonical artifact     |
                  | ELF bytes + manifest   |
                  | sha256 + provenance    |
                  +-----------+-----------+
                              |
             +----------------+----------------+
             v                                 v
  +-----------------------+          +-----------------------+
  | Deterministic binary  |          | Dynamic evidence      |
  | analysis pipeline     |          | transaction history   |
  | ELF -> sBPF -> IR     |          | logs/CPI/state/replay |
  +-----------+-----------+          +-----------+-----------+
              |                                  |
              +----------------+-----------------+
                               v
                  +-----------------------+
                  | Solana semantic graph |
                  | instructions/accounts |
                  | CPIs/syscalls/types   |
                  +-----------+-----------+
                              |
                              v
                  +-----------------------+
                  | AI evidence agents    |
                  | structure -> naming   |
                  | explain -> verify     |
                  +-----------+-----------+
                              |
                              v
                  +-----------------------+
                  | Evidence-linked output|
                  | pseudocode/report/API |
                  +-----------------------+
```

### 3.1 Chain acquisition layer

Implement a provider-neutral `ChainProvider` interface:

```rust
trait ChainProvider {
    fn get_account(&self, address: Pubkey, slot: Option<u64>) -> Result<AccountSnapshot>;
    fn get_transactions(&self, program: Pubkey, range: TxRange) -> Result<Vec<Transaction>>;
    fn get_transaction(&self, signature: Signature, slot: Option<u64>) -> Result<Transaction>;
}
```

The first implementation should use standard JSON-RPC and base64 account data. Keep
the RPC layer separate from analysis so users can provide a private/archive RPC
without changing the decompiler.

The resolver should:

1. Validate the address and cluster.
2. Fetch the program account with `getAccountInfo` using base64 encoding.
3. Classify the owner as native loader, legacy BPF loader, upgradeable loader, or
   loader v4.
4. For upgradeable loader programs, parse the Program state using the official
   loader interface and fetch the referenced ProgramData account.
5. For direct-code loaders, locate the ELF using the loader-specific adapter.
6. Validate the ELF magic, class, machine, section/segment bounds, and byte length.
7. Record the RPC context slot, deployment slot, loader, program address, data
   address, upgrade authority status, SHA-256, and commitment.
8. Store the byte-exact artifact in content-addressed storage.

Do not hardcode a single ProgramData offset in business logic. Put all loader layouts
behind versioned adapters backed by the Anza/Solana loader interfaces. This protects
the project from loader-v4 changes and future account-layout changes.

The resolver should support two modes:

- **latest**: current finalized artifact;
- **pinned**: artifact and account state at a requested slot/signature, when the RPC
  supports historical reads.

Every analysis must be reproducible from its artifact manifest, not from a mutable
program address alone. An upgradeable program can have multiple different binaries
over its lifetime.

### 3.2 Canonical artifact and evidence storage

Use a content-addressed artifact directory in the first release and add an optional
SQLite catalog:

```text
artifacts/
  sha256/<hash>.so
  sha256/<hash>.manifest.json
  sha256/<hash>.analysis.json
  sha256/<hash>.ir.json
  sha256/<hash>.evidence.jsonl
```

The manifest should include schema version, program address, loader, data address,
cluster, RPC context slot, deployment slot, byte hash, ELF hash, SBPF version,
analysis-tool version, and model/prompt versions for AI-produced fields.

Never overwrite an artifact when a program is upgraded. A new byte hash creates a
new immutable analysis record linked to the same program address.

### 3.3 Deterministic binary-analysis core

The core should be Rust and should use `solana-sbpf` as the runtime-aligned source
for instruction decoding, ELF handling, SBPF versions, verifier behavior, and
syscall/runtime conventions.

Pipeline:

```text
ELF validation
  -> SBPF version detection
  -> section/segment and relocation normalization
  -> instruction decode
  -> entrypoint discovery
  -> function boundary recovery
  -> basic blocks and CFG
  -> call graph and SCCs
  -> stack/frame and register dataflow
  -> SSA-like IR
  -> memory-offset and constant propagation
  -> typed Solana IR
```

Required deterministic outputs:

- instruction address, opcode, operands, and source ELF section;
- direct and indirect branch targets;
- basic blocks and CFG edges;
- function candidates and confidence;
- call graph, recursion/SCC information, and tail calls;
- stack-slot access ranges and register liveness;
- read-only data, strings, format strings, public keys, and numeric constants;
- relocation results and unresolved relocations;
- syscall calls with canonical signatures;
- verifier/runtime diagnostics.

Do not make a third-party GUI decompiler the core dependency. An optional Ghidra
adapter can provide a comparison baseline, but the open-source product needs its own
stable IR, JSON export, and testable native pipeline.

### 3.4 Solana semantic layer

Add a Solana-specific semantic graph on top of generic IR. Core entities:

```text
Program
  ├── EntryPoint
  ├── InstructionCandidate
  │     ├── discriminator/tag evidence
  │     ├── argument layout candidates
  │     └── account-role candidates
  ├── AccountTypeCandidate
  │     ├── field offsets and widths
  │     ├── owner/key/signer/writable checks
  │     └── serialization evidence
  ├── Syscall
  ├── CpiEdge
  ├── ErrorCandidate
  └── Constant/String/PublicKey
```

Implement these detectors in order:

1. Solana input-buffer and account-serialization offsets.
2. Syscall signatures and known memory effects.
3. CPI target extraction and instruction-data construction.
4. Anchor fingerprints and discriminator candidates.
5. System, SPL Token, Token-2022, Associated Token, Metaplex, and common protocol
   instruction fingerprints. Keep these definitions in a versioned public registry
   so projects can register domain-specific program IDs without changing the core
   analyzer.
6. PDA seed construction and bump handling.
7. Account ownership, signer, writable, executable, and key-equality checks.
8. Serialized field candidates from load/store widths, alignment, bounds checks,
   arithmetic, and repeated access patterns.
9. Error-code and log-string association.

The semantic layer must preserve alternatives. For example, an eight-byte constant
may be an Anchor discriminator, a packed integer, or random data. Store all plausible
interpretations until transaction evidence or control-flow constraints eliminate them.

### 3.5 Historical behavior and replay

Static decompilation alone cannot reveal all dynamic behavior. Add an optional,
read-only evidence pipeline:

```text
program address
  -> signatures involving program
  -> parsed instructions/logs/CPI tree
  -> account keys and instruction bytes
  -> account snapshots at the execution slot
  -> local SVM/LiteSVM replay
  -> execution trace and state diff
```

Replay must run in a sandbox and must never submit transactions. The replay result
should capture:

- return status and error;
- logs and events;
- compute-unit consumption;
- CPI targets and serialized CPI data;
- account lamport/data/owner changes;
- return data;
- branch/path coverage where instrumentation permits.

Use replay to validate hypotheses, not to replace static analysis. A historical
transaction only covers one path and one state.

### 3.6 AI evidence agents

The AI subsystem should consume a bounded evidence package, not arbitrary raw chain
content. Each agent returns strict JSON with evidence references.

#### Agent A: Structure recovery

Input: function IR, CFG, stack/frame facts, memory accesses, syscall calls.

Output: structured pseudocode or typed intermediate representation with generic names
such as `fn_0042`, `arg_0`, and `field_0x18`.

Rules:

- no invented APIs or source comments;
- preserve branch conditions and memory offsets;
- identify uncertainty;
- return byte ranges and IR node IDs supporting every statement.
- process known callees before callers and pass only bounded, accepted callee
  output as explicitly untrusted context;
- persist function-level stage bundles only after validation, and rebuild the
  scoped evidence pack plus dependency context before checkpoint reuse.

#### Agent B: Semantic labeling

Input: structure IR plus program strings, public keys, historical traces, and known
Solana/Anchor idioms.

Output: candidate names for functions, instructions, arguments, accounts, fields,
and errors, each with confidence and evidence.

Names should be namespaced by certainty, for example:

```json
{
  "name": "swap",
  "kind": "candidate",
  "confidence": 0.78,
  "evidence": ["trace.cpi[3].target=Tokenkeg...", "ir.fn_42.loads[7]"]
}
```

#### Agent C: Explanation/report generation

Input: verified graph and labeled candidates.

Output: human-readable explanations, call summaries, instruction tables, account
layouts, and security observations. This agent may simplify language but must not
change facts or confidence values.

#### Agent D: Verification/refinement loop

Use constrained iterative refinement:

```text
candidate pseudocode
  -> schema/type validation
  -> optional Rust/C-like compilation
  -> local replay or generated unit tests
  -> compare status/logs/CPI/account diffs
  -> mismatch report
  -> bounded repair pass
```

The model must never be allowed to silently “fix” a mismatch. The final report must
show whether a candidate was only syntactically valid, recompiled, or behaviorally
matched observed traces.

The compiler and replay adapters are separate from the model gateway. The compiler
receives only bounded generated snippets, runs without executing the candidate, and
returns `passed`, `failed`, `timed_out`, or `not_available`. Replay comparison is
reported independently in `validation.json`; observed trace agreement is evidence
for selected paths, not proof of equivalence or source recovery.

The deterministic SBPF dependency pass distinguishes the inputs that influence a
store destination from those that influence its value. Structure-stage ledgers
must cite both roles, and the MIR adapter computes the same split from projected
places and right-hand-side expressions. This rejects pointer/value swaps that a
coarser union-of-inputs check would accept; the current comparison remains
aggregated by effect kind rather than pairing every individual store.

After a whole candidate is rebuilt with the pinned Solana/SBPF toolchain,
`sos judge` provides a separate deterministic binary regression gate. It pairs
functions without trusting names and scores normalized instruction n-grams,
instruction counts, CFG shape, call shape, and memory effects. Missing or
unmatched candidate functions lower the score, and an optional exact mode
requires every normalized function shape to match. The versioned result records
both content hashes and per-function components. This remains structural
evidence—not semantic equivalence—and is interpreted alongside replay results.

`sos rebuild-judge` owns the reproducible compiler boundary around this gate. It
first validates exact `solana-cli` and `solana-cargo-build-sbf` versions, resolves
the package and cdylib target through locked Cargo metadata, hashes the package
source plus workspace lockfile, and calls `cargo-build-sbf` directly with bounded
time and logs. A unique `--sbf-out-dir` prevents stale-output reuse. Only a newly
emitted ELF is passed into the existing deterministic analyzer and judge, while
the complete non-secret build provenance is retained in versioned JSON.

Refinement output can carry directly comparable, evidence-linked observations for
logs, CPI targets, status, and return data. The validator matches these observations
against the trace and emits explicit mismatch records. A bounded repair loop returns
only compiler diagnostics and mismatch records as tagged untrusted context, records
each round, and cannot modify deterministic analysis facts.

### 3.7 Prompt and data security

On-chain strings, logs, account data, and transaction memo fields are untrusted input.
They can contain prompt-injection text. Keep them in tagged evidence fields and never
render them as system/developer instructions. Use a strict model gateway with:

- schema-constrained output;
- maximum function/evidence sizes;
- redaction of secrets and RPC credentials;
- model/provider audit logging;
- deterministic temperature/settings for reproducible reports;
- local-model support for sensitive analysis;
- no network access from the decompiler sandbox.

The executable bytes themselves must be treated as hostile data. ELF parsing,
emulation, replay, and optional tool adapters must run in isolated processes with
resource limits.

## 4. Recommended repository layout

```text
sos-decompiler/
  Cargo.toml
  crates/
    sos-types/           # versioned manifests, IR, evidence schemas
    sos-rpc/             # JSON-RPC providers and archive-provider traits
    sos-loader/          # loader v1/v2/v3/v4 account adapters
    sos-artifact/        # fetch, validate, hash, store
    sos-sbpf/             # decoder, relocations, memory model, verifier bridge
    sos-analysis/         # CFG, functions, dataflow, SSA-like IR
    sos-solana/           # syscalls, account ABI, Anchor/CPI/framework detectors
    sos-replay/            # optional local SVM/LiteSVM execution and tracing
    sos-ai/                # evidence packs, model gateway, validators, refinement
    sos-report/            # JSON, Markdown, Rust-like pseudocode, SARIF export
    sos-cli/               # fetch/analyze/decompile/explain/replay commands
  schemas/
    manifest.schema.json
    analysis.schema.json
    evidence.schema.json
  corpus/
    source-built/
    deployed-fixtures/
    expected/
  integrations/
    ghidra/                # optional comparison adapter
    web/                   # optional UI/API client
  docs/
    threat-model.md
    format-notes.md
    benchmark.md
  fixtures/
    corpus.toml
```

## 5. CLI/API contract

The first user-facing workflow should be:

```bash
sos fetch <PROGRAM_ADDRESS> --cluster mainnet-beta --out ./artifacts
sos analyze ./artifacts/<hash>.so --out ./analysis
sos decompile ./analysis --style rust-like --with-evidence
sos explain ./analysis --model local
sos benchmark ./artifacts/<hash>.so --repeats 3 --out ./benchmarks/result.json
sos replay <PROGRAM_ADDRESS> --signature <SIGNATURE> --out ./traces/trace.json
sos report ./analysis --format markdown
```

The program-address command should be a single convenient workflow:

```bash
sos inspect <PROGRAM_ADDRESS> \
  --cluster mainnet-beta \
  --history 100 \
  --replay \
  --model local \
  --out ./reports/<PROGRAM_ADDRESS>
```

Every command should also support machine-readable JSON so the web UI, notebooks,
and security tooling do not have to scrape human output.

## 6. Implementation phases

### Phase 0 — foundation

- Initialize the Rust workspace and versioned schemas.
- Add loader constants and fixture tests for v1/v2/v3/v4 layouts.
- Add content-addressed artifact storage.
- Add a fixture corpus with known `.so` files and source/build metadata.

### Phase 1 — address-to-ELF MVP

- Implement RPC provider and loader resolver.
- Support mainnet/devnet/testnet/custom RPC.
- Support latest and pinned-slot acquisition where available.
- Produce a verified manifest and exact `.so` artifact.
- Add negative tests for invalid owners, truncated accounts, non-ELF bytes, and
  upgraded programs.

### Phase 2 — deterministic binary analysis

- SBPF version detection and disassembly.
- ELF sections, relocations, strings, constants, and syscall recognition.
- Function candidates, CFG, call graph, stack and register dataflow.
- JSON/Graphviz exports.

### Phase 3 — Solana semantic recovery

- Runtime input/account ABI model.
- CPI and syscall graph.
- Anchor/native/Pinocchio classification.
- Instruction discriminator and account-layout candidates.
- Known program and error databases.

### Phase 4 — AI-assisted decompilation

- Evidence-pack builder.
- Structure-recovery agent with generic identifiers.
- Semantic-labeling agent with confidence/evidence.
- Markdown/Rust-like report generator.
- Provider abstraction for local and remote models.

### Phase 5 — behavioral validation

- Transaction history ingestion.
- Historical account hydration where supported.
- Sandboxed SVM/LiteSVM replay.
- Pinned candidate compilation and deterministic binary-similarity judging.
- Type-layout and call-graph non-regression checks.
- Trace comparison and bounded AI repair loop.

### Phase 6 — research-grade product

- Browser UI with synchronized CFG, pseudocode, bytes, traces, and evidence.
- Collaboration/export formats.
- Plugin API for detectors and known protocols, including registry-backed protocol
  definitions that preserve evidence and uncertainty.
- Public benchmark and reproducible evaluation harness.
- Optional Ghidra/Binary Ninja comparison adapters.

## 7. Evaluation plan

Build a corpus from open-source Anchor, native Rust, Pinocchio, and C programs. Compile
each with multiple compiler/toolchain versions, optimization levels, SBPF versions,
and program sizes. Add deployed fixtures and upgraded-program histories.

Track these metrics separately:

### Binary correctness

- instruction decoding success;
- verifier agreement with the reference runtime;
- function-boundary precision/recall;
- CFG edge precision/recall;
- syscall and relocation identification;
- stack/frame and memory-offset accuracy.

### Solana semantic correctness

- CPI target precision/recall;
- discriminator recognition precision/recall;
- account-role and signer/writable-check accuracy;
- field-offset and width accuracy;
- error/log association accuracy.

### AI quality

- schema-valid output rate;
- evidence citation coverage;
- recompilation rate;
- observed-trace behavioral agreement;
- hallucinated fact rate;
- naming usefulness judged against blinded human review.

### Product quality

- address-to-report latency;
- peak memory on large programs;
- RPC requests and cache hit rate;
- deterministic repeatability;
- sandbox escape/resource-exhaustion tests.

The headline metric should not be “looks like Rust.” It should be:

> percentage of reported facts and hypotheses that are supported by bytecode or
> observed execution evidence, plus percentage of generated candidates that preserve
> tested behavior.

## 8. Decisions to lock now

1. **Rust core, TypeScript UI.** Rust matches the Solana runtime ecosystem and gives
   safe, fast binary analysis; TypeScript is appropriate for the future UI/API client.
2. **Own stable IR.** Third-party decompilers can be adapters, never the product's
   canonical data model.
3. **AI after IR.** No raw-disassembly-to-Rust one-shot pipeline.
4. **Evidence-linked uncertainty.** Every inferred name, type, and behavior has a
   confidence score and references.
5. **Artifact immutability.** Program address alone is not an artifact identity;
   byte hash plus slot/provenance is.
6. **Read-only by default.** RPC reads and local replay only; no wallet or transaction
   submission capability in the core product.
7. **Open-core-friendly interfaces.** Keep RPC, model providers, known-program
   databases, and UI behind replaceable traits/APIs.

## 9. First concrete milestone

The first useful release should make this command reliable:

```bash
sos inspect <deployed-program-address> --cluster mainnet-beta
```

It should fetch the correct ELF, identify the loader and SBPF version, print a
disassembly and CFG, identify Solana syscalls/CPI targets, list strings/constants,
and produce a conservative evidence-linked report. AI naming and Rust-like output
should be layered on only after this deterministic report is correct and reproducible.
