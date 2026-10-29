# SOS Decompiler

Open-source, read-only Solana program reverse engineering from a deployed program
address. The architecture is documented in [ARCHITECTURE.md](ARCHITECTURE.md).

This repository contains native Rust/SBF, low-level native-ABI, and Pinocchio
fixtures that can be used to test the pipeline across implementation styles:

```text
source Rust -> deployable .so -> fetch by program address -> ELF/sBPF analysis
```

## Prerequisites

The example expects the Solana/Agave SBF toolchain:

```bash
solana --version
cargo-build-sbf --version
```

The build is pinned to the `solana-program` 2.x SDK because Solana CLI 3.0's SBF
toolchain uses a Rust/Cargo version that predates edition-2024 support. The program
does not require Anchor.

## Build the example `.so`

```bash
cargo-build-sbf --manifest-path programs/hello-world/Cargo.toml
```

The deployable artifact is written to:

```text
target/deploy/sos_hello_world.so
```

Inspect it locally with:

```bash
file target/deploy/sos_hello_world.so
shasum -a 256 target/deploy/sos_hello_world.so
```

For an on-chain program, `solana program dump` accepts the deployed address and
writes the executable bytes to a local file:

```bash
solana program dump --url localhost <PROGRAM_ADDRESS> /tmp/sos-hello-world.dump.so
```

## Deploy to a local validator

Start a local validator in another terminal:

```bash
solana-test-validator
```

Then deploy the fixture:

```bash
solana config set --url http://127.0.0.1:8899
solana program deploy target/deploy/sos_hello_world.so
```

Run the address-first acquisition, deterministic analysis, and four-stage local
model pipeline against a fresh validator with:

```bash
./scripts/test-local-inspect.sh
```

The Rust core scaffold can already inspect the generated ELF header:

```bash
cargo run -p sos-cli -- artifact-info target/deploy/sos_hello_world.so
```

Expected output includes:

```text
machine=0x107
sbpf=true
```

The command prints the deployed program address. That address is the only chain
identifier the SOS inspection command needs:

```bash
sos inspect <PROGRAM_ADDRESS> --rpc-url http://127.0.0.1:8899 --commitment processed
```

The implemented Phase 1 fetch path can already resolve a deployed address through
the upgradeable loader, fetch its ProgramData account, extract the ELF, hash it, and
write a content-addressed artifact plus manifest. It also creates an idempotent
`metadata.sqlite` index containing the provenance-aware artifact ID:

```bash
cargo run -p sos-cli -- fetch <PROGRAM_ADDRESS> \
  --rpc-url http://127.0.0.1:8899 \
  --commitment processed \
  --out ./artifacts
```

The inspection path composes fetching with deterministic SBPF analysis and writes an
`analysis.json`, `ir.json`, `semantics.json`, `security.json`, `report.md`, and
CFG-faithful `pseudocode.rs` alongside the content-addressed artifact:

```bash
cargo run -p sos-cli -- inspect <PROGRAM_ADDRESS> \
  --cluster localnet \
  --rpc-url http://127.0.0.1:8899 \
  --commitment processed \
  --out ./reports/<PROGRAM_ADDRESS>
```

Add `--history N` to save available transaction signatures and parsed transaction
evidence as `history.json`. Missing archive state is recorded as `not_available` and
does not prevent static analysis.

`inspect --replay` selects the latest available program transaction; pass
`--signature SIGNATURE` to select a specific transaction. If exact historical account
state is unavailable, `replay.json` records `not_available` without substituting
current account state.

The address-first workflow can also invoke an explicitly configured local model
adapter. It runs structure recovery, semantic labeling, explanation, and bounded
refinement through the same gateway, writing a versioned per-stage audit bundle to
`model.json`. Without a command, `--model` records `not_available` for every stage
rather than contacting a hosted service:

When a model is requested, `ai-report.md` is emitted alongside `model.json`; it is
explicitly labeled as untrusted hypothesis output and retains each claim's evidence.

```bash
sos inspect <PROGRAM_ADDRESS> \
  --rpc-url http://127.0.0.1:8899 \
  --out ./reports/<PROGRAM_ADDRESS> \
  --model local \
  --model-command ./fixtures/echo-model.sh

# Optional external validation and bounded repair in the same address-first run:
sos inspect <PROGRAM_ADDRESS> \
  --rpc-url http://127.0.0.1:8899 \
  --out ./reports/<PROGRAM_ADDRESS> \
  --model local \
  --model-command ./fixtures/repair-model.sh \
  --validate --repair --repair-rounds 3 \
  --validation-trace ./trace.json
```

Analyze an already downloaded ELF into a reusable analysis directory:

```bash
sos analyze ./program.so --out ./analysis
sos analyze ./program.so --out ./analysis --json
sos detect ./analysis --out ./analysis/security.json
sos detect ./analysis --format sarif --out ./analysis/security.sarif
sos validate ./analysis --compiler rustc --trace ./trace.json --out ./analysis/validation.json
sos validate ./analysis --compiler rustc --trace ./trace.json \
  --repair --model-command ./fixtures/repair-model.sh --repair-rounds 3 \
  --out ./analysis/validation.json
sos decompile ./analysis --format rust-like --out ./analysis/pseudocode.rs
sos decompile ./analysis --format c-like --out ./analysis/pseudocode.c
sos report ./analysis --format json --out ./analysis/report.json
sos report ./analysis --format graphviz --out ./analysis/cfg.dot
sos explain ./analysis --model-command ./fixtures/echo-model.sh --out ./analysis/explanation.json
```

### Local open-weight semantic lifting

SOS can now pass deterministic IR through a four-stage local semantic-lifting
pipeline and merge accepted structure/refinement candidates into readable
Rust-like output. The reviewed default is OpenAI's Apache-2.0 `gpt-oss:20b`
weights running locally through Ollama:

```bash
ollama pull gpt-oss:20b
cargo run -p sos-cli -- lift ./analysis \
  --model gpt-oss:20b \
  --ollama-url http://127.0.0.1:11434 \
  --out ./analysis/lifted
```

The same reviewed local pipeline is available directly from a deployed address;
no separate model-adapter command is required. Scope the first run to an entrypoint
or handler so large programs can be lifted and reviewed incrementally:

```bash
sos inspect <PROGRAM_ADDRESS> \
  --rpc-url https://api.mainnet.solana.com \
  --model gpt-oss:20b \
  --function fn_0 \
  --out ./reports/<PROGRAM_ADDRESS>
```

This address-first form writes `readable.rs` and `repair.json` in addition to the
deterministic evidence bundle and audited model outputs.

### Hosted open-weight lifting without local model load

For machines that cannot run a large local model, SOS supports Groq's hosted
`openai/gpt-oss-120b` endpoint. The weights are OpenAI-published under
Apache-2.0, and Groq supports strict JSON-schema output for this model. Create a
Groq API key, keep it in the process environment, and run:

```bash
export GROQ_API_KEY='...'
cargo run -p sos-cli -- lift ./analysis \
  --provider groq \
  --resume \
  --function fn_0 \
  --repair-rounds 2 \
  --out ./analysis/lifted-groq
```

The same provider works from only a deployed address:

```bash
export GROQ_API_KEY='...'
sos inspect <PROGRAM_ADDRESS> \
  --rpc-url https://api.mainnet.solana.com \
  --provider groq \
  --resume \
  --out ./reports/<PROGRAM_ADDRESS>
```

Without `--function`, address-first lifting processes the complete recovered
call graph. Known callees are lifted before callers, and accepted callee
reconstructions are supplied to the caller as bounded, explicitly untrusted
context. Final source is still rendered in deterministic function-address
order. Use `--function fn_ID` for a smaller first run.

`--resume` writes a completed four-stage checkpoint per function. SOS rebuilds
the scoped IR, Solana semantics, and callee context and then revalidates every
stage's provider/model/prompt identity, input and output hashes, strict schema,
anti-cheese rules, and evidence references before reuse. Invalid checkpoints
are ignored. Choose a stable directory with `--cache-dir DIR`; no API key is
stored. Fully cached Groq lifts can be reconstructed without loading a local
model or setting `GROQ_API_KEY`.

SOS never writes the key into an audit bundle and does not send raw `.so`
bytes. It sends a bounded projection of canonical IR and recovered Solana
facts. Those facts and generated hypotheses still leave the machine, so enable
Groq Zero Data Retention where available and do not use the hosted path for
artifacts whose analysis must remain entirely local. The provider deliberately
paces requests for the free developer-tier token limits.

If pseudocode already exists from SOS, Ghidra, IDA, Binary Ninja, or another
decompiler, use the one-stage humanizer instead of re-running binary analysis:

```bash
export GROQ_API_KEY='...'
sos humanize ./decompiler-output.txt \
  --provider groq \
  --resume \
  --out ./readable.rs \
  --audit-out ./humanize.json
```

Every source line receives one or more deterministic IR operation IDs. Model output must
cite those IDs, passes the same anti-cheese checks as binary lifting, and is
marked unverified. `humanize.json` stores the input hash, evidence pack, model
identity, and response without storing the API key. SOS automatically detects
function boundaries and partitions larger input into free-tier-sized units of
at most 128 evidence operations, following the GoonFi/Patina workflow. Hosted
rate-limit pacing applies between units. A run defaults to at most 16 units;
review the partition and pass `--max-units N` to authorize a larger run.
Detected function names and known call relationships are attached to each unit.
Up to two accepted earlier reconstructions are supplied to the next unit as
explicitly untrusted context; deterministic line evidence remains authoritative.
Dependency units are processed before callers even when file order is reversed,
then merged back into original source order.

`--resume` writes each accepted unit to a checkpoint directory (by default,
next to the audit output). On restart, SOS verifies the source, provider, model,
prompt, unit hash, complete staged evidence-pack hash, output schema,
anti-cheese rules, and every evidence reference before reuse. Stale or edited
checkpoints are ignored and recomputed. Use `--cache-dir DIR` with `--resume`
to choose a stable location. A fully cached hosted run can be reconstructed
without `GROQ_API_KEY`; any cache miss remains explicitly unresolved until the
key is available. API keys are never written to checkpoints.

Preview decomposition without loading or contacting any model:

```bash
sos humanize ./decompiler-output.txt --plan --audit-out ./humanize-plan.json
```

The command writes `readable.rs`, `model.json`, and `ai-report.md`. Generated
source remains evidence-linked hypothesis material; functions without an accepted
candidate are marked unresolved. The research basis, trust boundary, and staged
GoonFi/Patina-derived design are documented in
[docs/SEMANTIC_LIFTING.md](docs/SEMANTIC_LIFTING.md).

Run the complete checked fixture demonstration—C/SBF `.so` analysis, scoped
open-weight lifting, bounded compiler repair, and the deterministic behavior
coverage gate—with:

```bash
ollama serve
./scripts/demo-open-model.sh
```

The script uses only the reviewed `gpt-oss:20b` profile unless `SOS_MODEL` is
explicitly changed, and prints the paths to the readable source, audited model
bundle, local model-manifest provenance, and quality result.

Use `./scripts/demo-groq.sh` for the equivalent hosted demo. It requires only a
`GROQ_API_KEY`; it does not start Ollama or load the downloaded local weights.
Use `./scripts/demo-humanize-groq.sh` to transform the checked-in
Ghidra-style pseudocode fixture directly.

Hosted providers can implement the same provider-neutral JSON protocol through an
HTTP endpoint:

```bash
sos explain ./analysis --model hosted --model-endpoint http://127.0.0.1:8080/generate
```

For repeatable evaluation, benchmark an artifact directly. The command records
artifact and analysis hashes, structural/semantic/security counts, and whether
repeated analysis runs are byte-identical:

```bash
sos benchmark ./program.so --repeats 3 --out ./benchmark.json
```

Build the currently available fixture corpus with:

```bash
./scripts/build-fixtures.sh
```

The corpus manifest is [fixtures/corpus.toml](<fixtures/corpus.toml>); the
Pinocchio 0.9.0 and Anchor 0.31.1 fixtures are buildable with the pinned SBF
toolchain, and the C fixture is built through the active platform-tools C/SBF
SDK. The remaining reference entries are explicitly marked `planned` until
their source/build recipes are pinned and reproducible.

If authentication is required, `SOS_MODEL_TOKEN` is sent as a bearer token and is
never written into reports.

Replay accepts an address and transaction signature. It always records the exact
transaction evidence and attaches a conservative semantic comparison. The default
build emits an explicit `not_available` trace.
For local isolated execution, build the CLI with its explicit LiteSVM feature; the
command hydrates the transaction's account keys at the selected slot before
simulation:

```bash
cargo run -p sos-cli --features litesvm -- \
  replay <PROGRAM_ADDRESS> --signature <SIGNATURE> --out ./trace.json
```

The replay backend can also be exercised locally against the checked-in fixture:

```bash
cargo run -p sos-replay --features litesvm --example litesvm_fixture
```

The current implementation is an end-to-end, evidence-preserving decompiler
foundation: deterministic recovery is always available, AI refinement is optional,
and candidate source is never executed as proof. The normative requirements are in
[SPEC.md](SPEC.md), and the milestone gates are in [PLAN.md](PLAN.md).

Contribution and vulnerability-reporting guidance is available in
[CONTRIBUTING.md](CONTRIBUTING.md) and [SECURITY.md](SECURITY.md). The workspace
declares Apache-2.0 in its Cargo metadata, and the repository is licensed under
Apache-2.0; see [LICENSE](LICENSE).

Use `finalized` for production RPCs. A just-deployed local-validator account may
need `processed` or `confirmed` until it reaches the requested commitment.
Pinned-slot replay requires an RPC/archive provider that returns account data at
the exact requested context slot; a merely newer `minContextSlot` response is
recorded as historical state unavailable.

## Repository layout

```text
ARCHITECTURE.md                 research and system design
SPEC.md                         normative product and trust specification
PLAN.md                         phased implementation plan and acceptance gates
LICENSE                         Apache-2.0 license
CONTRIBUTING.md                 contribution and validation guidance
SECURITY.md                     security reporting and threat boundaries
crates/sos-types/               versioned manifests and evidence contracts
crates/sos-artifact/            hashing, manifests, content-addressed storage
crates/sos-rpc/                 provider-neutral and JSON-RPC contracts
crates/sos-storage/             filesystem-independent SQLite metadata index
crates/sos-replay/              versioned execution traces and replay boundary
  examples/litesvm_fixture.rs   successful local replay acceptance fixture
crates/sos-loader/              loader-v2/v3/v4 adapter boundaries
crates/sos-analysis/            deterministic analysis, IR, and public-key candidates
crates/sos-ai/                  bounded model gateway and evidence validation
schemas/v1/                    public JSON contracts for manifests, analysis, IR, semantics, evidence, refinement, validation, and reports
fuzz/                           offline cargo-fuzz targets for parser boundaries
.github/workflows/ci.yml        host checks and opt-in pinned SBF fixture CI
fixtures/echo-model.sh         local model-provider protocol fixture
fixtures/compile-model.sh      refinement fixture for external candidate compilation
fixtures/repair-model.sh       bounded mismatch-driven repair fixture
fixtures/corpus.toml            labeled fixture corpus manifest
crates/sos-solana/              syscall, protocol, and Solana semantic candidates
crates/sos-report/              Markdown, pseudocode, and SARIF renderers
crates/sos-cli/                 CLI entrypoint
programs/hello-world/           native Rust/SBF fixture
programs/native-abi/            native Rust/SBF dispatch and account-check fixture
programs/pinocchio-fixture/     pinned Pinocchio framework fixture
programs/anchor-fixture/        pinned Anchor framework fixture
  Cargo.toml
  src/lib.rs
programs/c-fixture/             pinned C/SBF ABI fixture
  README.md
  src/sos_c_fixture/sos_c_fixture.c
Cargo.toml                      workspace and release profile
```

## Design boundary

The hello-world source is a known-good fixture, not a decompiler implementation. It
gives us a controlled binary/source pair for tests. The decompiler must still treat
the deployed `.so` as the source of truth and must label reconstructed names and
types as hypotheses unless supported by bytecode or observed execution evidence.
