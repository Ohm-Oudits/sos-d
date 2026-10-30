# SOS Decompiler

Open-source, read-only Solana program reverse engineering from a deployed program
address. The architecture is documented in [ARCHITECTURE.md](ARCHITECTURE.md).
Model-agent credentials, providers, and trust boundaries are documented in
[docs/AGENT_PROVIDERS.md](docs/AGENT_PROVIDERS.md).

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

Run the address-first acquisition, deterministic analysis, and five-stage local
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

Call analysis resolves both unique immediate internal calls and `callx` sites whose
version-specific target register is proven, within one basic block, to contain one
aligned function-entry virtual address. Those edges become immutable source and
rustc-MIR target contracts for the model pipeline. A bounded lattice also carries up
to eight path-dependent function-pointer candidates across CFG joins and direct-call
argument handoffs. These appear as `call_candidates` with no exact `target_pc`; SOS
instructs the model to preserve the uncertain dispatch and never choose one callee or
claim the set is exhaustive.

Add `--history N` to save available transaction signatures and parsed transaction
evidence as `history.json`. Missing archive state is recorded as `not_available` and
does not prevent static analysis.

`inspect --replay` selects the latest available program transaction; pass
`--signature SIGNATURE` to select a specific transaction. If exact historical account
state is unavailable, `replay.json` records `not_available` without substituting
current account state.

The address-first workflow can also invoke an explicitly configured local model
adapter. It runs signature recovery, structure recovery, semantic labeling,
explanation, and bounded refinement through the same gateway, writing a
versioned per-stage audit bundle to `model.json`. Without a command, `--model`
records `not_available` for every stage rather than contacting a hosted service:

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
sos explain ./analysis --provider ollama --task structure --function fn_0 \
  --context-tokens 32768 --max-output-tokens 2400 \
  --out ./analysis/structure-fn-0.json
sos judge ./program.so ./recompiled-candidate.so \
  --minimum-score 0.90 --out ./binary-judge.json
sos rebuild-judge ./program.so ./candidate/Cargo.toml \
  --expected-solana-version 3.0.0 --minimum-score 0.90 \
  --out ./rebuild-judge.json
```

`sos judge` compares a candidate rebuilt with the pinned Solana/SBPF toolchain
against the original binary. It deterministically pairs functions without using
source names and scores normalized instruction n-grams, instruction counts, CFG
shape, call shape, and memory effects. Unmatched candidate functions lower the
score. Add `--require-exact` when every normalized function shape must match.
This is a strong reconstruction regression gate, not proof of semantic
equivalence; compiler, linker, optimization, and SBPF toolchain versions must be
pinned for meaningful comparisons.

`sos rebuild-judge` performs that pinned rebuild automatically. It requires both
`solana-cli` and `solana-cargo-build-sbf` to exactly match the requested version,
also verifies and explicitly pins platform-tools v1.51 and SBF rustc 1.84.1,
requires `Cargo.lock`, invokes `cargo-build-sbf` with `--locked` and an explicit
SBPF architecture, and writes the candidate into a fresh isolated directory so a
stale `.so` cannot pass. The versioned result retains tool output, exact arguments,
bounded build logs, manifest/source/lock hashes, and both ELF hashes before running
the same deterministic judge. The default pin is Solana 3.0.0 and the default build
timeout is 15 minutes (bounded to one hour).

`sos validate` also asks stable `rustc` to emit MIR for every compiled
refinement. SOS reduces that compiler output to a bounded function/CFG summary
and verifies that each SBPF-required effect category is present in the actual
candidate MIR. A model cannot satisfy this gate merely by returning an effect
ledger while omitting the corresponding write, branch, loop, call, or return
from its code. SOS also propagates incoming SBPF `r1`-`r5` identities through
register operations and CFG joins, then requires candidate MIR effects to retain
those positional input dependencies. Memory writes separately preserve which
inputs influence the destination address and which influence the stored value,
so swapping an account pointer with its data is rejected even when the same
inputs remain present overall. Distinct store role signatures are checked
separately, preventing two crossed writes from passing merely because their
aggregate target and value sets look correct. The MIR summary also independently
recovers branch arms, postdominating joins, natural/SCC loops, cycle edges, and
loop exits. Every binary-derived control-flow region must match a distinct MIR
region with the same branch arity and join/exit shape, and nested regions must
remain nested. Extra rustc regions from bounds checks or cleanup are allowed but
cannot be reused to satisfy binary contracts. MIR coverage is still a structural
non-regression check, not a claim of semantic equivalence.

### Local open-weight semantic lifting

SOS can now pass deterministic IR through a five-stage local semantic-lifting
pipeline and merge accepted structure/refinement candidates into readable
Rust-like output. The reviewed default is OpenAI's Apache-2.0 `gpt-oss:20b`
weights running locally through Ollama:

The stage order follows Patina's dependency discipline: recover the function
signature first, then reconstruct the body under that interface. Structure
responses must additionally cover every deterministic store, branch, recursive
loop, call, and return with an exact IR evidence location. SOS also derives
branch and loop regions from dominators, postdominators, backedges, and recursive
SCCs. Every recovered region requires its own concrete source header tied to the
exact CFG header block, preventing one generic `if` or `loop` from covering an
entire function. Signature recovery
is also constrained by a conservative SBPF live-in analysis: every incoming
`r1`-`r5` value read on a reachable path before overwrite must be mapped to one
model parameter and cited at its first-use instruction. Names and types remain
explicit hypotheses; the register mapping is deterministic evidence.

Evidence-pack schema v5 derives immutable fixed-offset memory-field
contracts. Each contract records its function, logical offset, observed machine
offset, access widths/kinds/PCs, conservative field path, and exact instruction
evidence, plus deterministic Rust layout type and field identifiers. Fully
recognized stack-materialized `SolAccountInfo` records use paths
such as `account_0.is_signer`; unresolved accesses retain neutral names such as
`unknown_memory_r1.field_pos_0x8`. Both structure and final refinement output
must map every contract to a unique source expression and may not replace an
unknown field with an unsupported semantic guess. Final refinements must also
declare the named layout as `#[repr(C)]`; SOS injects `offset_of!` and primitive
field-type assertions into the bounded rustc input. A cosmetic mapping, wrong
padding, or wrong primitive field type therefore fails compilation. Negative
machine-relative stack offsets remain evidence-only because they are not Rust
record offsets.

The same evidence pack now carries resolved internal-call contracts. Unique SBPF
function hashes are resolved to exact `fn_<pc>` identities; structure and
refinement output must map every call site to that exact target. Rustc MIR
validation checks target identity and multiplicity, so calling an unrelated
helper no longer satisfies a generic call effect. Indirect and unresolved calls
remain explicitly unresolved rather than being guessed.

```bash
ollama pull gpt-oss:20b
cargo run -p sos-cli -- lift ./analysis \
  --model gpt-oss:20b \
  --ollama-url http://127.0.0.1:11434 \
  --out ./analysis/lifted
```

Before trusting a model for a larger run, qualify that exact model/runtime pair
against the checked-in Solana, cybersecurity, and systems pseudocode corpus:

```bash
sos qualify-model fixtures/model-qualification-v1.json \
  --provider ollama --model gpt-oss:20b \
  --minimum-score 0.80 --out ./model-qualification.json
```

Qualification requires every response to pass the normal JSON schema,
evidence-reference, and anti-cheese gates, then scores hidden semantic concept
groups and forbidden reconstruction shortcuts. A passing score is bounded corpus
evidence—not a declaration that the weights are universally safe or correct.
The command also accepts the reviewed Groq, Hugging Face, and Cloudflare routes,
plus explicit OpenAI or Anthropic model IDs using the credential placeholders
documented below.

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

Cloudflare Workers AI is the second no-local-load option and currently includes
a daily free allocation on Workers Free. It hosts the same reviewed GPT-OSS
120B weights. Create a Workers AI token, export the account ID separately, and
run:

```bash
export CLOUDFLARE_ACCOUNT_ID='32-hex-character-account-id'
export CLOUDFLARE_API_TOKEN='...'
sos humanize ./decompiler-output.txt \
  --provider cloudflare \
  --resume \
  --out ./readable.rs \
  --audit-out ./humanize.json
```

The transport is pinned to `@cf/openai/gpt-oss-120b` and the native Workers AI
endpoint. SOS reads both values from the process environment, places the account
ID only in the endpoint URL and the token only in the authorization header, and
never stores either value in the prompt body, audit, or checkpoints.

Hugging Face Inference Providers is an additional reviewed hosted route. It forwards
the same Apache-2.0 GPT-OSS 120B weights through its OpenAI-compatible router.
SOS pins the Fireworks route because that route documents JSON-schema output.
Free Hugging Face accounts currently require purchased inference credits;
PRO/Team plans include monthly credits:

```bash
export HF_TOKEN='...'
sos humanize ./decompiler-output.txt \
  --provider huggingface \
  --resume \
  --out ./readable.rs \
  --audit-out ./humanize.json

# Address-only whole-program lifting is supported too:
sos inspect <PROGRAM_ADDRESS> \
  --rpc-url https://api.mainnet.solana.com \
  --provider huggingface \
  --resume \
  --out ./reports/<PROGRAM_ADDRESS>
```

Create `HF_TOKEN` with Inference Providers permission. The fixed router is
`https://router.huggingface.co/v1/chat/completions`; the reviewed default model
route is `openai/gpt-oss-120b:fireworks-ai`.

OpenAI Responses API and Anthropic Messages API adapters are available when an
operator wants to supply another agent model. SOS does not choose a mutable
default for these providers; pass the exact model identity or set its model
environment variable:

```bash
export OPENAI_API_KEY='...'
export SOS_OPENAI_MODEL='your-explicit-model-id'
sos humanize ./decompiler-output.txt --provider openai --resume

export ANTHROPIC_API_KEY='...'
export SOS_ANTHROPIC_MODEL='your-explicit-model-id'
sos inspect <PROGRAM_ADDRESS> \
  --provider anthropic \
  --structure-model your-reconstruction-model-id \
  --semantic-model your-labeling-model-id \
  --refinement-model your-reconstruction-model-id \
  --structure-tokens 2400 \
  --semantic-tokens 800 \
  --refinement-tokens 2400 \
  --resume \
  --out ./reports/<PROGRAM_ADDRESS>
```

These transports are not silently classified as reviewed open weights. Their
outputs pass the same local schemas, evidence-reference checks, anti-cheese
rules, and checkpoint validation. `--structure-model`, `--signature-model`, `--semantic-model`,
`--explanation-model`, and `--refinement-model` optionally assign distinct
models to the Patina-style agent stages; each identity is audited and included
in checkpoint validation. Matching `--structure-tokens`, `--signature-tokens`, `--semantic-tokens`,
`--explanation-tokens`, and `--refinement-tokens` flags enforce and audit bounded
per-stage output budgets. See
[docs/AGENT_PROVIDERS.md](docs/AGENT_PROVIDERS.md) and the empty
[.env.example](.env.example) placeholders.

Without `--function`, address-first lifting processes the complete recovered
call graph. Known callees are lifted before callers, and accepted callee
reconstructions are supplied to the caller as bounded, explicitly untrusted
context. Final source is still rendered in deterministic function-address
order. Use `--function fn_ID` for a smaller first run.

`--resume` writes a completed five-stage checkpoint per function. SOS rebuilds
the scoped IR, Solana semantics, and callee context and then revalidates every
stage's provider/model/prompt identity, input and output hashes, strict schema,
anti-cheese rules, and evidence references before reuse. Invalid checkpoints
are ignored. Choose a stable directory with `--cache-dir DIR`; no API key is
stored. Fully cached hosted lifts can be reconstructed without loading a local
model or setting the selected provider credential.

SOS never writes the key into an audit bundle and does not send raw `.so`
bytes. It sends a bounded projection of canonical IR and recovered Solana
facts. Those facts and generated hypotheses still leave the machine, so enable
Groq Zero Data Retention where available and do not use the hosted path for
artifacts whose analysis must remain entirely local. The provider deliberately
paces requests for the free developer-tier token limits.

If pseudocode already exists from SOS, Ghidra, IDA, Binary Ninja, or another
decompiler, use the one-stage humanizer instead of re-running binary analysis:

```bash
# Guaranteed no model load and no network request:
sos humanize ./decompiler-output.txt \
  --deterministic \
  --out ./readable.rs \
  --audit-out ./humanize.json
```

Use a reviewed hosted GPT-OSS provider when semantic renaming and higher-level
restructuring are desired:

```bash
export GROQ_API_KEY='...'
sos humanize ./decompiler-output.txt \
  --provider groq \
  --resume \
  --out ./readable.rs \
  --audit-out ./humanize.json
```

Every behavioral source line receives one or more deterministic IR operation
IDs plus a source obligation for its declarations, branches, backward loops,
calls, memory writes, returns, and statements. Model output must cite valid IDs
and map every obligation to one unique concrete code fragment retained in the
returned pseudocode. SOS rejects omitted behavior, comment-only mappings,
reused fragments, or fragments that do not visibly realize the required
construct. Accepted output still passes the same anti-cheese checks as binary
lifting and is marked unverified. `humanize.json` stores the input hash,
evidence pack, model identity, and response without storing the API key. SOS
automatically detects
function boundaries and partitions larger input into free-tier-sized units of
at most 128 evidence operations, following the GoonFi/Patina workflow. Hosted
rate-limit pacing applies between units. A run defaults to at most 16 units;
review the partition and pass `--max-units N` to authorize a larger run.
Detected function names and known call relationships are attached to each unit.
Evidence-pack v6 turns those relationships into exact textual symbol contracts:
the mapped declaration must retain its known function identity and every mapped
cross-function call must retain the known callee. The model may improve local names,
but it cannot silently redirect a call or rename one function differently across
units.
Up to two accepted earlier reconstructions are supplied to the next unit as
explicitly untrusted context; deterministic line evidence remains authoritative.
Dependency units are processed before callers even when file order is reversed,
then merged back into original source order.

If inference is unavailable or a response fails validation, SOS now emits an
evidence-preserving deterministic fallback instead of an empty `UNRESOLVED`
region. The fallback normalizes common Ghidra/IDA C types, guards, returns,
pointer reads and writes, and calls into explicitly unverified pseudo-Rust. Every
output line cites its original global line number; unsupported control flow or
syntax is retained verbatim as an `untranslated` comment and recorded in the
audit. A checked example is
[`fixtures/pseudocode/account_guard.deterministic.rs`](fixtures/pseudocode/account_guard.deterministic.rs).

`--resume` writes each accepted unit to a checkpoint directory (by default,
next to the audit output). On restart, SOS verifies the source, provider, model,
prompt, unit hash, complete staged evidence-pack hash, output schema,
anti-cheese rules, every source-obligation mapping, and every evidence reference
before reuse. Humanization prompt/cache v3 prevents checkpoints created before
exact declaration/callee contracts from being reused. Stale or edited
checkpoints are ignored and recomputed. Use `--cache-dir DIR` with `--resume`
to choose a stable location. A fully cached hosted run can be reconstructed
without the selected provider credential; any cache miss
uses the deterministic syntax-only fallback until the selected credentials are
available. API keys and account identifiers are never written to checkpoints.

Preview decomposition without loading or contacting any model:

```bash
sos humanize ./decompiler-output.txt --plan --audit-out ./humanize-plan.json
```

`sos humanize` writes the readable source and its JSON audit; address-first
`sos inspect` additionally writes `model.json` and `ai-report.md`. Generated
source remains evidence-linked hypothesis material. Functions without an
accepted model candidate use a clearly labeled deterministic fallback. The
research basis, trust boundary, and staged GoonFi/Patina-derived design are documented in
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
Ghidra-style pseudocode fixture directly. The provider-independent alternative
with a daily free allocation is `./scripts/demo-humanize-cloudflare.sh`, using
Workers AI credentials and no local model. Another fallback is
`./scripts/demo-humanize-huggingface.sh`, using `HF_TOKEN` and no local model.
For a completely offline, no-model run use
`./scripts/demo-humanize-deterministic.sh`.

Hosted providers can implement the same provider-neutral JSON protocol through an
HTTP endpoint:

```bash
sos explain ./analysis --model hosted --model-endpoint http://127.0.0.1:8080/generate
```

`sos explain` is also the cost-bounded one-agent path for inspecting one stage
and one recovered function before committing to a complete five-stage lift. It
supports the same reviewed Ollama, Groq, Hugging Face, Cloudflare, OpenAI, and
Anthropic providers as `sos lift`; generated output still passes the same local
schema, evidence-reference, and anti-cheese checks. It makes one provider
attempt so a rejected diagnostic cannot silently double local runtime or hosted
token spend.

For repeatable evaluation, benchmark an artifact directly. The command records
artifact and analysis hashes, structural/semantic/security counts, and whether
repeated analysis runs are byte-identical:

```bash
sos benchmark ./program.so --repeats 3 --out ./benchmark.json
```

After a reconstructed program has been compiled to SBPF, run the deterministic
binary judge and retain its versioned JSON result:

```bash
sos judge ./original.so ./recompiled-candidate.so \
  --minimum-score 0.90 \
  --out ./binary-judge.json
```

Use `--require-exact` for a GoonFi-style normalized exact-match gate. The default
score is intentionally more tolerant of compiler-produced differences, while
still penalizing missing and unmatched candidate functions.

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
scripts/test-callx-fixture.sh   compiled callx uncertainty regression check
fixtures/echo-model.sh         local model-provider protocol fixture
fixtures/compile-model.sh      refinement fixture for external candidate compilation
fixtures/repair-model.sh       bounded mismatch-driven repair fixture
fixtures/corpus.toml            labeled fixture corpus manifest
programs/callx-fixture/         compiled multi-target callx uncertainty fixture
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
