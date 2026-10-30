# Evidence-constrained semantic lifting

SOS separates deterministic SBPF recovery from model-assisted source
reconstruction. A model is allowed to propose readable code, names, types, and
Solana semantics; it is never allowed to rewrite the underlying instruction,
CFG, memory, or data-flow facts.

## Research basis

The design adapts two public reverse-engineering workflows:

- OtterSec's [GoonFi V2 decompilation](https://osec.io/blog/goonfi-decompilation/)
  divides work by instruction handler and call-graph slice, recompiles every
  candidate with the original toolchain, scores functions against the target,
  rejects regressions, and blocks source-level "cheese" such as embedded
  assembly or copied bytecode.
- `renato-osec/patina-research` commit
  `77b7c017741bfb54a176afbfb01da4f85b0530cd` describes a staged Rust pipeline:
  rename functions and variables, recover and layout-check signatures, rebuild
  regions with data-flow constraints, then reconstruct whole functions. Patina
  has no declared repository license at that revision, so SOS uses the published
  architecture as research input and does not copy its implementation.
- `renato-osec/goonfi-v2-decompiled` commit
  `381cc31945474ba6b5b56b32a8cad5266602d612` provides a public, byte-matched
  Solana reconstruction and its verification scripts. It is an evaluation
  reference, not vendored source.

## SOS pipeline

```text
ELF/SBPF facts
  -> canonical IR and Solana semantics
  -> signature recovery (bounded interface hypothesis)
  -> structure recovery (readable control flow + required-effect coverage)
  -> semantic labeling (accounts, types, CPI, PDA, checks)
  -> security-oriented explanation
  -> source refinement
  -> syntax/type validation
  -> rustc MIR effect/CFG comparison
  -> replay and deterministic recompilation comparison
```

Accepted output from one model stage is passed to the next as explicitly tagged
untrusted context. It never replaces deterministic evidence. Every generated
function must target a known IR function and cite valid instruction, block,
operation, memory, or read-only-data evidence. Signature recovery runs before
body reconstruction, following Patina's key dependency: the body agent receives
the accepted interface hypothesis instead of inventing an interface while
rewriting control flow. SOS promotes a gateway-accepted signature into an
explicit, response-hash-bound contract in evidence-pack schema v2. Structure
and refinement candidates are rejected if they change the contract's parameter
names, parameter types, order, or return type; raw prior-stage prose cannot
satisfy this gate. Structure output also includes a machine-checkable
effect ledger. Every deterministic store, branch, recursive CFG loop, call, and
return in the scoped IR must be covered by a matching exact IR location; prose
citations alone cannot satisfy this contract. SOS additionally derives explicit
control-flow region contracts from dominators, postdominators, natural
backedges, and recursive SCCs. Each branch contract records its header, arm
entries and members, join, and exits; each loop records its header, body, exits,
and cycle edges. The structure agent must map every contract to a distinct
one-line `if`/`match` or loop header in its returned pseudocode and cite the
exact deterministic header block. One generic branch or loop can no longer
satisfy several CFG regions.

Fixed-width machine accesses are also converted into immutable memory-field
contracts. Contracts preserve logical and observed machine offsets, all observed
widths, load/store kinds, access PCs, a conservative field path, and exact
instruction evidence. Each contract also names a deterministic Rust layout type
and field. Complete stack-materialized `SolAccountInfo` records use
recovered names such as `account_0.is_signer`; all other accesses keep stable
`unknown_memory_*` paths. Structure and refinement responses must map every
contract to a distinct source expression. An untrusted semantic-stage guess
cannot silently rename an unresolved offset to `balance`, `authority`, or another
plausible field. Refinement source must declare each nonnegative layout as an
immediately attributed `#[repr(C)]` struct. Before rustc runs, SOS appends
compile-time `std::mem::offset_of!` assertions and primitive reference-type
checks for every mapped field. This verifies the declared record layout; it does
not by itself prove that every runtime pointer has that record type.

Before accepting a signature, SOS runs a conservative forward live-in analysis
over the scoped SBPF CFG. Argument registers `r1` through `r5` begin as possible
inputs; a direct read on a reachable path before overwrite creates an observed
ABI obligation. The model must map each obligation to exactly one parameter and
cite its first-use instruction. Calls are treated as clobbering caller-saved
registers but are not assumed to consume all five arguments, avoiding invented
arity. This is a lower bound on the machine ABI, not proof of the original Rust
signature or its source-level types.

## Reviewed open-weight model

The default local profile is `gpt-oss:20b` from OpenAI:

- official weights: <https://huggingface.co/openai/gpt-oss-20b>
- license: Apache-2.0
- runtime target: 16 GiB memory
- reasons for selection: structured outputs, coding/reasoning training, 128K
  context, and local inference support

SOS does not automatically trust arbitrary Ollama tags. Unknown model IDs require
the explicit `--allow-unreviewed-model` option. For reproducible runs, record the
content digest printed by `ollama list`/the local Ollama manifest and use immutable
model storage in CI.

For hosted inference, the reviewed profile is OpenAI's Apache-2.0
`openai/gpt-oss-120b` weights through Groq's OpenAI-compatible API. SOS uses
Groq strict structured outputs, a fixed HTTPS endpoint, bounded completions,
and a free-tier-safe request cadence. The API key is read only from
`GROQ_API_KEY` and never enters the request body or audit files.

- model/runtime documentation: <https://console.groq.com/docs/model/openai/gpt-oss-120b>
- strict structured outputs: <https://console.groq.com/docs/structured-outputs>
- developer-tier rate limits: <https://console.groq.com/docs/rate-limits>
- retention and Zero Data Retention: <https://console.groq.com/docs/your-data>

Cloudflare Workers AI is the reviewed free-allocation alternative. Its Free plan
currently includes 10,000 Neurons per day, and its catalog hosts
`@cf/openai/gpt-oss-120b`. SOS calls the fixed native model endpoint with JSON
mode, validates the returned object again locally, and reads
`CLOUDFLARE_ACCOUNT_ID` plus `CLOUDFLARE_API_TOKEN` only from the environment.

- pricing/free allocation: <https://developers.cloudflare.com/workers-ai/platform/pricing/>
- GPT-OSS 120B model: <https://developers.cloudflare.com/workers-ai/models/gpt-oss-120b/>
- JSON mode: <https://developers.cloudflare.com/workers-ai/features/json-mode/>

The reviewed fallback is the same OpenAI GPT-OSS 120B weights through Hugging
Face Inference Providers, pinned to
`openai/gpt-oss-120b:fireworks-ai`. Hugging Face documents included monthly
inference credits for PRO/Team plans (free accounts require purchased credits),
an OpenAI-compatible router, and JSON-schema output for this route. SOS reads
`HF_TOKEN` only from the process environment and sends the same bounded evidence
projection used by Groq.

- GPT-OSS provider guide: <https://huggingface.co/docs/inference-providers/en/guides/gpt-oss>
- Inference Providers API: <https://huggingface.co/docs/inference-providers/index>
- current pricing: <https://huggingface.co/docs/inference-providers/en/pricing>
- reviewed weights: <https://huggingface.co/openai/gpt-oss-120b>

Direct OpenAI Responses API and Anthropic Messages API transports are also
available for operator-selected agents. They require an explicit model ID and
are not silently treated as reviewed open weights. Credentials, model
placeholders, request boundaries, and Patina-inspired stage isolation are
documented in [AGENT_PROVIDERS.md](AGENT_PROVIDERS.md).

The default `--model` can be overridden independently with
`--structure-model`, `--signature-model`, `--semantic-model`, `--explanation-model`, and
`--refinement-model`. Each stage records the selected model, and checkpoint
validation rejects any bundle created with a different stage assignment.
The parallel `--structure-tokens`, `--signature-tokens`, `--semantic-tokens`,
`--explanation-tokens`, and `--refinement-tokens` flags bound generated output;
the selected budget is sent to the provider, recorded in AI bundle schema v7,
and included in checkpoint validation.
The local GPT-OSS adapter selects Ollama's `low` reasoning mode so reasoning
tokens leave room for final schema-constrained JSON. Structure and signature
schemas require a candidate for every scoped IR function; an empty array is not
treated as successful decompilation.
Semantic-lift prompt v9 and lift-cache schema v10 bind checkpoints to the CFG
region mapping contract, so older effect-only structure responses are never
silently reused.

## Running a lift

### Hosted (recommended on lower-memory machines)

Create a Groq API key and export it in the shell that runs SOS:

```bash
export GROQ_API_KEY='...'
cargo run -p sos-cli -- analyze ./program.so --out ./analysis
cargo run -p sos-cli -- lift ./analysis \
  --provider groq \
  --resume \
  --function fn_0 \
  --repair-rounds 2 \
  --out ./analysis/lifted-groq
```

Or use Hugging Face's routed service without loading local weights:

```bash
export HF_TOKEN='...'
cargo run -p sos-cli -- lift ./analysis \
  --provider huggingface \
  --resume \
  --function fn_0 \
  --repair-rounds 2 \
  --out ./analysis/lifted-huggingface
```

For the daily no-charge Workers AI allocation:

```bash
export CLOUDFLARE_ACCOUNT_ID='32-hex-character-account-id'
export CLOUDFLARE_API_TOKEN='...'
cargo run -p sos-cli -- lift ./analysis \
  --provider cloudflare \
  --resume \
  --function fn_0 \
  --repair-rounds 2 \
  --out ./analysis/lifted-cloudflare
```

Address-first usage requires no local `.so` and defaults to the reviewed 120B
model when `--provider groq` is selected:

```bash
sos inspect <PROGRAM_ADDRESS> \
  --rpc-url https://api.mainnet.solana.com \
  --provider groq \
  --resume \
  --out ./reports/<PROGRAM_ADDRESS>
```

For complete-program lifting, SOS derives a deterministic function call graph,
processes known callees before callers, and supplies accepted direct-callee
reconstructions as bounded untrusted context to the caller's structure stage.
Recursive cycles retain stable address order. Rendering remains in address
order, independent of processing order.

`--resume` persists a fully accepted five-stage base bundle per function. On
reuse, SOS rebuilds the function-scoped analysis, semantic facts, and callee
context, then verifies request identity, all stage input/output hashes, strict
schemas, anti-cheese constraints, and evidence references. A stale, edited, or
partially failed bundle is never reused. `--cache-dir DIR` selects the durable
sidecar location. Compiler-driven repair remains outside the base checkpoint
and is rerun when required.

Only a compact evidence projection is hosted: canonical operation/block IDs,
bounded disassembly, and high-value Solana semantic facts. The raw ELF is not
sent. The complete pack stays local for hashing and evidence-reference
validation. Generated output is untrusted and cannot change deterministic IR.
Because the projection still reveals program behavior, enable provider retention
controls where available and use local inference for confidential binaries.

### Humanizing existing pseudocode

SOS can apply the same evidence-constrained model boundary to textual output
from another decompiler:

```bash
sos humanize ./decompiler-output.txt \
  --deterministic \
  --out ./readable.rs \
  --audit-out ./humanize.json
```

This explicit mode never constructs a model provider or performs a network
request. It produces the same line-addressed fallback used when hosted inference
is unavailable or rejected. For the model-assisted path:

```bash
export GROQ_API_KEY='...'
sos humanize ./decompiler-output.txt \
  --provider groq \
  --resume \
  --out ./readable.rs \
  --audit-out ./humanize.json
```

The same command accepts `--provider huggingface` with `HF_TOKEN`; the audited
model identity becomes `openai/gpt-oss-120b:fireworks-ai`, so Groq and Hugging
Face checkpoints cannot be confused or silently reused across providers.
`--provider cloudflare` similarly binds checkpoints to
`cloudflare_workers_ai_open_weights` and `@cf/openai/gpt-oss-120b`.

The adapter creates one synthetic `fn_0` and one or more
`op_<hex-operation-index>` records per source line. Original line and segment
numbers are embedded in each operation. It also derives a source obligation for
every non-trivial line, classifying function declarations, branches, backward
`goto` loops, calls, memory writes, returns, and ordinary statements. The model
must return exactly one unique, non-comment output fragment for every obligation;
SOS verifies that the fragment occurs in the returned pseudocode and visibly
realizes every classified construct. A citation alone cannot satisfy this gate,
and omitted or reused behavior causes the entire unit to fall back to the
deterministic humanizer. The model must also cite valid operation, block, or
function IDs, and SOS rejects unknown
references, inline assembly, raw byte arrays, include directives, unsafe code,
and raw-operation passthroughs. The output is a reconstruction hypothesis—not
original source or proof of equivalence.

Each free-tier projection is bounded to 128 evidence operations. SOS detects
common C/Ghidra/Rust-style function headers first, then automatically splits
oversized functions into bounded line regions while retaining global line
numbers. Successful and unresolved units are merged into one output and all
unit results remain in the audit. This follows GoonFi's function-by-function
reconstruction and Patina's staged region recovery while keeping each request
reviewable and within hosted token limits. The CLI defaults to 16 units per run
and requires an explicit `--max-units N` increase for larger submissions.

The partitioner extracts conservative function symbols and calls to other known
symbols. Each synthetic IR function keeps its recovered name. Accepted output
from up to two earlier units is passed to the next request as bounded, explicitly
untrusted context, following the iterative GoonFi/Patina pattern without allowing
model output to replace line-level evidence. A deterministic dependency walk
processes known callees before callers, handles cycles in stable source order,
and merges final fragments back into their original line order.

For interruption-safe hosted runs, `--resume` checkpoints a unit only after the
gateway accepts its schema and evidence references. Before reuse, SOS rebuilds
the current evidence pack and verifies the source, provider, model, prompt,
unit hash, full staged request hash, anti-cheese constraints, and evidence
references, including every source-obligation mapping. Humanization prompt and
cache schema v2 invalidate checkpoints produced before this coverage contract.
Invalid or stale records are warned about and recomputed rather
than trusted. The default directory is derived from `--audit-out`; specify
`--cache-dir DIR` for a stable path. Checkpoints contain no API credentials.
If every unit is cached, SOS can validate and reconstruct the accepted model
output without a selected provider token. If a unit is not cached or an
inference response is unavailable or rejected, the humanizer emits a
deterministic syntax-only pseudo-Rust fallback. It normalizes
common decompiler types, guards, returns, pointer reads/writes, and calls while
retaining unsupported lines verbatim as comments. Each emitted line references
its original source line, and the audit distinguishes `deterministic_fallback`
from `accepted_unverified` model output.

Use `--plan` to write the source hash, detected line ranges, synthetic function
IDs, operation counts, and per-unit hashes without requiring an API key or
contacting any model:

```bash
sos humanize ./decompiler-output.txt --plan --audit-out ./humanize-plan.json
```

### Local

Install Ollama, obtain the reviewed model, and start its local service:

```bash
ollama pull gpt-oss:20b
ollama serve
```

Analyze a program and run all five lifting stages:

```bash
cargo run -p sos-cli -- analyze ./program.so --out ./analysis
cargo run -p sos-cli -- lift ./analysis \
  --model gpt-oss:20b \
  --ollama-url http://127.0.0.1:11434 \
  --repair-rounds 2 \
  --out ./analysis/lifted
```

To evaluate one agent against one recovered function first, use the same
reviewed provider through the bounded one-stage command:

```bash
cargo run -p sos-cli -- explain ./analysis \
  --provider ollama \
  --task structure \
  --function fn_0 \
  --context-tokens 32768 \
  --max-output-tokens 2400 \
  --out ./analysis/structure-fn-0.json
```

This is an auditable diagnostic path, not a replacement for the full agent
chain. The response is validated against the same immutable scoped evidence
pack and remains an unverified hypothesis. The command makes one provider
attempt and fails closed when the candidate drops a deterministic loop or
memory write, invents a function identity, or cites nonexistent evidence.

Or fetch and lift directly from a deployed program address:

```bash
sos inspect <PROGRAM_ADDRESS> \
  --rpc-url https://api.mainnet.solana.com \
  --model gpt-oss:20b \
  --function fn_0 \
  --out ./reports/<PROGRAM_ADDRESS>
```

For iterative work on a large binary, scope a run to one canonical IR function
(`fn_` followed by its hexadecimal entry PC):

```bash
cargo run -p sos-cli -- lift ./analysis --function fn_0 --out ./analysis/lifted-fn-0
```

Outputs:

- `model.json`: complete per-stage audit bundle and input/output hashes;
- `ai-report.md`: readable hypotheses and evidence;
- `readable.rs`: best accepted source candidate per function, with unresolved
  functions left explicit.
- `repair.json`: bounded `rustc` diagnostics and compiler-repair rounds. A
  successful compile is still marked unverified until replay agrees.

The compilation record includes a versioned SOS summary of stable
`rustc --emit=mir` output. For each reconstructed `fn_<pc>`, SOS compares MIR-observed
writes, branches, loops, calls, and returns with the effect categories required
by canonical SBPF IR. Missing categories are fed into bounded repair as external
validation mismatches. Raw MIR is not persisted because rustc explicitly treats
its textual representation as unstable.

MIR summary schema v2 also derives structured regions from the compiler CFG.
It records branch entries/arms, immediate postdominating joins, exits, natural
loop backedges, and recursive SCC fallbacks. Validation assigns every
binary-derived region to a distinct shape-compatible MIR region, preserves
parent/child containment, and rejects flattened loops or changed join/exit
behavior. Additional compiler-generated regions are allowed, since bounds
checks, unwinding, and cleanup can add CFG nodes that did not exist as explicit
source constructs.

The same comparison carries positional data dependencies. SOS taints incoming
SBPF registers `r1` through `r5`, propagates those identities through register
moves, arithmetic, loads, CFG joins, and recursive blocks, and records the input
set influencing each observable effect. The MIR summarizer performs the
corresponding propagation from `_1` through `_5`. If a candidate write, branch,
loop, or return drops an input required by the binary, validation reports the
exact missing dependency to the repair agent. Unknown memory contents and
unknown callee arity are not guessed.

For source-backed corpus fixtures, run the lightweight text-contract smoke test
after lifting. It fails when required observable semantics are absent from the
readable candidate:

```bash
python3 scripts/evaluate-readable.py \
  ./analysis/lifted/readable.rs \
  fixtures/c/expected-semantics.json \
  --out ./analysis/lifted/quality.json
```

The stronger GoonFi-style gate recompiles the candidate with the same pinned
Solana/SBPF toolchain and compares the resulting ELF to the original:

```bash
sos judge ./original.so ./recompiled-candidate.so \
  --minimum-score 0.90 \
  --out ./analysis/lifted/binary-judge.json
```

The judge performs deterministic one-to-one function pairing without relying on
source names. Its weighted score covers normalized instruction n-grams,
instruction count, CFG shape, call shape, and memory effects; missing or
unmatched candidate functions reduce the score. `--require-exact` requires every
normalized function shape to match and rejects extra functions. Both inputs,
content hashes, per-function components, the threshold, and limitations are
recorded in versioned JSON.

This comparison is a reconstruction regression signal, not a proof of semantic
equivalence. Toolchain changes can alter otherwise equivalent binaries, while
untested behavior can still differ in structurally similar binaries. Pin the
compiler, linker, optimization flags, SDK, and SBPF tools, then combine this gate
with replay evidence.

## Current trust boundary

`readable.rs` is a hypothesis until independent checks pass. Compilation proves
syntax and type consistency; MIR comparison additionally proves bounded
structural effect coverage, not semantic equivalence. Replay proves behavior
only for the selected paths. Deterministic function-level recompilation scoring
and automatic pinned candidate rebuilding are implemented. The rebuild gate
rejects toolchain drift, requires a lockfile, captures bounded build provenance,
and prevents stale candidate reuse. Memory-write dependencies now distinguish
destination-address inputs from stored-value inputs across SBPF and MIR, and
distinct per-write role signatures are checked independently. Compiler CFG
validation now additionally requires one-to-one branch/loop region coverage,
branch arity, joins/exits, and nesting. Deterministic memory-field contracts now
preserve fixed widths and byte offsets through both source-producing stages, and
rustc now verifies nonnegative `#[repr(C)]` offsets and primitive field types.
Negative frame-relative offsets remain explicit evidence rather than record
layout assertions. The remaining judge work is stronger call-graph
non-regression constraints and multi-transaction path coverage.
