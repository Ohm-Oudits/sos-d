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
  -> structure recovery (readable control flow)
  -> semantic labeling (accounts, types, CPI, PDA, checks)
  -> security-oriented explanation
  -> source refinement
  -> syntax/type validation
  -> replay and recompilation comparison (planned judge expansion)
```

Accepted output from one model stage is passed to the next as explicitly tagged
untrusted context. It never replaces deterministic evidence. Every generated
function must target a known IR function and cite valid instruction, block,
operation, memory, or read-only-data evidence.

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

## Running a lift

### Hosted (recommended on lower-memory machines)

Create a Groq API key and export it in the shell that runs SOS:

```bash
export GROQ_API_KEY='...'
cargo run -p sos-cli -- analyze ./program.so --out ./analysis
cargo run -p sos-cli -- lift ./analysis \
  --provider groq \
  --function fn_0 \
  --repair-rounds 2 \
  --out ./analysis/lifted-groq
```

Address-first usage requires no local `.so` and defaults to the reviewed 120B
model when `--provider groq` is selected:

```bash
sos inspect <PROGRAM_ADDRESS> \
  --rpc-url https://api.mainnet.solana.com \
  --provider groq \
  --function fn_0 \
  --out ./reports/<PROGRAM_ADDRESS>
```

Only a compact evidence projection is hosted: canonical operation/block IDs,
bounded disassembly, and high-value Solana semantic facts. The raw ELF is not
sent. The complete pack stays local for hashing and evidence-reference
validation. Generated output is untrusted and cannot change deterministic IR.
Because the projection still reveals program behavior, enable Groq Zero Data
Retention where available and use local inference for confidential binaries.

### Humanizing existing pseudocode

SOS can apply the same evidence-constrained model boundary to textual output
from another decompiler:

```bash
export GROQ_API_KEY='...'
sos humanize ./decompiler-output.txt \
  --provider groq \
  --out ./readable.rs \
  --audit-out ./humanize.json
```

The adapter creates one synthetic `fn_0` and one or more
`op_<hex-operation-index>` records per source line. Original line and segment
numbers are embedded in each operation. The model
must cite valid operation, block, or function IDs, and SOS rejects unknown
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

Analyze a program and run all four lifting stages:

```bash
cargo run -p sos-cli -- analyze ./program.so --out ./analysis
cargo run -p sos-cli -- lift ./analysis \
  --model gpt-oss:20b \
  --ollama-url http://127.0.0.1:11434 \
  --repair-rounds 2 \
  --out ./analysis/lifted
```

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

For source-backed corpus fixtures, run the deterministic behavior judge after
lifting. It fails when required observable semantics are absent:

```bash
python3 scripts/evaluate-readable.py \
  ./analysis/lifted/readable.rs \
  fixtures/c/expected-semantics.json \
  --out ./analysis/lifted/quality.json
```

## Current trust boundary

`readable.rs` is a hypothesis until independent checks pass. Compilation proves
only syntax and type consistency. Replay proves behavior only for the selected
paths. The remaining judge work is function-level recompilation scoring,
non-regression enforcement across the call graph, type-layout verification, and
multi-transaction path coverage.
