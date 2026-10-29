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

## Running a lift

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
  --out ./analysis/lifted
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
