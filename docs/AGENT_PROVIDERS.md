# Agent providers and credential boundary

SOS uses models as untrusted reconstruction agents around a deterministic
decompiler core. A model may propose control-flow structure, names, types,
Solana concepts, explanations, and source refinements. It cannot alter the
decoded SBPF instructions, CFG, memory facts, data-flow facts, strings, or
artifact provenance.

This design is informed by:

- OtterSec's [GoonFi V2 decompilation](https://osec.io/blog/goonfi-decompilation/),
  especially its function/handler partitioning, compile-and-compare loop, and
  rejection of source-level shortcuts.
- `renato-osec/patina-research` at commit
  `77b7c017741bfb54a176afbfb01da4f85b0530cd`, especially its staged agents,
  per-stage turn/token budgets, durable JSON sidecars, verified stage
  hand-offs, and stop-on-failure behavior. SOS uses those architectural ideas
  as research input and does not copy Patina code.

## Trust boundary

```text
deployed address / .so
  -> deterministic ELF + SBPF analysis
  -> immutable, bounded evidence pack
  -> untrusted model stage
  -> schema + evidence-reference + anti-cheese validation
  -> accepted hypothesis or explicit rejection
  -> compiler/replay/binary comparison gates
```

The model receives a compact projection rather than the raw ELF. A stage result
is accepted only after local validation. Accepted output is still marked
unverified until compiler, replay, and binary-similarity checks support it.
Exact internal callees are supplied as immutable call contracts. Path-dependent
indirect calls are different: the compact projection may contain a bounded
`call_candidates` list with no exact target. Providers are instructed to preserve an
uncertain dispatch over those alternatives, never select one callee, claim the list
is exhaustive, or emit an exact-call mapping for it.

## Staged agents

The current agent chain is:

1. `signature_recovery`: infer evidence-linked function names, parameter roles,
   parameter types, and return types while preserving explicit unknowns.
2. `structure_recovery`: reconstruct readable regions and function bodies under
   the accepted signature. Every deterministic branch/loop region must map to a
   unique concrete source header citing its exact CFG header block.
3. `semantic_labeling`: propose account roles, types, CPIs, PDAs, and checks.
4. `explanation`: summarize security-relevant behavior.
5. `refinement`: improve source while preserving evidence references.
6. Local validation and bounded repair: reject malformed, unsupported, or
   evidence-free output and compare rebuilt candidates when a compiler is
   available.

Each stage has a distinct JSON schema, model identity, task-specific output
budget, and request hash. Outputs become bounded, explicitly untrusted context
for later stages. Checkpoints are scoped by source hash, provider, default and
stage model identities, prompt version, stage input, and stage output. A changed
provider or stage model cannot silently reuse another result.
Structure-stage region contracts are derived locally from dominators,
postdominators, natural backedges, and recursive SCCs; they are not model
claims. Prompt v13 and lift-cache schema v14 invalidate checkpoints that predate the
current CFG, memory-layout, exact-call, and indirect-call-candidate rules.

`--model` selects the default agent model. The following optional overrides
implement Patina-style role specialization while retaining one provider and one
credential boundary per run:

| Stage | Model override | Output-token override | Default |
| --- | --- | --- | ---: |
| Structure recovery | `--structure-model MODEL` | `--structure-tokens N` | 1,200 |
| Signature recovery | `--signature-model MODEL` | `--signature-tokens N` | 1,200 |
| Semantic labeling | `--semantic-model MODEL` | `--semantic-tokens N` | 1,200 |
| Explanation | `--explanation-model MODEL` | `--explanation-tokens N` | 640 |
| Refinement and compiler repair | `--refinement-model MODEL` | `--refinement-tokens N` | 1,200 |

Token overrides must be between 128 and 32,768. SOS sends the selected value to
Ollama as `num_predict` and to hosted APIs through their corresponding maximum
output-token field. It is also stored in the stage audit and checkpoint request
identity. This makes Patina-style stage budgets enforceable and reproducible.
For GPT-OSS through Ollama, SOS requests the model's supported `low` reasoning
level so hidden reasoning does not consume the entire bounded generation before
final JSON. If that still occurs, the stage is rejected with an explicit token-
budget diagnostic rather than accepting empty output.

For example, an API provider can use a stronger reconstruction/refinement model
and a smaller labeling/explanation model:

```bash
sos lift ./analysis \
  --provider openai \
  --model default-model-id \
  --structure-model reconstruction-model-id \
  --signature-model signature-model-id \
  --semantic-model labeling-model-id \
  --explanation-model explanation-model-id \
  --refinement-model reconstruction-model-id \
  --structure-tokens 2400 \
  --signature-tokens 1200 \
  --semantic-tokens 800 \
  --explanation-tokens 512 \
  --refinement-tokens 2400 \
  --resume \
  --out ./analysis/lifted
```

Every model identity appears in its stage audit record. Compiler-driven repair
uses both the refinement model and refinement token budget. Hosted providers that SOS pins to one reviewed model
(Groq, Hugging Face, and Cloudflare) reject a different override rather than
quietly routing to unreviewed weights. Ollama overrides must remain in the
reviewed registry unless `--allow-unreviewed-model` is explicitly supplied.

## Provider choices

For trusted open weights, prefer one of the reviewed GPT-OSS routes:

| Provider | Model identity | Credential |
| --- | --- | --- |
| Ollama | `gpt-oss:20b` | none; local process |
| Groq | `openai/gpt-oss-120b` | `GROQ_API_KEY` |
| Hugging Face | `openai/gpt-oss-120b:fireworks-ai` | `HF_TOKEN` |
| Cloudflare Workers AI | `@cf/openai/gpt-oss-120b` | `CLOUDFLARE_ACCOUNT_ID`, `CLOUDFLARE_API_TOKEN` |

Registry review establishes publisher, license, and expected weight family; it
does not establish task competence. Run the reproducible qualification corpus
against the exact local model/runtime before a security-sensitive lift:

```bash
sos qualify-model fixtures/model-qualification-v1.json \
  --provider ollama --model gpt-oss:20b \
  --out model-qualification.json
```

The same command supports `groq`, `huggingface`, `cloudflare`, `openai`, and
`anthropic` using the credential variables below. Provider-neutral adapters can
be assessed with `--provider command --model-command PATH`. The result records
each case's input hash, schema/evidence
acceptance, missing concept groups, forbidden matches, output, score, and the
explicit limitations of corpus-based evaluation.

SOS also has direct agent transports for the OpenAI Responses API and Anthropic
Messages API. These are execution providers, not entries in SOS's reviewed
open-weight registry. The operator must choose an exact model ID with `--model`
or `SOS_OPENAI_MODEL` / `SOS_ANTHROPIC_MODEL`; SOS intentionally has no mutable
"latest" default.

| Provider | Credential | Optional configuration |
| --- | --- | --- |
| OpenAI | `OPENAI_API_KEY` | `SOS_OPENAI_MODEL` |
| Anthropic | `ANTHROPIC_API_KEY` | `SOS_ANTHROPIC_MODEL`, `ANTHROPIC_WORKSPACE_ID` |

OpenAI requests use strict structured output through the Responses API.
Anthropic requests use structured output through the Messages API. Anthropic's
generation schema is reduced to its supported JSON Schema subset, then SOS
revalidates the response against the original stricter local schema.

## Configuration

Copy `.env.example` as a reference, but export values through your shell or
process manager. SOS does not parse `.env` files and never writes credentials to
model requests, checkpoints, audit output, logs, or cache identities.

```bash
export OPENAI_API_KEY='...'
export SOS_OPENAI_MODEL='your-explicit-model-id'
sos humanize ./pseudocode.c \
  --provider openai \
  --resume \
  --out ./readable.rs \
  --audit-out ./humanize.json
```

```bash
export ANTHROPIC_API_KEY='...'
export SOS_ANTHROPIC_MODEL='your-explicit-model-id'
# Optional: export ANTHROPIC_WORKSPACE_ID='...'
sos lift ./analysis \
  --provider anthropic \
  --resume \
  --function fn_0 \
  --out ./analysis/lifted-anthropic
```

The command-line model flag takes precedence over the model environment
variable:

```bash
sos inspect <PROGRAM_ADDRESS> \
  --provider openai \
  --model your-explicit-model-id \
  --structure-model your-reconstruction-model-id \
  --refinement-model your-reconstruction-model-id \
  --structure-tokens 2400 \
  --refinement-tokens 2400 \
  --resume \
  --out ./reports/<PROGRAM_ADDRESS>
```

With `--resume`, a missing credential still permits local checkpoint validation.
`humanize` emits an explicitly labeled deterministic syntax-only fallback for
cache misses; `lift` and `inspect` preserve misses as unavailable. Without
`--resume`, a missing credential is a configuration error.

## Operational safeguards

- Never commit `.env`; only the empty `.env.example` is tracked.
- Use least-privilege project/workspace keys and provider retention controls.
- Pin a concrete model/version where the provider supports it.
- Run one function or handler first (`--function fn_ID`) and review evidence
  references before scaling to a complete program.
- Treat readable output as a hypothesis. Recompile, replay, and use `sos judge`
  before making security claims.
- Use `sos humanize --deterministic` when code or metadata must not leave the
  machine.
