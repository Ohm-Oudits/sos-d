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

## Staged agents

The current agent chain is:

1. `structure_recovery`: reconstruct readable regions and function bodies.
2. `semantic_labeling`: propose account roles, types, CPIs, PDAs, and checks.
3. `explanation`: summarize security-relevant behavior.
4. `refinement`: improve source while preserving evidence references.
5. Local validation and bounded repair: reject malformed, unsupported, or
   evidence-free output and compare rebuilt candidates when a compiler is
   available.

Each stage has a distinct JSON schema, model identity, task-specific output
budget, and request hash. Outputs become bounded, explicitly untrusted context
for later stages. Checkpoints are scoped by source hash, provider, default and
stage model identities, prompt version, stage input, and stage output. A changed
provider or stage model cannot silently reuse another result.

`--model` selects the default agent model. The following optional overrides
implement Patina-style role specialization while retaining one provider and one
credential boundary per run:

| Stage | Override |
| --- | --- |
| Structure recovery | `--structure-model MODEL` |
| Semantic labeling | `--semantic-model MODEL` |
| Explanation | `--explanation-model MODEL` |
| Refinement and compiler repair | `--refinement-model MODEL` |

For example, an API provider can use a stronger reconstruction/refinement model
and a smaller labeling/explanation model:

```bash
sos lift ./analysis \
  --provider openai \
  --model default-model-id \
  --structure-model reconstruction-model-id \
  --semantic-model labeling-model-id \
  --explanation-model explanation-model-id \
  --refinement-model reconstruction-model-id \
  --resume \
  --out ./analysis/lifted
```

Every model identity appears in its stage audit record. Compiler-driven repair
uses the refinement model. Hosted providers that SOS pins to one reviewed model
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
