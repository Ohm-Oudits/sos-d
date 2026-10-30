# SOS JSON contracts

The files in `schemas/v1/`, `schemas/v2/`, and `schemas/v3/` are the public, forward-compatible
JSON contracts for SOS bundles. The Rust structs remain the source of serialization
behavior; these schemas document the wire format for external consumers and CI
tooling. The replay trace is v2 because account-state provenance was added after
the original trace contract. The AI pipeline bundle is v3 because an
evidence-constrained signature-recovery stage now sits between structure
recovery and semantic labeling. Version 2 added each stage's enforced
output-token budget in addition to model identity.
The evidence pack is v2 because gateway-accepted signature hypotheses are
carried as response-hash-bound contracts that later source stages must preserve.
For textual pseudocode, it also carries deterministic source-line obligations
for declarations, branches, loops, calls, writes, returns, and statements so a
humanization response can prove coverage with unique concrete output fragments.

Compatibility rules:

- `schema_version` is required on every top-level contract.
- A major schema change creates `schemas/v2/`; v1 fields are not silently
  reinterpreted.
- Consumers must ignore unknown fields so additive fields remain compatible.
- `status`, `confidence`, and `evidence` are part of the meaning of a claim and
  must not be discarded by adapters.

The checked-in contract tests in `sos-types` validate representative serialized
bundles, required top-level keys, and schema-version compatibility without
requiring a network or model provider.
The AI pipeline contract additionally records each stage's provider/model identity,
output-token budget, prompt version, input hash, output hash, and
accepted/rejected/unavailable status.
The refinement contract separately bounds source patches and directly comparable
candidate observations.
The validation contract keeps compiler status and behavioral status distinct;
successful compilation alone is never represented as behavioral agreement.
Its MIR summary also keeps memory-write destination dependencies separate from
stored-value dependencies, both in aggregate and per write, so pointer/value
role swaps and crossed write pairs remain explicit mismatches.
The binary-judge contract records content hashes, per-function structural scores,
the configured acceptance threshold, exact-match status, and explicit limitations;
it is a deterministic regression signal rather than equivalence proof.
The rebuild-judge contract additionally records the strict Solana toolchain gate,
locked dependency graph, package-source hash, exact build arguments, bounded logs,
build duration, isolated output directory, and emitted candidate hash.
The model-qualification corpus and result contracts separate reviewed weight
identity from measured task behavior and preserve reproducible per-case evidence.
