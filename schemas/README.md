# SOS JSON contracts

The files under `schemas/v*/` are the public, forward-compatible
JSON contracts for SOS bundles. The Rust structs remain the source of serialization
behavior; these schemas document the wire format for external consumers and CI
tooling. The replay trace is v2 because account-state provenance was added after
the original trace contract. AI pipeline bundle v3 introduced an
evidence-constrained signature-recovery stage between structure recovery and
semantic labeling; v2 had added each stage's enforced output-token budget in
addition to model identity. The current v7 contract also enforces the v4
compiler-verifiable memory-layout evidence pack described below.
The evidence pack is v2 because gateway-accepted signature hypotheses are
carried as response-hash-bound contracts that later source stages must preserve.
For textual pseudocode, it also carries deterministic source-line obligations
for declarations, branches, loops, calls, writes, returns, and statements so a
humanization response can prove coverage with unique concrete output fragments.
For binary structure recovery, it carries deterministic CFG region contracts
for branch arms/joins and loop bodies/exits/cycle edges. These contracts are
derived from dominators, postdominators, backedges, and recursive SCCs and must
map to distinct source headers in accepted model output.
Evidence-pack v3 adds immutable fixed-offset memory-field contracts. It records
logical and observed machine offsets, widths, access kinds/PCs, conservative
field paths, and whether a semantic SolAccountInfo name was deterministically
recovered. AI-bundle v6 invalidates older checkpoints that did not prove these
contracts in both structure and refinement output.
Evidence-pack v4 adds deterministic Rust layout type/field identifiers. Final
refinements must preserve those identifiers and offsets in their mappings and
declare matching `#[repr(C)]` records. SOS injects rustc `offset_of!` and
primitive field-type assertions. AI-bundle v7 invalidates checkpoints created
before this compiler proof was part of refinement schema v2.

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
