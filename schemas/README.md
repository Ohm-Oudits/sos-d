# SOS JSON contracts

The files in `schemas/v1/` and `schemas/v2/` are the public, forward-compatible
JSON contracts for SOS bundles. The Rust structs remain the source of serialization
behavior; these schemas document the wire format for external consumers and CI
tooling. The replay trace is currently v2 because account-state provenance was
added after the original trace contract.

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
prompt version, input hash, output hash, and accepted/rejected/unavailable status.
The refinement contract separately bounds source patches and directly comparable
candidate observations.
The validation contract keeps compiler status and behavioral status distinct;
successful compilation alone is never represented as behavioral agreement.
The binary-judge contract records content hashes, per-function structural scores,
the configured acceptance threshold, exact-match status, and explicit limitations;
it is a deterministic regression signal rather than equivalence proof.
