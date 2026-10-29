# Contributing to SOS Decompiler

SOS is an evidence-first Solana program analysis project. Contributions should
preserve the distinction between deterministic facts, recovered structure, and
hypotheses produced by heuristics or AI.

## Before opening a change

Run the host checks from the repository root:

```bash
cargo fmt --all -- --check
cargo test --workspace --all-features
cargo clippy --workspace --all-targets --all-features -- -D warnings
```

When changing fixture or SBF behavior, also run:

```bash
./scripts/build-fixtures.sh
./scripts/benchmark-fixtures.sh /tmp/sos-bench
```

Do not commit generated `target/` output, RPC credentials, model tokens, or
production transaction/account data.

## Design expectations

- Keep deterministic analysis independent of model providers and GUI tools.
- Attach evidence and an explicit status to every new semantic claim.
- Preserve competing interpretations when bytecode evidence is ambiguous.
- Treat on-chain strings, logs, account bytes, and transaction fields as
  untrusted data.
- Add malformed-input and regression tests for parser, loader, and validator
  changes.
- Do not add live-chain write or wallet-signing behavior to the core product.

Protocol and detector extensions should use the public registry/interfaces where
possible and document their evidence sources and limitations.
