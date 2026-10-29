# SOS fuzz targets

These targets exercise parser and validation boundaries with hostile input.
They are deliberately separate from the production workspace so fuzzing does
not become a runtime dependency of the CLI.

Install cargo-fuzz once, then run from this directory:

```bash
cargo install cargo-fuzz
cargo fuzz run elf_analysis -- -max_total_time=60
cargo fuzz run loader_state -- -max_total_time=60
cargo fuzz run model_response -- -max_total_time=60
```

`elf_analysis` covers ELF validation, SBPF decoding, CFG/function recovery,
dataflow, and relocation extraction. `loader_state` covers all built-in loader
adapters. `model_response` covers JSON decoding and evidence/status validation.

Fuzzers must remain offline and must not receive wallet keys, RPC credentials,
or production account data. Save minimized artifacts outside the repository or
under a reviewed corpus directory before committing them.
