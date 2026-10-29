# SOS C/SBF fixture

This small C program is a source-backed corpus member for the decompiler. It
uses the Solana C SDK entrypoint ABI, deserializes two account slots, branches
on instruction data, emits logs, and validates signer/writable account roles.

Build it with `scripts/build-c-fixture.sh`. The script uses the active Solana
platform-tools SBF SDK and produces `target/deploy/sos_c_fixture.so`.
