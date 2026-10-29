#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

command -v cargo-build-sbf >/dev/null 2>&1 || {
  echo "cargo-build-sbf is required; install the Solana/Agave CLI toolchain" >&2
  exit 1
}

cargo-build-sbf --manifest-path programs/hello-world/Cargo.toml
cargo-build-sbf --manifest-path programs/native-abi/Cargo.toml
cargo-build-sbf --manifest-path programs/pinocchio-fixture/Cargo.toml
cargo-build-sbf --manifest-path programs/anchor-fixture/Cargo.toml
./scripts/build-c-fixture.sh

test -s target/deploy/sos_hello_world.so
test -s target/deploy/sos_native_abi.so
test -s target/deploy/sos_pinocchio_fixture.so
test -s target/deploy/sos_anchor_fixture.so
test -s target/deploy/sos_c_fixture.so
echo "Built the available SOS fixture corpus."
