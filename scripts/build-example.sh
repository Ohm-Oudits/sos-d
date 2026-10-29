#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

command -v cargo-build-sbf >/dev/null 2>&1 || {
  echo "cargo-build-sbf is required; install the Solana/Agave CLI toolchain" >&2
  exit 1
}

cargo-build-sbf --manifest-path programs/hello-world/Cargo.toml

artifact="target/deploy/sos_hello_world.so"
test -s "$artifact"
echo "Built $artifact"
shasum -a 256 "$artifact"
