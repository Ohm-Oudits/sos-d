#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output_root="${1:-$repo_root/benchmarks/latest}"
cd "$repo_root"

./scripts/build-fixtures.sh
cargo build -p sos-cli --all-features
mkdir -p "$output_root"

target/debug/sos benchmark target/deploy/sos_hello_world.so \
  --repeats 3 --out "$output_root/sos-hello-world.json"
target/debug/sos benchmark target/deploy/sos_native_abi.so \
  --repeats 3 --out "$output_root/sos-native-abi.json"
target/debug/sos benchmark target/deploy/sos_pinocchio_fixture.so \
  --repeats 3 --out "$output_root/sos-pinocchio.json"
target/debug/sos benchmark target/deploy/sos_anchor_fixture.so \
  --repeats 3 --out "$output_root/sos-anchor.json"
target/debug/sos benchmark target/deploy/sos_c_fixture.so \
  --repeats 3 --out "$output_root/sos-c.json"

echo "Wrote benchmark results to $output_root"
