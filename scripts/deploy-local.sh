#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

artifact="target/deploy/sos_hello_world.so"
if [[ ! -s "$artifact" ]]; then
  ./scripts/build-example.sh
fi

solana program deploy --url localhost "$artifact"
