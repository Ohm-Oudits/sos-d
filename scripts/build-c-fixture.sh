#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
sbf_sdk="${SOS_SBF_SDK:-}"

if [[ -z "$sbf_sdk" ]]; then
  active_bin="$(dirname "$(command -v cargo-build-sbf 2>/dev/null || true)")"
  if [[ -d "$active_bin/platform-tools-sdk/sbf" ]]; then
    sbf_sdk="$active_bin/platform-tools-sdk/sbf"
  fi
fi

if [[ ! -f "$sbf_sdk/c/sbf.mk" ]]; then
  echo "SOS_SBF_SDK must point to platform-tools-sdk/sbf (Solana 3.0.0)" >&2
  exit 1
fi

cd "$repo_root/programs/c-fixture"
make -f "$sbf_sdk/c/sbf.mk" \
  SRC_DIR="src" \
  OUT_DIR="../../target/deploy" \
  sos_c_fixture

test -s "$repo_root/target/deploy/sos_c_fixture.so"
echo "Built C fixture at target/deploy/sos_c_fixture.so"
