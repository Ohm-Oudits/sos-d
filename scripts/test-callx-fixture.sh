#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

artifact="target/deploy/sos_callx_fixture.so"
test -s "$artifact"

audit_dir="$(mktemp -d "${TMPDIR:-/tmp}/sos-callx-fixture.XXXXXX")"
trap 'rm -rf -- "$audit_dir"' EXIT

cargo run --quiet -p sos-cli -- analyze "$artifact" --out "$audit_dir" >/dev/null

jq -e '[.instructions[] | select(.assembly | startswith("callx "))] | length > 0' \
  "$audit_dir/analysis.json" >/dev/null
jq -e '[.calls[] | select(.kind == "indirect" and .status == "unresolved")] | length > 0' \
  "$audit_dir/analysis.json" >/dev/null

echo "Verified compiled callx instructions and conservative unresolved-target evidence."
