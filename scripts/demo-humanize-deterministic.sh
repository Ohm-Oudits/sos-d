#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

input="${1:-fixtures/pseudocode/account_guard.txt}"
out_dir="${2:-$(mktemp -d /tmp/sos-humanize-deterministic-demo.XXXXXX)}"
mkdir -p "$out_dir"

cargo run -q -p sos-cli -- humanize "$input" \
  --deterministic \
  --out "$out_dir/readable.rs" \
  --audit-out "$out_dir/humanize.json"

echo "Input pseudocode: $input"
echo "Readable output:  $out_dir/readable.rs"
echo "Evidence audit:   $out_dir/humanize.json"
