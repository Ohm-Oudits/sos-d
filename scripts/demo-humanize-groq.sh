#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

test -n "${GROQ_API_KEY:-}" || {
  echo "GROQ_API_KEY is required; create a free Groq API key and export it in this shell" >&2
  exit 1
}

input="${1:-fixtures/pseudocode/account_guard.txt}"
out_dir="${2:-$(mktemp -d /tmp/sos-humanize-demo.XXXXXX)}"
mkdir -p "$out_dir"

cargo run -q -p sos-cli -- humanize "$input" \
  --provider groq \
  --resume \
  --cache-dir "$out_dir/checkpoints" \
  --out "$out_dir/readable.rs" \
  --audit-out "$out_dir/humanize.json"

echo "Input pseudocode: $input"
echo "Readable output:  $out_dir/readable.rs"
echo "Evidence audit:   $out_dir/humanize.json"
echo "Checkpoints:      $out_dir/checkpoints"
