#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

test -n "${GROQ_API_KEY:-}" || {
  echo "GROQ_API_KEY is required; create a free Groq API key and export it in this shell" >&2
  exit 1
}

artifact="target/deploy/sos_c_fixture.so"
contract="fixtures/c/expected-semantics.json"
out_dir="${1:-$(mktemp -d /tmp/sos-groq-demo.XXXXXX)}"
analysis_dir="$out_dir/analysis"
lift_dir="$out_dir/lifted"

command -v jq >/dev/null 2>&1 || {
  echo "jq is required" >&2
  exit 1
}
test -s "$artifact" || ./scripts/build-c-fixture.sh

mkdir -p "$analysis_dir" "$lift_dir"
jq -n \
  --arg model "openai/gpt-oss-120b" \
  --arg weights_url "https://huggingface.co/openai/gpt-oss-120b" \
  --arg license "Apache-2.0" \
  '{schema_version: 1, runtime: "groq", model: $model, weights_url: $weights_url, license: $license, raw_elf_uploaded: false}' \
  >"$lift_dir/model-provenance.json"

cargo run -q -p sos-cli -- analyze "$artifact" --out "$analysis_dir"
cargo run -q -p sos-cli -- lift "$analysis_dir" \
  --provider groq \
  --function fn_0 \
  --repair-rounds 2 \
  --out "$lift_dir"
python3 scripts/evaluate-readable.py \
  "$lift_dir/readable.rs" \
  "$contract" \
  --out "$lift_dir/quality.json"

jq '{model, provider, stages: [.stages[] | {task, status, error}]}' "$lift_dir/model.json"
jq '{score, accepted, passed, total}' "$lift_dir/quality.json"
echo "Readable output: $lift_dir/readable.rs"
echo "Audit bundle:    $lift_dir/model.json"
echo "Model provenance: $lift_dir/model-provenance.json"
echo "Quality result:  $lift_dir/quality.json"
