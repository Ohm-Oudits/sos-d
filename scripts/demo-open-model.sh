#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

model="${SOS_MODEL:-gpt-oss:20b}"
context_tokens="${SOS_CONTEXT_TOKENS:-32768}"
artifact="target/deploy/sos_c_fixture.so"
contract="fixtures/c/expected-semantics.json"
out_dir="${1:-$(mktemp -d /tmp/sos-open-model-demo.XXXXXX)}"
analysis_dir="$out_dir/analysis"
lift_dir="$out_dir/lifted"

command -v ollama >/dev/null 2>&1 || {
  echo "ollama is required: https://ollama.com/download" >&2
  exit 1
}
command -v jq >/dev/null 2>&1 || {
  echo "jq is required" >&2
  exit 1
}
curl --fail --silent --show-error http://127.0.0.1:11434/api/version >/dev/null || {
  echo "Ollama is not reachable at http://127.0.0.1:11434; start it with: ollama serve" >&2
  exit 1
}
ollama list | awk 'NR > 1 {print $1}' | grep -Fx "$model" >/dev/null || {
  echo "reviewed model is not installed; run: ollama pull $model" >&2
  exit 1
}
test -s "$artifact" || ./scripts/build-c-fixture.sh

mkdir -p "$analysis_dir" "$lift_dir"
model_digest="$(ollama list | awk -v selected="$model" '$1 == selected {print $2}')"
jq -n \
  --arg model "$model" \
  --arg digest "$model_digest" \
  --arg weights_url "https://huggingface.co/openai/gpt-oss-20b" \
  --arg license "Apache-2.0" \
  '{schema_version: 1, runtime: "ollama", model: $model, local_manifest_digest: $digest, weights_url: $weights_url, license: $license}' \
  >"$lift_dir/model-provenance.json"
cargo run -q -p sos-cli -- analyze "$artifact" --out "$analysis_dir"
cargo run -q -p sos-cli -- lift "$analysis_dir" \
  --function fn_0 \
  --model "$model" \
  --context-tokens "$context_tokens" \
  --repair-rounds 2 \
  --out "$lift_dir"
python3 scripts/evaluate-readable.py \
  "$lift_dir/readable.rs" \
  "$contract" \
  --out "$lift_dir/quality.json"

jq '{model, stages: [.stages[] | {task, status, error}]}' "$lift_dir/model.json"
jq '{score, accepted, passed, total}' "$lift_dir/quality.json"
echo "Readable output: $lift_dir/readable.rs"
echo "Audit bundle:    $lift_dir/model.json"
echo "Model provenance: $lift_dir/model-provenance.json"
echo "Quality result:  $lift_dir/quality.json"
