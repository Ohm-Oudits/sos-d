#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
rpc_url="http://127.0.0.1:8899"
cd "$repo_root"
artifact="target/deploy/sos_hello_world.so"
program_keypair="target/deploy/sos_hello_world-keypair.json"

command -v solana-test-validator >/dev/null 2>&1 || {
  echo "solana-test-validator is required" >&2
  exit 1
}
command -v solana >/dev/null 2>&1 || {
  echo "solana CLI is required" >&2
  exit 1
}
command -v solana-keygen >/dev/null 2>&1 || {
  echo "solana-keygen is required" >&2
  exit 1
}
test -s "$artifact" || {
  echo "missing $artifact; run ./scripts/build-fixtures.sh first" >&2
  exit 1
}
test -s "$program_keypair" || {
  echo "missing $program_keypair; run ./scripts/build-fixtures.sh first" >&2
  exit 1
}

ledger_dir="$(mktemp -d /tmp/sos-validator.XXXXXX)"
validator_log="$(mktemp /tmp/sos-validator-log.XXXXXX)"
out_dir="$(mktemp -d /tmp/sos-inspect.XXXXXX)"
validator_pid=""
cleanup() {
  if [[ -n "$validator_pid" ]]; then
    kill "$validator_pid" 2>/dev/null || true
    wait "$validator_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

solana-test-validator --ledger "$ledger_dir" --quiet >"$validator_log" 2>&1 &
validator_pid=$!
for _ in $(seq 1 60); do
  if solana cluster-version --url "$rpc_url" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done
solana cluster-version --url "$rpc_url" >/dev/null

program_id="$(solana-keygen pubkey "$program_keypair")"
solana program deploy \
  --url "$rpc_url" \
  "$artifact" >/dev/null
for _ in $(seq 1 60); do
  if solana account "$program_id" --url "$rpc_url" --commitment finalized >/dev/null 2>&1; then
    break
  fi
  sleep 1
done
solana account "$program_id" --url "$rpc_url" --commitment finalized >/dev/null

cargo build -q -p sos-cli
"$repo_root/target/debug/sos" inspect "$program_id" \
  --rpc-url "$rpc_url" \
  --commitment finalized \
  --out "$out_dir" \
  --model local \
  --model-command "$repo_root/fixtures/repair-model.sh" \
  --validate \
  --repair \
  --repair-rounds 3 \
  --validation-trace "$repo_root/fixtures/hello-candidate-trace.json" \
  --json >"$out_dir/summary.json"

test -s "$out_dir/manifest.json"
test -s "$out_dir/analysis.json"
test -s "$out_dir/semantics.json"
test -s "$out_dir/ir.json"
test -s "$out_dir/report.md"
test -s "$out_dir/pseudocode.rs"
test -s "$out_dir/pseudocode.c"
test -s "$out_dir/ai-report.md"
test "$(jq -r '.compilation.status' "$out_dir/validation.json")" = "passed"
test "$(jq -r '.behavioral_status' "$out_dir/validation.json")" = "behaviorally_validated"
test "$(jq '.repair.rounds | length' "$out_dir/validation.json")" -eq 2
test "$(jq -r '.repair.status' "$out_dir/validation.json")" = "behaviorally_validated"
test "$(jq '.stages | length' "$out_dir/model.json")" -eq 4
test "$(jq '[.stages[] | select(.status == "accepted")] | length' "$out_dir/model.json")" -eq 4
echo "local inspect E2E passed; bundle=$out_dir"
