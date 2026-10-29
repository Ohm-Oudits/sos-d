#!/bin/sh
# Minimal local-provider protocol fixture for integration tests.
# It reads a ModelRequest from stdin and emits a schema-valid explanation response.
IFS= read -r request || true
if printf '%s' "$request" | grep -q '"task":"structure_recovery"'; then
  printf '%s\n' '{"task":"structure_recovery","model":"local","prompt_version":"sos-v1","structured_output":{"functions":[]},"evidence":[]}'
elif printf '%s' "$request" | grep -q '"task":"semantic_labeling"'; then
  printf '%s\n' '{"task":"semantic_labeling","model":"local","prompt_version":"sos-v1","structured_output":{"labels":[]},"evidence":[]}'
elif printf '%s' "$request" | grep -q '"task":"refinement"'; then
  printf '%s\n' '{"task":"refinement","model":"local","prompt_version":"sos-v1","structured_output":{"patches":[],"validation":{"status":"not_available","notes":[]}},"evidence":[]}'
else
  printf '%s\n' '{"task":"explanation","model":"local","prompt_version":"sos-v1","structured_output":{"summary":"No additional explanation was generated.","observations":[]},"evidence":[]}'
fi
