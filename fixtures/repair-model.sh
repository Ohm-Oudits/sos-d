#!/bin/sh
# Deterministic local provider fixture for the bounded mismatch-driven repair loop.
IFS= read -r request || true
if printf '%s' "$request" | grep -q '"task":"structure_recovery"'; then
  printf '%s\n' '{"task":"structure_recovery","model":"local-repair","prompt_version":"sos-v1","structured_output":{"functions":[]},"evidence":[]}'
elif printf '%s' "$request" | grep -q '"task":"semantic_labeling"'; then
  printf '%s\n' '{"task":"semantic_labeling","model":"local-repair","prompt_version":"sos-v1","structured_output":{"labels":[]},"evidence":[]}'
elif printf '%s' "$request" | grep -q '"task":"refinement"' && printf '%s' "$request" | grep -q 'external_validation'; then
  printf '%s\n' '{"task":"refinement","model":"local-repair","prompt_version":"sos-v1","structured_output":{"schema_version":1,"patches":[{"function_id":"fn_0","replacement":"fn fn_0() { sos_raw_op!(0, 0, 0, 0, 0); unknown_return(); }","evidence":[{"source":"ir","location":"function:fn_0"}]}],"observations":[{"kind":"log_contains","value":"SOS hello world","evidence":[{"source":"ir","location":"function:fn_0"}]}],"validation":{"status":"unverified","notes":["external validation required"]}},"evidence":[]}'
elif printf '%s' "$request" | grep -q '"task":"refinement"'; then
  printf '%s\n' '{"task":"refinement","model":"local-repair","prompt_version":"sos-v1","structured_output":{"schema_version":1,"patches":[{"function_id":"fn_0","replacement":"fn broken( {","evidence":[{"source":"ir","location":"function:fn_0"}]}],"observations":[{"kind":"log_contains","value":"missing behavior","evidence":[{"source":"ir","location":"function:fn_0"}]}],"validation":{"status":"unverified","notes":["external validation required"]}},"evidence":[]}'
else
  printf '%s\n' '{"task":"explanation","model":"local-repair","prompt_version":"sos-v1","structured_output":{"summary":"A bounded candidate requires external validation.","observations":[]},"evidence":[]}'
fi
