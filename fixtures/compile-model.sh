#!/bin/sh
# Local provider fixture with one evidence-linked refinement patch. The patch is
# intentionally generic and is compiled only as a syntax/type candidate.
IFS= read -r request || true
if printf '%s' "$request" | grep -q '"task":"structure_recovery"'; then
  printf '%s\n' '{"task":"structure_recovery","model":"local-compile","prompt_version":"sos-v1","structured_output":{"functions":[]},"evidence":[]}'
elif printf '%s' "$request" | grep -q '"task":"semantic_labeling"'; then
  printf '%s\n' '{"task":"semantic_labeling","model":"local-compile","prompt_version":"sos-v1","structured_output":{"labels":[]},"evidence":[]}'
elif printf '%s' "$request" | grep -q '"task":"refinement"'; then
  printf '%s\n' '{"task":"refinement","model":"local-compile","prompt_version":"sos-v1","structured_output":{"schema_version":1,"patches":[{"function_id":"fn_0","replacement":"fn fn_0() { sos_raw_op!(0, 0, 0, 0, 0); unknown_return(); }","evidence":[{"source":"ir","location":"function:fn_0"}]}],"observations":[{"kind":"log_contains","value":"SOS hello world","evidence":[{"source":"ir","location":"function:fn_0"}]}],"validation":{"status":"unverified","notes":["external validation required"]}},"evidence":[]}'
else
  printf '%s\n' '{"task":"explanation","model":"local-compile","prompt_version":"sos-v1","structured_output":{"summary":"A bounded candidate requires external validation.","observations":[]},"evidence":[]}'
fi
