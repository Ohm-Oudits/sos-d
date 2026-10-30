#!/usr/bin/env python3
"""Synthetic provider fixture for the qualify-model CLI integration test."""

import json
import sys


request = json.load(sys.stdin)
function_id = request["evidence_pack"]["ir"]["functions"][0]["id"]
pseudocode = (
    f"fn {function_id}() {{ let instruction_tag = data[0]; "
    "require_signer(); verify_owner(); "
    "if invalid_input { return error; } "
    "match instruction_tag { transfer => token_program_cpi(), close => close_account_cpi() }; "
    "let updated_balance = checked_arithmetic(); state.balance = updated_balance; "
    "for destination in buffer { destination.clear(); } return 0; } "
    "// instruction length, discriminator, reject invalid input, bounds check on index, "
    "checked overflow arithmetic, transfer or close account, clear destination buffer"
)
obligations = request["evidence_pack"].get("pseudocode_obligations", [])
response = {
    "task": request["task"],
    "model": request["model"],
    "prompt_version": request["prompt_version"],
    "structured_output": {
        "functions": [
            {
                "function_id": function_id,
                "pseudocode": pseudocode,
                "source_mappings": [
                    {
                        "obligation_id": obligation["id"],
                        "output_fragment": pseudocode,
                    }
                    for obligation in obligations
                ],
                "evidence": [
                    {
                        "source": "ir",
                        "location": f"function:{function_id}",
                        "detail": None,
                    }
                ],
            }
        ]
    },
    "evidence": [],
}
json.dump(response, sys.stdout)
