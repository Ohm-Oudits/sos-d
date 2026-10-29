#!/usr/bin/env python3
"""Synthetic provider fixture for the qualify-model CLI integration test."""

import json
import sys


request = json.load(sys.stdin)
function_id = request["evidence_pack"]["ir"]["functions"][0]["id"]
response = {
    "task": request["task"],
    "model": request["model"],
    "prompt_version": request["prompt_version"],
    "structured_output": {
        "functions": [
            {
                "function_id": function_id,
                "pseudocode": (
                    f"fn {function_id}() {{ /* instruction tag and length */ "
                    "require signer; verify owner; reject invalid input; return 0; "
                    "match opcode; invoke token program CPI for transfer or close account; "
                    "perform bounds check on index and checked overflow arithmetic; "
                    "update balance; loop over destination buffer and clear to zero; }"
                ),
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
