#!/usr/bin/env python3
"""Score readable decompiler output against a versioned behavior contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def normalize(value: str) -> str:
    return " ".join(value.casefold().split())


def evaluate(source: str, contract: dict) -> dict:
    normalized = normalize(source)
    checks = []
    for expectation in contract.get("expectations", []):
        alternatives = [str(value) for value in expectation.get("any", [])]
        matched = next(
            (value for value in alternatives if normalize(value) in normalized), None
        )
        checks.append(
            {
                "id": expectation.get("id", "unknown"),
                "description": expectation.get("description", ""),
                "passed": matched is not None,
                "matched": matched,
                "alternatives": alternatives,
            }
        )
    passed = sum(1 for check in checks if check["passed"])
    score = passed / len(checks) if checks else 0.0
    minimum = float(contract.get("minimum_score", 1.0))
    return {
        "schema_version": 1,
        "fixture": contract.get("fixture"),
        "score": round(score, 4),
        "minimum_score": minimum,
        "accepted": score >= minimum,
        "passed": passed,
        "total": len(checks),
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("contract", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    source = args.source.read_text(encoding="utf-8")
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    result = evaluate(source, contract)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.write_text(rendered, encoding="utf-8")
    else:
        sys.stdout.write(rendered)
    return 0 if result["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
