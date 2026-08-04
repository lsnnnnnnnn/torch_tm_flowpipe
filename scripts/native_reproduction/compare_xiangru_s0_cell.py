#!/usr/bin/env python3
"""Compare one S0 policy cell while keeping source identity explicit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from compare_xiangru_s0 import compare_value, sha256


INPUT_HASH_FIELDS = (
    "source_dr_sha256",
    "source_q3_sha256",
    "fixture_sha256",
    "config_sha256",
    "controller_sha256",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--fresh", required=True, type=Path)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--method", default="complete_q3")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    args = parser.parse_args()

    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    fresh = json.loads(args.fresh.read_text(encoding="utf-8"))
    reference_cell = reference["cells"][args.policy][args.method]
    fresh_cell = fresh["cells"][args.policy][args.method]
    differences: list[dict] = []
    stats = {
        "numeric_fields_compared": 0,
        "maximum_absolute_error": 0.0,
        "maximum_relative_error": 0.0,
        "excluded_runtime_or_memory_fields": 0,
    }
    compare_value(
        reference_cell,
        fresh_cell,
        f"cells.{args.policy}.{args.method}",
        args.tolerance,
        differences,
        stats,
    )
    inputs = [
        {
            "field": field,
            "reference": reference["artifacts"].get(field),
            "fresh": fresh["artifacts"].get(field),
            "equal": (
                reference["artifacts"].get(field)
                == fresh["artifacts"].get(field)
            ),
        }
        for field in INPUT_HASH_FIELDS
    ]
    behavior_agrees = not differences and all(item["equal"] for item in inputs)
    source_identity = {
        "reference_sha": reference.get("crown_reach_commit"),
        "reference_dirty": reference.get("crown_reach_dirty"),
        "fresh_sha": fresh.get("crown_reach_commit"),
        "fresh_dirty": fresh.get("crown_reach_dirty"),
    }
    source_identity["exact"] = bool(
        source_identity["reference_sha"] == source_identity["fresh_sha"]
        and source_identity["reference_dirty"] is False
        and source_identity["fresh_dirty"] is False
    )
    payload = {
        "schema_version": 1,
        "status": (
            "PASS_BEHAVIOR_SOURCE_IDENTITY_MISMATCH"
            if behavior_agrees and not source_identity["exact"]
            else "PASS_WITH_DECLARED_TOLERANCE"
            if behavior_agrees
            else "FAIL"
        ),
        "reference": str(args.reference.resolve()),
        "reference_sha256": sha256(args.reference),
        "fresh": str(args.fresh.resolve()),
        "fresh_sha256": sha256(args.fresh),
        "policy": args.policy,
        "method": args.method,
        "source_identity": source_identity,
        "input_hashes": inputs,
        "comparison_policy": {
            "numeric_absolute_tolerance": args.tolerance,
            "tolerance_source": (
                "Xiangru run_s0_tora_static_partition_sweep.py "
                "CONTROLLER_TOLERANCE = 1e-6"
            ),
            "relative_error_denominator": (
                "max(abs(reference), numeric_absolute_tolerance)"
            ),
            "excluded_from_correctness": [
                "*.peak_cuda_memory_bytes",
                "*.timing",
                "*._seconds",
            ],
            "runtime_compared": False,
        },
        **stats,
        "differences": differences,
        "reproduction_eligible": behavior_agrees and source_identity["exact"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0 if behavior_agrees else 1


if __name__ == "__main__":
    raise SystemExit(main())
