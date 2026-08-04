#!/usr/bin/env python3
"""Compare Xiangru's S0 native result without treating timing as correctness.

The numerical tolerance is the experiment's own CONTROLLER_TOLERANCE from
run_s0_tora_static_partition_sweep.py.  Runtime and peak-memory measurements
are reported but are not correctness fields.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


EXACT_TOP_LEVEL_FIELDS = (
    "schema_version",
    "experiment",
    "status",
    "formal_scope",
    "verified_cells",
    "controls",
    "validity",
    "crown_reach_commit",
    "crown_reach_dirty",
    "diffreach_commit",
    "artifacts",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compare_value(
    reference: Any,
    fresh: Any,
    path: str,
    tolerance: float,
    differences: list[dict[str, Any]],
    stats: dict[str, Any],
) -> None:
    if path.endswith(".peak_cuda_memory_bytes") or path.endswith(".timing"):
        stats["excluded_runtime_or_memory_fields"] += 1
        return
    if path.endswith("_seconds"):
        stats["excluded_runtime_or_memory_fields"] += 1
        return
    if isinstance(reference, bool) or isinstance(fresh, bool):
        if type(reference) is not type(fresh) or reference != fresh:
            differences.append(
                {"path": path, "reference": reference, "fresh": fresh}
            )
        return
    if isinstance(reference, (int, float)) and isinstance(fresh, (int, float)):
        error = abs(float(reference) - float(fresh))
        stats["numeric_fields_compared"] += 1
        stats["maximum_absolute_error"] = max(
            stats["maximum_absolute_error"], error
        )
        relative_error = error / max(abs(float(reference)), tolerance)
        stats["maximum_relative_error"] = max(
            stats["maximum_relative_error"], relative_error
        )
        if not math.isfinite(error) or error > tolerance:
            differences.append(
                {
                    "path": path,
                    "reference": reference,
                    "fresh": fresh,
                    "absolute_error": error,
                }
            )
        return
    if isinstance(reference, dict) and isinstance(fresh, dict):
        if set(reference) != set(fresh):
            differences.append(
                {
                    "path": path,
                    "reference_keys": sorted(reference),
                    "fresh_keys": sorted(fresh),
                }
            )
        for key in sorted(set(reference) & set(fresh)):
            compare_value(
                reference[key],
                fresh[key],
                f"{path}.{key}",
                tolerance,
                differences,
                stats,
            )
        return
    if isinstance(reference, list) and isinstance(fresh, list):
        if len(reference) != len(fresh):
            differences.append(
                {
                    "path": path,
                    "reference_length": len(reference),
                    "fresh_length": len(fresh),
                }
            )
        for index, (reference_item, fresh_item) in enumerate(
            zip(reference, fresh)
        ):
            compare_value(
                reference_item,
                fresh_item,
                f"{path}[{index}]",
                tolerance,
                differences,
                stats,
            )
        return
    if type(reference) is not type(fresh) or reference != fresh:
        differences.append(
            {"path": path, "reference": reference, "fresh": fresh}
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--fresh", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    args = parser.parse_args()

    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    fresh = json.loads(args.fresh.read_text(encoding="utf-8"))
    exact_fields = [
        {
            "field": field,
            "equal": reference.get(field) == fresh.get(field),
        }
        for field in EXACT_TOP_LEVEL_FIELDS
    ]
    differences: list[dict[str, Any]] = []
    stats = {
        "numeric_fields_compared": 0,
        "maximum_absolute_error": 0.0,
        "maximum_relative_error": 0.0,
        "excluded_runtime_or_memory_fields": 0,
    }
    compare_value(
        reference["cells"],
        fresh["cells"],
        "cells",
        args.tolerance,
        differences,
        stats,
    )
    passed = all(item["equal"] for item in exact_fields) and not differences
    payload = {
        "schema_version": 1,
        "status": "PASS_WITH_DECLARED_TOLERANCE" if passed else "FAIL",
        "reference": str(args.reference.resolve()),
        "reference_sha256": sha256(args.reference),
        "fresh": str(args.fresh.resolve()),
        "fresh_sha256": sha256(args.fresh),
        "comparison_policy": {
            "exact_top_level_fields": list(EXACT_TOP_LEVEL_FIELDS),
            "cell_numeric_absolute_tolerance": args.tolerance,
            "relative_error_denominator": (
                "max(abs(reference), cell_numeric_absolute_tolerance)"
            ),
            "tolerance_source": (
                "Xiangru run_s0_tora_static_partition_sweep.py "
                "CONTROLLER_TOLERANCE = 1e-6"
            ),
            "excluded_from_correctness": [
                "cells.*.peak_cuda_memory_bytes",
                "cells.*.timing",
                "cells.*.*_seconds",
            ],
            "runtime_compared": False,
        },
        "exact_fields": exact_fields,
        **stats,
        "differences": differences,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
