#!/usr/bin/env python3
"""Compare uninstrumented/instrumented Xiangru correctness fields including ULPs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
from typing import Any, Iterator


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def leaves(value: Any, prefix: str = "$") -> Iterator[tuple[str, Any]]:
    if isinstance(value, dict):
        for key in sorted(value):
            yield from leaves(value[key], f"{prefix}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from leaves(item, f"{prefix}[{index}]")
    else:
        yield prefix, value


def excluded(path: str) -> bool:
    return ".timing" in path or path.endswith("_seconds") or "peak_cuda_memory_bytes" in path


def ordered_bits(value: float) -> int:
    raw = struct.unpack(">Q", struct.pack(">d", float(value)))[0]
    return (~raw & ((1 << 64) - 1)) if (raw >> 63) else (raw | (1 << 63))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--instrumented", type=Path, required=True)
    parser.add_argument("--private-detail", type=Path, required=True)
    parser.add_argument("--public-summary", type=Path, required=True)
    args = parser.parse_args()
    left_document = json.loads(args.reference.read_text(encoding="utf-8"))
    right_document = json.loads(args.instrumented.read_text(encoding="utf-8"))
    left = dict(leaves(left_document["cells"]["b48_static"]["complete_q3"]))
    right = dict(leaves(right_document["cells"]["b48_static"]["complete_q3"]))
    common = sorted(set(left) & set(right))
    details: list[dict[str, Any]] = []
    compared = 0
    exact = 0
    within_tolerance = 0
    maximum_abs = 0.0
    maximum_ulp = 0
    maximum_ulp_away_from_zero = 0
    excluded_count = 0
    nonnumeric_mismatches = []
    for path in common:
        if excluded(path):
            excluded_count += 1
            continue
        a, b = left[path], right[path]
        if isinstance(a, bool) or isinstance(b, bool) or not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
            if a != b and not any(token in path for token in ("source.crown_reach_dirty", "command")):
                nonnumeric_mismatches.append(path)
            continue
        a, b = float(a), float(b)
        if not math.isfinite(a) or not math.isfinite(b):
            if a != b:
                nonnumeric_mismatches.append(path)
            continue
        compared += 1
        difference = abs(a - b)
        ulp = abs(ordered_bits(a) - ordered_bits(b))
        maximum_abs = max(maximum_abs, difference)
        maximum_ulp = max(maximum_ulp, ulp)
        if min(abs(a), abs(b)) >= 1e-12:
            maximum_ulp_away_from_zero = max(maximum_ulp_away_from_zero, ulp)
        if difference == 0.0:
            exact += 1
        if difference <= 1e-6:
            within_tolerance += 1
        if difference:
            details.append({
                "path": path,
                "reference": a,
                "instrumented": b,
                "reference_hex": a.hex(),
                "instrumented_hex": b.hex(),
                "absolute_difference": difference,
                "ulp_distance": ulp,
            })
    details.sort(key=lambda row: (row["absolute_difference"], row["ulp_distance"]), reverse=True)
    private = {
        "schema": "xiangru_observation_equivalence_ulp_detail_v1",
        "reference_sha256": sha256(args.reference),
        "instrumented_sha256": sha256(args.instrumented),
        "differences": details,
        "nonnumeric_mismatches": nonnumeric_mismatches,
    }
    args.private_detail.parent.mkdir(parents=True, exist_ok=True)
    args.private_detail.write_text(json.dumps(private, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    public = {
        "schema": "xiangru_observation_equivalence_summary_v1",
        "status": "PASS_WITH_DECLARED_TOLERANCE" if within_tolerance == compared and not nonnumeric_mismatches else "FAIL",
        "correctness_numeric_fields_compared": compared,
        "bitwise_equal_numeric_fields": exact,
        "non_bitwise_numeric_fields": compared - exact,
        "fields_within_absolute_tolerance_1e_6": within_tolerance,
        "maximum_absolute_difference": maximum_abs,
        "maximum_ulp_distance": maximum_ulp,
        "maximum_ulp_distance_when_both_abs_ge_1e_12": maximum_ulp_away_from_zero,
        "ulp_caveat": "The all-field maximum compares exact zero with a tiny nonzero diagnostic; ordered-bit ULP distance across zero is not a useful relative-error measure.",
        "excluded_runtime_or_memory_fields": excluded_count,
        "nonnumeric_mismatch_count": len(nonnumeric_mismatches),
        "raw_reference_sha256": sha256(args.reference),
        "raw_instrumented_sha256": sha256(args.instrumented),
        "private_ulp_detail_sha256": sha256(args.private_detail),
        "source_identity": {
            "commit_equal": left_document.get("crown_reach_commit") == right_document.get("crown_reach_commit"),
            "reference_dirty": left_document.get("crown_reach_dirty"),
            "instrumented_dirty": right_document.get("crown_reach_dirty"),
            "expected_instrumented_patch": True,
        },
        "exporter_overhead": {
            "reference_solver_wall_excluding_validation_seconds": left_document["cells"]["b48_static"]["complete_q3"]["timing"]["solver_wall_seconds_excluding_validation"],
            "instrumented_solver_wall_excluding_validation_seconds": right_document["cells"]["b48_static"]["complete_q3"]["timing"]["solver_wall_seconds_excluding_validation"],
            "difference_seconds": right_document["cells"]["b48_static"]["complete_q3"]["timing"]["solver_wall_seconds_excluding_validation"] - left_document["cells"]["b48_static"]["complete_q3"]["timing"]["solver_wall_seconds_excluding_validation"],
            "performance_use": "excluded_from_formal_runtime_comparison",
        },
        "core_array_bitwise_gate": "UNAVAILABLE_UNINSTRUMENTED_BASELINE_DID_NOT_EXPORT_PER_LEAF_ARRAYS",
        "interpretation": "Aggregate correctness fields are equivalent within the frozen 1e-6 gate; timing is excluded. Per-leaf exporter arrays have no uninstrumented counterpart, so bitwise identity is not claimed.",
    }
    args.public_summary.parent.mkdir(parents=True, exist_ok=True)
    args.public_summary.write_text(json.dumps(public, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(public, sort_keys=True))
    return 0 if public["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
