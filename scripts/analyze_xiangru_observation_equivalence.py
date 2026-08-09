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

import numpy as np


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


def array_comparison(left: Any, right: Any) -> dict[str, Any]:
    a = np.asarray(left)
    b = np.asarray(right)
    if a.shape != b.shape:
        raise ValueError(f"exporter array shape mismatch: {a.shape} != {b.shape}")
    if a.dtype == np.bool_ or b.dtype == np.bool_:
        difference = a != b
        return {
            "element_count": int(a.size),
            "bitwise_equal_elements": int(np.count_nonzero(~difference)),
            "maximum_absolute_difference": 0.0 if not np.any(difference) else 1.0,
        }
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return {
        "element_count": int(a.size),
        "bitwise_equal_elements": int(np.count_nonzero(a == b)),
        "maximum_absolute_difference": float(np.max(np.abs(a - b), initial=0.0)),
    }


def merge_array_metrics(target: dict[str, Any], value: dict[str, Any]) -> None:
    target["element_count"] += value["element_count"]
    target["bitwise_equal_elements"] += value["bitwise_equal_elements"]
    target["maximum_absolute_difference"] = max(
        target["maximum_absolute_difference"],
        value["maximum_absolute_difference"],
    )


def empty_array_metrics() -> dict[str, Any]:
    return {
        "element_count": 0,
        "bitwise_equal_elements": 0,
        "maximum_absolute_difference": 0.0,
    }


def jsonl_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        first = json.loads(next(handle))
        rows = [json.loads(line) for line in handle]
    if first.get("schema", "").endswith("header_v1"):
        return rows
    return [first, *rows]


def exporter_regression(
    reference_plant: Path,
    instrumented_plant: Path,
    reference_controller: Path,
    instrumented_controller: Path,
) -> dict[str, Any]:
    left_rows = jsonl_rows(reference_plant)
    right_rows = jsonl_rows(instrumented_plant)
    if len(left_rows) != 200 or len(right_rows) != 200:
        raise ValueError("exporter regression requires two complete 200-segment traces")
    categories = {
        name: empty_array_metrics()
        for name in ("accepted_leaves", "endpoint", "tube", "remainder")
    }
    for expected_segment, (left_row, right_row) in enumerate(
        zip(left_rows, right_rows, strict=True), start=1
    ):
        if left_row["segment_index"] != expected_segment or right_row["segment_index"] != expected_segment:
            raise ValueError("exporter segment order mismatch")
        merge_array_metrics(
            categories["accepted_leaves"],
            array_comparison(left_row["accepted"], right_row["accepted"]),
        )
        for category, field in (
            ("endpoint", "endpoint"),
            ("tube", "tube"),
            ("remainder", "interval_remainder"),
        ):
            for side in ("lower", "upper"):
                merge_array_metrics(
                    categories[category],
                    array_comparison(
                        left_row[field][side], right_row[field][side]
                    ),
                )
    left_controller = json.loads(reference_controller.read_text(encoding="utf-8"))
    right_controller = json.loads(instrumented_controller.read_text(encoding="utf-8"))
    left_control_rows = left_controller["rows"]
    right_control_rows = right_controller["rows"]
    if len(left_control_rows) != 20 or len(right_control_rows) != 20:
        raise ValueError("exporter regression requires two 20-period controller traces")
    controller_metrics = empty_array_metrics()
    for left_row, right_row in zip(left_control_rows, right_control_rows, strict=True):
        for field in (
            "controller_output_interval_before_outward_composition",
            "controller_output_interval_after_outward_composition",
        ):
            for side in ("lower", "upper"):
                merge_array_metrics(
                    controller_metrics,
                    array_comparison(left_row[field][side], right_row[field][side]),
                )
    categories["controller_output"] = controller_metrics
    maximum = max(
        value["maximum_absolute_difference"] for value in categories.values()
    )
    return {
        "status": "PASS_WITH_DECLARED_TOLERANCE" if maximum <= 1e-12 else "FAIL",
        "scope": "prior validated exporter versus stage-contract exporter; this does not replace the unavailable uninstrumented raw-array gate",
        "segment_count": 200,
        "controller_period_count": 20,
        "categories": categories,
        "maximum_absolute_difference": maximum,
        "source_hashes": {
            "reference_plant": sha256(reference_plant),
            "instrumented_plant": sha256(instrumented_plant),
            "reference_controller": sha256(reference_controller),
            "instrumented_controller": sha256(instrumented_controller),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--instrumented", type=Path, required=True)
    parser.add_argument("--private-detail", type=Path, required=True)
    parser.add_argument("--public-summary", type=Path, required=True)
    parser.add_argument("--reference-plant", type=Path)
    parser.add_argument("--instrumented-plant", type=Path)
    parser.add_argument("--reference-controller", type=Path)
    parser.add_argument("--instrumented-controller", type=Path)
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
    lane_left = left_document["cells"]["b48_static"]["complete_q3"]
    lane_right = right_document["cells"]["b48_static"]["complete_q3"]
    public["uninstrumented_behavior_equivalence"] = {
        "status_equal": lane_left["status"] == lane_right["status"],
        "status": lane_right["status"],
        "certified_horizon_equal": lane_left["certified_horizon"] == lane_right["certified_horizon"],
        "certified_horizon": lane_right["certified_horizon"],
        "segment_count_equal": len(lane_left["segments"]) == len(lane_right["segments"]),
        "segment_count": len(lane_right["segments"]),
        "segments_attempted_equal": lane_left["segments_attempted"] == lane_right["segments_attempted"],
        "accepted_leaf_counts_equal": [row["accepted_leaves"] for row in lane_left["segments"]] == [row["accepted_leaves"] for row in lane_right["segments"]],
        "failed_leaf_indices_equal": [row["failed_leaf_indices"] for row in lane_left["segments"]] == [row["failed_leaf_indices"] for row in lane_right["segments"]],
        "first_failure_equal": lane_left["first_failure"] == lane_right["first_failure"],
        "controller_period_count_equal": len(lane_left["controller_periods"]) == len(lane_right["controller_periods"]),
        "aggregate_endpoint_tube_and_controller_fields_within_1e_6": within_tolerance == compared,
    }
    raw_arguments = (
        args.reference_plant,
        args.instrumented_plant,
        args.reference_controller,
        args.instrumented_controller,
    )
    if any(value is not None for value in raw_arguments):
        if not all(value is not None for value in raw_arguments):
            raise ValueError("all four exporter-regression paths must be supplied together")
        public["instrumented_exporter_regression"] = exporter_regression(
            args.reference_plant,
            args.instrumented_plant,
            args.reference_controller,
            args.instrumented_controller,
        )
        if public["instrumented_exporter_regression"]["status"] == "FAIL":
            public["status"] = "FAIL"
    args.public_summary.parent.mkdir(parents=True, exist_ok=True)
    args.public_summary.write_text(json.dumps(public, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(public, sort_keys=True))
    return 0 if public["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
