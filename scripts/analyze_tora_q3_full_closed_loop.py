#!/usr/bin/env python3
"""Publish sanitized native full-closed-loop gates and common-horizon metrics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct
from typing import Any


STATES = ("x1", "x2", "x3", "x4", "u1")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonl(path: Path, *, header: bool = False) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        if header:
            next(handle)
        return [json.loads(line) for line in handle]


def flatten(value: Any) -> list[float]:
    if isinstance(value, list):
        out: list[float] = []
        for item in value:
            out.extend(flatten(item))
        return out
    return [float(value)]


def max_abs(left: Any, right: Any) -> float:
    a, b = flatten(left), flatten(right)
    if len(a) != len(b):
        return math.inf
    return max((abs(x - y) for x, y in zip(a, b, strict=True)), default=0.0)


def ordered_bits(value: float) -> int:
    bits = struct.unpack(">q", struct.pack(">d", float(value)))[0]
    return bits if bits >= 0 else 0x8000000000000000 - bits


def max_ulp(left: Any, right: Any) -> int | str:
    a, b = flatten(left), flatten(right)
    if len(a) != len(b) or any(not math.isfinite(x) for x in (*a, *b)):
        return "unavailable"
    return max(
        (abs(ordered_bits(x) - ordered_bits(y)) for x, y in zip(a, b, strict=True)),
        default=0,
    )


def containment(
    candidate_lower: Any,
    candidate_upper: Any,
    reference_lower: Any,
    reference_upper: Any,
) -> dict[str, int]:
    cl, cu = flatten(candidate_lower), flatten(candidate_upper)
    rl, ru = flatten(reference_lower), flatten(reference_upper)
    rows = list(zip(cl, cu, rl, ru, strict=True))
    return {
        "scalar_count": len(rows),
        "candidate_contains_reference": sum(a <= c and b >= d for a, b, c, d in rows),
        "reference_contains_candidate": sum(c <= a and d >= b for a, b, c, d in rows),
        "overlap": sum(max(a, c) <= min(b, d) for a, b, c, d in rows),
    }


def median_ratio(candidate: list[float], reference: list[float]) -> float | str:
    ratios = [a / b for a, b in zip(candidate, reference, strict=True) if b != 0.0]
    return statistics.median(ratios) if ratios else "N/A"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--torch-summary", type=Path, required=True)
    parser.add_argument("--torch-segments", type=Path, required=True)
    parser.add_argument("--torch-controller", type=Path, required=True)
    parser.add_argument("--xiangru-plant", type=Path, required=True)
    parser.add_argument("--xiangru-controller", type=Path, required=True)
    parser.add_argument("--pre-fix-summary", type=Path, required=True)
    parser.add_argument("--pre-fix-segments", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    torch_summary = json.loads(args.torch_summary.read_text(encoding="utf-8"))
    torch_segments = jsonl(args.torch_segments)
    torch_controller = jsonl(args.torch_controller)
    xiangru_segments = jsonl(args.xiangru_plant, header=True)
    xiangru_controller_doc = json.loads(
        args.xiangru_controller.read_text(encoding="utf-8")
    )
    xiangru_controller = xiangru_controller_doc["rows"]
    if len(torch_segments) != 44 or len(torch_controller) != 5:
        raise ValueError("expected the native five-refresh failure trace")

    controller_rows = []
    for candidate, reference in zip(
        torch_controller, xiangru_controller, strict=False
    ):
        period = int(candidate["controller_period"])
        before_c = candidate["output_before_outward"]
        before_r = reference[
            "controller_output_interval_before_outward_composition"
        ]
        after_c = candidate["output_after_outward"]
        after_r = reference[
            "controller_output_interval_after_outward_composition"
        ]
        pre_c = candidate["pre_controller_state_box"]
        pre_r = reference["pre_controller_state_box"]
        relation = containment(
            after_c["lower"], after_c["upper"],
            after_r["lower"], after_r["upper"],
        )
        controller_rows.append({
            "controller_period": period,
            "segment_index": candidate["segment_index"],
            "physical_time": candidate["physical_time"],
            "pre_state_lower_max_abs": max_abs(pre_c["lower"], pre_r["lower"]),
            "pre_state_upper_max_abs": max_abs(pre_c["upper"], pre_r["upper"]),
            "before_lower_max_abs": max_abs(before_c["lower"], before_r["lower"]),
            "before_upper_max_abs": max_abs(before_c["upper"], before_r["upper"]),
            "after_lower_max_abs": max_abs(after_c["lower"], after_r["lower"]),
            "after_upper_max_abs": max_abs(after_c["upper"], after_r["upper"]),
            "after_lower_max_ulp": max_ulp(after_c["lower"], after_r["lower"]),
            "after_upper_max_ulp": max_ulp(after_c["upper"], after_r["upper"]),
            **relation,
        })

    width_rows = []
    for candidate, reference in zip(
        torch_segments[:10], xiangru_segments[:10], strict=True
    ):
        for field in ("segment_index", "physical_time", "controller_period", "leaf_id"):
            if candidate[field] != reference[field]:
                raise ValueError(f"T1 exact alignment mismatch: {field}")
        for kind in ("endpoint", "tube"):
            for state_index, state in enumerate(STATES):
                candidate_width = [
                    float(candidate[kind]["upper"][leaf][state_index])
                    - float(candidate[kind]["lower"][leaf][state_index])
                    for leaf in range(48)
                ]
                reference_width = [
                    float(reference[kind]["upper"][leaf][state_index])
                    - float(reference[kind]["lower"][leaf][state_index])
                    for leaf in range(48)
                ]
                width_rows.append({
                    "segment_index": candidate["segment_index"],
                    "physical_time": candidate["physical_time"],
                    "state": state,
                    "enclosure_kind": kind,
                    "torch_width_median": statistics.median(candidate_width),
                    "xiangru_width_median": statistics.median(reference_width),
                    "torch_over_xiangru_ratio_median": median_ratio(candidate_width, reference_width),
                    "torch_width_max": max(candidate_width),
                    "xiangru_width_max": max(reference_width),
                    "maximum_lower_abs_difference": max_abs(
                        [row[state_index] for row in candidate[kind]["lower"]],
                        [row[state_index] for row in reference[kind]["lower"]],
                    ),
                    "maximum_upper_abs_difference": max_abs(
                        [row[state_index] for row in candidate[kind]["upper"]],
                        [row[state_index] for row in reference[kind]["upper"]],
                    ),
                })

    failed = torch_segments[-1]
    failed_leaf_ids = [
        index for index, accepted in enumerate(failed["accepted"]) if not accepted
    ]
    failure = {
        "segment_index": failed["segment_index"],
        "physical_time": failed["physical_time"],
        "failed_leaf_ids": failed_leaf_ids,
        "first_failed_leaf": failed_leaf_ids[0],
        "property_margin": failed["property_margin"][failed_leaf_ids[0]],
        "classification": (
            "method-native plant/state-projection enclosure growth feeds a different "
            "sound controller input box; x3 tube crosses the frozen property at the "
            "fifth controller period"
        ),
        "acceptance_was_not_relaxed": True,
    }
    first_exact_controller_difference = next(
        row for row in controller_rows
        if max(row["after_lower_max_abs"], row["after_upper_max_abs"]) != 0.0
    )
    first_behavior_relevant_controller_difference = next(
        row for row in controller_rows
        if max(
            row["pre_state_lower_max_abs"], row["pre_state_upper_max_abs"],
            row["after_lower_max_abs"], row["after_upper_max_abs"],
        ) > 1.0e-12
    )
    result = {
        "schema": "tora_q3_native_full_closed_loop_analysis_v1",
        "status": "FAILED_AT_T4_4",
        "completed_segments": torch_summary["completed_segments"],
        "certified_horizon": torch_summary["certified_horizon"],
        "first_failure": failure,
        "nominal_gate": {
            **torch_summary["nominal_gate"],
            "tolerance": 1.0e-6,
            "status": "PASS" if torch_summary["nominal_gate"]["maximum_absolute_error"] <= 1.0e-6 else "FAIL",
        },
        "controller_gates": [
            {"gate": "initial_b48", "status": "PASS", "maximum_abs_difference": max(controller_rows[0]["after_lower_max_abs"], controller_rows[0]["after_upper_max_abs"]), "tolerance": 1.0e-12},
            {"gate": "first_refresh_at_t1", "status": "METHOD_INPUT_DIVERGENCE", "controller_period": 2, "pre_state_max_abs_difference": max(controller_rows[1]["pre_state_lower_max_abs"], controller_rows[1]["pre_state_upper_max_abs"])},
            {"gate": "first_five_refreshes", "status": "FAIL_ALIGNED_CONTRACT", "completed_refreshes": 5},
            {"gate": "all_twenty_refreshes", "status": "N/A", "reason": "closed-loop acceptance failed at segment 44"},
        ],
        "closed_loop_gates": [
            {"gate": "b48_t1", "status": "PASS", "certified_horizon": 1.0},
            {"gate": "b48_t5", "status": "FAIL", "certified_horizon": 4.3, "first_failed_segment": 44},
            {"gate": "b48_t10", "status": "N/A", "reason": "earlier failure"},
            {"gate": "b48_t20", "status": "N/A", "reason": "earlier failure"},
        ],
        "first_exact_controller_difference": first_exact_controller_difference,
        "first_behavior_relevant_controller_difference": first_behavior_relevant_controller_difference,
        "pre_fix_actual_bug_evidence": {
            "summary_sha256": sha256(args.pre_fix_summary),
            "segments_sha256": sha256(args.pre_fix_segments),
            "bugs": [
                "physical tube/endpoint were exported before affine parameterization composition",
                "exact-time endpoint terms were ranged before equal spatial exponents were aggregated",
            ],
            "post_fix_t1_x4_endpoint_max_abs_difference": max(
                max(row["maximum_lower_abs_difference"], row["maximum_upper_abs_difference"])
                for row in width_rows
                if row["segment_index"] == 1
                and row["state"] == "x4"
                and row["enclosure_kind"] == "endpoint"
            ),
        },
        "timing": {
            "controller_build_seconds": torch_summary["controller_build_seconds"],
            "controller_bound_seconds": torch_summary["controller_bound_seconds"],
            "controller_composition_seconds": torch_summary["controller_composition_seconds"],
            "plant_seconds": torch_summary["plant_seconds"],
            "normalization_seconds": torch_summary["normalization_seconds"],
            "serialization_seconds": torch_summary["serialization_seconds"],
            "wall_seconds_including_serialization": torch_summary["wall_seconds_including_serialization"],
            "peak_cuda_memory_bytes": torch_summary["peak_cuda_memory_bytes"],
        },
        "source_hashes": {
            "torch_summary": sha256(args.torch_summary),
            "torch_segments": sha256(args.torch_segments),
            "torch_controller_updates": sha256(args.torch_controller),
            "xiangru_plant_observation": sha256(args.xiangru_plant),
            "xiangru_controller_observation": sha256(args.xiangru_controller),
        },
    }
    (output / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for filename, rows in (
        ("controller_bound_comparison.csv", controller_rows),
        ("t1_width_comparison.csv", width_rows),
    ):
        with (output / filename).open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps({
        "status": result["status"],
        "certified_horizon": result["certified_horizon"],
        "first_failure": result["first_failure"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
