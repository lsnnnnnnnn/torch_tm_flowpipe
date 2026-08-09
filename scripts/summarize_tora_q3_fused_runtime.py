#!/usr/bin/env python3
"""Publish the final public runtime/tightness comparison from public inputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def build(output_root: Path) -> dict[Path, object]:
    fused = load(output_root / "fused_kernel/summary.json")
    resource = load(output_root / "fused_kernel/resource_recheck.json")
    aligned = load(output_root / "algorithm_aligned/common_control_t20.json")
    one_step = load(output_root / "algorithm_aligned/one_step_gates.json")
    root_cause = load(output_root / "stage_parity/root_cause.json")
    hierarchy = load(output_root / "native_full_loop/hierarchical_gates.json")

    if fused["status"] != "PASS" or aligned["status"] != "PASS":
        raise ValueError("fused and algorithm-aligned public inputs must pass")
    if fused["common_control_t20"]["runtime"]["repeat_count"] != 5:
        raise ValueError("formal T20 record must contain five measured repeats")
    if not fused["common_control_t20"]["checksum_stable"]:
        raise ValueError("formal T20 output checksum is unstable")
    expected_native_statuses = ["PASS", "PASS", "PASS", "FAIL", "NOT_RUN", "NOT_RUN"]
    for lane in hierarchy["implementations"].values():
        if [gate["status"] for gate in lane["gates"]] != expected_native_statuses:
            raise ValueError("native hierarchy does not fail closed at T5")

    fused_t20 = float(fused["common_control_t20"]["runtime"]["median_seconds"])
    frozen_t20 = float(fused["baseline_t20_seconds"])
    prior_t20 = 105.48005206231028
    xiangru_t20 = 1.20676
    common = {
        "schema": "tora_q3_common_control_final_summary_v1",
        "status": "PASS",
        "workload": {
            "batch": 48,
            "segments": 200,
            "horizon": 20.0,
            "controller_time_included": False,
            "period_local_frozen_input_restart": True,
            "independent_native_closed_loop": False,
        },
        "formal_runtime_lanes_seconds": {
            "torch_frozen_baseline": frozen_t20,
            "torch_prior_optimized": prior_t20,
            "torch_fused": fused_t20,
            "xiangru_matched_stack": xiangru_t20,
        },
        "runtime_protocol": {
            "excluded_complete_warmups": 1,
            "measured_repeats": 5,
            "cuda_synchronized_timing_boundaries": True,
            "checksum_stable": True,
        },
        "allowed_ratios": {
            "fused_internal_speedup_over_frozen_torch": frozen_t20 / fused_t20,
            "fused_internal_speedup_over_prior_torch": prior_t20 / fused_t20,
            "fused_over_xiangru_descriptive_ratio": fused_t20 / xiangru_t20,
        },
        "tightness": {
            "endpoint": aligned["endpoint"],
            "tube": aligned["tube"],
            "remainder": aligned["remainder"],
            "minimum_property_margin": aligned["minimum_property_margin"],
            "zero_width_denominator_policy": "N/A; excluded from width ratios",
        },
    }

    comparison = {
        "schema": "tora_q3_final_runtime_tightness_comparison_v1",
        "status": "CASE_C_PERFORMANCE_PASS_NATIVE_T5_GATE_FAIL",
        "common_control": common,
        "native_closed_loop": {
            "torch_target_widths": hierarchy["torch_target_width_availability"],
            "torch_best_certified_horizon": max(
                float(lane["certified_horizon"])
                for lane in hierarchy["implementations"].values()
            ),
            "xiangru_certified_horizon": hierarchy["xiangru_native_reference"][
                "certified_horizon"
            ],
            "formal_cross_implementation_t20_runtime_ratio": None,
            "reason": "Torch native lanes fail the T5 gate, so no common native T20 workload exists",
        },
        "resources": {
            "maximum_process_rss_bytes": resource["maximum_process_rss_bytes"],
            "peak_cuda_memory_bytes": resource["peak_cuda_memory_bytes"],
        },
    }

    stage_rows = []
    for row in root_cause["root_cause_table"]:
        stage_rows.append(
            {
                "stage": row["stage"],
                "classification": row["classification"],
                "input_contract_equal": row["input_contract_equal"],
                "coordinate_map_status": row["coordinate_map_status"],
                "first_segment": row["first_segment"],
                "first_leaf": row["first_leaf"],
                "center_difference_maximum_absolute": row["center_diff"],
                "width_difference_maximum_absolute": row["width_diff"],
                "remainder_contribution_difference": row[
                    "remainder_contribution_diff"
                ],
                "containment_relation": row["containment_relation"],
                "maximum_ulp_difference": row["max_ulp_diff"],
            }
        )

    replay_rows = []
    selected = {row["gate"]: row for row in one_step["gates"]}
    for replay, gate in (("R1", "G3"), ("R2", "G4")):
        row = selected[gate]
        for quantity in ("interval_remainder", "endpoint", "tube"):
            metrics = row[quantity]
            width_difference = float(metrics["width_difference_maximum_absolute"])
            replay_rows.append(
                {
                    "replay": replay,
                    "gate": gate,
                    "segment": row["segment_index"],
                    "quantity": quantity,
                    "center_difference_maximum_absolute": metrics[
                        "center_difference_maximum_absolute"
                    ],
                    "radius_difference_maximum_absolute_upper_bound": width_difference
                    / 2.0,
                    "width_difference_maximum_absolute": width_difference,
                    "candidate_contains_reference_coordinates": metrics[
                        "candidate_contains_reference_coordinates"
                    ],
                    "reference_contains_candidate_coordinates": metrics[
                        "reference_contains_candidate_coordinates"
                    ],
                }
            )

    return {
        output_root / "common_control/summary.json": common,
        output_root / "comparison/summary.json": comparison,
        output_root / "stage_parity/stage_first_divergence.csv": stage_rows,
        output_root
        / "stage_parity/r1_r2_center_radius_remainder.csv": replay_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/tora_q3_stage_parity_fused_20260809"),
    )
    args = parser.parse_args()
    generated = build(args.output_root)
    for path, payload in generated.items():
        if path.suffix == ".json":
            write_json(path, payload)  # type: ignore[arg-type]
        else:
            write_csv(path, payload)  # type: ignore[arg-type]
    print(json.dumps({"status": "PASS", "generated": len(generated)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
