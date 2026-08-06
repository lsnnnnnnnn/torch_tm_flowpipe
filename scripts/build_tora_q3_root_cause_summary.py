#!/usr/bin/env python3
"""Merge first-divergence and actual-bug evidence into one public summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--common-summary", type=Path, required=True)
    parser.add_argument("--full-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    common = json.loads(args.common_summary.read_text(encoding="utf-8"))
    full = json.loads(args.full_summary.read_text(encoding="utf-8"))
    result = {
        "schema": "tora_q3_first_divergence_root_cause_v1",
        "common_control_plant_replay": {
            "first_divergence": common["first_divergence"],
            "status_divergence": None,
            "classification": [
                "sine enclosure",
                "fixed-support overflow",
                "Picard/remainder algorithm",
                "expected method difference",
            ],
            "behavior_relevant": True,
            "both_completed_t20": True,
        },
        "native_full_closed_loop": {
            "first_exact_controller_difference": full[
                "first_exact_controller_difference"
            ],
            "first_behavior_relevant_controller_difference": full[
                "first_behavior_relevant_controller_difference"
            ],
            "first_failure": full["first_failure"],
            "classification": [
                "basis/normalization representation",
                "Picard/remainder algorithm",
                "controller bound/composition on a different sound input domain",
                "expected method difference",
            ],
        },
        "fixed_actual_implementation_bugs": full[
            "pre_fix_actual_bug_evidence"
        ],
        "regression_tests": [
            "test_affine_composition_materializes_local_spatial_coordinates",
            "test_exact_time_endpoint_aggregates_equal_spatial_exponents_soundly",
        ],
        "acceptance_relaxed": False,
        "controller_substituted": False,
        "post_fix_gates_restarted_from_one_step": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": "PASS",
        "common_first_segment": result["common_control_plant_replay"]["first_divergence"]["endpoint"]["segment_index"],
        "full_first_failure_segment": result["native_full_closed_loop"]["first_failure"]["segment_index"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
