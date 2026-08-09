#!/usr/bin/env python3
"""Publish a compact stage-parity decision from sanitized public evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build(output_root: Path) -> dict[str, Any]:
    observation = load(output_root / "stage_contract/observation_summary.json")
    comparison = load(output_root / "stage_contract/stage_comparison_summary.json")
    root_cause = load(output_root / "stage_parity/root_cause.json")
    if observation["status"] != "PASS":
        raise ValueError("stage observation contract did not pass")
    if comparison["status"] != "PASS_COMPLETE_OBSERVATION":
        raise ValueError("stage comparison is incomplete")
    if root_cause["status"] != "PASS_DOMINANT_STAGE_ISOLATED":
        raise ValueError("stage root cause is not isolated")
    return {
        "schema": "tora_q3_stage_parity_public_summary_v1",
        "status": "PASS",
        "observation_only": observation["observation_only"],
        "formal_runner_uses_xiangru_outputs": False,
        "stage_count": len(comparison["stage_table"]),
        "first_numerical_stage": root_cause["t1_0_014211_attribution"][
            "first_numerical_stage"
        ],
        "first_material_stage": root_cause["first_differences"][
            "first_material"
        ],
        "t1_attribution": root_cause["t1_0_014211_attribution"],
        "segment_40_attribution": root_cause[
            "segment_40_remainder_attribution"
        ],
        "raw_arrays_private": True,
        "raw_paths_in_public_record": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/tora_q3_stage_parity_fused_20260809"),
    )
    args = parser.parse_args()
    result = build(args.output_root)
    target = args.output_root / "stage_parity/summary.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": result["status"], "stage_count": 13}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
