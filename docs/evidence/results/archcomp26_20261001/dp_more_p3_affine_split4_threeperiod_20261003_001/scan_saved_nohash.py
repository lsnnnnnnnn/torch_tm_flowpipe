#!/usr/bin/env python3
"""Read the saved DP more three-period diagnostic without rerunning its solver."""

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "attempt/data"


def read_json(path):
    return json.loads(path.read_text())


def main():
    frozen = (HERE / "source/double_pendulum_more_robust_paper_p4.yaml").read_text()
    actual = (DATA / "config.yaml").read_text()
    uncommented = "\n".join(line for line in frozen.splitlines()
                            if line.strip() and not line.lstrip().startswith("#")) + "\n"
    assert uncommented.count("steps: 20\n") == 1
    assert actual == uncommented.replace("steps: 20\n", "steps: 3\n")
    assert "step_size: 0.02\n" in actual and "ode_step_size: 0.005\n" in actual
    assert actual.count("  splits: 5\n") == 2 and actual.count("  splits: 3\n") == 2
    model_line = next(line for line in actual.splitlines() if line.startswith("model_dir: "))
    model_path = model_line.removeprefix("model_dir: ")
    assert model_path.endswith("/controller_double_pendulum_more_robust.onnx")

    outer = read_json(HERE / "attempt/RESULT.json")
    inner = read_json(DATA / "RESULT.json")
    start = read_json(DATA / "START.json")
    metrics = read_json(DATA / "metrics.json")
    assert outer["status"] == "completed" and outer["exit_code"] == 0 and not outer["timed_out"]
    assert start["contract"] == "more" and start["mode"] == "three-period"
    assert start["controller_envelope"] == "affine-split4" and start["physical_gpu"] == 3
    assert start["controller"] == model_path
    assert inner["status"] == "completed" and inner["driver_return"] == 0
    assert inner["expected_substeps"] == inner["completed_substeps"] == 12
    assert inner["accepted_lane_substeps"] == 2700 and inner["all_lanes_accepted"]
    assert metrics["B"] == 225 and metrics["steps"] == 3
    assert metrics["substeps"] == 4 and metrics["order"] == 3 and metrics["broken"] == 0

    per_step = []
    smallest_margin = None
    smallest_location = None
    first_tube_union = None
    with (DATA / "observations.jsonl").open() as observations:
        for step, line in enumerate(observations, 1):
            row = json.loads(line)
            fields = ("accepted", "tube_endpoint_valid", "status", "whole_tube_inside_safe_box",
                      "tube", "endpoint")
            assert row["substep"] == step and all(len(row[key]) == 225 for key in fields)
            assert row["accepted"] == row["tube_endpoint_valid"]
            accepted = inside = 0
            tube_union = [[math.inf, -math.inf] for _ in range(4)]
            for lane in range(225):
                assert row["accepted"][lane] and row["status"][lane] == 0
                assert len(row["tube"][lane]) == len(row["endpoint"][lane]) == 4
                accepted += 1
                lane_inside = True
                for variable, (tube, endpoint) in enumerate(zip(row["tube"][lane], row["endpoint"][lane])):
                    assert len(tube) == len(endpoint) == 2
                    assert all(math.isfinite(value) for value in (*tube, *endpoint))
                    assert tube[0] <= endpoint[0] <= endpoint[1] <= tube[1]
                    tube_union[variable][0] = min(tube_union[variable][0], tube[0])
                    tube_union[variable][1] = max(tube_union[variable][1], tube[1])
                    lane_inside &= tube[0] >= -1.5 and tube[1] <= 1.5
                    for side, margin in (("lower", tube[0] + 1.5), ("upper", 1.5 - tube[1])):
                        if smallest_margin is None or margin < smallest_margin:
                            smallest_margin = margin
                            smallest_location = [step, lane, variable, side, *tube]
                assert row["whole_tube_inside_safe_box"][lane] == lane_inside
                inside += lane_inside
            if step == 1:
                first_tube_union = tube_union
                assert all(lo <= 1.0 and hi >= 1.3 for lo, hi in tube_union)
            per_step.append({"step": step, "accepted": accepted, "inside_safe_box": inside})
    assert len(per_step) == 12 and all(item == {"step": i, "accepted": 225, "inside_safe_box": 225}
                                           for i, item in enumerate(per_step, 1))
    assert inner["per_substep_accepted"] == [225] * 12
    assert inner["per_substep_inside_safe_box"] == [225] * 12
    assert inner["whole_tube_inside_safe_box"]

    audits = inner["controller_audit"]
    assert len(audits) == 3
    for period, audit in enumerate(audits):
        assert audit["period"] == period and audit["envelope"] == "affine-split4"
        detail = audit["affine_layer_diagnostics"]
        assert detail["subboxes_per_input_box"] == 256 and detail["global_T_reused"]
        assert math.isfinite(audit["directed_residual_width_max"])
        assert 0 <= audit["directed_residual_width_max"] == detail["tight_width_max"]

    stdout = (HERE / "attempt/stdout.log").read_text()
    assert [line for line in stdout.splitlines() if line.startswith("Step ")] == ["Step 0", "Step 1", "Step 2"]
    assert all(label not in stdout for label in ("Unsafe.", "Unknown.", "VERIFIED"))
    result = {
        "schema": "dp-more-p3-affine-split4-three-period-saved-scan-v1",
        "server_original": "/srv/local/shengenli/flowstar_acceleration_20260921T153643Z/runs/archcomp26_20261001/dp_more_p3_affine_split4_threeperiod_20261003_001",
        "config_diff_from_frozen_full_contract": {"steps": [20, 3]},
        "configured_boxes": 225,
        "completed_periods": 3,
        "completed_substeps": 12,
        "accepted_lane_substeps": 2700,
        "saved_tubes_inside_safe_box": 2700,
        "first_tube_union": first_tube_union,
        "minimum_saved_tube_safety_margin": smallest_margin,
        "minimum_margin_location": smallest_location,
        "per_step": per_step,
        "controller_residual_width_max_by_period": [audit["directed_residual_width_max"] for audit in audits],
        "controller_crown_bias_inversions_by_period": [audit["crown_bias_inversion_count"] for audit in audits],
        "author_final_property_label": None,
        "numeric_prefix_end_s": 0.06,
        "full_0p4_horizon_complete": False,
        "end_to_end_strict_certificate": inner["end_to_end_strict_certificate"],
        "outer_wall_s": outer["wall_s"],
    }
    (HERE / "INDEPENDENT_SAVED_INTERVAL_SCAN.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"substeps": 12, "accepted": 2700, "safe_saved_tubes": 2700,
                      "minimum_margin": smallest_margin, "widths": result["controller_residual_width_max_by_period"]}))


if __name__ == "__main__":
    main()
