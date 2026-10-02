#!/usr/bin/env python3
"""Independently read the saved DP more P3 full-request attempt."""

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
    assert actual == uncommented and "steps: 20\n" in actual
    assert "step_size: 0.02\n" in actual and "ode_step_size: 0.005\n" in actual
    assert actual.count("  splits: 5\n") == 2 and actual.count("  splits: 3\n") == 2
    model_path = next(line.removeprefix("model_dir: ") for line in actual.splitlines()
                      if line.startswith("model_dir: "))
    assert model_path.endswith("/controller_double_pendulum_more_robust.onnx")

    outer = read_json(HERE / "attempt/RESULT.json")
    inner = read_json(DATA / "RESULT.json")
    start = read_json(DATA / "START.json")
    metrics = read_json(DATA / "metrics.json")
    assert outer["status"] == "completed" and outer["exit_code"] == 0 and not outer["timed_out"]
    assert start["contract"] == "more" and start["mode"] == "full"
    assert start["controller"] == model_path and start["controller_envelope"] == "affine-split4"
    assert start["physical_gpu"] == 3
    assert inner["status"] == "incomplete" and inner["driver_return"] == 0
    assert inner["expected_substeps"] == 80 and inner["completed_substeps"] == 72
    assert inner["accepted_lane_substeps"] == 225 * 72
    assert metrics["B"] == 225 and metrics["steps"] == 20
    assert metrics["substeps"] == 4 and metrics["order"] == 3 and metrics["broken"] == 0

    per_step = []
    first_crossing = first_entirely_outside = None
    first_tube_union = terminal_endpoint_union = None
    minimum_safe_prefix_margin = None
    with (DATA / "observations.jsonl").open() as observations:
        for step, line in enumerate(observations, 1):
            row = json.loads(line)
            fields = ("accepted", "tube_endpoint_valid", "status", "whole_tube_inside_safe_box",
                      "tube", "endpoint")
            assert row["substep"] == step and all(len(row[key]) == 225 for key in fields)
            assert row["accepted"] == row["tube_endpoint_valid"]
            inside = crossing_lanes = entirely_outside_lanes = 0
            tube_union = [[math.inf, -math.inf] for _ in range(4)]
            endpoint_union = [[math.inf, -math.inf] for _ in range(4)]
            for lane in range(225):
                assert row["accepted"][lane] and row["status"][lane] == 0
                assert len(row["tube"][lane]) == len(row["endpoint"][lane]) == 4
                lane_inside = True
                lane_crossing = lane_entirely_outside = False
                for variable, (tube, endpoint) in enumerate(zip(row["tube"][lane], row["endpoint"][lane])):
                    assert len(tube) == len(endpoint) == 2
                    assert all(math.isfinite(value) for value in (*tube, *endpoint))
                    assert tube[0] <= endpoint[0] <= endpoint[1] <= tube[1]
                    tube_union[variable][0] = min(tube_union[variable][0], tube[0])
                    tube_union[variable][1] = max(tube_union[variable][1], tube[1])
                    endpoint_union[variable][0] = min(endpoint_union[variable][0], endpoint[0])
                    endpoint_union[variable][1] = max(endpoint_union[variable][1], endpoint[1])
                    if step <= 60:
                        margin = min(tube[0] + 1.5, 1.5 - tube[1])
                        minimum_safe_prefix_margin = (margin if minimum_safe_prefix_margin is None
                                                      else min(minimum_safe_prefix_margin, margin))
                    if tube[0] < -1.5 or tube[1] > 1.5:
                        lane_inside = False
                        lane_crossing = True
                        if first_crossing is None:
                            first_crossing = {"step": step, "lane": lane, "state": variable, "tube": tube}
                    if tube[1] < -1.5 or tube[0] > 1.5:
                        lane_entirely_outside = True
                        if first_entirely_outside is None:
                            first_entirely_outside = {"step": step, "lane": lane,
                                                      "state": variable, "tube": tube}
                assert row["whole_tube_inside_safe_box"][lane] == lane_inside
                inside += lane_inside
                crossing_lanes += lane_crossing
                entirely_outside_lanes += lane_entirely_outside
            if step == 1:
                first_tube_union = tube_union
                assert all(lo <= 1.0 and hi >= 1.3 for lo, hi in first_tube_union)
            terminal_endpoint_union = endpoint_union
            per_step.append({"step": step, "accepted": 225, "inside_safe_box": inside,
                             "crossing_lanes": crossing_lanes,
                             "entirely_outside_lanes": entirely_outside_lanes})
    assert len(per_step) == 72
    assert inner["per_substep_accepted"] == [225] * 72
    assert inner["per_substep_inside_safe_box"] == [row["inside_safe_box"] for row in per_step]
    assert all(row["inside_safe_box"] == 225 for row in per_step[:60])
    assert first_crossing["step"] == 61 and first_entirely_outside["step"] == 72

    audits = inner["controller_audit"]
    assert len(audits) == 18
    for period, audit in enumerate(audits):
        assert audit["period"] == period and audit["envelope"] == "affine-split4"
        detail = audit["affine_layer_diagnostics"]
        assert detail["subboxes_per_input_box"] == 256 and detail["global_T_reused"]
        assert math.isfinite(audit["directed_residual_width_max"])
        assert 0 <= audit["directed_residual_width_max"] == detail["tight_width_max"]

    stdout = (HERE / "attempt/stdout.log").read_text()
    assert [line for line in stdout.splitlines() if line.startswith("Step ")] == [f"Step {i}" for i in range(18)]
    assert stdout.count("Unsafe.") == 1 and "VERIFIED" not in stdout and "Unknown." not in stdout
    result = {
        "schema": "dp-more-p3-affine-split4-full-request-saved-scan-v1",
        "server_original": "/srv/local/shengenli/flowstar_acceleration_20260921T153643Z/runs/archcomp26_20261001/dp_more_p3_affine_split4_full20_20261003_001",
        "configured_boxes": 225,
        "requested_periods": 20,
        "observed_periods": 18,
        "requested_substeps": 80,
        "observed_substeps": 72,
        "accepted_lane_substeps": 225 * 72,
        "all_box_saved_safe_prefix_substeps": 60,
        "all_box_saved_safe_prefix_end_s": 0.3,
        "first_saved_band_crossing": first_crossing,
        "first_saved_tube_entirely_outside_band": first_entirely_outside,
        "minimum_safe_prefix_margin": minimum_safe_prefix_margin,
        "first_tube_union": first_tube_union,
        "last_observed_endpoint_union_at_0p36_s": terminal_endpoint_union,
        "per_step": per_step,
        "controller_residual_width_max_by_period": [audit["directed_residual_width_max"] for audit in audits],
        "author_checker_label": "Unsafe.",
        "full_0p4_horizon_complete": False,
        "independent_trajectory_counterexample": False,
        "end_to_end_strict_certificate": inner["end_to_end_strict_certificate"],
        "outer_process_status": outer["status"],
        "outer_wall_s": outer["wall_s"],
    }
    (HERE / "INDEPENDENT_SAVED_INTERVAL_SCAN.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"observed_substeps": 72, "accepted": 225 * 72,
                      "all_box_safe_prefix": 60, "first_crossing": first_crossing,
                      "first_entirely_outside": first_entirely_outside,
                      "author_label": "Unsafe."}))


if __name__ == "__main__":
    main()
