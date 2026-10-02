#!/usr/bin/env python3
"""Read the saved DP-more P3 intervals and receipts without rerunning the solver."""

import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
DATA = HERE / "attempt/data"
SAFE_LO, SAFE_HI = -1.5, 1.5


def read_json(path):
    return json.loads(path.read_text())


def main():
    source_contract = (HERE / "source/double_pendulum_more_robust_paper_p4.yaml").read_text()
    generated_contract = (DATA / "config.yaml").read_text()
    uncommented = lambda value: "\n".join(
        line for line in value.splitlines() if line.strip() and not line.lstrip().startswith("#")
    )
    assert uncommented(source_contract) == uncommented(generated_contract)
    assert "steps: 20" in generated_contract
    assert "step_size: 0.02" in generated_contract
    assert "ode_step_size: 0.005" in generated_contract

    outer = read_json(HERE / "attempt/RESULT.json")
    inner = read_json(DATA / "RESULT.json")
    start = read_json(DATA / "START.json")
    assert start["mode"] == "full" and start["physical_gpu"] == 2
    assert outer["status"] == "failed" and outer["exit_code"] == 1 and not outer["timed_out"]
    assert inner["status"] == "exception" and inner["completed_substeps"] == 9
    assert inner["error"] == "first P3 DP more substep rejection; stopped after saving raw observation"

    rows = [json.loads(line) for line in (DATA / "observations.jsonl").read_text().splitlines()]
    assert len(rows) == 9
    first_period = [json.loads(line) for line in (
        HERE.parent / "dp_more_p3_firstperiod_interval_20261002_001/attempt/data/observations.jsonl"
    ).read_text().splitlines()]
    first_period_equal = rows[:4] == first_period
    assert first_period_equal

    per_step = []
    valid_intervals = 0
    first_crossing = None
    for step, row in enumerate(rows, 1):
        assert row["substep"] == step
        assert all(len(row[key]) == 225 for key in (
            "accepted", "tube_endpoint_valid", "status", "whole_tube_inside_safe_box",
            "tube", "endpoint"
        ))
        assert row["tube_endpoint_valid"] == row["accepted"]
        accepted = sum(row["accepted"])
        assert accepted == (225 if step <= 8 else 0)
        assert set(row["status"]) == ({0} if step <= 8 else {1})
        crossings = [0, 0, 0, 0]
        entirely_outside = [0, 0, 0, 0]
        safe = 0
        for lane in range(225):
            if not row["accepted"][lane]:
                assert not row["whole_tube_inside_safe_box"][lane]
                continue  # rejected rows retain the previous state, not a new enclosure
            lane_safe = True
            assert len(row["tube"][lane]) == len(row["endpoint"][lane]) == 4
            for state, (tube, endpoint) in enumerate(zip(row["tube"][lane], row["endpoint"][lane])):
                assert len(tube) == len(endpoint) == 2
                assert all(math.isfinite(value) for value in (*tube, *endpoint))
                assert tube[0] <= tube[1] and endpoint[0] <= endpoint[1]
                assert tube[0] <= endpoint[0] and endpoint[1] <= tube[1]
                valid_intervals += 1
                if tube[0] < SAFE_LO or tube[1] > SAFE_HI:
                    crossings[state] += 1
                    lane_safe = False
                    if first_crossing is None:
                        first_crossing = {"substep": step, "lane": lane, "state": state, "tube": tube}
                if tube[1] < SAFE_LO or tube[0] > SAFE_HI:
                    entirely_outside[state] += 1
            assert row["whole_tube_inside_safe_box"][lane] == lane_safe
            safe += lane_safe
        per_step.append({
            "substep": step,
            "accepted": accepted,
            "safe": safe,
            "status_counts": {str(value): row["status"].count(value) for value in set(row["status"])},
            "crossing_by_state": crossings,
            "entire_tube_outside_by_state": entirely_outside,
        })

    assert sum(row["accepted"] for row in per_step) == inner["accepted_lane_substeps"] == 1800
    result = {
        "schema": "dp-more-p3-fullattempt-independent-scan-nohash-v1",
        "remote_original": "/srv/local/shengenli/flowstar_acceleration_20260921T153643Z/runs/archcomp26_20261001/dp_more_p3_full20_interval_20261002_001",
        "full_contract_config_matches_source_except_comments": True,
        "first_four_observations_equal_prior_gate": first_period_equal,
        "valid_accepted_state_intervals": valid_intervals,
        "accepted_lane_substeps": 1800,
        "first_rejection": {"substep": 9, "period_index_zero_based": 2, "rejected_lanes": 225,
                            "solver_status": "FAILED_CONTRACTION", "solver_status_code": 1},
        "first_saved_safe_band_crossing": first_crossing,
        "per_substep": per_step,
        "accepted_numerical_prefix_end_s": 0.04,
        "safe_numerical_prefix_end_s": 0.01,
        "full_horizon_complete": False,
        "property_verdict_from_full_driver": None,
    }
    (HERE / "INDEPENDENT_INTERVAL_SCAN_NOHASH.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"accepted": 1800, "first_rejection": 9, "valid_intervals": valid_intervals}))


if __name__ == "__main__":
    main()
