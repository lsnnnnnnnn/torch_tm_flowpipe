#!/usr/bin/env python3
"""Compare the two saved 40-step observer arms by direct numerical values."""

import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent


def read_json(path):
    return json.loads(path.read_text())


def arm(name):
    root = HERE / (name + "_002")
    data = root / "data"
    result = read_json(root / "RESULT.json")
    detail = read_json(data / "RESULT.json")
    metrics = read_json(data / "metrics.json")
    rows = [json.loads(line) for line in (data / "observations.jsonl").read_text().splitlines()]
    assert result["status"] == detail["status"] == "completed"
    assert detail["mode"] == name and detail["completed_substeps"] == 40
    assert detail["accepted_lane_substeps"] == 40960 and detail["all_lanes_accepted"]
    assert detail["nn_calls"] == 2 and metrics["broken"] == 0
    assert [x["substep"] for x in rows] == list(range(1, 41))
    assert all(x["accepted_count"] == 1024 and not x["rejected_lanes"] for x in rows)
    if name == "observer_on":
        assert all(x["tube_endpoint_union_12x4"] is not None for x in rows)
    else:
        assert all(x["tube_endpoint_union_12x4"] is None for x in rows)
    arrays = {}
    for field in ("final_tube_12x2", "final_endpoint_12x2", "final_status"):
        arrays[field] = np.load(data / (field + ".npy"), allow_pickle=False)
        assert np.isfinite(arrays[field]).all()
    return result, detail, metrics, rows, arrays


def main():
    on, off = arm("observer_on"), arm("observer_off")
    for field in ("B", "steps", "substeps", "broken", "ctrl_steps", "final_hull",
                  "final_hull_width_sum_mean", "order", "coupling"):
        assert on[2][field] == off[2][field], field
    for a, b in zip(on[3], off[3]):
        for field in ("substep", "accepted_count", "status_counts", "rejected_lanes",
                      "sr_length", "sr_epoch"):
            assert a[field] == b[field], (a["substep"], field)
    for field in on[4]:
        assert on[4][field].shape == off[4][field].shape
        assert on[4][field].dtype == off[4][field].dtype
        assert np.array_equal(on[4][field], off[4][field]), field
    comparison = {
        "schema": "quad-old-stage-a-p3-observer-nohash-comparison-v1",
        "source_arms": [str((HERE / (name + "_002")).resolve())
                        for name in ("observer_on", "observer_off")],
        "old_author_equations": True, "boxes": 1024, "substeps": 40,
        "per_step_acceptance_status_sr_direct_equal": True,
        "driver_final_hull_and_controller_record_summaries_equal": True,
        "final_per_lane_tube_endpoint_status_arrays_direct_equal": True,
        "on_driver_elapsed_s": on[1]["driver_elapsed_s"],
        "off_driver_elapsed_s": off[1]["driver_elapsed_s"],
        "on_outer_wall_s": on[0]["wall_s"],
        "off_outer_wall_s": off[0]["wall_s"],
        "on_observer_region_wall_s": on[1]["observer_region_wall_s"],
        "off_observer_region_wall_s": off[1]["observer_region_wall_s"],
        "qualification": "one sequential pair; CPU/GPU phase noise and changed synchronization prohibit full-horizon speed extrapolation",
    }
    (HERE / "COMPARISON.json").write_text(json.dumps(comparison, indent=2) + "\n")
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
