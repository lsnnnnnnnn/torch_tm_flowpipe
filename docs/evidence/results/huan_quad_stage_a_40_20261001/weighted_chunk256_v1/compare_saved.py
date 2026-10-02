#!/usr/bin/env python3
"""Directly compare the isolated 256-row run with both saved 128-row runs."""

import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
STAGE = HERE.parent
SOURCES = {
    "observer_on_002": STAGE / "observer_pair_v1/observer_on_002",
    "weighted_round_gate": STAGE / "weighted_round_gate_v1/run_001",
    "weighted_chunk256": HERE / "run_001",
}


def read(name):
    root = SOURCES[name]
    data = root / "data"
    outer = json.loads((root / "RESULT.json").read_text())
    inner = json.loads((data / "RESULT.json").read_text())
    metrics = json.loads((data / "metrics.json").read_text())
    observations = (data / "observations.jsonl").read_bytes()
    rows = [json.loads(line) for line in observations.splitlines()]
    assert outer["status"] == inner["status"] == "completed", name
    assert inner["completed_substeps"] == 40 and inner["accepted_lane_substeps"] == 40960, name
    assert inner["all_lanes_accepted"] and inner["nn_calls"] == 2, name
    assert metrics["broken"] == 0 and len(rows) == 40, name
    assert [row["substep"] for row in rows] == list(range(1, 41)), name
    assert all(row["accepted_count"] == 1024 and not row["rejected_lanes"] for row in rows), name
    arrays = {key: np.load(data / (key + ".npy"), allow_pickle=False)
              for key in ("final_tube_12x2", "final_endpoint_12x2", "final_status")}
    assert arrays["final_tube_12x2"].shape == arrays["final_endpoint_12x2"].shape == (1024, 12, 2), name
    assert arrays["final_status"].shape == (1024,), name
    return outer, inner, metrics, observations, arrays


def main():
    candidate = read("weighted_chunk256")
    counter = candidate[1]["weighted_counters"]
    assert counter["chunk_size"] == 256 and counter["refine_calls"] == 40, counter
    assert counter["graph_map_calls"] == 320 and counter["target_rows"] == 81920, counter
    assert counter["padded_rows"] == 0, counter
    assert candidate[1]["valid_counters"]["chunk_size"] == 128
    comparison = {"schema": "quad-old-author-p3-weighted256-saved-comparison-v1",
                  "candidate": str(SOURCES["weighted_chunk256"]),
                  "candidate_completed_substeps": 40,
                  "candidate_accepted_lane_substeps": 40960,
                  "candidate_weighted_counters": counter,
                  "candidate_outer_wall_s": candidate[0]["wall_s"],
                  "candidate_driver_elapsed_s": candidate[1]["driver_elapsed_s"],
                  "candidate_cuda_peak_reserved_bytes": candidate[1]["cuda_peak_reserved_bytes"],
                  "references": {}}
    fields = ("B", "steps", "substeps", "broken", "ctrl_steps", "final_hull",
              "final_hull_width_sum_mean", "order", "coupling", "crown_input_layout",
              "crown_transport", "engine")
    for name in ("observer_on_002", "weighted_round_gate"):
        prior = read(name)
        metric_fields = {key: candidate[2][key] == prior[2][key] for key in fields}
        array_fields = {key: bool(np.array_equal(candidate[4][key].view(np.uint8),
                                                 prior[4][key].view(np.uint8)))
                        for key in candidate[4]}
        item = {"source": str(SOURCES[name]),
                "saved_40_observation_rows_direct_bytes_equal": candidate[3] == prior[3],
                "numerical_metric_fields_direct_equal": metric_fields,
                "final_arrays_direct_bytes_equal": array_fields}
        comparison["references"][name] = item
        assert item["saved_40_observation_rows_direct_bytes_equal"], name
        assert all(metric_fields.values()) and all(array_fields.values()), name
    comparison["status"] = "all_saved_values_direct_equal"
    (HERE / "SAVED_COMPARISON.json").write_text(json.dumps(comparison, indent=2) + "\n")
    print(json.dumps({"status": comparison["status"], "weighted_counters": counter,
                      "candidate_outer_wall_s": comparison["candidate_outer_wall_s"]}))


if __name__ == "__main__":
    main()
