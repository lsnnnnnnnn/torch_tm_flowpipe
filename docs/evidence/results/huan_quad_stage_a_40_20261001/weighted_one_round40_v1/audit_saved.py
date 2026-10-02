#!/usr/bin/env python3
"""Independently scan saved one-round 40-step QUAD arrays; no solver calls."""

import argparse
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text())


def scan(run, reference):
    outer = read(run / "RESULT.json")
    variant = read(run / "ONE_ROUND_RESULT.json")
    inner = read(run / "data/RESULT.json")
    observations = [json.loads(line) for line in (run / "data/observations.jsonl").read_text().splitlines()]
    old_observations = [json.loads(line) for line in (reference / "observations.jsonl").read_text().splitlines()]
    if not (outer["status"] == "completed" and outer["exit_code"] == 0
            and inner["status"] == "completed" and inner["completed_substeps"] == 40
            and inner["accepted_lane_substeps"] == 40960 and inner["nn_calls"] == 2
            and variant["status"] == "completed_40_one_round"
            and variant["observed_advance_calls"] == 40
            and len(variant["round_rows"]) == len(observations) == len(old_observations) == 40):
        raise ValueError("outer, inner, round, or observation coverage mismatch")
    if any(row["step"] != i or row["eligible"] != 1024
           or row["round"]["attempted"] != 1024 or row["round"]["recovered"] != 1024
           or row["round"]["evaluations"] != 1024 or row["round"]["initial_mismatch"] != 0
           for i, row in enumerate(variant["round_rows"], 1)):
        raise ValueError("one-round diagnostic count mismatch")

    changed_steps = []
    union_wider_components = 0
    union_narrower_components = 0
    union_max_abs_delta = 0.0
    for i, (old, new) in enumerate(zip(old_observations, observations), 1):
        if (old["substep"] != new["substep"] or new["substep"] != i
                or old["accepted_count"] != new["accepted_count"] or new["accepted_count"] != 1024
                or old["status_counts"] != new["status_counts"]
                or old["sr_length"] != new["sr_length"] or old["sr_epoch"] != new["sr_epoch"]):
            raise ValueError(f"saved observation identity mismatch at {i}")
        a = np.asarray(old["tube_endpoint_union_12x4"], dtype=np.float64)
        b = np.asarray(new["tube_endpoint_union_12x4"], dtype=np.float64)
        if a.shape != b.shape or b.shape != (12, 4) or not np.isfinite(b).all():
            raise ValueError(f"invalid union at {i}")
        if not np.array_equal(a.view(np.uint64), b.view(np.uint64)):
            changed_steps.append(i)
        lower = (0, 2)
        upper = (1, 3)
        union_wider_components += int(np.count_nonzero(b[:, lower] < a[:, lower]))
        union_wider_components += int(np.count_nonzero(b[:, upper] > a[:, upper]))
        union_narrower_components += int(np.count_nonzero(b[:, lower] > a[:, lower]))
        union_narrower_components += int(np.count_nonzero(b[:, upper] < a[:, upper]))
        union_max_abs_delta = max(union_max_abs_delta, float(np.max(np.abs(a - b))))

    arrays = {}
    for stem in ("final_tube_12x2", "final_endpoint_12x2"):
        a = np.load(reference / (stem + ".npy"), allow_pickle=False)
        b = np.load(run / "data" / (stem + ".npy"), allow_pickle=False)
        if a.shape != b.shape or b.shape != (1024, 12, 2) or not np.isfinite(b).all():
            raise ValueError(f"invalid {stem} shape or finite gate")
        if np.any(b[..., 0] > b[..., 1]):
            raise ValueError(f"unordered {stem}")
        arrays[stem] = {
            "direct_equal": bool(np.array_equal(a.view(np.uint64), b.view(np.uint64))),
            "changed_endpoints": int(np.count_nonzero(a.view(np.uint64) != b.view(np.uint64))),
            "variant_wider_endpoints": int(np.count_nonzero(b[..., 0] < a[..., 0])
                                           + np.count_nonzero(b[..., 1] > a[..., 1])),
            "variant_narrower_endpoints": int(np.count_nonzero(b[..., 0] > a[..., 0])
                                             + np.count_nonzero(b[..., 1] < a[..., 1])),
            "reference_width_sum": float(np.sum(a[..., 1] - a[..., 0])),
            "variant_width_sum": float(np.sum(b[..., 1] - b[..., 0])),
            "max_abs_endpoint_delta": float(np.max(np.abs(a - b))),
        }
    old_status = np.load(reference / "final_status.npy", allow_pickle=False)
    new_status = np.load(run / "data/final_status.npy", allow_pickle=False)
    if old_status.shape != new_status.shape or old_status.shape != (1024,):
        raise ValueError("invalid final status shape")
    return {
        "schema": "quad-old-author-p3-one-round40-independent-saved-audit-nohash-v1",
        "status": "passed_saved_numeric_scan",
        "observed_steps": 40,
        "accepted_lane_steps": 40960,
        "one_round_accepts": 40960,
        "nn_calls": 2,
        "first_numerical_reject": None,
        "all_observations_changed_from_two_round": len(changed_steps) == 40,
        "changed_union_steps": changed_steps,
        "union_variant_wider_endpoints": union_wider_components,
        "union_variant_narrower_endpoints": union_narrower_components,
        "union_max_abs_endpoint_delta": union_max_abs_delta,
        "final_arrays": arrays,
        "final_status_direct_equal": bool(np.array_equal(old_status, new_status)),
        "final_status_unique": [int(v) for v in np.unique(new_status)],
        "outer_wall_s": outer["wall_s"],
        "inner_driver_elapsed_s": inner["driver_elapsed_s"],
        "same_horizon_as_reference": True,
        "horizon_s": 0.2,
        "full_1000_step_or_speed_inference": False,
        "end_to_end_certificate": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    args = parser.parse_args()
    result = scan(args.run, args.reference)
    (args.run / "SAVED_AUDIT.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "observed_steps": result["observed_steps"],
                      "final_tube_changed": result["final_arrays"]["final_tube_12x2"]["changed_endpoints"]}))


if __name__ == "__main__":
    main()
