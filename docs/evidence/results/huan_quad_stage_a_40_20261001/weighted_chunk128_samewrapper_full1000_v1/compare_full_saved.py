#!/usr/bin/env python3
"""Compare every saved candidate observer step with the archived P3/trig run."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def same_bytes(left, right):
    return (left.shape == right.shape and left.dtype == right.dtype
            and np.array_equal(left.view(np.uint8), right.view(np.uint8)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    data = args.candidate / "data"
    outer = json.loads((args.candidate / "RESULT.json").read_text())
    inner = json.loads((data / "RESULT.json").read_text())
    rows = [json.loads(line) for line in (data / "observations.jsonl").read_text().splitlines()]
    result = {"schema": "quad-old-author-p3-weighted128-full-observer-direct-comparison-v1",
              "reference": str(args.reference), "candidate": str(args.candidate),
              "reference_steps_available": sum((args.reference / f"observer_{i}.pt").is_file()
                                               for i in range(1, 1001)),
              "candidate_outer_status": outer["status"],
              "candidate_inner_status": inner["status"],
              "candidate_completed_substeps": inner["completed_substeps"],
              "candidate_accepted_lane_substeps": inner["accepted_lane_substeps"],
              "candidate_control_calls": inner.get("nn_calls"),
              "candidate_weighted_counters": inner.get("weighted_counters"),
              "candidate_observation_rows": len(rows),
              "compared_steps": 0, "bounds_equal_steps": 0,
              "accepted_equal_steps": 0, "status_equal_steps": 0,
              "pooled_equal_steps": 0, "first_issue": None}
    limit = min(1000, len(rows), inner["completed_substeps"])
    for step in range(1, limit + 1):
        old = torch.load(args.reference / f"observer_{step}.pt",
                         map_location="cpu", weights_only=True)
        saved = {key: np.load(data / f"observer_{step:04d}_{key}.npy", allow_pickle=False)
                 for key in ("bounds", "accepted", "status")}
        previous = {key: old[key].numpy() for key in saved}
        equal = {key: same_bytes(previous[key], saved[key]) for key in saved}
        pool = np.stack((previous["bounds"][:, :, 0].min(0),
                         previous["bounds"][:, :, 1].max(0),
                         previous["bounds"][:, :, 2].min(0),
                         previous["bounds"][:, :, 3].max(0)), -1)
        pooled_equal = same_bytes(pool, np.asarray(rows[step - 1]["tube_endpoint_union_12x4"],
                                                 dtype=np.float64))
        counts_equal = (rows[step - 1]["substep"] == step
                        and rows[step - 1]["accepted_count"] == int(previous["accepted"].sum())
                        and rows[step - 1]["status_counts"] ==
                        {str(int(k)): int(v) for k, v in zip(*np.unique(previous["status"], return_counts=True))})
        result["compared_steps"] = step
        for key in saved:
            result[key + "_equal_steps"] += int(equal[key])
        result["pooled_equal_steps"] += int(pooled_equal and counts_equal)
        if not all(equal.values()) or not pooled_equal or not counts_equal:
            result["first_issue"] = {"step": step, "per_box_direct_equal": equal,
                                     "pooled_direct_equal": pooled_equal,
                                     "saved_count_status_equal": counts_equal}
            break
    result["status"] = ("all_1000_saved_steps_direct_equal" if result["compared_steps"] == 1000
                        and result["first_issue"] is None else "incomplete_or_mismatch")
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "compared_steps", "first_issue")}))
    return 0 if result["status"] == "all_1000_saved_steps_direct_equal" else 1


if __name__ == "__main__":
    raise SystemExit(main())
