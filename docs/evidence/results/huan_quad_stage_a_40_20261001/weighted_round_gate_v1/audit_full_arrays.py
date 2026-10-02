#!/usr/bin/env python3
"""Recheck saved early-weighted arrays directly, without content digests."""

import argparse
import json
from pathlib import Path

import numpy as np


def audit(run, reference):
    data = run / "data"
    rows = [json.loads(line) for line in (data / "ROUND_ROWS.jsonl").read_text().splitlines()]
    before = np.load(data / "STEP_CURRENT_BEFORE.npy", allow_pickle=False)
    final = np.load(data / "STEP_CURRENT_AFTER.npy", allow_pickle=False)
    positions = np.load(data / "ROUND_STEP_INDEX.npy", allow_pickle=False)
    accepted = np.load(data / "ROUND_ACCEPTED_MASK.npy", allow_pickle=False)
    proposed = np.load(data / "ROUND_NEW_REMAINDER.npy", allow_pickle=False)
    if not (len(rows) == 40 and before.shape == final.shape == (40, 1024, 16, 2)
            and positions.shape == (80, 2) and accepted.shape == (80, 1024)
            and proposed.shape == (80, 1024, 16, 2)):
        raise ValueError("wrong 40-step, two-round, full-box array shapes")
    counts = [{"accepted": 0, "changed_lanes": 0, "changed_components": 0,
               "width_reduction_sum": 0.0, "gpu_stream_span_s": 0.0} for _ in range(2)]
    partial_second = []
    for step, row in enumerate(rows, 1):
        if row["step"] != step or len(row["rounds"]) != 2:
            raise ValueError(f"wrong row ordering at step {step}")
        current = before[step - 1].copy()
        for round_number, detail in enumerate(row["rounds"], 1):
            index = 2 * (step - 1) + round_number - 1
            mask = accepted[index]
            new = proposed[index]
            if not (np.array_equal(positions[index], (step, round_number))
                    and np.array_equal(mask.astype(np.uint8), detail["accepted_mask"])
                    and np.isfinite(new).all()):
                raise ValueError(f"round input mismatch at {step}/{round_number}")
            intersection = np.stack((np.maximum(current[..., 0], new[..., 0]),
                                     np.minimum(current[..., 1], new[..., 1])), -1)
            if np.any(mask[:, None] & (intersection[..., 0] > intersection[..., 1])):
                raise ValueError(f"disjoint accepted interval at {step}/{round_number}")
            after = np.where(mask[:, None, None], intersection, current)
            changed = current.view(np.uint64) != after.view(np.uint64)
            lane_count = int(changed.any(axis=(1, 2)).sum())
            component_count = int(changed.sum())
            if (int(mask.sum()), lane_count, component_count) != (
                    detail["accepted_count"], detail["changed_lane_count"],
                    detail["changed_component_count"]):
                raise ValueError(f"round change mismatch at {step}/{round_number}")
            bucket = counts[round_number - 1]
            bucket["accepted"] += int(mask.sum())
            bucket["changed_lanes"] += lane_count
            bucket["changed_components"] += component_count
            bucket["width_reduction_sum"] += float(np.sum(
                (current[..., 1] - current[..., 0]) - (after[..., 1] - after[..., 0])))
            bucket["gpu_stream_span_s"] += detail["gpu_stream_span_ms"] / 1000
            current = after
        if not np.array_equal(current.view(np.uint64), final[step - 1].view(np.uint64)):
            raise ValueError(f"actual final remainder mismatch at step {step}")
        if row["rounds"][1]["accepted_count"] != 1024:
            partial_second.append([step, row["rounds"][1]["accepted_count"]])
    direct = {}
    for name in ("final_tube_12x2", "final_endpoint_12x2", "final_status"):
        direct[name] = (data / (name + ".npy")).read_bytes() == (
            reference / (name + ".npy")).read_bytes()
    direct["observations_jsonl"] = [json.loads(line) for line in
                                    (data / "observations.jsonl").read_text().splitlines()] == [
                                        json.loads(line) for line in
                                        (reference / "observations.jsonl").read_text().splitlines()]
    return {"schema": "quad-old-p3-weighted-full-array-independent-audit-nohash-v1",
            "steps": 40, "boxes": 1024, "rounds": counts,
            "partial_second_round_steps": partial_second,
            "all_40_actual_final_remainders_bitwise_reconstructed": True,
            "reference_direct_equal": direct,
            "pass": all(direct.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.run, args.reference)
    (args.run / "FULL_ARRAY_AUDIT.json").write_text(json.dumps(result, indent=2,
                                                               allow_nan=False) + "\n")
    print(json.dumps({"pass": result["pass"], "rounds": result["rounds"]}))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
