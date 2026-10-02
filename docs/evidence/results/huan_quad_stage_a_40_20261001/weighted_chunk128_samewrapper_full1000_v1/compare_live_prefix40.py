#!/usr/bin/env python3
"""Read-only direct saved-value check of the first 40 steps of a live run."""

import json
from pathlib import Path

import numpy as np
import torch


BASE = Path("/srv/local/shengenli/flowstar_acceleration_20260921T153643Z")
OLD = BASE / "runs/quad_fullbatch_p3_20260928/full1024_p3_trig_gpu14_1000_v1"
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "run_001/data"


def same(left, right):
    return (left.shape == right.shape and left.dtype == right.dtype
            and np.array_equal(left.view(np.uint8), right.view(np.uint8)))


def main():
    rows = [json.loads(line) for line in (DATA / "observations.jsonl").read_text().splitlines()[:40]]
    assert len(rows) == 40, len(rows)
    result = {"schema": "quad-old-author-weighted128-live-prefix40-direct-v1",
              "compared_steps": 0, "first_issue": None}
    for step, row in enumerate(rows, 1):
        old = torch.load(OLD / f"observer_{step}.pt", map_location="cpu", weights_only=True)
        expected = {key: old[key].numpy() for key in ("bounds", "accepted", "status")}
        saved = {key: np.load(DATA / f"observer_{step:04d}_{key}.npy", allow_pickle=False)
                 for key in expected}
        equal = {key: same(expected[key], saved[key]) for key in expected}
        bounds = expected["bounds"]
        pooled = np.stack((bounds[:, :, 0].min(0), bounds[:, :, 1].max(0),
                           bounds[:, :, 2].min(0), bounds[:, :, 3].max(0)), -1)
        equal["pooled"] = same(pooled, np.asarray(row["tube_endpoint_union_12x4"], dtype=np.float64))
        equal["count_status"] = (row["substep"] == step and row["accepted_count"] == 1024
                                  and row["status_counts"] == {"0": 1024})
        result["compared_steps"] = step
        if not all(equal.values()):
            result["first_issue"] = {"step": step, "equal": equal}
            break
    result["status"] = "all_40_saved_steps_direct_equal" if result["first_issue"] is None else "mismatch"
    (ROOT / "LIVE_PREFIX40_COMPARISON.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    return 0 if result["first_issue"] is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
