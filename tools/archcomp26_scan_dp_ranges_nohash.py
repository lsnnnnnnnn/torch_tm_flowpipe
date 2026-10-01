#!/usr/bin/env python3
"""Inspect saved four-state DP range records without a content digest."""

import argparse
import json
from pathlib import Path

import numpy as np


DTYPE = np.dtype([
    ("lane", "<u8"), ("step", "<u8"), ("h", "<f8"),
    ("bounds", "<f8", (4, 4)),
])


def scan(path, lanes, steps, h, safe_lo, safe_hi, prefix_steps):
    size = path.stat().st_size
    if size != lanes * steps * DTYPE.itemsize:
        raise ValueError(f"record size mismatch: {size} bytes")
    data = np.fromfile(path, dtype=DTYPE)
    data = data[np.lexsort((data["lane"], data["step"]))]
    if not np.array_equal(data["step"], np.repeat(np.arange(1, steps + 1), lanes)):
        raise ValueError("missing, duplicate, or out-of-order step")
    if not np.array_equal(data["lane"], np.tile(np.arange(lanes), steps)):
        raise ValueError("missing, duplicate, or out-of-order lane")
    if not np.all(data["h"] == h):
        raise ValueError("unexpected ODE substep length")
    bounds = data["bounds"]
    if not np.isfinite(bounds).all():
        raise ValueError("nonfinite bound")
    if not (bounds[:, :, 0] <= bounds[:, :, 1]).all() or not (bounds[:, :, 2] <= bounds[:, :, 3]).all():
        raise ValueError("reversed interval")

    lower_gap = bounds[:, :, 0] - bounds[:, :, 2]
    upper_gap = bounds[:, :, 3] - bounds[:, :, 1]
    prefix = bounds[:prefix_steps * lanes]
    terminal = prefix[-lanes:, :, 2:4]
    width = terminal[:, :, 1] - terminal[:, :, 0]
    crossing = ((bounds[:, :, 0] < safe_lo) | (bounds[:, :, 1] > safe_hi)).any(axis=1)
    outside = ((bounds[:, :, 1] < safe_lo) | (bounds[:, :, 0] > safe_hi)).any(axis=1)
    crossing_counts = crossing.reshape(steps, lanes).sum(axis=1)
    outside_counts = outside.reshape(steps, lanes).sum(axis=1)
    return {
        "schema": "archcomp26-dp-four-state-range-scan-v1",
        "source": str(path.resolve()), "source_bytes": size,
        "record_layout": "uint64 lane, uint64 1-based step, float64 h, 4*(tube_lo,tube_hi,endpoint_lo,endpoint_hi)",
        "lanes": lanes, "steps": steps, "records": len(data), "ode_h": h,
        "statistics_through_step": prefix_steps,
        "all_records_finite_ordered": True,
        "tube_inside_safe_box": bool((bounds[:, :, 0] >= safe_lo).all() and (bounds[:, :, 1] <= safe_hi).all()),
        "endpoint_inside_safe_box": bool((bounds[:, :, 2] >= safe_lo).all() and (bounds[:, :, 3] <= safe_hi).all()),
        "prefix_tube_inside_safe_box": bool((prefix[:, :, 0] >= safe_lo).all() and (prefix[:, :, 1] <= safe_hi).all()),
        "first_observed_step_with_tube_crossing": int(np.flatnonzero(crossing_counts)[0] + 1) if crossing_counts.any() else None,
        "first_observed_step_with_tube_entirely_outside": int(np.flatnonzero(outside_counts)[0] + 1) if outside_counts.any() else None,
        "tube_crossing_lane_counts_by_step": [int(n) for n in crossing_counts],
        "tube_entirely_outside_lane_counts_by_step": [int(n) for n in outside_counts],
        "endpoint_below_tube_count": int((lower_gap > 0).sum()),
        "endpoint_above_tube_count": int((upper_gap > 0).sum()),
        "endpoint_tube_gap_max": float(max(lower_gap.max(), upper_gap.max(), 0.0)),
        "all_time_tube_union": [[float(prefix[:, i, 0].min()), float(prefix[:, i, 1].max())] for i in range(4)],
        "terminal_endpoint_union": [[float(terminal[:, i, 0].min()), float(terminal[:, i, 1].max())] for i in range(4)],
        "terminal_endpoint_union_width": [float(terminal[:, i, 1].max() - terminal[:, i, 0].min()) for i in range(4)],
        "terminal_per_box_width_mean": [float(width[:, i].mean()) for i in range(4)],
        "terminal_per_box_width_max": [float(width[:, i].max()) for i in range(4)],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranges", required=True, type=Path)
    parser.add_argument("--lanes", required=True, type=int)
    parser.add_argument("--steps", required=True, type=int)
    parser.add_argument("--prefix-steps", type=int)
    parser.add_argument("--h", required=True, type=float)
    parser.add_argument("--safe-lo", required=True, type=float)
    parser.add_argument("--safe-hi", required=True, type=float)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.lanes <= 0 or args.steps <= 0 or args.h <= 0 or args.safe_lo >= args.safe_hi:
        parser.error("invalid contract values")
    prefix_steps = args.steps if args.prefix_steps is None else args.prefix_steps
    if not 1 <= prefix_steps <= args.steps:
        parser.error("--prefix-steps must be within the observed step count")
    result = scan(args.ranges, args.lanes, args.steps, args.h, args.safe_lo, args.safe_hi, prefix_steps)
    with args.output.open("x") as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write("\n")
    print(json.dumps({k: result[k] for k in ("records", "tube_inside_safe_box", "endpoint_inside_safe_box", "endpoint_tube_gap_max")}))


if __name__ == "__main__":
    main()
