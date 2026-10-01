#!/usr/bin/env python3
"""Read native Flow* saved tube/endpoint intervals without content digests."""

import argparse
import json
import math
from pathlib import Path
import struct


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--boxes", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--physical", type=int, required=True)
    parser.add_argument("--safe-lo", type=float)
    parser.add_argument("--safe-hi", type=float)
    args = parser.parse_args()
    if (args.safe_lo is None) != (args.safe_hi is None):
        parser.error("safe-lo and safe-hi must be supplied together")
    if min(args.boxes, args.steps, args.physical) < 1:
        parser.error("boxes, steps and physical must be positive")

    row = struct.Struct("<QQd" + "d" * (4 * args.physical))
    seen = set()
    nonfinite = reversed_intervals = out_of_grid = 0
    endpoint_outside_tube = 0
    largest_endpoint_excursion = 0.0
    safe_crossings = 0
    first_safe_crossing_step = None
    tube_lo = [math.inf] * args.physical
    tube_hi = [-math.inf] * args.physical
    final_lo = [math.inf] * args.physical
    final_hi = [-math.inf] * args.physical
    final_width_sum = [0.0] * args.physical
    final_width_max = [0.0] * args.physical
    final_lanes = set()
    count = 0
    with args.input.open("rb") as source:
        while chunk := source.read(row.size):
            if len(chunk) != row.size:
                raise ValueError("trailing partial record")
            lane, step, h, *v = row.unpack(chunk)
            count += 1
            if (lane, step) in seen:
                raise ValueError(f"duplicate lane/step: {(lane, step)}")
            seen.add((lane, step))
            if lane >= args.boxes or step < 1 or step > args.steps:
                out_of_grid += 1
            if not math.isfinite(h) or h <= 0 or not all(map(math.isfinite, v)):
                nonfinite += 1
                continue
            for i in range(args.physical):
                tlo, thi, elo, ehi = v[4 * i:4 * i + 4]
                if tlo > thi or elo > ehi:
                    reversed_intervals += 1
                if elo < tlo or ehi > thi:
                    endpoint_outside_tube += 1
                    largest_endpoint_excursion = max(largest_endpoint_excursion, tlo - elo, ehi - thi)
                tube_lo[i] = min(tube_lo[i], tlo)
                tube_hi[i] = max(tube_hi[i], thi)
                if args.safe_lo is not None and (tlo < args.safe_lo or thi > args.safe_hi):
                    safe_crossings += 1
                    first_safe_crossing_step = step if first_safe_crossing_step is None else min(first_safe_crossing_step, step)
                if step == args.steps:
                    final_lanes.add(lane)
                    final_lo[i] = min(final_lo[i], elo)
                    final_hi[i] = max(final_hi[i], ehi)
                    final_width_sum[i] += ehi - elo
                    final_width_max[i] = max(final_width_max[i], ehi - elo)

    complete = count == args.boxes * args.steps and out_of_grid == 0 and all(
        (lane, step) in seen for lane in range(args.boxes) for step in range(1, args.steps + 1))
    result = {
        "schema": "archcomp26-native-range-scan-nohash-v1",
        "input": str(args.input), "record_bytes": row.size, "records": count,
        "expected_records": args.boxes * args.steps, "boxes": args.boxes,
        "steps": args.steps, "physical": args.physical, "complete_grid": complete,
        "out_of_grid": out_of_grid, "nonfinite_records": nonfinite,
        "reversed_component_intervals": reversed_intervals,
        "endpoint_outside_same_step_tube_components": endpoint_outside_tube,
        "largest_endpoint_excursion": largest_endpoint_excursion,
        "safe_band": None if args.safe_lo is None else [args.safe_lo, args.safe_hi],
        "safe_tube_crossing_components": safe_crossings,
        "first_safe_crossing_step": first_safe_crossing_step,
        "tube_union": [[tube_lo[i], tube_hi[i]] for i in range(args.physical)],
        "final_endpoint_lanes": len(final_lanes),
        "final_endpoint_union": None if len(final_lanes) != args.boxes else
            [[final_lo[i], final_hi[i]] for i in range(args.physical)],
        "final_endpoint_partition_mean_width": None if len(final_lanes) != args.boxes else
            [v / args.boxes for v in final_width_sum],
        "final_endpoint_partition_max_width": None if len(final_lanes) != args.boxes else final_width_max,
        "interpretation": "saved intervals and their observed coverage only; no independent NNCS floating-point proof",
    }
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: result[key] for key in (
        "records", "complete_grid", "nonfinite_records", "reversed_component_intervals",
        "safe_tube_crossing_components", "final_endpoint_lanes")}))
    return 0 if complete and nonfinite == 0 and reversed_intervals == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
