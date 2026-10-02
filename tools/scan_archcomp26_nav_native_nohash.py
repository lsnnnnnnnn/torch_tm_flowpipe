#!/usr/bin/env python3
"""Inspect saved NAV native intervals for coverage and the x/y property boxes."""

import argparse
import json
import math
from pathlib import Path
import struct


ROW = struct.Struct("<QQd" + "d" * 16)


def scan(source, boxes, steps):
    last_step = [0] * boxes
    count = invalid = obstacle_overlaps = target_misses = 0
    first_invalid = first_overlap = first_target_miss = None
    endpoint_union = [[math.inf, -math.inf] for _ in range(4)]
    with source.open("rb") as stream:
        while chunk := stream.read(ROW.size):
            count += 1
            if len(chunk) != ROW.size:
                raise ValueError("partial native range record")
            lane, step, h, *v = ROW.unpack(chunk)
            if (lane >= boxes or step != last_step[lane] + 1 or step > steps or
                    not math.isfinite(h) or abs(h - 0.01) > 1e-12 or
                    not all(math.isfinite(x) for x in v) or
                    any(v[i] > v[i + 1] or v[i + 2] > v[i + 3] for i in range(0, 16, 4))):
                invalid += 1
                if first_invalid is None:
                    first_invalid = [int(lane), int(step)]
                continue
            last_step[lane] = step
            xlo, xhi, _, _, ylo, yhi, _, _ = v[:8]
            if not (xlo > 2 or xhi < 1 or ylo > 2 or yhi < 1):
                obstacle_overlaps += 1
                if first_overlap is None:
                    first_overlap = [int(lane), int(step)]
            if step == steps:
                for i in range(4):
                    lo, hi = v[4 * i + 2:4 * i + 4]
                    endpoint_union[i][0] = min(endpoint_union[i][0], lo)
                    endpoint_union[i][1] = max(endpoint_union[i][1], hi)
                if steps == 600 and not (-0.5 <= v[2] <= v[3] <= 0.5 and
                                         -0.5 <= v[6] <= v[7] <= 0.5):
                    target_misses += 1
                    if first_target_miss is None:
                        first_target_miss = int(lane)
    complete = count == boxes * steps and invalid == 0 and all(s == steps for s in last_step)
    return {
        "schema": "archcomp26-nav-native-saved-property-scan-nohash-v1",
        "input": str(source), "record_bytes": ROW.size, "boxes": boxes, "steps": steps,
        "records": count, "expected_records": boxes * steps,
        "complete_grid": complete, "invalid_records": invalid,
        "first_invalid_lane_step": first_invalid,
        "saved_tubes_overlapping_closed_obstacle_box": obstacle_overlaps,
        "first_obstacle_overlap_lane_step": first_overlap,
        "terminal_target_checked": steps == 600 and complete,
        "terminal_endpoint_boxes_outside_closed_target": target_misses if steps == 600 and complete else None,
        "first_terminal_target_miss_lane": first_target_miss if steps == 600 and complete else None,
        "last_step_endpoint_union": endpoint_union if complete else None,
        "interpretation": "Checks saved interval records only; not an independent end-to-end NNCS floating-point certificate",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--boxes", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    args = parser.parse_args()
    if args.boxes < 1 or args.steps < 1:
        parser.error("boxes and steps must be positive")
    result = scan(args.input, args.boxes, args.steps)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: result[k] for k in ("records", "complete_grid", "invalid_records",
                                             "saved_tubes_overlapping_closed_obstacle_box",
                                             "terminal_target_checked",
                                             "terminal_endpoint_boxes_outside_closed_target")}))
    return 0 if result["complete_grid"] and result["saved_tubes_overlapping_closed_obstacle_box"] == 0 and result["terminal_endpoint_boxes_outside_closed_target"] in (None, 0) else 2


if __name__ == "__main__":
    raise SystemExit(main())
