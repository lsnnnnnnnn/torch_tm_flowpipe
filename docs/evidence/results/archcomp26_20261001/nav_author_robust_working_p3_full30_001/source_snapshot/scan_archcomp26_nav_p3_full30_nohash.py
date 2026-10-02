#!/usr/bin/env python3
"""Independently inspect saved NAV P3 full-horizon binary range records."""

import argparse
import json
import math
from pathlib import Path
import struct


PROFILES = {"nav-standard": 640, "nav-robust": 25}
STEPS = 600
RECORD = struct.Struct("<II16d")


def scan(run: Path) -> dict:
    source = run / "ranges.bin"
    instance = json.loads((run / "START.json").read_text())["instance"]
    box_count = PROFILES[instance]
    boxes = json.loads((run / "initial_boxes.json").read_text())
    if len(boxes) != box_count or any(len(box) != 7 for box in boxes):
        raise ValueError("NAV source grid differs")
    size = source.stat().st_size
    if size % RECORD.size:
        raise ValueError("NAV range file ends with an incomplete binary record")
    records = size // RECORD.size
    expected = box_count * STEPS
    out_of_order = nonfinite = reversed_intervals = endpoint_outside = 0
    obstacle_intersections = first_initial_outside = terminal_outside = 0
    final_endpoint = [[math.inf, -math.inf] for _ in range(4)]
    with source.open("rb") as stream:
        for index in range(records):
            lane, step, *values = RECORD.unpack(stream.read(RECORD.size))
            out_of_order += int(lane != index % box_count or step != index // box_count + 1)
            nonfinite += int(not all(map(math.isfinite, values)))
            tube = []
            endpoint = []
            for axis in range(4):
                lo, hi, end_lo, end_hi = values[4 * axis:4 * axis + 4]
                tube.append((lo, hi))
                endpoint.append((end_lo, end_hi))
                reversed_intervals += int(lo > hi or end_lo > end_hi)
                endpoint_outside += int(end_lo < lo or end_hi > hi)
                if step == 1 and lane < box_count:
                    first_initial_outside += int(
                        boxes[lane][axis][0] < lo or boxes[lane][axis][1] > hi)
                if step == STEPS:
                    final_endpoint[axis][0] = min(final_endpoint[axis][0], end_lo)
                    final_endpoint[axis][1] = max(final_endpoint[axis][1], end_hi)
            obstacle_intersections += int(
                tube[0][0] <= 2 and tube[0][1] >= 1 and
                tube[1][0] <= 2 and tube[1][1] >= 1)
            if step == STEPS:
                terminal_outside += int(any(
                    endpoint[axis][0] < -.5 or endpoint[axis][1] > .5
                    for axis in (0, 1)))
    complete = (records == expected and out_of_order == 0 and nonfinite == 0 and
                reversed_intervals == 0 and endpoint_outside == 0)
    return {
        "schema": "archcomp26-nav-p3-full30-saved-range-scan-nohash-v1",
        "instance": instance, "initial_boxes": box_count,
        "source": str(source), "record_bytes": RECORD.size,
        "records": records, "expected_records": expected,
        "complete_saved_grid": complete,
        "out_of_order_or_missing_identity_records": out_of_order,
        "nonfinite_records": nonfinite,
        "reversed_component_intervals": reversed_intervals,
        "endpoint_outside_same_step_tube_components": endpoint_outside,
        "first_step_source_box_outside_tube_components": first_initial_outside,
        "saved_tube_intersections_with_closed_obstacle": obstacle_intersections,
        "terminal_lanes_outside_closed_target": terminal_outside if complete else None,
        "final_endpoint_union": final_endpoint if complete else None,
        "interpretation": "saved numerical intervals only; no independent NNCS floating-point proof",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    result = scan(args.run)
    (args.run / "INDEPENDENT_SAVED_RANGE_SCAN.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
