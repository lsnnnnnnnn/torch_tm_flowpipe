#!/usr/bin/env python3
"""Independently check saved NAV P3 tube and endpoint intervals."""

import argparse
import json
import math
from pathlib import Path


def scan(run: Path, boxes: int) -> dict:
    start = json.loads((run / "START.json").read_text())
    rows = [json.loads(line) for line in (run / "observations.jsonl").read_text().splitlines()]
    expected_steps = start["expected_ode_substeps"]
    counts = {"nonfinite": 0, "reversed": 0, "endpoint_outside_tube": 0,
              "obstacle_intersections": 0, "first_initial_outside_tube": 0}
    source_boxes = (json.loads((run / "initial_boxes.json").read_text())
                    if boxes > 1 else [start["first_box_exact"]])
    if len(source_boxes) != boxes:
        raise ValueError("saved initial-box ledger count differs")
    tube_union = [[math.inf, -math.inf] for _ in range(4)]
    final_endpoints = []
    accepted_total = 0
    first_rejection = None
    for step, row in enumerate(rows, 1):
        if row["substep"] != step:
            raise ValueError(f"noncontiguous saved step: {row['substep']}")
        accepted = row["accepted"]
        accepted = [accepted] if isinstance(accepted, bool) else accepted
        if len(accepted) != boxes:
            raise ValueError(f"step {step} has {len(accepted)} rather than {boxes} lanes")
        ids = [i for i, good in enumerate(accepted) if good]
        if len(row.get("tube", [])) != len(ids) and boxes > 1:
            raise ValueError(f"step {step} saved tube count differs from accepted lanes")
        tubes = [row["tube"]] if boxes == 1 and ids else row.get("tube", [])
        endpoints = [row["endpoint"]] if boxes == 1 and ids else row.get("endpoint", [])
        if len(tubes) != len(ids) or len(endpoints) != len(ids):
            raise ValueError(f"step {step} saved endpoint count differs from accepted lanes")
        accepted_total += len(ids)
        if len(ids) < boxes and first_rejection is None:
            first_rejection = {"step": step, "lanes": [i for i, good in enumerate(accepted) if not good]}
        if step == expected_steps:
            final_endpoints = endpoints
        for lane, tube, endpoint in zip(ids, tubes, endpoints):
            if len(tube) != 4 or len(endpoint) != 4:
                raise ValueError(f"step {step} lane {lane} missing physical states")
            for coord, (seg, end) in enumerate(zip(tube, endpoint)):
                if len(seg) != 2 or len(end) != 2:
                    raise ValueError(f"step {step} lane {lane} malformed interval")
                counts["nonfinite"] += int(not all(map(math.isfinite, seg + end)))
                counts["reversed"] += int(seg[0] > seg[1] or end[0] > end[1])
                counts["endpoint_outside_tube"] += int(end[0] < seg[0] or end[1] > seg[1])
                tube_union[coord][0] = min(tube_union[coord][0], seg[0])
                tube_union[coord][1] = max(tube_union[coord][1], seg[1])
                if step == 1:
                    counts["first_initial_outside_tube"] += int(
                        source_boxes[lane][coord][0] < seg[0] or
                        source_boxes[lane][coord][1] > seg[1])
            counts["obstacle_intersections"] += int(
                tube[0][0] <= 2 and tube[0][1] >= 1 and
                tube[1][0] <= 2 and tube[1][1] >= 1)
    complete = (len(rows) == expected_steps and accepted_total == boxes * expected_steps
                and first_rejection is None and not any(counts.values()))
    return {
        "schema": "archcomp26-nav-p3-saved-interval-scan-nohash-v1",
        "source": str(run / "observations.jsonl"),
        "expected_boxes": boxes, "expected_substeps": expected_steps,
        "observed_substeps": len(rows), "accepted_lane_substeps": accepted_total,
        "expected_lane_substeps": boxes * expected_steps,
        "complete_saved_grid": complete, "first_rejection": first_rejection,
        "errors": counts, "tube_union": tube_union,
        "final_endpoint_union": (
            [[min(end[i][0] for end in final_endpoints),
              max(end[i][1] for end in final_endpoints)] for i in range(4)]
            if complete else None),
        "qualification": "saved numerical intervals only; no independent NNCS floating-point proof",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--boxes", type=int, required=True)
    args = parser.parse_args()
    result = scan(args.run, args.boxes)
    (args.run / "INDEPENDENT_SAVED_RANGE_AUDIT.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
