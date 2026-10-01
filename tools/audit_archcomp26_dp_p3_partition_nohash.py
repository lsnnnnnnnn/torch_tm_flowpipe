#!/usr/bin/env python3
"""Independently rescan saved DP P3 lane tubes; no model execution."""

import argparse
import json
import math
from pathlib import Path


def scan(run):
    attempt = run / "attempt"
    outer = json.loads((attempt / "RESULT.json").read_text())
    inner = json.loads((attempt / "data/RESULT.json").read_text())
    counts = {"substeps": 0, "accepted_lane_substeps": 0,
              "inside_lane_substeps": 0, "first_unsafe_substep": None,
              "first_failed_substep": None, "minimum_valid_tube_margin": None,
              "minimum_margin_location": None}
    for line in (attempt / "data/observations.jsonl").open():
        row = json.loads(line)
        counts["substeps"] += 1
        step = counts["substeps"]
        if row["substep"] != step:
            raise AssertionError("nonconsecutive DP substep ledger")
        fields = (row["accepted"], row["tube_endpoint_valid"], row["status"],
                  row["whole_tube_inside_safe_box"], row["tube"], row["endpoint"])
        if any(len(field) != 225 for field in fields):
            raise AssertionError("DP ledger lane count differs from 225")
        for lane, (accepted, tagged_valid, status, tagged_inside, tube, endpoint) in enumerate(zip(*fields)):
            if accepted != tagged_valid or len(tube) != 4 or len(endpoint) != 4:
                raise AssertionError("DP lane validity or physical dimension mismatch")
            if accepted and status != 0:
                raise AssertionError("accepted DP lane has nonactive status")
            if not accepted and counts["first_failed_substep"] is None:
                counts["first_failed_substep"] = step
            inside = accepted
            if accepted:
                counts["accepted_lane_substeps"] += 1
                if any(not (len(pair) == 2 and math.isfinite(pair[0])
                            and math.isfinite(pair[1]) and pair[0] <= pair[1])
                       for pair in endpoint):
                    raise AssertionError("invalid accepted DP endpoint interval")
                for variable, interval in enumerate(tube):
                    lo, hi = interval
                    if not (math.isfinite(lo) and math.isfinite(hi) and lo <= hi):
                        raise AssertionError("invalid accepted DP tube interval")
                    for side, margin in (("lower", lo + 1.7), ("upper", 2.0 - hi)):
                        if (counts["minimum_valid_tube_margin"] is None
                                or margin < counts["minimum_valid_tube_margin"]):
                            counts["minimum_valid_tube_margin"] = margin
                            counts["minimum_margin_location"] = [step, lane, variable, side, lo, hi]
                    inside &= lo >= -1.7 and hi <= 2.0
                counts["inside_lane_substeps"] += int(inside)
                if not inside and counts["first_unsafe_substep"] is None:
                    counts["first_unsafe_substep"] = step
            if inside != tagged_inside:
                raise AssertionError("saved DP safety tag differs from independent interval scan")
    counts["expected_lane_substeps"] = 225 * 100
    counts["full_numeric_safe"] = (outer["status"] == "completed" and inner["status"] == "completed"
                                   and counts["substeps"] == 100
                                   and counts["accepted_lane_substeps"] == 225 * 100
                                   and counts["inside_lane_substeps"] == 225 * 100)
    counts["outer_status"] = outer["status"]
    counts["inner_status"] = inner["status"]
    counts["end_to_end_strict_certificate"] = inner.get("end_to_end_strict_certificate")
    metrics_path = attempt / "data/metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
        counts["metrics_contract"] = {key: metrics[key]
                                      for key in ("B", "order", "steps", "substeps", "broken")}
        if counts["metrics_contract"] != {"B": 225, "order": 3, "steps": 20,
                                           "substeps": 5, "broken": 0}:
            raise AssertionError("completed DP metrics differ from full contract")
    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = scan(args.run)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
