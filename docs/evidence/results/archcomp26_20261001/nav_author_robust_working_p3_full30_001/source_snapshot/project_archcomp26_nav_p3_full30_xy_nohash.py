#!/usr/bin/env python3
"""Project saved NAV P3 full30 ranges to the existing x/y curve CSV layout."""

import argparse
import csv
import json
import math
from pathlib import Path
import struct


PROFILES = {"nav-standard": 640, "nav-robust": 25}
STEPS = 600
RECORD = struct.Struct("<II16d")
FIELDS = ("instance", "method", "source_generation", "step", "t_start", "t_end",
          "state", "tube_lo", "tube_hi", "tube_union_width", "endpoint_lo",
          "endpoint_hi", "endpoint_union_width", "partition_mean_tube_width",
          "partition_max_tube_width", "partition_mean_endpoint_width",
          "partition_max_endpoint_width")


def project(run: Path):
    instance = json.loads((run / "START.json").read_text())["instance"]
    box_count = PROFILES[instance]
    audit = json.loads((run / "INDEPENDENT_SAVED_RANGE_SCAN.json").read_text())
    if (not audit["complete_saved_grid"] or audit["records"] != box_count * STEPS or
            any(audit[key] for key in (
                "out_of_order_or_missing_identity_records", "nonfinite_records",
                "reversed_component_intervals", "endpoint_outside_same_step_tube_components"))):
        raise ValueError("NAV P3 raw ranges lack a complete valid independent scan")
    source = run / "ranges.bin"
    output = run / "xy_saved_curves.csv"
    with source.open("rb") as ranges, output.open("x", newline="") as saved:
        writer = csv.DictWriter(saved, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        for step in range(1, STEPS + 1):
            intervals = []
            for lane in range(box_count):
                record = ranges.read(RECORD.size)
                if len(record) != RECORD.size:
                    raise ValueError(f"missing NAV P3 saved range at step {step}, lane {lane}")
                got_lane, got_step, *values = RECORD.unpack(record)
                if (got_lane, got_step) != (lane, step):
                    raise ValueError(f"unexpected NAV P3 saved range identity at step {step}, lane {lane}")
                intervals.append(values)
            for axis, state in enumerate(("x", "y")):
                tube_lo = [v[4 * axis] for v in intervals]
                tube_hi = [v[4 * axis + 1] for v in intervals]
                end_lo = [v[4 * axis + 2] for v in intervals]
                end_hi = [v[4 * axis + 3] for v in intervals]
                lo, hi = min(tube_lo), max(tube_hi)
                elo, ehi = min(end_lo), max(end_hi)
                tube_widths = [b - a for a, b in zip(tube_lo, tube_hi)]
                end_widths = [b - a for a, b in zip(end_lo, end_hi)]
                writer.writerow(dict(zip(FIELDS, (
                    instance, "working_p3", "2026-10-02 new working P3",
                    step, (step - 1) / 100, step / 100, state,
                    lo, hi, hi - lo, elo, ehi, ehi - elo,
                    math.fsum(tube_widths) / box_count, max(tube_widths),
                    math.fsum(end_widths) / box_count, max(end_widths)))))
        if ranges.read(1):
            raise ValueError("extra NAV P3 range bytes after 600 steps")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    print(project(args.run))


if __name__ == "__main__":
    main()
