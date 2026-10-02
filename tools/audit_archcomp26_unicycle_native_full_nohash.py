#!/usr/bin/env python3
"""Audit saved native Unicycle full50 receipts and terminal target inclusion."""

import argparse
import json
import math
from pathlib import Path
import re
import struct


TARGET = [[-0.6, 0.6], [-0.2, 0.2], [-0.06, 0.06], [-0.3, 0.3]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run = args.run
    outer = json.loads((run / "RESULT.json").read_text())
    initial = json.loads((run / "INITIAL_FRACTION_AUDIT.json").read_text())
    scan = json.loads((run / "RANGE_SCAN.json").read_text())
    log = (run / "native.log").read_text()
    periods = [(int(a), int(b), int(c)) for a, b, c in
               re.findall(r"^PERIOD (\d+) STATUS (\d+) CUMULATIVE_SEGMENTS (\d+)$", log, re.M)]
    period_grid = len(periods) == 50 and all(row == (i, 2, 10*(i+1))
                                             for i, row in enumerate(periods))
    rpc = [json.loads(line) for line in (run / "controller_rpc.jsonl").read_text().splitlines()]
    valid_rpc = len(rpc) == 50
    for call in rpc:
        lower, upper, response = call["input_lower"], call["input_upper"], call["response"]
        valid_rpc &= len(lower) == len(upper) == 4 and all(
            math.isfinite(a) and math.isfinite(b) and a <= b for a, b in zip(lower, upper))
        valid_rpc &= len(response["T"]) == 2 and all(len(row) == 4 for row in response["T"])
        valid_rpc &= len(response["u_min"]) == len(response["u_max"]) == 2
        valid_rpc &= all(math.isfinite(x) for row in response["T"] for x in row)
        valid_rpc &= all(math.isfinite(a) and math.isfinite(b) and a <= b
                         for a, b in zip(response["u_min"], response["u_max"]))
    struct_row = struct.Struct("<QQd" + "d" * 16)
    with (run / "ranges.bin").open("rb") as stream:
        steps = [struct_row.unpack(chunk) for chunk in iter(lambda: stream.read(struct_row.size), b"")]
    fixed_step = len(steps) == 500 and all(lane == 0 and step == i+1 and h == 0.02
                                            for i, (lane, step, h, *_) in enumerate(steps))
    terminal = scan["final_endpoint_union"]
    target_inclusion = terminal is not None and len(terminal) == 4 and all(
        lo <= box[0] <= box[1] <= hi for box, (lo, hi) in zip(terminal, TARGET))
    target_margins = None if terminal is None else [
        [box[0]-lo, hi-box[1]] for box, (lo, hi) in zip(terminal, TARGET)]
    numerical_complete = (outer["status"] == "completed" and outer["exit_code"] == 0 and
                          initial["all_8_initial_coordinates_covered"] and
                          initial["repaired_states"] == ["x2"] and period_grid and valid_rpc and
                          fixed_step and scan["complete_grid"] and scan["nonfinite_records"] == 0 and
                          scan["reversed_component_intervals"] == 0 and
                          scan["endpoint_outside_same_step_tube_components"] == 0)
    result = {"schema": "archcomp26-unicycle-native-full50-audit-nohash-v1",
              "run": str(run), "full_50_periods_500_substeps": numerical_complete,
              "saved_period_grid_complete": period_grid, "rpc_calls": len(rpc),
              "rpc_shapes_and_ranges_valid": valid_rpc, "fixed_0p02_step_grid": fixed_step,
              "initial_full_box_covered": initial["all_8_initial_coordinates_covered"],
              "terminal_physical_endpoint": terminal, "target": TARGET,
              "terminal_subset_of_target": target_inclusion,
              "target_lower_upper_margins": target_margins,
              "author_checker_printed_verified": "\nVERIFIED\n" in log,
              "interpretation": "Saved full numerical terminal enclosure is within target, sufficient for reach within 10 s under chosen terminal rule; no independent end-to-end floating-point NNCS certificate."}
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"full_50_periods_500_substeps": numerical_complete,
                      "terminal_subset_of_target": target_inclusion,
                      "rpc_calls": len(rpc)}))
    return 0 if numerical_complete and target_inclusion else 2


if __name__ == "__main__":
    raise SystemExit(main())
