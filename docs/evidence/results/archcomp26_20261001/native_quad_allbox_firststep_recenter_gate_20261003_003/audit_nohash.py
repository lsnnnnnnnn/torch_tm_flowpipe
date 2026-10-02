#!/usr/bin/env python3
"""Audit saved first-control-call, first-step QUAD receipts without digests."""

import csv
from collections import Counter
from itertools import product
import json
import math
from pathlib import Path
import struct


ROOT = Path(__file__).resolve().parent
RUN = ROOT / "run"
N = 1024


def rows(name):
    with (RUN / name).open(newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    problems = []

    def require(condition, message):
        if not condition:
            problems.append(message)

    sources = rows("source_boxes.csv")
    source = {}
    for row in sources:
        lane, coord = int(row["lane"]), int(row["coord"])
        lo, hi = float(row["lo"]), float(row["hi"])
        require(math.isfinite(lo) and math.isfinite(hi) and lo <= hi,
                f"invalid source interval at {lane}/{coord}")
        require((lane, coord) not in source, f"duplicate source interval at {lane}/{coord}")
        source[lane, coord] = (lo, hi)
    require(len(sources) == N * 12 and len(source) == N * 12,
            "source box table is not a complete 1024 by 12 grid")
    axes = [set(source.get((lane, coord)) for lane in range(N)) for coord in range(1, 7)]
    expected_splits = [8, 8, 8, 2, 1, 1]
    require([len(axis) for axis in axes] == expected_splits,
            "source split cardinalities differ from 8,8,8,2,1,1")
    if all(None not in axis for axis in axes):
        observed = {tuple(source[lane, coord] for coord in range(1, 7))
                    for lane in range(N)}
        require(observed == set(product(*axes)), "source Cartesian partition is incomplete")
        for coord, axis in enumerate(axes, 1):
            ordered = sorted(axis)
            require(ordered[0][0] <= -0.4 and ordered[-1][1] >= 0.4,
                    f"source coordinate {coord} does not span [-0.4,0.4]")
            require(all(left[1] >= right[0] for left, right in zip(ordered, ordered[1:])),
                    f"source coordinate {coord} has a gap")
    require(all(source.get((lane, coord)) == (0.0, 0.0)
                for lane in range(N) for coord in range(7, 13)),
            "source x7 through x12 are not zero")

    rpc = json.loads((RUN / "rpc.json").read_text())
    params, bound = rpc["params"], rpc["coefficients"]
    lb, ub = params["input_lb"], params["input_ub"]
    require(len(lb) == N * 12 and len(ub) == N * 12,
            "saved RPC input cardinality mismatch")
    omitted = []
    exact_lower = exact_upper = 0
    for lane in range(N):
        for coord in range(1, 13):
            pos = 12 * lane + coord - 1
            if pos >= len(lb) or pos >= len(ub) or (lane, coord) not in source:
                continue
            a, b = lb[pos], ub[pos]
            lo, hi = source[lane, coord]
            if not (math.isfinite(a) and math.isfinite(b) and a <= b and
                    a <= lo and b >= hi):
                omitted.append([lane, coord, a, b, lo, hi])
            exact_lower += a == lo
            exact_upper += b == hi
    require(not omitted, f"RPC input omits {len(omitted)} source intervals")
    require(len(bound["T"]) == len(bound["u_min"]) == len(bound["u_max"]) == N,
            "saved CROWN output cardinality mismatch")
    invalid_coefficients = 0
    for lane in range(min(len(bound["T"]), N)):
        T, low, high = bound["T"][lane], bound["u_min"][lane], bound["u_max"][lane]
        if (len(T) != 3 or any(len(row) != 12 for row in T) or
                len(low) != 3 or len(high) != 3 or
                any(not math.isfinite(value) for row in T for value in row) or
                any(not math.isfinite(value) for value in low + high) or
                any(a > b for a, b in zip(low, high))):
            invalid_coefficients += 1
    require(invalid_coefficients == 0, "nonfinite, reversed, or malformed CROWN coefficients")
    calls = [json.loads(line) for line in (RUN / "controller_rpc.jsonl").read_text().splitlines()
             if line.strip()]
    require(len(calls) == 1 and calls[0] == {"input_boxes": N, "output_boxes": N},
            "expected exactly one recorded 1024-box CROWN call")

    states = rows("state.csv")
    status_counts = Counter(row["status"] for row in states)
    state_lanes = [int(row["lane"]) for row in states]
    require(state_lanes == list(range(N)), "state rows are not sequential lanes 0 to 1023")
    require(all(row["status"] == "2" and row["accepted_steps"] == "1" and
                math.isfinite(float(row["seconds"])) and float(row["seconds"]) >= 0
                for row in states), "one or more boxes refused or lacked one accepted step")
    scan = json.loads((RUN / "SCAN.json").read_text())
    require(scan["complete_grid"] and scan["records"] == N and
            scan["nonfinite_records"] == 0 and
            scan["reversed_component_intervals"] == 0,
            "saved range records incomplete, nonfinite, or reversed")
    record = struct.Struct("<QQd" + "d" * 48)
    binary = (RUN / "ranges.bin").read_bytes()
    require(len(binary) == N * record.size, "native range byte count mismatch")
    durations = []
    if len(binary) == N * record.size:
        for lane in range(N):
            saved_lane, step, duration, *_ = record.unpack_from(binary, lane * record.size)
            durations.append(duration)
            require(saved_lane == lane and step == 1 and
                    math.isfinite(duration) and abs(duration - 0.005) <= 1e-15,
                    f"unexpected saved range lane, step or duration at {lane}")

    octagon = rows("octagon.csv")
    oct_keys = {(int(row["lane"]), int(row["step"]), int(row["view"]),
                 int(row["form"])) for row in octagon}
    require(len(octagon) == N * 8 and
            oct_keys == {(lane, 1, view, form)
                         for lane in range(N) for view in range(2) for form in range(4)} and
            all(math.isfinite(float(row["lo"])) and math.isfinite(float(row["hi"])) and
                float(row["lo"]) <= float(row["hi"]) for row in octagon),
            "octagon observer receipt incomplete, nonfinite, or reversed")
    terminals = rows("terminal_axes.csv")
    terminal_keys = {(int(row["lane"]), int(row["coord"])) for row in terminals}
    require(len(terminals) == N * 12 and
            terminal_keys == {(lane, coord) for lane in range(N) for coord in range(1, 13)} and
            all(math.isfinite(float(row[key])) for row in terminals
                for key in ("pre_lo", "pre_hi", "composed_lo", "composed_hi")) and
            all(float(row["pre_lo"]) <= float(row["pre_hi"]) and
                float(row["composed_lo"]) <= float(row["composed_hi"])
                for row in terminals),
            "terminal axis receipt incomplete, nonfinite, or reversed")

    stdout = (RUN / "stdout.log").read_text()
    server = (RUN / "server.log").read_text()
    exit_code = int((RUN / "native_exit_code.txt").read_text().strip())
    require(exit_code == 0 and stdout.count("Calling CROWN.") == 1 and
            stdout.count("RPC_COVERS_ALL_SOURCE_BOXES") == 1 and
            stdout.count("\nBOX ") == N and
            "PROPERTY_NOT_CHECKED_FIRST_PERIOD_ONLY" in stdout,
            "native exit or log markers do not match the first-step gate")
    require(server.count('"POST / HTTP/1.1" 200') == 1,
            "server did not record exactly one successful RPC request")
    result = {
        "schema": "native-quad-allbox-firststep-recenter-nohash-v1",
        "run_id": ROOT.name,
        "source_boxes": N,
        "source_split_counts_x1_to_x6": [len(axis) for axis in axes],
        "rpc_calls": len(calls),
        "rpc_input_intervals": min(len(lb), len(ub)),
        "rpc_source_omissions": len(omitted),
        "first_rpc_omission": omitted[0] if omitted else None,
        "rpc_exact_lower_bounds": exact_lower,
        "rpc_exact_upper_bounds": exact_upper,
        "invalid_coefficient_boxes": invalid_coefficients,
        "native_exit_code": exit_code,
        "attempted_boxes": len(states),
        "status_counts": dict(status_counts),
        "accepted_one_step_boxes": sum(row["accepted_steps"] == "1" for row in states),
        "first_numerical_refusal": next((int(row["lane"]) for row in states
                                          if row["status"] not in ("2", "3", "4") or
                                          row["accepted_steps"] != "1"), None),
        "saved_range_records": scan["records"],
        "range_complete_grid": scan["complete_grid"],
        "range_nonfinite_records": scan["nonfinite_records"],
        "range_reversed_component_intervals": scan["reversed_component_intervals"],
        "range_endpoint_outside_same_step_tube_components":
            scan["endpoint_outside_same_step_tube_components"],
        "range_duration_min_max": [min(durations), max(durations)] if durations else None,
        "octagon_rows": len(octagon),
        "terminal_axis_rows": len(terminals),
        "errors": problems,
        "interpretation": "first control call and first h=0.005 ODE step only, isolated copied candidate; saved numerical coverage and finite records only; no independent CROWN/NN or later-step proof",
    }
    (ROOT / "AUDIT.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: result[key] for key in (
        "attempted_boxes", "status_counts", "rpc_calls", "rpc_source_omissions",
        "saved_range_records", "range_complete_grid", "octagon_rows", "errors")}))
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
