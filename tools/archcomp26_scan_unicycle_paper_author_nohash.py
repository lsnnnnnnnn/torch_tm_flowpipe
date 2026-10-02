#!/usr/bin/env python3
"""Scan saved Unicycle author tubes independently of the experiment process."""

import argparse
import json
import math
from pathlib import Path


INITIAL = [[9.5, 9.55], [-4.5, -4.45], [2.1, 2.11], [1.5, 1.51],
           [-0.0001, 0.0001], [0, 0], [0, 0], [0, 0]]
RHS = ["x4 * cos(x3)", "x4 * sin(x3)", "u2 - 20", "u1 + w - 20",
       "0", "1", "0", "0"]
TARGET = [[-0.6, 0.6], [-0.2, 0.2], [-0.06, 0.06], [-0.3, 0.3]]


def scan(run):
    payload = run / "payload"
    outer = json.loads((run / "RESULT.json").read_text())
    result = json.loads((payload / "RESULT.json").read_text())
    start = json.loads((payload / "START.json").read_text())
    rows = [json.loads(line) for line in (payload / "ranges.jsonl").read_text().splitlines()]
    config = (payload / "config.yaml").read_text()
    mode = result["mode"]
    planned = 10 if mode == "smoke1" else 500 if mode == "full" else None
    if (planned is None or outer["status"] != "completed" or
            outer["exit_code"] != 0 or
            result["method"] not in ("huan", "xiangru") or
            start["method"] != result["method"] or start["mode"] != mode or
            start["profile"] != "unicycle-paper-speed-w-constant-v1" or
            start["initial_set"] != INITIAL or start["dynamics_expressions"] != RHS or
            start["target"] != TARGET or start["period_s"] != 0.2 or
            start["ode_step_s"] != 0.02 or start["controller_periods"] != planned // 10 or
            start["model_preflight"]["input_shape"] != [1, 1, 1, 4] or
            start["model_preflight"]["output_shape"] != [1, 2] or
            f"steps: {planned // 10}" not in config or
            "ode_step_size: 0.02" not in config or
            result["expected_substeps"] != planned or len(rows) != result["observed_substeps"]):
        raise ValueError("saved receipts disagree with the paper-speed Unicycle contract")
    if mode == "smoke1" and "constraints_target:" in config:
        raise ValueError("one-period smoke includes a full-horizon target checker")

    first_rejection = None
    last_endpoint = None
    endpoint_outside_tube = 0
    largest_endpoint_tube_excess = 0.0
    tube_union = [[math.inf, -math.inf] for _ in INITIAL]
    accepted = 0
    for step, row in enumerate(rows, 1):
        if (row["substep"] != step or len(row["t_interval"]) != 2 or
                any(abs(actual - expected) > 1e-12 for actual, expected in zip(
                    row["t_interval"], [(step - 1) * 0.02, step * 0.02]))):
            raise ValueError(f"nonconsecutive or mistimed step {step}")
        if not row["accepted"]:
            if (first_rejection is not None or step != len(rows) or
                    "tube" in row or "endpoint" in row):
                raise ValueError("invalid rejection ledger")
            first_rejection = {"substep": step, "solver_status": row["solver_status"]}
            continue
        if first_rejection or len(row["tube"]) != 8 or len(row["endpoint"]) != 8:
            raise ValueError(f"invalid accepted state at {step}")
        for state, (tube, endpoint) in enumerate(zip(row["tube"], row["endpoint"])):
            if (len(tube) != 2 or len(endpoint) != 2 or
                    not all(math.isfinite(v) for v in tube + endpoint) or
                    tube[0] > tube[1] or endpoint[0] > endpoint[1]):
                raise ValueError(f"invalid interval at step {step}, state {state}")
            # Control outputs are assigned at t=0 before the first flowpipe.
            if step == 1 and state < 6 and not tube[0] <= INITIAL[state][0] <= INITIAL[state][1] <= tube[1]:
                raise ValueError(f"first tube omits initial state {state}")
            tube_union[state][0] = min(tube_union[state][0], tube[0])
            tube_union[state][1] = max(tube_union[state][1], tube[1])
            excess = max(tube[0] - endpoint[0], endpoint[1] - tube[1], 0.0)
            if excess:
                endpoint_outside_tube += 1
                largest_endpoint_tube_excess = max(largest_endpoint_tube_excess, excess)
        accepted += 1
        last_endpoint = row["endpoint"]
    if accepted != result["accepted_substeps"] or len(rows) > planned:
        raise ValueError("saved range count differs from receipt")
    full = mode == "full" and accepted == planned and first_rejection is None
    target_contains_endpoint = full and all(
        target[0] <= endpoint[0] <= endpoint[1] <= target[1]
        for endpoint, target in zip(last_endpoint[:4], TARGET))
    if (result["terminal_endpoint_in_target"] != target_contains_endpoint or
            (full and last_endpoint[:4] != result["terminal_physical_endpoint"])):
        raise ValueError("saved endpoint disagrees with author result")
    return {
        "schema": "archcomp26-unicycle-paper-author-independent-scan-nohash-v1",
        "source_run_dir": str(run), "method": result["method"], "mode": mode,
        "observed_substeps": len(rows), "accepted_substeps": accepted,
        "first_rejection": first_rejection, "complete_numerical_horizon": full,
        "saved_tube_union": tube_union if accepted else None,
        "last_saved_endpoint": last_endpoint,
        "endpoint_outside_same_step_tube_components": endpoint_outside_tube,
        "maximum_endpoint_tube_excess": largest_endpoint_tube_excess,
        "terminal_endpoint_in_target": target_contains_endpoint,
        "property_verdict": ("ENDPOINT_SUFFICIENT_FOR_REACH" if target_contains_endpoint
                             else "UNKNOWN" if full else "NOT_APPLICABLE_SHORT_PREFIX"),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    output = args.run_dir / "INDEPENDENT_INTERVAL_SCAN.json"
    with output.open("x") as file:
        json.dump(scan(args.run_dir.resolve()), file, indent=2, allow_nan=False)
        file.write("\n")
    print(output)
