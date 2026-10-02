#!/usr/bin/env python3
"""Check one-box TORA reach saved intervals without replaying the experiment."""

import argparse
import json
import math
from pathlib import Path
import struct


ROW = struct.Struct("<QQd16d")
TARGET = ((-0.1, 0.2), (-0.9, -0.6))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected", type=int, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    start = json.loads((root / "START.json").read_text())
    result = json.loads((root / "RESULT.json").read_text())
    observations = [json.loads(line) for line in (root / "observations.jsonl").read_text().splitlines()]
    raw = (root / "ranges.bin").read_bytes()
    if len(raw) % ROW.size:
        raise ValueError("truncated interval row")
    valid_count = 0
    for step, item in enumerate(observations, 1):
        if item.get("substep") != step:
            raise ValueError("nonconsecutive observation")
        if item.get("accepted") and item.get("interval_valid") is True:
            valid_count += 1
        elif step != len(observations):
            raise ValueError("observation after first numerical refusal")
    if len(raw) != valid_count * ROW.size:
        raise ValueError("saved row count differs from valid observations")
    if result.get("observed_substeps") != len(observations):
        raise ValueError("RESULT observation count differs")

    union = [[math.inf, -math.inf] for _ in range(4)]
    last_endpoint = None
    target_contained_endpoint_steps = []
    endpoint_outside_tube = 0
    maximum_endpoint_tube_excess = 0.0
    for index in range(valid_count):
        lane, step, h, *flat = ROW.unpack_from(raw, index * ROW.size)
        if lane != 0 or step != index + 1 or not math.isclose(h, 0.01, abs_tol=1e-12):
            raise ValueError("unexpected lane, step, or step size")
        endpoint = []
        for coordinate in range(4):
            lo, hi, end_lo, end_hi = flat[4 * coordinate:4 * coordinate + 4]
            if not (all(map(math.isfinite, (lo, hi, end_lo, end_hi))) and
                    lo <= hi and end_lo <= end_hi):
                raise ValueError(f"invalid interval at step {step}, state {coordinate + 1}")
            union[coordinate][0] = min(union[coordinate][0], lo)
            union[coordinate][1] = max(union[coordinate][1], hi)
            endpoint.append([end_lo, end_hi])
            excess = max(lo - end_lo, end_hi - hi, 0.0)
            endpoint_outside_tube += excess > 0
            maximum_endpoint_tube_excess = max(maximum_endpoint_tube_excess, excess)
        last_endpoint = endpoint
        if all(TARGET[i][0] <= endpoint[i][0] <= endpoint[i][1] <= TARGET[i][1]
               for i in range(2)):
            target_contained_endpoint_steps.append(step)

    full = valid_count == args.expected and result.get("full_horizon_completed") is True
    receipt = {
        "profile": start["profile"],
        "kind": "independent_saved_numeric_interval_scan_only",
        "expected_substeps": args.expected,
        "observed_substeps": len(observations),
        "valid_saved_substeps": valid_count,
        "first_refusal_substep": len(observations) if len(observations) > valid_count else None,
        "all_saved_rows_finite_ordered": True,
        "tube_union_by_state": dict(zip(("x1", "x2", "x3", "x4"), union)) if valid_count else None,
        "last_saved_endpoint_by_state": dict(zip(("x1", "x2", "x3", "x4"), last_endpoint)) if valid_count else None,
        "target_contained_saved_endpoint_steps": target_contained_endpoint_steps,
        "endpoint_outside_same_step_tube_components": endpoint_outside_tube,
        "maximum_endpoint_tube_excess": maximum_endpoint_tube_excess,
        "full_numerical_horizon_completed": full,
        "property_checker_run": False,
        "content_digest_computed": False,
        "qualification": "Saved numeric intervals do not establish end-to-end floating NNCS soundness.",
    }
    with (root / "INDEPENDENT_INTERVAL_SCAN.json").open("x") as output:
        json.dump(receipt, output, indent=2, allow_nan=False)
        output.write("\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
