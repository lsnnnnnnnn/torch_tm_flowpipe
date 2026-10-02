#!/usr/bin/env python3
"""Independently scan one new TORA author's saved box ranges without hashing."""

import argparse
import json
import math
from pathlib import Path
import struct


RECORD = struct.Struct("<QQd16d")  # lane, substep, h, four (tube lo/hi, endpoint lo/hi)


def scan(run_dir, expected_h=0.1):
    if not math.isfinite(expected_h) or expected_h <= 0:
        raise ValueError("expected ODE step must be positive and finite")
    observations = [json.loads(line) for line in
                    (run_dir / "payload/observations.jsonl").read_text().splitlines()]
    raw = (run_dir / "payload/ranges.bin").read_bytes()
    if len(raw) % RECORD.size:
        raise ValueError("truncated range record")
    if len(raw) != 12 * len(observations) * RECORD.size:
        raise ValueError("range count differs from 12 times observed substeps")
    rejected = {row["substep"]: set(row["rejected_lanes"]) for row in observations}
    if [row["substep"] for row in observations] != list(range(1, len(observations) + 1)):
        raise ValueError("nonconsecutive observation steps")
    accepted_count = 0
    first_rejection = None
    first_saved_tube_outside = None
    first_not_all_accepted_and_safe = None
    invalid_accepted = 0
    endpoint_outside_tube_components = 0
    maximum_endpoint_tube_excess = 0.0
    tube_union = [[math.inf, -math.inf] for _ in range(4)]
    last_endpoint_union = None
    for step in range(1, len(observations) + 1):
        last_endpoint_union = [[math.inf, -math.inf] for _ in range(4)]
        for lane in range(12):
            offset = ((step - 1) * 12 + lane) * RECORD.size
            row = RECORD.unpack_from(raw, offset)
            if (row[0], row[1]) != (lane, step) or row[2] != expected_h:
                raise ValueError(f"unexpected lane/step/h at {lane}/{step}")
            if lane in rejected[step]:
                if first_rejection is None:
                    first_rejection = step
                continue
            accepted_count += 1
            for i in range(4):
                tlo, thi, elo, ehi = row[3 + i * 4:7 + i * 4]
                if not (all(math.isfinite(x) for x in (tlo, thi, elo, ehi))
                        and tlo <= thi and elo <= ehi):
                    invalid_accepted += 1
                    continue
                tube_union[i][0] = min(tube_union[i][0], tlo)
                tube_union[i][1] = max(tube_union[i][1], thi)
                last_endpoint_union[i][0] = min(last_endpoint_union[i][0], elo)
                last_endpoint_union[i][1] = max(last_endpoint_union[i][1], ehi)
                if tlo < -2 or thi > 2:
                    if first_saved_tube_outside is None:
                        first_saved_tube_outside = step
                excess = max(tlo - elo, ehi - thi, 0.0)
                if excess > 0:
                    endpoint_outside_tube_components += 1
                    maximum_endpoint_tube_excess = max(maximum_endpoint_tube_excess, excess)
        if (first_not_all_accepted_and_safe is None and
                (first_rejection == step or first_saved_tube_outside == step)):
            first_not_all_accepted_and_safe = step
    if accepted_count != sum(row["accepted_count"] for row in observations):
        raise ValueError("accepted count differs from observations")
    if invalid_accepted:
        raise ValueError(f"invalid accepted ranges: {invalid_accepted}")
    return {
        "schema": "archcomp26-tora-ranges-scan-nohash-v1",
        "source_run_dir": str(run_dir),
        "expected_ode_step_s": expected_h,
        "record_count": len(raw) // RECORD.size,
        "observed_substeps": len(observations),
        "accepted_lane_substeps": accepted_count,
        "first_rejected_substep": first_rejection,
        "first_saved_tube_outside_safe_substep": first_saved_tube_outside,
        "all_accepted_and_saved_tube_safe_prefix_substeps":
            (first_not_all_accepted_and_safe - 1 if first_not_all_accepted_and_safe
             else len(observations)),
        "accepted_observed_tube_union": tube_union if accepted_count else None,
        "last_observed_accepted_endpoint_union": last_endpoint_union,
        "endpoint_outside_same_step_tube_components": endpoint_outside_tube_components,
        "maximum_endpoint_tube_excess": maximum_endpoint_tube_excess,
        "content_digest_computed": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--expected-h", type=float, default=0.1)
    args = parser.parse_args()
    result = scan(args.run_dir.resolve(), args.expected_h)
    output = args.run_dir / "INDEPENDENT_INTERVAL_SCAN.json"
    with output.open("x") as out:
        json.dump(result, out, indent=2, allow_nan=False)
        out.write("\n")
    print(output)


if __name__ == "__main__":
    main()
