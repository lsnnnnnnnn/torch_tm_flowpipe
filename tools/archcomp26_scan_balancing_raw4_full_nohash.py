#!/usr/bin/env python3
"""Independently check the saved Balancing raw4 full-attempt ledger."""

import argparse
import json
import math
from pathlib import Path


INITIAL = [[-0.1, 0.1], [-0.05, 0.05], [-0.1, 0.1], [-0.05, 0.05]]


def scan(directory):
    payload = directory / "payload"
    rows = [json.loads(line) for line in (payload / "ranges.jsonl").read_text().splitlines()]
    checks = [json.loads(line) for line in (payload / "property_checks.jsonl").read_text().splitlines()]
    outer = json.loads((directory / "RESULT.json").read_text())
    inner = json.loads((payload / "RESULT.json").read_text())
    config = (payload / "config.yaml").read_text()
    if ("steps: 500" not in config or "ode_step_size: 0.005" not in config
            or "constraints_safe_from: 8.0" not in config
            or "constraints_safe_until: 10.0" not in config):
        raise ValueError("full-horizon configuration or property window differs")
    if len(rows) > 2000 or len(rows) != inner["observed_substeps"]:
        raise ValueError("observed substep count differs")
    accepted = 0
    first_rejection = None
    tube_union = [[math.inf, -math.inf] for _ in INITIAL]
    last_endpoint = None
    for i, row in enumerate(rows, 1):
        if (row["substep"] != i or
                any(abs(a - b) > 1e-12 for a, b in zip(
                    row["t_interval"], [(i - 1) * 0.005, i * 0.005]))):
            raise ValueError(f"substep/time mismatch at {i}")
        if row["accepted"]:
            if first_rejection or row["solver_status_code"] != 0:
                raise ValueError(f"accepted row after rejection or inactive status at {i}")
            if len(row["tube"]) != 4 or len(row["endpoint"]) != 4:
                raise ValueError(f"physical state count mismatch at {i}")
            for j, (tube, end) in enumerate(zip(row["tube"], row["endpoint"])):
                if (len(tube) != 2 or len(end) != 2 or
                        not all(math.isfinite(v) for v in tube + end) or
                        not tube[0] <= end[0] <= end[1] <= tube[1]):
                    raise ValueError(f"invalid saved interval at {i}, state {j}")
                if i == 1 and not tube[0] <= INITIAL[j][0] <= INITIAL[j][1] <= tube[1]:
                    raise ValueError(f"initial state {j} absent from first tube")
                tube_union[j][0] = min(tube_union[j][0], tube[0])
                tube_union[j][1] = max(tube_union[j][1], tube[1])
            last_endpoint = row["endpoint"]
            accepted += 1
        else:
            first_rejection = {"substep": i, "status_code": row["solver_status_code"],
                               "status": row["solver_status"]}
            if i != len(rows) or "tube" in row or "endpoint" in row:
                raise ValueError("ledger continued or saved bounds after rejection")
    if (accepted != inner["accepted_substeps"] or
            len(checks) != inner["property_checks"] or
            inner["expected_substeps"] != 2000 or
            inner["expected_property_checks"] != 400):
        raise ValueError("saved ledger differs from inner receipt")
    if checks and accepted < 1600:
        raise ValueError("property checked before the 8 s window")
    full = accepted == 2000 and first_rejection is None
    return {
        "schema": "archcomp26-balancing-raw4-p3-full-independent-scan-nohash-v1",
        "source": str(directory), "planned_substeps": 2000,
        "observed_substeps": len(rows), "accepted_substeps": accepted,
        "first_rejection": first_rejection,
        "last_accepted_t_s": accepted * 0.005,
        "complete_numeric_horizon": full,
        "checked_property_substeps": len(checks), "planned_property_substeps": 400,
        "property_verdict": ("UNKNOWN_INCOMPLETE" if not full or len(checks) != 400
                             else inner["property_verdict"]),
        "accepted_tube_union": tube_union if accepted else None,
        "last_accepted_endpoint": last_endpoint,
        "outer_status": outer["status"], "outer_exit_code": outer["exit_code"],
        "outer_wall_s": outer["wall_s"], "inner_status": inner["status"],
        "cuda_peak_allocated_bytes": inner.get("cuda_peak_allocated_bytes"),
        "cuda_peak_reserved_bytes": inner.get("cuda_peak_reserved_bytes"),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(scan(args.directory), indent=2, allow_nan=False) + "\n")
