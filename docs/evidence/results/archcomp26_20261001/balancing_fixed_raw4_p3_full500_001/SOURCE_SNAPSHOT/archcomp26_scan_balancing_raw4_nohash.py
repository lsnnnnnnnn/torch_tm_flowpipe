#!/usr/bin/env python3
"""Independently scan one Balancing first-period saved range ledger."""

import argparse
import json
import math
from pathlib import Path


def scan(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) > 4:
        raise ValueError("first-period ledger exceeds four substeps")
    accepted = 0
    first_reject = None
    endpoint = None
    tube_union = [[math.inf, -math.inf] for _ in range(4)]
    for index, row in enumerate(rows, 1):
        if row["substep"] != index or row["t_interval"] != [(index - 1) * 0.005, index * 0.005]:
            raise ValueError(f"substep/time ledger mismatch at {index}")
        if row["accepted"]:
            if first_reject is not None or row["solver_status_code"] != 0:
                raise ValueError("accepted substep has inactive solver status")
            if len(row["tube"]) != 4 or len(row["endpoint"]) != 4:
                raise ValueError("physical state count differs")
            for state, (tube, end) in enumerate(zip(row["tube"], row["endpoint"])):
                if (len(tube) != 2 or len(end) != 2 or
                    not all(math.isfinite(value) for value in tube + end) or
                    not tube[0] <= end[0] <= end[1] <= tube[1]):
                    raise ValueError(f"invalid tube/endpoint at step {index}, state {state}")
                tube_union[state][0] = min(tube_union[state][0], tube[0])
                tube_union[state][1] = max(tube_union[state][1], tube[1])
            endpoint = row["endpoint"]
            accepted += 1
        else:
            first_reject = {"substep": index, "solver_status_code": row["solver_status_code"],
                            "solver_status": row["solver_status"]}
            if index != len(rows):
                raise ValueError("ledger advanced after first rejection")
    return {
        "schema": "balancing-raw4-p3-first-period-independent-scan-v1",
        "source": str(path), "expected_substeps": 4, "observed_substeps": len(rows),
        "accepted_substeps": accepted, "full_first_period_numeric": accepted == 4,
        "first_rejection": first_reject,
        "accepted_tube_union": tube_union if accepted else None,
        "last_accepted_endpoint": endpoint,
        "property_status": "NOT_APPLICABLE: [8,10] s window outside [0,0.02] s",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ranges", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(scan(args.ranges), indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
