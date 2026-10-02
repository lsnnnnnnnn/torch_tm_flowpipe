#!/usr/bin/env python3
"""Independently scan saved author Balancing raw4 flowpipe records."""

import argparse
import json
import math
from pathlib import Path


INITIAL = [[-0.1, 0.1], [-0.05, 0.05], [-0.1, 0.1], [-0.05, 0.05]]


def scan(path):
    result = json.loads((path / "RESULT.json").read_text())
    start = json.loads((path / "START.json").read_text())
    config = (path / "config.yaml").read_text()
    rows = [json.loads(line) for line in (path / "ranges.jsonl").read_text().splitlines()]
    checks = [json.loads(line) for line in (path / "property_checks.jsonl").read_text().splitlines()]
    mode = result["mode"]
    planned = 4 if mode == "smoke1" else 2000
    if (result["method"] != "xiangru" or start["method"] != "xiangru" or
            result["profile"] != "balancing-fixed-repo-raw4" or
            start["initial_physical_box"] != INITIAL or
            start["period_s"] != 0.02 or start["ode_step_s"] != 0.005 or
            start["ode_order"] != 6 or start["paper_feature5_contract"] or
            f"steps: {planned // 4}" not in config or
            "ode_step_size: 0.005" not in config or
            result["expected_substeps"] != planned or len(rows) != result["observed_substeps"]):
        raise ValueError("saved method, initial box, source profile, or time horizon differs")
    if mode == "smoke1":
        if "constraints_safe:" in config or checks:
            raise ValueError("short prefix must not claim the 8–10 s property")
    elif ("constraints_safe_from: 8.0" not in config or
          "constraints_safe_until: 10.0" not in config):
        raise ValueError("full attempt lacks the saved 8–10 s property window")

    accepted = 0
    first_rejection = None
    tube_union = [[math.inf, -math.inf] for _ in INITIAL]
    last_endpoint = None
    endpoint_outside_tube = []
    for index, row in enumerate(rows, 1):
        if (row["substep"] != index or len(row["t_interval"]) != 2 or
                any(abs(a - b) > 1e-12 for a, b in zip(
                    row["t_interval"], [(index - 1) * 0.005, index * 0.005]))):
            raise ValueError(f"invalid step/time index {index}")
        if row["accepted"]:
            if first_rejection or row["solver_status_code"] != 0:
                raise ValueError(f"accepted row follows rejection at {index}")
            if len(row["tube"]) != 4 or len(row["endpoint"]) != 4:
                raise ValueError(f"physical state count differs at {index}")
            for state, (tube, endpoint) in enumerate(zip(row["tube"], row["endpoint"])):
                if (len(tube) != 2 or len(endpoint) != 2 or
                        not all(math.isfinite(value) for value in tube + endpoint) or
                        tube[0] > tube[1] or endpoint[0] > endpoint[1]):
                    raise ValueError(f"nonfinite or inverted tube/endpoint at {index}, state {state}")
                for side, bound, value, outside in (("lower", tube[0], endpoint[0], endpoint[0] < tube[0]),
                                                    ("upper", tube[1], endpoint[1], endpoint[1] > tube[1])):
                    if outside:
                        delta = abs(value - bound)
                        endpoint_outside_tube.append({"substep": index, "state": state,
                                                      "side": side, "tube_bound": bound,
                                                      "endpoint_bound": value, "delta": delta,
                                                      "delta_in_tube_bound_ulps": delta / math.ulp(bound)})
                if index == 1 and not tube[0] <= INITIAL[state][0] <= INITIAL[state][1] <= tube[1]:
                    raise ValueError(f"initial state {state} absent from first tube")
                tube_union[state][0] = min(tube_union[state][0], tube[0])
                tube_union[state][1] = max(tube_union[state][1], tube[1])
            last_endpoint = row["endpoint"]
            accepted += 1
        else:
            first_rejection = {"substep": index, "status_code": row["solver_status_code"],
                               "status": row["solver_status"]}
            if index != len(rows) or "tube" in row or "endpoint" in row:
                raise ValueError("invalid ledger after first rejection")
    if (accepted != result["accepted_substeps"] or len(checks) != result["property_checks"] or
            (checks and accepted < 1600) or len(rows) > planned):
        raise ValueError("saved ledger and result receipt disagree")
    complete = accepted == planned and first_rejection is None
    return {
        "schema": "archcomp26-balancing-raw4-xiangru-independent-scan-nohash-v1",
        "source": str(path), "mode": mode, "planned_substeps": planned,
        "observed_substeps": len(rows), "accepted_substeps": accepted,
        "first_rejection": first_rejection, "last_accepted_t_s": accepted * 0.005,
        "complete_numeric_horizon": complete, "property_checks": len(checks),
        "property_verdict": ("NOT_APPLICABLE_SHORT_PREFIX" if mode == "smoke1" else
                             result["property_verdict"] if complete and len(checks) == 400 else
                             "UNKNOWN_INCOMPLETE"),
        "accepted_tube_union": tube_union if accepted else None,
        "last_accepted_endpoint": last_endpoint, "raw_status": result["status"],
        "raw_wall_s": result["wall_s"],
        "strict_saved_tube_contains_endpoint": not endpoint_outside_tube,
        "endpoint_outside_tube_count": len(endpoint_outside_tube),
        "endpoint_outside_tube_first": endpoint_outside_tube[0] if endpoint_outside_tube else None,
        "endpoint_outside_tube_max_absolute": max((x["delta"] for x in endpoint_outside_tube), default=0.0),
        "endpoint_outside_tube_max_tube_bound_ulps": max(
            (x["delta_in_tube_bound_ulps"] for x in endpoint_outside_tube), default=0.0),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(scan(args.directory), indent=2, allow_nan=False) + "\n")
