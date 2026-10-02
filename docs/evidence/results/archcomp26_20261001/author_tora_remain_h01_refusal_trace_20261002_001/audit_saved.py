#!/usr/bin/env python3
"""Audit saved TORA first-refusal traces by direct comparison, without digests."""

import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
OLD = HERE.parent / "author_tora_remain_v1/huan_full20_001/payload"
FIELDS = ("substep", "accepted_count", "rejected_lanes", "tube_inside_safe_for_accepted")


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def compare(name, expected_steps):
    payload = HERE / name / "payload"
    old_raw = (OLD / "ranges.bin").read_bytes()
    new_raw = (payload / "ranges.bin").read_bytes()
    old_obs = lines(OLD / "observations.jsonl")
    new_obs = lines(payload / "observations.jsonl")
    assert len(new_obs) == expected_steps
    assert old_raw[:len(new_raw)] == new_raw
    assert all(tuple(old[k] for k in FIELDS) == tuple(new[k] for k in FIELDS)
               for old, new in zip(old_obs, new_obs))
    return {"observed_steps": len(new_obs), "range_prefix_direct_bytes_equal": True,
            "observation_fields_equal": True}


def main():
    failed = compare("huan_step190_trace_001", 189)
    successful = compare("huan_step190_trace_002", 190)
    trace = lines(HERE / "huan_step190_trace_002/payload/refinement_step190.jsonl")
    assert len(trace) == 12 and sorted(row["lane"] for row in trace) == list(range(12))
    assert all(row["substep"] == 190 for row in trace)
    misses = []
    for row in trace:
        for component, (guess, proposal, included, lower, upper) in enumerate(zip(
                row["input_remainder"], row["proposal_interval"],
                row["subset_by_component"], row["lower_margin"], row["upper_margin"])):
            assert all(math.isfinite(value) for value in (*guess, *proposal, lower, upper))
            assert included == (guess[0] <= proposal[0] and proposal[1] <= guess[1])
            assert math.isclose(lower, proposal[0] - guess[0], abs_tol=1e-15)
            assert math.isclose(upper, guess[1] - proposal[1], abs_tol=1e-15)
            if not included:
                misses.append({"lane": row["lane"], "component": component,
                               "guess": guess, "proposal": proposal,
                               "lower_margin": lower, "upper_margin": upper})
    assert len(misses) == 1 and misses[0]["lane"] == 2 and misses[0]["component"] == 1
    result = {"schema": "archcomp26-tora-remain-h01-first-refusal-direct-audit-v1",
              "old_run": str(OLD.parent),
              "failed_hook_attempt": failed,
              "read_only_glue_trace_attempt": successful,
              "validpost_records": len(trace), "initial_self_map_misses": misses,
              "first_rejection": json.loads((HERE / "huan_step190_trace_002/payload/RESULT.json").read_text())["first_rejection"],
              "content_digest_computed": False}
    (HERE / "AUDIT.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
