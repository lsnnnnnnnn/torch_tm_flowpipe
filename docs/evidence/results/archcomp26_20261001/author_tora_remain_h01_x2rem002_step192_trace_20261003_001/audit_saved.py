#!/usr/bin/env python3
"""Compare a saved step-192 trace directly with the prior x2=.02 run."""

import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / "author_tora_remain_h01_x2rem002_20261002_001/huan_full20_firstreject_001/payload"
NEW = HERE / "huan_step192_trace_001/payload"
STEPS = 192
LANES = 12
RECORD_BYTES = 8 + 8 + 8 + 4 * 4 * 8


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    outer_start = json.loads((NEW.parent / "START.json").read_text())
    outer_result = json.loads((NEW.parent / "RESULT.json").read_text())
    assert outer_start["instance"] == "tora-remain"
    assert outer_start["method"] == "huan"
    assert outer_start["timeout_s"] == 120
    assert outer_start["selected_environment"]["CUDA_VISIBLE_DEVICES"] == "3"
    assert outer_result["status"] == "failed"
    assert outer_result["exit_code"] == 1 and not outer_result["timed_out"]
    old_start = json.loads((PRIOR / "START.json").read_text())
    new_start = json.loads((NEW / "START.json").read_text())
    for key in ("backend", "mode", "source_config", "boxes", "controller", "engine",
                "shared_driver", "controller_boundary", "method", "numerical_profile", "property"):
        assert old_start[key] == new_start[key], key
    assert (PRIOR / "config.yaml").read_text() == (NEW / "config.yaml").read_text()
    old_result = json.loads((PRIOR / "RESULT.json").read_text())
    new_result = json.loads((NEW / "RESULT.json").read_text())
    assert new_result["status"] == "stopped_first_numerical_rejection"
    assert new_result["probe_max_observed_substeps"] == STEPS
    assert new_result["first_rejection"] == old_result["first_rejection"]
    old_obs = rows(PRIOR / "observations.jsonl")
    new_obs = rows(NEW / "observations.jsonl")
    assert len(old_obs) == len(new_obs) == STEPS
    assert old_obs == new_obs
    old_raw = (PRIOR / "ranges.bin").read_bytes()
    new_raw = (NEW / "ranges.bin").read_bytes()
    assert len(old_raw) == len(new_raw) == STEPS * LANES * RECORD_BYTES
    assert old_raw == new_raw

    trace = rows(NEW / "refinement_step192.jsonl")
    assert len(trace) == LANES and sorted(row["lane"] for row in trace) == list(range(LANES))
    assert all(row["substep"] == STEPS for row in trace)
    expected_guess = [[-0.01, 0.01], [-0.02, 0.02], [-0.01, 0.01],
                      [-0.01, 0.01], [-0.01, 0.01], [-0.01, 0.01]]
    misses = []
    for row in trace:
        assert row["input_remainder"] == expected_guess
        assert len(row["proposal_interval"]) == len(row["subset_by_component"]) == 6
        for component, (guess, proposal, included, lower, upper, difference) in enumerate(zip(
                row["input_remainder"], row["proposal_interval"], row["subset_by_component"],
                row["lower_margin"], row["upper_margin"], row["difference_bound"])):
            assert all(math.isfinite(value) for value in (*guess, *proposal, *difference, lower, upper))
            assert proposal[0] <= proposal[1]
            assert included == (guess[0] <= proposal[0] and proposal[1] <= guess[1])
            assert math.isclose(lower, proposal[0] - guess[0], abs_tol=1e-15)
            assert math.isclose(upper, guess[1] - proposal[1], abs_tol=1e-15)
            if not included:
                misses.append({"lane": row["lane"], "component": component,
                               "guess": guess, "proposal": proposal,
                               "lower_margin": lower, "upper_margin": upper})
    result = {"schema": "archcomp26-tora-remain-h01-x2rem002-step192-direct-audit-v1",
              "prior_run": str(PRIOR.parent), "new_run": str(NEW.parent),
              "outer_status": outer_result["status"],
              "outer_exit_code": outer_result["exit_code"],
              "outer_timed_out": outer_result["timed_out"],
              "matched_config_text": True, "matched_192_observations": True,
              "matched_2304_range_records_directly": True,
              "first_rejection": new_result["first_rejection"],
              "validpost_records": len(trace), "initial_self_map_misses": misses}
    (HERE / "AUDIT.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
