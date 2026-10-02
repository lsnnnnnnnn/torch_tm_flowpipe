#!/usr/bin/env python3
"""Check the isolated TORA x2 remainder profile against saved runs."""

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / "author_tora_remain_v1"
EXPECTED_NEW_REMAINDER = (
    "- - -0.01\n  - 0.01\n- - -0.02\n  - 0.02\n"
    "- - -0.01\n  - 0.01\n- - -0.01\n  - 0.01\n"
    "- - -0.01\n  - 0.01\n- - -0.01\n  - 0.01\n"
)


def without_remainder(path):
    before, following = path.read_text().split("remainder_estimation:\n", 1)
    remainder, after = following.split("symbolic_remainder_queue:", 1)
    return before + "symbolic_remainder_queue:" + after, remainder


def audit(new_name, prior_name, steps, accepted, rejected):
    new = HERE / new_name
    old = PRIOR / prior_name
    new_other, new_remainder = without_remainder(new / "payload/config.yaml")
    old_other, old_remainder = without_remainder(old / "payload/config.yaml")
    assert new_other == old_other
    assert old_remainder == "- -0.01\n- 0.01\n"
    assert new_remainder == EXPECTED_NEW_REMAINDER
    result = json.loads((new / "payload/RESULT.json").read_text())
    scan = json.loads((new / "INDEPENDENT_INTERVAL_SCAN.json").read_text())
    assert result["observed_substeps"] == scan["observed_substeps"] == steps
    assert result["accepted_lane_substeps"] == scan["accepted_lane_substeps"] == accepted
    assert scan["record_count"] == steps * 12
    assert scan["first_rejected_substep"] == rejected
    assert scan["endpoint_outside_same_step_tube_components"] == 0
    return {"run": new_name, "config_other_fields_direct_text_equal": True,
            "prior_uniform_remainder": [-0.01, 0.01],
            "new_x2_remainder": [-0.02, 0.02],
            "observed_substeps": steps, "accepted_lane_substeps": accepted,
            "saved_range_records": scan["record_count"],
            "first_rejected_substep": rejected,
            "first_saved_tube_outside_safe_substep": scan["first_saved_tube_outside_safe_substep"],
            "result_status": result["status"]}


def main():
    smoke = audit("huan_smoke1_firstreject_001", "huan_smoke1_001", 10, 120, None)
    full = audit("huan_full20_firstreject_001", "huan_full20_001", 192, 2303, 192)
    (HERE / "AUDIT.json").write_text(json.dumps({
        "schema": "archcomp26-tora-remain-h01-x2rem002-saved-audit-v1",
        "smoke": smoke, "first_refusal_full_attempt": full,
        "content_digest_computed": False
    }, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
