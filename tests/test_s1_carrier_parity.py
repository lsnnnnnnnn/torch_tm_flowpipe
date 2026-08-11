import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments/audit_s1_boundary_drift.py"
SPEC = importlib.util.spec_from_file_location("audit_s1_boundary_drift_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)

SCHEDULE = (
    ROOT
    / "outputs/s1_prefix_integrated_complete_o4_20260810/20260810T095423Z"
    / "04_frozen_schedule_prefix/frozen_schedule.json"
)


def test_c1_shadow_and_c2_carrier_are_bit_exact_for_twenty_boundaries():
    schedule = json.loads(SCHEDULE.read_text(encoding="utf-8"))
    c0 = audit.replay_control("C0", schedule, max_attempt_index=19)
    c1 = audit.replay_control("C1", schedule, max_attempt_index=19)
    c2 = audit.replay_control("C2", schedule, max_attempt_index=19)
    assert len(c0["states"]) == len(c1["states"]) == len(c2["states"]) == 21
    assert audit._exact_state_sequence(c1) == audit._exact_state_sequence(c0)
    assert audit._attempt_decision_sequence(c1) == audit._attempt_decision_sequence(c0)
    assert audit._exact_state_sequence(c2) == audit._exact_state_sequence(c0)
    assert all(
        row["carrier_same_set_relation"] in {"not_applicable", "equal"}
        for row in c2["states"]
    )


def test_c3_no_renormalization_fails_the_exact_domain_gate_without_commit():
    schedule = json.loads(SCHEDULE.read_text(encoding="utf-8"))
    c3 = audit.replay_control("C3", schedule, max_attempt_index=12)
    assert c3["status"] == "domain_gate_failure"
    assert c3["failure_boundary"] == 11
    assert len(c3["states"]) == 12
    assert "normalized right-map total leaves [-1,1]" in c3["attempts"][-1]["message"]
    assert c3["attempts"][-1]["decision"] == "rejected"
