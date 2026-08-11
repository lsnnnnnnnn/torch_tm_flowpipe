import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments/run_s1_corrected_frozen_prefix.py"
SPEC = importlib.util.spec_from_file_location("run_s1_corrected_frozen_prefix_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
corrected = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(corrected)

SCHEDULE = (
    ROOT
    / "outputs/s1_prefix_integrated_complete_o4_20260810/20260810T095423Z"
    / "04_frozen_schedule_prefix/frozen_schedule.json"
)


def test_corrected_mode_uses_fixed_historical_accepted_steps_and_separate_diagnostics(
    tmp_path,
):
    output = tmp_path / "corrected"
    summary = corrected.run(SCHEDULE, output, max_boundaries=3)
    assert summary["outcome"] == "TEST_BOUNDARY_LIMIT_REACHED"
    assert summary["accepted_step_count"] == 3
    assert summary["processed_row_count"] == 3
    assert summary["all_attempted_diagnostics_immutable"]
    assert summary["all_candidate_gates_pass"]
    rows = [
        json.loads(line)
        for line in (output / "accepted_step_records.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(rows) == 3
    for row in rows:
        fixed = row["frozen_accepted_step"]
        assert fixed["decision"] == "accepted"
        assert fixed["h"]["hex"] == fixed["h_min_hex"] == fixed["h_max_hex"]
        assert fixed["returned_h_hex"] == fixed["h"]["hex"]
        assert fixed["step_rejections"] == 0
        assert row["historical_attempted_step"]["prestate_unchanged"]
        assert row["candidate_gates"]["passed"]
