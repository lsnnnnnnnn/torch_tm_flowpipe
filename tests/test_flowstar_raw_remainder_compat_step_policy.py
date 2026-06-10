from __future__ import annotations

import importlib.util
from pathlib import Path

from torch_tm_flowpipe.flowpipe import FLOWSTAR_COMPAT_STEP_GROW, FLOWSTAR_COMPAT_STEP_SHRINK

ROOT = Path(__file__).resolve().parents[1]
ATTEMPT_SCRIPT = ROOT / "experiments" / "flowstar_raw_remainder_compat_attempt_sequence.py"
STEP_POLICY_SCRIPT = ROOT / "experiments" / "flowstar_raw_remainder_compat_step_policy.py"

attempt_spec = importlib.util.spec_from_file_location("flowstar_raw_remainder_compat_attempt_sequence", ATTEMPT_SCRIPT)
assert attempt_spec is not None and attempt_spec.loader is not None
attempt = importlib.util.module_from_spec(attempt_spec)
attempt_spec.loader.exec_module(attempt)

step_spec = importlib.util.spec_from_file_location("flowstar_raw_remainder_compat_step_policy", STEP_POLICY_SCRIPT)
assert step_spec is not None and step_spec.loader is not None
step_policy = importlib.util.module_from_spec(step_spec)
step_spec.loader.exec_module(step_policy)

CSV_OUTPUTS = [
    ROOT / "outputs" / "flowstar_raw_remainder_compat_attempt_sequence" / "attempt_sequence_ledger.csv",
    ROOT / "outputs" / "flowstar_raw_remainder_compat_short_horizon" / "short_horizon_summary.csv",
    ROOT / "outputs" / "flowstar_raw_remainder_compat_step_policy" / "step_policy_summary.csv",
]

MD_OUTPUTS = [
    ROOT / "outputs" / "flowstar_raw_remainder_compat_attempt_sequence" / "attempt_sequence_report.md",
    ROOT / "outputs" / "flowstar_raw_remainder_compat_short_horizon" / "short_horizon_report.md",
    ROOT / "outputs" / "flowstar_raw_remainder_compat_step_policy" / "step_policy_report.md",
]


def test_default_solver_behavior_unchanged_by_step_policy_option():
    _diag, seg = attempt.run_torch_attempt("current_no_queue", 0.025)

    assert seg.status == "validated"
    assert seg.next_h == 0.025


def test_raw_remainder_compat_remains_opt_in():
    current_diag, current_seg = attempt.run_torch_attempt("current_no_queue", 0.025)
    compat_diag, compat_seg = attempt.run_torch_attempt("flowstar_raw_remainder_compat", 0.025)

    assert current_seg.status == "validated"
    assert compat_seg.status == "failed"
    assert current_diag["flowstar_raw_remainder_compat_enabled"] is False
    assert compat_diag["flowstar_raw_remainder_compat_enabled"] is True


def test_flowstar_step_policy_uses_shrink_0_5_on_failed_attempts():
    assert FLOWSTAR_COMPAT_STEP_SHRINK == 0.5
    assert step_policy.flowstar_policy_next_after_failure(0.1) == 0.05
    assert step_policy.flowstar_policy_next_after_failure(0.025) == 0.0125


def test_flowstar_step_policy_uses_audited_grow_rule_on_success():
    assert FLOWSTAR_COMPAT_STEP_GROW == 1.1
    assert abs(step_policy.flowstar_policy_next_after_success(0.0125) - 0.01375) <= 1e-15
    assert step_policy.flowstar_policy_next_after_success(0.1) == 0.1


def test_checked_in_csv_and_markdown_outputs_have_physical_lines():
    for path in CSV_OUTPUTS:
        assert path.exists(), path
        text = path.read_text(encoding="utf-8")
        assert len(text.splitlines()) > 5, path
        assert "\n" in text, path

    for path in MD_OUTPUTS:
        assert path.exists(), path
        lines = path.read_text(encoding="utf-8").splitlines()
        assert any(line.startswith("# ") for line in lines), path
        assert any(line.startswith("## ") for line in lines), path
        assert all("## " not in line[2:] for line in lines if line.startswith("# ")), path


def test_step_policy_script_does_not_target_h5_or_h10_outputs():
    assert "h5" not in str(step_policy.DEFAULT_OUT_DIR)
    assert "h10" not in str(step_policy.DEFAULT_OUT_DIR)
    assert step_policy.H_MAX == 0.1
    assert step_policy.H_MIN == 0.002
