import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments" / "flowstar_acceptance_predicate_audit.py"
spec = importlib.util.spec_from_file_location("flowstar_acceptance_predicate_audit", SCRIPT)
audit = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(audit)


def test_interval_subset_fails_when_narrow_interval_is_shifted_outside_target():
    assert (1.20 - 1.10) < (1.0 - -1.0)
    assert audit.interval_subset(1.10, 1.20, -1.0, 1.0) is False


def test_ledger_reports_shifted_component_failure_despite_smaller_width():
    row = {
        "trace_source": "fixture",
        "status": "rejected",
        "t_before": "0",
        "h_try": "0.025",
        "target_remainder_lo_x": "-1.0",
        "target_remainder_hi_x": "1.0",
        "target_remainder_lo_y": "-1.0",
        "target_remainder_hi_y": "1.0",
        "picard_ctrunc_normal_residual_lo_x": "1.10",
        "picard_ctrunc_normal_residual_hi_x": "1.20",
        "picard_ctrunc_normal_residual_lo_y": "-0.25",
        "picard_ctrunc_normal_residual_hi_y": "0.25",
    }

    ledger = audit.build_ledger_row("fixture", row)

    assert ledger["residual_width_sum"] < ledger["target_width_sum"]
    assert ledger["subset_x"] is False
    assert ledger["subset_y"] is True
    assert ledger["which_dim_failed"] == "x"
