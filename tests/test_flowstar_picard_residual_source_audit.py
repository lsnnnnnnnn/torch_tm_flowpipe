from __future__ import annotations

import importlib.util
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments" / "flowstar_picard_residual_source_audit.py"
spec = importlib.util.spec_from_file_location("flowstar_picard_residual_source_audit", SCRIPT)
assert spec is not None and spec.loader is not None
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_shifted_residual_fails_even_when_width_is_smaller_than_target():
    row = {
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

    ledger = audit.build_ledger_row("flowstar", row)

    residual_width_x = ledger["residual_x_hi"] - ledger["residual_x_lo"]
    target_width_x = ledger["target_x_hi"] - ledger["target_x_lo"]
    assert residual_width_x < target_width_x
    assert ledger["subset_x"] == "false"
    assert ledger["subset_y"] == "true"
    assert ledger["failed_dim"] == "x"


def test_missing_component_fields_are_unknown_not_zero():
    row = {
        "status": "accepted",
        "t_before": "0",
        "h_try": "0.025",
        "target_remainder_lo_x": "-0.0001",
        "target_remainder_hi_x": "0.0001",
        "target_remainder_lo_y": "-0.0001",
        "target_remainder_hi_y": "0.0001",
    }

    ledger = audit.build_ledger_row("flowstar", row)

    assert ledger["subset_x"] == "unknown"
    assert ledger["subset_y"] == "unknown"
    assert ledger["failed_dim"] == "unknown"
    assert ledger["picard_no_remainder_x_lo"] == ""
    assert ledger["picard_ctrunc_raw_y_hi"] == ""
    assert ledger["polynomial_diff_x_lo"] == ""
    assert ledger["cutoff_uncertainty_y_hi"] == ""
    assert "missing picard_no_remainder endpoints" in ledger["notes"]
    assert "missing picard_ctrunc_raw endpoints" in ledger["notes"]
    assert ledger["picard_no_remainder_x_lo"] != 0
