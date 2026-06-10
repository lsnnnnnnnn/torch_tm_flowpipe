from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ATTEMPT_SCRIPT = ROOT / "experiments" / "flowstar_raw_remainder_compat_attempt_sequence.py"
SHORT_SCRIPT = ROOT / "experiments" / "flowstar_raw_remainder_compat_short_horizon.py"

attempt_spec = importlib.util.spec_from_file_location("flowstar_raw_remainder_compat_attempt_sequence", ATTEMPT_SCRIPT)
assert attempt_spec is not None and attempt_spec.loader is not None
attempt = importlib.util.module_from_spec(attempt_spec)
attempt_spec.loader.exec_module(attempt)

short_spec = importlib.util.spec_from_file_location("flowstar_raw_remainder_compat_short_horizon", SHORT_SCRIPT)
assert short_spec is not None and short_spec.loader is not None
short = importlib.util.module_from_spec(short_spec)
short_spec.loader.exec_module(short)


def _flow_row(status: str, y_hi: str, h: float = 0.025) -> dict[str, object]:
    return {
        "status": status,
        "h_try": h,
        "t_before": 0.0,
        "picard_ctrunc_normal_residual_x_lo": "-0.00001",
        "picard_ctrunc_normal_residual_x_hi": "0.00001",
        "picard_ctrunc_normal_residual_y_lo": "-0.00008",
        "picard_ctrunc_normal_residual_y_hi": y_hi,
        "target_remainder_y_lo": "-0.0001",
        "target_remainder_y_hi": "0.0001",
    }


def _torch_row(attempt_index: int, mode: str, status: str, y_hi: str) -> dict[str, object]:
    return {
        "source": "torch",
        "mode": mode,
        "attempt_index": attempt_index,
        "t_before": 0.0,
        "h_try": attempt.H_ATTEMPTS[attempt_index - 1],
        "status": status,
        "residual_x_lo": "-0.00001",
        "residual_x_hi": "0.00001",
        "residual_y_lo": "-0.00008",
        "residual_y_hi": y_hi,
        "target_y_hi": "0.0001",
        "subset_y": float(y_hi) <= 0.0001,
        "failed_dim": "" if float(y_hi) <= 0.0001 else "y",
        "notes": "fixture",
    }


def test_one_step_compat_remains_opt_in():
    current_diag, current_seg = attempt.run_torch_attempt("current_no_queue", 0.025)
    compat_diag, compat_seg = attempt.run_torch_attempt("flowstar_raw_remainder_compat", 0.025)

    assert current_seg.status == "validated"
    assert compat_seg.status == "failed"
    assert current_diag["flowstar_raw_remainder_compat_enabled"] is False
    assert compat_diag["flowstar_raw_remainder_compat_enabled"] is True


def test_attempt_sequence_writer_works_on_fixtures(tmp_path):
    flow_rows = [
        _flow_row("rejected", "0.01", 0.1),
        _flow_row("rejected", "0.001", 0.05),
        _flow_row("rejected", "0.0001083", 0.025),
        _flow_row("accepted", "0.00002", 0.0125),
    ]
    torch_rows = [
        _torch_row(1, "flowstar_raw_remainder_compat", "rejected", "0.011"),
        _torch_row(2, "flowstar_raw_remainder_compat", "rejected", "0.0011"),
        _torch_row(3, "flowstar_raw_remainder_compat", "rejected", "0.0001082"),
        _torch_row(4, "flowstar_raw_remainder_compat", "accepted", "0.00002"),
    ]

    ledger = attempt.build_ledger(flow_rows, torch_rows)
    out_csv = tmp_path / "attempt_sequence_ledger.csv"
    out_md = tmp_path / "attempt_sequence_report.md"
    attempt.write_ledger(out_csv, ledger)
    attempt.write_report(out_md, ledger)

    with out_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 8
    assert rows[-1]["matches_flowstar_status"] == "true"
    assert out_md.read_text(encoding="utf-8").splitlines()[0] == "# Flow* Raw Remainder Compat Attempt Sequence"


def test_default_current_mode_remains_accepted_at_h_0025_fixture():
    _diag, seg = attempt.run_torch_attempt("current_no_queue", 0.025)

    assert seg.status == "validated"


def test_compat_mode_matches_flowstar_h_0025_reject_fixture():
    diag, seg = attempt.run_torch_attempt("flowstar_raw_remainder_compat", 0.025)
    flow = _flow_row("rejected", "0.0001083283903691475", 0.025)
    torch = attempt._torch_row(3, 0.025, "flowstar_raw_remainder_compat", diag, seg)
    ledger = attempt.build_ledger([_flow_row("rejected", "0.01", 0.1), _flow_row("rejected", "0.001", 0.05), flow, _flow_row("accepted", "0.00002", 0.0125)], [torch])
    compat_row = next(row for row in ledger if row.get("mode") == "flowstar_raw_remainder_compat")

    assert seg.status == "failed"
    assert compat_row["matches_flowstar_status"] is True
    assert compat_row["failed_dim"] == "y"


def test_overconservative_accepted_step_fixture_is_detected():
    flow = {"status": "accepted"}
    row = {"status": "rejected"}

    assert attempt.detect_overconservative(row, flow) is True
    assert attempt.detect_overconservative({"status": "accepted"}, flow) is False


def test_scripts_do_not_produce_h10_outputs(tmp_path):
    assert "h10" not in str(attempt.DEFAULT_OUT_DIR)
    assert "h10" not in str(short.DEFAULT_OUT_DIR)
    assert max(attempt.H_ATTEMPTS) == 0.1

    short_rows = short.finalize_summary([
        {"source": "flowstar", "mode": "probe_schedule", "_accepted_h": [0.0125], "final_width_sum": ""},
        {"source": "torch", "mode": "current_no_queue", "_accepted_h": [0.025], "final_width_sum": 1.0},
        {"source": "torch", "mode": "flowstar_raw_remainder_compat", "_accepted_h": [0.0125], "final_width_sum": 1.1},
    ])
    out_csv = tmp_path / "short_horizon_summary.csv"
    out_md = tmp_path / "short_horizon_report.md"
    short.write_summary(out_csv, short_rows)
    short.write_report(out_md, short_rows, 0.5)

    produced = [path.name for path in tmp_path.iterdir()]
    assert produced
    assert all("h10" not in name for name in produced)
