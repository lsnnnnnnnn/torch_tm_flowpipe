from __future__ import annotations

import csv
import importlib.util
import subprocess
import sys
from pathlib import Path

from torch_tm_flowpipe import Interval, flowpipe_step_flowstar_style_adaptive
from torch_tm_flowpipe.ode_examples import van_der_pol_ode

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments" / "flowstar_internal_intermediate_ranges_audit.py"
TRACE_COMPARE = ROOT / "experiments" / "flowstar_step_trace_compare.py"

spec = importlib.util.spec_from_file_location("flowstar_internal_intermediate_ranges_audit", SCRIPT)
assert spec is not None and spec.loader is not None
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)

trace_spec = importlib.util.spec_from_file_location("flowstar_step_trace_compare", TRACE_COMPARE)
assert trace_spec is not None and trace_spec.loader is not None
trace_compare = importlib.util.module_from_spec(trace_spec)
trace_spec.loader.exec_module(trace_compare)


def _put_box(row: dict[str, object], prefix: str, *, x: tuple[str, str] = ("0", "1"), y: tuple[str, str] = ("0", "1")) -> None:
    row[f"{prefix}_x_lo"] = x[0]
    row[f"{prefix}_x_hi"] = x[1]
    row[f"{prefix}_y_lo"] = y[0]
    row[f"{prefix}_y_hi"] = y[1]


def _row(
    source: str,
    *,
    raw_y_hi: str = "0.06",
    expression_y_hi: str | None = "0",
    int_trunc_y_hi: str | None = "0",
    int_trunc2_y_hi: str | None = "0",
    mul_y_hi: str | None = "0",
    before_x0_y_hi: str | None = "0",
    after_x0_y_hi: str | None = "0",
) -> dict[str, object]:
    row: dict[str, object] = {
        "trace_source": source,
        "source": source,
        "t_before": "0",
        "h_try": "0.025",
        "status": "accepted",
    }
    _put_box(row, "raw_ctrunc_residual", x=("-0.01", "0.01"), y=("-0.02", raw_y_hi))
    if expression_y_hi is not None:
        _put_box(row, "expression_evaluate_remainder", x=("0", "0"), y=("0", expression_y_hi))
    if int_trunc_y_hi is not None:
        _put_box(row, "int_trunc_dropped_terms", x=("0", "0"), y=("0", int_trunc_y_hi))
    if int_trunc2_y_hi is not None:
        _put_box(row, "int_trunc2_dropped_terms", x=("0", "0"), y=("0", int_trunc2_y_hi))
    if mul_y_hi is not None:
        _put_box(row, "mul_ctrunc_normal_remainder", x=("0", "0"), y=("0", mul_y_hi))
    if before_x0_y_hi is not None:
        _put_box(row, "accumulated_remainder_before_x0_add", x=("0", "0"), y=("0", before_x0_y_hi))
    if after_x0_y_hi is not None:
        _put_box(row, "accumulated_remainder_after_x0_add", x=("0", "0"), y=("0", after_x0_y_hi))
    return row


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = sorted({field for row in rows for field in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def test_missing_internal_fields_are_unknown_not_zero():
    ledger = audit.build_ledger_row(
        "flowstar",
        _row(
            "flowstar",
            expression_y_hi=None,
            int_trunc_y_hi=None,
            int_trunc2_y_hi=None,
            mul_y_hi=None,
            before_x0_y_hi=None,
            after_x0_y_hi=None,
        ),
    )

    assert ledger["expression_evaluate_remainder_y_hi"] == ""
    assert ledger["int_trunc_dropped_terms_y_hi"] == ""
    assert ledger["int_trunc_dropped_terms_y_hi"] != 0
    assert "flowstar.int_trunc_dropped_terms_y_hi" in ledger["missing_fields"]
    assert "unknown, not zero" in ledger["notes"]


def test_internal_component_y_hi_delta_matching_raw_delta_is_reported():
    flow = _row("flowstar", raw_y_hi="0.10", expression_y_hi="0.04")
    noqueue = _row("torch_noqueue", raw_y_hi="0.06", expression_y_hi="0")
    v2 = _row("torch_v2", raw_y_hi="0.06", expression_y_hi="0")

    ledger = audit.build_ledger([flow], [noqueue], [v2], t=0.0, h=0.025)
    summary = audit.summarize(ledger)

    assert next(row for row in ledger if row["source"] == "torch_noqueue")["component_matching_raw_y_hi_delta"] == "expression_evaluate_remainder_gap"
    assert summary["first_flowstar_internal_object_explaining_y_hi_delta"] == "expression_evaluate_remainder_gap"
    assert summary["cause_classification"] == "expression evaluate_remainder / evaluated RHS remainder"


def test_all_exposed_internals_match_but_raw_differs_reports_hidden_gap():
    flow = _row("flowstar", raw_y_hi="0.10")
    noqueue = _row("torch_noqueue", raw_y_hi="0.06")
    v2 = _row("torch_v2", raw_y_hi="0.06")

    ledger = audit.build_ledger([flow], [noqueue], [v2], t=0.0, h=0.025)
    summary = audit.summarize(ledger)

    assert next(row for row in ledger if row["source"] == "torch_noqueue")["component_matching_raw_y_hi_delta"] == "hidden_internal_intermediate_ranges_gap"
    assert summary["first_flowstar_internal_object_explaining_y_hi_delta"] == "hidden_internal_intermediate_ranges_gap"
    assert "AST_Node::evaluate NODE_VAR" in summary["exact_flowstar_internal_object_still_inaccessible"]


def _box_tuple(tm):
    if tm is None:
        return None
    return tuple(interval.to_tuple() for interval in tm.range_box())


def test_probe_export_fields_are_diagnostic_only():
    required = {
        "expression_evaluate_remainder_y_hi",
        "int_trunc_dropped_terms_y_hi",
        "int_trunc2_dropped_terms_y_hi",
        "mul_ctrunc_normal_remainder_y_hi",
        "accumulated_remainder_before_x0_add_y_hi",
        "accumulated_remainder_after_x0_add_y_hi",
    }
    assert required.issubset(set(trace_compare.TRACE_FIELDS))

    kwargs = dict(
        h=0.002,
        h_min=0.002,
        h_max=0.002,
        order=4,
        target_remainder_radius=1e-4,
        cutoff_threshold=1e-10,
        max_validation_attempts=2,
        validation_mode="target_remainder_flowstar_ctrunc",
        reset_mode="normalized_insertion",
    )
    baseline = flowpipe_step_flowstar_style_adaptive(
        van_der_pol_ode,
        [Interval(1.1, 1.4), Interval(2.35, 2.45)],
        **kwargs,
    )
    diagnostics: list[dict[str, object]] = []
    traced = flowpipe_step_flowstar_style_adaptive(
        van_der_pol_ode,
        [Interval(1.1, 1.4), Interval(2.35, 2.45)],
        diagnostics=diagnostics,
        diagnostics_context={"mode": "internal_ranges_passive_test", "segment_index": 0},
        **kwargs,
    )

    assert diagnostics
    assert traced.status == baseline.status
    assert traced.h == baseline.h
    assert _box_tuple(traced.tm) == _box_tuple(baseline.tm)
    assert _box_tuple(traced.final_tm) == _box_tuple(baseline.final_tm)
    assert _box_tuple(traced.reset_tm) == _box_tuple(baseline.reset_tm)


def test_cli_writes_ledger_and_report(tmp_path):
    trace_dir = tmp_path / "traces"
    out_dir = tmp_path / "out"
    _write_csv(trace_dir / "flowstar_trace.csv", [_row("flowstar", raw_y_hi="0.10")])
    _write_csv(trace_dir / "torch_noqueue_trace.csv", [_row("torch_noqueue", raw_y_hi="0.06")])
    _write_csv(trace_dir / "torch_v2_trace.csv", [_row("torch_v2", raw_y_hi="0.06")])

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--trace-dir",
            str(trace_dir),
            "--out-dir",
            str(out_dir),
        ],
        check=True,
        cwd=ROOT,
    )

    assert (out_dir / "internal_intermediate_ranges_ledger.csv").exists()
    report = (out_dir / "internal_intermediate_ranges_report.md").read_text(encoding="utf-8")
    assert "Flow* Source Inspection" in report
    assert "hidden_internal_intermediate_ranges_gap" in report
