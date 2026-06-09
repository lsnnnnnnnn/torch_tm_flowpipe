from __future__ import annotations

import csv
import importlib.util
import subprocess
import sys
from pathlib import Path

from torch_tm_flowpipe import Interval, flowpipe_step_flowstar_style_adaptive
from torch_tm_flowpipe.ode_examples import van_der_pol_ode

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments" / "flowstar_raw_ctrunc_residual_audit.py"

spec = importlib.util.spec_from_file_location("flowstar_raw_ctrunc_residual_audit", SCRIPT)
assert spec is not None and spec.loader is not None
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def _put_box(row: dict[str, object], prefix: str, *, x: tuple[str, str] = ("0", "1"), y: tuple[str, str] = ("0", "1")) -> None:
    row[f"{prefix}_x_lo"] = x[0]
    row[f"{prefix}_x_hi"] = x[1]
    row[f"{prefix}_y_lo"] = y[0]
    row[f"{prefix}_y_hi"] = y[1]


def _row(
    source: str,
    *,
    raw_y_hi: str = "0.000058",
    raw_poly_y_hi: str = "2.0",
    raw_remainder_y_hi: str | None = None,
    domain: str = "physical_remainder_interval_over_full_step_tau_domain_before_cutoff_polyDiff",
    include_picard_remainder: bool = True,
    include_ordinary: bool = True,
) -> dict[str, object]:
    raw_remainder_y_hi = raw_y_hi if raw_remainder_y_hi is None else raw_remainder_y_hi
    row: dict[str, object] = {
        "trace_source": source,
        "t_before": "0",
        "h_try": "0.025",
        "status": "accepted",
        "raw_ctrunc_residual_domain_semantics": domain,
        "raw_ctrunc_residual_includes_target_remainder": "false",
        "raw_ctrunc_residual_includes_ordinary_remainder": "false",
        "raw_ctrunc_residual_includes_cutoff_poly_diff": "false",
    }
    _put_box(row, "raw_ctrunc_residual", x=("-0.00001", "0.00001"), y=("-0.00008", raw_y_hi))
    _put_box(row, "raw_ctrunc_polynomial_range", x=("1", "2"), y=("1", raw_poly_y_hi))
    _put_box(row, "raw_ctrunc_remainder", x=("-0.00001", "0.00001"), y=("-0.00008", raw_remainder_y_hi))
    _put_box(row, "picard_no_remainder_range", x=("1", "2"), y=("1", "2"))
    _put_box(row, "target_remainder", x=("-0.0001", "0.0001"), y=("-0.0001", "0.0001"))
    if include_picard_remainder:
        _put_box(row, "picard_no_remainder_remainder", x=("0", "0"), y=("0", "0"))
    if include_ordinary:
        _put_box(row, "ordinary_remainder", x=("-0.00001", "0.00001"), y=("-0.00008", raw_y_hi))
    return row


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = sorted({field for row in rows for field in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def test_missing_flowstar_ordinary_no_remainder_endpoints_unknown_not_zero():
    row = _row("flowstar", include_picard_remainder=False, include_ordinary=False)

    ledger = audit.build_ledger_row("flowstar", row)

    assert ledger["picard_no_remainder_remainder_y_hi"] == ""
    assert ledger["picard_no_remainder_remainder_y_hi"] != 0
    assert "flowstar.picard_no_remainder_remainder_y_hi" in ledger["missing_fields"]
    assert "flowstar.ordinary_remainder_y_hi" in ledger["missing_fields"]
    assert "unknown, not zero" in ledger["notes"]


def test_same_polynomial_range_different_raw_residual_reports_raw_remainder_gap():
    flow = _row("flowstar", raw_y_hi="0.000108", raw_poly_y_hi="2.0")
    noqueue = _row("torch_noqueue", raw_y_hi="0.000058", raw_poly_y_hi="2.0")
    v2 = _row("torch_v2", raw_y_hi="0.000058", raw_poly_y_hi="2.0")

    ledger = audit.build_ledger([flow], [noqueue], [v2], t=0.0, h=0.025)
    summary = audit.summarize(ledger)
    noqueue_row = next(row for row in ledger if row["source"] == "torch_noqueue")

    assert noqueue_row["component_matching_raw_y_hi_delta"] == "raw_remainder_gap"
    assert summary["first_component_explaining_raw_y_hi_gap"] == "raw_remainder_gap"
    assert summary["raw_polynomial_range_component"] == "same"


def test_mismatched_domain_labels_mark_comparison_noncausal():
    flow = _row("flowstar", raw_y_hi="0.000108")
    noqueue = _row("torch_noqueue", raw_y_hi="0.000058", domain="normalized_remainder_interval")
    v2 = _row("torch_v2", raw_y_hi="0.000058")

    ledger = audit.build_ledger([flow], [noqueue], [v2], t=0.0, h=0.025)
    summary = audit.summarize(ledger)
    noqueue_row = next(row for row in ledger if row["source"] == "torch_noqueue")

    assert noqueue_row["component_matching_raw_y_hi_delta"] == "noncausal_domain_mismatch"
    assert summary["raw_objects_semantically_same"] == "false"
    assert summary["semantic_mismatch"] == "noncausal_domain_mismatch"


def _box_tuple(tm):
    if tm is None:
        return None
    return tuple(interval.to_tuple() for interval in tm.range_box())


def test_trace_export_is_passive_for_validation_decision_and_key_boxes():
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
        diagnostics_context={"mode": "passive_trace_test", "segment_index": 0},
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
    _write_csv(trace_dir / "flowstar_trace.csv", [_row("flowstar", raw_y_hi="0.000108", include_ordinary=False)])
    _write_csv(trace_dir / "torch_noqueue_trace.csv", [_row("torch_noqueue", raw_y_hi="0.000058")])
    _write_csv(trace_dir / "torch_v2_trace.csv", [_row("torch_v2", raw_y_hi="0.000058")])

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

    assert (out_dir / "raw_ctrunc_residual_ledger.csv").exists()
    report = (out_dir / "raw_ctrunc_residual_report.md").read_text(encoding="utf-8")
    assert "diagnostic-only" in report
    assert "raw_remainder_gap" in report
