from __future__ import annotations

import csv
import importlib.util
import subprocess
import sys
from pathlib import Path

from torch_tm_flowpipe import Interval, flowpipe_step_flowstar_style_adaptive
from torch_tm_flowpipe.ode_examples import van_der_pol_ode

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments" / "flowstar_raw_remainder_partition_audit.py"

spec = importlib.util.spec_from_file_location("flowstar_raw_remainder_partition_audit", SCRIPT)
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
    raw_y_hi: str = "0.06",
    dropped_y_hi: str | None = "0",
    multiplication_y_hi: str | None = "0",
    integration_y_hi: str | None = "0",
    after_cutoff_y_hi: str | None = "0",
) -> dict[str, object]:
    row: dict[str, object] = {
        "trace_source": source,
        "t_before": "0",
        "h_try": "0.025",
        "status": "accepted",
        "raw_remainder_range_enclosure_method": "same_method",
        "raw_remainder_normal_domain_scaling": "same_scaling",
    }
    _put_box(row, "raw_ctrunc_residual", x=("-0.01", "0.01"), y=("-0.02", raw_y_hi))
    if dropped_y_hi is not None:
        _put_box(row, "raw_remainder_dropped_terms_range", x=("0", "0"), y=("0", dropped_y_hi))
    if multiplication_y_hi is not None:
        _put_box(row, "raw_remainder_multiplication_remainder", x=("0", "0"), y=("0", multiplication_y_hi))
    if integration_y_hi is not None:
        _put_box(row, "raw_remainder_integration_remainder", x=("0", "0"), y=("0", integration_y_hi))
        _put_box(row, "raw_remainder_after_integration", x=("0", "0"), y=("0", integration_y_hi))
    if after_cutoff_y_hi is not None:
        _put_box(row, "raw_remainder_after_cutoff", x=("0", "0"), y=("0", after_cutoff_y_hi))
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
            dropped_y_hi=None,
            multiplication_y_hi=None,
            integration_y_hi=None,
            after_cutoff_y_hi=None,
        ),
    )

    assert ledger["dropped_terms_y_hi"] == ""
    assert ledger["multiplication_remainder_y_hi"] == ""
    assert ledger["dropped_terms_y_hi"] != 0
    assert "flowstar.dropped_terms_y_hi" in ledger["missing_fields"]
    assert "unknown, not zero" in ledger["notes"]


def test_dropped_term_delta_matching_raw_delta_reports_dropped_term_gap():
    flow = _row("flowstar", raw_y_hi="0.10", dropped_y_hi="0.04", multiplication_y_hi="0", integration_y_hi="0", after_cutoff_y_hi="0")
    noqueue = _row("torch_noqueue", raw_y_hi="0.06", dropped_y_hi="0", multiplication_y_hi="0", integration_y_hi="0", after_cutoff_y_hi="0")
    v2 = _row("torch_v2", raw_y_hi="0.06", dropped_y_hi="0", multiplication_y_hi="0", integration_y_hi="0", after_cutoff_y_hi="0")

    ledger = audit.build_ledger([flow], [noqueue], [v2], t=0.0, h=0.025)
    summary = audit.summarize(ledger)

    assert next(row for row in ledger if row["source"] == "torch_noqueue")["component_matching_raw_y_hi_delta"] == "dropped_term_gap"
    assert summary["first_component_explaining_y_hi_delta"] == "dropped_term_gap"
    assert summary["cause_classification"] == "dropped-term range"


def test_all_exposed_partitions_match_but_raw_differs_reports_hidden_gap():
    flow = _row("flowstar", raw_y_hi="0.10", dropped_y_hi="0", multiplication_y_hi="0", integration_y_hi="0", after_cutoff_y_hi="0")
    noqueue = _row("torch_noqueue", raw_y_hi="0.06", dropped_y_hi="0", multiplication_y_hi="0", integration_y_hi="0", after_cutoff_y_hi="0")
    v2 = _row("torch_v2", raw_y_hi="0.06", dropped_y_hi="0", multiplication_y_hi="0", integration_y_hi="0", after_cutoff_y_hi="0")

    ledger = audit.build_ledger([flow], [noqueue], [v2], t=0.0, h=0.025)
    summary = audit.summarize(ledger)

    assert next(row for row in ledger if row["source"] == "torch_noqueue")["component_matching_raw_y_hi_delta"] == "hidden_raw_remainder_gap"
    assert summary["first_component_explaining_y_hi_delta"] == "hidden_raw_remainder_gap"
    assert summary["flowstar_raw_returned_remainder_decomposable_from_exposed_fields"] == "false"


def _box_tuple(tm):
    if tm is None:
        return None
    return tuple(interval.to_tuple() for interval in tm.range_box())


def test_trace_export_remains_passive():
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
        diagnostics_context={"mode": "partition_passive_test", "segment_index": 0},
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

    assert (out_dir / "raw_remainder_partition_ledger.csv").exists()
    report = (out_dir / "raw_remainder_partition_report.md").read_text(encoding="utf-8")
    assert "Flow* Source Inspection" in report
    assert "hidden_raw_remainder_gap" in report
