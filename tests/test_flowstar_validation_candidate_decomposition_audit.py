from __future__ import annotations

import csv
import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments" / "flowstar_validation_candidate_decomposition_audit.py"
SAME_SOURCE_SCRIPT = ROOT / "experiments" / "flowstar_same_source_endpoint_tube_compare.py"

spec = importlib.util.spec_from_file_location("flowstar_validation_candidate_decomposition_audit", SCRIPT)
assert spec is not None and spec.loader is not None
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)

same_spec = importlib.util.spec_from_file_location("flowstar_same_source_endpoint_tube_compare", SAME_SOURCE_SCRIPT)
assert same_spec is not None and same_spec.loader is not None
same_source = importlib.util.module_from_spec(same_spec)
same_spec.loader.exec_module(same_source)


def _put_box(row: dict[str, object], prefix: str, *, x: tuple[str, str] = ("0", "1"), y: tuple[str, str] = ("0", "1")) -> None:
    row[f"{prefix}_x_lo"] = x[0]
    row[f"{prefix}_x_hi"] = x[1]
    row[f"{prefix}_y_lo"] = y[0]
    row[f"{prefix}_y_hi"] = y[1]


def _audit_row(
    source: str,
    *,
    status: str = "accepted",
    full_y: tuple[str, str] = ("0", "1"),
    residual_y_hi: str = "0.00005",
    include_ordinary: bool = True,
    include_raw: bool = True,
) -> dict[str, object]:
    row: dict[str, object] = {
        "trace_source": source,
        "t_before": "0",
        "h_try": "0.025",
        "status": status,
        "target_remainder_x_lo": "-0.0001",
        "target_remainder_x_hi": "0.0001",
        "target_remainder_y_lo": "-0.0001",
        "target_remainder_y_hi": "0.0001",
        "post_cutoff_residual_x_lo": "-0.00001",
        "post_cutoff_residual_x_hi": "0.00001",
        "post_cutoff_residual_y_lo": "-0.00008",
        "post_cutoff_residual_y_hi": residual_y_hi,
        "center_x": "1.25",
        "center_y": "2.4",
        "scale_x": "0.15",
        "scale_y": "0.05",
        "cutoff_polynomial_difference_width_x": "1e-12",
        "cutoff_polynomial_difference_width_y": "1e-12",
    }
    if source == "flowstar":
        full_prefix = "flowstar_full_step_tube"
    else:
        full_prefix = "torch_full_step_validation_candidate"
    row[f"{full_prefix}_domain_semantics"] = "physical_tube_over_full_step_tau_domain_before_tau_h_substitution"
    _put_box(row, full_prefix, y=full_y)
    if include_ordinary:
        _put_box(row, "picard_no_remainder_residual", x=("-0.00001", "0.00001"), y=("-0.00008", residual_y_hi))
    if include_raw:
        _put_box(row, "picard_ctrunc_raw_residual", x=("-0.00001", "0.00001"), y=("-0.00008", residual_y_hi))
    return row


def test_missing_component_fields_are_unknown_not_zero():
    row = _audit_row("flowstar", include_ordinary=False, include_raw=False)

    ledger = audit.build_ledger_row("flowstar", row)

    assert ledger["polynomial_range_x_lo"] == ""
    assert ledger["ordinary_remainder_y_hi"] == ""
    assert ledger["raw_ctrunc_residual_y_hi"] == ""
    assert ledger["cutoff_poly_diff_x_lo"] == ""
    assert ledger["ordinary_remainder_y_hi"] != 0
    assert "unknown ordinary_remainder endpoints" in ledger["notes"]
    assert "no zero inferred" in ledger["notes"]


def test_width_close_full_step_residual_fail_reports_decomposition_mismatch():
    flow = _audit_row("flowstar", status="rejected", full_y=("0", "1"), residual_y_hi="0.000108")
    noqueue = _audit_row("torch_noqueue", status="accepted", full_y=("0", "0.99995"), residual_y_hi="0.000058")
    v2 = _audit_row("torch_v2", status="accepted", full_y=("0", "0.99995"), residual_y_hi="0.000058")

    ledger = audit.build_ledger([flow], [noqueue], [v2], t=0.0, h=0.025)
    summary = audit.summarize(ledger)

    assert summary["full_step_width_close"] == "true"
    assert summary["verdict"] == "residual_decomposition_mismatch"
    assert summary["acceptance_residual_gap_equals_full_step_y_hi_gap"] == "true"


def test_present_component_endpoints_explain_y_hi_gap():
    flow = _audit_row("flowstar", status="rejected", full_y=("0", "1"), residual_y_hi="0.000108")
    noqueue = _audit_row("torch_noqueue", status="accepted", full_y=("0", "0.99995"), residual_y_hi="0.000058")
    v2 = _audit_row("torch_v2", status="accepted", full_y=("0", "0.99995"), residual_y_hi="0.000058")
    _put_box(flow, "polynomial_range", y=("0", "0.500000"))
    _put_box(noqueue, "polynomial_range", y=("0", "0.499950"))
    _put_box(v2, "polynomial_range", y=("0", "0.499950"))

    ledger = audit.build_ledger([flow], [noqueue], [v2], t=0.0, h=0.025)
    summary = audit.summarize(ledger)

    assert summary["exposed_gap_component"] == "polynomial_range"
    assert summary["polynomial_range_component"] == "differs"


def _same_source_row(source: str, *, old_y_hi: str, full_y_hi: str, tau_y_hi: str) -> dict[str, object]:
    row: dict[str, object] = {
        "trace_source": source,
        "t_before": "0",
        "h_try": "0.025",
        "status": "accepted",
        "endpoint_box_before_center_y_hi": old_y_hi,
    }
    if source == "flowstar":
        full_prefix = "flowstar_full_step_tube"
        tau_prefix = "flowstar_tau_h_endpoint"
    else:
        full_prefix = "torch_full_step_validation_candidate"
        tau_prefix = "torch_tau_h_endpoint"
    row[f"{full_prefix}_source_object"] = "full_candidate"
    row[f"{full_prefix}_domain_semantics"] = "full_domain"
    row[f"{tau_prefix}_source_object"] = "tau_candidate"
    row[f"{tau_prefix}_domain_semantics"] = "tau_domain"
    for prefix in (full_prefix, tau_prefix):
        row[f"{prefix}_includes_target_remainder"] = "false"
        row[f"{prefix}_includes_ordinary_remainder"] = "false"
        row[f"{prefix}_includes_cutoff_poly_diff"] = "true"
        row[f"{prefix}_includes_symbolic_output_width"] = "false"
    _put_box(row, full_prefix, y=("0", full_y_hi))
    _put_box(row, tau_prefix, y=("0", tau_y_hi))
    return row


def test_previous_gap_reduction_is_quantified_not_binary_only(tmp_path):
    rows, summary = same_source.build_same_source_ledger(
        [_same_source_row("flowstar", old_y_hi="10.0", full_y_hi="10.0", tau_y_hi="8.0")],
        [_same_source_row("torch_noqueue", old_y_hi="9.0", full_y_hi="9.99", tau_y_hi="7.98")],
        [_same_source_row("torch_v2", old_y_hi="9.0", full_y_hi="9.99", tau_y_hi="7.98")],
        t=0.0,
        h=0.025,
    )

    assert summary["previous_endpoint_before_center_y_hi_delta"] == -1.0
    assert abs(summary["same_source_full_step_y_hi_delta"] - -0.01) < 1e-12
    assert abs(summary["previous_gap_reduced_factor"] - 100.0) < 1e-9

    report = same_source._report(tmp_path, rows, summary, t=0.0, h=0.025)
    assert "previous_gap_reduced_factor" in report
    assert "Does the previous y_hi gap remain" not in report
    assert "The old large y_hi gap is mostly explained by source/stage mismatch" in report


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = sorted({field for row in rows for field in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_cli_writes_ledger_and_report(tmp_path):
    trace_dir = tmp_path / "traces"
    out_dir = tmp_path / "out"
    _write_csv(trace_dir / "flowstar_trace.csv", [_audit_row("flowstar", status="rejected", residual_y_hi="0.000108")])
    _write_csv(trace_dir / "torch_noqueue_trace.csv", [_audit_row("torch_noqueue", status="accepted", residual_y_hi="0.000058")])
    _write_csv(trace_dir / "torch_v2_trace.csv", [_audit_row("torch_v2", status="accepted", residual_y_hi="0.000058")])

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

    assert (out_dir / "validation_candidate_decomposition_ledger.csv").exists()
    assert (out_dir / "validation_candidate_decomposition_report.md").exists()


def test_checked_in_trace_headers_expose_decomposition_fields():
    traces = (
        ROOT / "outputs" / "flowstar_step_trace_compare" / "flowstar_trace.csv",
        ROOT / "outputs" / "flowstar_step_trace_compare" / "torch_noqueue_trace.csv",
        ROOT / "outputs" / "flowstar_step_trace_compare" / "torch_v2_trace.csv",
    )
    required_fields = (
        "polynomial_range_x_lo",
        "polynomial_range_y_hi",
        "ordinary_remainder_x_lo",
        "raw_ctrunc_residual_y_hi",
        "cutoff_poly_diff_y_hi",
        "post_cutoff_residual_y_hi",
    )

    for trace in traces:
        with trace.open(newline="", encoding="utf-8") as handle:
            fieldnames = csv.DictReader(handle).fieldnames or []
        for field in required_fields:
            assert field in fieldnames


def test_docs_and_outputs_recommend_decomposition_not_stale_alignment():
    docs = (ROOT / "docs" / "flowstar_step_trace_divergence_report.md").read_text(encoding="utf-8")
    output = (ROOT / "outputs" / "flowstar_step_trace_compare" / "trace_divergence_report.md").read_text(encoding="utf-8")

    assert "First align same-source tube/endpoint objects" not in docs
    assert "First align same-source tube/endpoint objects" not in output
    assert "Expose and compare polynomial/remainder/raw-ctrunc/no-remainder decomposition" in docs
    assert "Expose and compare polynomial/remainder/raw-ctrunc/no-remainder decomposition" in output
