from __future__ import annotations

import csv
import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments" / "flowstar_same_source_endpoint_tube_compare.py"
spec = importlib.util.spec_from_file_location("flowstar_same_source_endpoint_tube_compare", SCRIPT)
assert spec is not None and spec.loader is not None
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def _put_box(row: dict[str, object], prefix: str, *, y_hi: str = "2.0") -> None:
    row.update(
        {
            f"{prefix}_x_lo": "1.0",
            f"{prefix}_x_hi": "1.5",
            f"{prefix}_y_lo": "1.8",
            f"{prefix}_y_hi": y_hi,
        }
    )


def _row(
    *,
    source: str,
    flow_source_object: str = "same_full_source",
    torch_source_object: str = "same_full_source",
    tau_flow_source_object: str = "same_tau_source",
    tau_torch_source_object: str = "same_tau_source",
    full_y_hi: str = "2.0",
    tau_y_hi: str = "2.0",
    missing: str | None = None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "trace_source": source,
        "t_before": "0",
        "h_try": "0.025",
        "h": "0.025",
        "status": "accepted",
        "endpoint_box_before_center_y_hi": "2.5" if source == "flowstar" else "2.0",
    }
    if source == "flowstar":
        full_prefix = "flowstar_full_step_tube"
        tau_prefix = "flowstar_tau_h_endpoint"
        row[f"{full_prefix}_source_object"] = flow_source_object
        row[f"{tau_prefix}_source_object"] = tau_flow_source_object
    else:
        full_prefix = "torch_full_step_validation_candidate"
        tau_prefix = "torch_tau_h_endpoint"
        row[f"{full_prefix}_source_object"] = torch_source_object
        row[f"{tau_prefix}_source_object"] = tau_torch_source_object
    row[f"{full_prefix}_domain_semantics"] = "same_full_domain"
    row[f"{tau_prefix}_domain_semantics"] = "same_tau_domain"
    for prefix in (full_prefix, tau_prefix):
        row[f"{prefix}_includes_target_remainder"] = "false"
        row[f"{prefix}_includes_ordinary_remainder"] = "false"
        row[f"{prefix}_includes_cutoff_poly_diff"] = "true"
        row[f"{prefix}_includes_symbolic_output_width"] = "false"
    _put_box(row, full_prefix, y_hi=full_y_hi)
    _put_box(row, tau_prefix, y_hi=tau_y_hi)
    if missing:
        row.pop(missing, None)
    return row


def test_mismatched_source_labels_mark_semantic_comparison_invalid():
    rows, summary = audit.build_same_source_ledger(
        [_row(source="flowstar", flow_source_object="flow_tmv")],
        [_row(source="torch_noqueue", torch_source_object="torch_candidate")],
        [_row(source="torch_v2", torch_source_object="torch_candidate")],
        t=0.0,
        h=0.025,
    )

    full = next(row for row in rows if row["comparison_kind"] == "full_step_tube")
    assert full["source_objects_match"] == "false"
    assert full["semantic_comparison_valid"] == "false"
    assert full["verdict"] == "semantic_mismatch"
    assert summary["first_same_source_divergence"] == "full_step_tube:semantic_mismatch"


def test_same_source_labels_and_y_hi_difference_report_y_hi_divergence():
    rows, summary = audit.build_same_source_ledger(
        [_row(source="flowstar", full_y_hi="2.0")],
        [_row(source="torch_noqueue", full_y_hi="2.2")],
        [_row(source="torch_v2", full_y_hi="2.2")],
        t=0.0,
        h=0.025,
    )

    full = next(row for row in rows if row["comparison_kind"] == "full_step_tube")
    assert full["semantic_comparison_valid"] == "true"
    assert full["verdict"] == "same_source_y_hi_divergence"
    assert full["y_hi_delta"] == 0.20000000000000018
    assert summary["first_same_source_divergence"] == "full_step_tube:same_source_y_hi_divergence"


def test_missing_endpoint_fields_are_unknown_not_zero():
    rows, _summary = audit.build_same_source_ledger(
        [_row(source="flowstar")],
        [_row(source="torch_noqueue", missing="torch_tau_h_endpoint_y_hi")],
        [_row(source="torch_v2")],
        t=0.0,
        h=0.025,
    )

    tau = next(row for row in rows if row["comparison_kind"] == "tau_h_endpoint")
    assert tau["semantic_comparison_valid"] == "unknown"
    assert tau["verdict"] == "unknown_missing_fields"
    assert tau["torch_noqueue_y_hi"] == ""
    assert tau["torch_noqueue_width_sum"] is None
    assert tau["torch_noqueue_width_sum"] != 0


def test_generated_report_omits_stale_recommendation_text(tmp_path):
    rows, summary = audit.build_same_source_ledger(
        [_row(source="flowstar")],
        [_row(source="torch_noqueue")],
        [_row(source="torch_v2")],
        t=0.0,
        h=0.025,
    )

    report = audit._report(tmp_path, rows, summary, t=0.0, h=0.025)

    assert "Fix PyTorch acceptance policy/target residual validation" not in report
    assert "Full-step tube comparison semantically valid" in report


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
    _write_csv(trace_dir / "flowstar_trace.csv", [_row(source="flowstar")])
    _write_csv(trace_dir / "torch_noqueue_trace.csv", [_row(source="torch_noqueue")])
    _write_csv(trace_dir / "torch_v2_trace.csv", [_row(source="torch_v2")])

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

    assert (out_dir / "same_source_endpoint_tube_ledger.csv").exists()
    assert (out_dir / "same_source_endpoint_tube_report.md").exists()
