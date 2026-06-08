from __future__ import annotations

import csv
import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments" / "flowstar_box_lifecycle_alignment_audit.py"
spec = importlib.util.spec_from_file_location("flowstar_box_lifecycle_alignment_audit", SCRIPT)
assert spec is not None and spec.loader is not None
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def _row(source: str, **kwargs: object) -> dict[str, object]:
    row: dict[str, object] = {
        "trace_source": source,
        "status": kwargs.pop("status", "accepted"),
        "t_before": kwargs.pop("t_before", "0"),
        "h_try": kwargs.pop("h_try", "0.025"),
    }
    row.update(kwargs)
    return row


def _with_box(prefix: str, x: tuple[float, float], y: tuple[float, float]) -> dict[str, object]:
    return {
        f"{prefix}_x_lo": x[0],
        f"{prefix}_x_hi": x[1],
        f"{prefix}_y_lo": y[0],
        f"{prefix}_y_hi": y[1],
    }


def _stage_row(source: str, *, pre_x=(0.0, 1.0), pre_y=(0.0, 1.0), endpoint_x=(1.0, 2.0), endpoint_y=(1.0, 2.0), reset_x=(0.0, 1.0), reset_y=(0.0, 1.0), **kwargs: object) -> dict[str, object]:
    row = _row(source, **kwargs)
    row.update(_with_box("pre_step_box", pre_x, pre_y))
    row.update(_with_box("endpoint_box_before_center", endpoint_x, endpoint_y))
    row.update(_with_box("reset_box_after_center_scale", reset_x, reset_y))
    return row


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = sorted({field for row in rows for field in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def test_generic_center_scale_mismatch_is_not_called_same_local_box_mismatch():
    flow = [_row("flowstar", center_x="0", center_y="0", scale_x="1", scale_y="1")]
    noqueue = [_row("torch_noqueue", center_x="10", center_y="0", scale_x="2", scale_y="1")]
    v2 = [_row("torch_v2", center_x="10", center_y="0", scale_x="2", scale_y="1")]

    rows, summary = audit.build_lifecycle_alignment(flow, noqueue, v2, t=0.0, h=0.025)

    assert summary["first_lifecycle_stage_divergence"] == "unknown_missing_stage_fields"
    assert summary["residual_comparison_stage_valid"] == "unknown"
    assert "center" not in summary["first_lifecycle_stage_divergence"]
    assert "center/scale" in rows[1]["notes"]


def test_same_t_h_with_different_pre_step_box_is_noncausal():
    flow = [_stage_row("flowstar", status="rejected")]
    noqueue = [_stage_row("torch_noqueue", pre_x=(0.0, 2.0))]
    v2 = [_stage_row("torch_v2", pre_x=(0.0, 2.0))]

    rows, summary = audit.build_lifecycle_alignment(flow, noqueue, v2, t=0.0, h=0.025)

    assert summary["pre_step_boxes_equal"] == "false"
    assert summary["first_lifecycle_stage_divergence"] == "pre_step_box"
    assert summary["residual_comparison_stage_valid"] == "false"
    assert rows[1]["picard_residual_comparison"] == "noncausal/stage-misaligned"


def test_same_pre_step_but_different_reset_box_reports_reset_stage_divergence():
    flow = [_stage_row("flowstar", status="rejected")]
    noqueue = [_stage_row("torch_noqueue", reset_y=(0.0, 2.0))]
    v2 = [_stage_row("torch_v2", reset_y=(0.0, 2.0))]

    _, summary = audit.build_lifecycle_alignment(flow, noqueue, v2, t=0.0, h=0.025)

    assert summary["pre_step_boxes_equal"] == "true"
    assert summary["reset_after_center_boxes_equal"] == "false"
    assert summary["first_lifecycle_stage_divergence"] == "reset_box_after_center_scale"
    assert summary["residual_comparison_stage_valid"] == "false"


def test_missing_flowstar_raw_ctrunc_endpoints_are_unknown_not_zero():
    flow = [_stage_row("flowstar", status="rejected")]
    noqueue = [_stage_row("torch_noqueue")]
    v2 = [_stage_row("torch_v2")]

    rows, summary = audit.build_lifecycle_alignment(flow, noqueue, v2, t=0.0, h=0.025)

    assert "picard_ctrunc_raw_residual" in summary["flowstar_missing_residual_components"]
    assert rows[0].get("picard_ctrunc_raw_residual_x_lo", "") == ""
    assert rows[0].get("picard_ctrunc_raw_residual_y_hi", "") != 0


def test_cli_writes_lifecycle_ledger_and_report(tmp_path):
    trace_dir = tmp_path / "trace"
    out_dir = tmp_path / "out"
    _write_csv(trace_dir / "flowstar_trace.csv", [_stage_row("flowstar", status="rejected")])
    _write_csv(trace_dir / "torch_noqueue_trace.csv", [_stage_row("torch_noqueue")])
    _write_csv(trace_dir / "torch_v2_trace.csv", [_stage_row("torch_v2")])

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

    assert (out_dir / "box_lifecycle_ledger.csv").exists()
    report = (out_dir / "box_lifecycle_report.md").read_text(encoding="utf-8")
    assert "Are Flow* and PyTorch pre_step boxes equal?" in report


def test_cli_reports_unknown_when_trace_stage_columns_are_missing(tmp_path):
    trace_dir = tmp_path / "trace"
    out_dir = tmp_path / "out"
    _write_csv(trace_dir / "flowstar_trace.csv", [_row("flowstar", status="rejected")])
    _write_csv(trace_dir / "torch_noqueue_trace.csv", [_row("torch_noqueue")])
    _write_csv(trace_dir / "torch_v2_trace.csv", [_row("torch_v2")])

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

    ledger = list(csv.DictReader((out_dir / "box_lifecycle_ledger.csv").open(newline="", encoding="utf-8")))
    report = (out_dir / "box_lifecycle_report.md").read_text(encoding="utf-8")
    assert ledger[0]["first_lifecycle_stage_divergence"] == "unknown_missing_stage_fields"
    assert "unknown_missing_stage_fields" in report


def test_checked_in_trace_headers_have_lifecycle_stage_columns():
    required = {
        "pre_step_box_x_lo",
        "pre_step_box_x_hi",
        "pre_step_box_y_lo",
        "pre_step_box_y_hi",
        "endpoint_box_before_center_x_lo",
        "endpoint_box_before_center_x_hi",
        "endpoint_box_before_center_y_lo",
        "endpoint_box_before_center_y_hi",
    }
    for name in ("flowstar_trace.csv", "torch_noqueue_trace.csv", "torch_v2_trace.csv"):
        path = ROOT / "outputs" / "flowstar_step_trace_compare" / name
        with path.open(newline="", encoding="utf-8") as handle:
            fields = set(csv.DictReader(handle).fieldnames or [])
        assert required <= fields
