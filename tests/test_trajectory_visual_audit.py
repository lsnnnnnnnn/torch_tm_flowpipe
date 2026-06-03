from __future__ import annotations

import csv
import math
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "outputs" / "trajectory_audit"


def _csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames or [], list(reader)


def _as_float(value: str) -> float:
    assert value != ""
    out = float(value)
    assert math.isfinite(out)
    return out



def test_trajectory_required_csvs_exist_and_have_columns():
    required = {
        AUDIT / "torch_structured_summary.csv": {
            "case_id",
            "mode",
            "h",
            "steps",
            "requested_order",
            "endpoint_width_sum",
            "remainder_width_sum",
            "poly_range_width_sum",
            "max_final_degree",
            "term_count_total",
        },
        AUDIT / "flowstar_structured_summary.csv": {
            "case_id",
            "status",
            "failure_reason",
            "endpoint_box_available",
            "last_segment_width_sum",
            "tube_width_sum",
            "num_segments",
        },
        AUDIT / "flowstar_vs_torch_overlay_summary.csv": {
            "case_id",
            "torch_mode",
            "flowstar_status",
            "endpoint_ratio_available",
            "last_segment_ratio_available",
            "tube_ratio_available",
        },
        AUDIT / "crosscheck_summary.csv": {
            "comparison_id",
            "category",
            "metric",
            "new_value",
            "old_value",
            "pass_fail",
            "note",
        },
        AUDIT / "torch_segments" / "torch_range_only_h0p01_s10_o4_segments.csv": {
            "case_id",
            "segment_index",
            "t_lo",
            "t_hi",
            "x_lo",
            "x_hi",
            "y_lo",
            "y_hi",
            "width_sum",
        },
        AUDIT / "flowstar_segments" / "flowstar_rem1e-4_cut1e-10_h0p01_s10_o4_segments.csv": {
            "case_id",
            "segment_index",
            "t_lo",
            "t_hi",
            "x_lo",
            "x_hi",
            "y_lo",
            "y_hi",
            "width_sum",
        },
        AUDIT / "samples" / "torch_range_only_h0p01_s10_o4_samples.csv": {
            "case_id",
            "sample_id",
            "sample_kind",
            "diagnostic_only",
            "step_index",
            "t",
            "x",
            "y",
        },
    }
    for path, columns in required.items():
        assert path.exists(), path
        header, rows = _csv_rows(path)
        assert set(header) >= columns
        assert rows or path.name.endswith("_segments.csv")

def test_trajectory_flowstar_summary_and_segments_schema():
    required_summary = {
        "case_id",
        "system",
        "h",
        "steps",
        "horizon",
        "order",
        "setting_label",
        "remainder_estimation",
        "cutoff",
        "status",
        "failure_reason",
        "endpoint_box_available",
        "endpoint_width_x",
        "endpoint_width_y",
        "endpoint_width_sum",
        "last_segment_width_x",
        "last_segment_width_y",
        "last_segment_width_sum",
        "tube_width_x",
        "tube_width_y",
        "tube_width_sum",
        "flowstar_internal_reach_s",
        "flowstar_wall_compile_s",
        "flowstar_wall_run_s",
        "flowstar_wall_total_s",
        "num_segments",
        "box_source",
        "stdout_path",
        "stderr_path",
        "model_path",
        "plot_paths",
    }
    required_segments = {
        "case_id",
        "segment_index",
        "t_lo",
        "t_hi",
        "x_lo",
        "x_hi",
        "y_lo",
        "y_hi",
        "width_x",
        "width_y",
        "width_sum",
        "box_source",
    }
    header, rows = _csv_rows(AUDIT / "flowstar_structured_summary.csv")
    assert set(header) >= required_summary
    assert len(rows) == 3
    assert {row["status"] for row in rows} >= {"completed", "failed"}

    for row in rows:
        seg_path = AUDIT / "flowstar_segments" / f"{row['case_id']}_segments.csv"
        seg_header, segments = _csv_rows(seg_path)
        assert set(seg_header) >= required_segments
        assert row["endpoint_box_available"] == "false"
        assert row["endpoint_width_x"] == ""
        assert row["endpoint_width_y"] == ""
        assert row["endpoint_width_sum"] == ""
        if row["status"] == "completed":
            assert len(segments) == int(row["steps"])
            assert row["box_source"] == "flowstar_gnuplot_segment_boxes"
            assert row["failure_reason"] == ""
        else:
            assert row["failure_reason"]


def test_trajectory_overlay_summary_has_no_endpoint_ratio():
    _header, rows = _csv_rows(AUDIT / "flowstar_vs_torch_overlay_summary.csv")
    assert rows
    assert {row["flowstar_status"] for row in rows} >= {"completed", "failed"}
    for row in rows:
        assert row["endpoint_ratio_available"] == "false"
        assert row["endpoint_width_ratio_torch_over_flowstar"] == ""
        if row["flowstar_status"] == "completed":
            assert row["last_segment_ratio_available"] == "true"
            assert row["tube_ratio_available"] == "true"
        else:
            assert row["last_segment_ratio_available"] == "false"
            assert row["tube_ratio_available"] == "false"



def test_trajectory_crosscheck_summary_passes_required_comparisons():
    header, rows = _csv_rows(AUDIT / "crosscheck_summary.csv")
    assert set(header) >= {"comparison_id", "category", "metric", "case_id", "new_value", "old_value", "pass_fail", "note"}
    assert rows
    assert all(row["pass_fail"] == "pass" for row in rows)

    torch_rows = [row for row in rows if row["category"] == "torch_diagnostics"]
    flow_rows = [row for row in rows if row["category"] == "flowstar_sweep"]
    assert len(torch_rows) == 70
    assert len(flow_rows) == 15
    assert {row["metric"] for row in torch_rows} >= {
        "endpoint_width_sum_vs_final_width_sum",
        "remainder_width_sum",
        "poly_range_width_sum",
        "max_final_degree",
        "term_count_total",
    }
    assert {row["metric"] for row in flow_rows} >= {
        "status",
        "failure_reason",
        "num_segments",
        "last_segment_width_sum",
        "tube_width_sum",
    }
    failed_num_segments = [
        row for row in flow_rows
        if row["case_id"] == "flowstar_rem1e-4_cut1e-10_h0p01_s10_o2" and row["metric"] == "num_segments"
    ]
    assert failed_num_segments
    assert "failed case semantics" in failed_num_segments[0]["note"]

def test_trajectory_generated_cpp_and_runner_use_toolbox_api():
    cpp_paths = sorted((AUDIT / "flowstar_models").glob("*.cpp"))
    assert cpp_paths
    for path in cpp_paths:
        text = path.read_text(encoding="utf-8")
        assert '#include "Continuous.h"' in text
        assert "ode.reach(" in text
        assert "setting.setFixedStepsize(" in text
    runner = (ROOT / "comparisons" / "flowstar" / "run_flowstar.py").read_text(encoding="utf-8")
    assert "flowstar-toolbox" in runner
    assert '"-lflowstar"' in runner


def test_trajectory_torch_segments_samples_and_summary_consistency():
    _header, rows = _csv_rows(AUDIT / "torch_structured_summary.csv")
    assert len(rows) == 16
    for row in rows:
        seg_path = AUDIT / "torch_segments" / f"{row['case_id']}_segments.csv"
        sample_path = AUDIT / "samples" / f"{row['case_id']}_samples.csv"
        _seg_header, segments = _csv_rows(seg_path)
        _sample_header, samples = _csv_rows(sample_path)
        steps = int(row["steps"])
        assert len(segments) == steps
        assert row["endpoint_box_available"] == "true"
        assert samples
        assert {"corner", "center", "grid5x5"} <= {s["sample_kind"] for s in samples}

        prev_hi = None
        for i, seg in enumerate(segments):
            assert int(seg["segment_index"]) == i
            t_lo = _as_float(seg["t_lo"])
            t_hi = _as_float(seg["t_hi"])
            assert t_hi > t_lo
            if prev_hi is not None:
                assert t_lo >= prev_hi - 1e-12
            prev_hi = t_hi
            assert _as_float(seg["width_x"]) >= 0.0
            assert _as_float(seg["width_y"]) >= 0.0
            assert math.isclose(_as_float(seg["width_sum"]), _as_float(seg["width_x"]) + _as_float(seg["width_y"]), rel_tol=1e-10, abs_tol=1e-12)

        last = segments[-1]
        assert math.isclose(_as_float(row["last_segment_width_sum"]), _as_float(last["width_sum"]), rel_tol=1e-10, abs_tol=1e-12)
        tube_x = max(_as_float(s["x_hi"]) for s in segments) - min(_as_float(s["x_lo"]) for s in segments)
        tube_y = max(_as_float(s["y_hi"]) for s in segments) - min(_as_float(s["y_lo"]) for s in segments)
        assert math.isclose(_as_float(row["tube_width_x"]), tube_x, rel_tol=1e-10, abs_tol=1e-12)
        assert math.isclose(_as_float(row["tube_width_y"]), tube_y, rel_tol=1e-10, abs_tol=1e-12)


def test_trajectory_required_pngs_exist_and_are_real_images():
    required = [
        "torch_range_only_h0p01_s10_o4_torch_phase_xy.png",
        "torch_range_only_h0p01_s10_o4_torch_t_x.png",
        "torch_range_only_h0p01_s10_o4_torch_t_y.png",
        "torch_range_only_h0p01_s10_o4_torch_width_over_time.png",
        "torch_modes_h0p01_s10_o4_torch_modes_overlay_phase_xy.png",
        "flowstar_rem1e-4_cut1e-10_h0p01_s10_o4_overlay_phase_xy.png",
        "flowstar_rem1e-4_cut1e-10_h0p01_s10_o4_overlay_t_x.png",
        "flowstar_rem1e-4_cut1e-10_h0p01_s10_o4_overlay_t_y.png",
        "flowstar_rem1e-4_cut1e-10_h0p01_s10_o4_overlay_width_over_time.png",
        "flowstar_rem1e-10_cut1e-15_h0p0025_s10_o8_overlay_phase_xy.png",
        "flowstar_rem1e-10_cut1e-15_h0p0025_s10_o8_overlay_t_x.png",
        "flowstar_rem1e-10_cut1e-15_h0p0025_s10_o8_overlay_t_y.png",
        "flowstar_rem1e-10_cut1e-15_h0p0025_s10_o8_overlay_width_over_time.png",
        "contact_sheet_torch_orders.png",
        "contact_sheet_flowstar_overlays.png",
        "contact_sheet_width_trends.png",
    ]
    for name in required:
        path = AUDIT / "figures" / name
        data = path.read_bytes()
        assert len(data) > 1000
        assert data.startswith(b"\x89PNG\r\n\x1a\n")
        width, height = struct.unpack(">II", data[16:24])
        assert width > 100
        assert height > 100


def test_trajectory_docs_are_utf8_lf_without_cr():
    for path in [
        ROOT / "docs" / "trajectory_visual_audit.md",
        AUDIT / "README.md",
        AUDIT / "visual_audit_report.md",
        AUDIT / "crosscheck_summary.md",
    ]:
        data = path.read_bytes()
        data.decode("utf-8")
        assert b"\r" not in data
        assert data.endswith(b"\n")


def test_trajectory_audit_new_surface_scope_guard():
    files = [
        ROOT / "experiments" / "trajectory_visual_audit.py",
        ROOT / "docs" / "trajectory_visual_audit.md",
        AUDIT / "README.md",
        AUDIT / "visual_audit_report.md",
        AUDIT / "crosscheck_summary.md",
    ]
    banned = [
        "CROWN",
        "auto_LiRPA",
        "Jacobian",
        "sin/cos",
        "hybrid",
        "Flow* Python binding",
        "NN controller",
    ]
    allowed_markers = ["not", "no ", "disabled", "outside", "does not"]
    for path in files:
        for line in path.read_text(encoding="utf-8").splitlines():
            low = line.lower()
            for token in banned:
                if token.lower() in low:
                    assert any(marker in low for marker in allowed_markers), f"{path}: {line}"
