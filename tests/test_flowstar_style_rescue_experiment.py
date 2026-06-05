import csv
import importlib.util
import py_compile

import pandas as pd
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments" / "flowstar_style_rescue_vanderpol.py"
LOCALIZATION_SCRIPT = ROOT / "experiments" / "flowstar_style_failure_localization.py"
ORACLE_SCRIPT = ROOT / "experiments" / "flowstar_one_step_oracle.py"
FLOWPIPE = ROOT / "src" / "torch_tm_flowpipe" / "flowpipe.py"
EXPECTED_RUN_IDS = {
    "baseline_range_only_o6_s4",
    "baseline_dependency_preserving_o4_s1",
    "flowstar_style_o4_target",
    "flowstar_style_o6_target",
    "flowstar_style_o4_target_cutoff",
    "flowstar_style_o6_target_cutoff",
}


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _load_experiment_module():
    spec = importlib.util.spec_from_file_location("flowstar_style_rescue_vanderpol", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_flowstar_style_rescue_experiment_smoke_selected_config(tmp_path):
    out_dir = tmp_path / "rescue"

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--out-dir",
            str(out_dir),
            "--max-horizon",
            "0.02",
            "--wall-cap-s",
            "60",
            "--configs",
            "flowstar_style_o6_target",
        ],
        check=True,
    )

    required = [
        "rescue_summary.csv",
        "rescue_segments.csv",
        "rescue_validation_attempts.csv",
        "rescue_report.md",
        "rescue_t_x.png",
        "rescue_t_y.png",
        "rescue_phase_xy.png",
        "step_size_trace.png",
        "residual_vs_t.png",
    ]
    for name in required:
        assert (out_dir / name).exists()

    rows = _csv_rows(out_dir / "rescue_summary.csv")
    assert [row["run_id"] for row in rows] == ["flowstar_style_o6_target"]
    for column in [
        "min_h_used",
        "min_regular_h_used",
        "min_final_alignment_h",
        "h_below_flowstar_min_count",
        "max_h_used",
        "num_step_rejections",
        "num_accepted_steps",
        "num_rejected_steps",
    ]:
        assert column in rows[0]

    report = (out_dir / "rescue_report.md").read_text(encoding="utf-8")
    assert report.count("\n") > 10
    assert "Did flowstar_style_o6_target reach the requested horizon?" in report
    assert "min_regular_h_used" in report


def test_committed_rescue_artifacts_are_multiline_and_parseable():
    report_paths = [
        ROOT / "outputs" / "flowstar_style_rescue" / "rescue_report.md",
        ROOT / "outputs" / "flowstar_style_rescue_h2" / "rescue_report.md",
    ]
    for path in report_paths:
        text = path.read_text(encoding="utf-8")
        assert text.startswith("# Flowstar-Style Rescue Report")
        assert text.count("\n") > 10
        assert "## Summary Rows" in text

    summary_paths = [
        ROOT / "outputs" / "flowstar_style_rescue" / "rescue_summary.csv",
        ROOT / "outputs" / "flowstar_style_rescue_h2" / "rescue_summary.csv",
    ]
    for path in summary_paths:
        rows = _csv_rows(path)
        assert len(rows) >= 6
        assert {row["run_id"] for row in rows} >= EXPECTED_RUN_IDS


def test_flowstar_style_rescue_script_is_py_compileable():
    py_compile.compile(str(SCRIPT), doraise=True)
    py_compile.compile(str(LOCALIZATION_SCRIPT), doraise=True)
    py_compile.compile(str(ORACLE_SCRIPT), doraise=True)
    py_compile.compile(str(FLOWPIPE), doraise=True)


def test_flowstar_overlap_comparison_does_not_require_segment_count_match():
    module = _load_experiment_module()
    summary = {
        "run_id": "flowstar_style_o6_target",
        "status": "max_horizon_reached",
        "runtime_s": 1.25,
        "last_validated_t": 1.0,
    }
    py_rows = [
        {"run_id": "flowstar_style_o6_target", "status": "validated", "t_lo": 0.0, "t_hi": 0.4, "x_lo": 0.0, "x_hi": 1.0, "y_lo": 0.0, "y_hi": 1.0, "width_x": 1.0, "width_y": 1.0, "width_sum": 2.0},
        {"run_id": "flowstar_style_o6_target", "status": "validated", "t_lo": 0.4, "t_hi": 1.0, "x_lo": 0.0, "x_hi": 2.0, "y_lo": 0.0, "y_hi": 2.0, "width_x": 2.0, "width_y": 2.0, "width_sum": 4.0},
    ]
    flow_rows = [
        {"t_lo": 0.0, "t_hi": 1.0, "x_lo": 0.0, "x_hi": 4.0, "y_lo": 0.0, "y_hi": 4.0, "width_x": 4.0, "width_y": 4.0, "width_sum": 8.0},
    ]

    row, ratio_rows = module._comparison_row(summary, py_rows, flow_rows)

    assert row["flowstar_segments_over_same_horizon"] == 1
    assert row["py_segments"] == 2
    assert row["last_width_ratio"] == 0.5
    assert row["tube_width_ratio"] == 0.5
    assert len(ratio_rows) == 2



def test_h5_rescue_artifacts_are_multiline_and_parseable():
    out_dir = ROOT / "outputs" / "flowstar_style_rescue_h5"
    summary_rows = _csv_rows(out_dir / "rescue_summary.csv")
    comparison_rows = _csv_rows(out_dir / "rescue_vs_flowstar_comparison.csv")

    assert len(summary_rows) == 2
    assert [row["run_id"] for row in summary_rows] == [
        "flowstar_style_o6_target",
        "flowstar_style_o6_target_cutoff",
    ]
    assert len(comparison_rows) == 2
    assert [row["run_id"] for row in comparison_rows] == [
        "flowstar_style_o6_target",
        "flowstar_style_o6_target_cutoff",
    ]

    for name in ["rescue_report.md", "rescue_vs_flowstar_report.md"]:
        text = (out_dir / name).read_text(encoding="utf-8")
        assert text.count("\n") > 10
        assert "\\n" not in text


def test_flowstar_source_rescue_notes_are_multiline():
    text = (ROOT / "docs" / "flowstar_source_rescue_notes.md").read_text(encoding="utf-8")
    assert text.startswith("# Flow* Source Rescue Notes")
    assert text.count("\n") > 30
    assert "\\n" not in text



def test_requested_flowstar_style_artifacts_are_multiline_and_pandas_parseable():
    artifact_dirs = sorted(p for p in (ROOT / "outputs").glob("flowstar_style_*") if p.is_dir())
    text_paths = [ROOT / "docs" / "flowstar_source_rescue_notes.md"]
    csv_paths = []
    for out_dir in artifact_dirs:
        csv_paths.extend(sorted(out_dir.glob("*.csv")))
        text_paths.extend(sorted(out_dir.glob("*.md")))

    assert csv_paths
    assert text_paths
    for path in csv_paths:
        frame = pd.read_csv(path)
        text = path.read_text(encoding="utf-8")
        physical_lines = text.splitlines()
        assert text.endswith("\n")
        assert text.count("\n") >= len(frame) + 1
        assert len(physical_lines) >= len(frame) + 1
        if path.name.endswith("reset_boxes.csv"):
            assert "validation_mode" in frame.columns
    for path in text_paths:
        text = path.read_text(encoding="utf-8")
        assert text.count("\n") > 5
        assert "\\n" not in text


def test_adaptive_order_report_clarifies_aggregate_order8_count():
    out_dir = ROOT / "outputs" / "flowstar_style_rescue_adaptive_order"
    rows = _csv_rows(out_dir / "adaptive_order_summary.csv")
    report = (out_dir / "adaptive_order_report.md").read_text(encoding="utf-8")
    total_order8 = sum(int(row.get("num_order8_steps") or 0) for row in rows)

    assert total_order8 == 88
    assert "Across all configs" in report
    assert "not a single-run step count" in report


def test_candidate_order_specialized_outputs_smoke(tmp_path):
    out_dir = tmp_path / "flowstar_style_candidate_order"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--out-dir",
            str(out_dir),
            "--max-horizon",
            "0.02",
            "--wall-cap-s",
            "60",
            "--configs",
            "flowstar_style_o6_candidate8_output6",
        ],
        check=True,
    )

    assert (out_dir / "candidate_order_summary.csv").exists()
    assert (out_dir / "candidate_order_segments.csv").exists()
    report = (out_dir / "candidate_order_report.md").read_text(encoding="utf-8")
    assert "candidate_order=8/output_order=6" in report
    rows = _csv_rows(out_dir / "candidate_order_summary.csv")
    assert rows[0]["candidate_order"] == "8"
    assert rows[0]["output_order"] == "6"


def test_residual_centering_specialized_outputs_smoke(tmp_path):
    out_dir = tmp_path / "flowstar_style_residual_centering"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--out-dir",
            str(out_dir),
            "--max-horizon",
            "0.02",
            "--wall-cap-s",
            "60",
            "--configs",
            "flowstar_style_o6_target_centered",
        ],
        check=True,
    )

    for name in [
        "residual_centering_summary.csv",
        "residual_centering_segments.csv",
        "residual_centering_attempts.csv",
        "residual_centering_report.md",
    ]:
        assert (out_dir / name).exists()
    rows = _csv_rows(out_dir / "residual_centering_summary.csv")
    assert rows[0]["validation_mode"] == "target_remainder_centered"
    assert "center_corrections_applied" in rows[0]
    attempts = pd.read_csv(out_dir / "residual_centering_attempts.csv")
    assert "residual_before_lo_y" in attempts.columns
    report = (out_dir / "residual_centering_report.md").read_text(encoding="utf-8")
    assert report.count("\n") > 5
    assert "recomputing the Picard residual" in report


def test_selective_terms_specialized_outputs_smoke(tmp_path):
    out_dir = tmp_path / "flowstar_style_selective_terms"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--out-dir",
            str(out_dir),
            "--max-horizon",
            "0.02",
            "--wall-cap-s",
            "60",
            "--configs",
            "flowstar_style_o6_candidate8_output6_keep1",
        ],
        check=True,
    )

    for name in [
        "selective_terms_summary.csv",
        "selective_terms_segments.csv",
        "selective_terms_report.md",
        "retained_terms_near_failure.csv",
    ]:
        assert (out_dir / name).exists()
    rows = _csv_rows(out_dir / "selective_terms_summary.csv")
    assert rows[0]["selective_high_degree_terms_top_k"] == "1"
    segments = pd.read_csv(out_dir / "selective_terms_segments.csv")
    assert "selective_retained_terms_count" in segments.columns
    retained = pd.read_csv(out_dir / "retained_terms_near_failure.csv")
    assert "retained" in retained.columns
    report = (out_dir / "selective_terms_report.md").read_text(encoding="utf-8")
    assert report.count("\n") > 5
    assert "diagnostic-only" in report



def test_ctrunc_validation_specialized_outputs_smoke(tmp_path):
    out_dir = tmp_path / "flowstar_style_ctrunc_validation"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--out-dir",
            str(out_dir),
            "--max-horizon",
            "0.02",
            "--wall-cap-s",
            "60",
            "--configs",
            "flowstar_style_o6_target_flowstar_ctrunc",
        ],
        check=True,
    )

    for name in [
        "rescue_reset_boxes.csv",
        "ctrunc_validation_summary.csv",
        "ctrunc_validation_segments.csv",
        "ctrunc_validation_attempts.csv",
        "ctrunc_validation_report.md",
    ]:
        assert (out_dir / name).exists()
    reset_boxes = pd.read_csv(out_dir / "rescue_reset_boxes.csv")
    assert "validation_mode" in reset_boxes.columns
    attempts = pd.read_csv(out_dir / "ctrunc_validation_attempts.csv")
    assert "tmp_remainder_lo_x" in attempts.columns
    assert "subset_tmp_remainder" in attempts.columns
    report = (out_dir / "ctrunc_validation_report.md").read_text(encoding="utf-8")
    assert report.count("\n") > 5
    assert "flowstar_ctrunc validation" in report


def test_selective_validation_path_audit_outputs_smoke(tmp_path):
    out_dir = tmp_path / "flowstar_style_selective_terms"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--out-dir",
            str(out_dir),
            "--max-horizon",
            "0.02",
            "--wall-cap-s",
            "60",
            "--configs",
            "flowstar_style_o6_candidate8_output6_keep4",
        ],
        check=True,
    )

    terms = pd.read_csv(out_dir / "validation_path_terms.csv")
    assert {"before_validation", "after_selective", "inside_validation"} <= set(terms["stage"])
    assert "terms_hash" in terms.columns
    audit = (out_dir / "validation_path_audit.md").read_text(encoding="utf-8")
    assert "Selective Validation Path Audit" in audit
