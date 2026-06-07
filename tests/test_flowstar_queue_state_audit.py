from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments" / "flowstar_queue_state_audit.py"


def _write_source(root: Path, dirname: str, summary_name: str, segment_name: str, run_id: str, label: str) -> None:
    out = root / "outputs" / dirname
    out.mkdir(parents=True)
    out.joinpath(summary_name).write_text(
        "run_id,status,last_validated_t\n"
        f"{run_id},failed,0.02\n",
        encoding="utf-8",
    )
    out.joinpath(segment_name).write_text(
        "run_id,status,segment_index,t_hi,queue_size,j_count,phi_l_count,reset_box_width_sum,right_map_range_width_sum,"
        "target_check_width_sum,output_only_symbolic_width_sum,output_range_includes_symbolic_contributions\n"
        f"{run_id},validated,0,0.01,1,1,1,1.0,2.0,0.0,{0.1 if label == 'v2' else 0.0},true\n",
        encoding="utf-8",
    )


def _write_three_way_fixture(root: Path) -> None:
    _write_source(
        root,
        "flowstar_normalized_insertion_h10",
        "normalized_insertion_h10_summary.csv",
        "normalized_insertion_h10_segments.csv",
        "flowstar_style_o4_target_insert",
        "no",
    )
    _write_source(
        root,
        "flowstar_normalized_insertion_symqueue_split_h10",
        "symqueue_split_summary.csv",
        "symqueue_split_segments.csv",
        "flowstar_style_o4_target_insert_symqueue_split",
        "split",
    )
    _write_source(
        root,
        "flowstar_normalized_insertion_symqueue_v2_h10",
        "symqueue_v2_summary.csv",
        "symqueue_v2_segments.csv",
        "flowstar_style_o4_target_insert_symqueue_v2",
        "v2",
    )


def test_queue_state_audit_includes_all_three_sources(tmp_path):
    _write_three_way_fixture(tmp_path)
    out_dir = tmp_path / "audit"

    subprocess.run(
        [sys.executable, str(SCRIPT), "--repo-root", str(tmp_path), "--out-dir", str(out_dir)],
        check=True,
    )

    trace = pd.read_csv(out_dir / "queue_state_trace.csv")
    summary = pd.read_csv(out_dir / "queue_state_summary.csv")
    report = (out_dir / "queue_state_report.md").read_text(encoding="utf-8")
    assert set(trace["source"]) == {"no_queue", "split_symqueue", "v2_symqueue"}
    assert set(summary["source"]) == {"no_queue", "split_symqueue", "v2_symqueue"}
    assert "Did v2 reach h10?" in report
    assert "complete three-way input set loaded" in report


def test_queue_state_audit_reports_missing_v2_explicitly(tmp_path):
    _write_source(
        tmp_path,
        "flowstar_normalized_insertion_h10",
        "normalized_insertion_h10_summary.csv",
        "normalized_insertion_h10_segments.csv",
        "flowstar_style_o4_target_insert",
        "no",
    )
    _write_source(
        tmp_path,
        "flowstar_normalized_insertion_symqueue_split_h10",
        "symqueue_split_summary.csv",
        "symqueue_split_segments.csv",
        "flowstar_style_o4_target_insert_symqueue_split",
        "split",
    )
    out_dir = tmp_path / "audit"

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--repo-root",
            str(tmp_path),
            "--out-dir",
            str(out_dir),
            "--include-missing",
        ],
        check=True,
    )

    summary = pd.read_csv(out_dir / "queue_state_summary.csv")
    report = (out_dir / "queue_state_report.md").read_text(encoding="utf-8")
    missing = summary.loc[summary["source"] == "v2_symqueue"].iloc[0]
    assert missing["status"] == "missing"
    assert "flowstar_normalized_insertion_symqueue_v2_h10" in missing["missing_paths"]
    assert "Audit status: incomplete" in report
    assert "v2_symqueue" in report
