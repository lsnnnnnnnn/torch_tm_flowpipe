from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments" / "flowstar_vdp_width_trajectory_audit.py"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _minimal_fixture(root: Path) -> None:
    _write(
        root / "outputs/flowstar_benchmark_parity/generated_flowstar_vs_original_comparison.csv",
        "metric,value\n"
        "original_num_segments,2\n"
        "generated_num_segments,2\n"
        "segment_count_match,true\n"
        "max_abs_segment_field_diff,0\n",
    )
    _write(
        root / "outputs/flowstar_benchmark_parity/parity_summary.csv",
        "tool,status,num_segments,last_validated_t,last_attempted_t,last_segment_width_sum,tube_width_sum,endpoint_box_available,generated_flowstar_internal_reach_s,original_flowstar_wall_run_s\n"
        "original_flowstar,completed,2,10,10,0.7,9.5,false,,1.0\n"
        "generated_flowstar,completed,2,10,10,0.7,9.5,false,0.5,\n",
    )
    _write(
        root / "outputs/trajectory_audit/flowstar_vs_torch_overlay_summary.csv",
        "case_id,system,h,steps,horizon,order,setting_label,torch_mode,flowstar_status,torch_status,last_segment_ratio_available,last_segment_width_ratio_torch_over_flowstar,tube_ratio_available,tube_width_ratio_torch_over_flowstar,endpoint_ratio_available,endpoint_width_ratio_torch_over_flowstar,ratio_note\n"
        "toy_o4,van_der_pol,0.01,10,0.1,4,loose,range_only,completed,validated,true,1.1,true,1.05,false,,endpoint ratio disabled because Flow* GNUPLOT boxes are segment boxes\n",
    )
    _write(
        root / "outputs/flowstar_step_trace_compare/aligned_trace_diff.csv",
        "step_index,t_flowstar,t_noqueue,t_v2,flowstar_h,noqueue_h,v2_h,first_material_channel\n"
        "0,0,0,0,0.0125,0.025,0.025,center/scaling\n",
    )


def _run(root: Path, out_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--repo-root", str(root), "--out-dir", str(out_dir), "--no-strict-missing"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def test_endpoint_ratio_disabled_when_flowstar_endpoint_missing(tmp_path):
    _minimal_fixture(tmp_path)
    out_dir = tmp_path / "audit"

    _run(tmp_path, out_dir)

    ledger = _rows(out_dir / "width_comparison_ledger.csv")
    trajectory = [row for row in ledger if row["family"] == "trajectory_visual_audit"]
    assert trajectory
    assert all(row["endpoint_ratio_allowed"] == "false" for row in trajectory)
    assert all(row["endpoint_width_ratio"] == "" for row in trajectory)

    checks = {row["check_id"]: row for row in _rows(out_dir / "claim_boundary_checks.csv")}
    assert checks["endpoint_ratio_disabled_without_flowstar_endpoint"]["status"] == "pass"


def test_report_includes_exact_flowstar_parity_section(tmp_path):
    _minimal_fixture(tmp_path)
    out_dir = tmp_path / "audit"

    _run(tmp_path, out_dir)

    report = (out_dir / "report.md").read_text(encoding="utf-8")
    assert "What is already exact?" in report
    assert "Original Flow* vs generated Flow* parity" in report
    assert "max_abs_segment_field_diff=`0`" in report


def test_missing_artifacts_are_reported_explicitly(tmp_path):
    _minimal_fixture(tmp_path)
    out_dir = tmp_path / "audit"

    _run(tmp_path, out_dir)

    inventory_md = (out_dir / "evidence_inventory.md").read_text(encoding="utf-8")
    assert "Missing Paths" in inventory_md
    assert "outputs/flowstar_normalized_insertion_h10/" in inventory_md

    checks = {row["check_id"]: row for row in _rows(out_dir / "claim_boundary_checks.csv")}
    assert checks["missing_artifacts_recorded"]["status"] == "pass"


def test_accepted_step_h_mismatch_marked_noncausal(tmp_path):
    _minimal_fixture(tmp_path)
    out_dir = tmp_path / "audit"

    _run(tmp_path, out_dir)

    checks = {row["check_id"]: row for row in _rows(out_dir / "claim_boundary_checks.csv")}
    mismatch = checks["accepted_step_h_or_t_mismatch_noncausal"]
    assert mismatch["status"] == "noncausal_guarded"
    assert "adaptive_step_alignment_mismatch" in mismatch["details"]

    report = (out_dir / "report.md").read_text(encoding="utf-8")
    assert "accepted_ordinal_trace_diff_noncausal" in report
    assert "adaptive_step_alignment_mismatch" in report


def test_cli_writes_required_outputs(tmp_path):
    _minimal_fixture(tmp_path)
    out_dir = tmp_path / "audit"

    result = _run(tmp_path, out_dir)

    assert "Wrote Van der Pol width/trajectory audit" in result.stdout
    for name in [
        "summary.csv",
        "width_comparison_ledger.csv",
        "trajectory_overlay_ledger.csv",
        "claim_boundary_checks.csv",
        "report.md",
    ]:
        assert (out_dir / name).exists(), name
