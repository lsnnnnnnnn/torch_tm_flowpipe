from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "batched_dense_many_box_flowpipe_demo",
        ROOT / "experiments" / "batched_dense_many_box_flowpipe_demo.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_many_box_demo_tiny_cpu_writes_summary_report_and_contains_samples(tmp_path, monkeypatch):
    monkeypatch.delenv("FLOWSTAR_ROOT", raising=False)
    demo = _load_module()
    summary, report, rows, recommendation = demo.run_experiment(
        tmp_path,
        batches=[1, 2],
        steps_list=[2],
        orders=[2],
        h=0.01,
        devices=["cpu"],
        dtype=torch.float64,
        range_bound_modes=["interval"],
        dropped_merge_modes=["termwise", "merged"],
        scalar_cap=2,
    )

    assert summary.exists()
    assert report.exists()
    assert recommendation in demo.RECOMMENDATIONS
    dense_rows = [row for row in rows if row["implementation"] == "torch_dense" and row["status"] == "ok"]
    assert dense_rows
    assert all(row["containment_pass"] for row in dense_rows)
    assert all(int(row["sample_violations"]) == 0 for row in dense_rows)

    csv_rows = _rows(summary)
    assert csv_rows
    assert {"batch", "device", "range_bound_mode", "dropped_merge_mode", "containment_pass"} <= set(csv_rows[0])
    text = report.read_text(encoding="utf-8")
    assert "Does dense CPU beat scalar loop?" in text
    assert "Sampled trajectory containment: pass" in text


def test_many_box_demo_split_mode_tiny_cpu_contains_samples(tmp_path, monkeypatch):
    monkeypatch.delenv("FLOWSTAR_ROOT", raising=False)
    demo = _load_module()
    _summary, _report, rows, _recommendation = demo.run_experiment(
        tmp_path,
        batches=[1],
        steps_list=[1],
        orders=[2],
        h=0.01,
        devices=["cpu"],
        dtype=torch.float64,
        range_bound_modes=["interval", "split2"],
        dropped_merge_modes=["merged"],
        scalar_cap=1,
    )

    dense_rows = [row for row in rows if row["implementation"] == "torch_dense" and row["status"] == "ok"]
    assert len(dense_rows) == 2
    assert all(row["containment_pass"] for row in dense_rows)
    split_rows = [row for row in dense_rows if row["range_bound_mode"] == "split2"]
    assert split_rows
    assert "width_ratio_vs_interval" in split_rows[0]


@torch.no_grad()
def test_many_box_demo_cuda_smoke_if_available(tmp_path):
    if not torch.cuda.is_available():
        return
    demo = _load_module()
    _summary, _report, rows, _recommendation = demo.run_experiment(
        tmp_path,
        batches=[1],
        steps_list=[1],
        orders=[2],
        h=0.01,
        devices=["cpu", "cuda"],
        dtype=torch.float64,
        range_bound_modes=["interval"],
        dropped_merge_modes=["merged"],
        scalar_cap=1,
    )
    cuda_rows = [row for row in rows if row.get("device") == "cuda" and row.get("status") == "ok"]
    assert cuda_rows
    assert all(row["containment_pass"] for row in cuda_rows)



def test_checked_in_many_box_artifacts_are_physical_multiline_files():
    summary = ROOT / "outputs" / "batched_dense_many_box_flowpipe_demo" / "many_box_summary.csv"
    report = ROOT / "outputs" / "batched_dense_many_box_flowpipe_demo" / "many_box_report.md"
    bline_report = ROOT / "docs" / "batched_dense_gpu_b_line_report.md"
    flowstar_boundary = ROOT / "docs" / "flowstar_archeology_boundary_report.md"

    summary_lines = summary.read_text(encoding="utf-8").splitlines()
    report_lines = report.read_text(encoding="utf-8").splitlines()
    bline_lines = bline_report.read_text(encoding="utf-8").splitlines()
    boundary_lines = flowstar_boundary.read_text(encoding="utf-8").splitlines()

    assert len(summary_lines) > 10
    assert len(report_lines) > 10
    assert len(bline_lines) > 10
    assert len(boundary_lines) > 10
    assert report_lines[0] == "# Batched Dense Many-Box Plant Demo Report"
    assert "## Direct Answers" in report_lines
    assert "## Many-Box Plant Demo" in bline_lines
    assert "## Executive Conclusion" in boundary_lines
    assert any(line == "- First CUDA win batch: 512" for line in report_lines)
    assert any("GPU_PATH_CONTINUE" in line for line in report_lines)
    assert any(line == "Final B-line decision: `GPU_PATH_CONTINUE`." for line in bline_lines)
