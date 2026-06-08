from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def _load_microbench_module():
    spec = importlib.util.spec_from_file_location(
        "batched_tm_gpu_microbench",
        ROOT / "experiments" / "batched_tm_gpu_microbench.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_batched_tm_gpu_microbench_writes_csv_and_report(tmp_path):
    bench = _load_microbench_module()
    setting = bench.BenchmarkSetting("tiny_dim2_order2", 2, 2, "test_dense_total_degree")

    summary, report, rows = bench.run_benchmark(
        tmp_path,
        batch_sizes=[1, 2],
        settings=[setting],
        devices=["cpu"],
        dtype=torch.float64,
        warmup=0,
        repeats=1,
        max_working_bytes=64 * 1024 * 1024,
        max_scalar_batch=2,
        include_scalar=True,
    )

    assert summary.exists()
    assert report.exists()
    assert rows

    csv_rows = _rows(summary)
    assert csv_rows
    assert {"case_name", "batch", "operation", "implementation", "status", "median_ms"} <= set(csv_rows[0])

    ok_torch_ops = {
        row["operation"]
        for row in csv_rows
        if row["implementation"] == "torch_dense" and row["device"] == "cpu" and row["status"] == "ok"
    }
    assert set(bench.CORE_OPERATIONS) <= ok_torch_ops

    report_text = report.read_text(encoding="utf-8")
    assert "At batch=1, is PyTorch GPU slower than CPU?" in report_text
    assert "Final Recommendation:" in report_text
    assert any(choice in report_text for choice in bench.RECOMMENDATIONS)


def test_batched_tm_gpu_microbench_records_skips_without_speed_thresholds(tmp_path):
    bench = _load_microbench_module()
    setting = bench.BenchmarkSetting("tiny_dim2_order2", 2, 2, "test_dense_total_degree")

    summary, _report, _rows_out = bench.run_benchmark(
        tmp_path,
        batch_sizes=[1, 8],
        settings=[setting],
        devices=["cpu"],
        dtype=torch.float64,
        warmup=0,
        repeats=1,
        max_working_bytes=64 * 1024 * 1024,
        max_scalar_batch=1,
        include_scalar=True,
    )

    csv_rows = _rows(summary)
    skipped_scalar = [
        row
        for row in csv_rows
        if row["implementation"] == "python_scalar_sparse" and row["batch"] == "8"
    ]
    assert skipped_scalar
    assert all(row["status"] == "skipped" for row in skipped_scalar)
    assert all("max_scalar_batch" in row["skip_reason"] for row in skipped_scalar)

