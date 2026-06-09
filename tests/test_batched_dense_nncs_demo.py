from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "batched_dense_nncs_demo",
        ROOT / "experiments" / "batched_dense_nncs_demo.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_nncs_demo_affine_and_relu_tiny_cpu_write_outputs_and_contain_samples(tmp_path, monkeypatch):
    monkeypatch.delenv("FLOWSTAR_ROOT", raising=False)
    demo = _load_module()
    summary, report, rows, recommendation = demo.run_experiment(
        tmp_path,
        batches=[1, 2],
        num_control_steps_list=[2],
        plant_substeps_list=[2],
        controllers=["affine", "relu_ibp"],
        devices=["cpu"],
        dtype=torch.float64,
        order=2,
        h=0.01,
    )

    assert summary.exists()
    assert report.exists()
    assert recommendation in {"GPU_PATH_CONTINUE", "NEEDS_REMAINDER_REDESIGN"}
    ok_rows = [row for row in rows if row["status"] == "ok"]
    assert ok_rows
    assert {row["controller"] for row in ok_rows} == {"affine", "relu_ibp"}
    assert all(row["containment_pass"] for row in ok_rows)
    assert all(int(row["sample_violations"]) == 0 for row in ok_rows)

    csv_rows = _rows(summary)
    assert csv_rows
    assert {"batch", "controller", "controller_bound_ms", "plant_step_ms", "containment_pass"} <= set(csv_rows[0])
    text = report.read_text(encoding="utf-8")
    assert "End-to-end CPU run: yes" in text
    assert "Closed-loop sampled containment: pass" in text


def test_nncs_controller_bound_shapes():
    demo = _load_module()
    dtype = torch.float64
    device = torch.device("cpu")
    params = demo._controller_params(dtype, device)
    lo = torch.tensor([[-0.2, -0.1], [0.1, -0.3]], dtype=dtype)
    hi = torch.tensor([[0.3, 0.2], [0.4, 0.1]], dtype=dtype)
    u_lo, u_hi = demo._relu_ibp(lo, hi, params)
    assert u_lo.shape == u_hi.shape == (2, 1)
    assert bool(torch.all(u_lo <= u_hi))
    samples = torch.stack([lo, hi, 0.5 * (lo + hi)], dim=1)
    affine_u = demo._exact_controller_samples(samples, "affine", params)
    relu_u = demo._exact_controller_samples(samples, "relu_ibp", params)
    assert affine_u.shape == relu_u.shape == (2, 3)


@torch.no_grad()
def test_nncs_demo_cuda_smoke_if_available(tmp_path):
    if not torch.cuda.is_available():
        return
    demo = _load_module()
    _summary, _report, rows, _recommendation = demo.run_experiment(
        tmp_path,
        batches=[1],
        num_control_steps_list=[1],
        plant_substeps_list=[1],
        controllers=["affine"],
        devices=["cpu", "cuda"],
        dtype=torch.float64,
        order=2,
        h=0.01,
    )
    cuda_rows = [row for row in rows if row.get("device") == "cuda" and row.get("status") == "ok"]
    assert cuda_rows
    assert all(row["containment_pass"] for row in cuda_rows)
