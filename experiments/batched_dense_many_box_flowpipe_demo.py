#!/usr/bin/env python3
"""Many-box dense batched Taylor-model plant demo.

This is a B-line GPU workload experiment. It is plant-only, fixed-step,
fixed-order, explicit Euler-style Taylor-model propagation. It does not call
Flow*, does not implement adaptive rejection, and does not claim Flow* parity.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from torch_tm_flowpipe import Interval, TaylorModel  # noqa: E402
from torch_tm_flowpipe.batched_dense_tm import (  # noqa: E402
    BatchedMonomialBasis,
    BatchedTaylorModel,
    DenseTMProfiler,
)

SUMMARY_FIELDS = [
    "plant",
    "batch",
    "device",
    "implementation",
    "steps",
    "order",
    "h",
    "dtype",
    "range_bound_mode",
    "dropped_merge_mode",
    "status",
    "skip_reason",
    "elapsed_ms",
    "basis_mul_plan_ms",
    "rhs_construction_ms",
    "mul_trunc_ms",
    "dropped_range_bound_ms",
    "tm_remainder_propagation_ms",
    "range_bound_ms",
    "cuda_memory_allocated_bytes",
    "cuda_memory_reserved_bytes",
    "samples_checked",
    "sample_violations",
    "containment_pass",
    "max_width",
    "mean_width",
    "speedup_vs_scalar_cpu",
    "speedup_vs_dense_cpu",
    "width_ratio_vs_termwise",
    "width_ratio_vs_interval",
    "dominant_operation",
    "recommendation",
]

RECOMMENDATIONS = {"GPU_PATH_CONTINUE", "NEEDS_REMAINDER_REDESIGN", "STOP_DENSE_PLANT"}


def _split_csv_args(values: Sequence[Any] | Any | None) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        values = [values]
    parts: list[str] = []
    for value in values:
        parts.extend(part.strip() for part in str(value).split(",") if part.strip())
    return parts


def _parse_ints(values: Sequence[Any] | Any | None, default: Sequence[int]) -> list[int]:
    parts = _split_csv_args(values)
    return [int(part) for part in parts] if parts else list(default)


def _parse_strings(values: Sequence[Any] | Any | None, default: Sequence[str], allowed: set[str]) -> list[str]:
    parts = _split_csv_args(values) or list(default)
    bad = sorted(set(parts) - allowed)
    if bad:
        raise SystemExit(f"unsupported value(s): {', '.join(bad)}")
    return parts


def _dtype_from_name(name: str) -> torch.dtype:
    if name == "float64":
        return torch.float64
    if name == "float32":
        return torch.float32
    raise SystemExit("dtype must be float64 or float32")


def _format(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.9g}"
    return value


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _format(row.get(field, "")) for field in SUMMARY_FIELDS})


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _make_domains(batch: int, dtype: torch.dtype, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    idx = torch.arange(batch, dtype=dtype, device=device)
    phase_a = torch.remainder(idx * 7.0, 17.0) / 17.0
    phase_b = torch.remainder(idx * 5.0, 19.0) / 19.0
    center_x = 1.15 + 0.10 * (phase_a - 0.5)
    center_y = 2.00 + 0.12 * (phase_b - 0.5)
    radius_x = torch.full_like(center_x, 0.015)
    radius_y = torch.full_like(center_y, 0.020)
    lo = torch.stack([center_x - radius_x, center_y - radius_y], dim=1)
    hi = torch.stack([center_x + radius_x, center_y + radius_y], dim=1)
    return lo, hi


def _selected_indices(batch: int, device: torch.device) -> torch.Tensor:
    count = min(batch, 16)
    if count == batch:
        return torch.arange(batch, device=device)
    return torch.linspace(0, batch - 1, count, device=device).round().to(torch.long).unique()


def _sample_points(lo: torch.Tensor, hi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    indices = _selected_indices(int(lo.shape[0]), lo.device)
    lo_s = lo.index_select(0, indices)
    hi_s = hi.index_select(0, indices)
    center = 0.5 * (lo_s + hi_s)
    base = [
        lo_s,
        hi_s,
        center,
        torch.stack([lo_s[:, 0], hi_s[:, 1]], dim=1),
        torch.stack([hi_s[:, 0], lo_s[:, 1]], dim=1),
    ]
    gen = torch.Generator(device="cpu")
    gen.manual_seed(2026)
    rand = torch.rand((lo_s.shape[0], 4, lo_s.shape[1]), generator=gen, dtype=lo_s.dtype).to(lo_s.device)
    random_points = lo_s[:, None, :] + rand * (hi_s - lo_s)[:, None, :]
    points = torch.cat([torch.stack(base, dim=1), random_points], dim=1)
    return indices, points


def _advance_vdp_samples(points: torch.Tensor, h: float, steps: int) -> torch.Tensor:
    state = points.clone()
    h_t = torch.as_tensor(h, dtype=points.dtype, device=points.device)
    for _ in range(int(steps)):
        x = state[..., 0]
        y = state[..., 1]
        state = torch.stack([x + h_t * y, y + h_t * (y - x - x * x * y)], dim=-1)
    return state


def _contains(range_lo: torch.Tensor, range_hi: torch.Tensor, samples: torch.Tensor, tol: float) -> tuple[bool, int]:
    below = samples < range_lo[:, None, :] - tol
    above = samples > range_hi[:, None, :] + tol
    violations = int(torch.count_nonzero(below | above).detach().cpu())
    return violations == 0, violations


def _dominant(row: dict[str, Any]) -> str:
    timings = {
        "basis/mul plan": float(row.get("basis_mul_plan_ms") or 0.0),
        "rhs construction": float(row.get("rhs_construction_ms") or 0.0),
        "mul_trunc": float(row.get("mul_trunc_ms") or 0.0),
        "dropped range bound": float(row.get("dropped_range_bound_ms") or 0.0),
        "TM remainder propagation": float(row.get("tm_remainder_propagation_ms") or 0.0),
        "range bound": float(row.get("range_bound_ms") or 0.0),
    }
    return max(timings, key=timings.get)


def _run_dense_vdp(
    *,
    batch: int,
    steps: int,
    order: int,
    h: float,
    dtype: torch.dtype,
    device: torch.device,
    range_bound_mode: str,
    dropped_merge_mode: str,
    tol: float,
) -> dict[str, Any]:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _sync(device)
    basis_start = time.perf_counter()
    basis = BatchedMonomialBasis.build(dim=2, order=order, device=device)
    _sync(device)
    basis_ms = (time.perf_counter() - basis_start) * 1000.0
    domain_lo, domain_hi = _make_domains(batch, dtype, device)
    sample_indices, sample_points = _sample_points(domain_lo, domain_hi)
    sample_truth = _advance_vdp_samples(sample_points, h, steps)
    profiler = DenseTMProfiler(device=device)

    with torch.no_grad():
        tm = BatchedTaylorModel.variables_from_domain(domain_lo, domain_hi, basis)
        _sync(device)
        start = time.perf_counter()
        for _ in range(int(steps)):
            tm = tm.fixed_euler_tm_step_vdp(
                h,
                order=order,
                dropped_merge_mode=dropped_merge_mode,
                range_bound_mode=range_bound_mode,
                profile=profiler,
            )
        range_lo, range_hi = tm.range_bound(method=range_bound_mode, profile=profiler)
        _sync(device)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

    selected_lo = range_lo.index_select(0, sample_indices)
    selected_hi = range_hi.index_select(0, sample_indices)
    contained, violations = _contains(selected_lo, selected_hi, sample_truth, tol)
    profile = profiler.as_flat_dict()
    max_width = float(torch.max(range_hi - range_lo).detach().cpu())
    mean_width = float(torch.mean(range_hi - range_lo).detach().cpu())
    row = {
        "plant": "vdp",
        "batch": batch,
        "device": device.type,
        "implementation": "torch_dense",
        "steps": steps,
        "order": order,
        "h": h,
        "dtype": str(dtype).replace("torch.", ""),
        "range_bound_mode": range_bound_mode,
        "dropped_merge_mode": dropped_merge_mode,
        "status": "ok",
        "elapsed_ms": elapsed_ms,
        "basis_mul_plan_ms": basis_ms,
        "rhs_construction_ms": profile.get("rhs_construction", 0.0),
        "mul_trunc_ms": profile.get("mul_trunc", 0.0),
        "dropped_range_bound_ms": profile.get("dropped_range_bound", 0.0),
        "tm_remainder_propagation_ms": profile.get("tm_remainder_propagation", 0.0),
        "range_bound_ms": profile.get("range_bound", 0.0),
        "cuda_memory_allocated_bytes": profile.get("cuda_memory_allocated_bytes", 0),
        "cuda_memory_reserved_bytes": profile.get("cuda_memory_reserved_bytes", 0),
        "samples_checked": int(sample_truth.numel() // 2),
        "sample_violations": violations,
        "containment_pass": contained,
        "max_width": max_width,
        "mean_width": mean_width,
    }
    row["dominant_operation"] = _dominant(row)
    return row


def _scalar_domain(row_lo: torch.Tensor, row_hi: torch.Tensor) -> list[Interval]:
    return [Interval(row_lo[0], row_hi[0]), Interval(row_lo[1], row_hi[1])]


def _run_scalar_vdp(
    *,
    batch: int,
    steps: int,
    order: int,
    h: float,
    dtype: torch.dtype,
    tol: float,
) -> dict[str, Any]:
    domain_lo, domain_hi = _make_domains(batch, dtype, torch.device("cpu"))
    sample_indices, sample_points = _sample_points(domain_lo, domain_hi)
    sample_truth = _advance_vdp_samples(sample_points, h, steps)
    ranges_lo = torch.empty((batch, 2), dtype=dtype)
    ranges_hi = torch.empty((batch, 2), dtype=dtype)
    start = time.perf_counter()
    for batch_index in range(batch):
        domain = _scalar_domain(domain_lo[batch_index], domain_hi[batch_index])
        x = TaylorModel.variable(0, domain, order=order)
        y = TaylorModel.variable(1, domain, order=order)
        for _ in range(int(steps)):
            rhs_x = y
            rhs_y = y - x - x * x * y
            x = x + h * rhs_x
            y = y + h * rhs_y
        x_box = x.range_box()
        y_box = y.range_box()
        ranges_lo[batch_index, 0] = x_box.lo
        ranges_hi[batch_index, 0] = x_box.hi
        ranges_lo[batch_index, 1] = y_box.lo
        ranges_hi[batch_index, 1] = y_box.hi
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    contained, violations = _contains(
        ranges_lo.index_select(0, sample_indices),
        ranges_hi.index_select(0, sample_indices),
        sample_truth,
        tol,
    )
    return {
        "plant": "vdp",
        "batch": batch,
        "device": "cpu",
        "implementation": "scalar_loop",
        "steps": steps,
        "order": order,
        "h": h,
        "dtype": str(dtype).replace("torch.", ""),
        "range_bound_mode": "n/a",
        "dropped_merge_mode": "n/a",
        "status": "ok",
        "elapsed_ms": elapsed_ms,
        "samples_checked": int(sample_truth.numel() // 2),
        "sample_violations": violations,
        "containment_pass": contained,
        "max_width": float(torch.max(ranges_hi - ranges_lo)),
        "mean_width": float(torch.mean(ranges_hi - ranges_lo)),
        "dominant_operation": "scalar_loop",
    }


def _first_cuda_win(rows: Sequence[dict[str, Any]]) -> int | None:
    wins = [
        int(row["batch"])
        for row in rows
        if row.get("implementation") == "torch_dense"
        and row.get("device") == "cuda"
        and row.get("status") == "ok"
        and row.get("speedup_vs_dense_cpu") not in {None, ""}
        and float(row["speedup_vs_dense_cpu"]) > 1.0
    ]
    return min(wins) if wins else None


def _write_report(out_dir: Path, rows: Sequence[dict[str, Any]], recommendation: str, cuda_available: bool) -> Path:
    dense_rows = [row for row in rows if row.get("implementation") == "torch_dense" and row.get("status") == "ok"]
    scalar_rows = [row for row in rows if row.get("implementation") == "scalar_loop" and row.get("status") == "ok"]
    cpu_speedups = [float(row["speedup_vs_scalar_cpu"]) for row in dense_rows if row.get("speedup_vs_scalar_cpu") not in {None, ""}]
    cuda_speedups = [float(row["speedup_vs_dense_cpu"]) for row in dense_rows if row.get("speedup_vs_dense_cpu") not in {None, ""}]
    cpu_beats_scalar = any(value > 1.0 for value in cpu_speedups)
    cuda_beats_cpu = any(value > 1.0 for value in cuda_speedups)
    containment = all(bool(row.get("containment_pass")) for row in dense_rows) if dense_rows else False
    first_cuda = _first_cuda_win(rows)
    dominant_counts: dict[str, int] = {}
    for row in dense_rows:
        key = str(row.get("dominant_operation", "unknown"))
        dominant_counts[key] = dominant_counts.get(key, 0) + 1
    dominant = max(dominant_counts, key=dominant_counts.get) if dominant_counts else "not measured"

    merged_rows = [row for row in dense_rows if row.get("dropped_merge_mode") == "merged"]
    merged_ratios = [float(row["width_ratio_vs_termwise"]) for row in merged_rows if row.get("width_ratio_vs_termwise") not in {None, ""}]
    merged_reduces = any(ratio < 1.0 for ratio in merged_ratios)
    split_rows = [row for row in dense_rows if row.get("range_bound_mode") == "split2"]
    split_ratios = [float(row["width_ratio_vs_interval"]) for row in split_rows if row.get("width_ratio_vs_interval") not in {None, ""}]
    split_reduces = any(ratio < 1.0 for ratio in split_ratios)
    interval_elapsed = {
        (
            int(row["batch"]),
            int(row["steps"]),
            int(row["order"]),
            str(row["device"]),
            str(row["dropped_merge_mode"]),
        ): float(row["elapsed_ms"])
        for row in dense_rows
        if row.get("range_bound_mode") == "interval" and row.get("elapsed_ms") not in {None, ""}
    }
    split_costs = []
    for row in split_rows:
        key = (
            int(row["batch"]),
            int(row["steps"]),
            int(row["order"]),
            str(row["device"]),
            str(row["dropped_merge_mode"]),
        )
        base = interval_elapsed.get(key)
        if base and base > 0 and row.get("elapsed_ms") not in {None, ""}:
            split_costs.append(float(row["elapsed_ms"]) / base)
    if split_reduces and split_costs:
        split_answer = f"situational: width improves, median cost {sorted(split_costs)[len(split_costs) // 2]:.2f}x interval"
    elif split_reduces:
        split_answer = "yes for width, cost not measured"
    elif split_rows:
        split_answer = "no"
    else:
        split_answer = "not checked"

    lines = [
        "# Batched Dense Many-Box Plant Demo Report",
        "",
        "## Scope",
        "",
        "This is a plant-only, explicit Euler-style dense Taylor-model workload. It does not use Flow*, adaptive rejection, or Flow* Picard integration.",
        "",
        "## Direct Answers",
        "",
        f"- Does dense CPU beat scalar loop? {'yes' if cpu_beats_scalar else 'no' if cpu_speedups else 'not checked'}",
        f"- Does CUDA beat dense CPU? {'yes' if cuda_beats_cpu else 'no' if cuda_available else 'CUDA unavailable'}",
        f"- First CUDA win batch: {first_cuda if first_cuda is not None else 'none'}",
        f"- Merged dropped-term bounding reduces width: {'yes' if merged_reduces else 'no' if merged_ratios else 'not checked'}",
        f"- Split range bound cost/tightness: {split_answer}",
        f"- Sampled trajectory containment: {'pass' if containment else 'fail'}",
        f"- Dominant operation: {dominant}",
        f"- Recommendation: {recommendation}",
        "",
        "## Timing Rows",
        "",
        "| batch | device | impl | steps | mode | dropped | elapsed ms | containment | speedup scalar | speedup CPU | max width | dominant |",
        "| ---: | --- | --- | ---: | --- | --- | ---: | --- | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| {batch} | {device} | {impl} | {steps} | {mode} | {drop} | {elapsed} | {contain} | {scalar} | {cpu} | {width} | {dom} |".format(
                batch=row.get("batch", ""),
                device=row.get("device", ""),
                impl=row.get("implementation", ""),
                steps=row.get("steps", ""),
                mode=row.get("range_bound_mode", ""),
                drop=row.get("dropped_merge_mode", ""),
                elapsed=_format(row.get("elapsed_ms")),
                contain=_format(row.get("containment_pass")),
                scalar=_format(row.get("speedup_vs_scalar_cpu")),
                cpu=_format(row.get("speedup_vs_dense_cpu")),
                width=_format(row.get("max_width")),
                dom=row.get("dominant_operation", ""),
            )
        )
    if scalar_rows:
        lines.extend(["", f"Scalar loop was checked for {len(scalar_rows)} row(s); larger scalar batches are skipped by --scalar-cap."])
    report = out_dir / "many_box_report.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def run_experiment(
    out_dir: Path,
    *,
    batches: Sequence[int],
    steps_list: Sequence[int],
    orders: Sequence[int],
    h: float,
    devices: Sequence[str],
    dtype: torch.dtype,
    range_bound_modes: Sequence[str],
    dropped_merge_modes: Sequence[str],
    scalar_cap: int = 8,
) -> tuple[Path, Path, list[dict[str, Any]], str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cuda_available = torch.cuda.is_available()
    tol = 1e-9 if dtype == torch.float64 else 1e-5
    rows: list[dict[str, Any]] = []
    scalar_elapsed: dict[tuple[int, int, int], float] = {}

    for order in orders:
        for steps in steps_list:
            for batch in batches:
                if batch <= scalar_cap:
                    scalar = _run_scalar_vdp(batch=batch, steps=steps, order=order, h=h, dtype=dtype, tol=tol)
                    rows.append(scalar)
                    scalar_elapsed[(batch, steps, order)] = float(scalar["elapsed_ms"])
                else:
                    rows.append(
                        {
                            "plant": "vdp",
                            "batch": batch,
                            "device": "cpu",
                            "implementation": "scalar_loop",
                            "steps": steps,
                            "order": order,
                            "h": h,
                            "dtype": str(dtype).replace("torch.", ""),
                            "range_bound_mode": "n/a",
                            "dropped_merge_mode": "n/a",
                            "status": "skipped",
                            "skip_reason": f"batch exceeds scalar cap {scalar_cap}",
                        }
                    )

    for device_name in devices:
        if device_name == "cuda" and not cuda_available:
            for order in orders:
                for steps in steps_list:
                    for batch in batches:
                        for range_mode in range_bound_modes:
                            for drop_mode in dropped_merge_modes:
                                rows.append(
                                    {
                                        "plant": "vdp",
                                        "batch": batch,
                                        "device": "cuda",
                                        "implementation": "torch_dense",
                                        "steps": steps,
                                        "order": order,
                                        "h": h,
                                        "dtype": str(dtype).replace("torch.", ""),
                                        "range_bound_mode": range_mode,
                                        "dropped_merge_mode": drop_mode,
                                        "status": "skipped",
                                        "skip_reason": "CUDA unavailable",
                                    }
                                )
            continue
        device = torch.device(device_name)
        for order in orders:
            for steps in steps_list:
                for batch in batches:
                    for range_mode in range_bound_modes:
                        for drop_mode in dropped_merge_modes:
                            rows.append(
                                _run_dense_vdp(
                                    batch=batch,
                                    steps=steps,
                                    order=order,
                                    h=h,
                                    dtype=dtype,
                                    device=device,
                                    range_bound_mode=range_mode,
                                    dropped_merge_mode=drop_mode,
                                    tol=tol,
                                )
                            )

    dense_cpu_elapsed = {
        (
            int(row["batch"]),
            int(row["steps"]),
            int(row["order"]),
            str(row["range_bound_mode"]),
            str(row["dropped_merge_mode"]),
        ): float(row["elapsed_ms"])
        for row in rows
        if row.get("implementation") == "torch_dense" and row.get("device") == "cpu" and row.get("status") == "ok"
    }
    termwise_width = {
        (
            int(row["batch"]),
            int(row["steps"]),
            int(row["order"]),
            str(row["device"]),
            str(row["range_bound_mode"]),
        ): float(row["max_width"])
        for row in rows
        if row.get("implementation") == "torch_dense" and row.get("dropped_merge_mode") == "termwise" and row.get("status") == "ok"
    }
    interval_width = {
        (
            int(row["batch"]),
            int(row["steps"]),
            int(row["order"]),
            str(row["device"]),
            str(row["dropped_merge_mode"]),
        ): float(row["max_width"])
        for row in rows
        if row.get("implementation") == "torch_dense" and row.get("range_bound_mode") == "interval" and row.get("status") == "ok"
    }
    for row in rows:
        if row.get("implementation") != "torch_dense" or row.get("status") != "ok":
            continue
        batch = int(row["batch"])
        steps = int(row["steps"])
        order = int(row["order"])
        elapsed = float(row["elapsed_ms"])
        scalar_key = (batch, steps, order)
        if row.get("device") == "cpu" and scalar_key in scalar_elapsed and elapsed > 0:
            row["speedup_vs_scalar_cpu"] = scalar_elapsed[scalar_key] / elapsed
        cpu_key = (batch, steps, order, str(row["range_bound_mode"]), str(row["dropped_merge_mode"]))
        if row.get("device") == "cuda" and cpu_key in dense_cpu_elapsed and elapsed > 0:
            row["speedup_vs_dense_cpu"] = dense_cpu_elapsed[cpu_key] / elapsed
        term_key = (batch, steps, order, str(row["device"]), str(row["range_bound_mode"]))
        if row.get("dropped_merge_mode") == "merged" and term_key in termwise_width and termwise_width[term_key] > 0:
            row["width_ratio_vs_termwise"] = float(row["max_width"]) / termwise_width[term_key]
        int_key = (batch, steps, order, str(row["device"]), str(row["dropped_merge_mode"]))
        if row.get("range_bound_mode") == "split2" and int_key in interval_width and interval_width[int_key] > 0:
            row["width_ratio_vs_interval"] = float(row["max_width"]) / interval_width[int_key]

    dense_ok = [row for row in rows if row.get("implementation") == "torch_dense" and row.get("status") == "ok"]
    containment_ok = all(bool(row.get("containment_pass")) for row in dense_ok) if dense_ok else False
    cuda_win = _first_cuda_win(rows) is not None
    cpu_win = any(float(row.get("speedup_vs_scalar_cpu") or 0.0) > 1.0 for row in dense_ok)
    if not containment_ok:
        recommendation = "NEEDS_REMAINDER_REDESIGN"
    elif cuda_win or cpu_win:
        recommendation = "GPU_PATH_CONTINUE"
    else:
        recommendation = "STOP_DENSE_PLANT"
    for row in rows:
        row["recommendation"] = recommendation

    summary = out_dir / "many_box_summary.csv"
    _write_csv(summary, rows)
    report = _write_report(out_dir, rows, recommendation, cuda_available)
    return summary, report, rows, recommendation


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a many-box dense batched TM plant demo")
    parser.add_argument("--out-dir", default="outputs/batched_dense_many_box_flowpipe_demo")
    parser.add_argument("--batches", default="1,8,32,128,512,2048,8192")
    parser.add_argument("--steps", default="50")
    parser.add_argument("--order", default="4")
    parser.add_argument("--h", type=float, default=0.01)
    parser.add_argument("--devices", default="cpu,cuda")
    parser.add_argument("--dtype", default="float64", choices=["float64", "float32"])
    parser.add_argument("--range-bound-mode", default="interval")
    parser.add_argument("--dropped-merge-mode", default="termwise,merged")
    parser.add_argument("--scalar-cap", type=int, default=8)
    args = parser.parse_args()

    summary, report, _rows, recommendation = run_experiment(
        Path(args.out_dir),
        batches=_parse_ints(args.batches, [1, 8, 32, 128, 512]),
        steps_list=_parse_ints(args.steps, [50]),
        orders=_parse_ints(args.order, [4]),
        h=args.h,
        devices=_parse_strings(args.devices, ["cpu"], {"cpu", "cuda"}),
        dtype=_dtype_from_name(args.dtype),
        range_bound_modes=_parse_strings(args.range_bound_mode, ["interval"], {"interval", "split2", "subdivide"}),
        dropped_merge_modes=_parse_strings(args.dropped_merge_mode, ["termwise"], {"termwise", "merged"}),
        scalar_cap=args.scalar_cap,
    )
    print(f"many-box dense plant demo complete: recommendation={recommendation}")
    print(f"summary: {summary}")
    print(f"report: {report}")


if __name__ == "__main__":
    main()
