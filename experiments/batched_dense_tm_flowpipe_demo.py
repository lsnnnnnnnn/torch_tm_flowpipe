#!/usr/bin/env python3
"""Small dense batched Taylor-model prototype demo.

This experiment intentionally does not call Flow* and does not claim end-to-end
reachability speedups. It benchmarks a fixed dense Taylor-model step for a
Van der Pol-like vector field against a small scalar Python-loop baseline when
that baseline is still feasible.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from torch_tm_flowpipe import Interval, TaylorModel  # noqa: E402
from torch_tm_flowpipe.batched_dense_tm import BatchedMonomialBasis, BatchedTaylorModel  # noqa: E402

MICRO_FIELDS = [
    "batch",
    "device",
    "implementation",
    "steps",
    "order",
    "dtype",
    "status",
    "elapsed_ms",
    "rhs_ms",
    "update_ms",
    "range_ms",
    "dominant_operation",
    "samples_contained",
    "speedup_vs_scalar_cpu",
    "speedup_vs_dense_cpu",
    "note",
]

PARITY_FIELDS = [
    "batch",
    "device",
    "steps",
    "order",
    "dtype",
    "dense_contains_samples",
    "scalar_checked",
    "scalar_contains_samples",
    "dense_contains_scalar_ranges",
    "max_dense_width",
    "max_scalar_width",
    "note",
]


def _parse_batches(text: str) -> list[int]:
    batches = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not batches or any(batch <= 0 for batch in batches):
        raise argparse.ArgumentTypeError("batches must be positive comma-separated integers")
    return batches


def _parse_devices(text: str) -> list[str]:
    devices = [part.strip().lower() for part in text.split(",") if part.strip()]
    if not devices:
        raise argparse.ArgumentTypeError("at least one device is required")
    bad = [device for device in devices if device not in {"cpu", "cuda"}]
    if bad:
        raise argparse.ArgumentTypeError(f"unsupported device(s): {bad}")
    return devices


def _dtype_from_name(name: str) -> torch.dtype:
    if name == "float64":
        return torch.float64
    if name == "float32":
        return torch.float32
    raise argparse.ArgumentTypeError("dtype must be float64 or float32")


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


def _sample_points(lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    center = 0.5 * (lo + hi)
    return torch.stack(
        [
            lo,
            hi,
            center,
            torch.stack([lo[:, 0], hi[:, 1]], dim=1),
            torch.stack([hi[:, 0], lo[:, 1]], dim=1),
        ],
        dim=1,
    )


def _advance_samples(points: torch.Tensor, h: float, steps: int) -> torch.Tensor:
    state = points.clone()
    h_t = torch.as_tensor(h, dtype=points.dtype, device=points.device)
    for _ in range(int(steps)):
        x = state[..., 0]
        y = state[..., 1]
        state = torch.stack([x + h_t * y, y + h_t * (y - x - x * x * y)], dim=-1)
    return state


def _contains_samples(lo: torch.Tensor, hi: torch.Tensor, samples: torch.Tensor, tol: float) -> bool:
    return bool(torch.all(samples >= lo[:, None, :] - tol) and torch.all(samples <= hi[:, None, :] + tol))


def _dominant_operation(rhs_ms: float, update_ms: float, range_ms: float) -> str:
    timings = {
        "rhs_mul_trunc": rhs_ms,
        "euler_update": update_ms,
        "range_bound": range_ms,
    }
    return max(timings, key=timings.get)


def _run_dense(
    *,
    batch: int,
    steps: int,
    order: int,
    dtype: torch.dtype,
    device: torch.device,
    h: float,
    tol: float,
) -> dict[str, Any]:
    basis = BatchedMonomialBasis.build(dim=2, order=order, device=device)
    domain_lo, domain_hi = _make_domains(batch, dtype, device)
    samples = _sample_points(domain_lo, domain_hi)
    sample_truth = _advance_samples(samples, h, steps)

    with torch.no_grad():
        tm = BatchedTaylorModel.variables_from_domain(domain_lo, domain_hi, basis)
        _sync(device)
        start = time.perf_counter()
        rhs_ms = 0.0
        update_ms = 0.0
        for _ in range(steps):
            _sync(device)
            rhs_start = time.perf_counter()
            rhs = tm.vanderpol_rhs()
            _sync(device)
            rhs_ms += (time.perf_counter() - rhs_start) * 1000.0

            update_start = time.perf_counter()
            tm = tm.add(rhs.scale(h))
            _sync(device)
            update_ms += (time.perf_counter() - update_start) * 1000.0
        _sync(device)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        range_start = time.perf_counter()
        range_lo, range_hi = tm.range_bound()
        _sync(device)
        range_ms = (time.perf_counter() - range_start) * 1000.0

    contains = _contains_samples(range_lo, range_hi, sample_truth, tol)
    return {
        "status": "ok",
        "elapsed_ms": elapsed_ms,
        "rhs_ms": rhs_ms,
        "update_ms": update_ms,
        "range_ms": range_ms,
        "dominant_operation": _dominant_operation(rhs_ms, update_ms, range_ms),
        "samples_contained": contains,
        "range_lo": range_lo.detach().cpu(),
        "range_hi": range_hi.detach().cpu(),
        "sample_truth": sample_truth.detach().cpu(),
    }


def _scalar_domain(row_lo: torch.Tensor, row_hi: torch.Tensor) -> list[Interval]:
    return [Interval(row_lo[0], row_hi[0]), Interval(row_lo[1], row_hi[1])]


def _run_scalar(
    *,
    batch: int,
    steps: int,
    order: int,
    dtype: torch.dtype,
    h: float,
    tol: float,
) -> dict[str, Any]:
    domain_lo, domain_hi = _make_domains(batch, dtype, torch.device("cpu"))
    samples = _sample_points(domain_lo, domain_hi)
    sample_truth = _advance_samples(samples, h, steps)
    ranges_lo = torch.empty((batch, 2), dtype=dtype)
    ranges_hi = torch.empty((batch, 2), dtype=dtype)

    start = time.perf_counter()
    for batch_index in range(batch):
        domain = _scalar_domain(domain_lo[batch_index], domain_hi[batch_index])
        x = TaylorModel.variable(0, domain, order=order)
        y = TaylorModel.variable(1, domain, order=order)
        for _ in range(steps):
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

    contains = _contains_samples(ranges_lo, ranges_hi, sample_truth, tol)
    return {
        "status": "ok",
        "elapsed_ms": elapsed_ms,
        "samples_contained": contains,
        "range_lo": ranges_lo,
        "range_hi": ranges_hi,
        "sample_truth": sample_truth,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if row.get(key) is None else row.get(key, "") for key in fieldnames})


def _fmt(value: Any) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _write_reports(
    *,
    out_dir: Path,
    micro_rows: list[dict[str, Any]],
    parity_rows: list[dict[str, Any]],
    batches: list[int],
    steps: int,
    order: int,
    dtype_name: str,
    h: float,
    scalar_cap: int,
    cuda_available: bool,
) -> str:
    dense_ok = [row for row in micro_rows if row["implementation"] == "torch_dense" and row["status"] == "ok"]
    dense_contains = all(bool(row["samples_contained"]) for row in dense_ok) if dense_ok else False
    scalar_checked_rows = [row for row in parity_rows if row["scalar_checked"]]
    scalar_contains = all(bool(row["scalar_contains_samples"]) for row in scalar_checked_rows) if scalar_checked_rows else None
    cpu_speedups = [float(row["speedup_vs_scalar_cpu"]) for row in micro_rows if row.get("speedup_vs_scalar_cpu") not in {None, ""}]
    cuda_speedups = [float(row["speedup_vs_dense_cpu"]) for row in micro_rows if row.get("speedup_vs_dense_cpu") not in {None, ""}]
    cpu_beats_scalar = any(value > 1.0 for value in cpu_speedups)
    cuda_beats_cpu = any(value > 1.0 for value in cuda_speedups)
    dominant_counts: dict[str, int] = {}
    for row in dense_ok:
        dominant_counts[str(row["dominant_operation"])] = dominant_counts.get(str(row["dominant_operation"]), 0) + 1
    dominant = max(dominant_counts, key=dominant_counts.get) if dominant_counts else "not measured"

    if not dense_contains:
        recommendation = "NEEDS_CONSERVATIVE_REMAINDER_REDESIGN"
    elif cpu_beats_scalar or cuda_beats_cpu:
        recommendation = "GPU_PATH_CONTINUE"
    else:
        recommendation = "STOP_DENSE_TM_FOR_NOW"

    blockers = []
    if not dense_contains:
        blockers.append("sample containment failed in the dense prototype")
    if not cuda_available:
        blockers.append("CUDA was unavailable in this run")
    if not cpu_beats_scalar and not cuda_beats_cpu:
        blockers.append("no measured dense speed win in this configuration")
    if not blockers:
        blockers.append("remainder bounds are still interval-only and need tighter validation before larger claims")

    summary_lines = [
        "# Dense Batched Taylor Model Demo Report",
        "",
        "## Scope",
        "",
        "This is an experimental dense tensor prototype. It does not add a Flow* mechanism, does not modify the C++ probe, and does not replace production Polynomial/TaylorModel classes.",
        "",
        "## Configuration",
        "",
        f"- Batches: {','.join(str(batch) for batch in batches)}",
        f"- Steps: {steps}",
        f"- Order: {order}",
        f"- Step size h: {h}",
        f"- Dtype: {dtype_name}",
        f"- Scalar baseline cap: {scalar_cap}",
        f"- CUDA available: {_fmt(cuda_available)}",
        "",
        "## Answers",
        "",
        f"- Contains sampled trajectories: {_fmt(dense_contains)}",
        f"- Contains scalar sampled trajectories: {_fmt(scalar_contains) if scalar_contains is not None else 'not checked'}",
        f"- CPU beats scalar: {_fmt(cpu_beats_scalar) if cpu_speedups else 'not checked'}",
        f"- CUDA beats CPU: {_fmt(cuda_beats_cpu) if cuda_speedups else 'not measured'}",
        f"- Dominant operation: {dominant}",
        f"- Blockers: {'; '.join(blockers)}",
        f"- Next step: {recommendation}",
        "",
        "## Timing Rows",
        "",
        "| batch | device | implementation | elapsed ms | sample containment | speedup vs scalar | speedup vs CPU | dominant |",
        "| ---: | --- | --- | ---: | --- | ---: | ---: | --- |",
    ]
    for row in micro_rows:
        summary_lines.append(
            "| {batch} | {device} | {implementation} | {elapsed} | {contained} | {scalar} | {cpu} | {dominant} |".format(
                batch=row["batch"],
                device=row["device"],
                implementation=row["implementation"],
                elapsed=_fmt(row.get("elapsed_ms")),
                contained=_fmt(row.get("samples_contained")),
                scalar=_fmt(row.get("speedup_vs_scalar_cpu")),
                cpu=_fmt(row.get("speedup_vs_dense_cpu")),
                dominant=row.get("dominant_operation", ""),
            )
        )
    summary_lines.append("")
    (out_dir / "dense_tm_demo_report.md").write_text("\n".join(summary_lines), encoding="utf-8")

    parity_lines = [
        "# Dense Batched Taylor Model Parity Report",
        "",
        "The tests compare dense coefficient arithmetic against the sparse Polynomial path for small cases. This demo report adds sampled fixed-step Van der Pol containment checks.",
        "",
        "| batch | device | dense contains samples | scalar checked | scalar contains samples | dense contains scalar ranges | max dense width | max scalar width |",
        "| ---: | --- | --- | --- | --- | --- | ---: | ---: |",
    ]
    for row in parity_rows:
        parity_lines.append(
            "| {batch} | {device} | {dense} | {checked} | {scalar} | {ranges} | {dense_width} | {scalar_width} |".format(
                batch=row["batch"],
                device=row["device"],
                dense=_fmt(row.get("dense_contains_samples")),
                checked=_fmt(row.get("scalar_checked")),
                scalar=_fmt(row.get("scalar_contains_samples")),
                ranges=_fmt(row.get("dense_contains_scalar_ranges")),
                dense_width=_fmt(row.get("max_dense_width")),
                scalar_width=_fmt(row.get("max_scalar_width")),
            )
        )
    parity_lines.extend(
        [
            "",
            "## Limitations",
            "",
            "Remainder multiplication uses interval bounds for the retained polynomial ranges and dropped dense monomial products. This is conservative for the sampled small cases here, but it is not a proof of production reachability performance or tightness.",
        ]
    )
    (out_dir / "dense_tm_parity_report.md").write_text("\n".join(parity_lines), encoding="utf-8")
    return recommendation


def run(args: argparse.Namespace) -> str:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dtype = _dtype_from_name(args.dtype)
    batches = _parse_batches(args.batches)
    requested_devices = _parse_devices(args.devices)
    cuda_available = torch.cuda.is_available()
    tol = 1e-9 if dtype == torch.float64 else 1e-5

    micro_rows: list[dict[str, Any]] = []
    parity_rows: list[dict[str, Any]] = []
    scalar_results: dict[int, dict[str, Any]] = {}

    for batch in batches:
        if batch <= args.scalar_cap:
            scalar = _run_scalar(batch=batch, steps=args.steps, order=args.order, dtype=dtype, h=args.h, tol=tol)
            scalar_results[batch] = scalar
            micro_rows.append(
                {
                    "batch": batch,
                    "device": "cpu",
                    "implementation": "scalar_loop",
                    "steps": args.steps,
                    "order": args.order,
                    "dtype": args.dtype,
                    "status": scalar["status"],
                    "elapsed_ms": scalar["elapsed_ms"],
                    "samples_contained": scalar["samples_contained"],
                    "note": "existing sparse TaylorModel loop",
                }
            )
        else:
            micro_rows.append(
                {
                    "batch": batch,
                    "device": "cpu",
                    "implementation": "scalar_loop",
                    "steps": args.steps,
                    "order": args.order,
                    "dtype": args.dtype,
                    "status": "skipped",
                    "note": f"batch exceeds scalar cap {args.scalar_cap}",
                }
            )

    for device_name in requested_devices:
        if device_name == "cuda" and not cuda_available:
            for batch in batches:
                micro_rows.append(
                    {
                        "batch": batch,
                        "device": "cuda",
                        "implementation": "torch_dense",
                        "steps": args.steps,
                        "order": args.order,
                        "dtype": args.dtype,
                        "status": "skipped",
                        "note": "CUDA unavailable",
                    }
                )
            continue
        device = torch.device(device_name)
        for batch in batches:
            dense = _run_dense(batch=batch, steps=args.steps, order=args.order, dtype=dtype, device=device, h=args.h, tol=tol)
            micro_rows.append(
                {
                    "batch": batch,
                    "device": device_name,
                    "implementation": "torch_dense",
                    "steps": args.steps,
                    "order": args.order,
                    "dtype": args.dtype,
                    "status": dense["status"],
                    "elapsed_ms": dense["elapsed_ms"],
                    "rhs_ms": dense["rhs_ms"],
                    "update_ms": dense["update_ms"],
                    "range_ms": dense["range_ms"],
                    "dominant_operation": dense["dominant_operation"],
                    "samples_contained": dense["samples_contained"],
                    "note": "experimental dense tensor path",
                }
            )

            scalar = scalar_results.get(batch)
            if scalar is None:
                parity_rows.append(
                    {
                        "batch": batch,
                        "device": device_name,
                        "steps": args.steps,
                        "order": args.order,
                        "dtype": args.dtype,
                        "dense_contains_samples": dense["samples_contained"],
                        "scalar_checked": False,
                        "max_dense_width": float(torch.max(dense["range_hi"] - dense["range_lo"])),
                        "note": f"scalar skipped above cap {args.scalar_cap}",
                    }
                )
            else:
                dense_lo = dense["range_lo"]
                dense_hi = dense["range_hi"]
                scalar_lo = scalar["range_lo"]
                scalar_hi = scalar["range_hi"]
                dense_contains_scalar_ranges = bool(
                    torch.all(dense_lo <= scalar_lo + tol) and torch.all(dense_hi >= scalar_hi - tol)
                )
                parity_rows.append(
                    {
                        "batch": batch,
                        "device": device_name,
                        "steps": args.steps,
                        "order": args.order,
                        "dtype": args.dtype,
                        "dense_contains_samples": dense["samples_contained"],
                        "scalar_checked": True,
                        "scalar_contains_samples": scalar["samples_contained"],
                        "dense_contains_scalar_ranges": dense_contains_scalar_ranges,
                        "max_dense_width": float(torch.max(dense_hi - dense_lo)),
                        "max_scalar_width": float(torch.max(scalar_hi - scalar_lo)),
                        "note": "scalar range containment is diagnostic, not required for speed claims",
                    }
                )

    dense_cpu_elapsed = {
        int(row["batch"]): float(row["elapsed_ms"])
        for row in micro_rows
        if row["implementation"] == "torch_dense" and row["device"] == "cpu" and row["status"] == "ok"
    }
    scalar_elapsed = {batch: float(result["elapsed_ms"]) for batch, result in scalar_results.items()}
    for row in micro_rows:
        if row["implementation"] != "torch_dense" or row["status"] != "ok":
            continue
        batch = int(row["batch"])
        elapsed = float(row["elapsed_ms"])
        if row["device"] == "cpu" and batch in scalar_elapsed and elapsed > 0:
            row["speedup_vs_scalar_cpu"] = scalar_elapsed[batch] / elapsed
        if row["device"] == "cuda" and batch in dense_cpu_elapsed and elapsed > 0:
            row["speedup_vs_dense_cpu"] = dense_cpu_elapsed[batch] / elapsed

    _write_csv(out_dir / "dense_tm_microbench_summary.csv", micro_rows, MICRO_FIELDS)
    _write_csv(out_dir / "dense_tm_parity_summary.csv", parity_rows, PARITY_FIELDS)
    recommendation = _write_reports(
        out_dir=out_dir,
        micro_rows=micro_rows,
        parity_rows=parity_rows,
        batches=batches,
        steps=args.steps,
        order=args.order,
        dtype_name=args.dtype,
        h=args.h,
        scalar_cap=args.scalar_cap,
        cuda_available=cuda_available,
    )
    return recommendation


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the dense batched Taylor-model prototype demo")
    parser.add_argument("--out-dir", default="outputs/batched_dense_tm_prototype")
    parser.add_argument("--batches", default="1,8,32,128,512")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--order", type=int, default=4)
    parser.add_argument("--dtype", default="float64", choices=["float64", "float32"])
    parser.add_argument("--devices", default="cpu")
    parser.add_argument("--h", type=float, default=0.01)
    parser.add_argument("--scalar-cap", type=int, default=32)
    args = parser.parse_args()
    recommendation = run(args)
    print(f"dense batched TM prototype complete: next_step={recommendation}")
    print(f"outputs written to {Path(args.out_dir)}")


if __name__ == "__main__":
    main()
