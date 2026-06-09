#!/usr/bin/env python3
"""Batched dense Taylor-model NNCS workload demo.

This is a B-line workload: many state boxes, simple controller bounds, and a
fixed-step dense Taylor-model plant update. It is not CROWN-Reach parity and it
does not use Flow*.
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

from torch_tm_flowpipe.batched_dense_tm import BatchedMonomialBasis, BatchedTaylorModel  # noqa: E402

SUMMARY_FIELDS = [
    "batch",
    "device",
    "controller",
    "num_control_steps",
    "plant_substeps",
    "order",
    "h",
    "dtype",
    "range_bound_mode",
    "dropped_merge_mode",
    "status",
    "skip_reason",
    "elapsed_ms",
    "basis_mul_plan_ms",
    "controller_bound_ms",
    "plant_step_ms",
    "plant_rhs_mul_trunc_ms",
    "range_bound_ms",
    "reset_ms",
    "cuda_memory_allocated_bytes",
    "cuda_memory_reserved_bytes",
    "samples_checked",
    "sample_violations",
    "containment_pass",
    "max_width",
    "mean_width",
    "speedup_vs_dense_cpu",
    "controller_overhead_fraction",
    "plant_overhead_fraction",
    "dominant_operation",
    "recommendation",
]


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
    phase_a = torch.remainder(idx * 3.0, 17.0) / 17.0
    phase_b = torch.remainder(idx * 5.0, 19.0) / 19.0
    center_x = 0.45 + 0.30 * (phase_a - 0.5)
    center_y = -0.15 + 0.24 * (phase_b - 0.5)
    radius_x = torch.full_like(center_x, 0.025)
    radius_y = torch.full_like(center_y, 0.020)
    return torch.stack([center_x - radius_x, center_y - radius_y], dim=1), torch.stack(
        [center_x + radius_x, center_y + radius_y], dim=1
    )


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
    gen.manual_seed(314159)
    rand = torch.rand((lo_s.shape[0], 4, lo_s.shape[1]), generator=gen, dtype=lo_s.dtype).to(lo_s.device)
    random_points = lo_s[:, None, :] + rand * (hi_s - lo_s)[:, None, :]
    return indices, torch.cat([torch.stack(base, dim=1), random_points], dim=1)


def _controller_params(dtype: torch.dtype, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "affine_K": torch.tensor([[-0.70, -0.25]], dtype=dtype, device=device),
        "affine_b": torch.tensor([0.05], dtype=dtype, device=device),
        "W1": torch.tensor(
            [[0.8, -0.4], [-0.5, 0.7], [0.3, 0.2], [-0.2, -0.6]],
            dtype=dtype,
            device=device,
        ),
        "b1": torch.tensor([0.02, -0.01, 0.03, 0.04], dtype=dtype, device=device),
        "W2": torch.tensor([[0.35, -0.45, 0.25, 0.15]], dtype=dtype, device=device),
        "b2": torch.tensor([0.01], dtype=dtype, device=device),
    }


def _interval_linear(
    lo: torch.Tensor,
    hi: torch.Tensor,
    W: torch.Tensor,
    b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    center = 0.5 * (lo + hi)
    radius = 0.5 * (hi - lo)
    out_center = center @ W.T + b
    out_radius = radius @ torch.abs(W).T
    return out_center - out_radius, out_center + out_radius


def _relu_ibp(
    lo: torch.Tensor,
    hi: torch.Tensor,
    params: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    h_lo, h_hi = _interval_linear(lo, hi, params["W1"], params["b1"])
    h_lo = torch.clamp(h_lo, min=0.0)
    h_hi = torch.clamp(h_hi, min=0.0)
    return _interval_linear(h_lo, h_hi, params["W2"], params["b2"])


def _exact_controller_samples(
    samples: torch.Tensor,
    controller: str,
    params: dict[str, torch.Tensor],
) -> torch.Tensor:
    if controller == "affine":
        return torch.einsum("oi,bni->bno", params["affine_K"], samples).squeeze(-1) + params["affine_b"][0]
    hidden = torch.relu(torch.einsum("hi,bni->bnh", params["W1"], samples) + params["b1"])
    return (torch.einsum("oh,bnh->bno", params["W2"], hidden).squeeze(-1) + params["b2"][0])


def _advance_controlled_samples(points: torch.Tensor, controls: torch.Tensor, h: float, substeps: int) -> torch.Tensor:
    state = points.clone()
    h_t = torch.as_tensor(h, dtype=points.dtype, device=points.device)
    for _ in range(int(substeps)):
        x = state[..., 0]
        y = state[..., 1]
        state = torch.stack([x + h_t * y, y + h_t * (controls - x - 0.1 * y)], dim=-1)
    return state


def _contains(range_lo: torch.Tensor, range_hi: torch.Tensor, samples: torch.Tensor, tol: float) -> tuple[bool, int]:
    below = samples < range_lo[:, None, :] - tol
    above = samples > range_hi[:, None, :] + tol
    violations = int(torch.count_nonzero(below | above).detach().cpu())
    return violations == 0, violations


def _dominant(row: dict[str, Any]) -> str:
    timings = {
        "controller bound": float(row.get("controller_bound_ms") or 0.0),
        "plant rhs/mul_trunc": float(row.get("plant_rhs_mul_trunc_ms") or 0.0),
        "plant step": float(row.get("plant_step_ms") or 0.0),
        "range bound": float(row.get("range_bound_ms") or 0.0),
        "reset": float(row.get("reset_ms") or 0.0),
    }
    return max(timings, key=timings.get)


def _run_case(
    *,
    batch: int,
    num_control_steps: int,
    plant_substeps: int,
    order: int,
    h: float,
    controller: str,
    dtype: torch.dtype,
    device: torch.device,
    tol: float,
    range_bound_mode: str = "interval",
    dropped_merge_mode: str = "merged",
) -> dict[str, Any]:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _sync(device)
    basis_start = time.perf_counter()
    basis = BatchedMonomialBasis.build(dim=2, order=order, device=device)
    _sync(device)
    basis_ms = (time.perf_counter() - basis_start) * 1000.0
    params = _controller_params(dtype, device)
    current_lo, current_hi = _make_domains(batch, dtype, device)
    sample_indices, sample_state = _sample_points(current_lo, current_hi)
    controller_ms = 0.0
    plant_ms = 0.0
    range_ms = 0.0
    reset_ms = 0.0
    total_violations = 0
    containment_pass = True

    with torch.no_grad():
        _sync(device)
        start = time.perf_counter()
        state_tm = BatchedTaylorModel.variables_from_domain(current_lo, current_hi, basis)
        for _ in range(int(num_control_steps)):
            ctrl_start = time.perf_counter()
            if controller == "affine":
                control_tm = state_tm.affine_map(params["affine_K"], params["affine_b"])
            else:
                u_lo, u_hi = _relu_ibp(current_lo, current_hi, params)
                control_tm = BatchedTaylorModel.constant_interval(u_lo, u_hi, basis, current_lo, current_hi)
            sample_controls = _exact_controller_samples(sample_state, controller, params)
            _sync(device)
            controller_ms += (time.perf_counter() - ctrl_start) * 1000.0

            plant_start = time.perf_counter()
            for _substep in range(int(plant_substeps)):
                state_tm = state_tm.fixed_euler_tm_step_controlled(control_tm, h, order=order)
            sample_state = _advance_controlled_samples(sample_state, sample_controls, h, plant_substeps)
            _sync(device)
            plant_ms += (time.perf_counter() - plant_start) * 1000.0

            range_start = time.perf_counter()
            range_lo, range_hi = state_tm.range_bound(method=range_bound_mode)
            _sync(device)
            range_ms += (time.perf_counter() - range_start) * 1000.0
            selected_lo = range_lo.index_select(0, sample_indices)
            selected_hi = range_hi.index_select(0, sample_indices)
            step_contains, step_violations = _contains(selected_lo, selected_hi, sample_state, tol)
            containment_pass = containment_pass and step_contains
            total_violations += step_violations

            reset_start = time.perf_counter()
            current_lo = range_lo.detach()
            current_hi = range_hi.detach()
            state_tm = BatchedTaylorModel.variables_from_domain(current_lo, current_hi, basis)
            _sync(device)
            reset_ms += (time.perf_counter() - reset_start) * 1000.0
        _sync(device)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

    max_width = float(torch.max(current_hi - current_lo).detach().cpu())
    mean_width = float(torch.mean(current_hi - current_lo).detach().cpu())
    allocated = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    reserved = int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0
    row = {
        "batch": batch,
        "device": device.type,
        "controller": controller,
        "num_control_steps": num_control_steps,
        "plant_substeps": plant_substeps,
        "order": order,
        "h": h,
        "dtype": str(dtype).replace("torch.", ""),
        "range_bound_mode": range_bound_mode,
        "dropped_merge_mode": dropped_merge_mode,
        "status": "ok",
        "elapsed_ms": elapsed_ms,
        "basis_mul_plan_ms": basis_ms,
        "controller_bound_ms": controller_ms,
        "plant_step_ms": plant_ms,
        "plant_rhs_mul_trunc_ms": 0.0,
        "range_bound_ms": range_ms,
        "reset_ms": reset_ms,
        "cuda_memory_allocated_bytes": allocated,
        "cuda_memory_reserved_bytes": reserved,
        "samples_checked": int(sample_state.numel() // 2),
        "sample_violations": total_violations,
        "containment_pass": containment_pass,
        "max_width": max_width,
        "mean_width": mean_width,
    }
    if elapsed_ms > 0:
        row["controller_overhead_fraction"] = controller_ms / elapsed_ms
        row["plant_overhead_fraction"] = plant_ms / elapsed_ms
    row["dominant_operation"] = _dominant(row)
    return row


def _first_cuda_win(rows: Sequence[dict[str, Any]]) -> int | None:
    wins = [
        int(row["batch"])
        for row in rows
        if row.get("device") == "cuda"
        and row.get("status") == "ok"
        and row.get("speedup_vs_dense_cpu") not in {None, ""}
        and float(row["speedup_vs_dense_cpu"]) > 1.0
    ]
    return min(wins) if wins else None


def _write_report(out_dir: Path, rows: Sequence[dict[str, Any]], recommendation: str, cuda_available: bool) -> Path:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    cpu_ok = any(row.get("device") == "cpu" for row in ok_rows)
    cuda_ok = any(row.get("device") == "cuda" for row in ok_rows)
    first_cuda = _first_cuda_win(rows)
    containment = all(bool(row.get("containment_pass")) for row in ok_rows) if ok_rows else False
    controller_fracs = [float(row.get("controller_overhead_fraction") or 0.0) for row in ok_rows]
    plant_fracs = [float(row.get("plant_overhead_fraction") or 0.0) for row in ok_rows]
    controller_dominant = bool(controller_fracs and max(controller_fracs) > max(plant_fracs or [0.0]))
    dominant_counts: dict[str, int] = {}
    for row in ok_rows:
        key = str(row.get("dominant_operation", "unknown"))
        dominant_counts[key] = dominant_counts.get(key, 0) + 1
    dominant = max(dominant_counts, key=dominant_counts.get) if dominant_counts else "not measured"

    lines = [
        "# Batched Dense NNCS Demo Report",
        "",
        "## Scope",
        "",
        "This is a batched controller-bound plus dense plant loop. It is not CROWN-Reach parity and does not use Flow*.",
        "",
        "## Direct Answers",
        "",
        f"- End-to-end CPU run: {'yes' if cpu_ok else 'no'}",
        f"- End-to-end CUDA run: {'yes' if cuda_ok else 'no' if cuda_available else 'CUDA unavailable'}",
        f"- First CUDA win batch: {first_cuda if first_cuda is not None else 'none'}",
        f"- Dominant part: {dominant}",
        f"- Closed-loop sampled containment: {'pass' if containment else 'fail'}",
        f"- Controller bound overhead: {'dominant' if controller_dominant else 'not dominant'}",
        f"- Plant overhead: {'dominant' if not controller_dominant and ok_rows else 'not dominant'}",
        f"- Representation redesign needed next: tighter remainder/range bounding for wider closed-loop boxes, then richer controller linear bounds.",
        f"- Recommendation: {recommendation}",
        "",
        "## Timing Rows",
        "",
        "| batch | device | controller | elapsed ms | controller ms | plant ms | range ms | containment | speedup CPU | dominant |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| {batch} | {device} | {controller} | {elapsed} | {ctrl} | {plant} | {range_ms} | {contain} | {speedup} | {dominant} |".format(
                batch=row.get("batch", ""),
                device=row.get("device", ""),
                controller=row.get("controller", ""),
                elapsed=_format(row.get("elapsed_ms")),
                ctrl=_format(row.get("controller_bound_ms")),
                plant=_format(row.get("plant_step_ms")),
                range_ms=_format(row.get("range_bound_ms")),
                contain=_format(row.get("containment_pass")),
                speedup=_format(row.get("speedup_vs_dense_cpu")),
                dominant=row.get("dominant_operation", ""),
            )
        )
    report = out_dir / "nncs_report.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def run_experiment(
    out_dir: Path,
    *,
    batches: Sequence[int],
    num_control_steps_list: Sequence[int],
    plant_substeps_list: Sequence[int],
    controllers: Sequence[str],
    devices: Sequence[str],
    dtype: torch.dtype,
    range_bound_modes: Sequence[str] = ("interval",),
    dropped_merge_modes: Sequence[str] = ("merged",),
    order: int = 3,
    h: float = 0.01,
) -> tuple[Path, Path, list[dict[str, Any]], str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cuda_available = torch.cuda.is_available()
    tol = 1e-9 if dtype == torch.float64 else 1e-5
    rows: list[dict[str, Any]] = []
    for device_name in devices:
        if device_name == "cuda" and not cuda_available:
            for controller in controllers:
                for num_control_steps in num_control_steps_list:
                    for plant_substeps in plant_substeps_list:
                        for batch in batches:
                            for range_mode in range_bound_modes:
                                for drop_mode in dropped_merge_modes:
                                    rows.append(
                                        {
                                            "batch": batch,
                                            "device": "cuda",
                                            "controller": controller,
                                            "num_control_steps": num_control_steps,
                                            "plant_substeps": plant_substeps,
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
        for controller in controllers:
            for num_control_steps in num_control_steps_list:
                for plant_substeps in plant_substeps_list:
                    for batch in batches:
                        for range_mode in range_bound_modes:
                            for drop_mode in dropped_merge_modes:
                                rows.append(
                                    _run_case(
                                        batch=batch,
                                        num_control_steps=num_control_steps,
                                        plant_substeps=plant_substeps,
                                        order=order,
                                        h=h,
                                        controller=controller,
                                        dtype=dtype,
                                        device=device,
                                        tol=tol,
                                        range_bound_mode=range_mode,
                                        dropped_merge_mode=drop_mode,
                                    )
                                )

    dense_cpu_elapsed = {
        (
            int(row["batch"]),
            str(row["controller"]),
            int(row["num_control_steps"]),
            int(row["plant_substeps"]),
            str(row.get("range_bound_mode", "interval")),
            str(row.get("dropped_merge_mode", "merged")),
        ): float(row["elapsed_ms"])
        for row in rows
        if row.get("device") == "cpu" and row.get("status") == "ok"
    }
    for row in rows:
        if row.get("device") != "cuda" or row.get("status") != "ok":
            continue
        key = (
            int(row["batch"]),
            str(row["controller"]),
            int(row["num_control_steps"]),
            int(row["plant_substeps"]),
            str(row.get("range_bound_mode", "interval")),
            str(row.get("dropped_merge_mode", "merged")),
        )
        elapsed = float(row["elapsed_ms"])
        if key in dense_cpu_elapsed and elapsed > 0:
            row["speedup_vs_dense_cpu"] = dense_cpu_elapsed[key] / elapsed

    ok_rows = [row for row in rows if row.get("status") == "ok"]
    containment_ok = all(bool(row.get("containment_pass")) for row in ok_rows) if ok_rows else False
    if not containment_ok:
        recommendation = "NEEDS_REMAINDER_REDESIGN"
    elif _first_cuda_win(rows) is not None:
        recommendation = "GPU_PATH_CONTINUE"
    else:
        recommendation = "GPU_PATH_CONTINUE" if ok_rows else "NEEDS_REMAINDER_REDESIGN"
    for row in rows:
        row["recommendation"] = recommendation

    summary = out_dir / "nncs_summary.csv"
    _write_csv(summary, rows)
    report = _write_report(out_dir, rows, recommendation, cuda_available)
    return summary, report, rows, recommendation


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a batched dense TM NNCS demo")
    parser.add_argument("--out-dir", default="outputs/batched_dense_nncs_demo")
    parser.add_argument("--num-control-steps", default="10")
    parser.add_argument("--plant-substeps", default="5")
    parser.add_argument("--batches", default="1,8,32,128,512,2048")
    parser.add_argument("--controller", default="affine,relu_ibp")
    parser.add_argument("--devices", default="cpu,cuda")
    parser.add_argument("--dtype", default="float64", choices=["float64", "float32"])
    parser.add_argument("--range-bound-mode", default="interval")
    parser.add_argument("--dropped-merge-mode", default="merged")
    parser.add_argument("--order", type=int, default=3)
    parser.add_argument("--h", type=float, default=0.01)
    args = parser.parse_args()
    summary, report, _rows, recommendation = run_experiment(
        Path(args.out_dir),
        batches=_parse_ints(args.batches, [1, 8, 32, 128, 512]),
        num_control_steps_list=_parse_ints(args.num_control_steps, [10]),
        plant_substeps_list=_parse_ints(args.plant_substeps, [5]),
        controllers=_parse_strings(args.controller, ["affine"], {"affine", "relu_ibp"}),
        devices=_parse_strings(args.devices, ["cpu"], {"cpu", "cuda"}),
        dtype=_dtype_from_name(args.dtype),
        range_bound_modes=_parse_strings(args.range_bound_mode, ["interval"], {"interval", "blocked_interval"}),
        dropped_merge_modes=_parse_strings(args.dropped_merge_mode, ["merged"], {"merged", "grouped"}),
        order=args.order,
        h=args.h,
    )
    print(f"batched dense NNCS demo complete: recommendation={recommendation}")
    print(f"summary: {summary}")
    print(f"report: {report}")


if __name__ == "__main__":
    main()
