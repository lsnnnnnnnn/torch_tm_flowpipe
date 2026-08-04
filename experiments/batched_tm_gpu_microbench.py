#!/usr/bin/env python3
"""Production batched Taylor-model CPU/CUDA microbenchmark.

This file is diagnostic-only. It does not add a reachability algorithm, a Flow*
mechanism, or a symbolic queue. The benchmark asks whether dense, batched
Taylor-model operators have a speed path that the sparse Python object
representation cannot expose.  Every dense row calls the canonical package
implementation; this file contains no duplicate multiplication/range kernel.
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch

from torch_tm_flowpipe import Interval, Polynomial, PolynomialODE, TaylorModel, TMVector, flowpipe_step_from_tm
from torch_tm_flowpipe.batched_dense_tm import (
    BatchedMonomialBasis,
    BatchedPolynomial,
    BatchedTaylorModel,
    dense_picard_validate_step,
)


DEFAULT_BATCHES = [1, 8, 32, 48, 128]
CORE_OPERATIONS = [
    "tm_affine_map",
    "polynomial_add",
    "tm_mul_trunc",
    "tm_range_bound",
    "picard_validate_step",
]
CSV_FIELDS = [
    "case_name",
    "dim",
    "order",
    "monomial_profile",
    "monomial_count",
    "multiplication_pair_count",
    "batch",
    "operation",
    "implementation",
    "device",
    "dtype",
    "warmup",
    "repeats",
    "status",
    "skip_reason",
    "min_ms",
    "median_ms",
    "mean_ms",
    "samples_per_second",
    "speedup_vs_torch_cpu",
    "output_shape",
    "notes",
]
RECOMMENDATIONS = {
    "GPU_PATH_PROMISING",
    "DENSE_GPU_PATH_LIMITED",
    "NEEDS_REPRESENTATION_REDESIGN",
    "STOP_PYTHON_PLANT_TM_FOR_SPEED",
}


@dataclass(frozen=True)
class BenchmarkSetting:
    name: str
    dim: int
    order: int
    monomial_profile: str


@dataclass(frozen=True)
class MultiplicationPlan:
    left: torch.Tensor
    right: torch.Tensor
    target: torch.Tensor
    pair_count: int


def default_settings(include_order6: bool = False) -> list[BenchmarkSetting]:
    settings = [BenchmarkSetting("vdp_nvars3_order4", 2, 4, "vdp_tau_plus_two_generators")]
    if include_order6:
        settings.append(BenchmarkSetting("vdp_nvars3_order6_optional", 2, 6, "vdp_tau_plus_two_generators"))
    return settings


def total_degree_exponents(dim: int, order: int) -> list[tuple[int, ...]]:
    out: list[tuple[int, ...]] = []

    def rec(remaining_dim: int, remaining_degree: int, prefix: tuple[int, ...]) -> None:
        if remaining_dim == 1:
            out.append(prefix + (remaining_degree,))
            return
        for value in range(remaining_degree + 1):
            rec(remaining_dim - 1, remaining_degree - value, prefix + (value,))

    for total in range(order + 1):
        rec(dim, total, ())
    return out


def multiplication_plan(
    exponents: Sequence[tuple[int, ...]],
    order: int,
    device: torch.device,
) -> MultiplicationPlan:
    index = {exp: i for i, exp in enumerate(exponents)}
    left: list[int] = []
    right: list[int] = []
    target: list[int] = []
    for i, exp_i in enumerate(exponents):
        for j, exp_j in enumerate(exponents):
            exp = tuple(a + b for a, b in zip(exp_i, exp_j))
            if sum(exp) <= order:
                left.append(i)
                right.append(j)
                target.append(index[exp])
    return MultiplicationPlan(
        torch.as_tensor(left, dtype=torch.long, device=device),
        torch.as_tensor(right, dtype=torch.long, device=device),
        torch.as_tensor(target, dtype=torch.long, device=device),
        len(left),
    )


def _dtype_from_name(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float64":
        return torch.float64
    raise ValueError(f"unsupported dtype: {name}")


def _split_csv_args(values: Sequence[Any] | None) -> list[str]:
    if values is None:
        return []
    parts: list[str] = []
    for value in values:
        parts.extend(part.strip() for part in str(value).split(",") if part.strip())
    return parts


def _parse_int_list(values: Sequence[Any] | None, default: Sequence[int]) -> list[int]:
    parts = _split_csv_args(values)
    return [int(part) for part in parts] if parts else list(default)


def _parse_devices(values: Sequence[Any] | None) -> list[str] | None:
    parts = _split_csv_args(values)
    if not parts:
        return None
    allowed = {"cpu", "cuda"}
    unknown = sorted(set(parts) - allowed)
    if unknown:
        raise SystemExit(f"unknown device(s): {', '.join(unknown)}")
    return parts


def _format_number(value: Any) -> Any:
    if isinstance(value, float):
        if math.isfinite(value):
            return f"{value:.9g}"
        return ""
    return value


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _format_number(row.get(field, "")) for field in CSV_FIELDS})


def _torch_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _measure(
    fn: Callable[[], Any],
    *,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> tuple[float, float, float]:
    repeats = max(1, int(repeats))
    with torch.no_grad():
        for _ in range(max(0, int(warmup))):
            fn()
        _torch_sync(device)
        timings: list[float] = []
        for _ in range(repeats):
            start = time.perf_counter()
            fn()
            _torch_sync(device)
            timings.append((time.perf_counter() - start) * 1000.0)
    return min(timings), statistics.median(timings), statistics.fmean(timings)


def _row(
    *,
    setting: BenchmarkSetting,
    monomial_count: int,
    pair_count: int,
    batch: int,
    operation: str,
    implementation: str,
    device: str,
    dtype: torch.dtype,
    warmup: int,
    repeats: int,
    status: str,
    skip_reason: str = "",
    min_ms: float | str = "",
    median_ms: float | str = "",
    mean_ms: float | str = "",
    output_shape: str = "",
    notes: str = "",
) -> dict[str, Any]:
    samples_per_second: float | str = ""
    if isinstance(median_ms, float) and median_ms > 0:
        samples_per_second = float(batch) / (median_ms / 1000.0)
    return {
        "case_name": setting.name,
        "dim": setting.dim,
        "order": setting.order,
        "monomial_profile": setting.monomial_profile,
        "monomial_count": monomial_count,
        "multiplication_pair_count": pair_count,
        "batch": batch,
        "operation": operation,
        "implementation": implementation,
        "device": device,
        "dtype": str(dtype).replace("torch.", ""),
        "warmup": warmup,
        "repeats": repeats,
        "status": status,
        "skip_reason": skip_reason,
        "min_ms": min_ms,
        "median_ms": median_ms,
        "mean_ms": mean_ms,
        "samples_per_second": samples_per_second,
        "speedup_vs_torch_cpu": "",
        "output_shape": output_shape,
        "notes": notes,
    }


def _make_coefficients(
    batch: int,
    dim: int,
    terms: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    coeffs = 0.01 * torch.randn((batch, dim, terms), generator=gen, dtype=dtype)
    coeffs[:, :, 0] += 0.1
    return coeffs.to(device)


def _make_affine_data(
    batch: int,
    dim: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    center = 0.2 * torch.randn((batch, dim), generator=gen, dtype=dtype)
    radius = 0.01 + 0.05 * torch.rand((batch, dim), generator=gen, dtype=dtype)
    lo = center - radius
    hi = center + radius
    matrix = 0.2 * torch.randn((dim, dim), generator=gen, dtype=dtype)
    bias = 0.05 * torch.randn((dim,), generator=gen, dtype=dtype)
    return lo.to(device), hi.to(device), matrix.to(device), bias.to(device)


def _operation_working_bytes(
    operation: str,
    *,
    batch: int,
    dim: int,
    term_count: int,
    pair_count: int,
    dtype: torch.dtype,
) -> int:
    element_size = torch.empty((), dtype=dtype).element_size()
    coeff_bytes = batch * dim * term_count * element_size
    pair_bytes = batch * dim * pair_count * element_size
    if operation == "tm_affine_map":
        return batch * dim * element_size * 8 + dim * dim * element_size
    if operation == "polynomial_add":
        return coeff_bytes * 3
    if operation == "tm_mul_trunc":
        return coeff_bytes * 3 + pair_bytes
    if operation == "tm_range_bound":
        return coeff_bytes * 2 + batch * term_count * dim * element_size * 4
    if operation == "picard_validate_step":
        return coeff_bytes * 4 + pair_bytes * (2 if dim == 2 else 1)
    return coeff_bytes


def _memory_cap_for_device(device: torch.device, max_working_bytes: int) -> int:
    cap = int(max_working_bytes)
    if device.type == "cuda":
        try:
            free, _total = torch.cuda.mem_get_info(device)
            cap = min(cap, int(free * 0.60))
        except Exception:
            pass
    return cap


def _run_torch_operation(
    operation: str,
    *,
    setting: BenchmarkSetting,
    exponents: Sequence[tuple[int, ...]],
    batch: int,
    device: torch.device,
    dtype: torch.dtype,
    warmup: int,
    repeats: int,
    seed: int,
) -> tuple[float, float, float, str]:
    basis_dim = setting.dim + 1 if "vdp" in setting.monomial_profile else setting.dim
    if operation == "picard_validate_step":
        basis_dim = setting.dim + 1
    basis = BatchedMonomialBasis.build(basis_dim, setting.order, device)
    term_count = basis.num_terms
    coeffs = _make_coefficients(batch, setting.dim, term_count, dtype=dtype, device=device, seed=seed)
    domain_lo = torch.full((batch, basis_dim), -1.0, dtype=dtype, device=device)
    domain_hi = torch.full((batch, basis_dim), 1.0, dtype=dtype, device=device)
    if operation == "picard_validate_step":
        domain_lo[:, -1] = 0.0
        domain_hi[:, -1] = 0.01
    remainder_radius = torch.full((batch, setting.dim), 1e-8, dtype=dtype, device=device)
    tm = BatchedTaylorModel(
        BatchedPolynomial(coeffs, basis),
        -remainder_radius,
        remainder_radius,
        domain_lo,
        domain_hi,
    )
    other = BatchedTaylorModel(
        BatchedPolynomial(_make_coefficients(batch, setting.dim, term_count, dtype=dtype, device=device, seed=seed + 1), basis),
        -remainder_radius,
        remainder_radius,
        domain_lo,
        domain_hi,
    )
    if operation == "tm_affine_map":
        _lo, _hi, matrix, bias = _make_affine_data(batch, setting.dim, dtype=dtype, device=device, seed=seed + 2)

        def fn() -> Any:
            return tm.affine_map(matrix, bias)

        output_shape = f"tm=({batch},{setting.dim},{term_count})"
    elif operation == "polynomial_add":

        def fn() -> Any:
            return tm.poly.add(other.poly)

        output_shape = f"coeffs=({batch},{setting.dim},{term_count})"
    elif operation == "tm_mul_trunc":

        def fn() -> Any:
            return tm.component(0).mul_trunc(other.component(0))

        output_shape = f"tm=({batch},1,{term_count})"
    elif operation == "tm_range_bound":

        def fn() -> Any:
            return tm.range_bound()

        output_shape = f"lo_hi=({batch},{setting.dim})"
    elif operation == "picard_validate_step":
        if setting.dim != 2:
            raise ValueError("production Picard microbenchmark currently uses the two-state VDP plant")
        ode = PolynomialODE.from_system_spec(
            {
                "state_names": ["x", "y"],
                "rhs": [
                    {"terms": [{"coefficient": 1.0, "powers": [0, 1]}]},
                    {"terms": [
                        {"coefficient": 1.0, "powers": [0, 1]},
                        {"coefficient": -1.0, "powers": [1, 0]},
                        {"coefficient": -1.0, "powers": [2, 1]},
                    ]},
                ],
            }
        )

        def fn() -> Any:
            return dense_picard_validate_step(
                ode,
                tm,
                h=0.01,
                order=setting.order,
                tau_index=basis_dim - 1,
                target_remainder_radius=1.0,
                cutoff_threshold=1e-10,
                validation_mode="flowstar_raw_remainder_compat",
            )

        output_shape = f"validated_tm=({batch},{setting.dim},{term_count})"
    else:
        raise ValueError(f"unknown operation: {operation}")
    min_ms, median_ms, mean_ms = _measure(fn, device=device, warmup=warmup, repeats=repeats)
    return min_ms, median_ms, mean_ms, output_shape


def _poly_from_coeff_vector(coeffs: torch.Tensor, exponents: Sequence[tuple[int, ...]], n_vars: int) -> Polynomial:
    terms = {
        exp: coeffs[i].clone()
        for i, exp in enumerate(exponents)
        if bool(torch.any(coeffs[i] != 0))
    }
    return Polynomial(terms, n_vars=n_vars)


def _make_scalar_polys(
    batch: int,
    dim: int,
    exponents: Sequence[tuple[int, ...]],
    *,
    dtype: torch.dtype,
    seed: int,
) -> list[list[Polynomial]]:
    coeffs = _make_coefficients(
        batch,
        dim,
        len(exponents),
        dtype=dtype,
        device=torch.device("cpu"),
        seed=seed,
    )
    return [
        [_poly_from_coeff_vector(coeffs[b, component], exponents, dim) for component in range(dim)]
        for b in range(batch)
    ]


def _run_scalar_operation(
    operation: str,
    *,
    setting: BenchmarkSetting,
    exponents: Sequence[tuple[int, ...]],
    batch: int,
    dtype: torch.dtype,
    warmup: int,
    repeats: int,
    seed: int,
) -> tuple[float, float, float, str]:
    device = torch.device("cpu")
    if operation == "tm_affine_map":
        lo, hi, matrix, bias = _make_affine_data(
            batch,
            setting.dim,
            dtype=dtype,
            device=device,
            seed=seed,
        )
        boxes = [[Interval(lo[b, i], hi[b, i]) for i in range(setting.dim)] for b in range(batch)]

        def fn() -> Any:
            out = []
            for box in boxes:
                mapped = []
                for row in range(setting.dim):
                    acc = Interval.point(bias[row])
                    for col in range(setting.dim):
                        acc = acc + box[col] * matrix[row, col]
                    mapped.append(acc)
                out.append(mapped)
            return out

        output_shape = f"interval_objects={batch * setting.dim}"
    elif operation in {"polynomial_add", "tm_mul_trunc"}:
        polys_a = _make_scalar_polys(batch, setting.dim, exponents, dtype=dtype, seed=seed)
        polys_b = _make_scalar_polys(batch, setting.dim, exponents, dtype=dtype, seed=seed + 1)
        if operation == "polynomial_add":

            def fn() -> Any:
                return [
                    [left + right for left, right in zip(row_a, row_b)]
                    for row_a, row_b in zip(polys_a, polys_b)
                ]

        else:

            def fn() -> Any:
                return [
                    [left.mul_truncate(right, setting.order)[0] for left, right in zip(row_a, row_b)]
                    for row_a, row_b in zip(polys_a, polys_b)
                ]

        output_shape = f"polynomial_objects={batch * setting.dim}"
    elif operation == "tm_range_bound":
        domain = [Interval(-1.0, 1.0) for _ in range(setting.dim)]
        tms = [
            [
                TaylorModel(poly, Interval(-1e-6, 1e-6), domain, order=setting.order)
                for poly in row
            ]
            for row in _make_scalar_polys(batch, setting.dim, exponents, dtype=dtype, seed=seed)
        ]

        def fn() -> Any:
            return [[model.range_box() for model in row] for row in tms]

        output_shape = f"taylor_model_objects={batch * setting.dim}"
    elif operation == "picard_validate_step":
        domain = [Interval(-1.0, 1.0) for _ in range(setting.dim)]
        tmvs = [
            TMVector([TaylorModel(poly, Interval.zero(), domain, order=setting.order) for poly in row])
            for row in _make_scalar_polys(batch, setting.dim, exponents, dtype=dtype, seed=seed)
        ]
        ode = PolynomialODE.from_system_spec(
            {
                "state_names": ["x", "y"],
                "rhs": [
                    {"terms": [{"coefficient": 1.0, "powers": [0, 1]}]},
                    {"terms": [
                        {"coefficient": 1.0, "powers": [0, 1]},
                        {"coefficient": -1.0, "powers": [1, 0]},
                        {"coefficient": -1.0, "powers": [2, 1]},
                    ]},
                ],
            }
        )

        def fn() -> Any:
            return [
                flowpipe_step_from_tm(
                    ode,
                    tmv,
                    h=0.01,
                    order=setting.order,
                    target_remainder_radius=1.0,
                    cutoff_threshold=1e-10,
                    validation_mode="flowstar_raw_remainder_compat",
                )
                for tmv in tmvs
            ]

        output_shape = f"tmvector_objects={batch}"
    else:
        raise ValueError(f"unknown operation: {operation}")
    min_ms, median_ms, mean_ms = _measure(fn, device=device, warmup=warmup, repeats=repeats)
    return min_ms, median_ms, mean_ms, output_shape


def _add_speedups(rows: list[dict[str, Any]]) -> None:
    baselines: dict[tuple[str, int, str], float] = {}
    for row in rows:
        if (
            row["status"] == "ok"
            and row["implementation"] == "torch_dense"
            and row["device"] == "cpu"
            and isinstance(row["median_ms"], float)
        ):
            baselines[(row["case_name"], int(row["batch"]), row["operation"])] = row["median_ms"]
    for row in rows:
        key = (row["case_name"], int(row["batch"]), row["operation"])
        baseline = baselines.get(key)
        median = row.get("median_ms")
        if baseline and isinstance(median, float) and median > 0:
            row["speedup_vs_torch_cpu"] = baseline / median


def _ok_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("status") == "ok" and isinstance(row.get("median_ms"), float)]


def _first_gpu_win_by_operation(rows: Sequence[dict[str, Any]]) -> dict[str, int | None]:
    wins: dict[str, int | None] = {}
    for operation in CORE_OPERATIONS:
        op_rows = [
            row
            for row in _ok_rows(rows)
            if row["implementation"] == "torch_dense"
            and row["device"] == "cuda"
            and row["operation"] == operation
            and isinstance(row.get("speedup_vs_torch_cpu"), float)
            and row["speedup_vs_torch_cpu"] > 1.0
        ]
        wins[operation] = min((int(row["batch"]) for row in op_rows), default=None)
    return wins


def _dominant_operation(rows: Sequence[dict[str, Any]], *, implementation: str, device: str) -> str:
    candidates = [
        row
        for row in _ok_rows(rows)
        if row["implementation"] == implementation and row["device"] == device
    ]
    if not candidates:
        return "not measured"
    latest_by_case_op: dict[tuple[str, str], dict[str, Any]] = {}
    for row in candidates:
        key = (row["case_name"], row["operation"])
        old = latest_by_case_op.get(key)
        if old is None or int(row["batch"]) > int(old["batch"]):
            latest_by_case_op[key] = row
    totals: dict[str, float] = {}
    for row in latest_by_case_op.values():
        totals[row["operation"]] = totals.get(row["operation"], 0.0) + float(row["median_ms"])
    if not totals:
        return "not measured"
    operation, value = max(totals.items(), key=lambda item: item[1])
    return f"{operation} ({value:.3g} ms summed over largest measured batches)"


def _batch1_gpu_answer(rows: Sequence[dict[str, Any]]) -> str:
    comparisons = [
        row
        for row in _ok_rows(rows)
        if row["implementation"] == "torch_dense"
        and row["device"] == "cuda"
        and int(row["batch"]) == 1
        and isinstance(row.get("speedup_vs_torch_cpu"), float)
    ]
    if not comparisons:
        return "CUDA was unavailable or produced no batch=1 rows in this run, so no GPU-vs-CPU claim is made."
    slower = [row for row in comparisons if row["speedup_vs_torch_cpu"] < 1.0]
    faster = len(comparisons) - len(slower)
    return (
        f"Yes for {len(slower)}/{len(comparisons)} measured batch=1 CUDA rows; "
        f"{faster}/{len(comparisons)} were faster than torch CPU."
    )


def _choose_recommendation(rows: Sequence[dict[str, Any]]) -> tuple[str, str]:
    cuda_rows = [
        row
        for row in _ok_rows(rows)
        if row["implementation"] == "torch_dense"
        and row["device"] == "cuda"
        and isinstance(row.get("speedup_vs_torch_cpu"), float)
    ]
    if not cuda_rows:
        return (
            "NEEDS_REPRESENTATION_REDESIGN",
            "No CUDA speed evidence was produced. The current sparse dict/scalar object path is still not a GPU representation.",
        )
    meaningful = [row for row in cuda_rows if int(row["batch"]) >= 128]
    strong = [row for row in meaningful if row["speedup_vs_torch_cpu"] >= 1.5]
    wins = _first_gpu_win_by_operation(rows)
    core_win_count = sum(1 for op in CORE_OPERATIONS if wins.get(op) is not None and wins[op] <= 2048)
    if core_win_count >= 4 and len(strong) >= max(4, len(meaningful) // 3):
        return (
            "GPU_PATH_PROMISING",
            "Dense batched kernels show clear CUDA speedups at realistic batch sizes.",
        )
    if strong or core_win_count > 0:
        return (
            "DENSE_GPU_PATH_LIMITED",
            "The dense batched representation is operational, but its measured CUDA advantage is too narrow for an end-to-end GPU claim.",
        )
    return (
        "STOP_PYTHON_PLANT_TM_FOR_SPEED",
        "Dense CUDA kernels did not beat torch CPU at meaningful batch sizes in this run.",
    )


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def _build_report(rows: Sequence[dict[str, Any]], *, output_dir: Path, dtype: torch.dtype) -> str:
    recommendation, recommendation_reason = _choose_recommendation(rows)
    wins = _first_gpu_win_by_operation(rows)
    cuda_available = torch.cuda.is_available()
    cuda_name = torch.cuda.get_device_name(0) if cuda_available else "unavailable"
    ok = _ok_rows(rows)
    skipped = [row for row in rows if row.get("status") == "skipped"]

    win_rows = [
        [operation, wins[operation] if wins[operation] is not None else "no CUDA win measured"]
        for operation in CORE_OPERATIONS
    ]
    scalar_rows = [
        row
        for row in ok
        if row["implementation"] == "python_scalar_sparse"
        and isinstance(row.get("speedup_vs_torch_cpu"), float)
    ]
    scalar_note = "No scalar sparse rows were measured."
    if scalar_rows:
        ratios = [row["speedup_vs_torch_cpu"] for row in scalar_rows]
        scalar_note = (
            "Existing sparse Python rows ran at "
            f"{min(ratios):.3g}x to {max(ratios):.3g}x of torch dense CPU throughput "
            "for the measured scalar batches."
        )

    lines = [
        "# Batched TM GPU Microbenchmark Report",
        "",
        "This report is diagnostic-only. It does not claim a new reachability algorithm, "
        "and it does not use the Flow* C++ probe as an implementation route.",
        "",
        "## Run Metadata",
        "",
        f"- Output directory: `{output_dir}`",
        f"- PyTorch version: `{torch.__version__}`",
        f"- CUDA available: `{cuda_available}`",
        f"- CUDA device: `{cuda_name}`",
        f"- dtype: `{str(dtype).replace('torch.', '')}`",
        f"- OK rows: `{len(ok)}`",
        f"- Skipped rows: `{len(skipped)}`",
        "",
        "## Direct Answers",
        "",
        f"- At batch=1, is PyTorch GPU slower than CPU? {_batch1_gpu_answer(rows)}",
        "- What batch size is needed before GPU wins, if any?",
        "",
        _markdown_table(["operation", "first CUDA batch with speedup > 1.0"], win_rows),
        "",
        f"- Which operation dominates torch CPU runtime? {_dominant_operation(rows, implementation='torch_dense', device='cpu')}",
        f"- Which operation dominates torch CUDA runtime? {_dominant_operation(rows, implementation='torch_dense', device='cuda')}",
        f"- Does the production representation expose tensorized work? Yes: these dense rows call "
        f"`BatchedPolynomial`, `BatchedTaylorModel`, and `dense_picard_validate_step` directly. "
        f"The sparse reference remains dictionary based. {scalar_note}",
        "- What representation is measured? A canonical complete monomial basis with batched coefficient/domain/"
        "remainder tensors and cached multiplication/integration routes; VDP order 4 uses three variables and 35 slots.",
        "- Is the project still justified as PyTorch-native, or should plant remain Flow* C++? "
        f"{recommendation_reason}",
        "",
        f"## Final Recommendation: {recommendation}",
        "",
        "Allowed recommendation values are `GPU_PATH_PROMISING`, `DENSE_GPU_PATH_LIMITED`, "
        "`NEEDS_REPRESENTATION_REDESIGN`, and `STOP_PYTHON_PLANT_TM_FOR_SPEED`.",
    ]
    return "\n".join(lines) + "\n"


def run_benchmark(
    output_dir: str | Path = REPO_ROOT / "outputs" / "batched_tm_gpu_microbench",
    *,
    batch_sizes: Sequence[int] = DEFAULT_BATCHES,
    settings: Sequence[BenchmarkSetting] | None = None,
    devices: Sequence[str] | None = None,
    dtype: torch.dtype = torch.float64,
    warmup: int = 3,
    repeats: int = 7,
    max_working_bytes: int = 768 * 1024 * 1024,
    max_scalar_batch: int = 128,
    max_scalar_terms: int = 128,
    include_scalar: bool = True,
    include_order6: bool = False,
    seed: int = 20260608,
) -> tuple[Path, Path, list[dict[str, Any]]]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    selected_settings = list(settings if settings is not None else default_settings(include_order6))
    selected_devices = list(devices) if devices is not None else ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    rows: list[dict[str, Any]] = []

    for setting_index, setting in enumerate(selected_settings):
        exponents = total_degree_exponents(setting.dim, setting.order)
        dense_basis_dim = setting.dim + 1 if "vdp" in setting.monomial_profile else setting.dim
        dense_basis = BatchedMonomialBasis.build(dense_basis_dim, setting.order, "cpu")
        term_count = dense_basis.num_terms
        pair_count = int(dense_basis.mul_left_indices.numel())
        for batch in batch_sizes:
            for device_name in selected_devices:
                if device_name == "cuda" and not torch.cuda.is_available():
                    for operation in CORE_OPERATIONS:
                        rows.append(
                            _row(
                                setting=setting,
                                monomial_count=term_count,
                                pair_count=pair_count,
                                batch=int(batch),
                                operation=operation,
                                implementation="torch_dense",
                                device="cuda",
                                dtype=dtype,
                                warmup=warmup,
                                repeats=repeats,
                                status="skipped",
                                skip_reason="CUDA unavailable",
                            )
                        )
                    continue
                device = torch.device(device_name)
                memory_cap = _memory_cap_for_device(device, max_working_bytes)
                for op_index, operation in enumerate(CORE_OPERATIONS):
                    estimated = _operation_working_bytes(
                        operation,
                        batch=int(batch),
                        dim=setting.dim,
                        term_count=term_count,
                        pair_count=pair_count,
                        dtype=dtype,
                    )
                    if estimated > memory_cap:
                        rows.append(
                            _row(
                                setting=setting,
                                monomial_count=term_count,
                                pair_count=pair_count,
                                batch=int(batch),
                                operation=operation,
                                implementation="torch_dense",
                                device=device_name,
                                dtype=dtype,
                                warmup=warmup,
                                repeats=repeats,
                                status="skipped",
                                skip_reason=f"estimated working set {estimated} bytes exceeds cap {memory_cap} bytes",
                            )
                        )
                        continue
                    try:
                        min_ms, median_ms, mean_ms, output_shape = _run_torch_operation(
                            operation,
                            setting=setting,
                            exponents=exponents,
                            batch=int(batch),
                            device=device,
                            dtype=dtype,
                            warmup=warmup,
                            repeats=repeats,
                            seed=seed + setting_index * 1000 + op_index,
                        )
                        rows.append(
                            _row(
                                setting=setting,
                                monomial_count=term_count,
                                pair_count=pair_count,
                                batch=int(batch),
                                operation=operation,
                                implementation="torch_dense",
                                device=device_name,
                                dtype=dtype,
                                warmup=warmup,
                                repeats=repeats,
                                status="ok",
                                min_ms=min_ms,
                                median_ms=median_ms,
                                mean_ms=mean_ms,
                                output_shape=output_shape,
                            )
                        )
                    except RuntimeError as exc:
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                        rows.append(
                            _row(
                                setting=setting,
                                monomial_count=term_count,
                                pair_count=pair_count,
                                batch=int(batch),
                                operation=operation,
                                implementation="torch_dense",
                                device=device_name,
                                dtype=dtype,
                                warmup=warmup,
                                repeats=repeats,
                                status="skipped",
                                skip_reason=f"runtime error: {exc}",
                            )
                        )
            if include_scalar:
                for op_index, operation in enumerate(CORE_OPERATIONS):
                    if int(batch) > int(max_scalar_batch):
                        rows.append(
                            _row(
                                setting=setting,
                                monomial_count=term_count,
                                pair_count=pair_count,
                                batch=int(batch),
                                operation=operation,
                                implementation="python_scalar_sparse",
                                device="cpu",
                                dtype=dtype,
                                warmup=warmup,
                                repeats=repeats,
                                status="skipped",
                                skip_reason=f"batch exceeds max_scalar_batch={max_scalar_batch}",
                                notes="existing sparse object path is intentionally capped to avoid runaway diagnostic time",
                            )
                        )
                        continue
                    if term_count > int(max_scalar_terms):
                        rows.append(
                            _row(
                                setting=setting,
                                monomial_count=term_count,
                                pair_count=pair_count,
                                batch=int(batch),
                                operation=operation,
                                implementation="python_scalar_sparse",
                                device="cpu",
                                dtype=dtype,
                                warmup=warmup,
                                repeats=repeats,
                                status="skipped",
                                skip_reason=f"term count exceeds max_scalar_terms={max_scalar_terms}",
                                notes="existing sparse object path is intentionally capped to avoid runaway diagnostic time",
                            )
                        )
                        continue
                    try:
                        min_ms, median_ms, mean_ms, output_shape = _run_scalar_operation(
                            operation,
                            setting=setting,
                            exponents=exponents,
                            batch=int(batch),
                            dtype=dtype,
                            warmup=warmup,
                            repeats=repeats,
                            seed=seed + setting_index * 1000 + op_index,
                        )
                        rows.append(
                            _row(
                                setting=setting,
                                monomial_count=term_count,
                                pair_count=pair_count,
                                batch=int(batch),
                                operation=operation,
                                implementation="python_scalar_sparse",
                                device="cpu",
                                dtype=dtype,
                                warmup=warmup,
                                repeats=repeats,
                                status="ok",
                                min_ms=min_ms,
                                median_ms=median_ms,
                                mean_ms=mean_ms,
                                output_shape=output_shape,
                                notes="production sparse Polynomial/TaylorModel reference path",
                            )
                        )
                    except RuntimeError as exc:
                        rows.append(
                            _row(
                                setting=setting,
                                monomial_count=term_count,
                                pair_count=pair_count,
                                batch=int(batch),
                                operation=operation,
                                implementation="python_scalar_sparse",
                                device="cpu",
                                dtype=dtype,
                                warmup=warmup,
                                repeats=repeats,
                                status="skipped",
                                skip_reason=f"runtime error: {exc}",
                            )
                        )

    _add_speedups(rows)
    summary_path = output / "gpu_microbench_summary.csv"
    report_path = output / "gpu_microbench_report.md"
    _write_csv(summary_path, rows)
    report_path.write_text(_build_report(rows, output_dir=output, dtype=dtype), encoding="utf-8")
    return summary_path, report_path, rows


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark batched dense TM kernels on torch CPU/CUDA.")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "outputs" / "batched_tm_gpu_microbench"))
    parser.add_argument(
        "--batches",
        nargs="+",
        default=None,
        help="Batch sizes, either comma-separated or space-separated.",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument(
        "--devices",
        nargs="+",
        default=None,
        help="Devices, either comma-separated or space-separated. Defaults to CPU plus CUDA when available.",
    )
    parser.add_argument("--max-working-bytes-mib", type=float, default=768.0)
    parser.add_argument("--max-scalar-batch", type=int, default=1)
    parser.add_argument("--max-scalar-terms", type=int, default=70)
    parser.add_argument("--no-scalar", action="store_true")
    parser.add_argument("--include-order6", action="store_true")
    parser.add_argument("--torch-threads", type=int, default=None)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Short smoke run: dim=2/order=4, batches 1/8/32, warmup 1, repeats 2.",
    )
    parser.add_argument(
        "--tiny-config",
        action="store_true",
        help="Tiny dim=2/order=2 config for CPU-only CLI smoke tests.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.torch_threads is not None:
        torch.set_num_threads(int(args.torch_threads))
    dtype = _dtype_from_name(args.dtype)
    settings = None
    batches = _parse_int_list(args.batches, DEFAULT_BATCHES)
    devices = _parse_devices(args.devices)
    warmup = args.warmup
    repeats = args.repeats
    if args.tiny_config:
        settings = [BenchmarkSetting("tiny_dim2_order2", 2, 2, "tiny_cli_smoke")]
        if args.batches is None:
            batches = [1]
    if args.quick:
        settings = [BenchmarkSetting("vdp_nvars3_order4", 2, 4, "vdp_tau_plus_two_generators")]
        batches = [1, 8, 32]
        warmup = min(warmup, 1)
        repeats = min(repeats, 2)
    summary, report, _rows = run_benchmark(
        args.output_dir,
        batch_sizes=batches,
        settings=settings,
        devices=devices,
        dtype=dtype,
        warmup=warmup,
        repeats=repeats,
        max_working_bytes=int(args.max_working_bytes_mib * 1024 * 1024),
        max_scalar_batch=args.max_scalar_batch,
        max_scalar_terms=args.max_scalar_terms,
        include_scalar=not args.no_scalar,
        include_order6=args.include_order6,
    )
    print(f"wrote {summary}")
    print(f"wrote {report}")


if __name__ == "__main__":
    main()
