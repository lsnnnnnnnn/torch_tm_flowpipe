"""Experimental dense batched Taylor-model tensors.

This module is a correctness/parity prototype for small dense total-degree
Taylor models. It intentionally does not replace the sparse production
``Polynomial`` / ``TaylorModel`` path.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from itertools import product
import time
from typing import Any, Iterator, Sequence

import torch


@dataclass
class DenseTMProfiler:
    """Small opt-in profiler for dense batched TM experiments."""

    device: torch.device | str | None = None
    enabled: bool = True
    timings_ms: dict[str, float] = field(default_factory=dict)
    cuda_memory_allocated_bytes: int = 0
    cuda_memory_reserved_bytes: int = 0

    def _device(self) -> torch.device:
        return _as_device(self.device)

    def _sync(self) -> None:
        device = self._device()
        if self.enabled and device.type == "cuda":
            torch.cuda.synchronize(device)

    @contextmanager
    def measure(self, name: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        self._sync()
        start = time.perf_counter()
        try:
            yield
        finally:
            self._sync()
            self.timings_ms[name] = self.timings_ms.get(name, 0.0) + (time.perf_counter() - start) * 1000.0
            self.snapshot_cuda_memory()

    def snapshot_cuda_memory(self) -> None:
        device = self._device()
        if not self.enabled or device.type != "cuda":
            return
        self.cuda_memory_allocated_bytes = max(
            self.cuda_memory_allocated_bytes,
            int(torch.cuda.max_memory_allocated(device)),
        )
        self.cuda_memory_reserved_bytes = max(
            self.cuda_memory_reserved_bytes,
            int(torch.cuda.max_memory_reserved(device)),
        )

    def as_flat_dict(self) -> dict[str, float | int]:
        out: dict[str, float | int] = dict(self.timings_ms)
        out["cuda_memory_allocated_bytes"] = self.cuda_memory_allocated_bytes
        out["cuda_memory_reserved_bytes"] = self.cuda_memory_reserved_bytes
        return out


@contextmanager
def _profile_measure(profile: DenseTMProfiler | None, name: str) -> Iterator[None]:
    if profile is None:
        yield
        return
    with profile.measure(name):
        yield


def _as_device(device: torch.device | str | None) -> torch.device:
    return torch.device("cpu") if device is None else torch.device(device)


def _as_dtype(dtype: torch.dtype | None) -> torch.dtype:
    return torch.float64 if dtype is None else dtype


def _down(x: torch.Tensor) -> torch.Tensor:
    return torch.nextafter(x, torch.full_like(x, -torch.inf))


def _up(x: torch.Tensor) -> torch.Tensor:
    return torch.nextafter(x, torch.full_like(x, torch.inf))


def _interval_add(
    a_lo: torch.Tensor,
    a_hi: torch.Tensor,
    b_lo: torch.Tensor,
    b_hi: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _down(a_lo + b_lo), _up(a_hi + b_hi)


def _interval_sub(
    a_lo: torch.Tensor,
    a_hi: torch.Tensor,
    b_lo: torch.Tensor,
    b_hi: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _down(a_lo - b_hi), _up(a_hi - b_lo)


def _interval_mul(
    a_lo: torch.Tensor,
    a_hi: torch.Tensor,
    b_lo: torch.Tensor,
    b_hi: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    candidates = torch.stack(
        [a_lo * b_lo, a_lo * b_hi, a_hi * b_lo, a_hi * b_hi],
        dim=0,
    )
    return _down(torch.min(candidates, dim=0).values), _up(torch.max(candidates, dim=0).values)


def _interval_scale(
    lo: torch.Tensor,
    hi: torch.Tensor,
    scale: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    s = torch.as_tensor(scale, dtype=lo.dtype, device=lo.device)
    while s.ndim < lo.ndim:
        s = s.unsqueeze(-1)
    low = torch.minimum(lo * s, hi * s)
    high = torch.maximum(lo * s, hi * s)
    return _down(low), _up(high)


def _total_degree_exponents(dim: int, order: int) -> list[tuple[int, ...]]:
    exponents: list[tuple[int, ...]] = []

    def rec(pos: int, remaining: int, prefix: tuple[int, ...]) -> None:
        if pos == dim:
            if remaining == 0:
                exponents.append(prefix)
            return
        for value in range(remaining + 1):
            rec(pos + 1, remaining - value, prefix + (value,))

    for degree in range(order + 1):
        rec(0, degree, ())
    return exponents


def _power_interval_bounds(
    lo: torch.Tensor,
    hi: torch.Tensor,
    powers: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if powers.numel() == 0:
        empty = torch.empty((lo.shape[0], 0), dtype=lo.dtype, device=lo.device)
        return empty, empty
    powers = powers.to(device=lo.device, dtype=torch.long)
    max_power = int(torch.max(powers).detach().cpu())
    lo_cols: list[torch.Tensor] = []
    hi_cols: list[torch.Tensor] = []
    zero = torch.zeros_like(lo)
    one = torch.ones_like(lo)
    lo_abs = torch.minimum(torch.abs(lo), torch.abs(hi))
    hi_abs = torch.maximum(torch.abs(lo), torch.abs(hi))
    crosses_zero = (lo <= 0) & (hi >= 0)
    for power in range(max_power + 1):
        if power == 0:
            lo_cols.append(one)
            hi_cols.append(one)
        elif power % 2 == 1:
            endpoints = torch.stack([lo.pow(power), hi.pow(power)], dim=0)
            lo_cols.append(torch.min(endpoints, dim=0).values)
            hi_cols.append(torch.max(endpoints, dim=0).values)
        else:
            lo_cols.append(torch.where(crosses_zero, zero, lo_abs.pow(power)))
            hi_cols.append(hi_abs.pow(power))
    lo_table = torch.stack(lo_cols, dim=1)
    hi_table = torch.stack(hi_cols, dim=1)
    return _down(lo_table.index_select(1, powers)), _up(hi_table.index_select(1, powers))


def _monomial_interval_bounds_for_exponents(
    domain_lo: torch.Tensor,
    domain_hi: torch.Tensor,
    exponents: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if domain_lo.ndim == 1:
        domain_lo = domain_lo.unsqueeze(0)
        domain_hi = domain_hi.unsqueeze(0)
    if domain_lo.shape != domain_hi.shape:
        domain_lo, domain_hi = torch.broadcast_tensors(domain_lo, domain_hi)
    batch, dim = domain_lo.shape
    if exponents.shape[1] != dim:
        raise ValueError(f"exponent dimension {exponents.shape[1]} != domain dimension {dim}")
    exponents = exponents.to(device=domain_lo.device, dtype=torch.long)
    mono_lo = torch.ones((batch, exponents.shape[0]), dtype=domain_lo.dtype, device=domain_lo.device)
    mono_hi = torch.ones_like(mono_lo)
    for var_index in range(dim):
        power_lo, power_hi = _power_interval_bounds(
            domain_lo[:, var_index],
            domain_hi[:, var_index],
            exponents[:, var_index],
        )
        mono_lo, mono_hi = _interval_mul(mono_lo, mono_hi, power_lo, power_hi)
    return mono_lo, mono_hi


def _range_for_terms(
    coeffs: torch.Tensor,
    exponents: torch.Tensor,
    domain_lo: torch.Tensor,
    domain_hi: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if coeffs.shape[-1] == 0:
        out = torch.zeros(coeffs.shape[:-1], dtype=coeffs.dtype, device=coeffs.device)
        return out, out
    domain_lo = domain_lo.to(device=coeffs.device, dtype=coeffs.dtype)
    domain_hi = domain_hi.to(device=coeffs.device, dtype=coeffs.dtype)
    mono_lo, mono_hi = _monomial_interval_bounds_for_exponents(domain_lo, domain_hi, exponents)
    mono_lo = mono_lo[:, None, :]
    mono_hi = mono_hi[:, None, :]
    term_lo = torch.where(coeffs >= 0, coeffs * mono_lo, coeffs * mono_hi)
    term_hi = torch.where(coeffs >= 0, coeffs * mono_hi, coeffs * mono_lo)
    return _down(term_lo.sum(dim=-1)), _up(term_hi.sum(dim=-1))


def _subdivision_count(method: str, subdivisions: int = 2) -> int:
    if method == "interval":
        return 1
    if method == "split2":
        return 2
    if method == "subdivide":
        return max(2, int(subdivisions))
    if method.startswith("subdivide:"):
        return max(2, int(method.split(":", 1)[1]))
    raise ValueError(f"unknown range bound mode: {method}")


def _range_for_terms_split(
    coeffs: torch.Tensor,
    exponents: torch.Tensor,
    domain_lo: torch.Tensor,
    domain_hi: torch.Tensor,
    *,
    subdivisions: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    domain_lo = domain_lo.to(device=coeffs.device, dtype=coeffs.dtype)
    domain_hi = domain_hi.to(device=coeffs.device, dtype=coeffs.dtype)
    if domain_lo.ndim == 1:
        domain_lo = domain_lo.unsqueeze(0)
        domain_hi = domain_hi.unsqueeze(0)
    dim = int(domain_lo.shape[1])
    total_subboxes = int(subdivisions) ** dim
    if dim > 4 or total_subboxes > 16:
        raise ValueError("split range bounds are limited to small dimensions/subdivision counts")
    width = domain_hi - domain_lo
    out_lo: torch.Tensor | None = None
    out_hi: torch.Tensor | None = None
    for cell in product(range(int(subdivisions)), repeat=dim):
        cell_lo = torch.as_tensor(cell, dtype=coeffs.dtype, device=coeffs.device).view(1, dim) / float(subdivisions)
        cell_hi = (torch.as_tensor(cell, dtype=coeffs.dtype, device=coeffs.device).view(1, dim) + 1.0) / float(subdivisions)
        sub_lo = domain_lo + width * cell_lo
        sub_hi = domain_lo + width * cell_hi
        lo, hi = _range_for_terms(coeffs, exponents, sub_lo, sub_hi)
        out_lo = lo if out_lo is None else torch.minimum(out_lo, lo)
        out_hi = hi if out_hi is None else torch.maximum(out_hi, hi)
    if out_lo is None or out_hi is None:
        return _range_for_terms(coeffs, exponents, domain_lo, domain_hi)
    return _down(out_lo), _up(out_hi)


def _range_for_terms_mode(
    coeffs: torch.Tensor,
    exponents: torch.Tensor,
    domain_lo: torch.Tensor,
    domain_hi: torch.Tensor,
    *,
    method: str = "interval",
    subdivisions: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    count = _subdivision_count(method, subdivisions=subdivisions)
    if count == 1:
        return _range_for_terms(coeffs, exponents, domain_lo, domain_hi)
    return _range_for_terms_split(coeffs, exponents, domain_lo, domain_hi, subdivisions=count)


def _merge_coefficients_by_index(
    coeffs: torch.Tensor,
    merge_indices: torch.Tensor,
    unique_count: int,
) -> torch.Tensor:
    if unique_count == 0:
        return torch.zeros((*coeffs.shape[:-1], 0), dtype=coeffs.dtype, device=coeffs.device)
    merge_t = merge_indices.to(device=coeffs.device, dtype=torch.long)
    target = merge_t.view(*([1] * (coeffs.ndim - 1)), -1).expand_as(coeffs)
    out = torch.zeros((*coeffs.shape[:-1], int(unique_count)), dtype=coeffs.dtype, device=coeffs.device)
    out.scatter_add_(-1, target, coeffs)
    return out


@dataclass(frozen=True)
class BatchedMonomialBasis:
    """Dense total-degree monomial basis with precomputed scatter plans."""

    dim: int
    order: int
    exponents: torch.Tensor
    exponent_to_index: dict[tuple[int, ...], int]
    constant_index: int
    linear_indices: list[int]
    degree: torch.Tensor
    mul_left_indices: torch.Tensor
    mul_right_indices: torch.Tensor
    mul_out_indices: torch.Tensor
    trunc_left_indices: torch.Tensor
    trunc_right_indices: torch.Tensor
    trunc_exponents: torch.Tensor
    trunc_merge_indices: torch.Tensor
    trunc_unique_exponents: torch.Tensor

    @staticmethod
    def build(dim: int, order: int, device: torch.device | str | None = None) -> "BatchedMonomialBasis":
        if dim <= 0:
            raise ValueError("dim must be positive")
        if order < 0:
            raise ValueError("order must be nonnegative")
        device_t = _as_device(device)
        exps = _total_degree_exponents(int(dim), int(order))
        index = {exp: i for i, exp in enumerate(exps)}
        exponents_t = torch.as_tensor(exps, dtype=torch.long, device=device_t)
        degree_t = exponents_t.sum(dim=1)
        constant_index = index[(0,) * int(dim)]
        linear_indices: list[int] = []
        for var_index in range(int(dim)):
            exp = [0] * int(dim)
            exp[var_index] = 1
            if tuple(exp) in index:
                linear_indices.append(index[tuple(exp)])

        mul_left: list[int] = []
        mul_right: list[int] = []
        mul_out: list[int] = []
        trunc_left: list[int] = []
        trunc_right: list[int] = []
        trunc_exps: list[tuple[int, ...]] = []
        for left_index, left_exp in enumerate(exps):
            for right_index, right_exp in enumerate(exps):
                product_exp = tuple(a + b for a, b in zip(left_exp, right_exp))
                if sum(product_exp) <= int(order):
                    mul_left.append(left_index)
                    mul_right.append(right_index)
                    mul_out.append(index[product_exp])
                else:
                    trunc_left.append(left_index)
                    trunc_right.append(right_index)
                    trunc_exps.append(product_exp)

        trunc_unique: list[tuple[int, ...]] = []
        trunc_unique_index: dict[tuple[int, ...], int] = {}
        trunc_merge: list[int] = []
        for exp in trunc_exps:
            if exp not in trunc_unique_index:
                trunc_unique_index[exp] = len(trunc_unique)
                trunc_unique.append(exp)
            trunc_merge.append(trunc_unique_index[exp])

        return BatchedMonomialBasis(
            int(dim),
            int(order),
            exponents_t,
            index,
            constant_index,
            linear_indices,
            degree_t,
            torch.as_tensor(mul_left, dtype=torch.long, device=device_t),
            torch.as_tensor(mul_right, dtype=torch.long, device=device_t),
            torch.as_tensor(mul_out, dtype=torch.long, device=device_t),
            torch.as_tensor(trunc_left, dtype=torch.long, device=device_t),
            torch.as_tensor(trunc_right, dtype=torch.long, device=device_t),
            torch.as_tensor(trunc_exps, dtype=torch.long, device=device_t).reshape(-1, int(dim)),
            torch.as_tensor(trunc_merge, dtype=torch.long, device=device_t),
            torch.as_tensor(trunc_unique, dtype=torch.long, device=device_t).reshape(-1, int(dim)),
        )

    @property
    def device(self) -> torch.device:
        return self.exponents.device

    @property
    def num_terms(self) -> int:
        return int(self.exponents.shape[0])

    def to(self, device: torch.device | str) -> "BatchedMonomialBasis":
        device_t = torch.device(device)
        if device_t == self.device:
            return self
        return BatchedMonomialBasis(
            self.dim,
            self.order,
            self.exponents.to(device_t),
            dict(self.exponent_to_index),
            self.constant_index,
            list(self.linear_indices),
            self.degree.to(device_t),
            self.mul_left_indices.to(device_t),
            self.mul_right_indices.to(device_t),
            self.mul_out_indices.to(device_t),
            self.trunc_left_indices.to(device_t),
            self.trunc_right_indices.to(device_t),
            self.trunc_exponents.to(device_t),
            self.trunc_merge_indices.to(device_t),
            self.trunc_unique_exponents.to(device_t),
        )

    def term_index(self, exponent_tuple: Sequence[int]) -> int:
        exp = tuple(int(v) for v in exponent_tuple)
        return self.exponent_to_index[exp]

    def multiplication_plan(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.mul_left_indices, self.mul_right_indices, self.mul_out_indices

    def eval_monomials(self, points: torch.Tensor) -> torch.Tensor:
        points_t = torch.as_tensor(points)
        exponents = self.exponents.to(device=points_t.device)
        if points_t.shape[-1] != self.dim:
            raise ValueError(f"point dimension {points_t.shape[-1]} != basis dim {self.dim}")
        if points_t.ndim == 2:
            values = points_t[:, None, :].pow(exponents[None, :, :])
        elif points_t.ndim == 3:
            values = points_t[:, :, None, :].pow(exponents[None, None, :, :])
        else:
            raise ValueError("points must have shape [batch, dim] or [batch, n_points, dim]")
        return values.prod(dim=-1)

    def interval_monomial_bounds(
        self,
        domain_lo: torch.Tensor,
        domain_hi: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _monomial_interval_bounds_for_exponents(
            torch.as_tensor(domain_lo),
            torch.as_tensor(domain_hi),
            self.exponents,
        )


@dataclass(frozen=True)
class BatchedPolynomial:
    """Batched dense polynomial coefficients with shape ``[batch, out_dim, terms]``."""

    coeffs: torch.Tensor
    basis: BatchedMonomialBasis

    def __post_init__(self) -> None:
        if self.coeffs.ndim != 3:
            raise ValueError("coeffs must have shape [batch, out_dim, n_terms]")
        if self.coeffs.shape[-1] != self.basis.num_terms:
            raise ValueError(f"coeff term dimension {self.coeffs.shape[-1]} != {self.basis.num_terms}")
        if self.basis.device != self.coeffs.device:
            object.__setattr__(self, "basis", self.basis.to(self.coeffs.device))

    @staticmethod
    def zeros(
        batch: int,
        out_dim: int,
        basis: BatchedMonomialBasis,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "BatchedPolynomial":
        device_t = basis.device if device is None else torch.device(device)
        dtype_t = _as_dtype(dtype)
        basis_t = basis.to(device_t)
        coeffs = torch.zeros((int(batch), int(out_dim), basis_t.num_terms), dtype=dtype_t, device=device_t)
        return BatchedPolynomial(coeffs, basis_t)

    @staticmethod
    def constants(values: Any, basis: BatchedMonomialBasis) -> "BatchedPolynomial":
        if isinstance(values, torch.Tensor):
            values_t = values.clone()
            if not torch.is_floating_point(values_t):
                values_t = values_t.to(dtype=torch.float64)
        else:
            values_t = torch.as_tensor(values, dtype=torch.float64)
        if values_t.ndim == 0:
            values_t = values_t.reshape(1, 1)
        elif values_t.ndim == 1:
            values_t = values_t.unsqueeze(0)
        elif values_t.ndim != 2:
            raise ValueError("constant values must be scalar, [out_dim], or [batch, out_dim]")
        basis_t = basis.to(values_t.device)
        out = torch.zeros((*values_t.shape, basis_t.num_terms), dtype=values_t.dtype, device=values_t.device)
        out[..., basis_t.constant_index] = values_t
        return BatchedPolynomial(out, basis_t)

    @staticmethod
    def variables(
        batch: int,
        dim: int,
        basis: BatchedMonomialBasis,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "BatchedPolynomial":
        if int(dim) > basis.dim:
            raise ValueError("variable dim cannot exceed basis dim")
        if basis.order < 1:
            raise ValueError("basis order must be at least 1 to represent variables")
        out = BatchedPolynomial.zeros(batch, dim, basis, device=device, dtype=dtype)
        coeffs = out.coeffs.clone()
        for var_index in range(int(dim)):
            exp = tuple(1 if i == var_index else 0 for i in range(basis.dim))
            coeffs[:, var_index, out.basis.term_index(exp)] = 1.0
        return BatchedPolynomial(coeffs, out.basis)

    @property
    def batch(self) -> int:
        return int(self.coeffs.shape[0])

    @property
    def out_dim(self) -> int:
        return int(self.coeffs.shape[1])

    def clone(self) -> "BatchedPolynomial":
        return BatchedPolynomial(self.coeffs.clone(), self.basis)

    def to(self, device: torch.device | str) -> "BatchedPolynomial":
        device_t = torch.device(device)
        return BatchedPolynomial(self.coeffs.to(device_t), self.basis.to(device_t))

    def _check_basis(self, other: "BatchedPolynomial") -> None:
        if self.basis.dim != other.basis.dim or self.basis.order != other.basis.order:
            raise ValueError("basis mismatch")
        if self.basis.exponent_to_index != other.basis.exponent_to_index:
            raise ValueError("basis ordering mismatch")

    def add(self, other: "BatchedPolynomial") -> "BatchedPolynomial":
        self._check_basis(other)
        return BatchedPolynomial(self.coeffs + other.coeffs.to(device=self.coeffs.device, dtype=self.coeffs.dtype), self.basis)

    def sub(self, other: "BatchedPolynomial") -> "BatchedPolynomial":
        self._check_basis(other)
        return BatchedPolynomial(self.coeffs - other.coeffs.to(device=self.coeffs.device, dtype=self.coeffs.dtype), self.basis)

    def scale(self, scalar: Any) -> "BatchedPolynomial":
        s = torch.as_tensor(scalar, dtype=self.coeffs.dtype, device=self.coeffs.device)
        while s.ndim < self.coeffs.ndim:
            s = s.unsqueeze(-1)
        return BatchedPolynomial(self.coeffs * s, self.basis)

    def affine_map(self, W: torch.Tensor, b: torch.Tensor | None = None) -> "BatchedPolynomial":
        W_t = torch.as_tensor(W, dtype=self.coeffs.dtype, device=self.coeffs.device)
        if W_t.ndim == 2:
            out = torch.einsum("no,bot->bnt", W_t, self.coeffs)
        elif W_t.ndim == 3:
            out = torch.einsum("bno,bot->bnt", W_t, self.coeffs)
        else:
            raise ValueError("W must have shape [out_new, out_dim] or [batch, out_new, out_dim]")
        if b is not None:
            b_t = torch.as_tensor(b, dtype=self.coeffs.dtype, device=self.coeffs.device)
            if b_t.ndim == 1:
                out[:, :, self.basis.constant_index] += b_t.view(1, -1)
            elif b_t.ndim == 2:
                out[:, :, self.basis.constant_index] += b_t
            else:
                raise ValueError("b must have shape [out_new] or [batch, out_new]")
        return BatchedPolynomial(out, self.basis)

    def mul_trunc(
        self,
        other: "BatchedPolynomial",
        *,
        return_truncation_bound: bool = False,
        domain_lo: torch.Tensor | None = None,
        domain_hi: torch.Tensor | None = None,
        dropped_merge_mode: str = "termwise",
        range_bound_mode: str = "interval",
        profile: DenseTMProfiler | None = None,
    ) -> "BatchedPolynomial" | tuple["BatchedPolynomial", torch.Tensor, torch.Tensor]:
        if dropped_merge_mode not in {"termwise", "merged"}:
            raise ValueError("dropped_merge_mode must be 'termwise' or 'merged'")
        self._check_basis(other)
        other_coeffs = other.coeffs.to(device=self.coeffs.device, dtype=self.coeffs.dtype)
        basis = self.basis
        with _profile_measure(profile, "mul_trunc"):
            left = basis.mul_left_indices
            right = basis.mul_right_indices
            products = self.coeffs.index_select(-1, left) * other_coeffs.index_select(-1, right)
            out = torch.zeros((*products.shape[:-1], basis.num_terms), dtype=self.coeffs.dtype, device=self.coeffs.device)
            target = basis.mul_out_indices.view(*([1] * (products.ndim - 1)), -1).expand_as(products)
            out.scatter_add_(-1, target, products)
            poly = BatchedPolynomial(out, basis)
        if not return_truncation_bound:
            return poly
        if domain_lo is None or domain_hi is None:
            raise ValueError("domain_lo/domain_hi are required for truncation bounds")
        if basis.trunc_left_indices.numel() == 0:
            zeros = torch.zeros(products.shape[:-1], dtype=self.coeffs.dtype, device=self.coeffs.device)
            return poly, zeros, zeros
        with _profile_measure(profile, "dropped_range_bound"):
            dropped = self.coeffs.index_select(-1, basis.trunc_left_indices) * other_coeffs.index_select(
                -1, basis.trunc_right_indices
            )
            if dropped_merge_mode == "merged":
                dropped = _merge_coefficients_by_index(
                    dropped,
                    basis.trunc_merge_indices,
                    int(basis.trunc_unique_exponents.shape[0]),
                )
                trunc_exponents = basis.trunc_unique_exponents
            else:
                trunc_exponents = basis.trunc_exponents
            trunc_lo, trunc_hi = _range_for_terms_mode(
                dropped,
                trunc_exponents,
                domain_lo,
                domain_hi,
                method=range_bound_mode,
            )
        return poly, trunc_lo, trunc_hi

    def square_trunc(self, **kwargs: Any) -> "BatchedPolynomial" | tuple["BatchedPolynomial", torch.Tensor, torch.Tensor]:
        return self.mul_trunc(self, **kwargs)

    def range_bound(
        self,
        domain_lo: torch.Tensor,
        domain_hi: torch.Tensor,
        method: str = "interval",
        *,
        profile: DenseTMProfiler | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with _profile_measure(profile, "range_bound"):
            return _range_for_terms_mode(self.coeffs, self.basis.exponents, domain_lo, domain_hi, method=method)

    def evaluate(self, points: torch.Tensor) -> torch.Tensor:
        points_t = torch.as_tensor(points, dtype=self.coeffs.dtype, device=self.coeffs.device)
        monomials = self.basis.eval_monomials(points_t)
        if monomials.ndim == 2:
            return torch.einsum("bt,bot->bo", monomials, self.coeffs)
        return torch.einsum("bnt,bot->bno", monomials, self.coeffs)

    def component(self, index: int) -> "BatchedPolynomial":
        idx = int(index)
        return BatchedPolynomial(self.coeffs[:, idx : idx + 1, :], self.basis)

    @staticmethod
    def concat(polys: Sequence["BatchedPolynomial"]) -> "BatchedPolynomial":
        if not polys:
            raise ValueError("concat requires at least one polynomial")
        basis = polys[0].basis
        for poly in polys[1:]:
            polys[0]._check_basis(poly)
        return BatchedPolynomial(torch.cat([poly.coeffs for poly in polys], dim=1), basis)

    __add__ = add
    __sub__ = sub
    __mul__ = mul_trunc


@dataclass(frozen=True)
class BatchedTaylorModel:
    """Batched dense Taylor models with interval remainders."""

    poly: BatchedPolynomial
    rem_lo: torch.Tensor
    rem_hi: torch.Tensor
    domain_lo: torch.Tensor
    domain_hi: torch.Tensor

    def __post_init__(self) -> None:
        batch, out_dim, _terms = self.poly.coeffs.shape
        if self.rem_lo.shape != (batch, out_dim) or self.rem_hi.shape != (batch, out_dim):
            raise ValueError("remainder shape must be [batch, out_dim]")
        if self.domain_lo.shape != self.domain_hi.shape or self.domain_lo.shape != (batch, self.poly.basis.dim):
            raise ValueError("domain bounds must have shape [batch, dim]")

    @staticmethod
    def variables_from_domain(
        domain_lo: torch.Tensor,
        domain_hi: torch.Tensor,
        basis: BatchedMonomialBasis,
    ) -> "BatchedTaylorModel":
        lo = torch.as_tensor(domain_lo)
        hi = torch.as_tensor(domain_hi, dtype=lo.dtype, device=lo.device)
        if lo.ndim != 2:
            raise ValueError("domain bounds must have shape [batch, dim]")
        poly = BatchedPolynomial.variables(lo.shape[0], lo.shape[1], basis, device=lo.device, dtype=lo.dtype)
        rem = torch.zeros((lo.shape[0], lo.shape[1]), dtype=lo.dtype, device=lo.device)
        return BatchedTaylorModel(poly, rem, rem.clone(), lo, hi)

    @staticmethod
    def constant_interval(
        value_lo: Any,
        value_hi: Any,
        basis: BatchedMonomialBasis,
        domain_lo: torch.Tensor,
        domain_hi: torch.Tensor,
    ) -> "BatchedTaylorModel":
        dom_lo = torch.as_tensor(domain_lo)
        dom_hi = torch.as_tensor(domain_hi, dtype=dom_lo.dtype, device=dom_lo.device)
        lo = torch.as_tensor(value_lo, dtype=dom_lo.dtype, device=dom_lo.device)
        hi = torch.as_tensor(value_hi, dtype=dom_lo.dtype, device=dom_lo.device)
        if lo.ndim == 0:
            lo = lo.reshape(1, 1)
            hi = hi.reshape(1, 1)
        elif lo.ndim == 1:
            lo = lo.unsqueeze(-1)
            hi = hi.unsqueeze(-1)
        lo, hi = torch.broadcast_tensors(lo, hi)
        if lo.shape[0] == 1 and dom_lo.shape[0] != 1:
            lo = lo.expand(dom_lo.shape[0], -1)
            hi = hi.expand(dom_lo.shape[0], -1)
        if lo.ndim != 2 or lo.shape[0] != dom_lo.shape[0]:
            raise ValueError("constant interval values must broadcast to [batch, out_dim]")
        center = 0.5 * (lo + hi)
        poly = BatchedPolynomial.constants(center, basis.to(dom_lo.device))
        return BatchedTaylorModel(poly, _down(lo - center), _up(hi - center), dom_lo, dom_hi)

    def clone(self) -> "BatchedTaylorModel":
        return BatchedTaylorModel(
            self.poly.clone(),
            self.rem_lo.clone(),
            self.rem_hi.clone(),
            self.domain_lo.clone(),
            self.domain_hi.clone(),
        )

    def to(self, device: torch.device | str) -> "BatchedTaylorModel":
        device_t = torch.device(device)
        return BatchedTaylorModel(
            self.poly.to(device_t),
            self.rem_lo.to(device_t),
            self.rem_hi.to(device_t),
            self.domain_lo.to(device_t),
            self.domain_hi.to(device_t),
        )

    def _check_domain(self, other: "BatchedTaylorModel") -> None:
        self.poly._check_basis(other.poly)
        if self.domain_lo.shape != other.domain_lo.shape or not torch.allclose(self.domain_lo, other.domain_lo):
            raise ValueError("domain lower bounds mismatch")
        if self.domain_hi.shape != other.domain_hi.shape or not torch.allclose(self.domain_hi, other.domain_hi):
            raise ValueError("domain upper bounds mismatch")

    def add(self, other: "BatchedTaylorModel") -> "BatchedTaylorModel":
        self._check_domain(other)
        rem_lo, rem_hi = _interval_add(self.rem_lo, self.rem_hi, other.rem_lo, other.rem_hi)
        return BatchedTaylorModel(self.poly.add(other.poly), rem_lo, rem_hi, self.domain_lo, self.domain_hi)

    def sub(self, other: "BatchedTaylorModel") -> "BatchedTaylorModel":
        self._check_domain(other)
        rem_lo, rem_hi = _interval_sub(self.rem_lo, self.rem_hi, other.rem_lo, other.rem_hi)
        return BatchedTaylorModel(self.poly.sub(other.poly), rem_lo, rem_hi, self.domain_lo, self.domain_hi)

    def scale(self, scalar: Any) -> "BatchedTaylorModel":
        rem_lo, rem_hi = _interval_scale(self.rem_lo, self.rem_hi, scalar)
        return BatchedTaylorModel(self.poly.scale(scalar), rem_lo, rem_hi, self.domain_lo, self.domain_hi)

    def affine_map(self, W: torch.Tensor, b: torch.Tensor | None = None) -> "BatchedTaylorModel":
        poly = self.poly.affine_map(W, b)
        W_t = torch.as_tensor(W, dtype=self.rem_lo.dtype, device=self.rem_lo.device)
        center = 0.5 * (self.rem_lo + self.rem_hi)
        radius = 0.5 * (self.rem_hi - self.rem_lo)
        if W_t.ndim == 2:
            rem_center = torch.einsum("no,bo->bn", W_t, center)
            rem_radius = torch.einsum("no,bo->bn", torch.abs(W_t), radius)
        elif W_t.ndim == 3:
            rem_center = torch.einsum("bno,bo->bn", W_t, center)
            rem_radius = torch.einsum("bno,bo->bn", torch.abs(W_t), radius)
        else:
            raise ValueError("W must have shape [out_new, out_dim] or [batch, out_new, out_dim]")
        return BatchedTaylorModel(
            poly,
            _down(rem_center - rem_radius),
            _up(rem_center + rem_radius),
            self.domain_lo,
            self.domain_hi,
        )

    def mul_trunc(
        self,
        other: "BatchedTaylorModel",
        *,
        dropped_merge_mode: str = "termwise",
        range_bound_mode: str = "interval",
        profile: DenseTMProfiler | None = None,
    ) -> "BatchedTaylorModel":
        self._check_domain(other)
        poly, trunc_lo, trunc_hi = self.poly.mul_trunc(
            other.poly,
            return_truncation_bound=True,
            domain_lo=self.domain_lo,
            domain_hi=self.domain_hi,
            dropped_merge_mode=dropped_merge_mode,
            range_bound_mode=range_bound_mode,
            profile=profile,
        )
        p_lo, p_hi = self.poly.range_bound(self.domain_lo, self.domain_hi, method=range_bound_mode, profile=profile)
        q_lo, q_hi = other.poly.range_bound(self.domain_lo, self.domain_hi, method=range_bound_mode, profile=profile)
        with _profile_measure(profile, "tm_remainder_propagation"):
            p_j_lo, p_j_hi = _interval_mul(p_lo, p_hi, other.rem_lo, other.rem_hi)
            q_i_lo, q_i_hi = _interval_mul(q_lo, q_hi, self.rem_lo, self.rem_hi)
            i_j_lo, i_j_hi = _interval_mul(self.rem_lo, self.rem_hi, other.rem_lo, other.rem_hi)
            rem_lo, rem_hi = _interval_add(trunc_lo, trunc_hi, p_j_lo, p_j_hi)
            rem_lo, rem_hi = _interval_add(rem_lo, rem_hi, q_i_lo, q_i_hi)
            rem_lo, rem_hi = _interval_add(rem_lo, rem_hi, i_j_lo, i_j_hi)
        return BatchedTaylorModel(poly, rem_lo, rem_hi, self.domain_lo, self.domain_hi)

    def range_bound(
        self,
        method: str = "interval",
        *,
        profile: DenseTMProfiler | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        poly_lo, poly_hi = self.poly.range_bound(self.domain_lo, self.domain_hi, method=method, profile=profile)
        return _interval_add(poly_lo, poly_hi, self.rem_lo, self.rem_hi)

    def recenter_rescale(self) -> "BatchedTaylorModel":
        return self

    def component(self, index: int) -> "BatchedTaylorModel":
        idx = int(index)
        return BatchedTaylorModel(
            self.poly.component(idx),
            self.rem_lo[:, idx : idx + 1],
            self.rem_hi[:, idx : idx + 1],
            self.domain_lo,
            self.domain_hi,
        )

    @staticmethod
    def concat(models: Sequence["BatchedTaylorModel"]) -> "BatchedTaylorModel":
        if not models:
            raise ValueError("concat requires at least one model")
        first = models[0]
        polys = [model.poly for model in models]
        for model in models[1:]:
            first._check_domain(model)
        return BatchedTaylorModel(
            BatchedPolynomial.concat(polys),
            torch.cat([model.rem_lo for model in models], dim=1),
            torch.cat([model.rem_hi for model in models], dim=1),
            first.domain_lo,
            first.domain_hi,
        )

    def vanderpol_rhs(
        self,
        *,
        dropped_merge_mode: str = "termwise",
        range_bound_mode: str = "interval",
        profile: DenseTMProfiler | None = None,
    ) -> "BatchedTaylorModel":
        if self.poly.out_dim != 2:
            raise ValueError("Van der Pol RHS requires out_dim=2")
        with _profile_measure(profile, "rhs_construction"):
            x = self.component(0)
            y = self.component(1)
            x_sq = x.mul_trunc(
                x,
                dropped_merge_mode=dropped_merge_mode,
                range_bound_mode=range_bound_mode,
                profile=profile,
            )
            x_sq_y = x_sq.mul_trunc(
                y,
                dropped_merge_mode=dropped_merge_mode,
                range_bound_mode=range_bound_mode,
                profile=profile,
            )
            return BatchedTaylorModel.concat([y, y.sub(x).sub(x_sq_y)])

    def controlled_rhs(self, control: "BatchedTaylorModel") -> "BatchedTaylorModel":
        if self.poly.out_dim != 2 or control.poly.out_dim != 1:
            raise ValueError("controlled RHS requires state out_dim=2 and control out_dim=1")
        self._check_domain(control)
        x = self.component(0)
        y = self.component(1)
        return BatchedTaylorModel.concat([y, control.sub(x).sub(y.scale(0.1))])

    def fixed_euler_tm_step_vdp(
        self,
        h: float,
        order: int | None = None,
        *,
        dropped_merge_mode: str = "termwise",
        range_bound_mode: str = "interval",
        profile: DenseTMProfiler | None = None,
    ) -> "BatchedTaylorModel":
        _ = order
        rhs = self.vanderpol_rhs(
            dropped_merge_mode=dropped_merge_mode,
            range_bound_mode=range_bound_mode,
            profile=profile,
        )
        return self.add(rhs.scale(float(h)))

    def fixed_euler_tm_step_controlled(
        self,
        control: "BatchedTaylorModel",
        h: float,
        order: int | None = None,
    ) -> "BatchedTaylorModel":
        _ = order
        return self.add(self.controlled_rhs(control).scale(float(h)))

    def fixed_picard_step_vdp(self, h: float, order: int | None = None, **kwargs: Any) -> "BatchedTaylorModel":
        return self.fixed_euler_tm_step_vdp(h, order=order, **kwargs)

    def one_fixed_tm_step_vdp(self, h: float, order: int | None = None, **kwargs: Any) -> "BatchedTaylorModel":
        return self.fixed_euler_tm_step_vdp(h, order=order, **kwargs)

    __add__ = add
    __sub__ = sub
    __mul__ = mul_trunc


__all__ = [
    "BatchedMonomialBasis",
    "BatchedPolynomial",
    "BatchedTaylorModel",
    "DenseTMProfiler",
]
