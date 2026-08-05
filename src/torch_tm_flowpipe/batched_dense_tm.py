"""Canonical dense batched Taylor-model tensors and validated Picard core.

The sparse :mod:`torch_tm_flowpipe.polynomial` and
:mod:`torch_tm_flowpipe.taylor_model` implementations remain the semantic
reference.  This module implements the same complete-total-degree arithmetic
with a batch-first tensor layout and exposes explicit boundary conversions for
parity tests and the ``hybrid_dense_core`` flowpipe lane.  Dense Picard and
remainder validation never convert through a Python polynomial dictionary.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import hashlib
from itertools import product
import math
import time
from typing import Any, Callable, Mapping, Sequence

import torch


REMAINDER_LEDGER_CATEGORIES = (
    "initial_remainder",
    "polynomial_truncation",
    "cutoff",
    "integration_overflow",
    "composition_overflow",
    "poly_times_remainder",
    "remainder_times_poly",
    "remainder_times_remainder",
    "picard_residual",
    "roundoff_safeguard",
    "reset_or_reconditioning",
)

_INTEGRATION_PLAN_CACHE: dict[
    tuple[int, int, str, int, torch.dtype],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
] = {}
_MULTIPLICATION_DEGREE_PLAN_CACHE: dict[
    tuple[int, int, str, int],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
] = {}


@dataclass(frozen=True)
class DenseTMContract:
    """Machine-checkable coordinate and tensor contract for a dense solve."""

    batch_dim: int
    state_dim: int
    n_vars: int
    tau_index: int
    uncertainty_indices: tuple[int, ...]
    order: int
    domain_lo: torch.Tensor
    domain_hi: torch.Tensor
    local_time_semantics: str = "physical_[0,h]"
    time_scale: str = "integration_only"

    def __post_init__(self) -> None:
        if self.batch_dim <= 0 or self.state_dim <= 0 or self.n_vars <= 0:
            raise ValueError("batch_dim, state_dim, and n_vars must be positive")
        if self.order < 0:
            raise ValueError("order must be nonnegative")
        if not 0 <= self.tau_index < self.n_vars:
            raise ValueError("tau_index is outside the polynomial variables")
        if self.tau_index in self.uncertainty_indices:
            raise ValueError("tau_index cannot also be an uncertainty index")
        if len(set(self.uncertainty_indices)) != len(self.uncertainty_indices):
            raise ValueError("uncertainty indices must be unique")
        if any(index < 0 or index >= self.n_vars for index in self.uncertainty_indices):
            raise ValueError("uncertainty index is outside the polynomial variables")
        expected = (self.batch_dim, self.n_vars)
        if self.domain_lo.shape != expected or self.domain_hi.shape != expected:
            raise ValueError(f"domain tensors must have shape {expected}")
        if self.domain_lo.dtype != self.domain_hi.dtype or self.domain_lo.device != self.domain_hi.device:
            raise ValueError("domain bounds must share dtype and device")
        if not torch.is_floating_point(self.domain_lo):
            raise TypeError("dense Taylor-model domains must use a floating dtype")
        if not bool(torch.all(self.domain_lo <= self.domain_hi)):
            raise ValueError("domain lower bounds must not exceed upper bounds")
        if self.local_time_semantics != "physical_[0,h]" or self.time_scale != "integration_only":
            raise ValueError("the canonical dense lane uses physical tau and integration-only time scaling")

    @property
    def basis_size(self) -> int:
        return math.comb(self.n_vars + self.order, self.order)

    def as_dict(self) -> dict[str, Any]:
        return {
            "batch_dim": self.batch_dim,
            "state_dim": self.state_dim,
            "n_vars": self.n_vars,
            "tau_index": self.tau_index,
            "uncertainty_indices": list(self.uncertainty_indices),
            "order": self.order,
            "basis_size": self.basis_size,
            "domain_lo": self.domain_lo.detach().cpu().tolist(),
            "domain_hi": self.domain_hi.detach().cpu().tolist(),
            "dtype": str(self.domain_lo.dtype),
            "device": str(self.domain_lo.device),
            "local_time_semantics": self.local_time_semantics,
            "time_scale": self.time_scale,
        }


@dataclass
class DenseExecutionCounters:
    """Auditable boundary/fallback counters for a dense flowpipe lane."""

    sparse_to_dense_conversions: int = 0
    dense_to_sparse_conversions: int = 0
    segment_boundary_conversions: int = 0
    inner_loop_conversions: int = 0
    device_transfer_count: int = 0
    sparse_fallback_count: int = 0
    boundary_scalar_loop_count: int = 0
    inner_loop_scalar_count: int = 0
    range_subdivision_invocations: int = 0
    range_leaf_evaluations: int = 0

    def as_dict(self) -> dict[str, int]:
        return {name: int(getattr(self, name)) for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class DenseRemainderLedger:
    """Named interval contributions whose sum is a Taylor-model remainder."""

    entries: Mapping[str, tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        unknown = set(self.entries) - set(REMAINDER_LEDGER_CATEGORIES)
        if unknown:
            raise ValueError(f"unknown remainder ledger categories: {sorted(unknown)}")
        for name, (lo, hi) in self.entries.items():
            if lo.shape != hi.shape:
                raise ValueError(f"ledger shape mismatch for {name}")
            if not bool(torch.all(lo <= hi)):
                raise ValueError(f"invalid interval contribution for {name}")

    @staticmethod
    def empty() -> "DenseRemainderLedger":
        return DenseRemainderLedger({})

    def add(
        self,
        category: str,
        lo: torch.Tensor,
        hi: torch.Tensor,
    ) -> "DenseRemainderLedger":
        if category not in REMAINDER_LEDGER_CATEGORIES:
            raise ValueError(f"unknown remainder ledger category: {category}")
        entries = dict(self.entries)
        lo_t = lo.clone()
        hi_t = hi.clone()
        if category in entries:
            lo_t, hi_t = _interval_add(entries[category][0], entries[category][1], lo_t, hi_t)
        entries[category] = (lo_t, hi_t)
        return DenseRemainderLedger(entries)

    def merge(self, other: "DenseRemainderLedger") -> "DenseRemainderLedger":
        out = self
        for category, (lo, hi) in other.entries.items():
            out = out.add(category, lo, hi)
        return out

    def scale(self, scalar: Any) -> "DenseRemainderLedger":
        out = DenseRemainderLedger.empty()
        for category, (lo, hi) in self.entries.items():
            scaled_lo, scaled_hi = _interval_scale(lo, hi, scalar)
            out = out.add(category, scaled_lo, scaled_hi)
        return out

    def negate(self) -> "DenseRemainderLedger":
        return self.scale(-1.0)

    def total(self, like: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        lo = torch.zeros_like(like)
        hi = torch.zeros_like(like)
        for entry_lo, entry_hi in self.entries.values():
            lo, hi = _interval_add(lo, hi, entry_lo, entry_hi)
        return lo, hi

    def widths(self) -> dict[str, list[list[float]]]:
        return {
            name: (hi - lo).detach().cpu().tolist()
            for name, (lo, hi) in self.entries.items()
        }


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
    lo_sum = term_lo.sum(dim=-1)
    hi_sum = term_hi.sum(dim=-1)
    # ``scatter_add`` and the final reduction do not necessarily use the same
    # order as sparse dictionary aggregation (and CUDA is free to use another
    # order again).  Reserve a standard gamma_n absolute-error envelope so the
    # dense interval contains any such finite float64 reduction ordering.
    operation_count = max(1, 2 * int(coeffs.shape[-1]) + 1)
    eps = torch.finfo(coeffs.dtype).eps
    gamma_n = (operation_count * eps) / max(1.0 - operation_count * eps, eps)
    magnitude = torch.maximum(torch.abs(term_lo), torch.abs(term_hi)).sum(dim=-1)
    roundoff = magnitude * gamma_n + torch.finfo(coeffs.dtype).tiny
    return _down(lo_sum - roundoff), _up(hi_sum + roundoff)


@dataclass(frozen=True)
class DenseRangePolicy:
    """Explicit polynomial range semantics for the dense core."""

    method: str = "natural"
    max_depth: int = 0
    max_leaves: int = 64
    split_vars: tuple[int, ...] = (0, 1)
    trigger: str = "always"
    named_contexts: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.method not in {"natural", "subdivision", "adaptive_subdivision"}:
            raise ValueError("dense range method must be natural, subdivision, or adaptive_subdivision")
        if self.max_depth < 0:
            raise ValueError("dense range max_depth must be nonnegative")
        if self.max_leaves <= 0 or self.max_leaves > 64:
            raise ValueError("dense range max_leaves must lie in [1, 64]")
        if len(set(self.split_vars)) != len(self.split_vars):
            raise ValueError("dense range split_vars must be unique")
        if self.trigger not in {"always", "on_validation_failure", "proactive_depth1_on_named_contexts"}:
            raise ValueError("invalid dense range trigger")

    def applies_to(self, context: str) -> bool:
        return self.method != "natural" and (not self.named_contexts or context in self.named_contexts)


@dataclass(frozen=True)
class DenseSubdivisionCover:
    """Flat leaf tensor with explicit original-batch ownership."""

    lo: torch.Tensor
    hi: torch.Tensor
    owner: torch.Tensor
    requested_depth: int
    leaf_counts: tuple[int, ...]
    split_variables: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class DensePolynomialRangeResult:
    natural_lo: torch.Tensor
    natural_hi: torch.Tensor
    subdivision_lo: torch.Tensor
    subdivision_hi: torch.Tensor
    selected_lo: torch.Tensor
    selected_hi: torch.Tensor
    selected_method: str
    cover: DenseSubdivisionCover
    coverage_report: Mapping[str, Any]
    wall_s: float


def _subdivision_influence_scores(
    coeffs: torch.Tensor,
    exponents: torch.Tensor,
    domain_lo: torch.Tensor,
    domain_hi: torch.Tensor,
) -> torch.Tensor:
    """Deterministic width-times-derivative-magnitude split heuristic."""
    max_abs = torch.maximum(torch.abs(domain_lo), torch.abs(domain_hi))
    widths = domain_hi - domain_lo
    exponents_t = exponents.to(device=coeffs.device, dtype=torch.long)
    scores: list[torch.Tensor] = []
    abs_coeff = torch.abs(coeffs).sum(dim=1)
    for var_index in range(exponents_t.shape[1]):
        powers = exponents_t[:, var_index]
        derivative = torch.ones((coeffs.shape[0], exponents_t.shape[0]), dtype=coeffs.dtype, device=coeffs.device)
        for other_index in range(exponents_t.shape[1]):
            other_powers = exponents_t[:, other_index]
            if other_index == var_index:
                other_powers = torch.clamp(other_powers - 1, min=0)
            derivative = derivative * max_abs[:, other_index : other_index + 1].pow(other_powers.view(1, -1))
        derivative = derivative * powers.to(dtype=coeffs.dtype).view(1, -1)
        score = widths[:, var_index] * torch.sum(abs_coeff * derivative, dim=-1)
        scores.append(score)
    return torch.stack(scores, dim=1) if scores else torch.empty((coeffs.shape[0], 0), device=coeffs.device)


def _split_leaf_pair(lo: torch.Tensor, hi: torch.Tensor, var_index: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    midpoint = lo[var_index] + (hi[var_index] - lo[var_index]) * 0.5
    if not bool(midpoint > lo[var_index] and midpoint < hi[var_index]):
        return [(lo, hi)]
    left_hi = hi.clone()
    left_hi[var_index] = midpoint
    right_lo = lo.clone()
    right_lo[var_index] = midpoint
    if not bool(left_hi[var_index] == right_lo[var_index]):
        raise RuntimeError("subdivision children do not share an exact boundary")
    return [(lo, left_hi), (right_lo, hi)]


def build_dense_subdivision_cover(
    coeffs: torch.Tensor,
    exponents: torch.Tensor,
    domain_lo: torch.Tensor,
    domain_hi: torch.Tensor,
    *,
    depth: int,
    max_leaves: int = 64,
    split_vars: Sequence[int] = (0, 1),
) -> DenseSubdivisionCover:
    """Build the pre-registered 4/8/16/... leaf hierarchical box cover."""
    if coeffs.ndim != 3 or domain_lo.ndim != 2 or domain_hi.shape != domain_lo.shape:
        raise ValueError("subdivision expects coeffs [batch,output,term] and domains [batch,var]")
    if coeffs.shape[0] != domain_lo.shape[0] or exponents.shape != (coeffs.shape[-1], domain_lo.shape[1]):
        raise ValueError("subdivision coefficient/exponent/domain shape mismatch")
    if depth < 0:
        raise ValueError("subdivision depth must be nonnegative")
    if max_leaves <= 0 or max_leaves > 64:
        raise ValueError("max_leaves must lie in [1, 64]")
    if not bool(torch.all(torch.isfinite(coeffs)) and torch.all(torch.isfinite(domain_lo)) and torch.all(torch.isfinite(domain_hi))):
        raise FloatingPointError("subdivision inputs must be finite")
    if not bool(torch.all(domain_lo <= domain_hi)):
        raise ValueError("subdivision domain lower bounds must not exceed upper bounds")
    selected = tuple(int(index) for index in split_vars)
    if len(set(selected)) != len(selected) or any(index < 0 or index >= domain_lo.shape[1] for index in selected):
        raise ValueError("subdivision split_vars are invalid")
    scores = _subdivision_influence_scores(coeffs, exponents, domain_lo, domain_hi)
    all_lo: list[torch.Tensor] = []
    all_hi: list[torch.Tensor] = []
    owners: list[int] = []
    leaf_counts: list[int] = []
    histories: list[tuple[int, ...]] = []
    for batch_index in range(coeffs.shape[0]):
        leaves = [(domain_lo[batch_index].clone(), domain_hi[batch_index].clone())]
        history: list[int] = []
        if depth >= 1:
            for var_index in selected:
                proposed: list[tuple[torch.Tensor, torch.Tensor]] = []
                for leaf_lo, leaf_hi in leaves:
                    proposed.extend(_split_leaf_pair(leaf_lo, leaf_hi, var_index))
                if len(proposed) > max_leaves:
                    raise ValueError("max_leaves exceeded while building depth-1 subdivision cover")
                if len(proposed) > len(leaves):
                    history.append(var_index)
                leaves = proposed
        for _level in range(2, depth + 1):
            candidates = [index for index in selected if bool(domain_hi[batch_index, index] > domain_lo[batch_index, index])]
            if not candidates:
                break
            ranked = sorted(candidates, key=lambda index: (-float(scores[batch_index, index].detach().cpu()), index))
            var_index = ranked[0]
            proposed = []
            for leaf_lo, leaf_hi in leaves:
                proposed.extend(_split_leaf_pair(leaf_lo, leaf_hi, var_index))
            if len(proposed) > max_leaves:
                raise ValueError("max_leaves exceeded while building subdivision cover")
            if len(proposed) == len(leaves):
                break
            history.append(var_index)
            leaves = proposed
        leaf_counts.append(len(leaves))
        histories.append(tuple(history))
        for leaf_lo, leaf_hi in leaves:
            all_lo.append(leaf_lo)
            all_hi.append(leaf_hi)
            owners.append(batch_index)
    return DenseSubdivisionCover(
        torch.stack(all_lo, dim=0),
        torch.stack(all_hi, dim=0),
        torch.as_tensor(owners, dtype=torch.long, device=domain_lo.device),
        int(depth),
        tuple(leaf_counts),
        tuple(histories),
    )


def validate_dense_subdivision_cover(
    cover: DenseSubdivisionCover,
    domain_lo: torch.Tensor,
    domain_hi: torch.Tensor,
) -> dict[str, Any]:
    """Independently validate owner-local box coverage and interior disjointness."""
    reasons: list[str] = []
    if cover.lo.shape != cover.hi.shape or cover.lo.ndim != 2 or cover.owner.shape != (cover.lo.shape[0],):
        return {"valid": False, "reasons": ["cover shape mismatch"], "leaf_count": int(cover.lo.shape[0])}
    if not bool(torch.all(torch.isfinite(cover.lo)) and torch.all(torch.isfinite(cover.hi))):
        reasons.append("nonfinite leaf")
    if not bool(torch.all(cover.lo <= cover.hi)):
        reasons.append("invalid leaf interval")
    lo_cpu = domain_lo.detach().cpu()
    hi_cpu = domain_hi.detach().cpu()
    leaf_lo = cover.lo.detach().cpu()
    leaf_hi = cover.hi.detach().cpu()
    owner = cover.owner.detach().cpu()
    if owner.numel() and (int(torch.min(owner)) < 0 or int(torch.max(owner)) >= domain_lo.shape[0]):
        reasons.append("corrupted leaf ownership")
    checked_cells = 0
    for batch_index in range(domain_lo.shape[0]):
        indices = torch.nonzero(owner == batch_index, as_tuple=False).reshape(-1)
        if indices.numel() == 0:
            reasons.append(f"owner {batch_index} has no leaves")
            continue
        boxes_lo = leaf_lo.index_select(0, indices)
        boxes_hi = leaf_hi.index_select(0, indices)
        if bool(torch.any(boxes_lo < lo_cpu[batch_index])) or bool(torch.any(boxes_hi > hi_cpu[batch_index])):
            reasons.append(f"owner {batch_index} leaf exceeds parent")
        signatures = {
            tuple(float(value).hex() for value in torch.cat([lo_row, hi_row]).tolist())
            for lo_row, hi_row in zip(boxes_lo, boxes_hi)
        }
        if len(signatures) != int(indices.numel()):
            reasons.append(f"owner {batch_index} has duplicate leaves")
        axes: list[list[tuple[float, float]]] = []
        for var_index in range(domain_lo.shape[1]):
            endpoints = sorted(
                {
                    float(lo_cpu[batch_index, var_index]),
                    float(hi_cpu[batch_index, var_index]),
                    *[float(value) for value in boxes_lo[:, var_index]],
                    *[float(value) for value in boxes_hi[:, var_index]],
                }
            )
            cells = [(left, right) for left, right in zip(endpoints[:-1], endpoints[1:]) if left < right]
            axes.append(cells or [(endpoints[0], endpoints[0])])
        for cell in product(*axes):
            point = torch.tensor(
                [left if left == right else left + (right - left) * 0.5 for left, right in cell],
                dtype=boxes_lo.dtype,
            )
            memberships = torch.all((point >= boxes_lo) & (point <= boxes_hi), dim=1)
            checked_cells += 1
            if int(torch.count_nonzero(memberships)) != 1:
                reasons.append(f"owner {batch_index} gap or interior overlap")
                break
    return {
        "valid": not reasons,
        "reasons": sorted(set(reasons)),
        "leaf_count": int(cover.lo.shape[0]),
        "leaf_counts": list(cover.leaf_counts),
        "checked_cells": checked_cells,
        "split_variables": [list(items) for items in cover.split_variables],
    }


def _range_for_terms_with_policy(
    coeffs: torch.Tensor,
    exponents: torch.Tensor,
    domain_lo: torch.Tensor,
    domain_hi: torch.Tensor,
    *,
    policy: DenseRangePolicy,
    context: str,
    trace: list[dict[str, Any]] | None = None,
) -> DensePolynomialRangeResult:
    started = time.perf_counter()
    natural_lo, natural_hi = _range_for_terms(coeffs, exponents, domain_lo, domain_hi)
    if not policy.applies_to(context) or policy.max_depth == 0:
        owner = torch.arange(coeffs.shape[0], dtype=torch.long, device=coeffs.device)
        cover = DenseSubdivisionCover(domain_lo.clone(), domain_hi.clone(), owner, 0, tuple(1 for _ in range(coeffs.shape[0])), tuple(() for _ in range(coeffs.shape[0])))
        report = validate_dense_subdivision_cover(cover, domain_lo, domain_hi)
        result = DensePolynomialRangeResult(natural_lo, natural_hi, natural_lo, natural_hi, natural_lo, natural_hi, "natural", cover, report, time.perf_counter() - started)
    else:
        cover = build_dense_subdivision_cover(
            coeffs,
            exponents,
            domain_lo,
            domain_hi,
            depth=policy.max_depth,
            max_leaves=policy.max_leaves,
            split_vars=policy.split_vars,
        )
        report = validate_dense_subdivision_cover(cover, domain_lo, domain_hi)
        if not report["valid"]:
            raise RuntimeError(f"invalid subdivision cover: {report['reasons']}")
        leaf_coeffs = coeffs.index_select(0, cover.owner)
        leaf_lo, leaf_hi = _range_for_terms(leaf_coeffs, exponents, cover.lo, cover.hi)
        if not bool(torch.all(torch.isfinite(leaf_lo)) and torch.all(torch.isfinite(leaf_hi))):
            raise FloatingPointError("non-finite subdivision leaf range")
        hull_lo = torch.full_like(natural_lo, torch.inf)
        hull_hi = torch.full_like(natural_hi, -torch.inf)
        hull_index = cover.owner.view(-1, 1).expand(-1, coeffs.shape[1])
        hull_lo.scatter_reduce_(0, hull_index, leaf_lo, reduce="amin", include_self=True)
        hull_hi.scatter_reduce_(0, hull_index, leaf_hi, reduce="amax", include_self=True)
        hull_lo, hull_hi = _down(hull_lo), _up(hull_hi)
        use_subdivision = (hull_hi - hull_lo) <= (natural_hi - natural_lo)
        selected_lo = torch.where(use_subdivision, hull_lo, natural_lo)
        selected_hi = torch.where(use_subdivision, hull_hi, natural_hi)
        if bool(torch.all(use_subdivision)):
            selected_method = "subdivision"
        elif bool(torch.any(use_subdivision)):
            selected_method = "mixed_natural_subdivision"
        else:
            selected_method = "natural_subdivision_wider"
        result = DensePolynomialRangeResult(natural_lo, natural_hi, hull_lo, hull_hi, selected_lo, selected_hi, selected_method, cover, report, time.perf_counter() - started)
    if trace is not None:
        trace.append(
            {
                "phase": "polynomial_range",
                "context": context,
                "method_requested": policy.method,
                "method_used": result.selected_method,
                "natural_lo": result.natural_lo.detach().cpu().tolist(),
                "natural_hi": result.natural_hi.detach().cpu().tolist(),
                "natural_width": (result.natural_hi - result.natural_lo).detach().cpu().tolist(),
                "tightened_lo": result.subdivision_lo.detach().cpu().tolist(),
                "tightened_hi": result.subdivision_hi.detach().cpu().tolist(),
                "tightened_width": (result.subdivision_hi - result.subdivision_lo).detach().cpu().tolist(),
                "selected_lo": result.selected_lo.detach().cpu().tolist(),
                "selected_hi": result.selected_hi.detach().cpu().tolist(),
                "selected_width": (result.selected_hi - result.selected_lo).detach().cpu().tolist(),
                "leaf_count": int(result.cover.lo.shape[0]),
                "leaf_counts": list(result.cover.leaf_counts),
                "split_variables": [list(items) for items in result.cover.split_variables],
                "depth": int(result.cover.requested_depth),
                "coverage_valid": bool(result.coverage_report["valid"]),
                "finite": bool(torch.all(torch.isfinite(result.selected_lo)) and torch.all(torch.isfinite(result.selected_hi))),
                "device": str(coeffs.device),
                "wall_s": result.wall_s,
            }
        )
    return result


def _merge_coefficients_by_index(
    coeffs: torch.Tensor,
    merge_indices: torch.Tensor,
    unique_count: int,
) -> torch.Tensor:
    """Combine equal exponents before interval evaluation.

    Sparse :class:`Polynomial` construction performs this aggregation.  Doing
    it here is required for dense/sparse truncation-range equivalence; bounding
    every ordered multiply route independently loses cancellation.
    """
    if unique_count == 0:
        return torch.zeros((*coeffs.shape[:-1], 0), dtype=coeffs.dtype, device=coeffs.device)
    target = merge_indices.to(device=coeffs.device, dtype=torch.long)
    target = target.view(*([1] * (coeffs.ndim - 1)), -1).expand_as(coeffs)
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
    integrate_in_indices: torch.Tensor
    integrate_out_indices: torch.Tensor
    integrate_factors: torch.Tensor
    integrate_overflow_indices: torch.Tensor
    integrate_overflow_exponents: torch.Tensor
    integrate_overflow_factors: torch.Tensor
    fingerprint: str

    @staticmethod
    @lru_cache(maxsize=None)
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

        # Integration depends on tau_index, but one route table per possible
        # variable would duplicate storage.  Store the generic source slots and
        # construct the variable-specific kept/overflow view in
        # ``integration_plan`` below.
        integrate_sources = list(range(len(exps)))
        fingerprint_source = f"dim={int(dim)};order={int(order)};" + ";".join(
            ",".join(str(value) for value in exp) for exp in exps
        )
        fingerprint = hashlib.sha256(fingerprint_source.encode("ascii")).hexdigest()

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
            torch.as_tensor(integrate_sources, dtype=torch.long, device=device_t),
            torch.empty(0, dtype=torch.long, device=device_t),
            torch.empty(0, dtype=torch.float64, device=device_t),
            torch.empty(0, dtype=torch.long, device=device_t),
            torch.empty((0, int(dim)), dtype=torch.long, device=device_t),
            torch.empty(0, dtype=torch.float64, device=device_t),
            fingerprint,
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
            self.integrate_in_indices.to(device_t),
            self.integrate_out_indices.to(device_t),
            self.integrate_factors.to(device_t),
            self.integrate_overflow_indices.to(device_t),
            self.integrate_overflow_exponents.to(device_t),
            self.integrate_overflow_factors.to(device_t),
            self.fingerprint,
        )

    def term_index(self, exponent_tuple: Sequence[int]) -> int:
        exp = tuple(int(v) for v in exponent_tuple)
        return self.exponent_to_index[exp]

    def multiplication_plan(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.mul_left_indices, self.mul_right_indices, self.mul_out_indices

    def multiplication_plan_for_degree(
        self,
        max_degree: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return kept routes and exponent-grouped dropped routes."""
        degree = self.order if max_degree is None else int(max_degree)
        if degree < 0 or degree > self.order:
            raise ValueError("max_degree must be between zero and the basis order")
        cache_key = (self.dim, self.order, str(self.device), degree)
        cached = _MULTIPLICATION_DEGREE_PLAN_CACHE.get(cache_key)
        if cached is not None:
            return cached
        kept_mask = self.degree.index_select(0, self.mul_out_indices) <= degree
        kept_left = self.mul_left_indices[kept_mask]
        kept_right = self.mul_right_indices[kept_mask]
        kept_out = self.mul_out_indices[kept_mask]
        newly_dropped_left = self.mul_left_indices[~kept_mask]
        newly_dropped_right = self.mul_right_indices[~kept_mask]
        newly_dropped_exp = self.exponents.index_select(0, self.mul_out_indices[~kept_mask])
        dropped_left = torch.cat([newly_dropped_left, self.trunc_left_indices])
        dropped_right = torch.cat([newly_dropped_right, self.trunc_right_indices])
        dropped_exp = torch.cat([newly_dropped_exp, self.trunc_exponents], dim=0)
        if dropped_exp.shape[0] == 0:
            merge = torch.empty(0, dtype=torch.long, device=self.device)
            unique_exp = torch.empty((0, self.dim), dtype=torch.long, device=self.device)
        else:
            unique_exp, merge = torch.unique(dropped_exp, dim=0, sorted=True, return_inverse=True)
        plan = (kept_left, kept_right, kept_out, dropped_left, dropped_right, merge, unique_exp)
        # The annotation is deliberately structural; route tensors are immutable
        # by contract even though PyTorch tensors themselves are mutable objects.
        _MULTIPLICATION_DEGREE_PLAN_CACHE[cache_key] = plan
        return plan

    def integration_plan(
        self,
        var_index: int,
        *,
        dtype: torch.dtype = torch.float64,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return kept and overflow routes for integration in ``var_index``."""
        index = int(var_index)
        if index < 0 or index >= self.dim:
            raise IndexError(index)
        cache_key = (self.dim, self.order, str(self.device), index, dtype)
        cached = _INTEGRATION_PLAN_CACHE.get(cache_key)
        if cached is not None:
            return cached
        kept_in: list[int] = []
        kept_out: list[int] = []
        kept_factor: list[float] = []
        overflow_in: list[int] = []
        overflow_exp: list[tuple[int, ...]] = []
        overflow_factor: list[float] = []
        for source, exp in enumerate(self.exponent_to_index):
            out_exp = list(exp)
            out_exp[index] += 1
            factor = 1.0 / float(out_exp[index])
            if sum(out_exp) <= self.order:
                kept_in.append(source)
                kept_out.append(self.exponent_to_index[tuple(out_exp)])
                kept_factor.append(factor)
            else:
                overflow_in.append(source)
                overflow_exp.append(tuple(out_exp))
                overflow_factor.append(factor)
        device = self.device
        plan = (
            torch.as_tensor(kept_in, dtype=torch.long, device=device),
            torch.as_tensor(kept_out, dtype=torch.long, device=device),
            torch.as_tensor(kept_factor, dtype=dtype, device=device),
            torch.as_tensor(overflow_in, dtype=torch.long, device=device),
            torch.as_tensor(overflow_exp, dtype=torch.long, device=device).reshape(-1, self.dim),
            torch.as_tensor(overflow_factor, dtype=dtype, device=device),
        )
        _INTEGRATION_PLAN_CACHE[cache_key] = plan
        return plan

    def cutoff_mask(self, coeffs: torch.Tensor, threshold: float | None) -> torch.Tensor:
        if coeffs.shape[-1] != self.num_terms:
            raise ValueError("cutoff coefficient term dimension mismatch")
        if threshold is None:
            return torch.zeros_like(coeffs, dtype=torch.bool)
        return torch.abs(coeffs) <= abs(float(threshold))

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

    def _check_binary_shape(self, other: "BatchedPolynomial") -> None:
        if self.coeffs.shape != other.coeffs.shape:
            raise ValueError(
                "dense polynomial binary operands must have identical "
                f"[batch, output, terms] shape, got {tuple(self.coeffs.shape)} and {tuple(other.coeffs.shape)}"
            )

    def add(self, other: "BatchedPolynomial") -> "BatchedPolynomial":
        self._check_basis(other)
        self._check_binary_shape(other)
        return BatchedPolynomial(self.coeffs + other.coeffs.to(device=self.coeffs.device, dtype=self.coeffs.dtype), self.basis)

    def sub(self, other: "BatchedPolynomial") -> "BatchedPolynomial":
        self._check_basis(other)
        self._check_binary_shape(other)
        return BatchedPolynomial(self.coeffs - other.coeffs.to(device=self.coeffs.device, dtype=self.coeffs.dtype), self.basis)

    def scale(self, scalar: Any) -> "BatchedPolynomial":
        s = torch.as_tensor(scalar, dtype=self.coeffs.dtype, device=self.coeffs.device)
        if s.ndim > 2:
            raise ValueError("scale must be scalar, [batch], or [batch, output]")
        if s.ndim == 1 and s.shape[0] not in {1, self.batch}:
            raise ValueError("one-dimensional scale must have length one or batch")
        if s.ndim == 2 and s.shape not in {(1, 1), (self.batch, 1), (1, self.out_dim), (self.batch, self.out_dim)}:
            raise ValueError("two-dimensional scale cannot broadcast to [batch, output]")
        while s.ndim < self.coeffs.ndim:
            s = s.unsqueeze(-1)
        return BatchedPolynomial(self.coeffs * s, self.basis)

    def affine_map(self, W: torch.Tensor, b: torch.Tensor | None = None) -> "BatchedPolynomial":
        W_t = torch.as_tensor(W, dtype=self.coeffs.dtype, device=self.coeffs.device)
        if W_t.ndim == 2:
            if W_t.shape[1] != self.out_dim:
                raise ValueError("W input dimension does not match polynomial output dimension")
            out = torch.einsum("no,bot->bnt", W_t, self.coeffs)
        elif W_t.ndim == 3:
            if W_t.shape[0] != self.batch or W_t.shape[2] != self.out_dim:
                raise ValueError("batched W must have shape [batch, out_new, out_dim]")
            out = torch.einsum("bno,bot->bnt", W_t, self.coeffs)
        else:
            raise ValueError("W must have shape [out_new, out_dim] or [batch, out_new, out_dim]")
        if b is not None:
            b_t = torch.as_tensor(b, dtype=self.coeffs.dtype, device=self.coeffs.device)
            if b_t.ndim == 1:
                if b_t.shape[0] != out.shape[1]:
                    raise ValueError("b output dimension mismatch")
                out[:, :, self.basis.constant_index] += b_t.view(1, -1)
            elif b_t.ndim == 2:
                if b_t.shape != out.shape[:2]:
                    raise ValueError("batched b must have shape [batch, out_new]")
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
        dropped_merge_mode: str = "merged",
        max_degree: int | None = None,
        range_policy: DenseRangePolicy | None = None,
        range_trace: list[dict[str, Any]] | None = None,
        range_context: str = "polynomial_truncation",
    ) -> "BatchedPolynomial" | tuple["BatchedPolynomial", torch.Tensor, torch.Tensor]:
        if dropped_merge_mode not in {"merged", "termwise"}:
            raise ValueError("dropped_merge_mode must be 'merged' or 'termwise'")
        self._check_basis(other)
        self._check_binary_shape(other)
        other_coeffs = other.coeffs.to(device=self.coeffs.device, dtype=self.coeffs.dtype)
        basis = self.basis
        left, right, kept_out, dropped_left, dropped_right, dropped_merge, dropped_unique_exponents = (
            basis.multiplication_plan_for_degree(max_degree)
        )
        products = self.coeffs.index_select(-1, left) * other_coeffs.index_select(-1, right)
        out = torch.zeros((*products.shape[:-1], basis.num_terms), dtype=self.coeffs.dtype, device=self.coeffs.device)
        target = kept_out.view(*([1] * (products.ndim - 1)), -1).expand_as(products)
        out.scatter_add_(-1, target, products)
        poly = BatchedPolynomial(out, basis)
        if not return_truncation_bound:
            return poly
        if domain_lo is None or domain_hi is None:
            raise ValueError("domain_lo/domain_hi are required for truncation bounds")
        if dropped_left.numel() == 0:
            zeros = torch.zeros(products.shape[:-1], dtype=self.coeffs.dtype, device=self.coeffs.device)
            return poly, zeros, zeros
        dropped = self.coeffs.index_select(-1, dropped_left) * other_coeffs.index_select(-1, dropped_right)
        if dropped_merge_mode == "merged":
            dropped = _merge_coefficients_by_index(
                dropped,
                dropped_merge,
                int(dropped_unique_exponents.shape[0]),
            )
            dropped_exponents = dropped_unique_exponents
        else:
            # Reconstruct the route-level exponents for the diagnostic-only
            # termwise bound.  The production default is exponent-grouped.
            dropped_exponents = torch.cat(
                [
                    basis.exponents.index_select(0, basis.mul_out_indices[basis.degree.index_select(0, basis.mul_out_indices) > (basis.order if max_degree is None else int(max_degree))]),
                    basis.trunc_exponents,
                ],
                dim=0,
            )
        policy = range_policy or DenseRangePolicy()
        trunc_result = _range_for_terms_with_policy(
            dropped,
            dropped_exponents,
            domain_lo,
            domain_hi,
            policy=policy,
            context=range_context,
            trace=range_trace,
        )
        trunc_lo, trunc_hi = trunc_result.selected_lo, trunc_result.selected_hi
        return poly, trunc_lo, trunc_hi

    def square_trunc(self, **kwargs: Any) -> "BatchedPolynomial" | tuple["BatchedPolynomial", torch.Tensor, torch.Tensor]:
        return self.mul_trunc(self, **kwargs)

    def range_bound(
        self,
        domain_lo: torch.Tensor,
        domain_hi: torch.Tensor,
        method: str = "interval",
        *,
        subdivision_depth: int = 1,
        max_leaves: int = 64,
        split_vars: Sequence[int] = (0, 1),
        context: str = "retained_polynomial",
        trace: list[dict[str, Any]] | None = None,
        return_result: bool = False,
        policy: DenseRangePolicy | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | DensePolynomialRangeResult:
        normalized_method = "natural" if method in {"interval", "natural"} else method
        if normalized_method not in {"natural", "subdivision"}:
            raise ValueError("range method must be interval/natural or subdivision")
        policy = policy or DenseRangePolicy(
            method=normalized_method,
            max_depth=0 if normalized_method == "natural" else int(subdivision_depth),
            max_leaves=int(max_leaves),
            split_vars=tuple(int(index) for index in split_vars),
        )
        result = _range_for_terms_with_policy(
            self.coeffs,
            self.basis.exponents,
            domain_lo,
            domain_hi,
            policy=policy,
            context=context,
            trace=trace,
        )
        return result if return_result else (result.selected_lo, result.selected_hi)

    def integrate(
        self,
        var_index: int,
        *,
        domain_lo: torch.Tensor | None = None,
        domain_hi: torch.Tensor | None = None,
        return_overflow_bound: bool = False,
        range_policy: DenseRangePolicy | None = None,
        range_trace: list[dict[str, Any]] | None = None,
        range_context: str = "integration_overflow",
    ) -> "BatchedPolynomial" | tuple["BatchedPolynomial", torch.Tensor, torch.Tensor]:
        kept_in, kept_out, kept_factor, overflow_in, overflow_exp, overflow_factor = self.basis.integration_plan(
            var_index,
            dtype=self.coeffs.dtype,
        )
        kept = self.coeffs.index_select(-1, kept_in) * kept_factor.view(1, 1, -1)
        out = torch.zeros_like(self.coeffs)
        target = kept_out.view(1, 1, -1).expand_as(kept)
        out.scatter_add_(-1, target, kept)
        result = BatchedPolynomial(out, self.basis)
        if not return_overflow_bound:
            return result
        if domain_lo is None or domain_hi is None:
            raise ValueError("domain bounds are required for integration overflow")
        if overflow_in.numel() == 0:
            zeros = torch.zeros(self.coeffs.shape[:2], dtype=self.coeffs.dtype, device=self.coeffs.device)
            return result, zeros, zeros
        overflow_coeffs = self.coeffs.index_select(-1, overflow_in) * overflow_factor.view(1, 1, -1)
        overflow_result = _range_for_terms_with_policy(
            overflow_coeffs,
            overflow_exp,
            domain_lo,
            domain_hi,
            policy=range_policy or DenseRangePolicy(),
            context=range_context,
            trace=range_trace,
        )
        overflow_lo, overflow_hi = overflow_result.selected_lo, overflow_result.selected_hi
        return result, overflow_lo, overflow_hi

    def apply_cutoff(
        self,
        threshold: float | None,
        domain_lo: torch.Tensor,
        domain_hi: torch.Tensor,
        *,
        range_policy: DenseRangePolicy | None = None,
        range_trace: list[dict[str, Any]] | None = None,
        range_context: str = "cutoff",
    ) -> tuple["BatchedPolynomial", torch.Tensor, torch.Tensor]:
        if threshold is None:
            zeros = torch.zeros(self.coeffs.shape[:2], dtype=self.coeffs.dtype, device=self.coeffs.device)
            return self, zeros, zeros
        mask = self.basis.cutoff_mask(self.coeffs, threshold)
        removed = torch.where(mask, self.coeffs, torch.zeros_like(self.coeffs))
        kept = torch.where(mask, torch.zeros_like(self.coeffs), self.coeffs)
        removed_result = _range_for_terms_with_policy(
            removed,
            self.basis.exponents,
            domain_lo,
            domain_hi,
            policy=range_policy or DenseRangePolicy(),
            context=range_context,
            trace=range_trace,
        )
        removed_lo, removed_hi = removed_result.selected_lo, removed_result.selected_hi
        return BatchedPolynomial(kept, self.basis), removed_lo, removed_hi

    def substitute_const_and_drop(self, var_index: int, value: Any) -> "BatchedPolynomial":
        """Substitute one variable and return the complete basis with it removed."""
        index = int(var_index)
        if index < 0 or index >= self.basis.dim:
            raise IndexError(index)
        if self.basis.dim <= 1:
            raise ValueError("cannot drop the only polynomial variable")
        value_t = torch.as_tensor(value, dtype=self.coeffs.dtype, device=self.coeffs.device)
        if value_t.ndim == 0:
            value_t = value_t.expand(self.batch)
        if value_t.shape != (self.batch,):
            raise ValueError("substitution value must be scalar or [batch]")
        new_basis = BatchedMonomialBasis.build(self.basis.dim - 1, self.basis.order, str(self.coeffs.device))
        old_exponents = self.basis.exponents
        powers = old_exponents[:, index]
        factors = value_t[:, None].pow(powers[None, :])
        scaled = self.coeffs * factors[:, None, :]
        targets = []
        for exp in old_exponents.detach().cpu().tolist():
            reduced = tuple(value for i, value in enumerate(exp) if i != index)
            targets.append(new_basis.term_index(reduced))
        target_t = torch.as_tensor(targets, dtype=torch.long, device=self.coeffs.device)
        target_t = target_t.view(1, 1, -1).expand_as(scaled)
        out = torch.zeros((self.batch, self.out_dim, new_basis.num_terms), dtype=self.coeffs.dtype, device=self.coeffs.device)
        out.scatter_add_(-1, target_t, scaled)
        return BatchedPolynomial(out, new_basis)

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
    ledger: DenseRemainderLedger = field(default_factory=DenseRemainderLedger.empty)
    range_policy: DenseRangePolicy = field(default_factory=DenseRangePolicy)
    range_trace: list[dict[str, Any]] | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        batch, out_dim, _terms = self.poly.coeffs.shape
        if self.rem_lo.shape != (batch, out_dim) or self.rem_hi.shape != (batch, out_dim):
            raise ValueError("remainder shape must be [batch, out_dim]")
        if self.domain_lo.shape != self.domain_hi.shape or self.domain_lo.shape != (batch, self.poly.basis.dim):
            raise ValueError("domain bounds must have shape [batch, dim]")
        tensors = (self.rem_lo, self.rem_hi, self.domain_lo, self.domain_hi)
        if any(tensor.dtype != self.poly.coeffs.dtype for tensor in tensors):
            raise TypeError("polynomial, remainder, and domain tensors must share dtype")
        if any(tensor.device != self.poly.coeffs.device for tensor in tensors):
            raise ValueError("polynomial, remainder, and domain tensors must share device")
        if not bool(torch.all(self.rem_lo <= self.rem_hi)):
            raise ValueError("remainder lower bounds must not exceed upper bounds")
        if not bool(torch.all(self.domain_lo <= self.domain_hi)):
            raise ValueError("domain lower bounds must not exceed upper bounds")
        if not self.ledger.entries and bool(torch.any(self.rem_lo != 0) or torch.any(self.rem_hi != 0)):
            object.__setattr__(
                self,
                "ledger",
                DenseRemainderLedger.empty().add("initial_remainder", self.rem_lo, self.rem_hi),
            )

    @staticmethod
    def variables_from_domain(
        domain_lo: torch.Tensor,
        domain_hi: torch.Tensor,
        basis: BatchedMonomialBasis,
        *,
        range_policy: DenseRangePolicy | None = None,
        range_trace: list[dict[str, Any]] | None = None,
    ) -> "BatchedTaylorModel":
        lo = torch.as_tensor(domain_lo)
        hi = torch.as_tensor(domain_hi, dtype=lo.dtype, device=lo.device)
        if lo.ndim != 2:
            raise ValueError("domain bounds must have shape [batch, dim]")
        poly = BatchedPolynomial.variables(lo.shape[0], lo.shape[1], basis, device=lo.device, dtype=lo.dtype)
        rem = torch.zeros((lo.shape[0], lo.shape[1]), dtype=lo.dtype, device=lo.device)
        return BatchedTaylorModel(poly, rem, rem.clone(), lo, hi, DenseRemainderLedger.empty(), range_policy or DenseRangePolicy(), range_trace)

    @staticmethod
    def constants_like(values: Any, template: "BatchedTaylorModel") -> "BatchedTaylorModel":
        values_t = torch.as_tensor(values, dtype=template.poly.coeffs.dtype, device=template.poly.coeffs.device)
        if values_t.ndim == 0:
            values_t = values_t.expand(template.poly.batch, template.poly.out_dim)
        elif values_t.ndim == 1:
            if values_t.shape[0] == template.poly.out_dim:
                values_t = values_t.view(1, -1).expand(template.poly.batch, -1)
            elif values_t.shape[0] == template.poly.batch and template.poly.out_dim == 1:
                values_t = values_t.view(-1, 1)
            else:
                raise ValueError("constant vector cannot broadcast to [batch, output]")
        elif values_t.shape != (template.poly.batch, template.poly.out_dim):
            raise ValueError("constant tensor must have shape [batch, output]")
        poly = BatchedPolynomial.constants(values_t, template.poly.basis)
        zeros = torch.zeros_like(values_t)
        return BatchedTaylorModel(
            poly,
            zeros,
            zeros.clone(),
            template.domain_lo,
            template.domain_hi,
            DenseRemainderLedger.empty(),
            template.range_policy,
            template.range_trace,
        )

    def clone(self) -> "BatchedTaylorModel":
        return BatchedTaylorModel(
            self.poly.clone(),
            self.rem_lo.clone(),
            self.rem_hi.clone(),
            self.domain_lo.clone(),
            self.domain_hi.clone(),
            DenseRemainderLedger(
                {name: (lo.clone(), hi.clone()) for name, (lo, hi) in self.ledger.entries.items()}
            ),
            self.range_policy,
            self.range_trace,
        )

    def to(self, device: torch.device | str) -> "BatchedTaylorModel":
        device_t = torch.device(device)
        return BatchedTaylorModel(
            self.poly.to(device_t),
            self.rem_lo.to(device_t),
            self.rem_hi.to(device_t),
            self.domain_lo.to(device_t),
            self.domain_hi.to(device_t),
            DenseRemainderLedger(
                {name: (lo.to(device_t), hi.to(device_t)) for name, (lo, hi) in self.ledger.entries.items()}
            ),
            self.range_policy,
            self.range_trace,
        )

    def _check_domain(self, other: "BatchedTaylorModel") -> None:
        self.poly._check_basis(other.poly)
        if self.domain_lo.shape != other.domain_lo.shape or not torch.allclose(self.domain_lo, other.domain_lo):
            raise ValueError("domain lower bounds mismatch")
        if self.domain_hi.shape != other.domain_hi.shape or not torch.allclose(self.domain_hi, other.domain_hi):
            raise ValueError("domain upper bounds mismatch")
        if self.range_policy != other.range_policy:
            raise ValueError("dense range policy mismatch")
        if self.range_trace is not other.range_trace:
            raise ValueError("dense range trace ownership mismatch")

    def _coerce(self, other: Any) -> "BatchedTaylorModel":
        if isinstance(other, BatchedTaylorModel):
            return other
        return BatchedTaylorModel.constants_like(other, self)

    def with_remainder(
        self,
        rem_lo: Any,
        rem_hi: Any,
        *,
        category: str = "initial_remainder",
    ) -> "BatchedTaylorModel":
        lo = torch.as_tensor(rem_lo, dtype=self.poly.coeffs.dtype, device=self.poly.coeffs.device)
        hi = torch.as_tensor(rem_hi, dtype=self.poly.coeffs.dtype, device=self.poly.coeffs.device)
        lo, hi = torch.broadcast_tensors(lo, hi)
        if lo.shape != (self.poly.batch, self.poly.out_dim):
            try:
                lo = torch.broadcast_to(lo, (self.poly.batch, self.poly.out_dim))
                hi = torch.broadcast_to(hi, (self.poly.batch, self.poly.out_dim))
            except RuntimeError as exc:
                raise ValueError("remainder cannot broadcast to [batch, output]") from exc
        ledger = DenseRemainderLedger.empty().add(category, lo, hi)
        return BatchedTaylorModel(self.poly, lo, hi, self.domain_lo, self.domain_hi, ledger, self.range_policy, self.range_trace)

    def without_remainder(self) -> "BatchedTaylorModel":
        zeros = torch.zeros_like(self.rem_lo)
        return BatchedTaylorModel(
            self.poly,
            zeros,
            zeros.clone(),
            self.domain_lo,
            self.domain_hi,
            DenseRemainderLedger.empty(),
            self.range_policy,
            self.range_trace,
        )

    def add(self, other: Any) -> "BatchedTaylorModel":
        other = self._coerce(other)
        self._check_domain(other)
        rem_lo, rem_hi = _interval_add(self.rem_lo, self.rem_hi, other.rem_lo, other.rem_hi)
        ledger = self.ledger.merge(other.ledger)
        return BatchedTaylorModel(self.poly.add(other.poly), rem_lo, rem_hi, self.domain_lo, self.domain_hi, ledger, self.range_policy, self.range_trace)

    def sub(self, other: Any) -> "BatchedTaylorModel":
        other = self._coerce(other)
        self._check_domain(other)
        rem_lo, rem_hi = _interval_sub(self.rem_lo, self.rem_hi, other.rem_lo, other.rem_hi)
        ledger = self.ledger.merge(other.ledger.negate())
        return BatchedTaylorModel(self.poly.sub(other.poly), rem_lo, rem_hi, self.domain_lo, self.domain_hi, ledger, self.range_policy, self.range_trace)

    def __rsub__(self, other: Any) -> "BatchedTaylorModel":
        return self._coerce(other).sub(self)

    def __neg__(self) -> "BatchedTaylorModel":
        return self.scale(-1.0)

    def scale(self, scalar: Any) -> "BatchedTaylorModel":
        rem_lo, rem_hi = _interval_scale(self.rem_lo, self.rem_hi, scalar)
        return BatchedTaylorModel(
            self.poly.scale(scalar),
            rem_lo,
            rem_hi,
            self.domain_lo,
            self.domain_hi,
            self.ledger.scale(scalar),
            self.range_policy,
            self.range_trace,
        )

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
            DenseRemainderLedger.empty().add(
                "initial_remainder",
                _down(rem_center - rem_radius),
                _up(rem_center + rem_radius),
            ),
            self.range_policy,
            self.range_trace,
        )

    def mul_trunc(
        self,
        other: Any,
        *,
        dropped_merge_mode: str = "merged",
        max_degree: int | None = None,
    ) -> "BatchedTaylorModel":
        other = self._coerce(other)
        self._check_domain(other)
        poly, trunc_lo, trunc_hi = self.poly.mul_trunc(
            other.poly,
            return_truncation_bound=True,
            domain_lo=self.domain_lo,
            domain_hi=self.domain_hi,
            dropped_merge_mode=dropped_merge_mode,
            max_degree=max_degree,
            range_policy=self.range_policy,
            range_trace=self.range_trace,
            range_context="polynomial_truncation",
        )
        p_lo, p_hi = self.poly.range_bound(
            self.domain_lo,
            self.domain_hi,
            policy=self.range_policy,
            context="poly_times_remainder",
            trace=self.range_trace,
        )
        q_lo, q_hi = other.poly.range_bound(
            self.domain_lo,
            self.domain_hi,
            policy=self.range_policy,
            context="remainder_times_poly",
            trace=self.range_trace,
        )
        p_j_lo, p_j_hi = _interval_mul(p_lo, p_hi, other.rem_lo, other.rem_hi)
        q_i_lo, q_i_hi = _interval_mul(q_lo, q_hi, self.rem_lo, self.rem_hi)
        i_j_lo, i_j_hi = _interval_mul(self.rem_lo, self.rem_hi, other.rem_lo, other.rem_hi)
        rem_lo, rem_hi = _interval_add(trunc_lo, trunc_hi, p_j_lo, p_j_hi)
        rem_lo, rem_hi = _interval_add(rem_lo, rem_hi, q_i_lo, q_i_hi)
        rem_lo, rem_hi = _interval_add(rem_lo, rem_hi, i_j_lo, i_j_hi)
        ledger = DenseRemainderLedger.empty()
        ledger = ledger.add("polynomial_truncation", trunc_lo, trunc_hi)
        ledger = ledger.add("poly_times_remainder", p_j_lo, p_j_hi)
        ledger = ledger.add("remainder_times_poly", q_i_lo, q_i_hi)
        ledger = ledger.add("remainder_times_remainder", i_j_lo, i_j_hi)
        return BatchedTaylorModel(poly, rem_lo, rem_hi, self.domain_lo, self.domain_hi, ledger, self.range_policy, self.range_trace)

    def integrate(self, var_index: int) -> "BatchedTaylorModel":
        poly, overflow_lo, overflow_hi = self.poly.integrate(
            var_index,
            domain_lo=self.domain_lo,
            domain_hi=self.domain_hi,
            return_overflow_bound=True,
            range_policy=self.range_policy,
            range_trace=self.range_trace,
            range_context="integration_overflow",
        )
        tau_lo = self.domain_lo[:, int(var_index)].view(-1, 1)
        tau_hi = self.domain_hi[:, int(var_index)].view(-1, 1)
        integrated_rem_lo, integrated_rem_hi = _interval_mul(tau_lo, tau_hi, self.rem_lo, self.rem_hi)
        rem_lo, rem_hi = _interval_add(integrated_rem_lo, integrated_rem_hi, overflow_lo, overflow_hi)
        ledger = DenseRemainderLedger.empty()
        for category, (lo, hi) in self.ledger.entries.items():
            entry_lo, entry_hi = _interval_mul(tau_lo, tau_hi, lo, hi)
            ledger = ledger.add(category, entry_lo, entry_hi)
        ledger = ledger.add("integration_overflow", overflow_lo, overflow_hi)
        return BatchedTaylorModel(poly, rem_lo, rem_hi, self.domain_lo, self.domain_hi, ledger, self.range_policy, self.range_trace)

    def apply_cutoff(self, threshold: float | None) -> "BatchedTaylorModel":
        poly, cutoff_lo, cutoff_hi = self.poly.apply_cutoff(
            threshold,
            self.domain_lo,
            self.domain_hi,
            range_policy=self.range_policy,
            range_trace=self.range_trace,
            range_context="cutoff",
        )
        rem_lo, rem_hi = _interval_add(self.rem_lo, self.rem_hi, cutoff_lo, cutoff_hi)
        ledger = self.ledger.add("cutoff", cutoff_lo, cutoff_hi)
        return BatchedTaylorModel(poly, rem_lo, rem_hi, self.domain_lo, self.domain_hi, ledger, self.range_policy, self.range_trace)

    def range_bound(self, *, context: str = "retained_polynomial") -> tuple[torch.Tensor, torch.Tensor]:
        poly_lo, poly_hi = self.poly.range_bound(
            self.domain_lo,
            self.domain_hi,
            policy=self.range_policy,
            context=context,
            trace=self.range_trace,
        )
        return _interval_add(poly_lo, poly_hi, self.rem_lo, self.rem_hi)

    def recenter_rescale(self) -> "BatchedTaylorModel":
        raise NotImplementedError(
            "recenter_rescale is not a valid no-op; use the declared hybrid boundary carry or dense composition"
        )

    def endpoint(self, var_index: int, value: Any) -> "BatchedTaylorModel":
        poly = self.poly.substitute_const_and_drop(var_index, value)
        index = int(var_index)
        domain_lo = torch.cat([self.domain_lo[:, :index], self.domain_lo[:, index + 1 :]], dim=1)
        domain_hi = torch.cat([self.domain_hi[:, :index], self.domain_hi[:, index + 1 :]], dim=1)
        return BatchedTaylorModel(poly, self.rem_lo, self.rem_hi, domain_lo, domain_hi, self.ledger, self.range_policy, self.range_trace)

    def component(self, index: int) -> "BatchedTaylorModel":
        idx = int(index)
        return BatchedTaylorModel(
            self.poly.component(idx),
            self.rem_lo[:, idx : idx + 1],
            self.rem_hi[:, idx : idx + 1],
            self.domain_lo,
            self.domain_hi,
            DenseRemainderLedger(
                {name: (lo[:, idx : idx + 1], hi[:, idx : idx + 1]) for name, (lo, hi) in self.ledger.entries.items()}
            ),
            self.range_policy,
            self.range_trace,
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
            DenseRemainderLedger(
                {
                    category: (
                        torch.cat(
                            [
                                model.ledger.entries.get(category, (torch.zeros_like(model.rem_lo), torch.zeros_like(model.rem_hi)))[0]
                                for model in models
                            ],
                            dim=1,
                        ),
                        torch.cat(
                            [
                                model.ledger.entries.get(category, (torch.zeros_like(model.rem_lo), torch.zeros_like(model.rem_hi)))[1]
                                for model in models
                            ],
                            dim=1,
                        ),
                    )
                    for category in REMAINDER_LEDGER_CATEGORIES
                    if any(category in model.ledger.entries for model in models)
                }
            ),
            first.range_policy,
            first.range_trace,
        )

    @property
    def domain(self) -> list[Any]:
        """Sparse-style domain view for batch-one generic ODE callables."""
        if self.poly.batch != 1:
            raise ValueError("a list-valued domain view is only defined for batch=1")
        from .interval import Interval

        return [Interval(lo, hi) for lo, hi in zip(self.domain_lo[0], self.domain_hi[0])]

    @property
    def n_vars(self) -> int:
        return self.poly.basis.dim

    def __len__(self) -> int:
        return self.poly.out_dim

    def __iter__(self):
        for index in range(len(self)):
            yield self.component(index)

    def __getitem__(self, index: int) -> "BatchedTaylorModel":
        return self.component(index)

    def is_finite(self) -> bool:
        tensors = (self.poly.coeffs, self.rem_lo, self.rem_hi, self.domain_lo, self.domain_hi)
        return all(bool(torch.all(torch.isfinite(tensor))) for tensor in tensors)

    def vanderpol_rhs(self) -> "BatchedTaylorModel":
        if self.poly.out_dim != 2:
            raise ValueError("Van der Pol RHS requires out_dim=2")
        x = self.component(0)
        y = self.component(1)
        x_sq_y = x.mul_trunc(x).mul_trunc(y)
        return BatchedTaylorModel.concat([y, y.sub(x).sub(x_sq_y)])

    def fixed_picard_step_vdp(self, h: float, order: int | None = None) -> "BatchedTaylorModel":
        raise RuntimeError(
            "fixed_picard_step_vdp was an Euler prototype and has been removed; "
            "use dense_picard_validate_step with an explicit local-time contract"
        )

    def one_fixed_tm_step_vdp(self, h: float, order: int | None = None) -> "BatchedTaylorModel":
        raise RuntimeError(
            "one_fixed_tm_step_vdp was an Euler prototype and is not a validated flowpipe step"
        )

    __add__ = add
    __radd__ = add
    __sub__ = sub
    __mul__ = mul_trunc
    __rmul__ = mul_trunc


DenseRHS = Callable[[BatchedTaylorModel], Any]


@dataclass(frozen=True)
class DenseValidatedStep:
    """Fail-closed result of one dense local-time Picard/self-map solve."""

    segment_tm: BatchedTaylorModel
    raw_endpoint: BatchedTaylorModel | None
    status: str
    validation_attempts: int
    message: str
    contract: DenseTMContract
    counters: DenseExecutionCounters
    trace: tuple[Mapping[str, Any], ...]
    candidate_remainder_lo: torch.Tensor
    candidate_remainder_hi: torch.Tensor
    picard_image_remainder_lo: torch.Tensor
    picard_image_remainder_hi: torch.Tensor
    subset_margin: torch.Tensor

    @property
    def accepted(self) -> bool:
        return self.status == "validated"


def sparse_tmvector_to_dense(
    tmv: Any,
    *,
    order: int,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
    counters: DenseExecutionCounters | None = None,
    segment_boundary: bool = True,
    range_policy: DenseRangePolicy | None = None,
    range_trace: list[dict[str, Any]] | None = None,
) -> BatchedTaylorModel:
    """Convert a sparse batch-one TM vector by exponent, rejecting overflow."""
    from .tm_vector import TMVector

    if not isinstance(tmv, TMVector) or len(tmv) == 0:
        raise TypeError("sparse_tmvector_to_dense requires a non-empty TMVector")
    n_vars = tmv.n_vars
    basis = BatchedMonomialBasis.build(n_vars, int(order), str(torch.device(device)))
    coeffs = torch.zeros((1, len(tmv), basis.num_terms), dtype=dtype, device=device)
    rem_lo = torch.empty((1, len(tmv)), dtype=dtype, device=device)
    rem_hi = torch.empty_like(rem_lo)
    for output, model in enumerate(tmv):
        if model.n_vars != n_vars:
            raise ValueError("all sparse Taylor models must share n_vars")
        for exponent, coefficient in model.polynomial.terms.items():
            if sum(exponent) > int(order) or exponent not in basis.exponent_to_index:
                raise ValueError(f"sparse term {exponent} lies outside the dense order-{order} basis")
            coeffs[0, output, basis.term_index(exponent)] = coefficient.to(dtype=dtype, device=device)
        rem_lo[0, output] = model.remainder.lo.to(dtype=dtype, device=device)
        rem_hi[0, output] = model.remainder.hi.to(dtype=dtype, device=device)
    domain_lo = torch.stack([interval.lo.to(dtype=dtype, device=device) for interval in tmv.domain]).view(1, n_vars)
    domain_hi = torch.stack([interval.hi.to(dtype=dtype, device=device) for interval in tmv.domain]).view(1, n_vars)
    ledger = DenseRemainderLedger.empty().add("initial_remainder", rem_lo, rem_hi)
    if counters is not None:
        counters.sparse_to_dense_conversions += 1
        counters.boundary_scalar_loop_count += 1
        counters.segment_boundary_conversions += int(segment_boundary)
        counters.inner_loop_conversions += int(not segment_boundary)
        source_device = tmv[0].polynomial.device
        if torch.device(source_device) != torch.device(device):
            counters.device_transfer_count += 1
    return BatchedTaylorModel(
        BatchedPolynomial(coeffs, basis),
        rem_lo,
        rem_hi,
        domain_lo,
        domain_hi,
        ledger,
        range_policy or DenseRangePolicy(),
        range_trace,
    )


def dense_to_sparse_tmvector(
    model: BatchedTaylorModel,
    *,
    counters: DenseExecutionCounters | None = None,
    segment_boundary: bool = True,
) -> Any:
    """Convert one dense batch element to the sparse semantic-reference type."""
    from .interval import Interval
    from .polynomial import Polynomial
    from .taylor_model import TaylorModel
    from .tm_vector import TMVector

    if model.poly.batch != 1:
        raise ValueError("dense-to-sparse boundary conversion currently requires batch=1")
    exponents = [tuple(int(value) for value in row) for row in model.poly.basis.exponents.detach().cpu().tolist()]
    domain = [Interval(lo.detach().clone(), hi.detach().clone()) for lo, hi in zip(model.domain_lo[0], model.domain_hi[0])]
    models = []
    for output in range(model.poly.out_dim):
        terms = {
            exponent: coefficient.detach().clone()
            for exponent, coefficient in zip(exponents, model.poly.coeffs[0, output])
            if bool(coefficient != 0)
        }
        models.append(
            TaylorModel(
                Polynomial(terms, n_vars=model.poly.basis.dim),
                Interval(model.rem_lo[0, output], model.rem_hi[0, output]),
                domain,
                order=model.poly.basis.order,
            )
        )
    if counters is not None:
        counters.dense_to_sparse_conversions += 1
        counters.boundary_scalar_loop_count += 1
        counters.segment_boundary_conversions += int(segment_boundary)
        counters.inner_loop_conversions += int(not segment_boundary)
        if model.poly.coeffs.device.type != "cpu":
            counters.device_transfer_count += 1
    return TMVector(models)


def _coerce_dense_rhs_output(result: Any, template: BatchedTaylorModel) -> BatchedTaylorModel:
    if isinstance(result, BatchedTaylorModel):
        out = result
    elif hasattr(result, "models"):
        models = list(result.models)
        if not models or not all(isinstance(item, BatchedTaylorModel) for item in models):
            raise TypeError("generic dense RHS returned non-dense model components")
        out = BatchedTaylorModel.concat(models)
    elif isinstance(result, Sequence):
        models = list(result)
        if not models or not all(isinstance(item, BatchedTaylorModel) for item in models):
            raise TypeError("generic dense RHS sequence contains non-dense components")
        out = BatchedTaylorModel.concat(models)
    else:
        raise TypeError("generic dense RHS must return a BatchedTaylorModel or dense component sequence")
    template._check_domain(out)
    if out.poly.out_dim != template.poly.out_dim:
        raise ValueError("dense RHS output dimension does not match the state dimension")
    return out


def call_dense_rhs(rhs_fn: DenseRHS, state: BatchedTaylorModel) -> BatchedTaylorModel:
    """Call either the canonical dense callable or a batch-one sparse-style ODE."""
    try:
        result = rhs_fn(state)
    except TypeError as one_argument_error:
        try:
            result = rhs_fn(state, None)  # type: ignore[misc]
        except TypeError:
            raise one_argument_error
    return _coerce_dense_rhs_output(result, state)


class _DenseRawTraceScalar:
    """Dense equivalent of the sparse Flow* raw-remainder expression tracer."""

    def __init__(self, model: BatchedTaylorModel, *, effective_order: int, cutoff_threshold: float | None):
        if model.poly.out_dim != 1:
            raise ValueError("raw trace scalars require output dimension one")
        self.model = model
        self.effective_order = int(effective_order)
        self.cutoff_threshold = cutoff_threshold

    @property
    def domain(self) -> list[Any]:
        return self.model.domain

    @property
    def polynomial(self) -> BatchedPolynomial:
        return self.model.poly

    @property
    def remainder(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model.rem_lo, self.model.rem_hi

    def _wrap(self, model: BatchedTaylorModel) -> "_DenseRawTraceScalar":
        return _DenseRawTraceScalar(
            model,
            effective_order=self.effective_order,
            cutoff_threshold=self.cutoff_threshold,
        )

    @staticmethod
    def concat(models: Sequence["_DenseRawTraceScalar"]) -> "_DenseRawTraceVector":
        return _DenseRawTraceVector(models)

    def _coerce(self, other: Any) -> "_DenseRawTraceScalar":
        if isinstance(other, _DenseRawTraceScalar):
            return other
        if isinstance(other, BatchedTaylorModel):
            return self._wrap(other)
        return self._wrap(BatchedTaylorModel.constants_like(other, self.model))

    def __add__(self, other: Any) -> "_DenseRawTraceScalar":
        return self._wrap(self.model.add(self._coerce(other).model))

    __radd__ = __add__

    def __sub__(self, other: Any) -> "_DenseRawTraceScalar":
        return self._wrap(self.model.sub(self._coerce(other).model))

    def __rsub__(self, other: Any) -> "_DenseRawTraceScalar":
        return self._coerce(other).__sub__(self)

    def __neg__(self) -> "_DenseRawTraceScalar":
        return self._wrap(-self.model)

    def __mul__(self, other: Any) -> "_DenseRawTraceScalar":
        product = self.model.mul_trunc(
            self._coerce(other).model,
            max_degree=self.effective_order,
        ).apply_cutoff(self.cutoff_threshold)
        return self._wrap(product)

    __rmul__ = __mul__

    def pow_int(self, exponent: int) -> "_DenseRawTraceScalar":
        if exponent < 0:
            raise ValueError("raw trace only supports nonnegative integer powers")
        result = self._coerce(1.0)
        for _ in range(int(exponent)):
            result = result * self
        return result


class _DenseRawTraceVector:
    def __init__(self, models: Sequence[_DenseRawTraceScalar]):
        self.models = list(models)

    def __len__(self) -> int:
        return len(self.models)

    def __iter__(self):
        return iter(self.models)

    def __getitem__(self, index: int) -> _DenseRawTraceScalar:
        return self.models[index]

    @property
    def domain(self) -> list[Any]:
        return self.models[0].domain if self.models else []


def _call_dense_raw_trace_rhs(
    rhs_fn: DenseRHS,
    candidate: BatchedTaylorModel,
    *,
    effective_order: int,
    cutoff_threshold: float | None,
) -> BatchedTaylorModel:
    state = _DenseRawTraceVector(
        [
            _DenseRawTraceScalar(
                candidate.component(index),
                effective_order=effective_order,
                cutoff_threshold=cutoff_threshold,
            )
            for index in range(candidate.poly.out_dim)
        ]
    )
    try:
        output = rhs_fn(state)  # type: ignore[arg-type]
    except TypeError as one_argument_error:
        try:
            output = rhs_fn(state, None)  # type: ignore[misc,arg-type]
        except TypeError:
            raise one_argument_error
    values = list(output.models) if hasattr(output, "models") else list(output)
    if len(values) != candidate.poly.out_dim or not all(isinstance(value, _DenseRawTraceScalar) for value in values):
        raise TypeError("raw trace RHS must return one traced scalar per state component")
    return BatchedTaylorModel.concat([value.model for value in values])


def _dense_flowstar_raw_compat_image(
    rhs_fn: DenseRHS,
    base_ext: BatchedTaylorModel,
    candidate_with_target: BatchedTaylorModel,
    candidate_poly: BatchedTaylorModel,
    *,
    tau_index: int,
    order: int,
    cutoff_threshold: float | None,
    validation_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, Mapping[str, Any]]:
    raw_rhs = _call_dense_raw_trace_rhs(
        rhs_fn,
        candidate_with_target,
        effective_order=max(int(order) - 1, 0),
        cutoff_threshold=cutoff_threshold,
    )
    tau_lo = candidate_with_target.domain_lo[:, tau_index].view(-1, 1)
    tau_hi = candidate_with_target.domain_hi[:, tau_index].view(-1, 1)
    before_lo, before_hi = _interval_mul(tau_lo, tau_hi, raw_rhs.rem_lo, raw_rhs.rem_hi)
    before_lo, before_hi = _inflate_tensor_interval(before_lo, before_hi, validation_eps)

    regular_rhs = call_dense_rhs(rhs_fn, candidate_with_target)
    tmp = base_ext.add(regular_rhs.integrate(tau_index)).apply_cutoff(cutoff_threshold)
    poly_diff = tmp.poly.sub(candidate_poly.poly)
    diff_lo, diff_hi = poly_diff.range_bound(
        candidate_poly.domain_lo,
        candidate_poly.domain_hi,
        policy=candidate_poly.range_policy,
        context="raw_compat_poly_diff",
        trace=candidate_poly.range_trace,
    )
    diff_lo, diff_hi = _inflate_tensor_interval(diff_lo, diff_hi, validation_eps)
    check_lo, check_hi = _interval_add(base_ext.rem_lo, base_ext.rem_hi, before_lo, before_hi)
    check_lo, check_hi = _interval_add(check_lo, check_hi, diff_lo, diff_hi)
    check_lo, check_hi = _inflate_tensor_interval(check_lo, check_hi, validation_eps)
    return check_lo, check_hi, {
        "raw_rhs_remainder_lo": raw_rhs.rem_lo.detach().cpu().tolist(),
        "raw_rhs_remainder_hi": raw_rhs.rem_hi.detach().cpu().tolist(),
        "accumulated_before_x0_add_lo": before_lo.detach().cpu().tolist(),
        "accumulated_before_x0_add_hi": before_hi.detach().cpu().tolist(),
        "poly_diff_range_lo": diff_lo.detach().cpu().tolist(),
        "poly_diff_range_hi": diff_hi.detach().cpu().tolist(),
        "raw_remainder_ledger_widths": raw_rhs.ledger.widths(),
        "tmp_remainder_ledger_widths": tmp.ledger.widths(),
    }


def dense_polynomial_picard(
    rhs_fn: DenseRHS,
    base_poly: BatchedTaylorModel,
    *,
    tau_index: int,
    order: int,
    iterations: int | None = None,
    cutoff_threshold: float | None = None,
) -> tuple[BatchedTaylorModel, tuple[Mapping[str, Any], ...]]:
    """Construct the dense polynomial Picard candidate in physical local time."""
    if base_poly.poly.basis.order != int(order):
        raise ValueError("dense Picard order must match its complete basis")
    g = base_poly.without_remainder()
    rows: list[Mapping[str, Any]] = []
    for iteration in range(1, max(1, int(order) if iterations is None else int(iterations)) + 1):
        rhs = call_dense_rhs(rhs_fn, g)
        picard = base_poly.without_remainder().add(rhs.integrate(tau_index))
        # Match sparse _picard_polynomial: arithmetic remainder is audited but
        # is not fed into the next polynomial iterate.  Cutoff removal remains
        # visible as the candidate's static seed remainder.
        zeros = torch.zeros_like(picard.rem_lo)
        g = BatchedTaylorModel(
            picard.poly,
            zeros,
            zeros.clone(),
            picard.domain_lo,
            picard.domain_hi,
            DenseRemainderLedger.empty(),
            picard.range_policy,
            picard.range_trace,
        ).apply_cutoff(cutoff_threshold)
        rows.append(
            {
                "phase": "polynomial_picard",
                "iteration": iteration,
                "basis_hash": g.poly.basis.fingerprint,
                "effective_degree": int(
                    torch.max(g.poly.basis.degree[torch.any(g.poly.coeffs != 0, dim=(0, 1))]).item()
                )
                if bool(torch.any(g.poly.coeffs != 0))
                else 0,
                "nonzero_coefficients": int(torch.count_nonzero(g.poly.coeffs).item()),
                "coefficient_sha256": hashlib.sha256(g.poly.coeffs.detach().cpu().numpy().tobytes()).hexdigest(),
                "exponent_support_sha256": hashlib.sha256(
                    torch.any(g.poly.coeffs != 0, dim=(0, 1)).detach().cpu().numpy().tobytes()
                ).hexdigest(),
                "discarded_remainder_widths": picard.ledger.widths(),
                "cutoff_remainder_widths": g.ledger.widths(),
                "finite": g.is_finite(),
                "range_method": g.range_policy.method,
                "subdivision_depth": g.range_policy.max_depth,
            }
        )
        if not g.is_finite():
            break
    return g, tuple(rows)


def _inflate_tensor_interval(
    lo: torch.Tensor,
    hi: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    eps = torch.as_tensor(abs(float(epsilon)), dtype=lo.dtype, device=lo.device)
    return _down(lo - eps), _up(hi + eps)


def _subset_margin(
    outer_lo: torch.Tensor,
    outer_hi: torch.Tensor,
    inner_lo: torch.Tensor,
    inner_hi: torch.Tensor,
) -> torch.Tensor:
    return torch.minimum(inner_lo - outer_lo, outer_hi - inner_hi)


def dense_picard_validate_step(
    rhs_fn: DenseRHS,
    base_ext: BatchedTaylorModel,
    *,
    h: float,
    order: int,
    tau_index: int,
    target_remainder_radius: float,
    cutoff_threshold: float | None,
    max_validation_attempts: int = 2,
    validation_eps: float = 1e-12,
    validation_mode: str = "target_remainder",
    counters: DenseExecutionCounters | None = None,
) -> DenseValidatedStep:
    """Run true dense Picard and an unchanged interval subset self-map test."""
    supported_modes = {
        "target_remainder",
        "target_remainder_normal_eval",
        "target_remainder_refined",
        "target_remainder_flowstar_ctrunc",
        "flowstar_raw_remainder_compat",
    }
    if validation_mode not in supported_modes:
        raise ValueError(f"dense Picard currently requires one of {sorted(supported_modes)}")
    if h <= 0:
        raise ValueError("h must be positive")
    if max_validation_attempts <= 0:
        raise ValueError("max_validation_attempts must be positive")
    if base_ext.poly.basis.order != int(order):
        raise ValueError("requested order differs from the dense basis order")
    if base_ext.domain_lo.shape[1] != base_ext.poly.basis.dim:
        raise ValueError("dense domain/basis variable mismatch")
    if not torch.allclose(
        base_ext.domain_lo[:, tau_index],
        torch.zeros_like(base_ext.domain_lo[:, tau_index]),
    ) or not torch.allclose(
        base_ext.domain_hi[:, tau_index],
        torch.full_like(base_ext.domain_hi[:, tau_index], float(h)),
    ):
        raise ValueError("canonical local time must have physical domain [0,h]")
    counters = counters or DenseExecutionCounters()
    contract = DenseTMContract(
        batch_dim=base_ext.poly.batch,
        state_dim=base_ext.poly.out_dim,
        n_vars=base_ext.poly.basis.dim,
        tau_index=int(tau_index),
        uncertainty_indices=tuple(index for index in range(base_ext.poly.basis.dim) if index != int(tau_index)),
        order=int(order),
        domain_lo=base_ext.domain_lo,
        domain_hi=base_ext.domain_hi,
    )
    candidate, picard_trace = dense_polynomial_picard(
        rhs_fn,
        base_ext.without_remainder(),
        tau_index=tau_index,
        order=order,
        iterations=order,
        cutoff_threshold=cutoff_threshold,
    )
    trace: list[Mapping[str, Any]] = list(picard_trace)
    target_radius = abs(float(target_remainder_radius))
    target_lo = torch.full_like(candidate.rem_lo, -target_radius)
    target_hi = torch.full_like(candidate.rem_hi, target_radius)
    seed_lo, seed_hi = _interval_add(base_ext.rem_lo, base_ext.rem_hi, candidate.rem_lo, candidate.rem_hi)
    seed_lo, seed_hi = _inflate_tensor_interval(seed_lo, seed_hi, validation_eps)
    seed_margin = _subset_margin(target_lo, target_hi, seed_lo, seed_hi)
    empty_margin = torch.full_like(seed_margin, -torch.inf)
    if not candidate.is_finite() or not bool(torch.all(torch.isfinite(seed_lo)) and torch.all(torch.isfinite(seed_hi))):
        return DenseValidatedStep(
            candidate,
            None,
            "nonfinite",
            0,
            "non-finite candidate or initial remainder",
            contract,
            counters,
            tuple(trace),
            target_lo,
            target_hi,
            seed_lo,
            seed_hi,
            empty_margin,
        )
    if not bool(torch.all(seed_margin >= 0)):
        trace.append(
            {
                "phase": "remainder_validation",
                "attempt": 0,
                "validation_status": "failed",
                "rejection_reason": "initial or cutoff remainder exceeds target remainder",
                "subset_margin": seed_margin.detach().cpu().tolist(),
            }
        )
        return DenseValidatedStep(
            candidate,
            None,
            "failed",
            0,
            "initial or cutoff remainder exceeds target remainder",
            contract,
            counters,
            tuple(trace),
            target_lo,
            target_hi,
            seed_lo,
            seed_hi,
            seed_margin,
        )

    refined = validation_mode == "target_remainder_refined"
    current_lo, current_hi = (seed_lo, seed_hi) if refined else (target_lo, target_hi)
    last_image_lo = seed_lo
    last_image_hi = seed_hi
    last_margin = seed_margin
    last_segment_model = candidate.with_remainder(current_lo, current_hi, category="initial_remainder")
    last_rejection_reason = "Picard residual not subset of target remainder"
    for attempt in range(1, max_validation_attempts + 1):
        candidate_with_remainder = candidate.with_remainder(
            current_lo,
            current_hi,
            category="initial_remainder",
        )
        rhs = call_dense_rhs(rhs_fn, candidate_with_remainder)
        picard_image = base_ext.add(rhs.integrate(tau_index))
        residual = picard_image.sub(candidate.without_remainder())
        ordinary_lo, ordinary_hi = residual.range_bound(context="retained_polynomial")
        ordinary_lo, ordinary_hi = _inflate_tensor_interval(ordinary_lo, ordinary_hi, validation_eps)
        compat_extra: Mapping[str, Any] = {}
        if validation_mode == "flowstar_raw_remainder_compat":
            image_lo, image_hi, compat_extra = _dense_flowstar_raw_compat_image(
                rhs_fn,
                base_ext,
                candidate_with_remainder,
                candidate,
                tau_index=tau_index,
                order=order,
                cutoff_threshold=cutoff_threshold,
                validation_eps=validation_eps,
            )
        else:
            image_lo, image_hi = ordinary_lo, ordinary_hi
        finite = bool(
            torch.all(torch.isfinite(image_lo))
            and torch.all(torch.isfinite(image_hi))
            and torch.all(torch.isfinite(ordinary_lo))
            and torch.all(torch.isfinite(ordinary_hi))
        )
        self_margin = _subset_margin(current_lo, current_hi, image_lo, image_hi)
        target_margin = _subset_margin(target_lo, target_hi, image_lo, image_hi)
        self_subset = finite and bool(torch.all(self_margin >= 0))
        target_subset = finite and bool(torch.all(target_margin >= 0))
        last_image_lo, last_image_hi = image_lo, image_hi
        last_margin = self_margin if refined else target_margin
        last_segment_model = candidate.with_remainder(image_lo, image_hi, category="picard_residual")
        rejection_reason = "" if self_subset else (
            "Flowstar raw remainder compat residual not subset of target remainder"
            if validation_mode == "flowstar_raw_remainder_compat"
            else "Picard residual not subset of target remainder"
        )
        if rejection_reason:
            last_rejection_reason = rejection_reason
        trace.append(
            {
                "phase": "remainder_validation",
                "attempt": attempt,
                "validation_mode": validation_mode,
                "validation_status": "validated" if self_subset else "failed",
                "finite": finite,
                "subset_result": self_subset,
                "target_subset_result": target_subset,
                "candidate_remainder_lo": current_lo.detach().cpu().tolist(),
                "candidate_remainder_hi": current_hi.detach().cpu().tolist(),
                "picard_image_remainder_lo": image_lo.detach().cpu().tolist(),
                "picard_image_remainder_hi": image_hi.detach().cpu().tolist(),
                "ordinary_residual_lo": ordinary_lo.detach().cpu().tolist(),
                "ordinary_residual_hi": ordinary_hi.detach().cpu().tolist(),
                "subset_margin": last_margin.detach().cpu().tolist(),
                "remainder_ledger_widths": residual.ledger.widths(),
                "raw_ctrunc_residual_width_sum": float(torch.sum(image_hi - image_lo).detach().cpu()),
                "ordinary_residual_width_sum": float(torch.sum(ordinary_hi - ordinary_lo).detach().cpu()),
                "polynomial_range_width_sum": float(
                    torch.sum(
                        candidate.poly.range_bound(
                            candidate.domain_lo,
                            candidate.domain_hi,
                            policy=candidate.range_policy,
                            context="retained_polynomial",
                            trace=None,
                        )[1]
                        - candidate.poly.range_bound(
                            candidate.domain_lo,
                            candidate.domain_hi,
                            policy=candidate.range_policy,
                            context="retained_polynomial",
                            trace=None,
                        )[0]
                    ).detach().cpu()
                ),
                "rejection_reason": rejection_reason,
                "range_method": candidate.range_policy.method,
                "subdivision_depth": candidate.range_policy.max_depth,
                **compat_extra,
            }
        )
        if not finite:
            return DenseValidatedStep(
                last_segment_model,
                None,
                "nonfinite",
                attempt,
                "non-finite Picard residual interval",
                contract,
                counters,
                tuple(trace),
                image_lo,
                image_hi,
                image_lo,
                image_hi,
                last_margin,
            )
        if self_subset:
            endpoint = last_segment_model.endpoint(tau_index, float(h))
            return DenseValidatedStep(
                last_segment_model,
                endpoint,
                "validated",
                attempt,
                "",
                contract,
                counters,
                tuple(trace),
                image_lo,
                image_hi,
                image_lo,
                image_hi,
                last_margin,
            )
        if not refined or not target_subset:
            break
        next_lo = torch.minimum(seed_lo, image_lo)
        next_hi = torch.maximum(seed_hi, image_hi)
        current_lo, current_hi = _inflate_tensor_interval(next_lo, next_hi, validation_eps)
        if not bool(torch.all(_subset_margin(target_lo, target_hi, current_lo, current_hi) >= 0)):
            break

    return DenseValidatedStep(
        last_segment_model,
        None,
        "failed",
        min(max_validation_attempts, len([row for row in trace if row.get("phase") == "remainder_validation"])),
        last_rejection_reason,
        contract,
        counters,
        tuple(trace),
        last_image_lo,
        last_image_hi,
        last_image_lo,
        last_image_hi,
        last_margin,
    )


__all__ = [
    "BatchedMonomialBasis",
    "BatchedPolynomial",
    "BatchedTaylorModel",
    "DensePolynomialRangeResult",
    "DenseRangePolicy",
    "DenseSubdivisionCover",
    "DenseExecutionCounters",
    "DenseRemainderLedger",
    "DenseTMContract",
    "DenseValidatedStep",
    "REMAINDER_LEDGER_CATEGORIES",
    "call_dense_rhs",
    "dense_picard_validate_step",
    "dense_polynomial_picard",
    "dense_to_sparse_tmvector",
    "build_dense_subdivision_cover",
    "validate_dense_subdivision_cover",
    "sparse_tmvector_to_dense",
]
