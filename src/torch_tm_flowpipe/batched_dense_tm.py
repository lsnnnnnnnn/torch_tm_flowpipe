"""Canonical dense batched Taylor-model tensors and validated Picard core.

The sparse :mod:`torch_tm_flowpipe.polynomial` and
:mod:`torch_tm_flowpipe.taylor_model` implementations remain the semantic
reference.  This module implements the same complete-total-degree arithmetic
with a batch-first tensor layout and exposes explicit boundary conversions for
parity tests and the ``hybrid_dense_core`` flowpipe lane.  Dense Picard and
remainder validation never convert through a Python polynomial dictionary.
"""
from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import lru_cache
import hashlib
from itertools import product
import math
import time
from typing import Any, Callable, Iterator, Mapping, Sequence

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
_DEVICE_SCALAR_CACHE: dict[
    tuple[float, torch.dtype, str], torch.Tensor
] = {}
_DEVICE_INTEGER_CACHE: dict[tuple[int, str], torch.Tensor] = {}
_COMPILED_POINT_ENCLOSURE_CACHE: dict[
    int, Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]
] = {}
_COMPILED_POINT_ENCLOSURE_VERIFIED: set[
    tuple[int, str, torch.dtype, tuple[int, ...]]
] = set()
_COMPILED_POINT_ENCLOSURE_DISABLED: dict[
    tuple[int, str, torch.dtype, tuple[int, ...]], str
] = {}
_COMPILED_POINT_ENCLOSURE_TELEMETRY: dict[str, Any] = {
    "compiled_calls": 0,
    "eager_fallback_calls": 0,
    "verification_count": 0,
    "compile_and_first_call_seconds": {},
}
_MONOMIAL_INTERVAL_CACHE: OrderedDict[
    tuple[Any, ...],
    tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
] = OrderedDict()
_MONOMIAL_INTERVAL_CACHE_MAXSIZE = 512
_MONOMIAL_INTERVAL_CACHE_HITS = 0
_MONOMIAL_INTERVAL_CACHE_MISSES = 0


@dataclass
class _DenseValidationBatch:
    conditions: list[tuple[torch.Tensor, str, type[Exception]]] = field(
        default_factory=list
    )


_ACTIVE_DENSE_VALIDATION: ContextVar[_DenseValidationBatch | None] = ContextVar(
    "active_dense_validation", default=None
)
_SUPPRESS_TRANSIENT_LEDGER: ContextVar[bool] = ContextVar(
    "suppress_transient_dense_ledger", default=False
)


def _validation_is_deferred() -> bool:
    return _ACTIVE_DENSE_VALIDATION.get() is not None


def _transient_ledger_is_suppressed() -> bool:
    return _SUPPRESS_TRANSIENT_LEDGER.get()


def require_dense_condition(
    condition: torch.Tensor,
    message: str,
    error_type: type[Exception] = ValueError,
) -> None:
    """Check a device condition now, or queue it for one fail-closed boundary."""
    scalar = torch.all(condition)
    batch = _ACTIVE_DENSE_VALIDATION.get()
    if batch is None:
        if not bool(scalar):
            raise error_type(message)
        return
    batch.conditions.append((scalar, message, error_type))


@contextmanager
def dense_validation_batch() -> Iterator[None]:
    """Batch fixed-shape device checks into one synchronization per device.

    Shape, dtype, device, and Python contract checks remain immediate.  Tensor
    ordering/finiteness predicates are accumulated and checked before this
    context returns, so an invalid intermediate can never yield an accepted
    result.  Nested contexts share the outer fail-closed boundary.
    """
    existing = _ACTIVE_DENSE_VALIDATION.get()
    if existing is not None:
        yield
        return
    batch = _DenseValidationBatch()
    token = _ACTIVE_DENSE_VALIDATION.set(batch)
    try:
        yield
        grouped: dict[torch.device, list[tuple[torch.Tensor, str, type[Exception]]]] = {}
        for item in batch.conditions:
            grouped.setdefault(item[0].device, []).append(item)
        for items in grouped.values():
            combined = torch.stack([item[0] for item in items])
            if not bool(torch.all(combined)):
                results = combined.detach().cpu().tolist()
                for valid, (_condition, message, error_type) in zip(
                    results, items, strict=True
                ):
                    if not valid:
                        raise error_type(message)
                raise RuntimeError("deferred dense validation failed without a diagnostic")
    finally:
        _ACTIVE_DENSE_VALIDATION.reset(token)


@contextmanager
def dense_transient_ledger_suppressed() -> Iterator[None]:
    """Omit diagnostic-only ledgers inside a proved tensor math phase."""
    token = _SUPPRESS_TRANSIENT_LEDGER.set(True)
    try:
        yield
    finally:
        _SUPPRESS_TRANSIENT_LEDGER.reset(token)


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
        require_dense_condition(
            self.domain_lo <= self.domain_hi,
            "domain lower bounds must not exceed upper bounds",
        )
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
        if _transient_ledger_is_suppressed():
            return
        for name, (lo, hi) in self.entries.items():
            if lo.shape != hi.shape:
                raise ValueError(f"ledger shape mismatch for {name}")
            require_dense_condition(
                lo <= hi,
                f"invalid interval contribution for {name}",
            )

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
        if _transient_ledger_is_suppressed():
            return DenseRemainderLedger.empty()
        entries = dict(self.entries)
        lo_t = lo.clone()
        hi_t = hi.clone()
        if category in entries:
            lo_t, hi_t = _interval_add(entries[category][0], entries[category][1], lo_t, hi_t)
        entries[category] = (lo_t, hi_t)
        return DenseRemainderLedger(entries)

    def merge(self, other: "DenseRemainderLedger") -> "DenseRemainderLedger":
        if _transient_ledger_is_suppressed():
            return DenseRemainderLedger.empty()
        out = self
        for category, (lo, hi) in other.entries.items():
            out = out.add(category, lo, hi)
        return out

    def scale(self, scalar: Any) -> "DenseRemainderLedger":
        if _transient_ledger_is_suppressed():
            return DenseRemainderLedger.empty()
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

    def intervals(self) -> dict[str, dict[str, list[list[float]]]]:
        return {
            name: {
                "lo": lo.detach().cpu().tolist(),
                "hi": hi.detach().cpu().tolist(),
                "width": (hi - lo).detach().cpu().tolist(),
            }
            for name, (lo, hi) in self.entries.items()
        }


def _as_device(device: torch.device | str | None) -> torch.device:
    return torch.device("cpu") if device is None else torch.device(device)


def _as_dtype(dtype: torch.dtype | None) -> torch.dtype:
    return torch.float64 if dtype is None else dtype


def _to_layout(
    value: torch.Tensor,
    *,
    device: torch.device | str,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Return ``value`` directly when an explicit ``Tensor.to`` is a no-op."""
    device_t = torch.device(device)
    dtype_t = value.dtype if dtype is None else dtype
    if value.device == device_t and value.dtype == dtype_t:
        return value
    return value.to(device=device_t, dtype=dtype_t)


def _device_scalar(value: int | float, like: torch.Tensor) -> torch.Tensor:
    """Reuse an immutable 0-D scalar with the baseline tensor arithmetic path."""
    key = (float(value), like.dtype, str(like.device))
    cached = _DEVICE_SCALAR_CACHE.get(key)
    if cached is None:
        cached = torch.as_tensor(value, dtype=like.dtype, device=like.device)
        _DEVICE_SCALAR_CACHE[key] = cached
    return cached


def _device_integer(value: int, device: torch.device) -> torch.Tensor:
    key = (int(value), str(device))
    cached = _DEVICE_INTEGER_CACHE.get(key)
    if cached is None:
        cached = torch.tensor(int(value), dtype=torch.long, device=device)
        _DEVICE_INTEGER_CACHE[key] = cached
    return cached


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
    if isinstance(scale, torch.Tensor):
        s: Any = _to_layout(scale, device=lo.device, dtype=lo.dtype)
        while s.ndim < lo.ndim:
            s = s.unsqueeze(-1)
    elif isinstance(scale, (int, float)):
        s = _device_scalar(scale, lo)
    else:
        s = torch.as_tensor(scale, dtype=lo.dtype, device=lo.device)
        while s.ndim < lo.ndim:
            s = s.unsqueeze(-1)
    low = torch.minimum(lo * s, hi * s)
    high = torch.maximum(lo * s, hi * s)
    return _down(low), _up(high)


def _interval_div_positive_integer(
    lo: torch.Tensor,
    hi: torch.Tensor,
    divisor: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if int(divisor) <= 0:
        raise ValueError("divisor must be a positive integer")
    value = _device_scalar(divisor, lo)
    return _down(lo / value), _up(hi / value)


def _positive_power_over_factorial(
    magnitude: torch.Tensor,
    exponent: int,
) -> torch.Tensor:
    """Outward upper bound for ``magnitude**exponent / exponent!``."""
    if exponent < 0:
        raise ValueError("exponent must be nonnegative")
    result = torch.ones_like(magnitude)
    for factor in range(1, int(exponent) + 1):
        result = _up(result * magnitude)
        result = _up(result / _device_scalar(factor, magnitude))
    return result


def _point_sin_cos_enclosure(
    value: torch.Tensor,
    *,
    series_terms: int = 32,
    maximum_abs_center: float = 8.0,
    backend: str = "eager",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Enclose sin/cos of binary64 point values using rational series.

    Production formal code deliberately does not use ``torch.sin`` or
    ``torch.cos`` here.  Every multiply, add, and positive-integer division is
    expanded by ``nextafter`` and the analytic Taylor tail uses
    ``|f^(n)(x)| <= 1``.
    """
    if value.dtype != torch.float64:
        raise TypeError("formal trigonometric enclosure requires float64")
    if series_terms < 2:
        raise ValueError("series_terms must be at least two")
    require_dense_condition(
        torch.isfinite(value), "trigonometric center must be finite"
    )
    magnitude = torch.abs(value)
    require_dense_condition(
        magnitude <= float(maximum_abs_center),
        "trigonometric center exceeds the proved Maclaurin domain",
    )
    if backend not in {"eager", "compiled"}:
        raise ValueError("point sine/cosine enclosure backend must be eager or compiled")
    if backend == "compiled":
        return _compiled_point_sin_cos_enclosure(
            value, series_terms=series_terms
        )
    return _point_sin_cos_enclosure_kernel(value, series_terms=series_terms)


def _point_sin_cos_enclosure_kernel(
    value: torch.Tensor,
    *,
    series_terms: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure tensor core for optional fixed-shape compilation."""
    magnitude = torch.abs(value)

    x_lo = value
    x_hi = value
    square_lo, square_hi = _interval_mul(x_lo, x_hi, x_lo, x_hi)

    sin_lo = torch.zeros_like(value)
    sin_hi = torch.zeros_like(value)
    term_lo = value
    term_hi = value
    sin_lo, sin_hi = _interval_add(sin_lo, sin_hi, term_lo, term_hi)
    for index in range(1, int(series_terms)):
        term_lo, term_hi = _interval_mul(
            term_lo, term_hi, square_lo, square_hi
        )
        term_lo, term_hi = -term_hi, -term_lo
        term_lo, term_hi = _interval_div_positive_integer(
            term_lo, term_hi, (2 * index) * (2 * index + 1)
        )
        sin_lo, sin_hi = _interval_add(
            sin_lo, sin_hi, term_lo, term_hi
        )
    sin_tail = _positive_power_over_factorial(
        magnitude, 2 * int(series_terms) + 1
    )
    sin_lo = _down(sin_lo - sin_tail)
    sin_hi = _up(sin_hi + sin_tail)

    cos_lo = torch.ones_like(value)
    cos_hi = torch.ones_like(value)
    term_lo = torch.ones_like(value)
    term_hi = torch.ones_like(value)
    for index in range(1, int(series_terms)):
        term_lo, term_hi = _interval_mul(
            term_lo, term_hi, square_lo, square_hi
        )
        term_lo, term_hi = -term_hi, -term_lo
        term_lo, term_hi = _interval_div_positive_integer(
            term_lo, term_hi, (2 * index - 1) * (2 * index)
        )
        cos_lo, cos_hi = _interval_add(
            cos_lo, cos_hi, term_lo, term_hi
        )
    cos_tail = _positive_power_over_factorial(
        magnitude, 2 * int(series_terms)
    )
    cos_lo = _down(cos_lo - cos_tail)
    cos_hi = _up(cos_hi + cos_tail)
    return sin_lo, sin_hi, cos_lo, cos_hi


def _compiled_point_sin_cos_enclosure(
    value: torch.Tensor,
    *,
    series_terms: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the verified compiled point enclosure, or soundly fall back."""
    terms = int(series_terms)
    signature = (terms, str(value.device), value.dtype, tuple(value.shape))
    if value.device.type != "cuda" or signature in _COMPILED_POINT_ENCLOSURE_DISABLED:
        _COMPILED_POINT_ENCLOSURE_TELEMETRY["eager_fallback_calls"] += 1
        return _point_sin_cos_enclosure_kernel(value, series_terms=terms)
    compiled = _COMPILED_POINT_ENCLOSURE_CACHE.get(terms)
    if compiled is None:
        if not hasattr(torch, "compile"):
            _COMPILED_POINT_ENCLOSURE_DISABLED[signature] = "torch_compile_unavailable"
            _COMPILED_POINT_ENCLOSURE_TELEMETRY["eager_fallback_calls"] += 1
            return _point_sin_cos_enclosure_kernel(value, series_terms=terms)

        def fixed_kernel(
            argument: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            return _point_sin_cos_enclosure_kernel(
                argument, series_terms=terms
            )

        compiled = torch.compile(
            fixed_kernel,
            backend="inductor",
            fullgraph=True,
            dynamic=False,
        )
        _COMPILED_POINT_ENCLOSURE_CACHE[terms] = compiled
    started = time.perf_counter() if signature not in _COMPILED_POINT_ENCLOSURE_VERIFIED else None
    try:
        result = compiled(value)
        if signature not in _COMPILED_POINT_ENCLOSURE_VERIFIED:
            reference = _point_sin_cos_enclosure_kernel(
                value, series_terms=terms
            )
            if not all(
                torch.equal(compiled_value, eager_value)
                for compiled_value, eager_value in zip(
                    result, reference, strict=True
                )
            ):
                _COMPILED_POINT_ENCLOSURE_DISABLED[signature] = (
                    "first_call_bitwise_verification_failed"
                )
                _COMPILED_POINT_ENCLOSURE_TELEMETRY["eager_fallback_calls"] += 1
                return reference
            _COMPILED_POINT_ENCLOSURE_VERIFIED.add(signature)
            _COMPILED_POINT_ENCLOSURE_TELEMETRY["verification_count"] += 1
            _COMPILED_POINT_ENCLOSURE_TELEMETRY[
                "compile_and_first_call_seconds"
            ][str(signature)] = time.perf_counter() - started
        _COMPILED_POINT_ENCLOSURE_TELEMETRY["compiled_calls"] += 1
        return result
    except Exception as exc:
        _COMPILED_POINT_ENCLOSURE_DISABLED[signature] = type(exc).__name__
        _COMPILED_POINT_ENCLOSURE_TELEMETRY["eager_fallback_calls"] += 1
        return _point_sin_cos_enclosure_kernel(value, series_terms=terms)


def compiled_point_enclosure_status() -> dict[str, Any]:
    """Return sanitized compile/fallback telemetry for benchmark evidence."""
    return {
        "compiled_calls": int(
            _COMPILED_POINT_ENCLOSURE_TELEMETRY["compiled_calls"]
        ),
        "eager_fallback_calls": int(
            _COMPILED_POINT_ENCLOSURE_TELEMETRY["eager_fallback_calls"]
        ),
        "verification_count": int(
            _COMPILED_POINT_ENCLOSURE_TELEMETRY["verification_count"]
        ),
        "compile_and_first_call_seconds": dict(
            _COMPILED_POINT_ENCLOSURE_TELEMETRY[
                "compile_and_first_call_seconds"
            ]
        ),
        "verified_signatures": len(_COMPILED_POINT_ENCLOSURE_VERIFIED),
        "disabled_signatures": dict(_COMPILED_POINT_ENCLOSURE_DISABLED),
    }


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
    *,
    maximum_power: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if powers.numel() == 0:
        empty = torch.empty((lo.shape[0], 0), dtype=lo.dtype, device=lo.device)
        return empty, empty
    powers = _to_layout(powers, device=lo.device, dtype=torch.long)
    max_power = int(maximum_power)
    if max_power < 0:
        raise ValueError("maximum_power must be nonnegative")
    zero_power = _device_integer(0, powers.device)
    maximum_power_t = _device_integer(max_power, powers.device)
    require_dense_condition(
        (powers >= zero_power) & (powers <= maximum_power_t),
        "power table exceeds its declared maximum",
    )
    lookup_powers = (
        torch.minimum(torch.maximum(powers, zero_power), maximum_power_t)
        if _validation_is_deferred()
        else powers
    )
    lo_cols: list[torch.Tensor] = []
    hi_cols: list[torch.Tensor] = []
    zero = torch.zeros_like(lo)
    one = torch.ones_like(lo)
    lo_abs = torch.minimum(torch.abs(lo), torch.abs(hi))
    hi_abs = torch.maximum(torch.abs(lo), torch.abs(hi))
    crosses_zero = (lo <= 0) & (hi >= 0)
    for power in range(max_power + 1):
        power_t = _device_integer(power, lo.device)
        if power == 0:
            lo_cols.append(one)
            hi_cols.append(one)
        elif power % 2 == 1:
            endpoints = torch.stack([lo.pow(power_t), hi.pow(power_t)], dim=0)
            lo_cols.append(torch.min(endpoints, dim=0).values)
            hi_cols.append(torch.max(endpoints, dim=0).values)
        else:
            lo_cols.append(torch.where(crosses_zero, zero, lo_abs.pow(power_t)))
            hi_cols.append(hi_abs.pow(power_t))
    lo_table = torch.stack(lo_cols, dim=1)
    hi_table = torch.stack(hi_cols, dim=1)
    return (
        _down(lo_table.index_select(1, lookup_powers)),
        _up(hi_table.index_select(1, lookup_powers)),
    )


def _monomial_interval_bounds_for_exponents(
    domain_lo: torch.Tensor,
    domain_hi: torch.Tensor,
    exponents: torch.Tensor,
    *,
    maximum_power: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if domain_lo.ndim == 1:
        domain_lo = domain_lo.unsqueeze(0)
        domain_hi = domain_hi.unsqueeze(0)
    if domain_lo.shape != domain_hi.shape:
        domain_lo, domain_hi = torch.broadcast_tensors(domain_lo, domain_hi)
    batch, dim = domain_lo.shape
    if exponents.shape[1] != dim:
        raise ValueError(f"exponent dimension {exponents.shape[1]} != domain dimension {dim}")
    exponents = _to_layout(exponents, device=domain_lo.device, dtype=torch.long)
    mono_lo = torch.ones((batch, exponents.shape[0]), dtype=domain_lo.dtype, device=domain_lo.device)
    mono_hi = torch.ones_like(mono_lo)
    for var_index in range(dim):
        power_lo, power_hi = _power_interval_bounds(
            domain_lo[:, var_index],
            domain_hi[:, var_index],
            exponents[:, var_index],
            maximum_power=maximum_power,
        )
        mono_lo, mono_hi = _interval_mul(mono_lo, mono_hi, power_lo, power_hi)
    return mono_lo, mono_hi


def _range_for_terms(
    coeffs: torch.Tensor,
    exponents: torch.Tensor,
    domain_lo: torch.Tensor,
    domain_hi: torch.Tensor,
    *,
    maximum_power: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if coeffs.shape[-1] == 0:
        out = torch.zeros(coeffs.shape[:-1], dtype=coeffs.dtype, device=coeffs.device)
        return out, out
    domain_lo = _to_layout(domain_lo, device=coeffs.device, dtype=coeffs.dtype)
    domain_hi = _to_layout(domain_hi, device=coeffs.device, dtype=coeffs.dtype)
    global _MONOMIAL_INTERVAL_CACHE_HITS, _MONOMIAL_INTERVAL_CACHE_MISSES
    cache_key = (
        id(domain_lo),
        int(domain_lo._version),
        id(domain_hi),
        int(domain_hi._version),
        id(exponents),
        int(exponents._version),
        int(maximum_power),
        str(coeffs.device),
        coeffs.dtype,
    )
    cached = _MONOMIAL_INTERVAL_CACHE.get(cache_key)
    if (
        cached is not None
        and cached[0] is domain_lo
        and cached[1] is domain_hi
        and cached[2] is exponents
    ):
        _MONOMIAL_INTERVAL_CACHE_HITS += 1
        _MONOMIAL_INTERVAL_CACHE.move_to_end(cache_key)
        mono_lo, mono_hi = cached[3], cached[4]
    else:
        _MONOMIAL_INTERVAL_CACHE_MISSES += 1
        mono_lo, mono_hi = _monomial_interval_bounds_for_exponents(
            domain_lo,
            domain_hi,
            exponents,
            maximum_power=maximum_power,
        )
        _MONOMIAL_INTERVAL_CACHE[cache_key] = (
            domain_lo,
            domain_hi,
            exponents,
            mono_lo,
            mono_hi,
        )
        _MONOMIAL_INTERVAL_CACHE.move_to_end(cache_key)
        while len(_MONOMIAL_INTERVAL_CACHE) > _MONOMIAL_INTERVAL_CACHE_MAXSIZE:
            _MONOMIAL_INTERVAL_CACHE.popitem(last=False)
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


def monomial_interval_cache_status() -> dict[str, int]:
    """Return aggregate immutable-domain cache telemetry."""
    return {
        "entries": len(_MONOMIAL_INTERVAL_CACHE),
        "hits": int(_MONOMIAL_INTERVAL_CACHE_HITS),
        "misses": int(_MONOMIAL_INTERVAL_CACHE_MISSES),
        "capacity": _MONOMIAL_INTERVAL_CACHE_MAXSIZE,
    }


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()


def _interval_is_valid(lo: torch.Tensor, hi: torch.Tensor) -> bool:
    condition = torch.isfinite(lo) & torch.isfinite(hi) & (lo <= hi)
    if _validation_is_deferred():
        require_dense_condition(
            condition,
            "polynomial interval failed finite/ordering validation",
            FloatingPointError,
        )
        return True
    return bool(torch.all(condition))


@dataclass(frozen=True)
class DenseCanonicalPolynomial:
    """Deterministic exponent-grouped coefficient intervals.

    The input float coefficients are treated as exact binary64 values.  Equal
    exponents are aggregated in original term-index order with an outward
    interval addition at every step.  Consequently cancellation cannot erase
    the aggregation error, and no assumption is made that ``scatter_add`` is
    an exact reduction.
    """

    coefficient_lo: torch.Tensor
    coefficient_hi: torch.Tensor
    exponents: torch.Tensor
    exponent_tuples: tuple[tuple[int, ...], ...]
    source_term_count: int
    unique_term_count: int
    duplicate_group_count: int
    coefficient_interval_sha256: str
    exponent_sha256: str
    safeguard: str = "sequential_original_index_interval_add_nextafter"


def canonicalize_dense_polynomial(
    coeffs: torch.Tensor,
    exponents: torch.Tensor,
) -> DenseCanonicalPolynomial:
    """Canonicalize a generic dense polynomial without mutating its tensors."""
    if coeffs.ndim != 3 or exponents.ndim != 2:
        raise ValueError("canonicalization expects coeffs [batch,output,term] and exponents [term,var]")
    if coeffs.shape[-1] != exponents.shape[0]:
        raise ValueError("canonicalization coefficient/exponent term mismatch")
    if not torch.is_floating_point(coeffs):
        raise TypeError("canonicalization coefficients must use a floating dtype")
    if exponents.dtype not in {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}:
        raise TypeError("canonicalization exponents must use an integer dtype")
    if not bool(torch.all(torch.isfinite(coeffs))):
        raise FloatingPointError("canonicalization coefficients must be finite")
    exponents_cpu = exponents.detach().cpu().to(dtype=torch.long)
    if bool(torch.any(exponents_cpu < 0)):
        raise ValueError("polynomial exponents must be nonnegative")
    exponent_rows = tuple(tuple(int(item) for item in row) for row in exponents_cpu.tolist())
    unique_rows = tuple(sorted(set(exponent_rows)))
    members = {row: [] for row in unique_rows}
    for term_index, row in enumerate(exponent_rows):
        members[row].append(term_index)

    coefficient_lows: list[torch.Tensor] = []
    coefficient_highs: list[torch.Tensor] = []
    for row in unique_rows:
        indices = members[row]
        first = coeffs[..., indices[0]]
        aggregate_lo = first.clone()
        aggregate_hi = first.clone()
        for term_index in indices[1:]:
            coefficient = coeffs[..., term_index]
            aggregate_lo, aggregate_hi = _interval_add(
                aggregate_lo,
                aggregate_hi,
                coefficient,
                coefficient,
            )
        coefficient_lows.append(aggregate_lo)
        coefficient_highs.append(aggregate_hi)

    if unique_rows:
        coefficient_lo = torch.stack(coefficient_lows, dim=-1)
        coefficient_hi = torch.stack(coefficient_highs, dim=-1)
        canonical_exponents = torch.as_tensor(unique_rows, dtype=torch.long, device=exponents.device)
    else:
        coefficient_lo = torch.empty((*coeffs.shape[:-1], 0), dtype=coeffs.dtype, device=coeffs.device)
        coefficient_hi = coefficient_lo.clone()
        canonical_exponents = torch.empty((0, exponents.shape[1]), dtype=torch.long, device=exponents.device)
    if not _interval_is_valid(coefficient_lo, coefficient_hi):
        raise FloatingPointError("canonical coefficient aggregation produced an invalid interval")
    coefficient_hash = hashlib.sha256(
        coefficient_lo.detach().cpu().contiguous().numpy().tobytes()
        + coefficient_hi.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()
    return DenseCanonicalPolynomial(
        coefficient_lo,
        coefficient_hi,
        canonical_exponents,
        unique_rows,
        int(coeffs.shape[-1]),
        len(unique_rows),
        sum(len(indices) > 1 for indices in members.values()),
        coefficient_hash,
        _tensor_sha256(canonical_exponents),
    )


def registered_dense_horner_orders(dim: int) -> tuple[tuple[int, ...], ...]:
    """Return the finite generic order family used by production selection.

    For three variables this is exactly ``(u0,u1,tau)``, ``(u1,u0,tau)``,
    and ``(tau,u0,u1)``.  The construction remains valid for any positive
    dimension and removes duplicate permutations for dimensions one and two.
    """
    if int(dim) <= 0:
        raise ValueError("Horner variable dimension must be positive")
    natural = tuple(range(int(dim)))
    swapped = ((1, 0, *range(2, int(dim)))) if int(dim) >= 2 else natural
    last_first = (int(dim) - 1, *range(0, int(dim) - 1))
    ordered: list[tuple[int, ...]] = []
    for candidate in (natural, tuple(swapped), tuple(last_first)):
        if candidate not in ordered:
            ordered.append(candidate)
    return tuple(ordered)


def _validate_variable_order(order: Sequence[int], dim: int) -> tuple[int, ...]:
    normalized = tuple(int(index) for index in order)
    if len(normalized) != int(dim) or tuple(sorted(normalized)) != tuple(range(int(dim))):
        raise ValueError(f"Horner variable order must be a permutation of range({int(dim)})")
    return normalized


@dataclass(frozen=True)
class DenseHornerOrderResult:
    lo: torch.Tensor
    hi: torch.Tensor
    variable_order: tuple[int, ...]
    stages: tuple[Mapping[str, Any], ...]
    reconstructed_exponents: tuple[tuple[int, ...], ...]
    reconstruction_valid: bool
    validated: bool
    fallback_reason: str
    canonical: DenseCanonicalPolynomial


def _evaluate_canonical_horner_range(
    canonical: DenseCanonicalPolynomial,
    domain_lo: torch.Tensor,
    domain_hi: torch.Tensor,
    variable_order: Sequence[int],
    *,
    scope: str = "whole_domain",
) -> DenseHornerOrderResult:
    dim = int(canonical.exponents.shape[1])
    order = _validate_variable_order(variable_order, dim)
    lo = _to_layout(
        domain_lo,
        device=canonical.coefficient_lo.device,
        dtype=canonical.coefficient_lo.dtype,
    )
    hi = _to_layout(
        domain_hi,
        device=canonical.coefficient_lo.device,
        dtype=canonical.coefficient_lo.dtype,
    )
    expected_domain_shape = (canonical.coefficient_lo.shape[0], dim)
    if lo.shape != expected_domain_shape or hi.shape != expected_domain_shape:
        raise ValueError(f"Horner domains must have shape {expected_domain_shape}")
    if not bool(torch.all(torch.isfinite(lo)) and torch.all(torch.isfinite(hi))):
        raise FloatingPointError("Horner domains must be finite")
    if not bool(torch.all(lo <= hi)):
        raise ValueError("Horner domain lower bounds must not exceed upper bounds")

    stages: list[Mapping[str, Any]] = []
    visited: list[tuple[int, ...]] = []
    zero = torch.zeros(canonical.coefficient_lo.shape[:2], dtype=lo.dtype, device=lo.device)

    def recurse(term_indices: tuple[int, ...], depth: int, path: tuple[tuple[int, int], ...]) -> tuple[torch.Tensor, torch.Tensor]:
        if not term_indices:
            return zero, zero
        if depth == dim:
            if len(term_indices) != 1:
                raise RuntimeError("canonical Horner leaf does not contain exactly one exponent")
            term_index = term_indices[0]
            exponent = canonical.exponent_tuples[term_index]
            visited.append(exponent)
            return canonical.coefficient_lo[..., term_index], canonical.coefficient_hi[..., term_index]

        variable = order[depth]
        groups: dict[int, list[int]] = {}
        for term_index in term_indices:
            degree = canonical.exponent_tuples[term_index][variable]
            groups.setdefault(degree, []).append(term_index)
        coefficients = {
            degree: recurse(tuple(groups[degree]), depth + 1, path + ((variable, degree),))
            for degree in sorted(groups)
        }
        maximum_degree = max(coefficients)
        accumulator_lo, accumulator_hi = coefficients[maximum_degree]
        stages.append(
            {
                "scope": scope,
                "stage_depth": depth,
                "variable": variable,
                "degree": maximum_degree,
                "path": [list(item) for item in path],
                "operation": "seed",
                "coefficient_lo": accumulator_lo.detach().cpu().tolist(),
                "coefficient_hi": accumulator_hi.detach().cpu().tolist(),
                "intermediate_lo": accumulator_lo.detach().cpu().tolist(),
                "intermediate_hi": accumulator_hi.detach().cpu().tolist(),
                "safeguard": "canonical_coefficient_interval",
            }
        )
        variable_lo = lo[:, variable].view(-1, 1)
        variable_hi = hi[:, variable].view(-1, 1)
        for degree in range(maximum_degree - 1, -1, -1):
            prior_lo, prior_hi = accumulator_lo, accumulator_hi
            product_lo, product_hi = _interval_mul(prior_lo, prior_hi, variable_lo, variable_hi)
            coefficient_lo, coefficient_hi = coefficients.get(degree, (zero, zero))
            accumulator_lo, accumulator_hi = _interval_add(
                product_lo,
                product_hi,
                coefficient_lo,
                coefficient_hi,
            )
            stages.append(
                {
                    "scope": scope,
                    "stage_depth": depth,
                    "variable": variable,
                    "degree": degree,
                    "path": [list(item) for item in path],
                    "operation": "multiply_add",
                    "prior_lo": prior_lo.detach().cpu().tolist(),
                    "prior_hi": prior_hi.detach().cpu().tolist(),
                    "variable_lo": variable_lo.detach().cpu().tolist(),
                    "variable_hi": variable_hi.detach().cpu().tolist(),
                    "product_lo": product_lo.detach().cpu().tolist(),
                    "product_hi": product_hi.detach().cpu().tolist(),
                    "coefficient_lo": coefficient_lo.detach().cpu().tolist(),
                    "coefficient_hi": coefficient_hi.detach().cpu().tolist(),
                    "intermediate_lo": accumulator_lo.detach().cpu().tolist(),
                    "intermediate_hi": accumulator_hi.detach().cpu().tolist(),
                    "safeguard": "nextafter_each_interval_multiply_and_add",
                }
            )
        return accumulator_lo, accumulator_hi

    if canonical.unique_term_count == 0:
        result_lo, result_hi = zero, zero
    else:
        result_lo, result_hi = recurse(tuple(range(canonical.unique_term_count)), 0, ())
    reconstructed = tuple(sorted(visited))
    reconstruction_valid = reconstructed == canonical.exponent_tuples
    interval_valid = _interval_is_valid(result_lo, result_hi)
    validated = bool(reconstruction_valid and interval_valid)
    reasons = []
    if not reconstruction_valid:
        reasons.append("exponent coverage/reconstruction mismatch")
    if not interval_valid:
        reasons.append("nonfinite or inverted Horner interval")
    return DenseHornerOrderResult(
        result_lo,
        result_hi,
        order,
        tuple(stages),
        reconstructed,
        reconstruction_valid,
        validated,
        "; ".join(reasons),
        canonical,
    )


def evaluate_dense_horner_range(
    coeffs: torch.Tensor,
    exponents: torch.Tensor,
    domain_lo: torch.Tensor,
    domain_hi: torch.Tensor,
    variable_order: Sequence[int],
    *,
    scope: str = "whole_domain",
) -> DenseHornerOrderResult:
    """Evaluate one specified multivariate Horner factorization."""
    return _evaluate_canonical_horner_range(
        canonicalize_dense_polynomial(coeffs, exponents),
        domain_lo,
        domain_hi,
        variable_order,
        scope=scope,
    )


@dataclass(frozen=True)
class DenseRegisteredHornerResult:
    lo: torch.Tensor
    hi: torch.Tensor
    order_results: tuple[DenseHornerOrderResult, ...]
    selected_order_index: torch.Tensor
    validated: bool
    fallback_reason: str


def _evaluate_registered_horner_canonical(
    canonical: DenseCanonicalPolynomial,
    domain_lo: torch.Tensor,
    domain_hi: torch.Tensor,
    variable_orders: Sequence[Sequence[int]],
    *,
    scope: str,
) -> DenseRegisteredHornerResult:
    dim = int(canonical.exponents.shape[1])
    orders = tuple(sorted({_validate_variable_order(order, dim) for order in variable_orders}))
    if not orders:
        raise ValueError("registered Horner evaluation requires at least one variable order")
    results: list[DenseHornerOrderResult] = []
    failures: list[str] = []
    for order in orders:
        try:
            result = _evaluate_canonical_horner_range(
                canonical,
                domain_lo,
                domain_hi,
                order,
                scope=scope,
            )
        except (FloatingPointError, RuntimeError, ValueError) as exc:
            failures.append(f"{list(order)}:{type(exc).__name__}:{exc}")
            continue
        results.append(result)
        if not result.validated:
            failures.append(f"{list(order)}:{result.fallback_reason}")
    valid_results = [(index, result) for index, result in enumerate(results) if result.validated]
    if not valid_results:
        shape = canonical.coefficient_lo.shape[:2]
        nan = torch.full(shape, torch.nan, dtype=canonical.coefficient_lo.dtype, device=canonical.coefficient_lo.device)
        selected = torch.full(shape, -1, dtype=torch.long, device=canonical.coefficient_lo.device)
        return DenseRegisteredHornerResult(nan, nan.clone(), tuple(results), selected, False, "; ".join(failures) or "no valid Horner order")

    first_index, first_result = valid_results[0]
    best_lo = first_result.lo
    best_hi = first_result.hi
    selected_index = torch.full(best_lo.shape, first_index, dtype=torch.long, device=best_lo.device)
    for result_index, result in valid_results[1:]:
        use_result = (result.hi - result.lo) < (best_hi - best_lo)
        best_lo = torch.where(use_result, result.lo, best_lo)
        best_hi = torch.where(use_result, result.hi, best_hi)
        selected_index = torch.where(use_result, torch.full_like(selected_index, result_index), selected_index)
    return DenseRegisteredHornerResult(
        best_lo,
        best_hi,
        tuple(results),
        selected_index,
        True,
        "; ".join(failures),
    )


def evaluate_dense_registered_horner_range(
    coeffs: torch.Tensor,
    exponents: torch.Tensor,
    domain_lo: torch.Tensor,
    domain_hi: torch.Tensor,
    variable_orders: Sequence[Sequence[int]] | None = None,
    *,
    scope: str = "whole_domain",
) -> DenseRegisteredHornerResult:
    """Evaluate every pre-registered order and select width/lexicographically."""
    canonical = canonicalize_dense_polynomial(coeffs, exponents)
    orders = variable_orders or registered_dense_horner_orders(exponents.shape[1])
    return _evaluate_registered_horner_canonical(canonical, domain_lo, domain_hi, orders, scope=scope)


@dataclass(frozen=True)
class DenseRangePolicy:
    """Explicit polynomial range semantics for the dense core."""

    method: str = "natural"
    max_depth: int = 0
    max_leaves: int = 64
    split_vars: tuple[int, ...] = (0, 1)
    trigger: str = "always"
    named_contexts: tuple[str, ...] = ()
    variable_orders: tuple[tuple[int, ...], ...] = ()

    def __post_init__(self) -> None:
        supported = {
            "natural",
            "subdivision",
            "adaptive_subdivision",
            "horner_fixed",
            "horner_registered_best",
            "subdivision_then_horner",
            "horner_per_leaf",
        }
        if self.method not in supported:
            raise ValueError(f"dense range method must be one of {sorted(supported)}")
        if self.max_depth < 0:
            raise ValueError("dense range max_depth must be nonnegative")
        if self.max_leaves <= 0 or self.max_leaves > 64:
            raise ValueError("dense range max_leaves must lie in [1, 64]")
        if len(set(self.split_vars)) != len(self.split_vars):
            raise ValueError("dense range split_vars must be unique")
        if self.trigger not in {"always", "on_validation_failure", "proactive_depth1_on_named_contexts"}:
            raise ValueError("invalid dense range trigger")
        normalized_orders = tuple(tuple(int(index) for index in order) for order in self.variable_orders)
        if len(set(normalized_orders)) != len(normalized_orders):
            raise ValueError("dense Horner variable orders must be unique")
        if any(len(set(order)) != len(order) for order in normalized_orders):
            raise ValueError("each dense Horner variable order must contain unique indices")
        object.__setattr__(self, "variable_orders", normalized_orders)

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
    horner_lo: torch.Tensor
    horner_hi: torch.Tensor
    selected_lo: torch.Tensor
    selected_hi: torch.Tensor
    selected_method: str
    cover: DenseSubdivisionCover
    coverage_report: Mapping[str, Any]
    horner_report: Mapping[str, Any]
    horner_stages: tuple[Mapping[str, Any], ...]
    fallback_reason: str
    timings: Mapping[str, float]
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
    exponents_t = _to_layout(exponents, device=coeffs.device, dtype=torch.long)
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


def _identity_dense_cover(
    coeffs: torch.Tensor,
    domain_lo: torch.Tensor,
    domain_hi: torch.Tensor,
) -> tuple[DenseSubdivisionCover, Mapping[str, Any]]:
    owner = torch.arange(coeffs.shape[0], dtype=torch.long, device=coeffs.device)
    cover = DenseSubdivisionCover(
        domain_lo.clone(),
        domain_hi.clone(),
        owner,
        0,
        tuple(1 for _ in range(coeffs.shape[0])),
        tuple(() for _ in range(coeffs.shape[0])),
    )
    # This cover is exact by construction: every parent box is cloned once and
    # ``owner`` is the matching arange.  Running the generic cell-enumerating
    # validator here repeats thousands of scalar checks without adding a new
    # proof obligation.  Covers built by subdivision still use the independent
    # validator below.
    report: Mapping[str, Any] = {
        "valid": True,
        "reasons": [],
        "leaf_count": int(coeffs.shape[0]),
        "leaf_counts": [1 for _ in range(coeffs.shape[0])],
        "checked_cells": 0,
        "split_variables": [[] for _ in range(coeffs.shape[0])],
        "validation": "identity_cover_exact_by_construction",
    }
    return cover, report


def _range_for_canonical_terms(
    canonical: DenseCanonicalPolynomial,
    domain_lo: torch.Tensor,
    domain_hi: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Natural monomial range with interval-valued canonical coefficients."""
    if canonical.unique_term_count == 0:
        zero = torch.zeros(
            canonical.coefficient_lo.shape[:2],
            dtype=canonical.coefficient_lo.dtype,
            device=canonical.coefficient_lo.device,
        )
        return zero, zero
    lo = _to_layout(
        domain_lo,
        device=canonical.coefficient_lo.device,
        dtype=canonical.coefficient_lo.dtype,
    )
    hi = _to_layout(
        domain_hi,
        device=canonical.coefficient_lo.device,
        dtype=canonical.coefficient_lo.dtype,
    )
    maximum_power = max(
        (max(row, default=0) for row in canonical.exponent_tuples), default=0
    )
    monomial_lo, monomial_hi = _monomial_interval_bounds_for_exponents(
        lo,
        hi,
        canonical.exponents,
        maximum_power=maximum_power,
    )
    term_lo, term_hi = _interval_mul(
        canonical.coefficient_lo,
        canonical.coefficient_hi,
        monomial_lo[:, None, :],
        monomial_hi[:, None, :],
    )
    result_lo = term_lo[..., 0]
    result_hi = term_hi[..., 0]
    for term_index in range(1, canonical.unique_term_count):
        result_lo, result_hi = _interval_add(
            result_lo,
            result_hi,
            term_lo[..., term_index],
            term_hi[..., term_index],
        )
    return result_lo, result_hi


def _policy_horner_orders(policy: DenseRangePolicy, dim: int) -> tuple[tuple[int, ...], ...]:
    registered = policy.variable_orders or registered_dense_horner_orders(dim)
    validated = tuple(_validate_variable_order(order, dim) for order in registered)
    if policy.method == "horner_fixed":
        return (validated[0],)
    return validated


def _horner_report(
    result: DenseRegisteredHornerResult,
    *,
    requested_orders: Sequence[Sequence[int]],
) -> tuple[dict[str, Any], tuple[Mapping[str, Any], ...]]:
    orders: list[dict[str, Any]] = []
    stages: list[Mapping[str, Any]] = []
    for order_index, order_result in enumerate(result.order_results):
        selected_mask = result.selected_order_index == order_index
        orders.append(
            {
                "variable_order": list(order_result.variable_order),
                "lo": order_result.lo.detach().cpu().tolist(),
                "hi": order_result.hi.detach().cpu().tolist(),
                "width": (order_result.hi - order_result.lo).detach().cpu().tolist(),
                "validated": order_result.validated,
                "reconstruction_valid": order_result.reconstruction_valid,
                "fallback_reason": order_result.fallback_reason,
                "selected_mask": selected_mask.detach().cpu().tolist(),
                "canonical_coefficient_interval_sha256": order_result.canonical.coefficient_interval_sha256,
                "canonical_exponent_sha256": order_result.canonical.exponent_sha256,
                "source_term_count": order_result.canonical.source_term_count,
                "unique_term_count": order_result.canonical.unique_term_count,
                "duplicate_group_count": order_result.canonical.duplicate_group_count,
                "coefficient_aggregation_safeguard": order_result.canonical.safeguard,
            }
        )
        for stage_index, stage in enumerate(order_result.stages):
            stages.append(
                {
                    "variable_order": list(order_result.variable_order),
                    "order_result_index": order_index,
                    "stage_index": stage_index,
                    **stage,
                }
            )
    report = {
        "requested": True,
        "validated": result.validated,
        "fallback_reason": result.fallback_reason,
        "requested_orders": [list(order) for order in requested_orders],
        "selection_rule": "minimum_width_then_lexicographic_variable_order",
        "selected_order_index": result.selected_order_index.detach().cpu().tolist(),
        "orders": orders,
    }
    return report, tuple(stages)


def _hull_owner_intervals(
    leaf_lo: torch.Tensor,
    leaf_hi: torch.Tensor,
    owner: torch.Tensor,
    owner_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    hull_shape = (int(owner_count), leaf_lo.shape[1])
    hull_lo = torch.full(hull_shape, torch.inf, dtype=leaf_lo.dtype, device=leaf_lo.device)
    hull_hi = torch.full(hull_shape, -torch.inf, dtype=leaf_hi.dtype, device=leaf_hi.device)
    hull_index = owner.view(-1, 1).expand(-1, leaf_lo.shape[1])
    hull_lo.scatter_reduce_(0, hull_index, leaf_lo, reduce="amin", include_self=True)
    hull_hi.scatter_reduce_(0, hull_index, leaf_hi, reduce="amax", include_self=True)
    return _down(hull_lo), _up(hull_hi)


def _range_for_terms_with_policy(
    coeffs: torch.Tensor,
    exponents: torch.Tensor,
    domain_lo: torch.Tensor,
    domain_hi: torch.Tensor,
    *,
    maximum_power: int,
    policy: DenseRangePolicy,
    context: str,
    trace: list[dict[str, Any]] | None = None,
) -> DensePolynomialRangeResult:
    started = time.perf_counter()
    natural_started = time.perf_counter()
    horner_method = policy.method in {
        "horner_fixed",
        "horner_registered_best",
        "subdivision_then_horner",
        "horner_per_leaf",
    }
    canonical: DenseCanonicalPolynomial | None = None
    if policy.applies_to(context) and horner_method:
        canonical = canonicalize_dense_polynomial(coeffs, exponents)
        natural_lo, natural_hi = _range_for_canonical_terms(canonical, domain_lo, domain_hi)
    else:
        natural_lo, natural_hi = _range_for_terms(
            coeffs,
            exponents,
            domain_lo,
            domain_hi,
            maximum_power=maximum_power,
        )
    natural_range_s = time.perf_counter() - natural_started
    natural_valid = _interval_is_valid(natural_lo, natural_hi)
    subdivision_method = policy.method in {"subdivision", "adaptive_subdivision"}
    if not natural_valid and not subdivision_method:
        raise FloatingPointError("natural polynomial range failed finite/ordering validation")

    cover_started = time.perf_counter()
    cover, report = _identity_dense_cover(coeffs, domain_lo, domain_hi)
    cover_validation_s = time.perf_counter() - cover_started
    horner_report: Mapping[str, Any] = {
        "requested": False,
        "validated": False,
        "fallback_reason": "Horner not requested for this named context",
        "requested_orders": [],
        "orders": [],
    }
    horner_stages: tuple[Mapping[str, Any], ...] = ()
    horner_lo, horner_hi = natural_lo, natural_hi
    subdivision_lo, subdivision_hi = natural_lo, natural_hi
    selected_lo, selected_hi = natural_lo, natural_hi
    selected_method = "natural"
    fallback_reason = ""
    leaf_evaluation_s = 0.0
    hull_s = 0.0
    horner_range_s = 0.0
    selection_s = 0.0

    applies = policy.applies_to(context)
    combined_method = policy.method in {"subdivision_then_horner", "horner_per_leaf"}
    if not applies or (subdivision_method and policy.max_depth == 0):
        pass
    elif subdivision_method:
        cover_started = time.perf_counter()
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
        cover_validation_s = time.perf_counter() - cover_started
        leaf_started = time.perf_counter()
        leaf_coeffs = coeffs.index_select(0, cover.owner)
        leaf_lo, leaf_hi = _range_for_terms(
            leaf_coeffs,
            exponents,
            cover.lo,
            cover.hi,
            maximum_power=maximum_power,
        )
        if not _interval_is_valid(leaf_lo, leaf_hi):
            raise FloatingPointError("non-finite subdivision leaf range")
        if not natural_valid:
            raise FloatingPointError("natural polynomial range failed finite/ordering validation")
        leaf_evaluation_s = time.perf_counter() - leaf_started
        hull_started = time.perf_counter()
        subdivision_lo, subdivision_hi = _hull_owner_intervals(
            leaf_lo,
            leaf_hi,
            cover.owner,
            coeffs.shape[0],
        )
        hull_s = time.perf_counter() - hull_started
        selection_started = time.perf_counter()
        use_subdivision = (subdivision_hi - subdivision_lo) <= (natural_hi - natural_lo)
        selected_lo = torch.where(use_subdivision, subdivision_lo, natural_lo)
        selected_hi = torch.where(use_subdivision, subdivision_hi, natural_hi)
        if bool(torch.all(use_subdivision)):
            selected_method = "subdivision"
        elif bool(torch.any(use_subdivision)):
            selected_method = "mixed_natural_subdivision"
        else:
            selected_method = "natural_subdivision_wider"
            fallback_reason = "validated subdivision enclosure is wider than natural"
        selection_s = time.perf_counter() - selection_started
    elif horner_method:
        assert canonical is not None
        orders = _policy_horner_orders(policy, exponents.shape[1])
        horner_started = time.perf_counter()
        registered = _evaluate_registered_horner_canonical(
            canonical,
            domain_lo,
            domain_hi,
            orders,
            scope="whole_domain",
        )
        horner_range_s = time.perf_counter() - horner_started
        report_value, stage_value = _horner_report(registered, requested_orders=orders)
        horner_report = report_value
        horner_stages = stage_value
        if registered.validated:
            horner_lo, horner_hi = registered.lo, registered.hi
            selection_started = time.perf_counter()
            use_horner = (horner_hi - horner_lo) <= (natural_hi - natural_lo)
            selected_lo = torch.where(use_horner, horner_lo, natural_lo)
            selected_hi = torch.where(use_horner, horner_hi, natural_hi)
            if bool(torch.all(use_horner)):
                selected_method = "horner_fixed" if policy.method == "horner_fixed" else "horner_registered_best"
            elif bool(torch.any(use_horner)):
                selected_method = "mixed_natural_horner"
            else:
                selected_method = "natural_horner_wider"
                fallback_reason = "validated Horner enclosure is wider than natural"
            selection_s = time.perf_counter() - selection_started
        else:
            fallback_reason = f"explicit natural fallback: {registered.fallback_reason}"

        if combined_method and registered.validated and policy.max_depth > 0:
            cover_started = time.perf_counter()
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
            cover_validation_s = time.perf_counter() - cover_started
            leaf_started = time.perf_counter()
            leaf_coeffs = coeffs.index_select(0, cover.owner)
            leaf_canonical = canonicalize_dense_polynomial(leaf_coeffs, exponents)
            leaf_natural_lo, leaf_natural_hi = _range_for_canonical_terms(
                leaf_canonical,
                cover.lo,
                cover.hi,
            )
            if not _interval_is_valid(leaf_natural_lo, leaf_natural_hi):
                raise FloatingPointError("per-leaf natural polynomial range failed validation")
            leaf_registered = _evaluate_registered_horner_canonical(
                leaf_canonical,
                cover.lo,
                cover.hi,
                orders,
                scope="subdivision_leaf",
            )
            if not leaf_registered.validated:
                fallback_reason = f"explicit whole-domain fallback: invalid per-leaf Horner: {leaf_registered.fallback_reason}"
            else:
                leaf_report, leaf_stages = _horner_report(leaf_registered, requested_orders=orders)
                use_leaf_horner = (
                    (leaf_registered.hi - leaf_registered.lo)
                    <= (leaf_natural_hi - leaf_natural_lo)
                )
                leaf_selected_lo = torch.where(use_leaf_horner, leaf_registered.lo, leaf_natural_lo)
                leaf_selected_hi = torch.where(use_leaf_horner, leaf_registered.hi, leaf_natural_hi)
                horner_report = {
                    **dict(horner_report),
                    "per_leaf": {
                        **leaf_report,
                        "natural_lo": leaf_natural_lo.detach().cpu().tolist(),
                        "natural_hi": leaf_natural_hi.detach().cpu().tolist(),
                        "natural_width": (leaf_natural_hi - leaf_natural_lo).detach().cpu().tolist(),
                        "horner_selected_mask": use_leaf_horner.detach().cpu().tolist(),
                        "sound_selection_rule": "validated_horner_and_width_not_greater_than_natural_per_leaf",
                    },
                }
                horner_stages = (*horner_stages, *leaf_stages)
                leaf_evaluation_s = time.perf_counter() - leaf_started
                hull_started = time.perf_counter()
                subdivision_lo, subdivision_hi = _hull_owner_intervals(
                    leaf_selected_lo,
                    leaf_selected_hi,
                    cover.owner,
                    coeffs.shape[0],
                )
                hull_s = time.perf_counter() - hull_started
                selection_started = time.perf_counter()
                use_combined = (subdivision_hi - subdivision_lo) < (selected_hi - selected_lo)
                selected_lo = torch.where(use_combined, subdivision_lo, selected_lo)
                selected_hi = torch.where(use_combined, subdivision_hi, selected_hi)
                if bool(torch.all(use_combined)):
                    selected_method = "subdivision_then_horner"
                elif bool(torch.any(use_combined)):
                    selected_method = "mixed_natural_horner_subdivision"
                selection_s += time.perf_counter() - selection_started

    timings = {
        "natural_range_s": natural_range_s,
        "horner_range_s": horner_range_s,
        "cover_validation_s": cover_validation_s,
        "leaf_evaluation_s": leaf_evaluation_s,
        "hull_s": hull_s,
        "selection_s": selection_s,
    }
    result = DensePolynomialRangeResult(
        natural_lo=natural_lo,
        natural_hi=natural_hi,
        subdivision_lo=subdivision_lo,
        subdivision_hi=subdivision_hi,
        horner_lo=horner_lo,
        horner_hi=horner_hi,
        selected_lo=selected_lo,
        selected_hi=selected_hi,
        selected_method=selected_method,
        cover=cover,
        coverage_report=report,
        horner_report=horner_report,
        horner_stages=horner_stages,
        fallback_reason=fallback_reason,
        timings=timings,
        wall_s=time.perf_counter() - started,
    )
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
                "horner_lo": result.horner_lo.detach().cpu().tolist(),
                "horner_hi": result.horner_hi.detach().cpu().tolist(),
                "horner_width": (result.horner_hi - result.horner_lo).detach().cpu().tolist(),
                "horner": result.horner_report,
                "horner_stages": list(result.horner_stages),
                "fallback_reason": result.fallback_reason,
                "selected_lo": result.selected_lo.detach().cpu().tolist(),
                "selected_hi": result.selected_hi.detach().cpu().tolist(),
                "selected_width": (result.selected_hi - result.selected_lo).detach().cpu().tolist(),
                "leaf_count": int(result.cover.lo.shape[0]),
                "leaf_counts": list(result.cover.leaf_counts),
                "split_variables": [list(items) for items in result.cover.split_variables],
                "depth": int(result.cover.requested_depth),
                "coverage_valid": bool(result.coverage_report["valid"]),
                "natural_validated": _interval_is_valid(result.natural_lo, result.natural_hi),
                "horner_validated": bool(result.horner_report.get("validated", False)),
                "finite": bool(torch.all(torch.isfinite(result.selected_lo)) and torch.all(torch.isfinite(result.selected_hi))),
                "device": str(coeffs.device),
                **result.timings,
                "range_attribution_s": result.wall_s,
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
    target = _to_layout(merge_indices, device=coeffs.device, dtype=torch.long)
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
        exponents = _to_layout(self.exponents, device=points_t.device)
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
            _to_layout(domain_lo, device=self.device),
            _to_layout(domain_hi, device=self.device),
            self.exponents,
            maximum_power=self.order,
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
        if device_t == self.coeffs.device:
            return self
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
        other_coeffs = _to_layout(
            other.coeffs, device=self.coeffs.device, dtype=self.coeffs.dtype
        )
        return BatchedPolynomial(self.coeffs + other_coeffs, self.basis)

    def sub(self, other: "BatchedPolynomial") -> "BatchedPolynomial":
        self._check_basis(other)
        self._check_binary_shape(other)
        other_coeffs = _to_layout(
            other.coeffs, device=self.coeffs.device, dtype=self.coeffs.dtype
        )
        return BatchedPolynomial(self.coeffs - other_coeffs, self.basis)

    def scale(self, scalar: Any) -> "BatchedPolynomial":
        if isinstance(scalar, torch.Tensor):
            s = _to_layout(
                scalar, dtype=self.coeffs.dtype, device=self.coeffs.device
            )
        elif isinstance(scalar, (int, float)):
            s = _device_scalar(scalar, self.coeffs)
        else:
            s = torch.as_tensor(
                scalar, dtype=self.coeffs.dtype, device=self.coeffs.device
            )
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
        other_coeffs = _to_layout(
            other.coeffs, device=self.coeffs.device, dtype=self.coeffs.dtype
        )
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
        policy = range_policy or DenseRangePolicy()
        factorized_method = policy.method in {
            "horner_fixed",
            "horner_registered_best",
            "subdivision_then_horner",
            "horner_per_leaf",
        }
        factorized_context = factorized_method and policy.applies_to(range_context)
        if dropped_merge_mode == "merged" and not factorized_context:
            dropped = _merge_coefficients_by_index(
                dropped,
                dropped_merge,
                int(dropped_unique_exponents.shape[0]),
            )
            dropped_exponents = dropped_unique_exponents
        else:
            # Horner canonicalization receives the original route coefficients
            # and exponents so it can enclose aggregation error before equal
            # exponents are combined.  The candidate/kept coefficient tensor
            # above is deliberately untouched.
            dropped_exponents = (
                basis.exponents.index_select(0, dropped_left)
                + basis.exponents.index_select(0, dropped_right)
            )
        trunc_result = _range_for_terms_with_policy(
            dropped,
            dropped_exponents,
            domain_lo,
            domain_hi,
            maximum_power=2 * basis.order,
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
        variable_orders: Sequence[Sequence[int]] = (),
        context: str = "retained_polynomial",
        trace: list[dict[str, Any]] | None = None,
        return_result: bool = False,
        policy: DenseRangePolicy | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | DensePolynomialRangeResult:
        normalized_method = "natural" if method in {"interval", "natural"} else method
        supported = {
            "natural",
            "subdivision",
            "horner_fixed",
            "horner_registered_best",
            "subdivision_then_horner",
            "horner_per_leaf",
        }
        if normalized_method not in supported:
            raise ValueError(f"range method must be one of {sorted(supported)}")
        policy = policy or DenseRangePolicy(
            method=normalized_method,
            max_depth=int(subdivision_depth) if normalized_method in {"subdivision", "subdivision_then_horner", "horner_per_leaf"} else 0,
            max_leaves=int(max_leaves),
            split_vars=tuple(int(index) for index in split_vars),
            variable_orders=tuple(tuple(int(index) for index in order) for order in variable_orders),
        )
        result = _range_for_terms_with_policy(
            self.coeffs,
            self.basis.exponents,
            domain_lo,
            domain_hi,
            maximum_power=self.basis.order,
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
            maximum_power=self.basis.order + 1,
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
            maximum_power=self.basis.order,
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
        if not _transient_ledger_is_suppressed():
            require_dense_condition(
                self.rem_lo <= self.rem_hi,
                "remainder lower bounds must not exceed upper bounds",
            )
            require_dense_condition(
                self.domain_lo <= self.domain_hi,
                "domain lower bounds must not exceed upper bounds",
            )
        elif not _validation_is_deferred():
            raise RuntimeError(
                "transient ledger suppression requires a validation batch"
            )
        if _transient_ledger_is_suppressed():
            return
        if not self.ledger.entries and _validation_is_deferred():
            require_dense_condition(
                (self.rem_lo == 0) & (self.rem_hi == 0),
                "nonzero remainder inside a validation batch requires an explicit ledger",
            )
            infer_initial = False
        else:
            infer_initial = not self.ledger.entries and bool(
                torch.any(self.rem_lo != 0) or torch.any(self.rem_hi != 0)
            )
        if infer_initial:
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
        if device_t == self.poly.coeffs.device:
            return self
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
        if self.domain_lo.shape != other.domain_lo.shape:
            raise ValueError("domain lower bounds mismatch")
        if self.domain_hi.shape != other.domain_hi.shape:
            raise ValueError("domain upper bounds mismatch")
        if self.domain_lo is not other.domain_lo:
            require_dense_condition(
                torch.isclose(self.domain_lo, other.domain_lo),
                "domain lower bounds mismatch",
            )
        if self.domain_hi is not other.domain_hi:
            require_dense_condition(
                torch.isclose(self.domain_hi, other.domain_hi),
                "domain upper bounds mismatch",
            )
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
        if not ledger.entries:
            ledger = ledger.add("initial_remainder", rem_lo, rem_hi)
        return BatchedTaylorModel(self.poly.add(other.poly), rem_lo, rem_hi, self.domain_lo, self.domain_hi, ledger, self.range_policy, self.range_trace)

    def sub(self, other: Any) -> "BatchedTaylorModel":
        other = self._coerce(other)
        self._check_domain(other)
        rem_lo, rem_hi = _interval_sub(self.rem_lo, self.rem_hi, other.rem_lo, other.rem_hi)
        ledger = self.ledger.merge(other.ledger.negate())
        if not ledger.entries:
            ledger = ledger.add("initial_remainder", rem_lo, rem_hi)
        return BatchedTaylorModel(self.poly.sub(other.poly), rem_lo, rem_hi, self.domain_lo, self.domain_hi, ledger, self.range_policy, self.range_trace)

    def __rsub__(self, other: Any) -> "BatchedTaylorModel":
        return self._coerce(other).sub(self)

    def __neg__(self) -> "BatchedTaylorModel":
        return self.scale(-1.0)

    def scale(self, scalar: Any) -> "BatchedTaylorModel":
        rem_lo, rem_hi = _interval_scale(self.rem_lo, self.rem_hi, scalar)
        ledger = self.ledger.scale(scalar)
        if not ledger.entries:
            ledger = ledger.add("initial_remainder", rem_lo, rem_hi)
        return BatchedTaylorModel(
            self.poly.scale(scalar),
            rem_lo,
            rem_hi,
            self.domain_lo,
            self.domain_hi,
            ledger,
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
        index = int(var_index)
        if index < 0 or index >= self.poly.basis.dim:
            raise IndexError(index)
        if self.poly.basis.dim <= 1:
            raise ValueError("cannot drop the only polynomial variable")
        value_t = torch.as_tensor(
            value, dtype=self.poly.coeffs.dtype, device=self.poly.coeffs.device
        )
        if value_t.ndim == 0:
            value_t = value_t.expand(self.poly.batch)
        if value_t.shape != (self.poly.batch,):
            raise ValueError("endpoint value must be scalar or [batch]")

        old_exponents = self.poly.basis.exponents
        powers = old_exponents[:, index]
        factors = value_t[:, None].pow(powers[None, :])
        scaled = self.poly.coeffs * factors[:, None, :]
        factor_lo = torch.ones_like(factors)
        factor_hi = torch.ones_like(factors)
        exponent_rows = tuple(self.poly.basis.exponent_to_index)
        for power in range(1, self.poly.basis.order + 1):
            mask = powers == power
            power_count = sum(
                exponent[index] == power for exponent in exponent_rows
            )
            previous_lo = torch.ones(
                (self.poly.batch, power_count),
                dtype=self.poly.coeffs.dtype,
                device=self.poly.coeffs.device,
            )
            previous_hi = previous_lo.clone()
            value_lo = value_t[:, None].expand_as(previous_lo)
            value_hi = value_lo
            for _ in range(power):
                previous_lo, previous_hi = _interval_mul(
                    previous_lo, previous_hi, value_lo, value_hi
                )
            factor_lo[:, mask] = previous_lo
            factor_hi[:, mask] = previous_hi
        scaled_lo, scaled_hi = _interval_mul(
            self.poly.coeffs,
            self.poly.coeffs,
            factor_lo[:, None, :],
            factor_hi[:, None, :],
        )

        new_basis = BatchedMonomialBasis.build(
            self.poly.basis.dim - 1,
            self.poly.basis.order,
            str(self.poly.coeffs.device),
        )
        targets = []
        for exponent in exponent_rows:
            reduced = tuple(
                exponent_value
                for variable, exponent_value in enumerate(exponent)
                if variable != index
            )
            targets.append(new_basis.term_index(reduced))
        target_t = torch.as_tensor(
            targets, dtype=torch.long, device=self.poly.coeffs.device
        )
        expanded_targets = target_t.view(1, 1, -1).expand_as(scaled)
        coefficients = torch.zeros(
            (self.poly.batch, self.poly.out_dim, new_basis.num_terms),
            dtype=self.poly.coeffs.dtype,
            device=self.poly.coeffs.device,
        )
        coefficients.scatter_add_(-1, expanded_targets, scaled)

        exact_lo = torch.zeros_like(coefficients)
        exact_hi = torch.zeros_like(coefficients)
        for source_index, target_index in enumerate(targets):
            lo, hi = _interval_add(
                exact_lo[..., target_index],
                exact_hi[..., target_index],
                scaled_lo[..., source_index],
                scaled_hi[..., source_index],
            )
            exact_lo[..., target_index] = lo
            exact_hi[..., target_index] = hi
        coefficient_error = torch.maximum(
            torch.abs(coefficients - exact_lo),
            torch.abs(exact_hi - coefficients),
        )
        domain_lo = torch.cat([self.domain_lo[:, :index], self.domain_lo[:, index + 1 :]], dim=1)
        domain_hi = torch.cat([self.domain_hi[:, :index], self.domain_hi[:, index + 1 :]], dim=1)
        radius = _polynomial_error_radius(
            coefficient_error, new_basis, domain_lo, domain_hi
        )
        rem_lo, rem_hi = _interval_add(
            self.rem_lo, self.rem_hi, -radius, radius
        )
        ledger = self.ledger.add("roundoff_safeguard", -radius, radius)
        return BatchedTaylorModel(
            BatchedPolynomial(coefficients, new_basis),
            rem_lo,
            rem_hi,
            domain_lo,
            domain_hi,
            ledger,
            self.range_policy,
            self.range_trace,
        )

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
        if _validation_is_deferred():
            require_dense_condition(
                torch.stack([torch.all(torch.isfinite(tensor)) for tensor in tensors]),
                "Taylor-model tensors must be finite",
            )
            return True
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


def _polynomial_error_radius(
    coefficient_error_magnitude: torch.Tensor,
    basis: BatchedMonomialBasis,
    domain_lo: torch.Tensor,
    domain_hi: torch.Tensor,
) -> torch.Tensor:
    """Outward range radius for symmetric coefficient errors."""
    if coefficient_error_magnitude.shape[-1] != basis.num_terms:
        raise ValueError("coefficient error term dimension mismatch")
    monomial_lo, monomial_hi = basis.interval_monomial_bounds(
        domain_lo, domain_hi
    )
    monomial_abs = torch.maximum(
        torch.abs(monomial_lo), torch.abs(monomial_hi)
    )
    products = coefficient_error_magnitude * monomial_abs[:, None, :]
    radius = torch.sum(products, dim=-1)
    # The reduction itself contains at most ``num_terms - 1`` additions.  This
    # standard gamma bound is deliberately based on machine epsilon rather
    # than unit roundoff, so it also covers the computation of ``products``.
    epsilon = torch.finfo(radius.dtype).eps
    operations = basis.num_terms + 1
    gamma = (operations * epsilon) / (1.0 - operations * epsilon)
    return _up(_up(radius) * (1.0 + 2.0 * gamma))


def _with_symmetric_roundoff(
    model: BatchedTaylorModel,
    radius: torch.Tensor,
) -> BatchedTaylorModel:
    zeros = torch.zeros_like(radius)
    radius = torch.maximum(radius, zeros)
    lo, hi = _interval_add(model.rem_lo, model.rem_hi, -radius, radius)
    ledger = model.ledger.add("roundoff_safeguard", -radius, radius)
    return BatchedTaylorModel(
        model.poly,
        lo,
        hi,
        model.domain_lo,
        model.domain_hi,
        ledger,
        model.range_policy,
        model.range_trace,
    )


def _sound_add_tm(
    left: BatchedTaylorModel,
    right: BatchedTaylorModel,
) -> BatchedTaylorModel:
    left._check_domain(right)
    result = left.add(right)
    exact_lo = _down(left.poly.coeffs + right.poly.coeffs)
    exact_hi = _up(left.poly.coeffs + right.poly.coeffs)
    computed = result.poly.coeffs
    coefficient_error = torch.maximum(
        torch.abs(computed - exact_lo), torch.abs(exact_hi - computed)
    )
    radius = _polynomial_error_radius(
        coefficient_error,
        result.poly.basis,
        result.domain_lo,
        result.domain_hi,
    )
    return _with_symmetric_roundoff(result, radius)


def _sound_scale_tm_interval(
    model: BatchedTaylorModel,
    coefficient_lo: torch.Tensor,
    coefficient_hi: torch.Tensor,
) -> BatchedTaylorModel:
    """Scale a Taylor model by an interval coefficient, retaining its center."""
    if coefficient_lo.shape != (model.poly.batch, model.poly.out_dim):
        raise ValueError("coefficient interval must have shape [batch, output]")
    if coefficient_hi.shape != coefficient_lo.shape:
        raise ValueError("coefficient interval shape mismatch")
    require_dense_condition(
        coefficient_lo <= coefficient_hi,
        "invalid coefficient interval",
    )
    midpoint = coefficient_lo + 0.5 * (coefficient_hi - coefficient_lo)
    midpoint = torch.maximum(coefficient_lo, torch.minimum(coefficient_hi, midpoint))
    result = model.scale(midpoint)

    exact_lo = _down(model.poly.coeffs * midpoint.unsqueeze(-1))
    exact_hi = _up(model.poly.coeffs * midpoint.unsqueeze(-1))
    coefficient_error = torch.maximum(
        torch.abs(result.poly.coeffs - exact_lo),
        torch.abs(exact_hi - result.poly.coeffs),
    )
    roundoff_radius = _polynomial_error_radius(
        coefficient_error,
        model.poly.basis,
        model.domain_lo,
        model.domain_hi,
    )
    result = _with_symmetric_roundoff(result, roundoff_radius)

    uncertainty_lo = _down(coefficient_lo - midpoint)
    uncertainty_hi = _up(coefficient_hi - midpoint)
    model_lo, model_hi = model.range_bound(
        context="sine_coefficient_uncertainty"
    )
    extra_lo, extra_hi = _interval_mul(
        uncertainty_lo,
        uncertainty_hi,
        model_lo,
        model_hi,
    )
    rem_lo, rem_hi = _interval_add(
        result.rem_lo, result.rem_hi, extra_lo, extra_hi
    )
    ledger = result.ledger.add(
        "composition_overflow", extra_lo, extra_hi
    )
    return BatchedTaylorModel(
        result.poly,
        rem_lo,
        rem_hi,
        result.domain_lo,
        result.domain_hi,
        ledger,
        result.range_policy,
        result.range_trace,
    )


def _sound_mul_tm(
    left_model: BatchedTaylorModel,
    right_model: BatchedTaylorModel,
) -> BatchedTaylorModel:
    """Taylor-model product with an explicit retained-route FP error bound."""
    left_model._check_domain(right_model)
    result = left_model.mul_trunc(right_model)
    basis = left_model.poly.basis
    (
        kept_left,
        kept_right,
        kept_out,
        dropped_left,
        dropped_right,
        _dropped_merge,
        _dropped_unique_exponents,
    ) = basis.multiplication_plan_for_degree(None)
    left_coeffs = left_model.poly.coeffs
    right_coeffs = right_model.poly.coeffs

    kept_products = left_coeffs.index_select(-1, kept_left) * right_coeffs.index_select(-1, kept_right)
    kept_lo = _down(
        left_coeffs.index_select(-1, kept_left)
        * right_coeffs.index_select(-1, kept_right)
    )
    kept_hi = _up(
        left_coeffs.index_select(-1, kept_left)
        * right_coeffs.index_select(-1, kept_right)
    )
    route_error = torch.maximum(
        torch.abs(kept_products - kept_lo),
        torch.abs(kept_hi - kept_products),
    )
    target = kept_out.view(1, 1, -1).expand_as(route_error)
    coefficient_error = torch.zeros_like(result.poly.coeffs)
    coefficient_error.scatter_add_(-1, target, route_error)
    absolute_route_sum = torch.zeros_like(result.poly.coeffs)
    absolute_route_sum.scatter_add_(-1, target, torch.abs(kept_products))
    route_counts = _to_layout(
        torch.bincount(kept_out, minlength=basis.num_terms),
        dtype=result.poly.coeffs.dtype,
        device=result.poly.coeffs.device,
    )
    epsilon = torch.finfo(result.poly.coeffs.dtype).eps
    gamma = (route_counts + 1.0) * epsilon
    gamma = gamma / (1.0 - gamma)
    coefficient_error = _up(
        coefficient_error
        + _up(absolute_route_sum) * gamma.view(1, 1, -1) * 2.0
    )
    radius = _polynomial_error_radius(
        coefficient_error,
        basis,
        result.domain_lo,
        result.domain_hi,
    )

    if dropped_left.numel():
        dropped_exponents = (
            basis.exponents.index_select(0, dropped_left)
            + basis.exponents.index_select(0, dropped_right)
        )
        dropped_products = (
            left_coeffs.index_select(-1, dropped_left)
            * right_coeffs.index_select(-1, dropped_right)
        )
        dropped_lo = _down(
            left_coeffs.index_select(-1, dropped_left)
            * right_coeffs.index_select(-1, dropped_right)
        )
        dropped_hi = _up(
            left_coeffs.index_select(-1, dropped_left)
            * right_coeffs.index_select(-1, dropped_right)
        )
        dropped_error = torch.maximum(
            torch.abs(dropped_products - dropped_lo),
            torch.abs(dropped_hi - dropped_products),
        )
        monomial_lo, monomial_hi = _monomial_interval_bounds_for_exponents(
            result.domain_lo,
            result.domain_hi,
            dropped_exponents,
            maximum_power=2 * basis.order,
        )
        monomial_abs = torch.maximum(
            torch.abs(monomial_lo), torch.abs(monomial_hi)
        )
        dropped_radius = torch.sum(
            dropped_error * monomial_abs[:, None, :], dim=-1
        )
        dropped_ops = int(dropped_left.numel()) + 1
        dropped_gamma = (dropped_ops * epsilon) / (
            1.0 - dropped_ops * epsilon
        )
        dropped_value_sum = torch.sum(
            torch.abs(dropped_products) * monomial_abs[:, None, :],
            dim=-1,
        )
        dropped_radius = _up(
            _up(dropped_radius)
            + _up(dropped_value_sum) * (2.0 * dropped_gamma)
        )
        radius = _up(radius + dropped_radius)
    return _with_symmetric_roundoff(result, radius)


def sin_tm(
    model: BatchedTaylorModel,
    order: int = 2,
    *,
    maximum_delta_radius: float = 4.0,
    maximum_abs_center: float = 8.0,
    series_terms: int = 32,
    point_enclosure_backend: str = "eager",
) -> BatchedTaylorModel:
    """Sound centered Taylor-model sine for batched CPU/CUDA float64 models.

    The polynomial is the centered Taylor expansion through ``order`` (0--3).
    Transcendental coefficients are enclosed by an outward rational Maclaurin
    evaluator; fixed-support multiplication overflow, coefficient uncertainty,
    floating-point route error, and the analytic Lagrange tail all enter the
    interval remainder.  Domains wider than the proved local composition
    radius fail closed.
    """
    degree = int(order)
    if degree not in {0, 1, 2, 3}:
        raise ValueError("sin_tm order must be one of 0, 1, 2, or 3")
    if model.poly.coeffs.dtype != torch.float64:
        raise TypeError("formal sin_tm requires float64")
    if not model.is_finite():
        raise ValueError("sin_tm input must be finite")

    constant = model.poly.coeffs[..., model.poly.basis.constant_index]
    sin_lo, sin_hi, cos_lo, cos_hi = _point_sin_cos_enclosure(
        constant,
        series_terms=series_terms,
        maximum_abs_center=maximum_abs_center,
        backend=point_enclosure_backend,
    )
    delta_coeffs = model.poly.coeffs.clone()
    delta_coeffs[..., model.poly.basis.constant_index] = 0.0
    delta = BatchedTaylorModel(
        BatchedPolynomial(delta_coeffs, model.poly.basis),
        model.rem_lo,
        model.rem_hi,
        model.domain_lo,
        model.domain_hi,
        model.ledger,
        model.range_policy,
        model.range_trace,
    )
    delta_lo, delta_hi = delta.range_bound(context="sine_delta")
    delta_radius = torch.maximum(torch.abs(delta_lo), torch.abs(delta_hi))
    require_dense_condition(
        delta_radius <= abs(float(maximum_delta_radius)),
        "sin_tm composition radius exceeds maximum_delta_radius; "
        "split the input domain or fail closed",
    )

    constant_poly = BatchedPolynomial.constants(
        sin_lo + 0.5 * (sin_hi - sin_lo), model.poly.basis
    )
    constant_mid = constant_poly.coeffs[
        ..., model.poly.basis.constant_index
    ]
    constant_error_lo = _down(sin_lo - constant_mid)
    constant_error_hi = _up(sin_hi - constant_mid)
    result = BatchedTaylorModel(
        constant_poly,
        constant_error_lo,
        constant_error_hi,
        model.domain_lo,
        model.domain_hi,
        DenseRemainderLedger.empty().add(
            "composition_overflow",
            constant_error_lo,
            constant_error_hi,
        ),
        model.range_policy,
        model.range_trace,
    )

    powers = [delta]
    for _power in range(2, degree + 1):
        powers.append(_sound_mul_tm(powers[-1], delta))
    coefficient_intervals: list[tuple[torch.Tensor, torch.Tensor]] = []
    if degree >= 1:
        coefficient_intervals.append((cos_lo, cos_hi))
    if degree >= 2:
        second_lo, second_hi = _interval_div_positive_integer(
            -sin_hi, -sin_lo, 2
        )
        coefficient_intervals.append((second_lo, second_hi))
    if degree >= 3:
        third_lo, third_hi = _interval_div_positive_integer(
            -cos_hi, -cos_lo, 6
        )
        coefficient_intervals.append((third_lo, third_hi))
    for power_model, (coefficient_lo, coefficient_hi) in zip(
        powers[:degree], coefficient_intervals, strict=True
    ):
        result = _sound_add_tm(
            result,
            _sound_scale_tm_interval(
                power_model, coefficient_lo, coefficient_hi
            ),
        )

    tail_radius = _positive_power_over_factorial(
        delta_radius, degree + 1
    )
    tail_lo = -tail_radius
    tail_hi = tail_radius
    rem_lo, rem_hi = _interval_add(
        result.rem_lo, result.rem_hi, tail_lo, tail_hi
    )
    ledger = result.ledger.add(
        "composition_overflow", tail_lo, tail_hi
    )
    return BatchedTaylorModel(
        result.poly,
        rem_lo,
        rem_hi,
        result.domain_lo,
        result.domain_hi,
        ledger,
        result.range_policy,
        result.range_trace,
    )


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
        "raw_remainder_ledger_intervals": raw_rhs.ledger.intervals(),
        "tmp_remainder_ledger_widths": tmp.ledger.widths(),
        "tmp_remainder_ledger_intervals": tmp.ledger.intervals(),
    }


def dense_polynomial_picard(
    rhs_fn: DenseRHS,
    base_poly: BatchedTaylorModel,
    *,
    tau_index: int,
    order: int,
    iterations: int | None = None,
    cutoff_threshold: float | None = None,
    capture_trace: bool = True,
    profiler_stage_prefix: str | None = None,
) -> tuple[BatchedTaylorModel, tuple[Mapping[str, Any], ...]]:
    """Construct the dense polynomial Picard candidate in physical local time."""
    if base_poly.poly.basis.order != int(order):
        raise ValueError("dense Picard order must match its complete basis")
    g = base_poly.without_remainder()
    rows: list[Mapping[str, Any]] = []
    for iteration in range(1, max(1, int(order) if iterations is None else int(iterations)) + 1):
        profiler_scope = (
            torch.profiler.record_function(
                f"stage::{profiler_stage_prefix}_k{iteration}"
            )
            if profiler_stage_prefix is not None
            else None
        )
        if profiler_scope is not None:
            profiler_scope.__enter__()
        try:
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
        finally:
            if profiler_scope is not None:
                profiler_scope.__exit__(None, None, None)
        if capture_trace:
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
                "discarded_remainder_intervals": picard.ledger.intervals(),
                "cutoff_remainder_widths": g.ledger.widths(),
                "cutoff_remainder_intervals": g.ledger.intervals(),
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
    polynomial_picard_iterations: int | None = None,
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
        iterations=polynomial_picard_iterations,
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
                "remainder_ledger_intervals": residual.ledger.intervals(),
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
    "DenseCanonicalPolynomial",
    "DenseHornerOrderResult",
    "DensePolynomialRangeResult",
    "DenseRegisteredHornerResult",
    "DenseRangePolicy",
    "DenseSubdivisionCover",
    "DenseExecutionCounters",
    "DenseRemainderLedger",
    "DenseTMContract",
    "DenseValidatedStep",
    "REMAINDER_LEDGER_CATEGORIES",
    "call_dense_rhs",
    "compiled_point_enclosure_status",
    "dense_picard_validate_step",
    "dense_polynomial_picard",
    "dense_transient_ledger_suppressed",
    "dense_validation_batch",
    "dense_to_sparse_tmvector",
    "build_dense_subdivision_cover",
    "canonicalize_dense_polynomial",
    "evaluate_dense_horner_range",
    "evaluate_dense_registered_horner_range",
    "registered_dense_horner_orders",
    "monomial_interval_cache_status",
    "sin_tm",
    "validate_dense_subdivision_cover",
    "sparse_tmvector_to_dense",
]
