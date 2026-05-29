"""Small interval-arithmetic layer backed by scalar torch tensors.

The implementation is intentionally conservative but lightweight.  It keeps the
API simple enough for the sparse Taylor-model prototype while still using
``torch.nextafter`` to nudge results outward when possible.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


def _as_tensor(x: Any, *, like: torch.Tensor | None = None) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        t = x.detach().clone() if x.requires_grad else x.clone()
    else:
        dtype = like.dtype if like is not None else torch.float64
        device = like.device if like is not None else None
        t = torch.as_tensor(x, dtype=dtype, device=device)
    if not torch.is_floating_point(t):
        dtype = like.dtype if like is not None else torch.float64
        t = t.to(dtype=dtype)
    return t.reshape(()) if t.numel() == 1 else t


def _neg_inf_like(x: torch.Tensor) -> torch.Tensor:
    return torch.full_like(x, -torch.inf)


def _pos_inf_like(x: torch.Tensor) -> torch.Tensor:
    return torch.full_like(x, torch.inf)


def _down(x: torch.Tensor) -> torch.Tensor:
    return torch.nextafter(x, _neg_inf_like(x))


def _up(x: torch.Tensor) -> torch.Tensor:
    return torch.nextafter(x, _pos_inf_like(x))


@dataclass(frozen=True)
class Interval:
    """Closed interval ``[lo, hi]``.

    The bounds may be scalar tensors or tensors of the same shape.  Most of the
    Taylor-model code uses scalar intervals, but vector-shaped tensors are useful
    for smoke tests and diagnostics.
    """

    lo: torch.Tensor
    hi: torch.Tensor

    def __init__(self, lo: Any, hi: Any | None = None):
        if hi is None:
            hi = lo
        lo_t = _as_tensor(lo)
        hi_t = _as_tensor(hi, like=lo_t)
        if lo_t.shape != hi_t.shape:
            lo_t, hi_t = torch.broadcast_tensors(lo_t, hi_t)
        lo_t = lo_t.clone()
        hi_t = hi_t.clone()
        if torch.any(lo_t > hi_t):
            raise ValueError(f"invalid interval with lo > hi: {lo_t} > {hi_t}")
        object.__setattr__(self, "lo", lo_t)
        object.__setattr__(self, "hi", hi_t)

    @staticmethod
    def point(x: Any) -> "Interval":
        return Interval(x, x)

    @staticmethod
    def zero(*, dtype: torch.dtype = torch.float64, device: torch.device | str | None = None) -> "Interval":
        return Interval(torch.zeros((), dtype=dtype, device=device), torch.zeros((), dtype=dtype, device=device))

    @staticmethod
    def hull(*intervals: "Interval") -> "Interval":
        if not intervals:
            raise ValueError("Interval.hull requires at least one interval")
        lo = intervals[0].lo
        hi = intervals[0].hi
        for iv in intervals[1:]:
            lo = torch.minimum(lo, iv.lo)
            hi = torch.maximum(hi, iv.hi)
        return Interval(_down(lo), _up(hi))

    @property
    def dtype(self) -> torch.dtype:
        return self.lo.dtype

    @property
    def device(self) -> torch.device:
        return self.lo.device

    @property
    def lower(self) -> torch.Tensor:
        return self.lo

    @property
    def upper(self) -> torch.Tensor:
        return self.hi

    def width(self) -> torch.Tensor:
        return _up(self.hi - self.lo)

    def radius(self) -> torch.Tensor:
        return _up((self.hi - self.lo) / 2)

    def mid(self) -> torch.Tensor:
        return (self.lo + self.hi) / 2

    def is_finite(self) -> bool:
        return bool(torch.all(torch.isfinite(self.lo)) and torch.all(torch.isfinite(self.hi)))

    def contains(self, x: Any, *, tol: float = 0.0) -> bool:
        t = _as_tensor(x, like=self.lo)
        return bool(torch.all(t >= self.lo - tol) and torch.all(t <= self.hi + tol))

    def contains_interval(self, other: "Interval", *, tol: float = 0.0) -> bool:
        return bool(torch.all(other.lo >= self.lo - tol) and torch.all(other.hi <= self.hi + tol))

    def inflate(self, eps: Any) -> "Interval":
        e = torch.abs(_as_tensor(eps, like=self.lo))
        return Interval(_down(self.lo - e), _up(self.hi + e))

    def scale_about_mid(self, factor: float, *, min_radius: float = 0.0) -> "Interval":
        c = self.mid()
        r = torch.maximum(self.radius() * factor, torch.as_tensor(min_radius, dtype=self.dtype, device=self.device))
        return Interval(_down(c - r), _up(c + r))

    def __add__(self, other: Any) -> "Interval":
        other = ensure_interval(other)
        return Interval(_down(self.lo + other.lo), _up(self.hi + other.hi))

    __radd__ = __add__

    def __sub__(self, other: Any) -> "Interval":
        other = ensure_interval(other)
        return Interval(_down(self.lo - other.hi), _up(self.hi - other.lo))

    def __rsub__(self, other: Any) -> "Interval":
        other = ensure_interval(other)
        return other.__sub__(self)

    def __neg__(self) -> "Interval":
        return Interval(_down(-self.hi), _up(-self.lo))

    def __mul__(self, other: Any) -> "Interval":
        other = ensure_interval(other)
        candidates = torch.stack(
            [self.lo * other.lo, self.lo * other.hi, self.hi * other.lo, self.hi * other.hi], dim=0
        )
        return Interval(_down(torch.min(candidates, dim=0).values), _up(torch.max(candidates, dim=0).values))

    __rmul__ = __mul__

    def reciprocal(self) -> "Interval":
        if bool(torch.any((self.lo <= 0) & (self.hi >= 0))):
            raise ZeroDivisionError("interval reciprocal crosses zero")
        vals = torch.stack([1.0 / self.lo, 1.0 / self.hi], dim=0)
        return Interval(_down(torch.min(vals, dim=0).values), _up(torch.max(vals, dim=0).values))

    def __truediv__(self, other: Any) -> "Interval":
        other = ensure_interval(other)
        return self * other.reciprocal()

    def __rtruediv__(self, other: Any) -> "Interval":
        other = ensure_interval(other)
        return other.__truediv__(self)

    def pow_int(self, exponent: int) -> "Interval":
        if exponent < 0:
            return self.reciprocal().pow_int(-exponent)
        if exponent == 0:
            return Interval.point(torch.ones_like(self.lo))
        if exponent == 1:
            return self
        if exponent % 2 == 1:
            vals = torch.stack([self.lo.pow(exponent), self.hi.pow(exponent)], dim=0)
            return Interval(_down(torch.min(vals, dim=0).values), _up(torch.max(vals, dim=0).values))
        # even power
        zero = torch.zeros_like(self.lo)
        lo_abs = torch.minimum(torch.abs(self.lo), torch.abs(self.hi))
        hi_abs = torch.maximum(torch.abs(self.lo), torch.abs(self.hi))
        crosses = (self.lo <= 0) & (self.hi >= 0)
        lo = torch.where(crosses, zero, lo_abs.pow(exponent))
        hi = hi_abs.pow(exponent)
        return Interval(_down(lo), _up(hi))

    def to_tuple(self) -> tuple[float, float]:
        return (float(self.lo.detach().cpu()), float(self.hi.detach().cpu()))

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        if self.lo.numel() == 1 and self.hi.numel() == 1:
            return f"Interval({float(self.lo):.17g}, {float(self.hi):.17g})"
        return f"Interval(lo={self.lo}, hi={self.hi})"


def ensure_interval(x: Any) -> Interval:
    return x if isinstance(x, Interval) else Interval.point(x)
