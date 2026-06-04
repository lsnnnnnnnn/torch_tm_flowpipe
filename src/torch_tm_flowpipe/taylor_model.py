"""Taylor model scalar: bounded-degree sparse polynomial plus interval remainder."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List

import torch

from .interval import Interval, ensure_interval
from .polynomial import Polynomial


def _same_domain(a: list[Interval], b: list[Interval]) -> bool:
    if len(a) != len(b):
        return False
    return all(torch.equal(x.lo, y.lo) and torch.equal(x.hi, y.hi) for x, y in zip(a, b))



def _interval_width_float(iv: Interval) -> float:
    return float(iv.width().detach().cpu())


def _interval_finite_with_width(iv: Interval) -> bool:
    return iv.is_finite() and bool(torch.all(torch.isfinite(iv.width())))


def taylor_model_mul_breakdown(a: "TaylorModel", b: "TaylorModel", order: int) -> dict[str, Any]:
    """Return diagnostic-only width terms for Taylor-model multiplication.

    This mirrors :meth:`TaylorModel.__mul__` but does not change or replace the
    normal arithmetic path.
    """
    other = a._coerce(b)
    poly, dropped = a.polynomial.mul_truncate(other.polynomial, int(order))
    p_self_range = a.polynomial.evaluate_interval(a.domain)
    p_other_range = other.polynomial.evaluate_interval(other.domain)
    trunc_range = dropped.evaluate_interval(a.domain)
    p_self_times_other_remainder = p_self_range * other.remainder
    p_other_times_self_remainder = p_other_range * a.remainder
    remainder_times_remainder = a.remainder * other.remainder
    rem = (
        p_self_times_other_remainder
        + p_other_times_self_remainder
        + remainder_times_remainder
        + trunc_range
    )
    kept_poly_range = poly.evaluate_interval(a.domain)
    output_total_range = kept_poly_range + rem
    finite_terms = {
        "kept_poly_range_finite": _interval_finite_with_width(kept_poly_range),
        "dropped_trunc_finite": _interval_finite_with_width(trunc_range),
        "p_self_times_other_remainder_finite": _interval_finite_with_width(p_self_times_other_remainder),
        "p_other_times_self_remainder_finite": _interval_finite_with_width(p_other_times_self_remainder),
        "remainder_times_remainder_finite": _interval_finite_with_width(remainder_times_remainder),
        "total_remainder_finite": _interval_finite_with_width(rem),
        "output_total_range_finite": _interval_finite_with_width(output_total_range),
    }
    return {
        "kept_poly_range_width": _interval_width_float(kept_poly_range),
        "dropped_trunc_width": _interval_width_float(trunc_range),
        "p_self_times_other_remainder_width": _interval_width_float(p_self_times_other_remainder),
        "p_other_times_self_remainder_width": _interval_width_float(p_other_times_self_remainder),
        "remainder_times_remainder_width": _interval_width_float(remainder_times_remainder),
        "total_remainder_width": _interval_width_float(rem),
        "output_total_range_width": _interval_width_float(output_total_range),
        **finite_terms,
        "finite": all(finite_terms.values()),
    }


@dataclass(frozen=True)
class TaylorModel:
    polynomial: Polynomial
    remainder: Interval
    domain: List[Interval]
    order: int | None = None

    def __init__(self, polynomial: Polynomial, remainder: Interval | Any | None = None, domain: Iterable[Interval] | None = None, order: int | None = None):
        domain_l = list(domain or [])
        if polynomial.n_vars != len(domain_l):
            raise ValueError(f"polynomial.n_vars={polynomial.n_vars} but domain has length {len(domain_l)}")
        rem = ensure_interval(0.0 if remainder is None else remainder)
        object.__setattr__(self, "polynomial", polynomial)
        object.__setattr__(self, "remainder", rem)
        object.__setattr__(self, "domain", domain_l)
        object.__setattr__(self, "order", polynomial.degree() if order is None else int(order))

    @staticmethod
    def zero(domain: Iterable[Interval], *, order: int | None = None) -> "TaylorModel":
        domain_l = list(domain)
        return TaylorModel(Polynomial.zero(len(domain_l)), Interval.zero(), domain_l, order=order)

    @staticmethod
    def constant(value: Any, domain: Iterable[Interval], *, order: int | None = None, remainder: Interval | Any | None = None) -> "TaylorModel":
        domain_l = list(domain)
        return TaylorModel(Polynomial.constant(value, len(domain_l)), Interval.zero() if remainder is None else ensure_interval(remainder), domain_l, order=order)

    @staticmethod
    def variable(index: int, domain: Iterable[Interval], *, order: int | None = None) -> "TaylorModel":
        domain_l = list(domain)
        return TaylorModel(Polynomial.variable(index, len(domain_l)), Interval.zero(), domain_l, order=order)

    @property
    def n_vars(self) -> int:
        return self.polynomial.n_vars

    def clone(self) -> "TaylorModel":
        return TaylorModel(self.polynomial.clone(), self.remainder, list(self.domain), self.order)

    def _coerce(self, other: Any) -> "TaylorModel":
        if isinstance(other, TaylorModel):
            if not _same_domain(self.domain, other.domain):
                raise ValueError("TaylorModel domain mismatch")
            return other
        if isinstance(other, Polynomial):
            if other.n_vars != self.n_vars:
                raise ValueError("Polynomial n_vars mismatch")
            return TaylorModel(other, Interval.zero(), list(self.domain), order=self.order)
        if isinstance(other, Interval):
            return TaylorModel(Polynomial.zero(self.n_vars), other, list(self.domain), order=self.order)
        return TaylorModel.constant(other, list(self.domain), order=self.order)

    def with_remainder(self, remainder: Interval | Any) -> "TaylorModel":
        return TaylorModel(self.polynomial, ensure_interval(remainder), list(self.domain), self.order)

    def apply_cutoff(self, threshold: float | None) -> "TaylorModel":
        if threshold is None:
            return self
        poly, removed_range = self.polynomial.cutoff(threshold, self.domain)
        return TaylorModel(poly, self.remainder + removed_range, list(self.domain), self.order)

    def __add__(self, other: Any) -> "TaylorModel":
        other = self._coerce(other)
        return TaylorModel(self.polynomial + other.polynomial, self.remainder + other.remainder, list(self.domain), self.order)

    __radd__ = __add__

    def __sub__(self, other: Any) -> "TaylorModel":
        other = self._coerce(other)
        return TaylorModel(self.polynomial - other.polynomial, self.remainder - other.remainder, list(self.domain), self.order)

    def __rsub__(self, other: Any) -> "TaylorModel":
        return self._coerce(other).__sub__(self)

    def __neg__(self) -> "TaylorModel":
        return TaylorModel(-self.polynomial, -self.remainder, list(self.domain), self.order)

    def __mul__(self, other: Any) -> "TaylorModel":
        other = self._coerce(other)
        order = max(self.order or self.polynomial.degree(), other.order or other.polynomial.degree())
        poly, dropped = self.polynomial.mul_truncate(other.polynomial, order)
        p_self_range = self.polynomial.evaluate_interval(self.domain)
        p_other_range = other.polynomial.evaluate_interval(other.domain)
        trunc_range = dropped.evaluate_interval(self.domain)
        rem = (p_self_range * other.remainder) + (p_other_range * self.remainder) + (self.remainder * other.remainder) + trunc_range
        return TaylorModel(poly, rem, list(self.domain), order=order)

    __rmul__ = __mul__

    def pow_int(self, exponent: int) -> "TaylorModel":
        if exponent < 0:
            raise ValueError("TaylorModel.pow_int only supports nonnegative exponents")
        if exponent == 0:
            return TaylorModel.constant(1.0, self.domain, order=self.order)
        out = TaylorModel.constant(1.0, self.domain, order=self.order)
        for _ in range(exponent):
            out = out * self
        return out

    def range_box(self) -> Interval:
        return self.polynomial.evaluate_interval(self.domain) + self.remainder

    def evaluate_point(self, values: Iterable[Any]) -> torch.Tensor:
        return self.polynomial.evaluate_point(values) + self.remainder.mid()

    def integrate(self, var_index: int) -> "TaylorModel":
        poly = self.polynomial.integrate(var_index)
        # The current project uses local time tau in [0,h].  The interval part of
        # the integrand is integrated as tau * R and kept as an interval remainder.
        tau_dom = self.domain[var_index]
        rem = tau_dom * self.remainder
        return TaylorModel(poly, rem, list(self.domain), self.order)

    def substitute_const(self, var_index: int, value: Any) -> "TaylorModel":
        poly = self.polynomial.substitute_const(var_index, value)
        # Domain is unchanged here.  Call drop_variable afterwards when the local
        # variable should disappear from the representation.
        return TaylorModel(poly, self.remainder, list(self.domain), self.order)

    def drop_variable(self, var_index: int, *, require_zero_exponent: bool = True) -> "TaylorModel":
        poly = self.polynomial.drop_variable(var_index, require_zero_exponent=require_zero_exponent)
        new_domain = [d for i, d in enumerate(self.domain) if i != var_index]
        return TaylorModel(poly, self.remainder, new_domain, self.order)

    def extend_domain(self, new_interval: Interval) -> "TaylorModel":
        return TaylorModel(self.polynomial.extend_vars(1), self.remainder, list(self.domain) + [new_interval], self.order)

    def active_variables(self) -> set[int]:
        return self.polynomial.active_variables()

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"TaylorModel(poly={self.polynomial}, rem={self.remainder})"
