"""Sparse bounded-degree multivariate polynomials over torch scalar tensors."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Tuple

import torch

from .interval import Interval

Exponent = Tuple[int, ...]


def _coef(x: Any, *, like: torch.Tensor | None = None) -> torch.Tensor:
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


def _zero_like_from_terms(terms: Mapping[Exponent, torch.Tensor] | None = None) -> torch.Tensor:
    if terms:
        first = next(iter(terms.values()))
        return torch.zeros_like(first)
    return torch.zeros((), dtype=torch.float64)


def _is_zero_tensor(x: torch.Tensor) -> bool:
    return bool(torch.all(x == 0))


@dataclass(frozen=True)
class Polynomial:
    """Sparse polynomial represented by ``{exponent_tuple: coefficient}``.

    Exponents always have length ``n_vars``.  Coefficients are torch tensors,
    typically scalar float64 tensors.
    """

    terms: Dict[Exponent, torch.Tensor]
    n_vars: int

    def __init__(self, terms: Mapping[Exponent, Any] | None = None, n_vars: int | None = None):
        terms = dict(terms or {})
        if n_vars is None:
            if terms:
                n_vars = len(next(iter(terms.keys())))
            else:
                n_vars = 0
        clean: Dict[Exponent, torch.Tensor] = {}
        for exp, c in terms.items():
            exp_t = tuple(int(e) for e in exp)
            if len(exp_t) != n_vars:
                raise ValueError(f"exponent length {len(exp_t)} != n_vars {n_vars}")
            if any(e < 0 for e in exp_t):
                raise ValueError(f"negative exponent in {exp_t}")
            c_t = _coef(c)
            if not _is_zero_tensor(c_t):
                clean[exp_t] = clean.get(exp_t, torch.zeros_like(c_t)) + c_t
        clean = {e: c for e, c in clean.items() if not _is_zero_tensor(c)}
        object.__setattr__(self, "terms", clean)
        object.__setattr__(self, "n_vars", int(n_vars))

    @staticmethod
    def zero(n_vars: int, *, dtype: torch.dtype = torch.float64, device: torch.device | str | None = None) -> "Polynomial":
        return Polynomial({}, n_vars=n_vars)

    @staticmethod
    def constant(value: Any, n_vars: int) -> "Polynomial":
        c = _coef(value)
        if _is_zero_tensor(c):
            return Polynomial.zero(n_vars, dtype=c.dtype, device=c.device)
        return Polynomial({(0,) * n_vars: c}, n_vars=n_vars)

    @staticmethod
    def variable(index: int, n_vars: int, *, dtype: torch.dtype = torch.float64, device: torch.device | str | None = None) -> "Polynomial":
        if index < 0 or index >= n_vars:
            raise IndexError(index)
        exp = [0] * n_vars
        exp[index] = 1
        return Polynomial({tuple(exp): torch.ones((), dtype=dtype, device=device)}, n_vars=n_vars)

    def clone(self) -> "Polynomial":
        return Polynomial({e: c.clone() for e, c in self.terms.items()}, self.n_vars)

    @property
    def dtype(self) -> torch.dtype:
        return next(iter(self.terms.values())).dtype if self.terms else torch.float64

    @property
    def device(self) -> torch.device:
        return next(iter(self.terms.values())).device if self.terms else torch.device("cpu")

    def degree(self) -> int:
        return max((sum(e) for e in self.terms), default=0)

    def _coerce(self, other: Any) -> "Polynomial":
        if isinstance(other, Polynomial):
            if other.n_vars != self.n_vars:
                raise ValueError(f"n_vars mismatch {self.n_vars} != {other.n_vars}")
            return other
        return Polynomial.constant(other, self.n_vars)

    def __add__(self, other: Any) -> "Polynomial":
        other = self._coerce(other)
        out = {e: c.clone() for e, c in self.terms.items()}
        for e, c in other.terms.items():
            out[e] = out.get(e, torch.zeros_like(c)) + c
        return Polynomial(out, self.n_vars)

    __radd__ = __add__

    def __sub__(self, other: Any) -> "Polynomial":
        other = self._coerce(other)
        return self + (-other)

    def __rsub__(self, other: Any) -> "Polynomial":
        return self._coerce(other) - self

    def __neg__(self) -> "Polynomial":
        return Polynomial({e: -c for e, c in self.terms.items()}, self.n_vars)

    def __mul__(self, other: Any) -> "Polynomial":
        other = self._coerce(other)
        out: Dict[Exponent, torch.Tensor] = {}
        for e1, c1 in self.terms.items():
            for e2, c2 in other.terms.items():
                exp = tuple(a + b for a, b in zip(e1, e2))
                val = c1 * c2
                out[exp] = out.get(exp, torch.zeros_like(val)) + val
        return Polynomial(out, self.n_vars)

    __rmul__ = __mul__

    def mul_truncate(self, other: Any, order: int) -> tuple["Polynomial", "Polynomial"]:
        other = self._coerce(other)
        kept: Dict[Exponent, torch.Tensor] = {}
        dropped: Dict[Exponent, torch.Tensor] = {}
        for e1, c1 in self.terms.items():
            for e2, c2 in other.terms.items():
                exp = tuple(a + b for a, b in zip(e1, e2))
                val = c1 * c2
                target = kept if sum(exp) <= order else dropped
                target[exp] = target.get(exp, torch.zeros_like(val)) + val
        return Polynomial(kept, self.n_vars), Polynomial(dropped, self.n_vars)

    def truncate(self, order: int) -> tuple["Polynomial", "Polynomial"]:
        kept = {e: c for e, c in self.terms.items() if sum(e) <= order}
        dropped = {e: c for e, c in self.terms.items() if sum(e) > order}
        return Polynomial(kept, self.n_vars), Polynomial(dropped, self.n_vars)

    def pow_int(self, exponent: int, *, order: int | None = None) -> tuple["Polynomial", "Polynomial"] | "Polynomial":
        if exponent < 0:
            raise ValueError("Polynomial.pow_int only supports nonnegative exponents")
        result = Polynomial.constant(1.0, self.n_vars)
        dropped_total = Polynomial.zero(self.n_vars)
        base = self
        n = exponent
        # For the small degrees in this prototype, repeated multiplication is clearer.
        for _ in range(exponent):
            if order is None:
                result = result * self
            else:
                result, dropped = result.mul_truncate(self, order)
                dropped_total = dropped_total + dropped
        return (result, dropped_total) if order is not None else result

    def integrate(self, var_index: int) -> "Polynomial":
        if var_index < 0 or var_index >= self.n_vars:
            raise IndexError(var_index)
        out: Dict[Exponent, torch.Tensor] = {}
        for exp, c in self.terms.items():
            exp_l = list(exp)
            exp_l[var_index] += 1
            denom = torch.as_tensor(exp_l[var_index], dtype=c.dtype, device=c.device)
            out[tuple(exp_l)] = c / denom
        return Polynomial(out, self.n_vars)

    def derivative(self, var_index: int) -> "Polynomial":
        if var_index < 0 or var_index >= self.n_vars:
            raise IndexError(var_index)
        out: Dict[Exponent, torch.Tensor] = {}
        for exp, c in self.terms.items():
            if exp[var_index] == 0:
                continue
            exp_l = list(exp)
            power = exp_l[var_index]
            exp_l[var_index] -= 1
            out[tuple(exp_l)] = c * power
        return Polynomial(out, self.n_vars)

    def evaluate_interval(self, domain: Iterable[Interval]) -> Interval:
        domain_l = list(domain)
        if len(domain_l) != self.n_vars:
            raise ValueError(f"domain length {len(domain_l)} != n_vars {self.n_vars}")
        if not self.terms:
            return Interval.zero()
        acc = Interval.zero(dtype=self.dtype, device=self.device)
        for exp, c in self.terms.items():
            term_iv = Interval.point(c)
            for power, dom in zip(exp, domain_l):
                if power:
                    term_iv = term_iv * dom.pow_int(power)
            acc = acc + term_iv
        return acc

    def evaluate_point(self, values: Iterable[Any]) -> torch.Tensor:
        vals = [_coef(v) for v in values]
        if len(vals) != self.n_vars:
            raise ValueError(f"values length {len(vals)} != n_vars {self.n_vars}")
        total = torch.zeros_like(next(iter(self.terms.values()))) if self.terms else torch.zeros((), dtype=torch.float64)
        for exp, c in self.terms.items():
            term = c
            for power, val in zip(exp, vals):
                if power:
                    term = term * val.pow(power)
            total = total + term
        return total

    def substitute_const(self, var_index: int, value: Any) -> "Polynomial":
        if var_index < 0 or var_index >= self.n_vars:
            raise IndexError(var_index)
        v = _coef(value, like=next(iter(self.terms.values())) if self.terms else None)
        out: Dict[Exponent, torch.Tensor] = {}
        for exp, c in self.terms.items():
            power = exp[var_index]
            new_exp = list(exp)
            new_exp[var_index] = 0
            val = c * (v.pow(power) if power else torch.ones_like(c))
            new_exp_t = tuple(new_exp)
            out[new_exp_t] = out.get(new_exp_t, torch.zeros_like(val)) + val
        return Polynomial(out, self.n_vars)

    def drop_variable(self, var_index: int, *, require_zero_exponent: bool = True) -> "Polynomial":
        if var_index < 0 or var_index >= self.n_vars:
            raise IndexError(var_index)
        out: Dict[Exponent, torch.Tensor] = {}
        for exp, c in self.terms.items():
            if require_zero_exponent and exp[var_index] != 0:
                raise ValueError(f"cannot drop active variable {var_index}; exponent {exp[var_index]} in term {exp}")
            new_exp = tuple(e for i, e in enumerate(exp) if i != var_index)
            out[new_exp] = out.get(new_exp, torch.zeros_like(c)) + c
        return Polynomial(out, self.n_vars - 1)

    def extend_vars(self, n_new: int = 1) -> "Polynomial":
        if n_new < 0:
            raise ValueError("n_new must be nonnegative")
        if n_new == 0:
            return self
        return Polynomial({e + (0,) * n_new: c for e, c in self.terms.items()}, self.n_vars + n_new)

    def active_variables(self) -> set[int]:
        active: set[int] = set()
        for exp in self.terms:
            for i, power in enumerate(exp):
                if power:
                    active.add(i)
        return active

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Polynomial(n_vars={self.n_vars}, terms={self.terms})"
