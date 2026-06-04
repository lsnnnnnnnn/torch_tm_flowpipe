"""Experimental symbolic remainder helpers.

This module is intentionally diagnostic-only.  Noise symbols are represented as
ordinary polynomial variables with domain [-1, 1], so the existing TaylorModel
arithmetic can carry recent residual sources symbolically without changing the
default TaylorModel implementation.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterable, Mapping, Sequence

from .interval import Interval
from .polynomial import Polynomial
from .taylor_model import TaylorModel
from .tm_vector import TMVector


def symbolic_noise_domain() -> Interval:
    return Interval(-1.0, 1.0)


@dataclass(frozen=True)
class SymbolicNoiseSymbol:
    """Metadata for one bounded noise variable eps_i in [-1, 1]."""

    symbol_id: int
    var_index: int
    state_dim: int
    source: str = "picard_residual"


@dataclass(frozen=True)
class SymbolicRemainderState:
    """Queue metadata for diagnostic symbolic remainders."""

    symbols: tuple[SymbolicNoiseSymbol, ...] = ()
    next_symbol_id: int = 0
    max_symbolic_remainders: int = 0

    @staticmethod
    def empty(max_symbolic_remainders: int = 0) -> "SymbolicRemainderState":
        return SymbolicRemainderState((), 0, int(max_symbolic_remainders))

    def active_var_indices(self) -> tuple[int, ...]:
        return tuple(symbol.var_index for symbol in self.symbols)

    def with_queue_size(self, max_symbolic_remainders: int) -> "SymbolicRemainderState":
        return replace(self, max_symbolic_remainders=int(max_symbolic_remainders))


def _zero_interval_like(domain: Sequence[Interval]) -> Interval:
    if domain:
        return Interval.zero(dtype=domain[0].dtype, device=domain[0].device)
    return Interval.zero()


def _interval_width(iv: Interval) -> float:
    return float(iv.width().detach().cpu())


def _zero_remainder_model(model: TaylorModel) -> TaylorModel:
    return TaylorModel(model.polynomial, _zero_interval_like(model.domain), list(model.domain), order=model.order)


def _with_polynomial(model: TaylorModel, polynomial: Polynomial) -> TaylorModel:
    return TaylorModel(polynomial, model.remainder, list(model.domain), order=model.order)


def split_polynomial_by_variables(
    polynomial: Polynomial,
    variable_indices: Iterable[int],
) -> tuple[Polynomial, Polynomial]:
    """Split a polynomial into terms independent of and dependent on variables."""

    indices = set(int(i) for i in variable_indices)
    independent: dict[tuple[int, ...], Any] = {}
    dependent: dict[tuple[int, ...], Any] = {}
    for exp, coeff in polynomial.terms.items():
        target = dependent if any(exp[i] for i in indices if i < len(exp)) else independent
        target[exp] = coeff
    return Polynomial(independent, polynomial.n_vars), Polynomial(dependent, polynomial.n_vars)


def symbolic_remainder_widths(tm: TMVector, state: SymbolicRemainderState) -> list[float]:
    indices = state.active_var_indices()
    if not indices:
        return [0.0 for _ in tm]
    widths: list[float] = []
    for model in tm:
        _plain, symbolic = split_polynomial_by_variables(model.polynomial, indices)
        widths.append(_interval_width(symbolic.evaluate_interval(model.domain)))
    return widths


def ordinary_remainder_widths(tm: TMVector) -> list[float]:
    return [_interval_width(model.remainder) for model in tm]


def _drop_one_variable(
    tm: TMVector,
    state: SymbolicRemainderState,
    var_index: int,
    *,
    remove_symbol_id: int | None = None,
) -> tuple[TMVector, SymbolicRemainderState, list[float]]:
    new_models: list[TaylorModel] = []
    materialized_widths: list[float] = []
    for model in tm:
        kept: dict[tuple[int, ...], Any] = {}
        dropped: dict[tuple[int, ...], Any] = {}
        for exp, coeff in model.polynomial.terms.items():
            if exp[var_index]:
                dropped[exp] = coeff
            else:
                kept[exp] = coeff

        dropped_poly = Polynomial(dropped, model.n_vars)
        contribution = dropped_poly.evaluate_interval(model.domain) if dropped else _zero_interval_like(model.domain)
        kept_poly = Polynomial(kept, model.n_vars).drop_variable(var_index, require_zero_exponent=True)
        new_domain = [dom for i, dom in enumerate(model.domain) if i != var_index]
        new_models.append(TaylorModel(kept_poly, model.remainder + contribution, new_domain, order=model.order))
        materialized_widths.append(_interval_width(contribution))

    new_symbols: list[SymbolicNoiseSymbol] = []
    for symbol in state.symbols:
        if remove_symbol_id is not None and symbol.symbol_id == remove_symbol_id:
            continue
        if symbol.var_index == var_index:
            continue
        if symbol.var_index > var_index:
            new_symbols.append(replace(symbol, var_index=symbol.var_index - 1))
        else:
            new_symbols.append(symbol)

    return (
        TMVector(new_models),
        replace(state, symbols=tuple(new_symbols)),
        materialized_widths,
    )


def materialize_oldest_symbols(
    tm: TMVector,
    state: SymbolicRemainderState,
    max_symbolic_remainders: int,
) -> tuple[TMVector, SymbolicRemainderState, dict[str, Any]]:
    """Materialize oldest queue entries until at most max symbols remain."""

    max_count = max(0, int(max_symbolic_remainders))
    state = state.with_queue_size(max_count)
    materialized_symbol_ids: list[int] = []
    width_sums = [0.0 for _ in tm]
    current_tm = tm
    current_state = state
    while len(current_state.symbols) > max_count:
        oldest = current_state.symbols[0]
        current_tm, current_state, widths = _drop_one_variable(
            current_tm,
            current_state,
            oldest.var_index,
            remove_symbol_id=oldest.symbol_id,
        )
        materialized_symbol_ids.append(oldest.symbol_id)
        width_sums = [a + b for a, b in zip(width_sums, widths)]

    return (
        current_tm,
        current_state,
        {
            "materialized_symbol_ids": tuple(materialized_symbol_ids),
            "materialized_remainder_widths": tuple(width_sums),
            "materialized_remainder_width_sum": sum(width_sums),
        },
    )


def materialize_non_symbolic_variables(
    tm: TMVector,
    state: SymbolicRemainderState,
) -> tuple[TMVector, SymbolicRemainderState, dict[str, Any]]:
    """Materialize all variables that are not tracked noise symbols."""

    noise_indices = set(state.active_var_indices())
    current_tm = tm
    current_state = state
    width_sums = [0.0 for _ in tm]
    dropped_indices: list[int] = []
    for var_index in sorted((i for i in range(tm.n_vars) if i not in noise_indices), reverse=True):
        current_tm, current_state, widths = _drop_one_variable(current_tm, current_state, var_index)
        dropped_indices.append(var_index)
        width_sums = [a + b for a, b in zip(width_sums, widths)]
    return (
        current_tm,
        current_state,
        {
            "materialized_variable_indices": tuple(dropped_indices),
            "materialized_remainder_widths": tuple(width_sums),
            "materialized_remainder_width_sum": sum(width_sums),
        },
    )


def materialize_all_symbols(tm: TMVector, state: SymbolicRemainderState) -> TMVector:
    materialized, _state, _stats = materialize_oldest_symbols(tm, state, 0)
    return materialized


def introduce_symbolic_remainders(
    tm: TMVector,
    state: SymbolicRemainderState | None,
    *,
    max_symbolic_remainders: int,
    source: str = "picard_residual",
) -> tuple[TMVector, SymbolicRemainderState, dict[str, Any]]:
    """Move each component's interval remainder into a fresh noise symbol."""

    if state is None:
        state = SymbolicRemainderState.empty(max_symbolic_remainders)
    else:
        state = state.with_queue_size(max_symbolic_remainders)

    remainders = [model.remainder for model in tm]
    models = [_zero_remainder_model(model) for model in tm]
    current_symbols = list(state.symbols)
    next_symbol_id = state.next_symbol_id
    introduced: list[int] = []

    for state_dim, remainder in enumerate(remainders):
        eps_index = models[0].n_vars if models else 0
        models = [model.extend_domain(symbolic_noise_domain()) for model in models]
        n_vars = models[0].n_vars if models else 0
        center = remainder.mid()
        radius = remainder.radius()
        residual_poly = (
            Polynomial.constant(center, n_vars)
            + Polynomial.variable(eps_index, n_vars, dtype=radius.dtype, device=radius.device) * radius
        )
        models[state_dim] = _with_polynomial(models[state_dim], models[state_dim].polynomial + residual_poly)
        current_symbols.append(SymbolicNoiseSymbol(next_symbol_id, eps_index, state_dim, source))
        introduced.append(next_symbol_id)
        next_symbol_id += 1

    current_tm = TMVector(models)
    current_state = SymbolicRemainderState(tuple(current_symbols), next_symbol_id, int(max_symbolic_remainders))
    current_tm, current_state, materialized_stats = materialize_oldest_symbols(
        current_tm,
        current_state,
        max_symbolic_remainders,
    )
    symbolic_width = symbolic_remainder_widths(current_tm, current_state)
    ordinary_width = ordinary_remainder_widths(current_tm)
    stats = {
        **materialized_stats,
        "introduced_symbol_ids": tuple(int(i) for i in introduced),
        "introduced_symbols": len(introduced),
        "active_noise_symbols": len(current_state.symbols),
        "symbolic_remainder_widths": tuple(symbolic_width),
        "symbolic_remainder_width_sum": sum(symbolic_width),
        "ordinary_remainder_widths": tuple(ordinary_width),
        "ordinary_remainder_width_sum": sum(ordinary_width),
    }
    return current_tm, current_state, stats


def _merge_states(a: SymbolicRemainderState, b: SymbolicRemainderState) -> SymbolicRemainderState:
    by_id = {symbol.symbol_id: symbol for symbol in a.symbols}
    for symbol in b.symbols:
        by_id.setdefault(symbol.symbol_id, symbol)
    symbols = tuple(sorted(by_id.values(), key=lambda symbol: symbol.symbol_id))
    return SymbolicRemainderState(
        symbols,
        max(a.next_symbol_id, b.next_symbol_id),
        max(a.max_symbolic_remainders, b.max_symbolic_remainders),
    )


@dataclass(frozen=True)
class SymbolicTaylorModel:
    """Small wrapper around a TaylorModel that carries noise metadata."""

    base: TaylorModel
    state: SymbolicRemainderState = SymbolicRemainderState()

    @property
    def symbolic_remainder_terms(self) -> tuple[SymbolicNoiseSymbol, ...]:
        return self.state.symbols

    @property
    def noise_domains(self) -> Mapping[int, Interval]:
        return {symbol.var_index: self.base.domain[symbol.var_index] for symbol in self.state.symbols}

    @property
    def max_symbolic_remainders(self) -> int:
        return self.state.max_symbolic_remainders

    def _coerce(self, other: Any) -> tuple[TaylorModel, SymbolicRemainderState]:
        if isinstance(other, SymbolicTaylorModel):
            return other.base, _merge_states(self.state, other.state)
        return self.base._coerce(other), self.state

    def __add__(self, other: Any) -> "SymbolicTaylorModel":
        other_base, state = self._coerce(other)
        return SymbolicTaylorModel(self.base + other_base, state)

    __radd__ = __add__

    def __sub__(self, other: Any) -> "SymbolicTaylorModel":
        other_base, state = self._coerce(other)
        return SymbolicTaylorModel(self.base - other_base, state)

    def __rsub__(self, other: Any) -> "SymbolicTaylorModel":
        other_base, state = self._coerce(other)
        return SymbolicTaylorModel(other_base - self.base, state)

    def __neg__(self) -> "SymbolicTaylorModel":
        return SymbolicTaylorModel(-self.base, self.state)

    def __mul__(self, other: Any) -> "SymbolicTaylorModel":
        other_base, state = self._coerce(other)
        return SymbolicTaylorModel(self.base * other_base, state)

    __rmul__ = __mul__

    def range_box(self) -> Interval:
        return self.base.range_box()

    def materialize(self) -> TaylorModel:
        return materialize_all_symbols(TMVector([self.base]), self.state)[0]
