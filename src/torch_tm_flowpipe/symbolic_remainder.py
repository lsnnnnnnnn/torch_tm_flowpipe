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


IntervalColumn = tuple[Interval, ...]
RealMatrix = tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class FlowstarSymbolicRemainderQueue:
    """Clean-room skeleton of Flow*'s J/Phi_L/scalars remainder queue.

    This is separate from ``SymbolicRemainderState`` above. Flow* does not add
    ordinary polynomial noise variables for each residual; it keeps a queue of
    interval remainder columns and propagates older columns through the linear
    part of each accepted reset map.
    """

    J: tuple[IntervalColumn, ...]
    Phi_L: tuple[RealMatrix, ...]
    scalars: tuple[float, ...]
    max_size: int

    @staticmethod
    def empty(dim: int, max_size: int = 100) -> "FlowstarSymbolicRemainderQueue":
        return FlowstarSymbolicRemainderQueue((), (), tuple(1.0 for _ in range(int(dim))), int(max_size))

    @property
    def dim(self) -> int:
        return len(self.scalars)

    def reset(self, dim: int | None = None) -> "FlowstarSymbolicRemainderQueue":
        return FlowstarSymbolicRemainderQueue.empty(self.dim if dim is None else int(dim), self.max_size)


def _zero_interval_like(domain: Sequence[Interval]) -> Interval:
    if domain:
        return Interval.zero(dtype=domain[0].dtype, device=domain[0].device)
    return Interval.zero()


def _zero_interval_like_interval(iv: Interval) -> Interval:
    return Interval.zero(dtype=iv.dtype, device=iv.device)


def _linear_coefficients(tm: TMVector) -> RealMatrix:
    dim = len(tm)
    rows: list[list[float]] = []
    for model in tm:
        row = [0.0 for _ in range(dim)]
        for exp, coeff in model.polynomial.terms.items():
            if sum(exp) != 1:
                continue
            for var_index in range(min(dim, len(exp))):
                if exp[var_index] == 1 and all(power == 0 for j, power in enumerate(exp) if j != var_index):
                    row[var_index] = float(coeff.detach().cpu())
                    break
        rows.append(row)
    return tuple(tuple(row) for row in rows)


def _right_scale_matrix(matrix: RealMatrix, scalars: Sequence[float]) -> RealMatrix:
    return tuple(tuple(value * float(scalars[j]) for j, value in enumerate(row)) for row in matrix)


def _matmul_real(a: RealMatrix, b: RealMatrix) -> RealMatrix:
    if not a:
        return ()
    cols = len(b[0]) if b else 0
    out: list[tuple[float, ...]] = []
    for row in a:
        out_row = []
        for col in range(cols):
            out_row.append(sum(float(row[k]) * float(b[k][col]) for k in range(len(b))))
        out.append(tuple(out_row))
    return tuple(out)


def _matmul_interval_col(matrix: RealMatrix, column: IntervalColumn, reference: Interval) -> IntervalColumn:
    out: list[Interval] = []
    for row in matrix:
        acc = _zero_interval_like_interval(reference)
        for scalar, iv in zip(row, column):
            if scalar:
                acc = acc + iv * float(scalar)
        out.append(acc)
    return tuple(out)


def _add_interval_columns(a: IntervalColumn, b: IntervalColumn) -> IntervalColumn:
    return tuple(x + y for x, y in zip(a, b))


def _column_width_sum(column: IntervalColumn) -> float:
    return sum(_interval_width(iv) for iv in column)


def _updated_phi_and_propagated_remainder(
    state: FlowstarSymbolicRemainderQueue,
    phi_l_i: RealMatrix,
    reference: Interval,
) -> tuple[tuple[RealMatrix, ...], IntervalColumn]:
    updated_phi = list(state.Phi_L)
    for i in range(1, len(updated_phi)):
        updated_phi[i] = _matmul_real(phi_l_i, updated_phi[i])
    updated_phi.append(phi_l_i)

    propagated = tuple(_zero_interval_like_interval(reference) for _ in range(state.dim))
    for i in range(1, len(updated_phi)):
        if i - 1 >= len(state.J):
            break
        propagated = _add_interval_columns(propagated, _matmul_interval_col(updated_phi[i], state.J[i - 1], reference))
    return tuple(updated_phi), propagated


def flowstar_symbolic_remainder_queue_reset(
    tm: TMVector,
    state: FlowstarSymbolicRemainderQueue | None,
    *,
    max_size: int = 100,
) -> tuple[TMVector, FlowstarSymbolicRemainderQueue, dict[str, Any]]:
    """Propagate endpoint remainders through a Flow*-style linear queue.

    The helper is intentionally conservative: it preserves the endpoint
    polynomial dependency and replaces the ordinary remainder by the current
    endpoint remainder plus the queued linear propagation of older remainders.
    """

    dim = len(tm)
    if dim == 0:
        empty = FlowstarSymbolicRemainderQueue.empty(0, max_size)
        return tm, empty, {"active_queue_size": 0, "queue_reset": False}
    if state is None or state.dim != dim or int(state.max_size) != int(max_size):
        state = FlowstarSymbolicRemainderQueue.empty(dim, max_size)

    reference = tm[0].remainder
    linear = _linear_coefficients(tm)
    phi_l_i = _right_scale_matrix(linear, state.scalars)
    updated_phi, propagated = _updated_phi_and_propagated_remainder(state, phi_l_i, reference)
    current_j = tuple(model.remainder for model in tm)
    total_remainders = _add_interval_columns(current_j, propagated)
    reset_tm = TMVector(model.with_remainder(rem) for model, rem in zip(tm, total_remainders))

    widths = reset_tm.range_box()
    scalars: list[float] = []
    for box in widths:
        mag = max(abs(float(box.lo.detach().cpu())), abs(float(box.hi.detach().cpu())))
        scalars.append(0.0 if mag == 0 else 1.0 / mag)

    new_j = state.J + (current_j,)
    queue_reset = bool(int(max_size) > 0 and len(new_j) >= int(max_size))
    if queue_reset:
        new_state = FlowstarSymbolicRemainderQueue.empty(dim, max_size)
    else:
        new_state = FlowstarSymbolicRemainderQueue(tuple(new_j), updated_phi, tuple(scalars), int(max_size))

    stats = {
        "queue_size_before": len(state.J),
        "queue_size_after": len(new_state.J),
        "queue_reset": queue_reset,
        "current_remainder_width_sum": _column_width_sum(current_j),
        "propagated_remainder_width_sum": _column_width_sum(propagated),
        "total_remainder_width_sum": _column_width_sum(total_remainders),
        "linear_map_abs_sum": sum(abs(v) for row in linear for v in row),
    }
    return reset_tm, new_state, stats


def flowstar_normalized_insertion_symbolic_queue_reset(
    inserted_endpoint: TMVector,
    reset_tm: TMVector,
    state: FlowstarSymbolicRemainderQueue | None,
    *,
    scales: Sequence[float],
    max_size: int = 100,
    materialize_propagated_on_reset: bool = True,
) -> tuple[TMVector, FlowstarSymbolicRemainderQueue, dict[str, Any]]:
    """Conservative symbolic queue update after normalized insertion.

    ``inserted_endpoint`` is the accepted nonconstant endpoint after Flow*-style
    normal insertion. ``reset_tm`` is the fresh normalized local initial set for
    the next step. The helper propagates older queued interval columns through
    the linear part of ``inserted_endpoint``. Existing symqueue mode materializes
    that propagated width on ``reset_tm``; split mode returns a clean ordinary
    reset and exposes the propagated width for output/range materialization.
    """

    dim = len(reset_tm)
    if dim == 0:
        empty = FlowstarSymbolicRemainderQueue.empty(0, max_size)
        return reset_tm, empty, {"queue_size": 0, "queue_reset": False}
    if state is None or state.dim != dim or int(state.max_size) != int(max_size):
        state = FlowstarSymbolicRemainderQueue.empty(dim, max_size)

    reference = reset_tm[0].remainder
    linear = _linear_coefficients(inserted_endpoint)
    phi_l_i = _right_scale_matrix(linear, state.scalars)
    updated_phi, propagated = _updated_phi_and_propagated_remainder(state, phi_l_i, reference)
    current_j = tuple(model.remainder for model in inserted_endpoint)

    if materialize_propagated_on_reset:
        reset_with_queue = TMVector(
            model.with_remainder(model.remainder + propagated_i)
            for model, propagated_i in zip(reset_tm, propagated)
        )
        output_symbolic_remainders = tuple(_zero_interval_like_interval(reference) for _ in range(dim))
    else:
        reset_with_queue = reset_tm
        output_symbolic_remainders = propagated

    new_j = state.J + (current_j,)
    queue_reset = bool(int(max_size) > 0 and len(new_j) >= int(max_size))
    if queue_reset:
        new_state = FlowstarSymbolicRemainderQueue.empty(dim, max_size)
    else:
        scalar_tuple = tuple(float(s) for s in scales[:dim])
        if len(scalar_tuple) < dim:
            scalar_tuple = scalar_tuple + tuple(1.0 for _ in range(dim - len(scalar_tuple)))
        new_state = FlowstarSymbolicRemainderQueue(tuple(new_j), updated_phi, scalar_tuple, int(max_size))

    propagated_widths = [_interval_width(iv) for iv in propagated]
    new_widths = [_interval_width(iv) for iv in current_j]
    ordinary_remainder_widths = [_interval_width(model.remainder) for model in reset_tm]
    materialized_widths = (
        [_interval_width(model.remainder) for model in reset_with_queue]
        if materialize_propagated_on_reset
        else propagated_widths
    )
    ordinary_only_range_width = sum(_interval_width(iv) for iv in reset_tm.range_box())
    total_with_symbolic_tm = TMVector(
        model.with_remainder(model.remainder + rem)
        for model, rem in zip(reset_with_queue, output_symbolic_remainders)
    )
    total_range_width_with_symbolic = sum(_interval_width(iv) for iv in total_with_symbolic_tm.range_box())
    linear_norm = sum(abs(v) for row in linear for v in row)
    stats = {
        "queue_size_before": len(state.J),
        "queue_size_after": len(new_state.J),
        "queue_size": len(new_state.J),
        "queue_reset": queue_reset,
        "semantic_split": not materialize_propagated_on_reset,
        "propagated_symbolic_width_x": propagated_widths[0] if len(propagated_widths) > 0 else "",
        "propagated_symbolic_width_y": propagated_widths[1] if len(propagated_widths) > 1 else "",
        "propagated_symbolic_width_sum": sum(propagated_widths),
        "new_symbolic_width_x": new_widths[0] if len(new_widths) > 0 else "",
        "new_symbolic_width_y": new_widths[1] if len(new_widths) > 1 else "",
        "new_symbolic_width_sum": sum(new_widths),
        "materialized_width_x": materialized_widths[0] if len(materialized_widths) > 0 else "",
        "materialized_width_y": materialized_widths[1] if len(materialized_widths) > 1 else "",
        "materialized_width_sum": sum(materialized_widths),
        "materialized_for_output_width": sum(materialized_widths),
        "ordinary_only_range_width": ordinary_only_range_width,
        "symbolic_contribution_width": sum(propagated_widths),
        "total_range_width_with_symbolic": total_range_width_with_symbolic,
        "target_checked_width": sum(ordinary_remainder_widths),
        "linear_map_norm": linear_norm,
        "linear_map_abs_sum": linear_norm,
        "scalars": ";".join(f"{float(s):.17g}" for s in scales[:dim]),
        "_symbolic_output_remainders": output_symbolic_remainders,
        "approximation": (
            "limited_normalized_insertion_symqueue_split; current insertion uncertainty "
            "is queued for future propagation and propagated old queue width is "
            "materialized for output/range only"
            if not materialize_propagated_on_reset
            else
            "limited_normalized_insertion_symqueue; current insertion uncertainty "
            "is queued for future propagation and propagated old queue width is "
            "materialized on the next normalized reset"
        ),
    }
    return reset_with_queue, new_state, stats


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
