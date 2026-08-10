"""Generic affine common-basis transforms for batched Taylor polynomials.

The transform is diagnostic infrastructure: it makes coordinate identity,
normalization, variable order, and physical local time explicit before any
cross-tool coefficient comparison is attempted.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import comb
from typing import Iterable, Sequence

import torch


def _as_float64(value: torch.Tensor | Sequence[Sequence[float]]) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float64)
    if tensor.ndim != 2:
        raise ValueError("basis tensors must have shape [batch, variables]")
    return tensor


def _down(value: torch.Tensor) -> torch.Tensor:
    return torch.nextafter(value, torch.full_like(value, -torch.inf))


def _up(value: torch.Tensor) -> torch.Tensor:
    return torch.nextafter(value, torch.full_like(value, torch.inf))


def _interval_add(
    left_lo: torch.Tensor,
    left_hi: torch.Tensor,
    right_lo: torch.Tensor,
    right_hi: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _down(left_lo + right_lo), _up(left_hi + right_hi)


def _interval_mul(
    left_lo: torch.Tensor,
    left_hi: torch.Tensor,
    right_lo: torch.Tensor,
    right_hi: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    products = torch.stack(
        (
            left_lo * right_lo,
            left_lo * right_hi,
            left_hi * right_lo,
            left_hi * right_hi,
        ),
        dim=0,
    )
    return _down(torch.amin(products, dim=0)), _up(torch.amax(products, dim=0))


@dataclass(frozen=True)
class AffineCoordinateBasis:
    """Physical meaning of normalized polynomial variables.

    For variable ``i``, ``physical = center[:, i] + scale[:, i] * u_i`` and
    ``u_i`` ranges over ``[domain_lo[:, i], domain_hi[:, i]]``.
    """

    names: tuple[str, ...]
    center: torch.Tensor
    scale: torch.Tensor
    domain_lo: torch.Tensor
    domain_hi: torch.Tensor
    time_name: str | None = None
    time_semantics: str | None = None

    def __post_init__(self) -> None:
        center = _as_float64(self.center)
        scale = _as_float64(self.scale)
        domain_lo = _as_float64(self.domain_lo)
        domain_hi = _as_float64(self.domain_hi)
        shape = center.shape
        if not self.names or len(set(self.names)) != len(self.names):
            raise ValueError("coordinate names must be nonempty and unique")
        if shape[1] != len(self.names):
            raise ValueError("basis names and tensor variable dimension disagree")
        if scale.shape != shape or domain_lo.shape != shape or domain_hi.shape != shape:
            raise ValueError("all basis tensors must have the same shape")
        if not bool(torch.all(torch.isfinite(center)) and torch.all(torch.isfinite(scale))):
            raise ValueError("basis center and scale must be finite")
        if not bool(torch.all(torch.isfinite(domain_lo)) and torch.all(torch.isfinite(domain_hi))):
            raise ValueError("basis domains must be finite")
        if not bool(torch.all(domain_lo <= domain_hi)):
            raise ValueError("basis domain is reversed")
        if self.time_name is not None and self.time_name not in self.names:
            raise ValueError("time_name is not a basis coordinate")
        if (self.time_name is None) != (self.time_semantics is None):
            raise ValueError("time_name and time_semantics must be specified together")
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "domain_lo", domain_lo)
        object.__setattr__(self, "domain_hi", domain_hi)

    @property
    def batch(self) -> int:
        return int(self.center.shape[0])

    @property
    def variables(self) -> int:
        return len(self.names)

    def physical_domain(self) -> tuple[torch.Tensor, torch.Tensor]:
        first = self.center + self.scale * self.domain_lo
        second = self.center + self.scale * self.domain_hi
        return _down(torch.minimum(first, second)), _up(torch.maximum(first, second))


@dataclass(frozen=True)
class IntervalPolynomialBatch:
    """Sparse batched/state polynomial with interval coefficients."""

    coeff_lo: torch.Tensor
    coeff_hi: torch.Tensor
    exponents: torch.Tensor

    def __post_init__(self) -> None:
        coeff_lo = torch.as_tensor(self.coeff_lo, dtype=torch.float64)
        coeff_hi = torch.as_tensor(self.coeff_hi, dtype=torch.float64)
        exponents = torch.as_tensor(self.exponents, dtype=torch.int64)
        if coeff_lo.ndim != 3 or coeff_hi.shape != coeff_lo.shape:
            raise ValueError("coefficients must share shape [batch, state, support]")
        if exponents.ndim != 2 or exponents.shape[0] != coeff_lo.shape[2]:
            raise ValueError("exponents must have shape [support, variables]")
        if not bool(torch.all(coeff_lo <= coeff_hi)):
            raise ValueError("coefficient interval is reversed")
        if bool(torch.any(exponents < 0)):
            raise ValueError("polynomial exponents must be nonnegative")
        support = [tuple(int(item) for item in row) for row in exponents.tolist()]
        if len(set(support)) != len(support):
            raise ValueError("polynomial support contains duplicate exponents")
        object.__setattr__(self, "coeff_lo", coeff_lo)
        object.__setattr__(self, "coeff_hi", coeff_hi)
        object.__setattr__(self, "exponents", exponents)

    @classmethod
    def from_point_coefficients(
        cls,
        coefficients: torch.Tensor,
        exponents: torch.Tensor | Sequence[Sequence[int]],
    ) -> "IntervalPolynomialBatch":
        values = torch.as_tensor(coefficients, dtype=torch.float64)
        return cls(values, values.clone(), torch.as_tensor(exponents, dtype=torch.int64))

    @property
    def batch(self) -> int:
        return int(self.coeff_lo.shape[0])

    @property
    def states(self) -> int:
        return int(self.coeff_lo.shape[1])

    @property
    def variables(self) -> int:
        return int(self.exponents.shape[1])

    @property
    def support(self) -> tuple[tuple[int, ...], ...]:
        return tuple(tuple(int(item) for item in row) for row in self.exponents.tolist())


@dataclass(frozen=True)
class CommonBasisTransformResult:
    transformed: IntervalPolynomialBatch
    retained: IntervalPolynomialBatch
    intervalized_discarded_lo: torch.Tensor
    intervalized_discarded_hi: torch.Tensor
    transformed_range_lo: torch.Tensor
    transformed_range_hi: torch.Tensor
    retained_range_lo: torch.Tensor
    retained_range_hi: torch.Tensor
    dropped_zero_variables: tuple[str, ...]
    coordinate_identity_known: bool
    time_treatment: str


def _broadcast_basis(basis: AffineCoordinateBasis, batch: int) -> AffineCoordinateBasis:
    if basis.batch == batch:
        return basis
    if basis.batch != 1:
        raise ValueError("basis batch dimension cannot broadcast to polynomial batch")
    return AffineCoordinateBasis(
        basis.names,
        basis.center.expand(batch, -1),
        basis.scale.expand(batch, -1),
        basis.domain_lo.expand(batch, -1),
        basis.domain_hi.expand(batch, -1),
        basis.time_name,
        basis.time_semantics,
    )


def _natural_range(
    polynomial: IntervalPolynomialBatch,
    domain_lo: torch.Tensor,
    domain_hi: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    domain_lo = torch.as_tensor(domain_lo, dtype=torch.float64)
    domain_hi = torch.as_tensor(domain_hi, dtype=torch.float64)
    if domain_lo.shape != (polynomial.batch, polynomial.variables) or domain_hi.shape != domain_lo.shape:
        raise ValueError("range domain shape does not match polynomial")
    total_lo = torch.zeros((polynomial.batch, polynomial.states), dtype=torch.float64)
    total_hi = torch.zeros_like(total_lo)
    for slot, exponent in enumerate(polynomial.support):
        monomial_lo = torch.ones((polynomial.batch, 1), dtype=torch.float64)
        monomial_hi = torch.ones_like(monomial_lo)
        for variable, power in enumerate(exponent):
            if power == 0:
                continue
            lo = domain_lo[:, variable : variable + 1]
            hi = domain_hi[:, variable : variable + 1]
            if power % 2 == 0:
                values = torch.stack((lo**power, hi**power), dim=0)
                factor_lo = torch.where((lo <= 0) & (hi >= 0), torch.zeros_like(lo), torch.amin(values, dim=0))
                factor_hi = torch.amax(values, dim=0)
            else:
                factor_lo, factor_hi = lo**power, hi**power
            monomial_lo, monomial_hi = _interval_mul(
                monomial_lo, monomial_hi, _down(factor_lo), _up(factor_hi)
            )
        term_lo, term_hi = _interval_mul(
            polynomial.coeff_lo[:, :, slot],
            polynomial.coeff_hi[:, :, slot],
            monomial_lo,
            monomial_hi,
        )
        total_lo, total_hi = _interval_add(total_lo, total_hi, term_lo, term_hi)
    return total_lo, total_hi


def _empty_polynomial(batch: int, states: int, variables: int) -> IntervalPolynomialBatch:
    return IntervalPolynomialBatch(
        torch.zeros((batch, states, 0), dtype=torch.float64),
        torch.zeros((batch, states, 0), dtype=torch.float64),
        torch.empty((0, variables), dtype=torch.int64),
    )


def affine_common_basis_transform(
    polynomial: IntervalPolynomialBatch,
    source: AffineCoordinateBasis,
    target: AffineCoordinateBasis,
    *,
    retain_support: Iterable[Sequence[int]] | None = None,
) -> CommonBasisTransformResult:
    """Rewrite ``polynomial`` from ``source`` coordinates into ``target``.

    Missing source variables may be dropped only after proving that every
    input monomial has exponent zero in that variable. This is the fail-closed
    guard against comparing coefficients with unknown identity.
    """
    if polynomial.variables != source.variables:
        raise ValueError("source basis and polynomial variable dimensions disagree")
    source = _broadcast_basis(source, polynomial.batch)
    target = _broadcast_basis(target, polynomial.batch)
    if source.time_name is not None:
        if target.time_name != source.time_name or target.time_semantics != source.time_semantics:
            raise ValueError("source and target local-time identity/semantics differ")

    target_index = {name: index for index, name in enumerate(target.names)}
    dropped: list[str] = []
    source_routes: list[tuple[int | None, torch.Tensor | None, torch.Tensor | None]] = []
    for source_index, name in enumerate(source.names):
        if name not in target_index:
            if bool(torch.any(polynomial.exponents[:, source_index] != 0)):
                raise ValueError(f"coordinate identity is unknown for nonconstant source variable {name!r}")
            dropped.append(name)
            source_routes.append((None, None, None))
            continue
        target_i = target_index[name]
        scale = source.scale[:, source_index]
        if bool(torch.any(scale == 0)):
            if bool(torch.any(polynomial.exponents[:, source_index] != 0)):
                raise ValueError(f"cannot invert zero physical scale for source variable {name!r}")
            dropped.append(name)
            source_routes.append((None, None, None))
            continue
        offset = (target.center[:, target_i] - source.center[:, source_index]) / scale
        multiplier = target.scale[:, target_i] / scale
        source_routes.append((target_i, offset, multiplier))

    accumulated: dict[tuple[int, ...], tuple[torch.Tensor, torch.Tensor]] = {}
    for slot, source_exponent in enumerate(polynomial.support):
        expansion: dict[tuple[int, ...], torch.Tensor] = {
            (0,) * target.variables: torch.ones(polynomial.batch, dtype=torch.float64)
        }
        for source_i, power in enumerate(source_exponent):
            if power == 0:
                continue
            target_i, offset, multiplier = source_routes[source_i]
            if target_i is None or offset is None or multiplier is None:
                raise ValueError("nonconstant source variable was dropped")
            next_expansion: dict[tuple[int, ...], torch.Tensor] = {}
            for exponent, factor in expansion.items():
                for target_power in range(power + 1):
                    output_exponent = list(exponent)
                    output_exponent[target_i] += target_power
                    output_key = tuple(output_exponent)
                    route_factor = (
                        float(comb(power, target_power))
                        * offset ** (power - target_power)
                        * multiplier**target_power
                    )
                    value = factor * route_factor
                    next_expansion[output_key] = next_expansion.get(
                        output_key, torch.zeros_like(value)
                    ) + value
            expansion = next_expansion
        for output_exponent, factor in expansion.items():
            factor_2d = factor[:, None]
            term_lo, term_hi = _interval_mul(
                polynomial.coeff_lo[:, :, slot],
                polynomial.coeff_hi[:, :, slot],
                factor_2d,
                factor_2d,
            )
            if output_exponent in accumulated:
                previous_lo, previous_hi = accumulated[output_exponent]
                accumulated[output_exponent] = _interval_add(previous_lo, previous_hi, term_lo, term_hi)
            else:
                accumulated[output_exponent] = (term_lo, term_hi)

    ordered = sorted(accumulated)
    if ordered:
        transformed = IntervalPolynomialBatch(
            torch.stack([accumulated[item][0] for item in ordered], dim=2),
            torch.stack([accumulated[item][1] for item in ordered], dim=2),
            torch.tensor(ordered, dtype=torch.int64),
        )
    else:
        transformed = _empty_polynomial(polynomial.batch, polynomial.states, target.variables)

    retain_set = None if retain_support is None else {
        tuple(int(value) for value in exponent) for exponent in retain_support
    }
    if retain_set is not None and any(len(item) != target.variables for item in retain_set):
        raise ValueError("retained support exponent dimension differs from target basis")
    retained_slots = [
        slot for slot, exponent in enumerate(transformed.support)
        if retain_set is None or exponent in retain_set
    ]
    discarded_slots = [slot for slot in range(len(transformed.support)) if slot not in retained_slots]
    retained = (
        IntervalPolynomialBatch(
            transformed.coeff_lo[:, :, retained_slots],
            transformed.coeff_hi[:, :, retained_slots],
            transformed.exponents[retained_slots],
        )
        if retained_slots
        else _empty_polynomial(polynomial.batch, polynomial.states, target.variables)
    )
    discarded = (
        IntervalPolynomialBatch(
            transformed.coeff_lo[:, :, discarded_slots],
            transformed.coeff_hi[:, :, discarded_slots],
            transformed.exponents[discarded_slots],
        )
        if discarded_slots
        else _empty_polynomial(polynomial.batch, polynomial.states, target.variables)
    )
    transformed_lo, transformed_hi = _natural_range(transformed, target.domain_lo, target.domain_hi)
    retained_lo, retained_hi = _natural_range(retained, target.domain_lo, target.domain_hi)
    discarded_lo, discarded_hi = _natural_range(discarded, target.domain_lo, target.domain_hi)
    return CommonBasisTransformResult(
        transformed=transformed,
        retained=retained,
        intervalized_discarded_lo=discarded_lo,
        intervalized_discarded_hi=discarded_hi,
        transformed_range_lo=transformed_lo,
        transformed_range_hi=transformed_hi,
        retained_range_lo=retained_lo,
        retained_range_hi=retained_hi,
        dropped_zero_variables=tuple(dropped),
        coordinate_identity_known=True,
        time_treatment=(
            f"preserved {source.time_name}: {source.time_semantics}"
            if source.time_name is not None
            else "no distinguished time variable"
        ),
    )


def evaluate_point(
    polynomial: IntervalPolynomialBatch,
    point: torch.Tensor | Sequence[Sequence[float]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate an interval-coefficient polynomial at batched point values."""
    point = _as_float64(point)
    if point.shape != (polynomial.batch, polynomial.variables):
        raise ValueError("point shape does not match polynomial batch/variables")
    lo = torch.zeros((polynomial.batch, polynomial.states), dtype=torch.float64)
    hi = torch.zeros_like(lo)
    for slot, exponent in enumerate(polynomial.support):
        factor = torch.ones((polynomial.batch, 1), dtype=torch.float64)
        for variable, power in enumerate(exponent):
            factor = factor * point[:, variable : variable + 1] ** power
        term_lo, term_hi = _interval_mul(
            polynomial.coeff_lo[:, :, slot], polynomial.coeff_hi[:, :, slot], factor, factor
        )
        lo, hi = _interval_add(lo, hi, term_lo, term_hi)
    return lo, hi
