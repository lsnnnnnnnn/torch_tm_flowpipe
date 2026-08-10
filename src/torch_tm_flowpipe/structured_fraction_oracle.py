"""Independent exact-rational oracle for complete polynomial structured images.

This module intentionally does not import or call the production outward
interval implementation. Every input binary64 scalar is converted to its exact
``fractions.Fraction`` value before interval arithmetic.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import product
import math
from typing import Sequence

import torch


@dataclass(frozen=True)
class FractionInterval:
    lo: Fraction
    hi: Fraction

    def __post_init__(self) -> None:
        if self.lo > self.hi:
            raise ValueError("rational interval is inverted")

    @staticmethod
    def point(value: Fraction) -> "FractionInterval":
        return FractionInterval(value, value)

    @staticmethod
    def zero() -> "FractionInterval":
        return FractionInterval.point(Fraction(0))

    @staticmethod
    def one() -> "FractionInterval":
        return FractionInterval.point(Fraction(1))

    def add(self, other: "FractionInterval") -> "FractionInterval":
        return FractionInterval(self.lo + other.lo, self.hi + other.hi)

    def mul(self, other: "FractionInterval") -> "FractionInterval":
        values = (
            self.lo * other.lo,
            self.lo * other.hi,
            self.hi * other.lo,
            self.hi * other.hi,
        )
        return FractionInterval(min(values), max(values))

    def scale(self, value: Fraction | int) -> "FractionInterval":
        return self.mul(FractionInterval.point(Fraction(value)))

    def power(self, exponent: int) -> "FractionInterval":
        result = FractionInterval.one()
        for _ in range(int(exponent)):
            result = result.mul(self)
        return result


@dataclass(frozen=True)
class FractionStructuredImageOracle:
    affine_map: tuple[tuple[tuple[FractionInterval, ...], ...], ...]
    affine_image: tuple[tuple[FractionInterval, ...], ...]
    nonlinear_residual: tuple[tuple[FractionInterval, ...], ...]
    total_difference: tuple[tuple[FractionInterval, ...], ...]


def _fraction(value: float) -> Fraction:
    return Fraction.from_float(float(value))


def _interval_tensor(lo: torch.Tensor, hi: torch.Tensor) -> list[list[FractionInterval]]:
    return [
        [FractionInterval(_fraction(lo[b, i]), _fraction(hi[b, i])) for i in range(lo.shape[1])]
        for b in range(lo.shape[0])
    ]


def fraction_complete_polynomial_difference_oracle(
    coefficients: torch.Tensor,
    exponents: torch.Tensor,
    base_lo: torch.Tensor,
    base_hi: torch.Tensor,
    structured_lo: torch.Tensor,
    structured_hi: torch.Tensor,
    coordinate_map_lo: torch.Tensor,
    coordinate_map_hi: torch.Tensor,
) -> FractionStructuredImageOracle:
    """Evaluate the binomial difference with exact rational interval arithmetic."""
    if coefficients.ndim != 3 or exponents.ndim != 2:
        raise ValueError("oracle expects coefficients [B,O,M] and exponents [M,V]")
    batch, output_dim, term_count = coefficients.shape
    variable_dim = exponents.shape[1]
    structured_dim = structured_lo.shape[1]
    if term_count != exponents.shape[0]:
        raise ValueError("oracle coefficient/exponent count mismatch")
    if base_lo.shape != (batch, variable_dim) or base_hi.shape != base_lo.shape:
        raise ValueError("oracle base shape mismatch")
    if structured_lo.shape != (batch, structured_dim) or structured_hi.shape != structured_lo.shape:
        raise ValueError("oracle structured shape mismatch")
    if coordinate_map_lo.shape != (batch, variable_dim, structured_dim):
        raise ValueError("oracle coordinate map shape mismatch")
    if coordinate_map_hi.shape != coordinate_map_lo.shape:
        raise ValueError("oracle coordinate map endpoint mismatch")

    base = _interval_tensor(base_lo, base_hi)
    structured = _interval_tensor(structured_lo, structured_hi)
    coordinate = [
        [
            [
                FractionInterval(
                    _fraction(coordinate_map_lo[b, v, j]),
                    _fraction(coordinate_map_hi[b, v, j]),
                )
                for j in range(structured_dim)
            ]
            for v in range(variable_dim)
        ]
        for b in range(batch)
    ]
    delta = [
        [
            _sum_intervals(
                [coordinate[b][v][j].mul(structured[b][j]) for j in range(structured_dim)]
            )
            for v in range(variable_dim)
        ]
        for b in range(batch)
    ]
    exponent_rows: Sequence[Sequence[int]] = exponents.detach().cpu().to(torch.long).tolist()

    affine_map: list[list[list[FractionInterval]]] = [
        [
            [FractionInterval.zero() for _ in range(structured_dim)]
            for _ in range(output_dim)
        ]
        for _ in range(batch)
    ]
    total = [
        [FractionInterval.zero() for _ in range(output_dim)]
        for _ in range(batch)
    ]
    nonlinear = [
        [FractionInterval.zero() for _ in range(output_dim)]
        for _ in range(batch)
    ]
    for b in range(batch):
        for output in range(output_dim):
            for term_index, exponent_row in enumerate(exponent_rows):
                coefficient = FractionInterval.point(_fraction(coefficients[b, output, term_index]))
                for route in product(*(range(int(power) + 1) for power in exponent_row)):
                    structured_degree = sum(route)
                    if structured_degree == 0:
                        continue
                    term = coefficient
                    for variable, (power, selected) in enumerate(zip(exponent_row, route)):
                        factor = (
                            base[b][variable]
                            .power(int(power) - int(selected))
                            .mul(delta[b][variable].power(int(selected)))
                            .scale(math.comb(int(power), int(selected)))
                        )
                        term = term.mul(factor)
                    total[b][output] = total[b][output].add(term)
                    if structured_degree >= 2:
                        nonlinear[b][output] = nonlinear[b][output].add(term)

                for variable, power in enumerate(exponent_row):
                    if int(power) == 0:
                        continue
                    derivative = coefficient.scale(int(power))
                    for base_variable, base_power in enumerate(exponent_row):
                        derivative = derivative.mul(
                            base[b][base_variable].power(
                                int(base_power) - (1 if base_variable == variable else 0)
                            )
                        )
                    for structured_variable in range(structured_dim):
                        contribution = derivative.mul(
                            coordinate[b][variable][structured_variable]
                        )
                        affine_map[b][output][structured_variable] = (
                            affine_map[b][output][structured_variable].add(contribution)
                        )

    affine_image = [
        [
            _sum_intervals(
                [
                    affine_map[b][output][j].mul(structured[b][j])
                    for j in range(structured_dim)
                ]
            )
            for output in range(output_dim)
        ]
        for b in range(batch)
    ]
    return FractionStructuredImageOracle(
        tuple(tuple(tuple(row) for row in batch_rows) for batch_rows in affine_map),
        tuple(tuple(row) for row in affine_image),
        tuple(tuple(row) for row in nonlinear),
        tuple(tuple(row) for row in total),
    )


def _sum_intervals(values: Sequence[FractionInterval]) -> FractionInterval:
    result = FractionInterval.zero()
    for value in values:
        result = result.add(value)
    return result


__all__ = [
    "FractionInterval",
    "FractionStructuredImageOracle",
    "fraction_complete_polynomial_difference_oracle",
]
