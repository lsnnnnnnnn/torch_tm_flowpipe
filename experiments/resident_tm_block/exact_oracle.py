"""Independent Fraction oracle for the resident normal-composition contract.

This module is diagnostic-only.  It intentionally does not call the CUDA
interval helpers or the sparse Taylor-model multiplication.  Python floats
reproduce the declared point-coefficient RN sequence; Fractions independently
carry the exact coefficient and interval expressions that those point values
must enclose.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Sequence

import torch

from torch_tm_flowpipe.resident_tm_block import (
    ResidentNormalCompositionRequest,
    ResidentNormalCompositionResult,
    canonical_exponents,
)
from torch_tm_flowpipe.taylor_model import TaylorModel
from torch_tm_flowpipe.tm_vector import TMVector


@dataclass(frozen=True)
class RationalInterval:
    lo: Fraction
    hi: Fraction

    def __post_init__(self) -> None:
        if self.lo > self.hi:
            raise ValueError("invalid rational interval")


ZERO = RationalInterval(Fraction(0), Fraction(0))
ONE = RationalInterval(Fraction(1), Fraction(1))


def _fraction(value: float | torch.Tensor) -> Fraction:
    return Fraction(float(value))


def _add(left: RationalInterval, right: RationalInterval) -> RationalInterval:
    return RationalInterval(left.lo + right.lo, left.hi + right.hi)


def _mul(left: RationalInterval, right: RationalInterval) -> RationalInterval:
    products = (
        left.lo * right.lo,
        left.lo * right.hi,
        left.hi * right.lo,
        left.hi * right.hi,
    )
    return RationalInterval(min(products), max(products))


def _power(value: RationalInterval, exponent: int) -> RationalInterval:
    result = ONE
    for _ in range(int(exponent)):
        result = _mul(result, value)
    return result


def _term_index(first: int, second: int) -> int:
    degree = int(first) + int(second)
    return degree * (degree + 1) // 2 + int(first)


def _term_count(order: int) -> int:
    return (int(order) + 1) * (int(order) + 2) // 2


def _partition_endpoint(lo: float, hi: float, index: int, pieces: int) -> float:
    if index <= 0:
        return lo
    if index >= pieces:
        return hi
    width = hi - lo
    fraction = float(index) / float(pieces)
    value = lo + width * fraction
    return min(hi, max(lo, value))


def _range_on_box(
    coefficient_lo: Sequence[Fraction],
    coefficient_hi: Sequence[Fraction],
    order: int,
    domain: Sequence[RationalInterval],
) -> RationalInterval:
    result = ZERO
    for degree in range(int(order) + 1):
        for first in range(degree + 1):
            second = degree - first
            index = _term_index(first, second)
            coefficient = RationalInterval(
                coefficient_lo[index], coefficient_hi[index]
            )
            if coefficient == ZERO:
                continue
            monomial = _mul(_power(domain[0], first), _power(domain[1], second))
            result = _add(result, _mul(coefficient, monomial))
    return result


def _range_polynomial(
    coefficient_lo: Sequence[Fraction],
    coefficient_hi: Sequence[Fraction],
    order: int,
    domain_lo: Sequence[float],
    domain_hi: Sequence[float],
    split: int,
) -> RationalInterval:
    pieces = int(split) if int(split) > 1 else 1
    hull: RationalInterval | None = None
    for first_cell in range(pieces):
        for second_cell in range(pieces):
            box = (
                RationalInterval(
                    Fraction(
                        _partition_endpoint(
                            domain_lo[0], domain_hi[0], first_cell, pieces
                        )
                    ),
                    Fraction(
                        _partition_endpoint(
                            domain_lo[0], domain_hi[0], first_cell + 1, pieces
                        )
                    ),
                ),
                RationalInterval(
                    Fraction(
                        _partition_endpoint(
                            domain_lo[1], domain_hi[1], second_cell, pieces
                        )
                    ),
                    Fraction(
                        _partition_endpoint(
                            domain_lo[1], domain_hi[1], second_cell + 1, pieces
                        )
                    ),
                ),
            )
            value = _range_on_box(coefficient_lo, coefficient_hi, order, box)
            hull = (
                value
                if hull is None
                else RationalInterval(min(hull.lo, value.lo), max(hull.hi, value.hi))
            )
    assert hull is not None
    return hull


@dataclass
class OracleTM:
    point: list[float]
    coefficient_lo: list[Fraction]
    coefficient_hi: list[Fraction]
    remainder: RationalInterval
    split: int

    @classmethod
    def zero(cls, terms: int, split: int) -> "OracleTM":
        return cls(
            [0.0] * terms,
            [Fraction(0)] * terms,
            [Fraction(0)] * terms,
            ZERO,
            int(split),
        )

    @classmethod
    def constant(
        cls,
        terms: int,
        split: int,
        point: float,
        lo: Fraction,
        hi: Fraction,
    ) -> "OracleTM":
        result = cls.zero(terms, split)
        result.point[0] = float(point)
        result.coefficient_lo[0] = lo
        result.coefficient_hi[0] = hi
        return result


def _active(point: float, lo: Fraction, hi: Fraction) -> bool:
    return point != 0.0 or lo != 0 or hi != 0


def _cutoff(
    value: OracleTM,
    cutoff: float | None,
    order: int,
    domain_lo: Sequence[float],
    domain_hi: Sequence[float],
) -> None:
    if cutoff is None:
        return
    terms = _term_count(order)
    removed_lo = [Fraction(0)] * terms
    removed_hi = [Fraction(0)] * terms
    for index in range(terms):
        if abs(value.point[index]) <= float(cutoff):
            removed_lo[index] = value.coefficient_lo[index]
            removed_hi[index] = value.coefficient_hi[index]
            value.point[index] = 0.0
            value.coefficient_lo[index] = Fraction(0)
            value.coefficient_hi[index] = Fraction(0)
    removed = _range_polynomial(
        removed_lo, removed_hi, order, domain_lo, domain_hi, 0
    )
    value.remainder = _add(value.remainder, removed)


def _add_constant(
    value: OracleTM, point: float, lo: Fraction, hi: Fraction
) -> None:
    value.point[0] = float(value.point[0] + float(point))
    value.coefficient_lo[0] += lo
    value.coefficient_hi[0] += hi


def _add_tm(left: OracleTM, right: OracleTM) -> None:
    for index in range(len(left.point)):
        left.point[index] = float(left.point[index] + right.point[index])
        left.coefficient_lo[index] += right.coefficient_lo[index]
        left.coefficient_hi[index] += right.coefficient_hi[index]
    left.remainder = _add(left.remainder, right.remainder)
    left.split = max(left.split, right.split)


def _multiply(
    left: OracleTM,
    right: OracleTM,
    order: int,
    cutoff: float | None,
    domain_lo: Sequence[float],
    domain_hi: Sequence[float],
) -> None:
    terms = _term_count(order)
    full_terms = _term_count(2 * order)
    point = [0.0] * terms
    coefficient_lo = [Fraction(0)] * terms
    coefficient_hi = [Fraction(0)] * terms
    dropped_lo = [Fraction(0)] * full_terms
    dropped_hi = [Fraction(0)] * full_terms
    exponents = canonical_exponents(order)
    for left_index, (left_first, left_second) in enumerate(exponents):
        if (
            left.coefficient_lo[left_index] == 0
            and left.coefficient_hi[left_index] == 0
        ):
            continue
        for right_index, (right_first, right_second) in enumerate(exponents):
            if (
                right.coefficient_lo[right_index] == 0
                and right.coefficient_hi[right_index] == 0
            ):
                continue
            output_first = left_first + right_first
            output_second = left_second + right_second
            output_index = _term_index(output_first, output_second)
            coefficient_product = _mul(
                RationalInterval(
                    left.coefficient_lo[left_index],
                    left.coefficient_hi[left_index],
                ),
                RationalInterval(
                    right.coefficient_lo[right_index],
                    right.coefficient_hi[right_index],
                ),
            )
            if output_first + output_second <= order:
                point_product = float(
                    left.point[left_index] * right.point[right_index]
                )
                point[output_index] = float(point[output_index] + point_product)
                coefficient_lo[output_index] += coefficient_product.lo
                coefficient_hi[output_index] += coefficient_product.hi
            else:
                dropped_lo[output_index] += coefficient_product.lo
                dropped_hi[output_index] += coefficient_product.hi
    split = max(left.split, right.split)
    left_range = _range_polynomial(
        left.coefficient_lo,
        left.coefficient_hi,
        order,
        domain_lo,
        domain_hi,
        0,
    )
    right_range = _range_polynomial(
        right.coefficient_lo,
        right.coefficient_hi,
        order,
        domain_lo,
        domain_hi,
        0,
    )
    truncation = _range_polynomial(
        dropped_lo, dropped_hi, 2 * order, domain_lo, domain_hi, split
    )
    remainder = ZERO
    remainder = _add(remainder, _mul(left_range, right.remainder))
    remainder = _add(remainder, _mul(right_range, left.remainder))
    remainder = _add(remainder, _mul(left.remainder, right.remainder))
    remainder = _add(remainder, truncation)
    left.point = point
    left.coefficient_lo = coefficient_lo
    left.coefficient_hi = coefficient_hi
    left.remainder = remainder
    left.split = split
    _cutoff(left, cutoff, order, domain_lo, domain_hi)


def _input_model(
    point: torch.Tensor,
    lo: torch.Tensor,
    hi: torch.Tensor,
    rem_lo: torch.Tensor,
    rem_hi: torch.Tensor,
    split: int,
) -> OracleTM:
    return OracleTM(
        [float(value) for value in point],
        [_fraction(value) for value in lo],
        [_fraction(value) for value in hi],
        RationalInterval(_fraction(rem_lo), _fraction(rem_hi)),
        int(split),
    )


@dataclass(frozen=True)
class OracleOutput:
    models: tuple[OracleTM, ...]
    coefficient_error: tuple[tuple[RationalInterval, ...], ...]


def compose_exact(request: ResidentNormalCompositionRequest) -> OracleOutput:
    order = int(request.order)
    terms = _term_count(order)
    domain_lo = [float(value) for value in request.domain_lo]
    domain_hi = [float(value) for value in request.domain_hi]
    inner = [
        _input_model(
            request.inner_point[index],
            request.inner_lo[index],
            request.inner_hi[index],
            request.inner_rem_lo[index],
            request.inner_rem_hi[index],
            request.inner_splits[index],
        )
        for index in range(2)
    ]
    outputs: list[OracleTM] = []
    errors: list[tuple[RationalInterval, ...]] = []
    for component in range(request.output_dim):
        outer_point = [float(value) for value in request.outer_point[component]]
        outer_lo = [_fraction(value) for value in request.outer_lo[component]]
        outer_hi = [_fraction(value) for value in request.outer_hi[component]]
        maximum = -1
        for first in range(order + 1):
            for second in range(order - first + 1):
                index = _term_index(first, second)
                if _active(outer_point[index], outer_lo[index], outer_hi[index]):
                    maximum = first

        def branch(first: int) -> OracleTM | None:
            second_max = -1
            for second in range(order - first + 1):
                index = _term_index(first, second)
                if _active(outer_point[index], outer_lo[index], outer_hi[index]):
                    second_max = second
            if second_max < 0:
                return None
            start = _term_index(first, second_max)
            result = OracleTM.constant(
                terms,
                request.outer_splits[component],
                outer_point[start],
                outer_lo[start],
                outer_hi[start],
            )
            for second in range(second_max - 1, -1, -1):
                _multiply(
                    result,
                    inner[1],
                    order,
                    request.cutoff,
                    domain_lo,
                    domain_hi,
                )
                index = _term_index(first, second)
                if _active(outer_point[index], outer_lo[index], outer_hi[index]):
                    _add_constant(
                        result,
                        outer_point[index],
                        outer_lo[index],
                        outer_hi[index],
                    )
                    _cutoff(
                        result,
                        request.cutoff,
                        order,
                        domain_lo,
                        domain_hi,
                    )
            return result

        if maximum < 0:
            accumulator = OracleTM.zero(
                terms, request.outer_splits[component]
            )
        else:
            accumulator = branch(maximum)
            assert accumulator is not None
            for first in range(maximum - 1, -1, -1):
                _multiply(
                    accumulator,
                    inner[0],
                    order,
                    request.cutoff,
                    domain_lo,
                    domain_hi,
                )
                addend = branch(first)
                if addend is not None:
                    _add_tm(accumulator, addend)
                    _cutoff(
                        accumulator,
                        request.cutoff,
                        order,
                        domain_lo,
                        domain_hi,
                    )
        accumulator.remainder = _add(
            accumulator.remainder,
            RationalInterval(
                _fraction(request.outer_rem_lo[component]),
                _fraction(request.outer_rem_hi[component]),
            ),
        )
        coefficient_errors = tuple(
            RationalInterval(
                accumulator.coefficient_lo[index]
                - Fraction(accumulator.point[index]),
                accumulator.coefficient_hi[index]
                - Fraction(accumulator.point[index]),
            )
            for index in range(terms)
        )
        error_range = _range_polynomial(
            [value.lo for value in coefficient_errors],
            [value.hi for value in coefficient_errors],
            order,
            domain_lo,
            domain_hi,
            accumulator.split,
        )
        accumulator.remainder = _add(accumulator.remainder, error_range)
        outputs.append(accumulator)
        errors.append(coefficient_errors)
    return OracleOutput(tuple(outputs), tuple(errors))


def verify_result(
    request: ResidentNormalCompositionRequest,
    result: ResidentNormalCompositionResult,
) -> dict[str, object]:
    if not result.ok or result.output is None:
        raise AssertionError(f"resident result is not successful: {result.status}")
    oracle = compose_exact(request)
    actual_models = (
        list(result.output.models)
        if isinstance(result.output, TMVector)
        else [result.output]
    )
    if len(actual_models) != len(oracle.models):
        raise AssertionError("resident output component count mismatch")
    assert result.coefficient_error_lo is not None
    assert result.coefficient_error_hi is not None
    exponents = canonical_exponents(request.order)
    coefficient_checks = 0
    error_checks = 0
    remainder_checks = 0
    for component, (expected, actual) in enumerate(zip(oracle.models, actual_models)):
        for index, exponent in enumerate(exponents):
            actual_point = float(
                actual.polynomial.terms.get(exponent, torch.tensor(0.0))
            )
            if actual_point.hex() != float(expected.point[index]).hex():
                raise AssertionError(
                    f"point coefficient mismatch at component={component}, exponent={exponent}: "
                    f"{actual_point.hex()} != {expected.point[index].hex()}"
                )
            coefficient_checks += 1
            actual_error_lo = Fraction(
                float(result.coefficient_error_lo[component, index])
            )
            actual_error_hi = Fraction(
                float(result.coefficient_error_hi[component, index])
            )
            expected_error = oracle.coefficient_error[component][index]
            if not (
                actual_error_lo <= expected_error.lo
                and expected_error.hi <= actual_error_hi
            ):
                raise AssertionError(
                    f"coefficient error not enclosed at component={component}, exponent={exponent}"
                )
            error_checks += 1
        actual_rem_lo = Fraction(float(actual.remainder.lo))
        actual_rem_hi = Fraction(float(actual.remainder.hi))
        if not (
            actual_rem_lo <= expected.remainder.lo
            and expected.remainder.hi <= actual_rem_hi
        ):
            raise AssertionError(
                f"remainder not enclosed at component={component}: "
                f"[{float(actual.remainder.lo).hex()}, {float(actual.remainder.hi).hex()}]"
            )
        remainder_checks += 1
    return {
        "point_coefficient_checks": coefficient_checks,
        "coefficient_error_interval_checks": error_checks,
        "remainder_interval_checks": remainder_checks,
        "independent_numeric_type": "fractions.Fraction",
        "cuda_interval_helpers_called": False,
        "legacy_cpu_result_used_as_truth": False,
    }


__all__ = [
    "OracleOutput",
    "OracleTM",
    "RationalInterval",
    "compose_exact",
    "verify_result",
]
