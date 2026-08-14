"""Independent exact-rational primitives for the frozen VDP step-1 audit.

This module deliberately does not import either Taylor-model implementation.
All polynomial coefficients and interval endpoints are :class:`Fraction`
objects.  It is therefore suitable as the discrete/exact half of the
independent oracle, while the MPFR executable supplies a separately compiled
directed-rounding check.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from typing import Any, Iterable, Mapping, Sequence


Exponent = tuple[int, ...]


def fraction_text(value: Fraction) -> str:
    """Return the unique ``numerator/denominator`` exact encoding."""
    value = Fraction(value)
    return f"{value.numerator}/{value.denominator}"


def parse_fraction(value: str | int | Fraction) -> Fraction:
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value)
    text = str(value)
    if "/" in text:
        numerator, denominator = text.split("/", 1)
        return Fraction(int(numerator), int(denominator))
    return Fraction(text)


def dyadic_from_hex(value: str) -> Fraction:
    """Decode a finite C/Python hexadecimal binary64 string exactly."""
    return Fraction.from_float(float.fromhex(value))


def canonical_mpfr_fraction(value: str) -> Fraction:
    """Decode ``precision:sign:hex-mantissa:binary-exponent`` exactly."""
    fields = value.split(":")
    if len(fields) != 4:
        raise ValueError(f"invalid canonical MPFR value: {value!r}")
    precision, sign, mantissa, exponent = fields
    if int(precision) < 2 or int(sign) not in (-1, 1):
        raise ValueError(f"invalid canonical MPFR metadata: {value!r}")
    integer = int(mantissa, 16) * int(sign)
    power = int(exponent)
    if power >= 0:
        return Fraction(integer * (1 << power))
    return Fraction(integer, 1 << (-power))


@dataclass(frozen=True)
class RationalInterval:
    """Closed exact-rational interval with elementary natural arithmetic."""

    lo: Fraction
    hi: Fraction

    def __post_init__(self) -> None:
        object.__setattr__(self, "lo", Fraction(self.lo))
        object.__setattr__(self, "hi", Fraction(self.hi))
        if self.lo > self.hi:
            raise ValueError("reversed rational interval")

    @classmethod
    def point(cls, value: Fraction | int) -> "RationalInterval":
        value_q = Fraction(value)
        return cls(value_q, value_q)

    @classmethod
    def symmetric(cls, radius: Fraction | int) -> "RationalInterval":
        radius_q = abs(Fraction(radius))
        return cls(-radius_q, radius_q)

    def __add__(self, other: "RationalInterval") -> "RationalInterval":
        return RationalInterval(self.lo + other.lo, self.hi + other.hi)

    def __neg__(self) -> "RationalInterval":
        return RationalInterval(-self.hi, -self.lo)

    def __sub__(self, other: "RationalInterval") -> "RationalInterval":
        return self + (-other)

    def __mul__(self, other: "RationalInterval") -> "RationalInterval":
        values = (
            self.lo * other.lo,
            self.lo * other.hi,
            self.hi * other.lo,
            self.hi * other.hi,
        )
        return RationalInterval(min(values), max(values))

    def scale(self, value: Fraction | int) -> "RationalInterval":
        return self * RationalInterval.point(Fraction(value))

    def subseteq(self, other: "RationalInterval") -> bool:
        return other.lo <= self.lo and self.hi <= other.hi

    def contains(self, value: Fraction | int) -> bool:
        value_q = Fraction(value)
        return self.lo <= value_q <= self.hi

    @property
    def width(self) -> Fraction:
        return self.hi - self.lo

    def to_json(self) -> dict[str, str]:
        return {"lower": fraction_text(self.lo), "upper": fraction_text(self.hi)}


class RationalPolynomial:
    """Sparse multivariate polynomial with exact coefficients."""

    def __init__(self, n_vars: int, terms: Mapping[Exponent, Fraction | int] | None = None):
        if int(n_vars) <= 0:
            raise ValueError("a polynomial needs at least one variable")
        self.n_vars = int(n_vars)
        normalized: dict[Exponent, Fraction] = {}
        for exponent, coefficient in (terms or {}).items():
            exponent_t = tuple(int(item) for item in exponent)
            if len(exponent_t) != self.n_vars or any(item < 0 for item in exponent_t):
                raise ValueError(f"invalid exponent {exponent!r}")
            coefficient_q = Fraction(coefficient)
            if coefficient_q:
                normalized[exponent_t] = normalized.get(exponent_t, Fraction(0)) + coefficient_q
        self.terms = {key: value for key, value in normalized.items() if value}

    @classmethod
    def constant(cls, n_vars: int, value: Fraction | int) -> "RationalPolynomial":
        return cls(n_vars, {(0,) * int(n_vars): Fraction(value)})

    def copy(self) -> "RationalPolynomial":
        return RationalPolynomial(self.n_vars, self.terms)

    def _require_compatible(self, other: "RationalPolynomial") -> None:
        if self.n_vars != other.n_vars:
            raise ValueError("polynomial dimensions differ")

    def __add__(self, other: "RationalPolynomial") -> "RationalPolynomial":
        self._require_compatible(other)
        result = dict(self.terms)
        for exponent, coefficient in other.terms.items():
            result[exponent] = result.get(exponent, Fraction(0)) + coefficient
        return RationalPolynomial(self.n_vars, result)

    def __neg__(self) -> "RationalPolynomial":
        return RationalPolynomial(self.n_vars, {key: -value for key, value in self.terms.items()})

    def __sub__(self, other: "RationalPolynomial") -> "RationalPolynomial":
        return self + (-other)

    def __mul__(self, other: "RationalPolynomial") -> "RationalPolynomial":
        self._require_compatible(other)
        result: dict[Exponent, Fraction] = {}
        for left_exp, left_value in self.terms.items():
            for right_exp, right_value in other.terms.items():
                exponent = tuple(a + b for a, b in zip(left_exp, right_exp, strict=True))
                result[exponent] = result.get(exponent, Fraction(0)) + left_value * right_value
        return RationalPolynomial(self.n_vars, result)

    def scale(self, value: Fraction | int) -> "RationalPolynomial":
        value_q = Fraction(value)
        return RationalPolynomial(self.n_vars, {key: value_q * item for key, item in self.terms.items()})

    def truncate(self, total_degree: int) -> tuple["RationalPolynomial", "RationalPolynomial"]:
        degree = int(total_degree)
        retained = {key: value for key, value in self.terms.items() if sum(key) <= degree}
        discarded = {key: value for key, value in self.terms.items() if sum(key) > degree}
        return RationalPolynomial(self.n_vars, retained), RationalPolynomial(self.n_vars, discarded)

    def cutoff(self, threshold: Fraction) -> tuple["RationalPolynomial", "RationalPolynomial"]:
        threshold_q = abs(Fraction(threshold))
        retained = {key: value for key, value in self.terms.items() if abs(value) > threshold_q}
        discarded = {key: value for key, value in self.terms.items() if abs(value) <= threshold_q}
        return RationalPolynomial(self.n_vars, retained), RationalPolynomial(self.n_vars, discarded)

    def integrate(self, variable: int, max_total_degree: int | None = None) -> tuple["RationalPolynomial", "RationalPolynomial"]:
        variable_i = int(variable)
        if not 0 <= variable_i < self.n_vars:
            raise ValueError("integration variable out of range")
        retained: dict[Exponent, Fraction] = {}
        discarded: dict[Exponent, Fraction] = {}
        for exponent, coefficient in self.terms.items():
            new_exp = list(exponent)
            new_exp[variable_i] += 1
            exponent_t = tuple(new_exp)
            integrated = coefficient / exponent_t[variable_i]
            target = (
                discarded
                if max_total_degree is not None and sum(exponent_t) > int(max_total_degree)
                else retained
            )
            target[exponent_t] = target.get(exponent_t, Fraction(0)) + integrated
        return RationalPolynomial(self.n_vars, retained), RationalPolynomial(self.n_vars, discarded)

    def substitute(self, variable: int, value: Fraction) -> "RationalPolynomial":
        variable_i = int(variable)
        value_q = Fraction(value)
        terms: dict[Exponent, Fraction] = {}
        for exponent, coefficient in self.terms.items():
            new_exp = list(exponent)
            power = new_exp[variable_i]
            new_exp[variable_i] = 0
            exponent_t = tuple(new_exp)
            terms[exponent_t] = terms.get(exponent_t, Fraction(0)) + coefficient * value_q**power
        return RationalPolynomial(self.n_vars, terms)

    def natural_range(self, domain: Sequence[RationalInterval]) -> RationalInterval:
        if len(domain) != self.n_vars:
            raise ValueError("polynomial/domain dimensions differ")
        result = RationalInterval.point(0)
        for exponent, coefficient in self.terms.items():
            term = RationalInterval.point(coefficient)
            for power, interval in zip(exponent, domain, strict=True):
                if power == 0:
                    factor = RationalInterval.point(1)
                elif power % 2 == 1:
                    factor = RationalInterval(interval.lo**power, interval.hi**power)
                elif interval.lo <= 0 <= interval.hi:
                    factor = RationalInterval(Fraction(0), max(abs(interval.lo), abs(interval.hi)) ** power)
                else:
                    values = (interval.lo**power, interval.hi**power)
                    factor = RationalInterval(min(values), max(values))
                term = term * factor
            result = result + term
        return result

    @property
    def degree(self) -> int:
        return max((sum(key) for key in self.terms), default=0)

    def support(self) -> tuple[Exponent, ...]:
        return tuple(sorted(self.terms, key=lambda item: (sum(item), item)))

    def to_json(self, *, variable_order: Sequence[str]) -> dict[str, Any]:
        if len(variable_order) != self.n_vars:
            raise ValueError("variable-order dimension mismatch")
        return {
            "variable_order": list(variable_order),
            "support_size": len(self.terms),
            "degree": self.degree,
            "terms": [
                {"exponents": list(exponent), "coefficient": fraction_text(self.terms[exponent])}
                for exponent in self.support()
            ],
        }

    def __eq__(self, other: object) -> bool:
        return isinstance(other, RationalPolynomial) and self.n_vars == other.n_vars and self.terms == other.terms

    def __repr__(self) -> str:
        return f"RationalPolynomial(n_vars={self.n_vars}, terms={self.terms!r})"


TAU = 0
UX = 1
UY = 2
N_VARS = 3
H = Fraction(1, 100)
TARGET_RADIUS = Fraction(1, 10_000)
CUTOFF_RADIUS = Fraction(1, 10_000_000_000)


def exact_initial_polynomials() -> tuple[RationalPolynomial, RationalPolynomial]:
    x = RationalPolynomial(
        N_VARS,
        {
            (0, 0, 0): Fraction(5, 4),
            (0, 1, 0): Fraction(3, 20),
        },
    )
    y = RationalPolynomial(
        N_VARS,
        {
            (0, 0, 0): Fraction(12, 5),
            (0, 0, 1): Fraction(1, 20),
        },
    )
    return x, y


def vdp_polynomial_rhs(
    x: RationalPolynomial, y: RationalPolynomial
) -> tuple[RationalPolynomial, RationalPolynomial]:
    return y, y - x - (x * x * y)


@dataclass(frozen=True)
class ExactPicardIteration:
    iteration: int
    rhs_degree_limit: int
    rhs_x: RationalPolynomial
    rhs_y: RationalPolynomial
    discarded_rhs_x: RationalPolynomial
    discarded_rhs_y: RationalPolynomial
    image_x: RationalPolynomial
    image_y: RationalPolynomial


def exact_picard_iterations(
    *, order: int = 4, construction: str = "flowstar_staged"
) -> tuple[ExactPicardIteration, ...]:
    """Return exact Picard images for the two production construction styles.

    ``flowstar_staged`` evaluates iteration ``i`` at RHS degree ``i-1``.
    ``torch_complete`` evaluates in the complete O4 basis each time and lets
    time integration discard total degree greater than four.
    """
    if construction not in {"flowstar_staged", "torch_complete"}:
        raise ValueError(f"unknown construction: {construction}")
    if int(order) <= 0:
        raise ValueError("order must be positive")
    x0, y0 = exact_initial_polynomials()
    x, y = x0, y0
    rows: list[ExactPicardIteration] = []
    for iteration in range(1, int(order) + 1):
        rhs_x_full, rhs_y_full = vdp_polynomial_rhs(x, y)
        limit = iteration - 1 if construction == "flowstar_staged" else int(order) - 1
        rhs_x, discarded_x = rhs_x_full.truncate(limit)
        rhs_y, discarded_y = rhs_y_full.truncate(limit)
        integrated_x, overflow_x = rhs_x.integrate(TAU, max_total_degree=order)
        integrated_y, overflow_y = rhs_y.integrate(TAU, max_total_degree=order)
        discarded_x = discarded_x + overflow_x
        discarded_y = discarded_y + overflow_y
        x = x0 + integrated_x
        y = y0 + integrated_y
        rows.append(
            ExactPicardIteration(
                iteration=iteration,
                rhs_degree_limit=limit,
                rhs_x=rhs_x,
                rhs_y=rhs_y,
                discarded_rhs_x=discarded_x,
                discarded_rhs_y=discarded_y,
                image_x=x,
                image_y=y,
            )
        )
    return tuple(rows)


def exact_step1_polynomials() -> tuple[RationalPolynomial, RationalPolynomial]:
    staged = exact_picard_iterations(construction="flowstar_staged")[-1]
    complete = exact_picard_iterations(construction="torch_complete")[-1]
    if staged.image_x != complete.image_x or staged.image_y != complete.image_y:
        raise AssertionError("the two exact complete-O4 Picard constructions disagree")
    return staged.image_x, staged.image_y


def common_domain(*, endpoint: bool = False) -> tuple[RationalInterval, ...]:
    tau = RationalInterval.point(H) if endpoint else RationalInterval(Fraction(0), H)
    return (tau, RationalInterval(-1, 1), RationalInterval(-1, 1))


def _nonlinear_remainder(
    px_range: RationalInterval,
    py_range: RationalInterval,
    rx: RationalInterval,
    ry: RationalInterval,
) -> RationalInterval:
    two = RationalInterval.point(2)
    # (Px+rx)^2(Py+ry) - Px^2 Py, expanded so that every summand contains
    # at least one explicit remainder source.
    return (
        px_range * px_range * ry
        + two * px_range * rx * py_range
        + two * px_range * rx * ry
        + rx * rx * py_range
        + rx * rx * ry
    )


@dataclass(frozen=True)
class RemainderOracleIteration:
    iteration: int
    candidate_x: RationalInterval
    candidate_y: RationalInterval
    image_x: RationalInterval
    image_y: RationalInterval
    subset_x: bool
    subset_y: bool
    margin_x: Fraction
    margin_y: Fraction


@dataclass(frozen=True)
class ExactStep1OracleResult:
    polynomial_x: RationalPolynomial
    polynomial_y: RationalPolynomial
    truncation_x: RationalInterval
    truncation_y: RationalInterval
    cutoff_x: RationalInterval
    cutoff_y: RationalInterval
    refinement: tuple[RemainderOracleIteration, ...]
    final_remainder_x: RationalInterval
    final_remainder_y: RationalInterval
    segment_polynomial_x: RationalInterval
    segment_polynomial_y: RationalInterval
    endpoint_polynomial_x: RationalInterval
    endpoint_polynomial_y: RationalInterval
    segment_final_x: RationalInterval
    segment_final_y: RationalInterval
    endpoint_final_x: RationalInterval
    endpoint_final_y: RationalInterval

    def to_json(self) -> dict[str, Any]:
        names = ("tau", "ux", "uy")
        return {
            "schema": "independent_exact_step1_oracle_v1",
            "range_algorithm": "exact_rational_natural_interval",
            "polynomial": {
                "x": self.polynomial_x.to_json(variable_order=names),
                "y": self.polynomial_y.to_json(variable_order=names),
            },
            "truncation_remainder": {
                "x": self.truncation_x.to_json(),
                "y": self.truncation_y.to_json(),
            },
            "cutoff_remainder": {
                "x": self.cutoff_x.to_json(),
                "y": self.cutoff_y.to_json(),
            },
            "refinement": [
                {
                    "iteration": row.iteration,
                    "candidate": {"x": row.candidate_x.to_json(), "y": row.candidate_y.to_json()},
                    "image": {"x": row.image_x.to_json(), "y": row.image_y.to_json()},
                    "subset": {"x": row.subset_x, "y": row.subset_y},
                    "margin": {"x": fraction_text(row.margin_x), "y": fraction_text(row.margin_y)},
                }
                for row in self.refinement
            ],
            "final_remainder": {
                "x": self.final_remainder_x.to_json(),
                "y": self.final_remainder_y.to_json(),
            },
            "range": {
                "segment_polynomial": {
                    "x": self.segment_polynomial_x.to_json(),
                    "y": self.segment_polynomial_y.to_json(),
                },
                "endpoint_polynomial": {
                    "x": self.endpoint_polynomial_x.to_json(),
                    "y": self.endpoint_polynomial_y.to_json(),
                },
                "segment_final": {
                    "x": self.segment_final_x.to_json(),
                    "y": self.segment_final_y.to_json(),
                },
                "endpoint_final": {
                    "x": self.endpoint_final_x.to_json(),
                    "y": self.endpoint_final_y.to_json(),
                },
            },
        }


def exact_step1_remainder_oracle(*, refinement_steps: int = 5) -> ExactStep1OracleResult:
    """Build a conservative exact-rational Picard self-map enclosure.

    The range algorithm is deliberately named and simple: termwise natural
    interval evaluation.  It makes no sampling or hidden dependency claim.
    """
    px, py = exact_step1_polynomials()
    x0, y0 = exact_initial_polynomials()
    full_rhs_x, full_rhs_y = vdp_polynomial_rhs(px, py)
    retained_rhs_x, discarded_rhs_x = full_rhs_x.truncate(3)
    retained_rhs_y, discarded_rhs_y = full_rhs_y.truncate(3)
    integrated_x, overflow_x = retained_rhs_x.integrate(TAU, max_total_degree=4)
    integrated_y, overflow_y = retained_rhs_y.integrate(TAU, max_total_degree=4)
    discarded_rhs_x = discarded_rhs_x + overflow_x
    discarded_rhs_y = discarded_rhs_y + overflow_y
    residual_x = (x0 + integrated_x) - px
    residual_y = (y0 + integrated_y) - py
    segment_domain = common_domain(endpoint=False)
    tau_domain = RationalInterval(Fraction(0), H)
    # Polynomial truncation is integrated symbolically before it is ranged.
    # Multiplying a derivative range by [0,h] would erase the known tau power
    # and is needlessly (sometimes dramatically) wider.
    integrated_discard_x, _ = discarded_rhs_x.integrate(TAU)
    integrated_discard_y, _ = discarded_rhs_y.integrate(TAU)
    truncation_x = integrated_discard_x.natural_range(segment_domain)
    truncation_y = integrated_discard_y.natural_range(segment_domain)
    cutoff_x = RationalInterval.point(0)
    cutoff_y = RationalInterval.point(0)
    px_range = px.natural_range(segment_domain)
    py_range = py.natural_range(segment_domain)
    residual_x_range = residual_x.natural_range(segment_domain)
    residual_y_range = residual_y.natural_range(segment_domain)

    rx = RationalInterval.symmetric(TARGET_RADIUS)
    ry = RationalInterval.symmetric(TARGET_RADIUS)
    rows: list[RemainderOracleIteration] = []
    for iteration in range(int(refinement_steps)):
        nonlinear = _nonlinear_remainder(px_range, py_range, rx, ry)
        derivative_rx = ry
        derivative_ry = ry - rx - nonlinear
        image_x = residual_x_range + truncation_x + cutoff_x + derivative_rx * tau_domain
        image_y = residual_y_range + truncation_y + cutoff_y + derivative_ry * tau_domain
        subset_x = image_x.subseteq(rx)
        subset_y = image_y.subseteq(ry)
        rows.append(
            RemainderOracleIteration(
                iteration=iteration,
                candidate_x=rx,
                candidate_y=ry,
                image_x=image_x,
                image_y=image_y,
                subset_x=subset_x,
                subset_y=subset_y,
                margin_x=min(image_x.lo - rx.lo, rx.hi - image_x.hi),
                margin_y=min(image_y.lo - ry.lo, ry.hi - image_y.hi),
            )
        )
        if not (subset_x and subset_y):
            break
        if image_x == rx and image_y == ry:
            break
        rx, ry = image_x, image_y

    segment_poly_x = px.natural_range(segment_domain)
    segment_poly_y = py.natural_range(segment_domain)
    endpoint_domain = common_domain(endpoint=True)
    # Endpoint semantics require algebraic substitution before ranging.  Merely
    # evaluating each still-unmerged tau term on the point interval {h} loses
    # dependencies between equal post-substitution monomials.
    endpoint_px = px.substitute(TAU, H)
    endpoint_py = py.substitute(TAU, H)
    endpoint_poly_x = endpoint_px.natural_range(endpoint_domain)
    endpoint_poly_y = endpoint_py.natural_range(endpoint_domain)
    return ExactStep1OracleResult(
        polynomial_x=px,
        polynomial_y=py,
        truncation_x=truncation_x,
        truncation_y=truncation_y,
        cutoff_x=cutoff_x,
        cutoff_y=cutoff_y,
        refinement=tuple(rows),
        final_remainder_x=rx,
        final_remainder_y=ry,
        segment_polynomial_x=segment_poly_x,
        segment_polynomial_y=segment_poly_y,
        endpoint_polynomial_x=endpoint_poly_x,
        endpoint_polynomial_y=endpoint_poly_y,
        segment_final_x=segment_poly_x + rx,
        segment_final_y=segment_poly_y + ry,
        endpoint_final_x=endpoint_poly_x + rx,
        endpoint_final_y=endpoint_poly_y + ry,
    )


def complete_support(n_vars: int, order: int) -> tuple[Exponent, ...]:
    return tuple(
        sorted(
            (
                exponent
                for exponent in product(range(int(order) + 1), repeat=int(n_vars))
                if sum(exponent) <= int(order)
            ),
            key=lambda item: (sum(item), item),
        )
    )


def exact_fixture_polynomials() -> dict[str, RationalPolynomial]:
    """Small hand-checkable affine through quartic fixtures."""
    return {
        "affine": RationalPolynomial(2, {(0, 0): 1, (1, 0): Fraction(1, 2), (0, 1): -2}),
        "quadratic": RationalPolynomial(2, {(2, 0): 3, (1, 1): -1, (0, 2): Fraction(1, 3)}),
        "cubic": RationalPolynomial(2, {(3, 0): Fraction(2, 5), (2, 1): -4, (0, 3): 7}),
        "quartic": RationalPolynomial(2, {(4, 0): 1, (3, 1): -2, (2, 2): 3, (0, 4): -4}),
    }


def exact_vdp_time_series(
    x0: Fraction, y0: Fraction, *, degree: int
) -> tuple[tuple[Fraction, ...], tuple[Fraction, ...]]:
    """Return exact Taylor coefficients of the point-initialized VDP ODE."""
    degree_i = int(degree)
    if degree_i < 0:
        raise ValueError("series degree must be nonnegative")
    x = [Fraction(x0)] + [Fraction(0)] * degree_i
    y = [Fraction(y0)] + [Fraction(0)] * degree_i
    for n in range(degree_i):
        x_squared_y = sum(
            x[i] * x[j] * y[n - i - j]
            for i in range(n + 1)
            for j in range(n - i + 1)
        )
        x[n + 1] = y[n] / (n + 1)
        y[n + 1] = (y[n] - x[n] - x_squared_y) / (n + 1)
    return tuple(x), tuple(y)


def _evaluate_univariate(coefficients: Sequence[Fraction], value: Fraction) -> Fraction:
    result = Fraction(0)
    for coefficient in reversed(coefficients):
        result = result * value + Fraction(coefficient)
    return result


@dataclass(frozen=True)
class FormalTrueSolutionEnclosure:
    """Rigorous point-corner enclosure plus a monotonicity reduction proof."""

    series_degree: int
    complex_disk_radius: Fraction
    state_ball_radius: Fraction
    contraction_bound: Fraction
    self_map_x_bound: Fraction
    self_map_y_bound: Fraction
    tail_x: Fraction
    tail_y: Fraction
    corners: Mapping[str, tuple[RationalInterval, RationalInterval]]
    endpoint_x: RationalInterval
    endpoint_y: RationalInterval
    segment_x: RationalInterval
    segment_y: RationalInterval

    def to_json(self) -> dict[str, Any]:
        return {
            "schema": "formal_vdp_step1_true_solution_enclosure_v1",
            "method": "exact Taylor coefficients at four monotone corners plus Cauchy tail",
            "series_degree": self.series_degree,
            "complex_disk_radius": fraction_text(self.complex_disk_radius),
            "state_ball_radius": fraction_text(self.state_ball_radius),
            "picard_self_map_bounds": {
                "x": fraction_text(self.self_map_x_bound),
                "y": fraction_text(self.self_map_y_bound),
            },
            "picard_contraction_bound": fraction_text(self.contraction_bound),
            "cauchy_tail": {"x": fraction_text(self.tail_x), "y": fraction_text(self.tail_y)},
            "corners": {
                name: {"x": values[0].to_json(), "y": values[1].to_json()}
                for name, values in self.corners.items()
            },
            "endpoint": {"x": self.endpoint_x.to_json(), "y": self.endpoint_y.to_json()},
            "segment": {"x": self.segment_x.to_json(), "y": self.segment_y.to_json()},
            "monotonicity_proof": {
                "real_region": "x>0 and y>0 from the radius-one existence ball; x'>0; because x starts >=11/10, y'=y(1-x^2)-x<0",
                "sensitivity_norm": "||S-I||_inf <= exp(Lh)-1 <= Lh/(1-Lh) < 1",
                "signs": "S11>0, S22>0; S12'=S22 with S12(0)=0; at every S21=0 boundary, S21'=(-1-2xy)S11<0",
                "endpoint_extrema": "x: (x0,y0)=(11/10,47/20),(7/5,49/20); y: (7/5,47/20),(11/10,49/20)",
                "segment_extrema": "x minimum 11/10 at t=0 and maximum at upper/upper endpoint; y maximum 49/20 at t=0 and minimum at upper/lower endpoint",
            },
        }


def formal_true_solution_enclosure(*, series_degree: int = 100) -> FormalTrueSolutionEnclosure:
    """Certify the exact VDP step-1 endpoint/tube independently of both tools.

    On the complex time disk ``|t| <= 1/50``, a radius-one state ball around
    every initial corner is a strict Picard self-map and contraction.  Cauchy's
    coefficient estimate therefore bounds the degree-``N`` tail at ``h=1/100``
    by ``M/2**N``.  A short sensitivity sign proof reduces the interval initial
    set to four point corners; no sampling is used.
    """
    degree = int(series_degree)
    if degree < 16:
        raise ValueError("formal corner enclosure requires series degree >= 16")
    disk = Fraction(1, 50)
    ball = Fraction(1)
    max_x = Fraction(12, 5)  # max |x0| + radius
    max_y = Fraction(69, 20)  # max |y0| + radius
    self_x = disk * max_y
    self_y = disk * (max_y + max_x + max_x * max_x * max_y)
    lipschitz = Fraction(2) + 2 * max_x * max_y + max_x * max_x
    contraction = disk * lipschitz
    if max(self_x, self_y) >= ball or contraction >= 1:
        raise AssertionError("declared analytic Picard ball does not close")
    tail_x = max_x / (2**degree)
    tail_y = max_y / (2**degree)
    corner_values = {
        "lower_lower": (Fraction(11, 10), Fraction(47, 20)),
        "lower_upper": (Fraction(11, 10), Fraction(49, 20)),
        "upper_lower": (Fraction(7, 5), Fraction(47, 20)),
        "upper_upper": (Fraction(7, 5), Fraction(49, 20)),
    }
    corners: dict[str, tuple[RationalInterval, RationalInterval]] = {}
    for name, (x0, y0) in corner_values.items():
        x_coefficients, y_coefficients = exact_vdp_time_series(x0, y0, degree=degree)
        x_value = _evaluate_univariate(x_coefficients, H)
        y_value = _evaluate_univariate(y_coefficients, H)
        corners[name] = (
            RationalInterval(x_value - tail_x, x_value + tail_x),
            RationalInterval(y_value - tail_y, y_value + tail_y),
        )
    endpoint_x = RationalInterval(
        corners["lower_lower"][0].lo,
        corners["upper_upper"][0].hi,
    )
    endpoint_y = RationalInterval(
        corners["upper_lower"][1].lo,
        corners["lower_upper"][1].hi,
    )
    segment_x = RationalInterval(Fraction(11, 10), endpoint_x.hi)
    segment_y = RationalInterval(endpoint_y.lo, Fraction(49, 20))
    return FormalTrueSolutionEnclosure(
        series_degree=degree,
        complex_disk_radius=disk,
        state_ball_radius=ball,
        contraction_bound=contraction,
        self_map_x_bound=self_x,
        self_map_y_bound=self_y,
        tail_x=tail_x,
        tail_y=tail_y,
        corners=corners,
        endpoint_x=endpoint_x,
        endpoint_y=endpoint_y,
        segment_x=segment_x,
        segment_y=segment_y,
    )
