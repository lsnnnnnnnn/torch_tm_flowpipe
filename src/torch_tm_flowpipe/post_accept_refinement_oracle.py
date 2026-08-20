"""Independent exact-rational oracle for C2 remainder-refinement ledgers.

The production closure converts interval-coefficient tensors to Bernstein
form in :mod:`batched_dense_tm`.  This oracle does not import that module or
reuse its transforms: it decodes binary64 values as exact dyadic rationals and
uses the separate sparse :class:`RationalPolynomial` implementation.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Mapping, Sequence

from .step1_oracle import RationalInterval, RationalPolynomial, fraction_text


def _q(value: float | str) -> Fraction:
    if isinstance(value, str) and value.lower().startswith(("0x", "-0x", "+0x")):
        return Fraction.from_float(float.fromhex(value))
    return Fraction.from_float(float(value))


def _interval(lo: float | str, hi: float | str) -> RationalInterval:
    return RationalInterval(_q(lo), _q(hi))


def polynomial_from_binary64_hex(
    coefficient_hex: Sequence[str],
    exponents: Sequence[Sequence[int]],
) -> RationalPolynomial:
    if len(coefficient_hex) != len(exponents):
        raise ValueError("coefficient/exponent length mismatch")
    if not exponents:
        raise ValueError("polynomial exponent table is empty")
    n_vars = len(exponents[0])
    return RationalPolynomial(
        n_vars,
        {
            tuple(int(power) for power in exponent): _q(coefficient)
            for coefficient, exponent in zip(coefficient_hex, exponents, strict=True)
            if _q(coefficient) != 0
        },
    )


def _extend(polynomial: RationalPolynomial, n_vars: int) -> RationalPolynomial:
    if polynomial.n_vars > int(n_vars):
        raise ValueError("cannot shrink an exact polynomial")
    return RationalPolynomial(
        int(n_vars),
        {
            exponent + (0,) * (int(n_vars) - polynomial.n_vars): coefficient
            for exponent, coefficient in polynomial.terms.items()
        },
    )


def _variable(n_vars: int, index: int) -> RationalPolynomial:
    exponent = [0] * int(n_vars)
    exponent[int(index)] = 1
    return RationalPolynomial(int(n_vars), {tuple(exponent): Fraction(1)})


@dataclass(frozen=True)
class RefinementComponentCertificate:
    component: int
    exact_raw_rhs_residual: RationalInterval
    production_raw_rhs_remainder: RationalInterval
    exact_poly_diff: RationalInterval
    production_poly_diff: RationalInterval
    exact_final_image: RationalInterval
    production_final_image: RationalInterval
    raw_rhs_contained: bool
    poly_diff_contained: bool
    final_image_contained: bool

    def to_json(self) -> dict[str, Any]:
        return {
            "component": self.component,
            "exact_raw_rhs_residual": self.exact_raw_rhs_residual.to_json(),
            "production_raw_rhs_remainder": self.production_raw_rhs_remainder.to_json(),
            "exact_poly_diff": self.exact_poly_diff.to_json(),
            "production_poly_diff": self.production_poly_diff.to_json(),
            "exact_final_image": self.exact_final_image.to_json(),
            "production_final_image": self.production_final_image.to_json(),
            "raw_rhs_contained": self.raw_rhs_contained,
            "poly_diff_contained": self.poly_diff_contained,
            "final_image_contained": self.final_image_contained,
        }


@dataclass(frozen=True)
class RefinementIterationCertificate:
    iteration: int
    components: tuple[RefinementComponentCertificate, ...]
    all_contained: bool
    method: str = "exact_binary64_rational_sparse_tensor_product_bernstein"

    def to_json(self) -> dict[str, Any]:
        return {
            "schema": "vdp_c2_refinement_iteration_oracle_v1",
            "iteration": self.iteration,
            "method": self.method,
            "all_contained": self.all_contained,
            "components": [component.to_json() for component in self.components],
        }


def verify_refinement_iteration(
    row: Mapping[str, Any],
    *,
    candidate_coefficient_hex: Sequence[Sequence[str]],
    candidate_exponents: Sequence[Sequence[int]],
    domain: Sequence[Sequence[float | str]],
    base_remainder: Sequence[Sequence[float | str]],
    tau_interval: Sequence[float | str],
    validation_eps: float | str,
) -> RefinementIterationCertificate:
    """Verify one proposed C2 image against an exact rational/Bernstein map."""

    if row.get("phase") != "post_accept_refinement" or int(row.get("refinement_iteration", 0)) <= 0:
        raise ValueError("row is not a C2 refinement proposal")
    if len(candidate_coefficient_hex) != 2:
        raise ValueError("the frozen VDP oracle requires exactly two candidate components")
    candidate = tuple(
        polynomial_from_binary64_hex(coefficients, candidate_exponents)
        for coefficients in candidate_coefficient_hex
    )
    base_dim = candidate[0].n_vars
    if candidate[1].n_vars != base_dim or len(domain) != base_dim:
        raise ValueError("candidate/domain dimension mismatch")
    joint_dim = base_dim + 2
    px = _extend(candidate[0], joint_dim) + _variable(joint_dim, base_dim)
    py = _extend(candidate[1], joint_dim) + _variable(joint_dim, base_dim + 1)

    raw_exponents = row["raw_rhs_polynomial_exponents"]
    raw_coefficients = row["raw_rhs_polynomial_coefficient_hex"][0]
    if len(raw_coefficients) != 2:
        raise ValueError("raw RHS trace does not have two components")
    retained_raw = tuple(
        _extend(polynomial_from_binary64_hex(coefficients, raw_exponents), joint_dim)
        for coefficients in raw_coefficients
    )
    exact_raw_polynomials = (
        py - retained_raw[0],
        (RationalPolynomial.constant(joint_dim, 1) - px * px) * py - px - retained_raw[1],
    )
    input_lo = row["input_remainder_lo"][0]
    input_hi = row["input_remainder_hi"][0]
    joint_domain = tuple(_interval(lo, hi) for lo, hi in domain) + (
        _interval(input_lo[0], input_hi[0]),
        _interval(input_lo[1], input_hi[1]),
    )

    poly_diff_exponents = row["poly_diff_exponents"]
    poly_diff_coefficients = row["poly_diff_coefficient_hex"][0]
    poly_diff = tuple(
        polynomial_from_binary64_hex(coefficients, poly_diff_exponents)
        for coefficients in poly_diff_coefficients
    )
    base_domain = tuple(_interval(lo, hi) for lo, hi in domain)
    raw_production_lo = row["raw_rhs_remainder_lo"][0]
    raw_production_hi = row["raw_rhs_remainder_hi"][0]
    diff_production_lo = row["poly_diff_range_lo"][0]
    diff_production_hi = row["poly_diff_range_hi"][0]
    final_lo = row["proposed_remainder_lo"][0]
    final_hi = row["proposed_remainder_hi"][0]
    tau = _interval(tau_interval[0], tau_interval[1])
    eps = RationalInterval.symmetric(_q(validation_eps))

    certificates = []
    for component in range(2):
        exact_raw = exact_raw_polynomials[component].bernstein_range(joint_domain)
        production_raw = _interval(raw_production_lo[component], raw_production_hi[component])
        exact_diff = poly_diff[component].bernstein_range(base_domain)
        production_diff = _interval(
            diff_production_lo[component],
            diff_production_hi[component],
        )
        base = _interval(base_remainder[component][0], base_remainder[component][1])
        exact_final = base + tau * exact_raw + eps + exact_diff + eps + eps
        production_final = _interval(final_lo[component], final_hi[component])
        certificates.append(
            RefinementComponentCertificate(
                component=component,
                exact_raw_rhs_residual=exact_raw,
                production_raw_rhs_remainder=production_raw,
                exact_poly_diff=exact_diff,
                production_poly_diff=production_diff,
                exact_final_image=exact_final,
                production_final_image=production_final,
                raw_rhs_contained=exact_raw.subseteq(production_raw),
                poly_diff_contained=exact_diff.subseteq(production_diff),
                final_image_contained=exact_final.subseteq(production_final),
            )
        )
    all_contained = all(
        component.raw_rhs_contained
        and component.poly_diff_contained
        and component.final_image_contained
        for component in certificates
    )
    return RefinementIterationCertificate(
        iteration=int(row["refinement_iteration"]),
        components=tuple(certificates),
        all_contained=all_contained,
    )


def assert_refinement_certificate(certificate: RefinementIterationCertificate) -> None:
    if certificate.all_contained:
        return
    failed = [
        {
            "component": component.component,
            "raw_rhs": component.raw_rhs_contained,
            "poly_diff": component.poly_diff_contained,
            "final_image": component.final_image_contained,
            "exact_final_width": fraction_text(component.exact_final_image.width),
        }
        for component in certificate.components
        if not (
            component.raw_rhs_contained
            and component.poly_diff_contained
            and component.final_image_contained
        )
    ]
    raise ValueError(f"refinement exact oracle containment failed: {failed}")


__all__ = [
    "RefinementComponentCertificate",
    "RefinementIterationCertificate",
    "assert_refinement_certificate",
    "polynomial_from_binary64_hex",
    "verify_refinement_iteration",
]
