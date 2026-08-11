from fractions import Fraction

import torch

from torch_tm_flowpipe import BatchedMonomialBasis, BatchedPolynomial
from torch_tm_flowpipe.fixed_support_outward import OutwardIntervalTensor
from torch_tm_flowpipe.structured_fraction_oracle import (
    fraction_complete_polynomial_difference_oracle,
)
from torch_tm_flowpipe.structured_remainder import (
    compare_complete_polynomial_contracts,
)


DTYPE = torch.float64


def _polynomial(dim, terms, *, outputs=1):
    basis = BatchedMonomialBasis.build(dim, 4)
    coefficients = torch.zeros((1, outputs, basis.num_terms), dtype=DTYPE)
    for output, exponent, value in terms:
        coefficients[0, output, basis.term_index(exponent)] = value
    return BatchedPolynomial(coefficients, basis)


def _comparison(polynomial, q_lo, q_hi, ordinary_lo, ordinary_hi, z_lo, z_hi):
    q = OutwardIntervalTensor(
        torch.tensor([q_lo], dtype=DTYPE),
        torch.tensor([q_hi], dtype=DTYPE),
    )
    ordinary = OutwardIntervalTensor(
        torch.tensor([ordinary_lo], dtype=DTYPE),
        torch.tensor([ordinary_hi], dtype=DTYPE),
    )
    structured = OutwardIntervalTensor(
        torch.tensor([z_lo], dtype=DTYPE),
        torch.tensor([z_hi], dtype=DTYPE),
    )
    current_base = q.add(ordinary)
    coordinate = torch.eye(len(q_lo), dtype=DTYPE).unsqueeze(0)
    result = compare_complete_polynomial_contracts(
        polynomial,
        polynomial_base_domain=(q.lo, q.hi),
        current_base_domain=(current_base.lo, current_base.hi),
        ordinary_box=(ordinary.lo, ordinary.hi),
        structured_box=(structured.lo, structured.hi),
        coordinate_map=coordinate,
    )
    return result, q, ordinary.add(structured), coordinate


def _contains_fraction(lo, hi, exact):
    assert Fraction.from_float(float(lo)) <= exact.lo
    assert Fraction.from_float(float(hi)) >= exact.hi


def test_total_delta_is_exact_affine_for_two_dimensional_harmonic_map():
    harmonic = _polynomial(
        2,
        [
            (0, (0, 1), 1.0),
            (1, (1, 0), -1.0),
        ],
        outputs=2,
    )
    comparison, _, _, _ = _comparison(
        harmonic,
        [-1.0, -2.0],
        [2.0, 3.0],
        [-0.1, -0.2],
        [0.15, 0.25],
        [-0.05, -0.125],
        [0.075, 0.1],
    )
    total = comparison.total_delta_image
    assert torch.equal(
        total.nonlinear_residual_lo,
        torch.zeros_like(total.nonlinear_residual_lo),
    )
    assert torch.equal(
        total.nonlinear_residual_hi,
        torch.zeros_like(total.nonlinear_residual_hi),
    )
    assert torch.equal(total.reconstruction_lo, total.total_difference_lo)
    assert torch.equal(total.reconstruction_hi, total.total_difference_hi)


def test_total_delta_includes_ordinary_structured_quadratic_interaction_once():
    quadratic = _polynomial(1, [(0, (2,), 1.0)])
    comparison, q, delta, coordinate = _comparison(
        quadratic,
        [1.0],
        [1.0],
        [0.25],
        [0.25],
        [0.125],
        [0.125],
    )
    oracle = fraction_complete_polynomial_difference_oracle(
        quadratic.coeffs,
        quadratic.basis.exponents,
        q.lo,
        q.hi,
        delta.lo,
        delta.hi,
        coordinate,
        coordinate,
    )
    exact = oracle.total_difference[0][0]
    _contains_fraction(
        comparison.total_delta_reconstruction_lo[0, 0],
        comparison.total_delta_reconstruction_hi[0, 0],
        exact,
    )
    assert exact.lo <= Fraction(57, 64) <= exact.hi
    assert not comparison.current_contains_total_delta_mask.item()
    # N_total = Delta^2 = R_o^2 + 2 R_o Z + Z^2.  The cross term is
    # therefore represented once inside the single materialized Delta image.
    nonlinear = oracle.nonlinear_residual[0][0]
    assert nonlinear.lo <= Fraction(9, 64) <= nonlinear.hi


def test_both_contract_primitives_contain_their_fraction_oracles_on_asymmetric_boxes():
    polynomial = _polynomial(
        2,
        [
            (0, (3, 1), -0.75),
            (0, (1, 1), 0.5),
        ],
    )
    comparison, q, delta, coordinate = _comparison(
        polynomial,
        [-0.5, -1.0],
        [1.0, 0.5],
        [-0.08, -0.03],
        [0.11, 0.07],
        [-0.25, -0.125],
        [0.375, 0.25],
    )
    oracle = fraction_complete_polynomial_difference_oracle(
        polynomial.coeffs,
        polynomial.basis.exponents,
        q.lo,
        q.hi,
        delta.lo,
        delta.hi,
        coordinate,
        coordinate,
    )
    _contains_fraction(
        comparison.total_delta_image.total_difference_lo[0, 0],
        comparison.total_delta_image.total_difference_hi[0, 0],
        oracle.total_difference[0][0],
    )
    assert comparison.total_delta_image.containment_mask.tolist() == [True]
    assert comparison.current_image.containment_mask.tolist() == [True]
