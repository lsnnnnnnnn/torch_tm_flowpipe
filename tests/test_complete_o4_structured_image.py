from fractions import Fraction

import pytest
import torch

from torch_tm_flowpipe import BatchedMonomialBasis, BatchedPolynomial
from torch_tm_flowpipe.structured_fraction_oracle import (
    fraction_complete_polynomial_difference_oracle,
)
from torch_tm_flowpipe.structured_remainder import (
    complete_polynomial_structured_image,
)


DTYPE = torch.float64


def _polynomial(dim, order, terms, *, batch=1, outputs=1):
    basis = BatchedMonomialBasis.build(dim, order)
    coeffs = torch.zeros((batch, outputs, basis.num_terms), dtype=DTYPE)
    for output, exponent, value in terms:
        coeffs[:, output, basis.term_index(exponent)] = torch.as_tensor(value, dtype=DTYPE)
    return BatchedPolynomial(coeffs, basis)


def _identity_map(batch, variable_dim, structured_dim=None):
    structured_dim = variable_dim if structured_dim is None else structured_dim
    out = torch.zeros((batch, variable_dim, structured_dim), dtype=DTYPE)
    for index in range(min(variable_dim, structured_dim)):
        out[:, index, index] = 1.0
    return out


def _assert_contains_fraction(lo, hi, oracle):
    for batch in range(lo.shape[0]):
        for output in range(lo.shape[1]):
            assert Fraction.from_float(float(lo[batch, output])) <= oracle[batch][output].lo
            assert Fraction.from_float(float(hi[batch, output])) >= oracle[batch][output].hi


@pytest.mark.parametrize(
    "polynomial,base,z",
    [
        (
            _polynomial(1, 4, [(0, (0,), 3.0), (0, (1,), -2.0)]),
            ([-0.75], [1.25]),
            ([-0.2], [0.4]),
        ),
        (
            _polynomial(1, 4, [(0, (2,), 1.5)]),
            ([-1.0], [2.0]),
            ([-0.125], [0.375]),
        ),
        (
            _polynomial(2, 4, [(0, (1, 1), -3.0)]),
            ([-0.5, -0.25], [1.0, 0.75]),
            ([-0.2, -0.4], [0.3, 0.1]),
        ),
        (
            _polynomial(2, 4, [(0, (2, 1), 2.0)]),
            ([-0.75, -0.5], [1.25, 0.25]),
            ([-0.1, -0.2], [0.3, 0.4]),
        ),
        (
            _polynomial(2, 4, [(0, (3, 1), -0.75)]),
            ([-0.5, -1.0], [1.0, 0.5]),
            ([-0.25, -0.125], [0.375, 0.25]),
        ),
    ],
)
def test_complete_retained_monomials_contain_independent_fraction_oracle(polynomial, base, z):
    base_lo = torch.tensor([base[0]], dtype=DTYPE)
    base_hi = torch.tensor([base[1]], dtype=DTYPE)
    z_lo = torch.tensor([z[0]], dtype=DTYPE)
    z_hi = torch.tensor([z[1]], dtype=DTYPE)
    coordinate = _identity_map(1, polynomial.basis.dim)
    result = complete_polynomial_structured_image(
        polynomial,
        (base_lo, base_hi),
        (z_lo, z_hi),
        coordinate,
    )
    oracle = fraction_complete_polynomial_difference_oracle(
        polynomial.coeffs,
        polynomial.basis.exponents,
        base_lo,
        base_hi,
        z_lo,
        z_hi,
        coordinate,
        coordinate,
    )
    _assert_contains_fraction(
        result.total_difference_lo,
        result.total_difference_hi,
        oracle.total_difference,
    )
    assert result.containment_mask.tolist() == [True]


def test_affine_and_harmonic_maps_have_exactly_zero_nonlinear_residual():
    harmonic = _polynomial(
        2,
        4,
        [
            (0, (1, 0), 0.0),
            (0, (0, 1), 1.0),
            (1, (1, 0), -1.0),
            (1, (0, 1), 0.0),
        ],
        outputs=2,
    )
    base_lo = torch.tensor([[-1.0, -2.0]], dtype=DTYPE)
    base_hi = torch.tensor([[2.0, 3.0]], dtype=DTYPE)
    z_lo = torch.tensor([[-0.2, -0.4]], dtype=DTYPE)
    z_hi = torch.tensor([[0.3, 0.1]], dtype=DTYPE)
    result = complete_polynomial_structured_image(
        harmonic,
        (base_lo, base_hi),
        (z_lo, z_hi),
        _identity_map(1, 2),
    )
    assert torch.equal(result.nonlinear_residual_lo, torch.zeros_like(result.nonlinear_residual_lo))
    assert torch.equal(result.nonlinear_residual_hi, torch.zeros_like(result.nonlinear_residual_hi))
    assert result.total_difference_lo[0, 0] <= -0.4
    assert result.total_difference_hi[0, 0] >= 0.1
    assert result.total_difference_lo[0, 1] <= -0.3
    assert result.total_difference_hi[0, 1] >= 0.2


def test_duplicate_monomial_routes_and_cancellation_remain_outward():
    coefficients = torch.tensor([[[1.0, -1.0, 2.0 ** -1022]]], dtype=DTYPE)
    exponents = torch.tensor([[2], [2], [1]], dtype=torch.long)
    base_lo = torch.tensor([[-1.0]], dtype=DTYPE)
    base_hi = torch.tensor([[1.0]], dtype=DTYPE)
    z_lo = torch.tensor([[-2.0 ** -52]], dtype=DTYPE)
    z_hi = torch.tensor([[2.0 ** -51]], dtype=DTYPE)
    coordinate = torch.ones((1, 1, 1), dtype=DTYPE)
    result = complete_polynomial_structured_image(
        (coefficients, exponents),
        (base_lo, base_hi),
        (z_lo, z_hi),
        coordinate,
    )
    oracle = fraction_complete_polynomial_difference_oracle(
        coefficients,
        exponents,
        base_lo,
        base_hi,
        z_lo,
        z_hi,
        coordinate,
        coordinate,
    )
    _assert_contains_fraction(result.total_difference_lo, result.total_difference_hi, oracle.total_difference)


def test_endpoint_image_is_contained_in_independent_tube_image():
    # P(x,tau) = x + tau*x^2, with tau not perturbed by the structured box.
    polynomial = _polynomial(
        2,
        4,
        [(0, (1, 0), 1.0), (0, (2, 1), 1.0)],
    )
    base_lo = torch.tensor([[-0.5, 0.0]], dtype=DTYPE)
    base_hi = torch.tensor([[0.75, 0.1]], dtype=DTYPE)
    z_lo = torch.tensor([[-0.1]], dtype=DTYPE)
    z_hi = torch.tensor([[0.2]], dtype=DTYPE)
    coordinate = torch.tensor([[[1.0], [0.0]]], dtype=DTYPE)
    endpoint = complete_polynomial_structured_image(
        polynomial,
        (base_lo, base_hi),
        (z_lo, z_hi),
        coordinate,
        (torch.tensor([0.1], dtype=DTYPE), torch.tensor([0.1], dtype=DTYPE)),
        tau_index=1,
    )
    tube = complete_polynomial_structured_image(
        polynomial,
        (base_lo, base_hi),
        (z_lo, z_hi),
        coordinate,
        (torch.tensor([0.0], dtype=DTYPE), torch.tensor([0.1], dtype=DTYPE)),
        tau_index=1,
    )
    assert tube.total_difference_lo.item() <= endpoint.total_difference_lo.item()
    assert tube.total_difference_hi.item() >= endpoint.total_difference_hi.item()
    assert endpoint.domain_scope == "endpoint_tau_point"
    assert tube.domain_scope == "tube_tau_interval"


def test_batch_permutation_is_deterministic_and_nonfinite_fails_closed():
    polynomial = _polynomial(1, 4, [(0, (3,), 1.0)], batch=3)
    base_lo = torch.tensor([[-1.0], [-0.5], [-2.0]], dtype=DTYPE)
    base_hi = torch.tensor([[0.5], [1.0], [0.25]], dtype=DTYPE)
    z_lo = torch.tensor([[-0.1], [-0.2], [-0.3]], dtype=DTYPE)
    z_hi = torch.tensor([[0.3], [0.1], [0.2]], dtype=DTYPE)
    coordinate = _identity_map(3, 1)
    result = complete_polynomial_structured_image(
        polynomial,
        (base_lo, base_hi),
        (z_lo, z_hi),
        coordinate,
    )
    permutation = torch.tensor([2, 0, 1])
    permuted_polynomial = BatchedPolynomial(polynomial.coeffs[permutation], polynomial.basis)
    permuted = complete_polynomial_structured_image(
        permuted_polynomial,
        (base_lo[permutation], base_hi[permutation]),
        (z_lo[permutation], z_hi[permutation]),
        coordinate[permutation],
    )
    inverse = torch.argsort(permutation)
    assert torch.equal(result.total_difference_lo, permuted.total_difference_lo[inverse])
    assert torch.equal(result.total_difference_hi, permuted.total_difference_hi[inverse])

    bad = base_hi.clone()
    bad[0, 0] = torch.inf
    with pytest.raises(FloatingPointError, match="nonfinite"):
        complete_polynomial_structured_image(
            polynomial,
            (base_lo, bad),
            (z_lo, z_hi),
            coordinate,
        )
