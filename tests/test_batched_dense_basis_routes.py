import math

import pytest
import torch

from torch_tm_flowpipe.batched_dense_tm import BatchedMonomialBasis, BatchedPolynomial


@pytest.mark.parametrize("n_vars", [1, 2, 3])
@pytest.mark.parametrize("order", [1, 2, 3, 4])
def test_complete_basis_count_fingerprint_and_cache(n_vars, order):
    basis = BatchedMonomialBasis.build(n_vars, order, "cpu")
    repeated = BatchedMonomialBasis.build(n_vars, order, "cpu")
    assert basis is repeated
    assert basis.num_terms == math.comb(n_vars + order, order)
    assert len(basis.exponent_to_index) == basis.num_terms
    assert len(set(basis.exponent_to_index)) == basis.num_terms
    assert basis.fingerprint == repeated.fingerprint
    assert basis.fingerprint == basis.to("cpu").fingerprint
    assert basis.constant_index == basis.term_index((0,) * n_vars)
    assert len(basis.linear_indices) == n_vars


def test_vdp_order4_basis_has_tau_and_35_slots():
    basis = BatchedMonomialBasis.build(3, 4)
    assert basis.num_terms == 35
    assert basis.term_index((0, 0, 1)) in basis.linear_indices


def test_integration_route_keeps_and_exposes_overflow():
    basis = BatchedMonomialBasis.build(2, 3)
    coeffs = torch.zeros((1, 1, basis.num_terms), dtype=torch.float64)
    coeffs[0, 0, basis.term_index((0, 0))] = 2.0
    coeffs[0, 0, basis.term_index((0, 2))] = 3.0
    coeffs[0, 0, basis.term_index((1, 2))] = 4.0
    poly = BatchedPolynomial(coeffs, basis)
    domain_lo = torch.tensor([[-1.0, 0.0]], dtype=torch.float64)
    domain_hi = torch.tensor([[1.0, 0.5]], dtype=torch.float64)

    integrated, overflow_lo, overflow_hi = poly.integrate(
        1,
        domain_lo=domain_lo,
        domain_hi=domain_hi,
        return_overflow_bound=True,
    )

    assert integrated.coeffs[0, 0, basis.term_index((0, 1))] == 2.0
    assert integrated.coeffs[0, 0, basis.term_index((0, 3))] == 1.0
    # 4*x*tau^2 integrates to (4/3)*x*tau^3, which is degree four.
    assert float(overflow_lo[0, 0]) <= -(4.0 / 3.0) * 0.5**3
    assert float(overflow_hi[0, 0]) >= (4.0 / 3.0) * 0.5**3


def test_degree_specific_multiply_route_moves_degree_four_to_overflow():
    basis = BatchedMonomialBasis.build(1, 4)
    coeffs = torch.zeros((1, 1, basis.num_terms), dtype=torch.float64)
    coeffs[0, 0, basis.term_index((2,))] = 2.0
    poly = BatchedPolynomial(coeffs, basis)
    kept, lo, hi = poly.mul_trunc(
        poly,
        max_degree=3,
        return_truncation_bound=True,
        domain_lo=torch.tensor([[-1.0]], dtype=torch.float64),
        domain_hi=torch.tensor([[1.0]], dtype=torch.float64),
    )
    assert torch.count_nonzero(kept.coeffs) == 0
    assert float(lo[0, 0]) <= 0.0
    assert float(hi[0, 0]) >= 4.0

