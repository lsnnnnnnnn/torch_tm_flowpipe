import itertools

import pytest
import torch

from torch_tm_flowpipe import Interval, Polynomial, TaylorModel, TMVector
from torch_tm_flowpipe.batched_dense_tm import (
    BatchedMonomialBasis,
    BatchedPolynomial,
    DenseExecutionCounters,
    dense_to_sparse_tmvector,
    sparse_tmvector_to_dense,
)


def _sparse_from_coefficients(coefficients, basis):
    terms = {}
    for index, exponent in enumerate(basis.exponents.tolist()):
        value = coefficients[index]
        if bool(value != 0):
            terms[tuple(exponent)] = value
    return Polynomial(terms, n_vars=basis.dim)


@pytest.mark.parametrize("n_vars,order", [(1, 1), (1, 4), (2, 2), (2, 4), (3, 2), (3, 4)])
def test_random_multiply_retained_and_grouped_overflow_match_sparse(n_vars, order):
    basis = BatchedMonomialBasis.build(n_vars, order)
    generator = torch.Generator().manual_seed(1000 + 10 * n_vars + order)
    left_coeffs = torch.randn((3, 1, basis.num_terms), dtype=torch.float64, generator=generator) * 0.1
    right_coeffs = torch.randn((3, 1, basis.num_terms), dtype=torch.float64, generator=generator) * 0.1
    left = BatchedPolynomial(left_coeffs, basis)
    right = BatchedPolynomial(right_coeffs, basis)
    domain_lo = torch.full((3, n_vars), -1.0, dtype=torch.float64)
    domain_hi = torch.full((3, n_vars), 1.0, dtype=torch.float64)
    dense_product, dense_lo, dense_hi = left.mul_trunc(
        right,
        return_truncation_bound=True,
        domain_lo=domain_lo,
        domain_hi=domain_hi,
    )

    sparse_domain = [Interval(-1.0, 1.0) for _ in range(n_vars)]
    for batch in range(3):
        sparse_left = _sparse_from_coefficients(left_coeffs[batch, 0], basis)
        sparse_right = _sparse_from_coefficients(right_coeffs[batch, 0], basis)
        kept, dropped = sparse_left.mul_truncate(sparse_right, order)
        expected = torch.zeros(basis.num_terms, dtype=torch.float64)
        for exponent, coefficient in kept.terms.items():
            expected[basis.term_index(exponent)] = coefficient
        assert torch.allclose(dense_product.coeffs[batch, 0], expected, atol=2e-15, rtol=2e-15)
        sparse_overflow = dropped.evaluate_interval(sparse_domain)
        assert float(dense_lo[batch, 0]) <= float(sparse_overflow.lo) + 2e-15
        assert float(dense_hi[batch, 0]) >= float(sparse_overflow.hi) - 2e-15


def test_equal_dropped_exponents_are_combined_before_intervalization():
    basis = BatchedMonomialBasis.build(1, 2)
    left_coeffs = torch.zeros((1, 1, basis.num_terms), dtype=torch.float64)
    right_coeffs = torch.zeros_like(left_coeffs)
    left_coeffs[0, 0, basis.term_index((1,))] = 1.0
    left_coeffs[0, 0, basis.term_index((2,))] = 1.0
    right_coeffs[0, 0, basis.term_index((1,))] = -1.0
    right_coeffs[0, 0, basis.term_index((2,))] = 1.0
    left = BatchedPolynomial(left_coeffs, basis)
    right = BatchedPolynomial(right_coeffs, basis)
    domain_lo = torch.tensor([[-1.0]], dtype=torch.float64)
    domain_hi = torch.tensor([[1.0]], dtype=torch.float64)
    _, merged_lo, merged_hi = left.mul_trunc(
        right,
        return_truncation_bound=True,
        domain_lo=domain_lo,
        domain_hi=domain_hi,
        dropped_merge_mode="merged",
    )
    _, termwise_lo, termwise_hi = left.mul_trunc(
        right,
        return_truncation_bound=True,
        domain_lo=domain_lo,
        domain_hi=domain_hi,
        dropped_merge_mode="termwise",
    )
    # The two cubic routes cancel; only x^4 remains after exponent grouping.
    assert float(merged_lo) <= 0.0
    assert float(merged_hi) >= 1.0
    assert float(merged_hi - merged_lo) < float(termwise_hi - termwise_lo)


def test_sparse_dense_roundtrip_maps_by_exponent_and_counts_only_boundaries():
    domain = [Interval(-1.0, 1.0), Interval(-0.25, 0.5)]
    models = TMVector(
        [
            TaylorModel(Polynomial({(0, 0): 1.0, (1, 0): 2.0, (0, 2): -3.0}, 2), Interval(-0.01, 0.02), domain, order=3),
            TaylorModel(Polynomial({(0, 1): 4.0, (2, 0): 5.0}, 2), Interval(-0.03, 0.04), domain, order=3),
        ]
    )
    counters = DenseExecutionCounters()
    dense = sparse_tmvector_to_dense(models, order=3, counters=counters)
    roundtrip = dense_to_sparse_tmvector(dense, counters=counters)

    assert counters.sparse_to_dense_conversions == 1
    assert counters.dense_to_sparse_conversions == 1
    assert counters.segment_boundary_conversions == 2
    assert counters.inner_loop_conversions == 0
    assert counters.sparse_fallback_count == 0
    for actual, expected in zip(roundtrip, models):
        assert actual.polynomial.terms.keys() == expected.polynomial.terms.keys()
        for exponent in actual.polynomial.terms:
            assert torch.equal(actual.polynomial.terms[exponent], expected.polynomial.terms[exponent])
        assert float(actual.remainder.lo) <= float(expected.remainder.lo)
        assert float(actual.remainder.hi) >= float(expected.remainder.hi)


def test_sparse_conversion_rejects_out_of_basis_term():
    domain = [Interval(-1.0, 1.0)]
    sparse = TMVector([TaylorModel(Polynomial({(3,): 1.0}, 1), Interval.zero(), domain, order=3)])
    with pytest.raises(ValueError, match="outside the dense order-2 basis"):
        sparse_tmvector_to_dense(sparse, order=2)


def test_strict_binary_shapes_do_not_silently_broadcast_batch_or_output():
    basis = BatchedMonomialBasis.build(2, 2)
    a = BatchedPolynomial.zeros(1, 2, basis)
    b = BatchedPolynomial.zeros(3, 2, basis)
    c = BatchedPolynomial.zeros(1, 1, basis)
    for other in (b, c):
        with pytest.raises(ValueError, match="identical"):
            a.add(other)
        with pytest.raises(ValueError, match="identical"):
            a.mul_trunc(other)

