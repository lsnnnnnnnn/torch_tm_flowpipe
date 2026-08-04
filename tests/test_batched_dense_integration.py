import torch

from torch_tm_flowpipe import Interval, Polynomial, TaylorModel, TMVector
from torch_tm_flowpipe.batched_dense_tm import (
    BatchedMonomialBasis,
    dense_to_sparse_tmvector,
    sparse_tmvector_to_dense,
)


def test_dense_integration_matches_sparse_physical_tau_semantics():
    domain = [Interval(-1.0, 1.0), Interval(0.0, 0.25)]
    poly = Polynomial({(0, 0): 2.0, (1, 1): 3.0, (0, 3): -4.0}, n_vars=2)
    sparse = TMVector([TaylorModel(poly, Interval(-0.1, 0.2), domain, order=4)])
    dense = sparse_tmvector_to_dense(sparse, order=4)

    integrated = dense.integrate(1)
    sparse_integrated = sparse[0].integrate(1)
    roundtrip = dense_to_sparse_tmvector(integrated)[0]

    assert set(roundtrip.polynomial.terms) == set(sparse_integrated.polynomial.terms)
    for exponent, expected in sparse_integrated.polynomial.terms.items():
        assert torch.allclose(roundtrip.polynomial.terms[exponent], expected, atol=0.0, rtol=0.0)
    assert float(roundtrip.remainder.lo) <= float(sparse_integrated.remainder.lo)
    assert float(roundtrip.remainder.hi) >= float(sparse_integrated.remainder.hi)


def test_constant_integral_uses_h_once_not_twice():
    basis = BatchedMonomialBasis.build(2, 4)
    domain = [Interval(-1.0, 1.0), Interval(0.0, 0.2)]
    sparse = TMVector([TaylorModel.constant(3.0, domain, order=4)])
    dense = sparse_tmvector_to_dense(sparse, order=4)
    integrated = dense.integrate(1)
    tau_slot = basis.term_index((0, 1))
    assert torch.allclose(integrated.poly.coeffs[0, 0, tau_slot], torch.tensor(3.0, dtype=torch.float64))
    endpoint = integrated.endpoint(1, 0.2)
    assert torch.allclose(
        endpoint.poly.coeffs[0, 0, endpoint.poly.basis.constant_index],
        torch.tensor(0.6, dtype=torch.float64),
    )
