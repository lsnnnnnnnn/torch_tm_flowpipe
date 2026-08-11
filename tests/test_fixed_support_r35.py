from __future__ import annotations

from fractions import Fraction

import pytest
import torch

from torch_tm_flowpipe.batched_dense_tm import BatchedMonomialBasis
from torch_tm_flowpipe.fixed_support import (
    FixedSupportDescriptor,
    FixedSupportPolynomial,
    diffreach_vdp_polynomial_rhs,
)


def _r35() -> FixedSupportDescriptor:
    return FixedSupportDescriptor.complete_total_degree(
        variable_names=("t", "xi0", "xi1"), order=4, local_time_index=0
    )


@pytest.mark.unit
def test_r35_has_complete_deterministic_support_and_tables() -> None:
    support = _r35()
    dense = BatchedMonomialBasis.build(3, 4, "cpu")
    assert support.num_slots == 35
    assert support.exponents == tuple(tuple(row) for row in dense.exponents.tolist())
    assert support.support_sha256 == _r35().support_sha256
    tables = support.manifest()["complete_algebra_tables"]
    assert len(tables["multiplication"]) == 35 * 35
    assert len(tables["integration"]) == 35
    assert len(tables["endpoint_substitution"]) == 35


@pytest.mark.unit
def test_r35_every_product_destination_and_overflow_is_exact() -> None:
    support = _r35()
    table = support.manifest()["complete_algebra_tables"]["multiplication"]
    for row in table:
        left = support.exponents[row["left_slot"]]
        right = support.exponents[row["right_slot"]]
        product = tuple(a + b for a, b in zip(left, right))
        assert tuple(row["product_exponent"]) == product
        assert row["overflow"] == (sum(product) > 4)
        if row["overflow"]:
            assert row["output_slot"] is None
        else:
            assert row["output_slot"] == support.slot(product)


@pytest.mark.unit
def test_r35_monomial_multiply_integrate_differentiate_and_endpoint() -> None:
    support = _r35()
    for left_slot, left_exp in enumerate(support.exponents):
        left = torch.zeros((1, 1, 35), dtype=torch.float64)
        left[..., left_slot] = 1.0
        left_poly = FixedSupportPolynomial(left, support)
        for right_slot, right_exp in enumerate(support.exponents):
            right = torch.zeros_like(left)
            right[..., right_slot] = 1.0
            product = left_poly.mul_trunc(FixedSupportPolynomial(right, support))
            exponent = tuple(a + b for a, b in zip(left_exp, right_exp))
            if sum(exponent) <= 4:
                expected = torch.zeros_like(left)
                expected[..., support.slot(exponent)] = 1.0
                assert torch.equal(product.coeffs, expected)
            else:
                assert torch.count_nonzero(product.coeffs) == 0

        derivative = left_poly.differentiate(1)
        expected_derivative = torch.zeros_like(left)
        if left_exp[1]:
            out = list(left_exp)
            out[1] -= 1
            expected_derivative[..., support.slot(out)] = left_exp[1]
        assert torch.equal(derivative.coeffs, expected_derivative)

        integrated = left_poly.integrate_time_trunc()
        expected_integrated = torch.zeros_like(left)
        integrated_exp = list(left_exp)
        denominator = integrated_exp[0] + 1
        integrated_exp[0] += 1
        if sum(integrated_exp) <= 4:
            expected_integrated[..., support.slot(integrated_exp)] = 1.0 / denominator
        assert torch.equal(integrated.coeffs, expected_integrated)

        endpoint = left_poly.evaluate_time(0.25)
        reduced = list(left_exp)
        power = reduced[0]
        reduced[0] = 0
        expected_endpoint = torch.zeros_like(left)
        expected_endpoint[..., support.slot(reduced)] = 0.25**power
        assert torch.equal(endpoint.coeffs, expected_endpoint)


@pytest.mark.unit
def test_r35_sparse_roundtrip_and_degree_fixtures() -> None:
    support = _r35()
    coefficients = torch.zeros((1, 1, 35), dtype=torch.float64)
    sparse = {
        (0, 0, 0): 1.0,
        (0, 1, 0): -2.0,
        (1, 1, 0): 3.0,
        (0, 0, 3): -4.0,
        (0, 2, 2): 5.0,
    }
    for exponent, value in sparse.items():
        coefficients[..., support.slot(exponent)] = value
    polynomial = FixedSupportPolynomial(coefficients, support)
    recovered = {
        exponent: float(polynomial.coeffs[..., slot])
        for slot, exponent in enumerate(support.exponents)
        if float(polynomial.coeffs[..., slot]) != 0.0
    }
    assert recovered == sparse


@pytest.mark.unit
@pytest.mark.parametrize("batch", [1, 64])
def test_r35_vdp_rhs_exact_support_and_batch_shape(batch: int) -> None:
    support = _r35()
    coefficients = torch.zeros((batch, 2, 35), dtype=torch.float64)
    coefficients[:, 0, support.slot((0, 1, 0))] = 1.0
    coefficients[:, 1, support.slot((0, 0, 1))] = 1.0
    state = FixedSupportPolynomial(coefficients, support)
    box_lo = torch.tensor([[0.0, -1.0, -1.0]], dtype=torch.float64).expand(batch, -1)
    box_hi = torch.tensor([[0.01, 1.0, 1.0]], dtype=torch.float64).expand(batch, -1)
    rhs = diffreach_vdp_polynomial_rhs(state, box_lo, box_hi)
    assert rhs.coeffs.shape == (batch, 2, 35)
    assert torch.equal(rhs.coeffs[:, 0], coefficients[:, 1])
    assert torch.all(rhs.coeffs[:, 1, support.slot((0, 0, 1))] == 1.0)
    assert torch.all(rhs.coeffs[:, 1, support.slot((0, 1, 0))] == -1.0)
    assert torch.all(rhs.coeffs[:, 1, support.slot((0, 2, 1))] == -1.0)
    assert torch.count_nonzero(rhs.coeffs[:, 1]) == 3 * batch


@pytest.mark.unit
def test_r35_fraction_fixture_matches_binary64_coefficients() -> None:
    support = _r35()
    a = Fraction(3, 8)
    b = Fraction(-5, 16)
    left = torch.zeros((1, 1, 35), dtype=torch.float64)
    right = torch.zeros_like(left)
    left[..., support.slot((1, 1, 0))] = float(a)
    right[..., support.slot((0, 0, 2))] = float(b)
    product = FixedSupportPolynomial(left, support).mul_trunc(
        FixedSupportPolynomial(right, support)
    )
    value = product.coeffs[..., support.slot((1, 1, 2))].item()
    assert Fraction.from_float(value) == a * b


@pytest.mark.unit
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_r35_cpu_cuda_polynomial_decision_parity() -> None:
    support = _r35()
    generator = torch.Generator().manual_seed(20260811)
    left = torch.randn((64, 2, 35), generator=generator, dtype=torch.float64)
    right = torch.randn((64, 2, 35), generator=generator, dtype=torch.float64)
    cpu = FixedSupportPolynomial(left, support).mul_trunc(FixedSupportPolynomial(right, support))
    cuda = FixedSupportPolynomial(left.cuda(), support).mul_trunc(
        FixedSupportPolynomial(right.cuda(), support)
    )
    assert torch.allclose(cpu.coeffs, cuda.coeffs.cpu(), rtol=1e-12, atol=1e-12)
    assert torch.equal(torch.isfinite(cpu.coeffs), torch.isfinite(cuda.coeffs.cpu()))
