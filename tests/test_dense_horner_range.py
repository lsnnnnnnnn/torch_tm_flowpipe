import hashlib
import json
import math

import pytest
import torch

from torch_tm_flowpipe import (
    BatchedMonomialBasis,
    BatchedPolynomial,
    DenseRangePolicy,
    Interval,
    Polynomial,
    TMVector,
    canonicalize_dense_polynomial,
    evaluate_dense_horner_range,
    evaluate_dense_registered_horner_range,
    flowpipe_step_flowstar_style_adaptive,
    registered_dense_horner_orders,
)


DTYPE = torch.float64


def _data(terms, *, dim, batch=1, outputs=1):
    exponents = torch.tensor([term[0] for term in terms], dtype=torch.long).reshape(-1, dim)
    values = torch.tensor([term[1] for term in terms], dtype=DTYPE)
    coeffs = values.view(1, 1, -1).repeat(batch, outputs, 1)
    if outputs > 1:
        coeffs[:, 1::2] *= -0.75
    return coeffs, exponents


def _evaluate_samples(coeffs, exponents, points):
    monomials = points[:, :, None, :].pow(exponents[None, None, :, :]).prod(dim=-1)
    return torch.einsum("bnt,bot->bno", monomials, coeffs)


def _assert_contains_samples(result, coeffs, exponents, domain_lo, domain_hi, *, count=257):
    generator = torch.Generator(device=domain_lo.device).manual_seed(20260805)
    unit = torch.rand(
        (coeffs.shape[0], count, exponents.shape[1]),
        generator=generator,
        dtype=coeffs.dtype,
        device=coeffs.device,
    )
    points = domain_lo[:, None, :] + unit * (domain_hi - domain_lo)[:, None, :]
    values = _evaluate_samples(coeffs, exponents.to(coeffs.device), points)
    assert torch.all(values >= result.lo[:, None, :])
    assert torch.all(values <= result.hi[:, None, :])


@pytest.mark.parametrize(
    ("terms", "dim", "domain", "reference"),
    [
        ([((0,), 2.5)], 1, ((-3.0,), (7.0,)), (2.5, 2.5)),
        ([((0, 0), 1.0), ((1, 0), 2.0), ((0, 1), -3.0)], 2, ((-1.0, 2.0), (2.0, 4.0)), (-13.0, -1.0)),
        ([((3,), 1.0)], 1, ((-2.0,), (3.0,)), (-8.0, 27.0)),
        ([((4,), 1.0)], 1, ((-2.0,), (3.0,)), (0.0, 81.0)),
        ([((0,), 3.0), ((1,), -2.0), ((3,), 0.5)], 1, ((2.0,), (4.0,)), (3.0, 27.0)),
        ([((0,), -1.0), ((2,), 2.0)], 1, ((-4.0,), (-2.0,)), (7.0, 31.0)),
        ([((0,), 1.0), ((1,), -3.0), ((5,), 0.25)], 1, ((1.125,), (1.1250001,)), (-1.924491982098353, -1.9244918823242188)),
        ([((0, 0), 1.0), ((1, 1), -2.0), ((2, 0), 0.5)], 2, ((0.25, -1.0), (0.25, -1.0)), (1.53125, 1.53125)),
    ],
)
def test_analytic_constant_affine_power_shifted_narrow_and_zero_width_cases(terms, dim, domain, reference):
    coeffs, exponents = _data(terms, dim=dim)
    lo = torch.tensor([domain[0]], dtype=DTYPE)
    hi = torch.tensor([domain[1]], dtype=DTYPE)
    for order in registered_dense_horner_orders(dim):
        result = evaluate_dense_horner_range(coeffs, exponents, lo, hi, order)
        assert result.validated
        assert result.reconstruction_valid
        assert float(result.lo) <= reference[0] + 1e-12
        assert float(result.hi) >= reference[1] - 1e-12
        _assert_contains_samples(result, coeffs, exponents, lo, hi)


@pytest.mark.parametrize("degree", range(4, 13))
def test_degrees_four_through_twelve_and_missing_degrees(degree):
    terms = [((0,), -0.25), ((2,), 0.5), ((degree,), -1.0 if degree % 3 == 0 else 1.0)]
    coeffs, exponents = _data(terms, dim=1)
    lo = torch.tensor([[-1.25]], dtype=DTYPE)
    hi = torch.tensor([[0.75]], dtype=DTYPE)
    result = evaluate_dense_horner_range(coeffs, exponents, lo, hi, (0,))
    assert result.validated
    assert any(stage["degree"] == degree - 1 for stage in result.stages)
    _assert_contains_samples(result, coeffs, exponents, lo, hi, count=1025)


def test_mixed_monomials_missing_degrees_and_all_registered_orders_reconstruct():
    terms = [
        ((0, 0, 0), 0.75),
        ((3, 0, 0), -1.25),
        ((1, 2, 0), 0.5),
        ((0, 1, 4), -0.125),
        ((2, 0, 3), 0.875),
    ]
    coeffs, exponents = _data(terms, dim=3)
    lo = torch.tensor([[-1.0, -0.5, 0.0]], dtype=DTYPE)
    hi = torch.tensor([[1.0, 1.5, 0.1]], dtype=DTYPE)
    assert registered_dense_horner_orders(3) == ((0, 1, 2), (1, 0, 2), (2, 0, 1))
    for order in registered_dense_horner_orders(3):
        result = evaluate_dense_horner_range(coeffs, exponents, lo, hi, order)
        assert result.reconstructed_exponents == tuple(sorted(term[0] for term in terms))
        assert result.reconstruction_valid
        assert {stage["variable"] for stage in result.stages} == {0, 1, 2}
        _assert_contains_samples(result, coeffs, exponents, lo, hi)


def test_equal_exponent_duplicates_exact_near_cancellation_and_aggregation_envelope():
    terms = [
        ((2,), 1.0e16),
        ((2,), -1.0e16),
        ((2,), 3.0),
        ((1,), 1.0),
        ((1,), -math.nextafter(1.0, 0.0)),
        ((0,), -2.0),
    ]
    coeffs, exponents = _data(terms, dim=1)
    before_coeffs = coeffs.clone()
    before_exponents = exponents.clone()
    canonical = canonicalize_dense_polynomial(coeffs, exponents)
    assert canonical.source_term_count == 6
    assert canonical.unique_term_count == 3
    assert canonical.duplicate_group_count == 2
    x2 = canonical.exponent_tuples.index((2,))
    assert float(canonical.coefficient_lo[..., x2]) <= 3.0 <= float(canonical.coefficient_hi[..., x2])
    assert float(canonical.coefficient_hi[..., x2] - canonical.coefficient_lo[..., x2]) > 0.0
    assert torch.equal(coeffs, before_coeffs)
    assert torch.equal(exponents, before_exponents)
    result = evaluate_dense_horner_range(
        coeffs,
        exponents,
        torch.tensor([[-1.0]], dtype=DTYPE),
        torch.tensor([[1.0]], dtype=DTYPE),
        (0,),
    )
    assert result.validated
    # Use an exact-order-independent summation here: direct tensor reduction of
    # the deliberately catastrophic 1e16/-1e16 routes is not a faithful sample
    # of the mathematical polynomial being enclosed.
    for x in torch.linspace(-1.0, 1.0, 257, dtype=DTYPE).tolist():
        value = math.fsum(float(coefficient) * x ** exponent[0] for exponent, coefficient in terms)
        assert float(result.lo) <= value <= float(result.hi)


def test_alternating_huge_and_subnormal_coefficients_remain_safeguarded_float64():
    smallest = torch.nextafter(torch.tensor(0.0, dtype=DTYPE), torch.tensor(1.0, dtype=DTYPE)).item()
    huge = torch.finfo(DTYPE).max / 32.0
    terms = [((degree,), (-1.0 if degree % 2 else 1.0) * (huge if degree == 0 else 2.0 ** -degree)) for degree in range(13)]
    terms.extend([((1,), smallest), ((1,), -smallest)])
    coeffs, exponents = _data(terms, dim=1)
    lo = torch.tensor([[0.0]], dtype=DTYPE)
    hi = torch.tensor([[0.25]], dtype=DTYPE)
    result = evaluate_dense_horner_range(coeffs, exponents, lo, hi, (0,))
    assert result.validated
    assert torch.isfinite(result.lo).all() and torch.isfinite(result.hi).all()
    _assert_contains_samples(result, coeffs, exponents, lo, hi)


@pytest.mark.parametrize("batch", [1, 16, 48])
def test_batch_multiple_components_natural_horner_and_randomized_sample_containment(batch):
    generator = torch.Generator().manual_seed(17 + batch)
    exponents = torch.tensor(
        [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (2, 1, 0), (1, 2, 1), (0, 0, 4)],
        dtype=torch.long,
    )
    coeffs = torch.randn((batch, 3, exponents.shape[0]), generator=generator, dtype=DTYPE) * 0.25
    lo = torch.tensor([[-1.0, -0.75, 0.0]], dtype=DTYPE).repeat(batch, 1)
    hi = torch.tensor([[1.0, 1.25, 0.05]], dtype=DTYPE).repeat(batch, 1)
    registered = evaluate_dense_registered_horner_range(coeffs, exponents, lo, hi)
    assert registered.validated
    _assert_contains_samples(registered, coeffs, exponents, lo, hi, count=97)

    basis = BatchedMonomialBasis.build(3, 4)
    dense_coeffs = torch.zeros((batch, 3, basis.num_terms), dtype=DTYPE)
    for term_index, exponent in enumerate(exponents.tolist()):
        dense_coeffs[..., basis.term_index(exponent)] = coeffs[..., term_index]
    polynomial = BatchedPolynomial(dense_coeffs, basis)
    result = polynomial.range_bound(
        lo,
        hi,
        policy=DenseRangePolicy(method="horner_registered_best"),
        return_result=True,
    )
    assert result.horner_report["validated"]
    values = _evaluate_samples(coeffs, exponents, lo[:, None, :].expand(-1, 1, -1))[:, 0]
    assert torch.all(values >= result.natural_lo) and torch.all(values <= result.natural_hi)
    assert torch.all(values >= result.horner_lo) and torch.all(values <= result.horner_hi)
    assert torch.all(values >= result.selected_lo) and torch.all(values <= result.selected_hi)


def test_dense_sparse_zero_width_parity_and_original_tensors_unchanged():
    terms = [((0, 0), 1.25), ((1, 0), -0.5), ((0, 2), 3.0), ((2, 1), -0.125)]
    coeffs, exponents = _data(terms, dim=2)
    point = torch.tensor([[0.375, -1.25]], dtype=DTYPE)
    coeff_hash = hashlib.sha256(coeffs.numpy().tobytes()).hexdigest()
    exponent_hash = hashlib.sha256(exponents.numpy().tobytes()).hexdigest()
    result = evaluate_dense_horner_range(coeffs, exponents, point, point, (1, 0))
    sparse = Polynomial({exponent: coefficient for exponent, coefficient in terms}, n_vars=2)
    sparse_value = sparse.evaluate_interval([Interval(0.375, 0.375), Interval(-1.25, -1.25)])
    exact = math.fsum(coefficient * 0.375 ** exponent[0] * (-1.25) ** exponent[1] for exponent, coefficient in terms)
    assert float(result.lo) <= exact <= float(result.hi)
    assert float(sparse_value.lo) <= exact <= float(sparse_value.hi)
    assert hashlib.sha256(coeffs.numpy().tobytes()).hexdigest() == coeff_hash
    assert hashlib.sha256(exponents.numpy().tobytes()).hexdigest() == exponent_hash


def test_registered_order_tie_break_is_lexicographic_and_repeated_cpu_run_is_stable():
    coeffs, exponents = _data([((0, 0, 0), 1.0)], dim=3, batch=16, outputs=2)
    lo = torch.tensor([[-1.0, -1.0, 0.0]], dtype=DTYPE).repeat(16, 1)
    hi = torch.tensor([[1.0, 1.0, 0.1]], dtype=DTYPE).repeat(16, 1)
    first = evaluate_dense_registered_horner_range(coeffs, exponents, lo, hi)
    second = evaluate_dense_registered_horner_range(coeffs, exponents, lo, hi)
    assert torch.equal(first.lo, second.lo)
    assert torch.equal(first.hi, second.hi)
    assert torch.equal(first.selected_order_index, torch.zeros_like(first.selected_order_index))
    assert [row.variable_order for row in first.order_results] == sorted(registered_dense_horner_orders(3))
    assert [row.canonical.coefficient_interval_sha256 for row in first.order_results] == [
        row.canonical.coefficient_interval_sha256 for row in second.order_results
    ]
    assert json.dumps([row.stages for row in first.order_results], sort_keys=True) == json.dumps(
        [row.stages for row in second.order_results], sort_keys=True
    )


def test_nonfinite_fail_closed_and_invalid_horner_explicitly_falls_back_to_natural():
    coeffs = torch.tensor([[[-1.0e308, 1.0e308, 1.0e308]]], dtype=DTYPE)
    exponents = torch.tensor([[0], [1], [2]], dtype=torch.long)
    domain = torch.tensor([[1.0]], dtype=DTYPE)
    basis = BatchedMonomialBasis.build(1, 2)
    polynomial = BatchedPolynomial(coeffs, basis)
    result = polynomial.range_bound(domain, domain, method="horner_registered_best", return_result=True)
    assert result.selected_method == "natural"
    assert result.horner_report["validated"] is False
    assert result.fallback_reason.startswith("explicit natural fallback")
    assert torch.isfinite(result.selected_lo).all() and torch.isfinite(result.selected_hi).all()

    with pytest.raises(FloatingPointError, match="coefficients must be finite"):
        canonicalize_dense_polynomial(
            torch.tensor([[[torch.inf]]], dtype=DTYPE),
            torch.tensor([[0]], dtype=torch.long),
        )
    with pytest.raises(TypeError, match="integer dtype"):
        canonicalize_dense_polynomial(
            torch.tensor([[[1.0]]], dtype=DTYPE),
            torch.tensor([[0.5]], dtype=DTYPE),
        )
    with pytest.raises(ValueError, match="permutation"):
        evaluate_dense_horner_range(coeffs, exponents, domain, domain, (1,))


def test_subdivision_then_horner_selects_sound_enclosure_per_leaf():
    basis = BatchedMonomialBasis.build(2, 4)
    coeffs = torch.zeros((1, 1, basis.num_terms), dtype=DTYPE)
    for exponent, coefficient in {((0, 0)): 0.25, ((1, 0)): -1.0, ((2, 0)): 1.0, ((1, 2)): -0.5}.items():
        coeffs[0, 0, basis.term_index(exponent)] = coefficient
    polynomial = BatchedPolynomial(coeffs, basis)
    lo = torch.tensor([[-1.0, -1.0]], dtype=DTYPE)
    hi = torch.tensor([[1.0, 1.0]], dtype=DTYPE)
    result = polynomial.range_bound(
        lo,
        hi,
        policy=DenseRangePolicy(
            method="subdivision_then_horner",
            max_depth=1,
            max_leaves=4,
            split_vars=(0, 1),
        ),
        return_result=True,
    )
    assert result.coverage_report["valid"]
    assert result.horner_report["validated"]
    assert result.horner_report["per_leaf"]["validated"]
    assert result.cover.lo.shape[0] == 4
    points = torch.tensor([[[-1.0, -1.0], [-0.25, 0.5], [0.0, 0.0], [0.75, -0.5], [1.0, 1.0]]], dtype=DTYPE)
    values = polynomial.evaluate(points)
    assert torch.all(values >= result.selected_lo[:, None, :])
    assert torch.all(values <= result.selected_hi[:, None, :])


def test_harmonic_oscillator_one_step_regression_with_factorized_policy():
    def harmonic(x, u=None):
        return TMVector([x[1], -x[0]])

    segment = flowpipe_step_flowstar_style_adaptive(
        harmonic,
        [Interval(0.9, 1.1), Interval(-0.1, 0.1)],
        h=0.01,
        h_min=0.002,
        h_max=0.1,
        order=4,
        target_remainder_radius=1e-4,
        cutoff_threshold=1e-10,
        validation_mode="flowstar_raw_remainder_compat",
        reset_mode="normalized_insertion",
        step_policy_mode="flowstar_compat",
        tm_backend="dense",
        dense_range_policy=DenseRangePolicy(
            method="horner_registered_best",
            named_contexts=("polynomial_truncation",),
        ),
    )
    assert segment.status == "validated"
    endpoint = segment.endpoint_raw_tm.range_box()
    for x0, y0 in ((0.9, -0.1), (0.9, 0.1), (1.1, -0.1), (1.1, 0.1)):
        exact_x = x0 * math.cos(0.01) + y0 * math.sin(0.01)
        exact_y = y0 * math.cos(0.01) - x0 * math.sin(0.01)
        assert endpoint[0].contains(exact_x, tol=2e-9)
        assert endpoint[1].contains(exact_y, tol=2e-9)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("batch", [1, 16, 48])
def test_cuda_registered_horner_matches_cpu_within_safeguard_contract(batch):
    generator = torch.Generator().manual_seed(100 + batch)
    exponents = torch.tensor([(0, 0), (1, 0), (0, 1), (2, 1), (1, 3), (4, 0)], dtype=torch.long)
    coeffs = torch.randn((batch, 2, exponents.shape[0]), generator=generator, dtype=DTYPE) * 0.125
    lo = torch.tensor([[-1.0, -0.5]], dtype=DTYPE).repeat(batch, 1)
    hi = torch.tensor([[1.0, 0.75]], dtype=DTYPE).repeat(batch, 1)
    cpu = evaluate_dense_registered_horner_range(coeffs, exponents, lo, hi)
    cuda = evaluate_dense_registered_horner_range(coeffs.cuda(), exponents.cuda(), lo.cuda(), hi.cuda())
    assert cuda.validated
    torch.testing.assert_close(cuda.lo.cpu(), cpu.lo, rtol=2e-15, atol=2e-15)
    torch.testing.assert_close(cuda.hi.cpu(), cpu.hi, rtol=2e-15, atol=2e-15)
    _assert_contains_samples(cuda, coeffs.cuda(), exponents.cuda(), lo.cuda(), hi.cuda(), count=97)
