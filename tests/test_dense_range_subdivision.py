import itertools
import math

import pytest
import torch

from torch_tm_flowpipe import (
    BatchedMonomialBasis,
    BatchedPolynomial,
    DenseSubdivisionCover,
    Interval,
    Polynomial,
    build_dense_subdivision_cover,
    validate_dense_subdivision_cover,
)


def _poly(terms, *, dim, order, batch=1, device="cpu"):
    basis = BatchedMonomialBasis.build(dim, order, device=device)
    coeffs = torch.zeros((batch, 1, basis.num_terms), dtype=torch.float64, device=device)
    for exponent, value in terms.items():
        coeffs[:, 0, basis.term_index(exponent)] = torch.as_tensor(value, dtype=torch.float64, device=device)
    return BatchedPolynomial(coeffs, basis)


def _samples(lo, hi, count=401, *, device="cpu"):
    lo_t = torch.as_tensor(lo, dtype=torch.float64, device=device)
    hi_t = torch.as_tensor(hi, dtype=torch.float64, device=device)
    if lo_t.numel() == 1:
        return torch.linspace(float(lo_t[0]), float(hi_t[0]), count, dtype=torch.float64, device=device).view(1, -1, 1)
    generator = torch.Generator(device=device).manual_seed(20260805)
    unit = torch.rand((1, count, lo_t.numel()), generator=generator, dtype=torch.float64, device=device)
    return lo_t.view(1, 1, -1) + unit * (hi_t - lo_t).view(1, 1, -1)


@pytest.mark.parametrize(
    "terms,domain",
    [
        ({(0,): 3.0}, ([-1.0], [1.0])),
        ({(1,): 2.0}, ([-1.0], [1.0])),
        ({(0,): 0.25, (1,): -1.0, (2,): 1.0}, ([-1.0], [2.0])),
        ({(3,): 1.0, (4,): -0.25}, ([-2.0], [1.0])),
        ({(2,): 1.0}, ([0.125], [0.625])),
        ({(3,): -2.0}, ([-2.0], [-0.25])),
        ({(8,): 1e-10, (12,): -1e-12}, ([0.0], [0.0039859994324420315])),
    ],
)
def test_analytic_scalar_cases_contain_dense_samples(terms, domain):
    order = max(sum(exponent) for exponent in terms)
    poly = _poly(terms, dim=1, order=order)
    lo = torch.tensor([domain[0]], dtype=torch.float64)
    hi = torch.tensor([domain[1]], dtype=torch.float64)
    result = poly.range_bound(lo, hi, method="subdivision", subdivision_depth=3, split_vars=(0,), return_result=True)
    values = poly.evaluate(_samples(domain[0], domain[1]))
    assert bool(torch.all(values >= result.selected_lo[:, None, :]))
    assert bool(torch.all(values <= result.selected_hi[:, None, :]))
    assert result.coverage_report["valid"]


def test_subdivision_tightens_shifted_square_without_sample_hull():
    # (u - 0.375)^2 over a shifted off-origin domain.
    poly = _poly({(0,): 0.140625, (1,): -0.75, (2,): 1.0}, dim=1, order=2)
    lo = torch.tensor([[0.125]], dtype=torch.float64)
    hi = torch.tensor([[0.625]], dtype=torch.float64)
    result = poly.range_bound(lo, hi, method="subdivision", subdivision_depth=3, split_vars=(0,), return_result=True)
    assert float(result.subdivision_hi - result.subdivision_lo) < float(result.natural_hi - result.natural_lo)
    assert float(result.selected_lo) <= 0.0 <= float(result.selected_hi)
    assert result.selected_method == "subdivision"


def test_mixed_monomials_and_cross_zero_boxes_are_contained():
    poly = _poly({(1, 1): 1.0, (2, 1): -1.0, (0, 0): 0.125}, dim=2, order=3)
    lo = torch.tensor([[-1.0, -0.5]], dtype=torch.float64)
    hi = torch.tensor([[1.0, 1.5]], dtype=torch.float64)
    result = poly.range_bound(lo, hi, method="subdivision", subdivision_depth=3, split_vars=(0, 1), return_result=True)
    values = poly.evaluate(_samples(lo[0], hi[0], count=2000))
    assert bool(torch.all(values >= result.selected_lo[:, None, :]))
    assert bool(torch.all(values <= result.selected_hi[:, None, :]))


@pytest.mark.parametrize("depth,expected", [(0, 1), (1, 4), (2, 8), (3, 16), (4, 32), (5, 64)])
def test_pre_registered_depth_leaf_counts_and_complete_cover(depth, expected):
    poly = _poly({(1, 0, 0): 1.0, (0, 1, 0): -2.0, (0, 0, 4): 0.5}, dim=3, order=4)
    domain_lo = torch.tensor([[-1.0, -1.0, 0.0]], dtype=torch.float64)
    domain_hi = torch.tensor([[1.0, 1.0, 0.01]], dtype=torch.float64)
    cover = build_dense_subdivision_cover(poly.coeffs, poly.basis.exponents, domain_lo, domain_hi, depth=depth, split_vars=(0, 1))
    report = validate_dense_subdivision_cover(cover, domain_lo, domain_hi)
    assert cover.leaf_counts == (expected,)
    assert report["valid"], report
    assert all(2 not in history for history in cover.split_variables)


def test_zero_width_domain_is_not_duplicated_and_shifted_cover_has_shared_boundaries():
    zero = _poly({(1,): 1.0}, dim=1, order=1)
    zlo = torch.tensor([[0.125]], dtype=torch.float64)
    zhi = zlo.clone()
    cover = build_dense_subdivision_cover(zero.coeffs, zero.basis.exponents, zlo, zhi, depth=5, split_vars=(0,))
    assert cover.leaf_counts == (1,)
    assert validate_dense_subdivision_cover(cover, zlo, zhi)["valid"]

    shifted = _poly({(1,): 1.0}, dim=1, order=1)
    lo = torch.tensor([[-2.75]], dtype=torch.float64)
    hi = torch.tensor([[-1.125]], dtype=torch.float64)
    cover = build_dense_subdivision_cover(shifted.coeffs, shifted.basis.exponents, lo, hi, depth=3, split_vars=(0,))
    ordered = sorted(zip(cover.lo[:, 0].tolist(), cover.hi[:, 0].tolist()))
    assert ordered[0][0] == float(lo)
    assert ordered[-1][1] == float(hi)
    assert all(left[1] == right[0] for left, right in zip(ordered[:-1], ordered[1:]))


@pytest.mark.parametrize("batch", [1, 16, 48])
def test_batch_ownership_and_shape_are_preserved(batch):
    basis = BatchedMonomialBasis.build(2, 4)
    generator = torch.Generator().manual_seed(11)
    coeffs = torch.randn((batch, 2, basis.num_terms), generator=generator, dtype=torch.float64) * 0.05
    poly = BatchedPolynomial(coeffs, basis)
    domain_lo = torch.stack([torch.tensor([-1.0 + i * 1e-3, -0.5]) for i in range(batch)]).to(torch.float64)
    domain_hi = domain_lo + torch.tensor([1.5, 1.25], dtype=torch.float64)
    result = poly.range_bound(domain_lo, domain_hi, method="subdivision", subdivision_depth=2, split_vars=(0, 1), return_result=True)
    assert result.selected_lo.shape == (batch, 2)
    assert result.cover.leaf_counts == tuple(8 for _ in range(batch))
    assert torch.bincount(result.cover.owner, minlength=batch).tolist() == [8] * batch
    assert result.coverage_report["valid"]


def test_dense_sparse_leafwise_parity_after_equal_exponent_merge():
    terms = {(0, 0): 0.125, (1, 0): 0.75, (0, 1): -1.25, (2, 1): 0.5, (4, 0): -0.125}
    dense = _poly(terms, dim=2, order=4)
    domain_lo = torch.tensor([[-0.75, -0.25]], dtype=torch.float64)
    domain_hi = torch.tensor([[1.25, 0.5]], dtype=torch.float64)
    result = dense.range_bound(domain_lo, domain_hi, method="subdivision", subdivision_depth=3, split_vars=(0, 1), return_result=True)
    sparse = Polynomial(terms, n_vars=2)
    leaf_ranges = []
    for lo, hi in zip(result.cover.lo, result.cover.hi):
        leaf_ranges.append(sparse.evaluate_interval([Interval(lo[0], hi[0]), Interval(lo[1], hi[1])]))
    sparse_lo = min(float(interval.lo) for interval in leaf_ranges)
    sparse_hi = max(float(interval.hi) for interval in leaf_ranges)
    tolerance = 2e-13
    assert float(result.subdivision_lo) <= sparse_lo + tolerance
    assert float(result.subdivision_hi) >= sparse_hi - tolerance
    assert abs(float(result.subdivision_lo) - sparse_lo) <= tolerance
    assert abs(float(result.subdivision_hi) - sparse_hi) <= tolerance


@pytest.mark.parametrize("seed", [3, 17, 91])
@pytest.mark.parametrize("dim,degree", [(1, 1), (1, 8), (2, 3), (3, 4), (3, 6)])
def test_randomized_sample_containment_is_deterministic_and_input_unchanged(seed, dim, degree):
    basis = BatchedMonomialBasis.build(dim, degree)
    generator = torch.Generator().manual_seed(seed)
    coeffs = torch.randn((1, 2, basis.num_terms), generator=generator, dtype=torch.float64) * 0.1
    before = coeffs.clone()
    poly = BatchedPolynomial(coeffs, basis)
    lo = -torch.rand((1, dim), generator=generator, dtype=torch.float64)
    hi = lo + 0.05 + torch.rand((1, dim), generator=generator, dtype=torch.float64)
    first = poly.range_bound(lo, hi, method="subdivision", subdivision_depth=3, split_vars=tuple(range(min(2, dim))), return_result=True)
    second = poly.range_bound(lo, hi, method="subdivision", subdivision_depth=3, split_vars=tuple(range(min(2, dim))), return_result=True)
    values = poly.evaluate(_samples(lo[0], hi[0], count=1000))
    assert bool(torch.all(values >= first.selected_lo[:, None, :]))
    assert bool(torch.all(values <= first.selected_hi[:, None, :]))
    assert torch.equal(first.selected_lo, second.selected_lo)
    assert torch.equal(first.selected_hi, second.selected_hi)
    assert torch.equal(coeffs, before)


def test_adversarial_magnitudes_alternating_signs_and_subnormals_remain_finite():
    terms = {(0,): 1e200, (1,): -1e200, (2,): 1e-300, (3,): -5e199, (4,): torch.finfo(torch.float64).tiny}
    poly = _poly(terms, dim=1, order=4)
    lo = torch.tensor([[0.999]], dtype=torch.float64)
    hi = torch.tensor([[1.001]], dtype=torch.float64)
    result = poly.range_bound(lo, hi, method="subdivision", subdivision_depth=4, split_vars=(0,), return_result=True)
    assert bool(torch.all(torch.isfinite(result.selected_lo)))
    assert bool(torch.all(torch.isfinite(result.selected_hi)))


def test_invalid_domain_max_leaves_corrupted_ownership_and_nonfinite_leaf_fail_closed():
    poly = _poly({(2, 0): 1.0, (0, 2): -1.0}, dim=2, order=2)
    lo = torch.tensor([[-1.0, -1.0]], dtype=torch.float64)
    hi = torch.tensor([[1.0, 1.0]], dtype=torch.float64)
    with pytest.raises(ValueError, match="lower bounds"):
        build_dense_subdivision_cover(poly.coeffs, poly.basis.exponents, hi, lo, depth=1)
    with pytest.raises(ValueError, match="max_leaves exceeded"):
        build_dense_subdivision_cover(poly.coeffs, poly.basis.exponents, lo, hi, depth=3, max_leaves=8)
    cover = build_dense_subdivision_cover(poly.coeffs, poly.basis.exponents, lo, hi, depth=2)
    corrupted = DenseSubdivisionCover(cover.lo, cover.hi, torch.ones_like(cover.owner), cover.requested_depth, cover.leaf_counts, cover.split_variables)
    assert not validate_dense_subdivision_cover(corrupted, lo, hi)["valid"]

    huge = _poly({(2,): torch.finfo(torch.float64).max}, dim=1, order=2)
    with pytest.raises(FloatingPointError, match="non-finite subdivision leaf range"):
        huge.range_bound(torch.tensor([[1.5]]), torch.tensor([[2.0]]), method="subdivision", subdivision_depth=1, split_vars=(0,))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("batch", [1, 16, 48])
def test_cuda_subdivision_contains_samples_and_matches_cpu_with_safeguard(batch):
    basis_cpu = BatchedMonomialBasis.build(3, 4)
    generator = torch.Generator().manual_seed(101)
    coeffs = torch.randn((batch, 2, basis_cpu.num_terms), generator=generator, dtype=torch.float64) * 0.01
    lo = torch.tensor([[-1.0, -1.0, 0.0]], dtype=torch.float64).repeat(batch, 1)
    hi = torch.tensor([[1.0, 1.0, 0.01]], dtype=torch.float64).repeat(batch, 1)
    cpu = BatchedPolynomial(coeffs, basis_cpu).range_bound(lo, hi, method="subdivision", subdivision_depth=3, return_result=True)
    cuda_poly = BatchedPolynomial(coeffs.cuda(), BatchedMonomialBasis.build(3, 4, "cuda"))
    cuda = cuda_poly.range_bound(lo.cuda(), hi.cuda(), method="subdivision", subdivision_depth=3, return_result=True)
    tolerance = 2e-12
    assert torch.allclose(cpu.selected_lo, cuda.selected_lo.cpu(), atol=tolerance, rtol=tolerance)
    assert torch.allclose(cpu.selected_hi, cuda.selected_hi.cpu(), atol=tolerance, rtol=tolerance)
    points = _samples(lo[0].cuda(), hi[0].cuda(), count=300, device="cuda").repeat(batch, 1, 1)
    values = cuda_poly.evaluate(points)
    assert bool(torch.all(values >= cuda.selected_lo[:, None, :]))
    assert bool(torch.all(values <= cuda.selected_hi[:, None, :]))
