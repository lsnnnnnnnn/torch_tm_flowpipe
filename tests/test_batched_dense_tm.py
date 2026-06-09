
import math

import pytest
import torch

from torch_tm_flowpipe import Interval, Polynomial
from torch_tm_flowpipe.batched_dense_tm import (
    BatchedMonomialBasis,
    BatchedPolynomial,
    BatchedTaylorModel,
    DenseTMProfiler,
)


def _expected_exponents(dim, order):
    out = []

    def rec(pos, remaining, prefix):
        if pos == dim:
            if remaining == 0:
                out.append(tuple(prefix))
            return
        for value in range(remaining + 1):
            rec(pos + 1, remaining - value, prefix + [value])

    for degree in range(order + 1):
        rec(0, degree, [])
    return out


def _dense_to_scalar(coeffs, basis):
    terms = {}
    coeffs_cpu = coeffs.detach().cpu()
    for idx, exp in enumerate(_expected_exponents(basis.dim, basis.order)):
        value = coeffs_cpu[idx]
        if bool(torch.abs(value) > 0):
            terms[exp] = value.clone()
    return Polynomial(terms, n_vars=basis.dim)


def _scalar_to_dense(poly, basis, *, dtype=torch.float64, device="cpu"):
    coeffs = torch.zeros((basis.num_terms,), dtype=dtype, device=device)
    for exp, coeff in poly.terms.items():
        if sum(exp) <= basis.order:
            coeffs[basis.term_index(exp)] = coeff.to(dtype=dtype, device=device)
    return coeffs


def _assert_contains(lo, hi, values, tol=1e-10):
    assert bool(torch.all(values >= lo - tol)), (lo, values)
    assert bool(torch.all(values <= hi + tol)), (hi, values)


def test_basis_term_counts_ordering_and_multiplication_plan():
    for order in [0, 1, 2, 3]:
        basis = BatchedMonomialBasis.build(dim=2, order=order)
        assert basis.num_terms == math.comb(2 + order, order)
        assert basis.constant_index == basis.term_index((0, 0))
        assert basis.exponents.tolist() == [list(exp) for exp in _expected_exponents(2, order)]
        assert bool(torch.all(basis.degree <= order))
        if order == 0:
            assert basis.linear_indices == []
        else:
            assert basis.linear_indices == [basis.term_index((1, 0)), basis.term_index((0, 1))]

    basis = BatchedMonomialBasis.build(dim=2, order=2)
    x_idx = basis.term_index((1, 0))
    y_idx = basis.term_index((0, 1))
    xy_idx = basis.term_index((1, 1))
    left, right, out = basis.multiplication_plan()
    mask = (left == x_idx) & (right == y_idx)
    assert bool(torch.any(mask))
    assert int(out[mask][0]) == xy_idx


def test_polynomial_add_scale_affine_and_evaluate_match_scalar():
    basis = BatchedMonomialBasis.build(dim=2, order=3)
    gen = torch.Generator().manual_seed(7)
    coeffs_a = torch.randn((3, 2, basis.num_terms), generator=gen, dtype=torch.float64) * 0.25
    coeffs_b = torch.randn((3, 2, basis.num_terms), generator=gen, dtype=torch.float64) * 0.25
    p = BatchedPolynomial(coeffs_a, basis)
    q = BatchedPolynomial(coeffs_b, basis)

    assert torch.allclose(p.add(q).coeffs, coeffs_a + coeffs_b)
    assert torch.allclose(p.sub(q).coeffs, coeffs_a - coeffs_b)
    assert torch.allclose(p.scale(-1.5).coeffs, coeffs_a * -1.5)

    W = torch.tensor([[1.5, -0.25], [-2.0, 0.5], [0.0, 3.0]], dtype=torch.float64)
    b = torch.tensor([1.0, -1.0, 0.5], dtype=torch.float64)
    mapped = p.affine_map(W, b)
    expected = torch.einsum("no,bot->bnt", W, coeffs_a)
    expected[:, :, basis.constant_index] += b.view(1, -1)
    assert torch.allclose(mapped.coeffs, expected)

    points = torch.tensor([[0.2, -0.3], [0.7, 0.1], [-0.4, 0.5]], dtype=torch.float64)
    dense_values = p.evaluate(points)
    for batch in range(3):
        for out_dim in range(2):
            scalar = _dense_to_scalar(coeffs_a[batch, out_dim], basis)
            expected_value = scalar.evaluate_point(points[batch])
            assert torch.allclose(dense_values[batch, out_dim], expected_value)

    point_cloud = torch.stack([points, points + 0.1], dim=1)
    assert p.evaluate(point_cloud).shape == (3, 2, 2)


def test_mul_trunc_matches_sparse_polynomial_coefficients():
    basis = BatchedMonomialBasis.build(dim=2, order=3)
    gen = torch.Generator().manual_seed(19)
    coeffs_a = torch.randn((2, 1, basis.num_terms), generator=gen, dtype=torch.float64) * 0.2
    coeffs_b = torch.randn((2, 1, basis.num_terms), generator=gen, dtype=torch.float64) * 0.2
    p = BatchedPolynomial(coeffs_a, basis)
    q = BatchedPolynomial(coeffs_b, basis)

    product = p.mul_trunc(q)
    for batch in range(2):
        scalar_p = _dense_to_scalar(coeffs_a[batch, 0], basis)
        scalar_q = _dense_to_scalar(coeffs_b[batch, 0], basis)
        kept, _dropped = scalar_p.mul_truncate(scalar_q, basis.order)
        expected = _scalar_to_dense(kept, basis)
        assert torch.allclose(product.coeffs[batch, 0], expected, atol=1e-12, rtol=1e-12)


def test_taylor_model_multiplication_moves_dropped_terms_to_remainder():
    basis = BatchedMonomialBasis.build(dim=2, order=2)
    domain_lo = torch.tensor([[-0.5, -0.25]], dtype=torch.float64)
    domain_hi = torch.tensor([[0.75, 0.5]], dtype=torch.float64)
    variables = BatchedTaylorModel.variables_from_domain(domain_lo, domain_hi, basis)
    x = variables.component(0)
    y = variables.component(1)

    left = x.mul_trunc(x).add(y)
    right = x.add(y)
    product = left.mul_trunc(right)
    lo, hi = product.range_bound()

    xs = torch.linspace(float(domain_lo[0, 0]), float(domain_hi[0, 0]), 7, dtype=torch.float64)
    ys = torch.linspace(float(domain_lo[0, 1]), float(domain_hi[0, 1]), 7, dtype=torch.float64)
    samples = torch.tensor([[float(a), float(b)] for a in xs for b in ys], dtype=torch.float64).view(1, -1, 2)
    exact = ((samples[..., 0] ** 2 + samples[..., 1]) * (samples[..., 0] + samples[..., 1])).unsqueeze(-1)
    _assert_contains(lo[:, None, :], hi[:, None, :], exact)
    assert bool(torch.any(product.rem_lo < 0)) or bool(torch.any(product.rem_hi > 0))


def test_range_bounds_contain_sampled_values():
    basis = BatchedMonomialBasis.build(dim=2, order=2)
    domain_lo = torch.tensor([[-1.0, -0.5], [-0.25, -1.0], [-1.0, 0.0]], dtype=torch.float64)
    domain_hi = torch.tensor([[2.0, 1.5], [1.0, 0.75], [1.0, 2.0]], dtype=torch.float64)
    coeffs = torch.zeros((3, 1, basis.num_terms), dtype=torch.float64)
    coeffs[0, 0, basis.term_index((0, 0))] = 1.0
    coeffs[0, 0, basis.term_index((1, 0))] = 2.0
    coeffs[0, 0, basis.term_index((0, 1))] = -3.0
    coeffs[1, 0, basis.term_index((1, 0))] = 1.0
    coeffs[1, 0, basis.term_index((0, 1))] = 1.0
    coeffs[2, 0, basis.term_index((2, 0))] = 1.0
    coeffs[2, 0, basis.term_index((0, 1))] = 1.0
    polys = BatchedPolynomial(coeffs, basis)

    lo, hi = polys.range_bound(domain_lo, domain_hi)
    samples = torch.stack(
        [
            domain_lo,
            domain_hi,
            0.5 * (domain_lo + domain_hi),
            torch.stack([domain_lo[:, 0], domain_hi[:, 1]], dim=1),
            torch.stack([domain_hi[:, 0], domain_lo[:, 1]], dim=1),
        ],
        dim=1,
    )
    values = polys.evaluate(samples)
    _assert_contains(lo[:, None, :], hi[:, None, :], values)


def test_batched_taylor_model_shapes_and_sample_containment():
    basis = BatchedMonomialBasis.build(dim=2, order=3)
    domain_lo = torch.tensor([[-0.4, -0.3], [0.1, -0.2]], dtype=torch.float64)
    domain_hi = torch.tensor([[0.5, 0.4], [0.6, 0.3]], dtype=torch.float64)
    tm = BatchedTaylorModel.variables_from_domain(domain_lo, domain_hi, basis)

    mapped = tm.affine_map(torch.tensor([[1.0, 2.0], [-0.5, 0.25]], dtype=torch.float64), torch.tensor([0.1, -0.2]))
    product = tm.component(0).mul_trunc(tm.component(1))
    stepped = tm.one_fixed_tm_step_vdp(0.01, order=3)

    assert mapped.poly.coeffs.shape == (2, 2, basis.num_terms)
    assert product.rem_lo.shape == (2, 1)
    assert isinstance(stepped.poly.coeffs, torch.Tensor)
    assert isinstance(stepped.rem_lo, torch.Tensor)
    assert not isinstance(stepped.poly, Polynomial)

    samples = torch.stack(
        [
            domain_lo,
            domain_hi,
            0.5 * (domain_lo + domain_hi),
        ],
        dim=1,
    )
    x0 = samples[..., 0]
    y0 = samples[..., 1]
    exact_step = torch.stack(
        [x0 + 0.01 * y0, y0 + 0.01 * (y0 - x0 - x0 * x0 * y0)],
        dim=-1,
    )
    lo, hi = stepped.range_bound()
    _assert_contains(lo[:, None, :], hi[:, None, :], exact_step)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")),
    ],
)
def test_device_roundtrip_and_cuda_if_available(device):
    basis = BatchedMonomialBasis.build(dim=2, order=2, device=device)
    domain_lo = torch.tensor([[-0.2, -0.1]], dtype=torch.float64, device=device)
    domain_hi = torch.tensor([[0.3, 0.4]], dtype=torch.float64, device=device)
    tm = BatchedTaylorModel.variables_from_domain(domain_lo, domain_hi, basis)
    out = tm.one_fixed_tm_step_vdp(0.01, order=2)
    lo, hi = out.range_bound()

    assert out.poly.coeffs.device.type == torch.device(device).type
    assert lo.device.type == torch.device(device).type
    assert bool(torch.all(lo <= hi))



def test_dropped_term_merged_bound_contains_samples_and_is_not_wider():
    basis = BatchedMonomialBasis.build(dim=2, order=1)
    domain_lo = torch.tensor([[-1.0, -0.5]], dtype=torch.float64)
    domain_hi = torch.tensor([[1.0, 0.75]], dtype=torch.float64)
    coeffs_p = torch.zeros((1, 1, basis.num_terms), dtype=torch.float64)
    coeffs_q = torch.zeros_like(coeffs_p)
    coeffs_p[0, 0, basis.term_index((1, 0))] = 1.0
    coeffs_p[0, 0, basis.term_index((0, 1))] = 1.0
    coeffs_q[0, 0, basis.term_index((1, 0))] = 1.0
    coeffs_q[0, 0, basis.term_index((0, 1))] = -1.0
    p = BatchedPolynomial(coeffs_p, basis)
    q = BatchedPolynomial(coeffs_q, basis)

    kept_termwise, term_lo, term_hi = p.mul_trunc(
        q,
        return_truncation_bound=True,
        domain_lo=domain_lo,
        domain_hi=domain_hi,
        dropped_merge_mode="termwise",
    )
    kept_merged, merged_lo, merged_hi = p.mul_trunc(
        q,
        return_truncation_bound=True,
        domain_lo=domain_lo,
        domain_hi=domain_hi,
        dropped_merge_mode="merged",
    )

    assert kept_termwise.coeffs.shape == kept_merged.coeffs.shape
    assert term_lo.shape == merged_lo.shape == (1, 1)
    assert bool(torch.all(merged_hi - merged_lo <= term_hi - term_lo + 1e-12))

    xs = torch.linspace(float(domain_lo[0, 0]), float(domain_hi[0, 0]), 9, dtype=torch.float64)
    ys = torch.linspace(float(domain_lo[0, 1]), float(domain_hi[0, 1]), 9, dtype=torch.float64)
    samples = torch.tensor([[float(x), float(y)] for x in xs for y in ys], dtype=torch.float64).view(1, -1, 2)
    dropped_values = p.evaluate(samples) * q.evaluate(samples)
    _assert_contains(term_lo[:, None, :], term_hi[:, None, :], dropped_values)
    _assert_contains(merged_lo[:, None, :], merged_hi[:, None, :], dropped_values)


def test_split_range_bound_contains_samples_and_is_comparable_to_interval():
    basis = BatchedMonomialBasis.build(dim=2, order=2)
    domain_lo = torch.tensor([[0.0, -0.25]], dtype=torch.float64)
    domain_hi = torch.tensor([[1.0, 0.25]], dtype=torch.float64)
    coeffs = torch.zeros((1, 1, basis.num_terms), dtype=torch.float64)
    coeffs[0, 0, basis.term_index((1, 0))] = 1.0
    coeffs[0, 0, basis.term_index((2, 0))] = -1.0
    coeffs[0, 0, basis.term_index((0, 1))] = 0.25
    poly = BatchedPolynomial(coeffs, basis)

    interval_lo, interval_hi = poly.range_bound(domain_lo, domain_hi, method="interval")
    split_lo, split_hi = poly.range_bound(domain_lo, domain_hi, method="split2")
    assert bool(torch.all(split_hi - split_lo <= interval_hi - interval_lo + 1e-12))

    xs = torch.linspace(0.0, 1.0, 17, dtype=torch.float64)
    ys = torch.linspace(-0.25, 0.25, 9, dtype=torch.float64)
    samples = torch.tensor([[float(x), float(y)] for x in xs for y in ys], dtype=torch.float64).view(1, -1, 2)
    values = poly.evaluate(samples)
    _assert_contains(interval_lo[:, None, :], interval_hi[:, None, :], values)
    _assert_contains(split_lo[:, None, :], split_hi[:, None, :], values)


def test_fixed_euler_alias_and_profiler_hooks():
    basis = BatchedMonomialBasis.build(dim=2, order=2)
    domain_lo = torch.tensor([[-0.2, -0.1]], dtype=torch.float64)
    domain_hi = torch.tensor([[0.3, 0.4]], dtype=torch.float64)
    tm = BatchedTaylorModel.variables_from_domain(domain_lo, domain_hi, basis)
    profiler = DenseTMProfiler(device="cpu")
    euler = tm.fixed_euler_tm_step_vdp(0.01, dropped_merge_mode="merged", profile=profiler)
    old_alias = tm.fixed_picard_step_vdp(0.01, dropped_merge_mode="merged")
    one_alias = tm.one_fixed_tm_step_vdp(0.01, dropped_merge_mode="merged")

    assert torch.allclose(euler.poly.coeffs, old_alias.poly.coeffs)
    assert torch.allclose(euler.poly.coeffs, one_alias.poly.coeffs)
    assert profiler.timings_ms["mul_trunc"] > 0.0
    assert profiler.timings_ms["dropped_range_bound"] > 0.0
    assert profiler.timings_ms["range_bound"] > 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cpu_cuda_dense_tm_results_close_for_small_case():
    basis_cpu = BatchedMonomialBasis.build(dim=2, order=2, device="cpu")
    domain_lo = torch.tensor([[-0.2, -0.1], [0.1, -0.2]], dtype=torch.float64)
    domain_hi = torch.tensor([[0.3, 0.4], [0.4, 0.2]], dtype=torch.float64)
    tm_cpu = BatchedTaylorModel.variables_from_domain(domain_lo, domain_hi, basis_cpu)
    for _ in range(3):
        tm_cpu = tm_cpu.fixed_euler_tm_step_vdp(0.01, dropped_merge_mode="merged")
    lo_cpu, hi_cpu = tm_cpu.range_bound(method="split2")

    basis_cuda = BatchedMonomialBasis.build(dim=2, order=2, device="cuda")
    tm_cuda = BatchedTaylorModel.variables_from_domain(domain_lo.cuda(), domain_hi.cuda(), basis_cuda)
    for _ in range(3):
        tm_cuda = tm_cuda.fixed_euler_tm_step_vdp(0.01, dropped_merge_mode="merged")
    lo_cuda, hi_cuda = tm_cuda.range_bound(method="split2")

    assert torch.allclose(lo_cpu, lo_cuda.cpu(), atol=1e-10, rtol=1e-10)
    assert torch.allclose(hi_cpu, hi_cuda.cpu(), atol=1e-10, rtol=1e-10)
