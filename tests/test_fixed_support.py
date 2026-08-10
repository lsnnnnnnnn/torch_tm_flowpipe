from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from torch_tm_flowpipe.fixed_support import (
    FixedSupportDescriptor,
    FixedSupportInterval,
    FixedSupportPolynomial,
    FixedSupportReachability,
    FixedSupportTaylorModel,
    diffreach_vdp_polynomial_rhs,
    diffreach_vdp_tm_rhs,
    fixed_support_dr_remainder_picard,
    fixed_support_polynomial_picard,
)


EXPECTED_DR7_SHA256 = "0ae11ee9d911d45e42294df74ef2896ecb9aeb9f3d7851c09ea90e2bb2631f5e"


def _support() -> FixedSupportDescriptor:
    return FixedSupportDescriptor.diffreach_restricted_quadratic(2)


def _boxes(batch: int = 2, h: float = 0.01):
    lo = torch.tensor([[0.0, -1.0, -1.0]], dtype=torch.float64).expand(batch, -1).clone()
    hi = torch.tensor([[h, 1.0, 1.0]], dtype=torch.float64).expand(batch, -1).clone()
    return lo, hi


@pytest.mark.unit
def test_diffreach_descriptor_slot_order_routes_and_hash_are_frozen():
    support = _support()
    assert support.exponents == (
        (0, 0, 0),
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (2, 0, 0),
        (1, 1, 0),
        (1, 0, 1),
    )
    assert support.linear_slots == (1, 2, 3)
    assert support.time_cross_slots == (4, 5, 6)
    assert support.num_slots == 7
    assert support.support_sha256 == EXPECTED_DR7_SHA256
    assert json.dumps(support.manifest(), sort_keys=True)
    frozen_manifest = json.loads(
        (Path(__file__).parents[1] / "benchmarks" / "diffreach_fixed_support_dr7_20260810.json").read_text(
            encoding="utf-8"
        )
    )
    frozen_hash = frozen_manifest.pop("support_sha256")
    assert frozen_manifest == support.manifest()
    assert frozen_hash == support.support_sha256

    t2_routes = [route for route in support.multiply_routes if route.output_slot == 4]
    assert [(route.left_slot, route.right_slot, route.sign) for route in t2_routes] == [
        (0, 4, 1),
        (4, 0, 1),
        (1, 1, 1),
        (1, 1, 1),
        (1, 1, -1),
    ]


@pytest.mark.unit
def test_diffreach_multiply_routes_match_c_l_lt_expression_order():
    support = _support()
    generator = torch.Generator().manual_seed(20260810)
    left = torch.randn((2, 3, 7), generator=generator, dtype=torch.float64)
    right = torch.randn((2, 3, 7), generator=generator, dtype=torch.float64)
    actual = FixedSupportPolynomial(left, support).mul_trunc(FixedSupportPolynomial(right, support)).coeffs

    c1, l1, lt1 = left[..., 0], left[..., 1:4], left[..., 4:7]
    c2, l2, lt2 = right[..., 0], right[..., 1:4], right[..., 4:7]
    expected_c = c1 * c2
    expected_l = c1[..., None] * l2 + c2[..., None] * l1
    expected_lt = (
        c1[..., None] * lt2
        + c2[..., None] * lt1
        + l1[..., 0:1] * l2
        + l2[..., 0:1] * l1
    )
    expected_lt[..., 0] = expected_lt[..., 0] - l1[..., 0] * l2[..., 0]
    expected = torch.cat((expected_c[..., None], expected_l, expected_lt), dim=-1)
    assert torch.equal(actual, expected)


@pytest.mark.unit
def test_diffreach_integration_routes_match_c_l_lt_mapping():
    support = _support()
    coeffs = torch.arange(2 * 3 * 7, dtype=torch.float64).reshape(2, 3, 7) / 7.0
    actual = FixedSupportPolynomial(coeffs, support).integrate_time_trunc().coeffs
    expected = torch.zeros_like(coeffs)
    expected[..., 1] = coeffs[..., 0]
    expected[..., 4] = 0.5 * coeffs[..., 1]
    expected[..., 5:7] = coeffs[..., 2:4]
    assert torch.equal(actual, expected)


@pytest.mark.unit
def test_range_separates_endpoint_and_full_step_tube():
    support = _support()
    coefficients = torch.zeros((1, 1, 7), dtype=torch.float64)
    coefficients[..., 0] = 1.0
    coefficients[..., 1] = 2.0
    coefficients[..., 2] = -0.5
    coefficients[..., 4] = 3.0
    coefficients[..., 5] = 0.25
    polynomial = FixedSupportPolynomial(coefficients, support)
    tube_lo, tube_hi = _boxes(1, h=0.1)
    endpoint_lo = tube_lo.clone()
    endpoint_lo[:, 0] = 0.1
    tube = polynomial.range(tube_lo, tube_hi)
    endpoint = polynomial.range(endpoint_lo, tube_hi)
    assert tube.lo.item() < endpoint.lo.item()
    assert tube.hi.item() == pytest.approx(endpoint.hi.item(), abs=1e-15)


@pytest.mark.unit
def test_grouped_multiplication_and_integration_ledgers_sum_to_remainder():
    support = _support()
    lo, hi = _boxes()
    generator = torch.Generator().manual_seed(17)
    left = FixedSupportPolynomial(torch.randn((2, 2, 7), generator=generator, dtype=torch.float64), support)
    right = FixedSupportPolynomial(torch.randn((2, 2, 7), generator=generator, dtype=torch.float64), support)
    _, multiply_ledger = left.mul_ctrunc(right, lo, hi)
    assert tuple(multiply_ledger.as_dict()) == (
        "pure_spatial_quadratic",
        "time_cubic",
        "time_quartic",
    )
    multiply_total = multiply_ledger.total_like(left.coeffs[..., 0])
    assert torch.all(multiply_total.lo <= multiply_total.hi)

    _, integration_ledger = left.integrate_time_ctrunc(lo, hi)
    assert tuple(integration_ledger.as_dict()) == (
        "integration_time_cubic",
        "integration_time_squared_spatial",
    )
    integration_total = integration_ledger.total_like(left.coeffs[..., 0])
    assert torch.all(integration_total.lo <= integration_total.hi)


@pytest.mark.unit
def test_two_polynomial_picard_iterates_have_fixed_support():
    support = _support()
    lo, hi = _boxes()
    base_coefficients = torch.zeros((2, 2, 7), dtype=torch.float64)
    base_coefficients[:, :, 0] = torch.tensor([[1.25, 2.4], [-0.3, 0.7]], dtype=torch.float64)
    base_coefficients[:, 0, 2] = torch.tensor([0.15, 0.2], dtype=torch.float64)
    base_coefficients[:, 1, 3] = torch.tensor([0.05, 0.1], dtype=torch.float64)
    base = FixedSupportPolynomial(base_coefficients, support)
    final, trace = fixed_support_polynomial_picard(
        base, diffreach_vdp_polynomial_rhs, lo, hi, iterations=2
    )
    assert len(trace) == 2
    assert trace[0].coeffs.shape == (2, 2, 7)
    assert final.support.support_sha256 == EXPECTED_DR7_SHA256
    assert torch.isfinite(final.coeffs).all()
    assert not torch.equal(trace[0].coeffs, trace[1].coeffs)


@pytest.mark.unit
def test_dr_picard_reports_failed_initial_inclusion_for_rejection():
    support = _support()
    lo, hi = _boxes(batch=1, h=0.1)
    base = FixedSupportPolynomial.zeros(1, 1, support)
    new_x0 = FixedSupportTaylorModel.from_polynomial(base)
    seed = FixedSupportTaylorModel(
        base,
        FixedSupportInterval(
            torch.tensor([[-1e-4]], dtype=torch.float64),
            torch.tensor([[1e-4]], dtype=torch.float64),
        ),
    )

    def expanding_rhs(state, box_lo, box_hi):
        del box_lo, box_hi
        return FixedSupportTaylorModel(
            state.polynomial,
            FixedSupportInterval(
                torch.full_like(state.remainder.lo, -1.0),
                torch.full_like(state.remainder.hi, 1.0),
            ),
        )

    result = fixed_support_dr_remainder_picard(
        expanding_rhs, new_x0, seed, lo, hi, rounds=3
    )
    assert not result.initial_inclusion_passed
    assert not result.initial_inclusion_mask.item()
    assert result.round_inclusion_masks.shape == (3, 1, 1)
    assert torch.equal(result.round_remainder_lo[0], seed.remainder.lo)
    assert torch.equal(result.round_remainder_hi[0], seed.remainder.hi)


@pytest.mark.unit
def test_vdp_tm_rhs_emits_named_overflow_and_finite_ranges():
    support = _support()
    lo, hi = _boxes(batch=1)
    coefficients = torch.zeros((1, 2, 7), dtype=torch.float64)
    coefficients[..., 0] = torch.tensor([[1.25, 2.4]], dtype=torch.float64)
    coefficients[:, 0, 2] = 0.15
    coefficients[:, 1, 3] = 0.05
    model = FixedSupportTaylorModel.from_polynomial(FixedSupportPolynomial(coefficients, support))
    rhs = diffreach_vdp_tm_rhs(model, lo, hi)
    assert rhs.polynomial.coeffs.shape == (1, 2, 7)
    assert rhs.ledger.entries
    interval = rhs.range(lo, hi)
    assert torch.isfinite(interval.lo).all()
    assert torch.isfinite(interval.hi).all()
    assert torch.all(interval.lo <= interval.hi)


@pytest.mark.regression
def test_frozen_diffreach_polynomial_picard_and_every_dr_rp_round_are_exact():
    fixture_path = Path(__file__).parent / "fixtures" / "diffreach_dr7_vdp_one_step_float64.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    support = _support()
    lo, hi = _boxes(batch=2, h=fixture["h"])
    centers = torch.tensor([[1.25, 2.4], [-0.3, 0.7]], dtype=torch.float64)
    scales = torch.tensor([[0.15, 0.05], [0.2, 0.1]], dtype=torch.float64)
    base_coefficients = torch.zeros((2, 2, 7), dtype=torch.float64)
    base_coefficients[..., 0] = centers
    base_coefficients[:, 0, 2] = scales[:, 0]
    base_coefficients[:, 1, 3] = scales[:, 1]
    base = FixedSupportPolynomial(base_coefficients, support)

    poly2, polynomial_rounds = fixed_support_polynomial_picard(
        base, diffreach_vdp_polynomial_rhs, lo, hi, iterations=2
    )
    assert torch.equal(
        polynomial_rounds[0].coeffs,
        torch.tensor(fixture["poly1_slots"], dtype=torch.float64),
    )
    assert torch.equal(
        polynomial_rounds[1].coeffs,
        torch.tensor(fixture["poly2_slots"], dtype=torch.float64),
    )

    seed = FixedSupportTaylorModel(
        poly2,
        FixedSupportInterval(
            torch.full((2, 2), -0.01, dtype=torch.float64),
            torch.full((2, 2), 0.01, dtype=torch.float64),
        ),
    )
    result = fixed_support_dr_remainder_picard(
        diffreach_vdp_tm_rhs,
        FixedSupportTaylorModel.from_polynomial(base),
        seed,
        lo,
        hi,
        rounds=10,
    )
    assert torch.equal(result.initial_inclusion_mask, torch.tensor(fixture["initial_inclusion_mask"]))
    assert torch.equal(result.round_inclusion_masks, torch.tensor(fixture["round_masks"]))
    assert torch.equal(
        result.round_remainder_lo,
        torch.tensor(fixture["round_accepted_lo"], dtype=torch.float64),
    )
    assert torch.equal(
        result.round_remainder_hi,
        torch.tensor(fixture["round_accepted_hi"], dtype=torch.float64),
    )

    tube = result.model.range(lo, hi)
    endpoint_lo = lo.clone()
    endpoint_lo[:, 0] = fixture["h"]
    endpoint = result.model.range(endpoint_lo, hi)
    assert torch.equal(tube.lo, torch.tensor(fixture["tube_lo"], dtype=torch.float64))
    assert torch.equal(tube.hi, torch.tensor(fixture["tube_hi"], dtype=torch.float64))
    assert torch.equal(endpoint.lo, torch.tensor(fixture["endpoint_lo"], dtype=torch.float64))
    assert torch.equal(endpoint.hi, torch.tensor(fixture["endpoint_hi"], dtype=torch.float64))


def _vdp_solver(support=None, **kwargs):
    support = support or _support()
    return FixedSupportReachability(
        support=support,
        state_dim=2,
        polynomial_rhs=diffreach_vdp_polynomial_rhs,
        tm_rhs=diffreach_vdp_tm_rhs,
        step_size=0.01,
        initial_remainder=0.01,
        polynomial_picard_iterations=2,
        remainder_rounds=10,
        symbolic_window_size=1000,
        **kwargs,
    )


@pytest.mark.integration
def test_one_complete_segment_returns_distinct_endpoint_and_tube():
    solver = _vdp_solver()
    initial_lo = torch.tensor([[1.1, 2.35]], dtype=torch.float64)
    initial_hi = torch.tensor([[1.4, 2.45]], dtype=torch.float64)
    result = solver.verify(initial_lo, initial_hi, steps=1)
    assert result.completed
    assert result.validated_steps == 1
    assert result.endpoint_lo.shape == (1, 2, 2)
    assert result.tube_lo.shape == (1, 1, 2)
    assert result.host_synchronizations == 1
    assert result.device_transfers == 0
    assert result.endpoint_lo[0, 1].tolist() == pytest.approx(
        [1.1233050850213477, 2.3116066270348252], abs=2e-15
    )
    assert result.endpoint_hi[0, 1].tolist() == pytest.approx(
        [1.4244270012524727, 2.4343238753471987], abs=2e-15
    )
    assert result.tube_lo[0, 0, 0] < result.endpoint_lo[0, 1, 0]
    assert result.tube_hi[0, 0, 1] > result.endpoint_hi[0, 1, 1]


@pytest.mark.integration
def test_float64_short_horizon_matches_pinned_diffreach_operations():
    """Reference was exported with pinned builders forced explicitly to f64."""

    solver = _vdp_solver()
    result = solver.verify(
        torch.tensor([[1.1, 2.35]], dtype=torch.float64),
        torch.tensor([[1.4, 2.45]], dtype=torch.float64),
        steps=10,
    )
    assert result.completed
    assert torch.all(result.initial_inclusion_masks)
    assert result.endpoint_lo[0, -1].tolist() == pytest.approx(
        [1.3214892996883922, 1.9114552016480633], abs=2e-9
    )
    assert result.endpoint_hi[0, -1].tolist() == pytest.approx(
        [1.6274613072475235, 2.2240727893613332], abs=2e-9
    )


@pytest.mark.integration
def test_failed_initial_inclusion_is_propagated_to_solver_completion():
    support = FixedSupportDescriptor.diffreach_restricted_quadratic(1)

    def zero_polynomial_rhs(state, box_lo, box_hi):
        del box_lo, box_hi
        return FixedSupportPolynomial.zeros(
            state.batch,
            state.output_dim,
            state.support,
            dtype=state.coeffs.dtype,
            device=state.coeffs.device,
        )

    def expanding_tm_rhs(state, box_lo, box_hi):
        del box_lo, box_hi
        zero = FixedSupportPolynomial.zeros(
            state.polynomial.batch,
            state.polynomial.output_dim,
            state.polynomial.support,
            dtype=state.polynomial.coeffs.dtype,
            device=state.polynomial.coeffs.device,
        )
        return FixedSupportTaylorModel(
            zero,
            FixedSupportInterval(
                torch.full_like(state.remainder.lo, -1.0),
                torch.full_like(state.remainder.hi, 1.0),
            ),
        )

    solver = FixedSupportReachability(
        support=support,
        state_dim=1,
        polynomial_rhs=zero_polynomial_rhs,
        tm_rhs=expanding_tm_rhs,
        step_size=0.1,
        initial_remainder=1e-4,
        remainder_rounds=3,
        symbolic_window_size=4,
    )
    result = solver.verify(
        torch.tensor([[0.0]], dtype=torch.float64),
        torch.tensor([[0.1]], dtype=torch.float64),
        steps=2,
    )
    assert not result.completed
    assert result.validated_steps == 0
    assert result.first_failure_step == 0
    assert result.first_failure_reason == "failed_initial_DR_RP_inclusion"
    assert not result.initial_inclusion_masks[0].item()


@pytest.mark.property
def test_batch_permutation_equivariance_on_short_horizon():
    initial_lo = torch.tensor(
        [[1.1, 2.35], [1.15, 2.37], [1.25, 2.4]], dtype=torch.float64
    )
    initial_hi = torch.tensor(
        [[1.4, 2.45], [1.3, 2.43], [1.35, 2.44]], dtype=torch.float64
    )
    permutation = torch.tensor([2, 0, 1])
    direct = _vdp_solver().verify(initial_lo, initial_hi, steps=3)
    permuted = _vdp_solver().verify(
        initial_lo.index_select(0, permutation),
        initial_hi.index_select(0, permutation),
        steps=3,
    )
    inverse = torch.argsort(permutation)
    assert torch.equal(direct.endpoint_lo, permuted.endpoint_lo.index_select(0, inverse))
    assert torch.equal(direct.endpoint_hi, permuted.endpoint_hi.index_select(0, inverse))
    assert torch.equal(direct.tube_lo, permuted.tube_lo.index_select(0, inverse))
    assert torch.equal(direct.tube_hi, permuted.tube_hi.index_select(0, inverse))


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_float64_fixed_support_matches_cpu_short_horizon():
    initial_lo = torch.tensor([[1.1, 2.35], [1.2, 2.38]], dtype=torch.float64)
    initial_hi = torch.tensor([[1.4, 2.45], [1.35, 2.44]], dtype=torch.float64)
    cpu = _vdp_solver().verify(initial_lo, initial_hi, steps=3)
    cuda = _vdp_solver().verify(initial_lo.cuda(), initial_hi.cuda(), steps=3)
    assert cuda.device_transfers == 0
    assert cuda.endpoint_lo.device.type == "cuda"
    assert torch.allclose(cpu.endpoint_lo, cuda.endpoint_lo.cpu(), atol=2e-12, rtol=2e-12)
    assert torch.allclose(cpu.endpoint_hi, cuda.endpoint_hi.cpu(), atol=2e-12, rtol=2e-12)
