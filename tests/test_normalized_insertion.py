import itertools

import pytest
import torch

from torch_tm_flowpipe import (
    BatchedMonomialBasis,
    BatchedPolynomial,
    BatchedTaylorModel,
    FlowstarNormalFlowpipeState,
    Interval,
    TaylorModel,
    TMVector,
    preserve_complete_polynomial_carry,
    flowpipe_step_flowstar_style_adaptive,
    insert_ctrunc_normal_horner_diagnostic,
    insert_ctrunc_normal_like,
)
from torch_tm_flowpipe.ode_examples import scalar_quadratic_ode, van_der_pol_ode


def test_insert_ctrunc_normal_like_contains_sampled_direct_composition():
    domain = [Interval(-1.0, 1.0), Interval(-1.0, 1.0)]
    x = TaylorModel.variable(0, domain, order=5)
    y = TaylorModel.variable(1, domain, order=5)
    outer = x * x + 0.5 * x * y - y + Interval(-1e-6, 1e-6)
    inner = TMVector([x + 0.25 * y, y - 0.2 * x + 0.1])

    diagnostics = {}
    composed = insert_ctrunc_normal_like(outer, inner, order=4, cutoff_threshold=1e-12, domain=domain, diagnostics=diagnostics)

    box = composed.range_box()
    for point in itertools.product([-1.0, -0.5, 0.0, 0.5, 1.0], repeat=2):
        inner_value = [model.evaluate_point(point) for model in inner]
        direct = outer.evaluate_point(inner_value)
        assert box.contains(direct, tol=1e-9)
    assert diagnostics["composed_poly_range_width"] > 0.0
    assert diagnostics["output_remainder_width"] >= 0.0


def test_insert_ctrunc_normal_like_moves_truncation_and_cutoff_to_remainder():
    domain = [Interval(-1.0, 1.0), Interval(-1.0, 1.0)]
    x = TaylorModel.variable(0, domain, order=4)
    y = TaylorModel.variable(1, domain, order=4)
    outer = x.pow_int(3) + 1e-11 * y
    diagnostics = {}

    composed = insert_ctrunc_normal_like(outer, TMVector([x, y]), order=2, cutoff_threshold=1e-10, domain=domain, diagnostics=diagnostics)

    assert composed.remainder.width().item() > 1.0
    assert diagnostics["insertion_truncation_width"] > 1.0
    assert diagnostics["insertion_cutoff_width"] > 0.0


def test_insert_ctrunc_normal_horner_diagnostic_contains_sampled_direct_values():
    domain = [Interval(-1.0, 1.0), Interval(-1.0, 1.0)]
    x = TaylorModel.variable(0, domain, order=5)
    y = TaylorModel.variable(1, domain, order=5)
    outer = x.pow_int(3) - 0.25 * x * y + 0.125 * y.pow_int(2) + Interval(-1e-7, 1e-7)
    inner = TMVector([x + 0.3 * y, y - 0.1 * x + 0.05])

    diagnostic = insert_ctrunc_normal_horner_diagnostic(
        outer,
        inner,
        order=3,
        cutoff_threshold=1e-12,
        domain=domain,
    )

    assert diagnostic.stage_ranges
    assert diagnostic.top_components
    assert diagnostic.summary["horner_stage_count"] == len(diagnostic.stage_ranges)
    box = diagnostic.horner_result.range_box()
    for point in itertools.product([-1.0, -0.5, 0.0, 0.5, 1.0], repeat=2):
        inner_value = [model.evaluate_point(point) for model in inner]
        direct = outer.evaluate_point(inner_value)
        assert box.contains(direct, tol=1e-8)


def test_insert_ctrunc_normal_horner_uncertainty_is_recorded_and_conservative():
    domain = [Interval(-1.0, 1.0), Interval(-1.0, 1.0)]
    x = TaylorModel.variable(0, domain, order=5)
    y = TaylorModel.variable(1, domain, order=5)
    outer = x.pow_int(4) + 0.5 * y.pow_int(3) + 1e-11 * x
    inner = TMVector([x + 0.2 * y + Interval(-1e-5, 1e-5), y - 0.15 * x])

    diagnostic = insert_ctrunc_normal_horner_diagnostic(
        outer,
        inner,
        order=2,
        cutoff_threshold=1e-10,
        domain=domain,
    )

    assert diagnostic.summary["horner_truncation_width_sum"] > 0.0
    assert diagnostic.summary["horner_cutoff_width_sum"] > 0.0
    assert diagnostic.summary["horner_range_width_sum"] >= diagnostic.summary["horner_normal_range_width_sum"]
    box = diagnostic.horner_result.range_box()
    for point in itertools.product([-1.0, 0.0, 1.0], repeat=2):
        inner_value = [model.evaluate_point(point) for model in inner]
        direct = outer.evaluate_point(inner_value)
        assert box.contains(direct, tol=1e-7)


def test_normalized_insertion_state_contains_sampled_endpoint_values():
    x0 = [Interval(1.1, 1.4), Interval(2.35, 2.45)]
    seg = flowpipe_step_flowstar_style_adaptive(
        van_der_pol_ode,
        x0,
        h=0.002,
        h_min=0.002,
        h_max=0.002,
        order=4,
        target_remainder_radius=1e-4,
        cutoff_threshold=1e-10,
        reset_mode="normalized_insertion",
    )

    assert seg.status == "validated"
    assert seg.flowstar_normal_state is not None
    assert seg.reset_tm is not None
    reset_box = seg.reset_tm.range_box()
    for point in itertools.product([-1.0, 0.0, 1.0], repeat=2):
        endpoint = [model.evaluate_point(point) for model in seg.final_tm]
        for value, box in zip(endpoint, reset_box):
            assert box.contains(value, tol=1e-8)


def test_complete_polynomial_normalized_carry_retains_validated_endpoint_terms_and_remainder():
    x0 = [Interval(1.1, 1.4), Interval(2.35, 2.45)]
    first = flowpipe_step_flowstar_style_adaptive(
        van_der_pol_ode,
        x0,
        h=0.002,
        h_min=0.002,
        h_max=0.002,
        order=4,
        target_remainder_radius=1e-4,
        cutoff_threshold=1e-10,
        reset_mode="normalized_insertion_complete_polynomial",
    )

    assert first.status == "validated"
    assert first.reset_tm is not None
    assert first.flowstar_normal_state is not None
    assert first.flowstar_normal_state.complete_initial_tm is not None
    for endpoint, carried in zip(first.final_tm, first.reset_tm):
        assert endpoint.polynomial.terms.keys() == carried.polynomial.terms.keys()
        for exponent in endpoint.polynomial.terms:
            assert endpoint.polynomial.terms[exponent].item() == carried.polynomial.terms[exponent].item()
        assert endpoint.remainder.to_tuple() == carried.remainder.to_tuple()
    stats = first.flowstar_normal_stats
    assert stats is not None
    assert stats["complete_polynomial_carry"] is True
    assert stats["complete_carry_retained_terms"] == sum(
        len(model.polynomial.terms) for model in first.final_tm
    )
    assert stats["complete_carry_intervalized_term_count"] == 0

    second = flowpipe_step_flowstar_style_adaptive(
        van_der_pol_ode,
        first.reset_tm,
        h=0.002,
        h_min=0.002,
        h_max=0.002,
        order=4,
        target_remainder_radius=1e-4,
        cutoff_threshold=1e-10,
        reset_mode="normalized_insertion_complete_polynomial",
        flowstar_normal_state=first.flowstar_normal_state,
    )
    assert second.status == "validated"


def test_complete_polynomial_carry_preserves_generic_correlated_model_and_clones_it():
    domain = [Interval(-1.0, 1.0), Interval(-1.0, 1.0)]
    u = TaylorModel.variable(0, domain, order=4)
    v = TaylorModel.variable(1, domain, order=4)
    complete = TMVector(
        [
            1.0 + u + 0.25 * u * v + Interval(-1e-8, 2e-8),
            2.0 - u * u + 0.5 * v + Interval(-3e-8, 4e-8),
        ]
    )
    state = FlowstarNormalFlowpipeState(
        tmv_pre=complete,
        tmv_right=TMVector.identity(domain, order=4),
        domain=domain,
        center=[1.0, 2.0],
        scales=[1.25, 1.5],
        complete_initial_tm=complete,
    )

    carried = state.normalized_initial_tm(4)
    assert carried is not complete
    for point in itertools.product([-1.0, -0.5, 0.0, 0.5, 1.0], repeat=2):
        for original, copied in zip(complete, carried):
            assert original.evaluate_point(point) == copied.evaluate_point(point)
            assert original.remainder.to_tuple() == copied.remainder.to_tuple()
    diagnostics = state.diagnostic_widths()
    assert diagnostics["complete_polynomial_carry"] is True
    assert diagnostics["complete_carry_max_degree"] == 2
    assert diagnostics["complete_carry_retained_terms"] == 6


@pytest.mark.parametrize("batch", [1, 8, 64])
def test_complete_polynomial_carry_is_batch_generic_and_permutation_equivariant(batch):
    basis = BatchedMonomialBasis.build(3, 4, "cpu")
    coeffs = torch.zeros((batch, 2, basis.num_terms), dtype=torch.float64)
    row = torch.arange(batch, dtype=torch.float64)
    coeffs[:, 0, basis.constant_index] = 1.0 + row / 100.0
    coeffs[:, 1, basis.constant_index] = 2.0 - row / 200.0
    coeffs[:, 0, basis.term_index((1, 0, 0))] = 0.25 + row / 1000.0
    coeffs[:, 1, basis.term_index((1, 1, 0))] = -0.125 - row / 2000.0
    rem_lo = torch.stack((-1e-8 - row * 1e-11, -2e-8 - row * 2e-11), dim=1)
    rem_hi = torch.stack((2e-8 + row * 1e-11, 3e-8 + row * 2e-11), dim=1)
    domain_lo = torch.stack((-torch.ones_like(row), -0.9 * torch.ones_like(row), torch.zeros_like(row)), dim=1)
    domain_hi = torch.stack((torch.ones_like(row), (0.9 + row / 10000.0), 0.01 + row / 100000.0), dim=1)
    endpoint = BatchedTaylorModel(
        BatchedPolynomial(coeffs, basis), rem_lo, rem_hi, domain_lo, domain_hi
    )

    carried = preserve_complete_polynomial_carry(endpoint)
    assert carried is not endpoint
    assert carried.poly.coeffs.data_ptr() != endpoint.poly.coeffs.data_ptr()
    for actual, expected in zip(
        (carried.poly.coeffs, carried.rem_lo, carried.rem_hi, carried.domain_lo, carried.domain_hi),
        (endpoint.poly.coeffs, endpoint.rem_lo, endpoint.rem_hi, endpoint.domain_lo, endpoint.domain_hi),
    ):
        assert torch.equal(actual, expected)

    permutation = torch.arange(batch - 1, -1, -1)
    permuted_endpoint = BatchedTaylorModel(
        BatchedPolynomial(endpoint.poly.coeffs[permutation], basis),
        endpoint.rem_lo[permutation],
        endpoint.rem_hi[permutation],
        endpoint.domain_lo[permutation],
        endpoint.domain_hi[permutation],
    )
    permuted_carry = preserve_complete_polynomial_carry(permuted_endpoint)
    assert torch.equal(permuted_carry.poly.coeffs, carried.poly.coeffs[permutation])
    assert torch.equal(permuted_carry.rem_lo, carried.rem_lo[permutation])
    assert torch.equal(permuted_carry.rem_hi, carried.rem_hi[permutation])


def test_default_flowstar_style_adaptive_reset_is_unchanged():
    x0 = [Interval(0.0, 0.1)]
    default = flowpipe_step_flowstar_style_adaptive(
        scalar_quadratic_ode,
        x0,
        h=0.01,
        h_min=0.01,
        h_max=0.01,
        order=4,
        target_remainder_radius=1e-4,
        cutoff_threshold=None,
    )
    explicit = flowpipe_step_flowstar_style_adaptive(
        scalar_quadratic_ode,
        x0,
        h=0.01,
        h_min=0.01,
        h_max=0.01,
        order=4,
        target_remainder_radius=1e-4,
        cutoff_threshold=None,
        reset_mode="normalized_endpoint_box",
    )

    assert default.status == explicit.status
    assert default.flowstar_normal_state is None
    assert explicit.flowstar_normal_state is None
    assert default.reset_tm is not None
    assert explicit.reset_tm is not None
    assert default.reset_tm.range_box()[0].to_tuple() == explicit.reset_tm.range_box()[0].to_tuple()


def test_normalized_insertion_symqueue_carries_queue_state():
    x0 = [Interval(1.1, 1.4), Interval(2.35, 2.45)]
    first = flowpipe_step_flowstar_style_adaptive(
        van_der_pol_ode,
        x0,
        h=0.002,
        h_min=0.002,
        h_max=0.002,
        order=4,
        target_remainder_radius=1e-4,
        cutoff_threshold=1e-10,
        reset_mode="normalized_insertion_symqueue",
        flowstar_symbolic_queue_max_size=100,
    )

    assert first.status == "validated"
    assert first.flowstar_normal_state is not None
    assert first.flowstar_normal_state.symbolic_queue is not None
    assert first.reset_tm is not None
    assert first.flowstar_symbolic_queue_stats is not None
    assert first.flowstar_symbolic_queue_stats["queue_size_after"] == 1
    assert "propagated_symbolic_width_sum" in first.flowstar_symbolic_queue_stats

    second = flowpipe_step_flowstar_style_adaptive(
        van_der_pol_ode,
        first.reset_tm,
        h=0.002,
        h_min=0.002,
        h_max=0.002,
        order=4,
        target_remainder_radius=1e-4,
        cutoff_threshold=1e-10,
        reset_mode="normalized_insertion_symqueue",
        flowstar_symbolic_queue_max_size=100,
        flowstar_normal_state=first.flowstar_normal_state,
    )

    assert second.status == "validated"
    assert second.flowstar_symbolic_queue_stats is not None
    assert second.flowstar_symbolic_queue_stats["queue_size_after"] == 2
    assert second.flowstar_normal_state is not None
    assert second.flowstar_normal_state.initial_remainders is not None



def test_normalized_insertion_symqueue_split_keeps_target_seed_clean():
    x0 = [Interval(1.1, 1.4), Interval(2.35, 2.45)]
    first = flowpipe_step_flowstar_style_adaptive(
        van_der_pol_ode,
        x0,
        h=0.002,
        h_min=0.002,
        h_max=0.002,
        order=4,
        target_remainder_radius=1e-4,
        cutoff_threshold=1e-10,
        reset_mode="normalized_insertion_symqueue_split",
        flowstar_symbolic_queue_max_size=100,
    )

    assert first.status == "validated"
    assert first.flowstar_normal_state is not None
    assert first.reset_tm is not None

    second = flowpipe_step_flowstar_style_adaptive(
        van_der_pol_ode,
        first.reset_tm,
        h=0.002,
        h_min=0.002,
        h_max=0.002,
        order=4,
        target_remainder_radius=1e-4,
        cutoff_threshold=1e-10,
        reset_mode="normalized_insertion_symqueue_split",
        flowstar_symbolic_queue_max_size=100,
        flowstar_normal_state=first.flowstar_normal_state,
    )

    assert second.status == "validated"
    assert second.flowstar_normal_state is not None
    assert second.flowstar_normal_state.initial_remainders is None
    assert second.flowstar_symbolic_queue_stats is not None
    stats = second.flowstar_symbolic_queue_stats
    assert stats["semantic_split"] is True
    assert stats["queue_size_after"] == 2
    assert stats["target_checked_width"] <= 1e-15
    assert stats["symbolic_contribution_width"] > 0.0
    assert abs(stats["materialized_for_output_width"] - stats["symbolic_contribution_width"]) < 1e-15
    assert stats["total_range_width_with_symbolic"] >= stats["ordinary_only_range_width"]


def test_normalized_insertion_symqueue_v2_keeps_target_clean_and_records_linear_state():
    x0 = [Interval(1.1, 1.4), Interval(2.35, 2.45)]
    first = flowpipe_step_flowstar_style_adaptive(
        van_der_pol_ode,
        x0,
        h=0.002,
        h_min=0.002,
        h_max=0.002,
        order=4,
        target_remainder_radius=1e-4,
        cutoff_threshold=1e-10,
        reset_mode="normalized_insertion_symqueue_v2",
        symbolic_queue_mode="flowstar_linear_v2",
        flowstar_symbolic_queue_max_size=100,
    )

    assert first.status == "validated"
    assert first.flowstar_normal_state is not None
    assert first.flowstar_normal_state.symbolic_queue is not None

    second = flowpipe_step_flowstar_style_adaptive(
        van_der_pol_ode,
        first.reset_tm,
        h=0.002,
        h_min=0.002,
        h_max=0.002,
        order=4,
        target_remainder_radius=1e-4,
        cutoff_threshold=1e-10,
        reset_mode="normalized_insertion_symqueue_v2",
        symbolic_queue_mode="flowstar_linear_v2",
        flowstar_symbolic_queue_max_size=100,
        flowstar_normal_state=first.flowstar_normal_state,
    )

    assert second.status == "validated"
    assert second.flowstar_normal_state is not None
    assert second.flowstar_normal_state.initial_remainders is None
    assert second.flowstar_symbolic_queue_stats is not None
    stats = second.flowstar_symbolic_queue_stats
    assert stats["symbolic_queue_mode"] == "flowstar_linear_v2"
    assert stats["semantic_split"] is True
    assert stats["j_count"] == 2
    assert stats["phi_l_count"] == 2
    assert stats["target_check_width_sum"] <= 1e-15
    assert stats["output_only_symbolic_width_sum"] > 0.0
    assert stats["output_range_includes_symbolic_contributions"] is True
    assert stats["current_linear_map_norm"] > 0.0
    assert stats["scalar_x"] > 0.0
    assert stats["scalar_y"] > 0.0


def test_normalized_insertion_normal_eval_range_mode_records_old_and_normal_ranges():
    x0 = [Interval(1.1, 1.4), Interval(2.35, 2.45)]
    seg = flowpipe_step_flowstar_style_adaptive(
        van_der_pol_ode,
        x0,
        h=0.002,
        h_min=0.002,
        h_max=0.002,
        order=4,
        target_remainder_radius=1e-4,
        cutoff_threshold=1e-10,
        reset_mode="normalized_insertion",
        right_map_range_mode="normal_eval",
    )

    assert seg.status == "validated"
    assert seg.flowstar_normal_stats is not None
    stats = seg.flowstar_normal_stats
    assert stats["right_map_range_mode"] == "normal_eval"
    assert stats["old_right_map_range_width_sum"] >= 0.0
    assert stats["normal_right_map_range_width_sum"] >= 0.0
    assert stats["inserted_endpoint_width_sum"] == stats["normal_right_map_range_width_sum"]
