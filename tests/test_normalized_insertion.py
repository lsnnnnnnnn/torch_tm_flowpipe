import itertools

from torch_tm_flowpipe import (
    Interval,
    TaylorModel,
    TMVector,
    flowpipe_step_flowstar_style_adaptive,
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
