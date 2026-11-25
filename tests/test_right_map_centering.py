from __future__ import annotations
from torch_tm_flowpipe import Interval, flowpipe_step_flowstar_style_adaptive
from torch_tm_flowpipe.ode_examples import van_der_pol_ode


def _one_step(center_mode: str | None = None):
    kwargs = {}
    if center_mode is not None:
        kwargs["right_map_center_mode"] = center_mode
    return flowpipe_step_flowstar_style_adaptive(
        van_der_pol_ode,
        [Interval(1.1, 1.4), Interval(2.35, 2.45)],
        h=0.002,
        h_min=0.002,
        h_max=0.002,
        order=4,
        target_remainder_radius=1e-4,
        cutoff_threshold=1e-10,
        reset_mode="normalized_insertion",
        **kwargs,
    )


def test_default_right_map_center_mode_is_constant_and_matches_explicit_constant():
    default = _one_step()
    explicit = _one_step("constant")

    assert default.status == "validated"
    assert explicit.status == "validated"
    assert default.flowstar_normal_stats is not None
    assert explicit.flowstar_normal_stats is not None
    assert default.flowstar_normal_stats["right_map_center_mode"] == "constant"
    assert explicit.flowstar_normal_stats["right_map_center_mode"] == "constant"
    assert default.flowstar_normal_stats["center_x"] == explicit.flowstar_normal_stats["center_x"]
    assert default.flowstar_normal_stats["center_y"] == explicit.flowstar_normal_stats["center_y"]
    assert default.flowstar_normal_stats["centered_reset_width_sum"] == explicit.flowstar_normal_stats["centered_reset_width_sum"]


def test_range_midpoint_reconstruction_uses_polynomial_and_remainder_diffs():
    seg = _one_step("range_midpoint")

    assert seg.status == "validated"
    assert seg.flowstar_normal_stats is not None
    stats = seg.flowstar_normal_stats
    assert stats["right_map_center_mode"] == "range_midpoint"
    assert stats["reconstruction_polynomial_max_abs_diff"] <= 1e-15
    assert stats["reconstruction_remainder_lo_diff"] <= 1e-15
    assert stats["reconstruction_remainder_hi_diff"] <= 1e-15
    assert stats["inserted_range_midpoint_shift_abs_sum"] > 0.0


def test_centered_inserted_range_is_translated_without_width_growth_and_scale_reduces():
    seg = _one_step("range_midpoint")
    assert seg.flowstar_normal_stats is not None
    stats = seg.flowstar_normal_stats

    for dim in ("x", "y"):
        assert abs(stats[f"inserted_range_width_{dim}"] - stats[f"centered_inserted_range_width_{dim}"]) <= 1e-14
        assert stats[f"centered_scale_{dim}"] <= stats[f"baseline_scale_{dim}"] + 1e-14
    assert stats["centered_reset_width_sum"] <= stats["baseline_reset_width_sum"] + 1e-14


def test_reset_reconstruction_contains_validated_endpoint_box():
    seg = _one_step("range_midpoint")
    assert seg.flowstar_normal_state is not None
    reconstructed = seg.flowstar_normal_state.endpoint_tm().range_box()
    endpoint = seg.final_tm.range_box()

    for reconstructed_dim, endpoint_dim in zip(reconstructed, endpoint):
        assert reconstructed_dim.contains_interval(endpoint_dim, tol=1e-8)
