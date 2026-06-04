import math

from torch_tm_flowpipe import Interval, flowpipe_multi_step, flowpipe_step, flowpipe_step_flowstar_style_adaptive
from torch_tm_flowpipe.ode_examples import affine_controlled_ode, scalar_quadratic_ode, van_der_pol_ode


def test_scalar_quadratic_one_step_validated_and_contains_exact_samples():
    seg = flowpipe_step(scalar_quadratic_ode, [Interval(0.0, 0.1)], h=0.01, order=4)
    assert seg.status == "validated"
    box = seg.final_tm.range_box()[0]
    for i in range(11):
        x0 = 0.1 * i / 10
        exact = math.tan(0.01 + math.atan(x0))
        assert box.contains(exact, tol=1e-10)


def test_dependency_preserving_multi_step_scalar_quadratic():
    dep = flowpipe_multi_step(
        scalar_quadratic_ode,
        [Interval(0.0, 0.1)],
        h=0.01,
        steps=5,
        order=4,
        mode="dependency_preserving",
    )
    rng = flowpipe_multi_step(
        scalar_quadratic_ode,
        [Interval(0.0, 0.1)],
        h=0.01,
        steps=5,
        order=4,
        mode="range_only",
    )
    assert dep.status == "validated"
    assert rng.status == "validated"
    assert dep.final_tm.n_vars == 1
    assert dep.final_tm.active_variables() == {0}
    assert float(dep.final_tm.max_width()) <= float(rng.final_tm.max_width()) * 1.05
    box = dep.final_tm.range_box()[0]
    for i in range(11):
        x0 = 0.1 * i / 10
        exact = math.tan(0.05 + math.atan(x0))
        assert box.contains(exact, tol=1e-9)


def test_local_tau_not_active_after_each_dependency_preserving_step():
    res = flowpipe_multi_step(
        scalar_quadratic_ode,
        [Interval(0.0, 0.1)],
        h=0.01,
        steps=3,
        order=4,
        mode="dependency_preserving",
    )
    for seg in res.segments:
        assert seg.final_tm.n_vars == 1
        assert seg.final_tm.active_variables() == {0}


def test_affine_control_smoke_validated():
    seg = flowpipe_step(
        affine_controlled_ode,
        [Interval(-0.1, 0.1), Interval(-0.1, 0.1)],
        h=0.01,
        order=3,
        affine_u={"A": [[0.5, -0.25]], "b": [0.0], "error": [0.01]},
    )
    assert seg.status == "validated"
    assert len(seg.final_tm.range_box()) == 2


def test_van_der_pol_short_smoke_validated():
    seg = flowpipe_step(van_der_pol_ode, [Interval(1.0, 1.01), Interval(0.0, 0.01)], h=0.002, order=4)
    assert seg.status == "validated"


def test_flowpipe_step_diagnostics_are_optional_and_do_not_change_result():
    plain = flowpipe_step(scalar_quadratic_ode, [Interval(0.0, 0.1)], h=0.01, order=4)
    diagnostics = []
    instrumented = flowpipe_step(
        scalar_quadratic_ode,
        [Interval(0.0, 0.1)],
        h=0.01,
        order=4,
        diagnostics=diagnostics,
        diagnostics_mode="unit",
        diagnostics_segment_index=7,
    )

    assert instrumented.status == plain.status
    assert instrumented.validation_attempts == plain.validation_attempts
    assert instrumented.message == plain.message
    assert instrumented.final_tm.range_box()[0].to_tuple() == plain.final_tm.range_box()[0].to_tuple()
    assert diagnostics
    assert diagnostics[-1]["mode"] == "unit"
    assert diagnostics[-1]["segment_index"] == 7
    assert diagnostics[-1]["validation_status"] == "validated"

def test_default_flowpipe_step_matches_explicit_growth_validation():
    plain = flowpipe_step(scalar_quadratic_ode, [Interval(0.0, 0.1)], h=0.01, order=4)
    explicit = flowpipe_step(
        scalar_quadratic_ode,
        [Interval(0.0, 0.1)],
        h=0.01,
        order=4,
        validation_mode="growth",
    )

    assert explicit.status == plain.status
    assert explicit.validation_attempts == plain.validation_attempts
    assert explicit.final_tm.range_box()[0].to_tuple() == plain.final_tm.range_box()[0].to_tuple()


def test_flowstar_style_multi_step_uses_fresh_normalized_variables():
    x0 = [Interval(1.1, 1.4), Interval(2.35, 2.45)]
    rescue = flowpipe_multi_step(van_der_pol_ode, x0, h=0.002, steps=2, order=4, mode="flowstar_style")
    dep = flowpipe_multi_step(van_der_pol_ode, x0, h=0.002, steps=2, order=4, mode="dependency_preserving")

    assert rescue.status == "validated"
    assert dep.status == "validated"
    assert rescue.final_tm.n_vars == 2
    assert all(model.polynomial.degree() <= 1 for model in rescue.final_tm)
    assert any(model.polynomial.degree() > 1 for model in dep.final_tm)
    assert [iv.to_tuple() for iv in rescue.final_tm.domain] == [(-1.0, 1.0), (-1.0, 1.0)]
    assert [iv.to_tuple() for iv in dep.final_tm.domain] != [(-1.0, 1.0), (-1.0, 1.0)]

    for seg in rescue.segments:
        assert seg.reset_tm is not None
        assert [iv.to_tuple() for iv in seg.reset_tm.domain] == [(-1.0, 1.0), (-1.0, 1.0)]
        assert seg.reset_tm.active_variables() == {0, 1}
        for reset_iv, raw_iv in zip(seg.reset_tm.range_box(), seg.final_tm.range_box()):
            assert reset_iv.contains_interval(raw_iv, tol=1e-12)


def test_target_remainder_validation_rejects_without_remainder_growth():
    diagnostics = []
    seg = flowpipe_step(
        scalar_quadratic_ode,
        [Interval(0.0, 0.1)],
        h=0.05,
        order=2,
        validation_mode="target_remainder",
        target_remainder_radius=1e-8,
        max_validation_attempts=1,
        diagnostics=diagnostics,
    )

    assert seg.status == "failed"
    assert seg.message == "Picard residual not subset of target remainder"
    assert diagnostics[-1]["subset_result"] is False
    assert diagnostics[-1]["remainder_width_sum"] <= 2.1e-8
    assert diagnostics[-1]["residual_width_sum"] > diagnostics[-1]["target_remainder_width_sum"]


def test_flowstar_style_adaptive_shrinks_after_rejection():
    diagnostics = []
    seg = flowpipe_step_flowstar_style_adaptive(
        scalar_quadratic_ode,
        [Interval(0.0, 0.1)],
        h=0.05,
        order=2,
        h_min=0.0125,
        h_max=0.05,
        target_remainder_radius=1e-16,
        cutoff_threshold=None,
        max_validation_attempts=1,
        diagnostics=diagnostics,
    )

    assert seg.status == "failed"
    assert seg.step_rejections == 3
    assert [row["h"] for row in diagnostics] == [0.05, 0.025, 0.0125]
