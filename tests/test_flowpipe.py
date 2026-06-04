import math

from torch_tm_flowpipe import Interval, flowpipe_multi_step, flowpipe_step
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
