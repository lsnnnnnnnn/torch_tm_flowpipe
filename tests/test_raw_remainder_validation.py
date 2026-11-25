from __future__ import annotations
from torch_tm_flowpipe import Interval, flowpipe_step_flowstar_style_adaptive
from torch_tm_flowpipe.flowpipe import _flowstar_raw_remainder_compat_check
from torch_tm_flowpipe.ode_examples import van_der_pol_ode


def _box_tuple(tm):
    if tm is None:
        return None
    return tuple(interval.to_tuple() for interval in tm.range_box())


def _one_step(validation_mode: str, *, diagnostics: list[dict[str, object]] | None = None, ode=van_der_pol_ode):
    return flowpipe_step_flowstar_style_adaptive(
        ode,
        [Interval(1.1, 1.4), Interval(2.35, 2.45)],
        h=0.025,
        h_min=0.025,
        h_max=0.025,
        order=4,
        target_remainder_radius=1e-4,
        cutoff_threshold=1e-10,
        max_validation_attempts=2,
        validation_mode=validation_mode,
        reset_mode="normalized_insertion",
        grow_factor=1.0,
        diagnostics=diagnostics,
        diagnostics_context={"mode": validation_mode, "segment_index": 0, "t_before": 0.0} if diagnostics is not None else None,
    )


def test_default_flowstar_ctrunc_mode_output_unchanged():
    diagnostics: list[dict[str, object]] = []
    seg = _one_step("target_remainder_flowstar_ctrunc", diagnostics=diagnostics)

    assert seg.status == "validated"
    assert diagnostics[-1]["validation_mode"] == "target_remainder_flowstar_ctrunc"
    assert diagnostics[-1]["flowstar_raw_remainder_compat_enabled"] is False
    assert abs(float(diagnostics[-1]["raw_ctrunc_residual_hi_y"]) - 5.8769251659495075e-05) < 1e-14
    assert diagnostics[-1]["subset_tmp_remainder"] is True


def test_diagnostics_do_not_alter_default_validation():
    baseline = _one_step("target_remainder_flowstar_ctrunc")
    diagnostics: list[dict[str, object]] = []
    traced = _one_step("target_remainder_flowstar_ctrunc", diagnostics=diagnostics)

    assert diagnostics
    assert traced.status == baseline.status
    assert traced.h == baseline.h
    assert _box_tuple(traced.tm) == _box_tuple(baseline.tm)
    assert _box_tuple(traced.final_tm) == _box_tuple(baseline.final_tm)
    assert _box_tuple(traced.reset_tm) == _box_tuple(baseline.reset_tm)


def test_accumulated_before_x0_add_fixture_exceeding_target_rejects():
    target = [Interval(-1e-4, 1e-4), Interval(-1e-4, 1e-4)]
    before = [Interval(-1e-5, 1e-5), Interval(-8e-5, 1.1e-4)]
    base = [Interval(0.0, 0.0), Interval(0.0, 0.0)]
    poly_diff = [Interval(0.0, 0.0), Interval(0.0, 0.0)]

    check, subset = _flowstar_raw_remainder_compat_check(target, before, base, poly_diff)

    assert target[0].contains_interval(check[0])
    assert not target[1].contains_interval(check[1])
    assert subset == [True, False]
