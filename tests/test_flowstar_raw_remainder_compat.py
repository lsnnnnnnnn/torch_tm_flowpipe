from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

from torch_tm_flowpipe import Interval, flowpipe_step_flowstar_style_adaptive
from torch_tm_flowpipe.flowpipe import _flowstar_raw_remainder_compat_check
from torch_tm_flowpipe.ode_examples import van_der_pol_ode

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments" / "flowstar_raw_remainder_compat_experiment.py"

spec = importlib.util.spec_from_file_location("flowstar_raw_remainder_compat_experiment", SCRIPT)
assert spec is not None and spec.loader is not None
compat_exp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compat_exp)


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


def test_compat_mode_is_opt_in_and_rejects_one_step():
    default_diagnostics: list[dict[str, object]] = []
    compat_diagnostics: list[dict[str, object]] = []

    default_seg = _one_step("target_remainder_flowstar_ctrunc", diagnostics=default_diagnostics, ode=compat_exp.van_der_pol_flowstar_expression_ode)
    compat_seg = _one_step("flowstar_raw_remainder_compat", diagnostics=compat_diagnostics, ode=compat_exp.van_der_pol_flowstar_expression_ode)

    assert default_seg.status == "validated"
    assert compat_seg.status == "failed"
    assert default_diagnostics[-1]["flowstar_raw_remainder_compat_enabled"] is False
    assert compat_diagnostics[-1]["flowstar_raw_remainder_compat_enabled"] is True
    assert compat_diagnostics[-1]["validation_mode"] == "flowstar_raw_remainder_compat"
    assert float(compat_diagnostics[-1]["raw_ctrunc_residual_hi_y"]) > 1e-4
    assert compat_diagnostics[-1]["subset_flowstar_raw_remainder_compat_y"] is False


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


def test_one_step_ledger_writer_works(tmp_path):
    flowstar_trace_row = {
        "status": "rejected",
        "picard_ctrunc_normal_residual_x_lo": "-0.00001",
        "picard_ctrunc_normal_residual_x_hi": "0.00001",
        "picard_ctrunc_normal_residual_y_lo": "-0.000083",
        "picard_ctrunc_normal_residual_y_hi": "0.0001083",
        "target_remainder_x_lo": "-0.0001",
        "target_remainder_x_hi": "0.0001",
        "target_remainder_y_lo": "-0.0001",
        "target_remainder_y_hi": "0.0001",
        "raw_ctrunc_residual_y_hi": "0.0001083",
        "accumulated_remainder_before_x0_add_y_hi": "0.0001083",
        "raw_ctrunc_polynomial_range_y_hi": "2.47",
        "flowstar_full_step_tube_y_hi": "2.48",
        "cutoff_poly_diff_y_hi": "0",
    }
    torch_row = {
        "source": "torch",
        "mode": "flowstar_raw_remainder_compat",
        "t_before": 0.0,
        "h_try": 0.025,
        "status": "rejected",
        "residual_x_lo": "-0.00001",
        "residual_x_hi": "0.00001",
        "residual_y_lo": "-0.000083",
        "residual_y_hi": "0.00010831",
        "target_x_lo": "-0.0001",
        "target_x_hi": "0.0001",
        "target_y_lo": "-0.0001",
        "target_y_hi": "0.0001",
        "subset_x": True,
        "subset_y": False,
        "failed_dim": "y",
        "raw_ctrunc_residual_y_hi": "0.00010831",
        "accumulated_before_x0_add_y_hi": "0.00010831",
        "polynomial_range_y_hi": "2.47",
        "full_step_tube_y_hi": "2.48",
        "cutoff_poly_diff_y_hi": "0",
        "notes": "fixture",
    }

    ledger = compat_exp.build_ledger(flowstar_trace_row, [torch_row])
    ledger_path = tmp_path / "raw_remainder_compat_ledger.csv"
    report_path = tmp_path / "raw_remainder_compat_report.md"
    compat_exp.write_ledger(ledger_path, ledger)
    compat_exp.write_report(report_path, ledger)

    with ledger_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert rows[1]["matches_flowstar_accept_reject"] == "true"
    assert report_path.read_text(encoding="utf-8").splitlines()[0] == "# Flow* Raw Remainder Compatibility Experiment"
