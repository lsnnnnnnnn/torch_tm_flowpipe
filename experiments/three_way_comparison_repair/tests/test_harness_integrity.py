from __future__ import annotations

import inspect
from pathlib import Path

from collect_results import (
    _common_time_summary,
    _failure_summary,
    _late_point_violations,
    collect,
)
from common import (
    FAILURE_CATEGORIES,
    FROZEN_RESULT,
    PROTOCOL_BOX,
    RAW_FIELDS,
    load_spec,
    make_row,
    manifest_digest,
    sha256_manifest,
)
from run_flowstar_audit import render_fixed_cpp, render_parity_cpp


def _row(**changes):
    values = make_row(
        tool="flowstar",
        variant="flowstar_stock",
        protocol=PROTOCOL_BOX,
        system="riccati",
        h=0.01,
        horizon=0.02,
        step_index=1,
        absolute_time=0.01,
        state_index=0,
        interval_kind="endpoint_raw",
        lower=-0.01,
        upper=0.11,
        exact=(0.0, 0.1001001001001001),
        native_validation_status="advance_returned_1",
        analytic_reference_status="passed",
        failure_category="",
    )
    values.update(changes)
    return {key: str(values[key]) for key in RAW_FIELDS}


def test_stock_source_has_no_post_advance_mutation_and_diagnostic_does() -> None:
    spec = load_spec()
    system = spec["systems"]["riccati"]
    keyword = "setting.tm_setting.remainder_estimation[state]"
    stock = render_fixed_cpp(
        system,
        protocol=PROTOCOL_BOX,
        h=0.01,
        horizon=0.01,
        order=2,
        candidate=1e-4,
        cutoff=1e-15,
        variant="flowstar_stock",
    )
    diagnostic = render_fixed_cpp(
        system,
        protocol=PROTOCOL_BOX,
        h=0.01,
        horizon=0.01,
        order=2,
        candidate=1e-4,
        cutoff=1e-15,
        variant="flowstar_candidate_reinjection_diagnostic",
    )
    assert keyword not in stock
    assert keyword in diagnostic
    assert diagnostic.count(keyword) == 1


def test_candidate_reinjection_renderer_reproduces_historical_assignment() -> None:
    spec = load_spec()
    source = render_fixed_cpp(
        spec["systems"]["riccati"],
        protocol=PROTOCOL_BOX,
        h=0.01,
        horizon=0.01,
        order=2,
        candidate=1e-4,
        cutoff=1e-15,
        variant="flowstar_candidate_reinjection_diagnostic",
    )
    assert "next.tmvPre.tms[state].remainder =" in source
    assert "setting.tm_setting.remainder_estimation[state];" in source
    native_capture = source.index("native_widths[state] =")
    mutation = source.index("next.tmvPre.tms[state].remainder =")
    post_capture = source.index("post_widths[state] =")
    assert native_capture < mutation < post_capture


def test_no_failed_segment_contributes_later_points() -> None:
    failure = _row(
        step_index="2",
        absolute_time="0.02",
        interval_kind="failure",
        lower="",
        upper="",
        width="",
        failure_category="fixed_step_validation_failed",
    )
    assert not _late_point_violations([_row(), failure])
    later = _row(step_index="3", absolute_time="0.03")
    assert _late_point_violations([_row(), failure, later])


def test_failure_categories_are_nonempty_and_summaries_preserve_them() -> None:
    failure = _row(
        step_index="2",
        absolute_time="0.02",
        interval_kind="failure",
        lower="",
        upper="",
        width="",
        failure_category="first_picard_inclusion_failed",
    )
    summary = _failure_summary([_row(), failure])
    assert summary[0]["failure_category"] in FAILURE_CATEGORIES
    assert all(category for category in FAILURE_CATEGORIES)


def test_frozen_historical_result_manifest_is_unchanged() -> None:
    assert FROZEN_RESULT.exists()
    assert (
        manifest_digest(sha256_manifest(FROZEN_RESULT))
        == "3da0feed583dde3055fbbe039de32edec9516a2028562585eab448567a1c3f02"
    )


def test_report_rows_keep_equal_explicit_absolute_times() -> None:
    rows = [
        _row(tool="torch_tm_flowpipe"),
        _row(tool="diffreach"),
        _row(tool="flowstar"),
    ]
    summary = _common_time_summary(rows)
    assert {row["absolute_time"] for row in summary} == {"0.01"}
    assert {row["interval_kind"] for row in rows} == {"endpoint_raw"}


def test_plots_filter_the_requested_interval_kind() -> None:
    import plot_results

    source = inspect.getsource(plot_results.inflation)
    assert "(raw.interval_kind == kind)" in source
    source = inspect.getsource(plot_results.width_curves)
    assert '(raw.interval_kind == "endpoint_raw")' in source


def test_original_flowstar_parity_configuration_is_exact() -> None:
    spec = load_spec()
    original = spec["flowstar"]["original_vanderpol"]
    assert original == {
        "horizon": 10.0,
        "step_policy": "adaptive_0.002_to_0.1",
        "order": 4,
        "cutoff": 1e-10,
        "candidate_remainder": 1e-4,
        "symbolic_remainder_window": 100,
        "preconditioning": "QR",
    }
    source = render_parity_cpp("parity_test")
    for fragment in (
        'ODE<Real> ode({"y", "(1 - x^2) * y - x", "1"}, vars);',
        "box[x_id] = Interval(1.1, 1.4);",
        "box[y_id] = Interval(2.35, 2.45);",
        "Symbolic_Remainder symbolic(initial_set, 100);",
        "ode.reach(result, initial_set, 10.0, setting, safe_set, symbolic);",
    ):
        assert fragment in source


def test_partial_outcome_does_not_hide_flowstar_violations(tmp_path: Path) -> None:
    source = inspect.getsource(collect)
    assert 'row["tool"] != "flowstar"' in source
    assert '"torch_and_diffreach_sampled_trajectories"' in source
    assert '"sampled_trajectories"' in source
