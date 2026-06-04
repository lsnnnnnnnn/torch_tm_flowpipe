from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

from torch_tm_flowpipe import (
    Interval,
    SymbolicRemainderState,
    TMVector,
    TaylorModel,
    flowpipe_step,
    introduce_symbolic_remainders,
    materialize_all_symbols,
)
from torch_tm_flowpipe.ode_examples import scalar_quadratic_ode

ROOT = Path(__file__).resolve().parents[1]


def _load_stage3_module():
    spec = importlib.util.spec_from_file_location(
        "flowstar_benchmark_symbolic_remainder_diagnostics",
        ROOT / "experiments" / "flowstar_benchmark_symbolic_remainder_diagnostics.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _small_params():
    return {
        "initial_set": {"x": [1.1, 1.4], "y": [2.35, 2.45]},
        "taylor_order": 4,
        "time_horizon": 0.0005,
    }


def test_flowpipe_step_default_and_symbolic_false_match_plain_result():
    plain = flowpipe_step(scalar_quadratic_ode, [Interval(0.0, 0.1)], h=0.01, order=4)
    explicit_false = flowpipe_step(
        scalar_quadratic_ode,
        [Interval(0.0, 0.1)],
        h=0.01,
        order=4,
        symbolic_remainder=False,
    )

    assert explicit_false.status == plain.status
    assert explicit_false.validation_attempts == plain.validation_attempts
    assert explicit_false.message == plain.message
    assert explicit_false.final_tm.range_box()[0].to_tuple() == plain.final_tm.range_box()[0].to_tuple()
    assert explicit_false.symbolic_remainder is False
    assert explicit_false.symbolic_remainder_state is None


def test_symbolic_noise_variable_has_unit_interval_domain():
    seg = flowpipe_step(
        scalar_quadratic_ode,
        [Interval(0.0, 0.1)],
        h=0.01,
        order=4,
        symbolic_remainder=True,
        max_symbolic_remainders=4,
        symbolic_remainder_state=SymbolicRemainderState.empty(4),
    )

    assert seg.status == "validated"
    assert seg.symbolic_remainder_state is not None
    assert len(seg.symbolic_remainder_state.symbols) == 1
    symbol = seg.symbolic_remainder_state.symbols[0]
    assert seg.final_tm.domain[symbol.var_index].to_tuple() == (-1.0, 1.0)


def test_materializing_symbolic_remainder_is_conservative():
    domain = [Interval(0.0, 1.0)]
    tm = TMVector([TaylorModel.variable(0, domain, order=3).with_remainder(Interval(-0.2, 0.4))])
    symbolic, state, _stats = introduce_symbolic_remainders(
        tm,
        SymbolicRemainderState.empty(4),
        max_symbolic_remainders=4,
    )

    materialized = materialize_all_symbols(symbolic, state)
    assert materialized[0].range_box().contains_interval(symbolic[0].range_box(), tol=1e-12)


def test_symbolic_multiplication_conservative_against_samples():
    domain = [Interval(-0.1, 0.1)]
    tm = TMVector([TaylorModel.variable(0, domain, order=4).with_remainder(Interval(-0.02, 0.04))])
    symbolic, _state, _stats = introduce_symbolic_remainders(
        tm,
        SymbolicRemainderState.empty(4),
        max_symbolic_remainders=4,
    )
    product = symbolic[0] * symbolic[0]
    box = product.range_box()

    for x0 in (-0.1, -0.05, 0.0, 0.05, 0.1):
        for eps in (-1.0, -0.5, 0.0, 0.5, 1.0):
            value = product.evaluate_point([x0, eps])
            assert box.contains(value, tol=1e-12)


def test_queue_overflow_materializes_oldest_symbol():
    domain = [Interval(0.0, 0.1), Interval(0.2, 0.3)]
    tm = TMVector(
        [
            TaylorModel.variable(0, domain, order=3).with_remainder(Interval(-0.01, 0.01)),
            TaylorModel.variable(1, domain, order=3).with_remainder(Interval(-0.02, 0.02)),
        ]
    )
    symbolic, state, stats = introduce_symbolic_remainders(
        tm,
        SymbolicRemainderState.empty(1),
        max_symbolic_remainders=1,
    )

    assert len(state.symbols) == 1
    assert state.symbols[0].symbol_id == 1
    assert stats["materialized_symbol_ids"] == (0,)
    assert stats["materialized_remainder_width_sum"] > 0.0
    assert symbolic.n_vars == 3


def test_stage3_short_run_writes_required_outputs_and_sanitized_csv(tmp_path):
    stage3 = _load_stage3_module()
    refs = [{"segment_index": 0, "t_lo": 0.0, "t_hi": 0.0005}]

    baseline_summary, baseline_segments, baseline_breakdowns = stage3.run_symbolic_diagnostic(
        _small_params(),
        refs,
        mode="range_only",
        order=4,
        substep_factor=1,
        symbolic_remainder=False,
        queue_size="",
        max_wall_s_per_run=10,
        max_horizon=0.0005,
    )
    symbolic_summary, symbolic_segments, symbolic_breakdowns = stage3.run_symbolic_diagnostic(
        _small_params(),
        refs,
        mode="range_only",
        order=4,
        substep_factor=1,
        symbolic_remainder=True,
        queue_size=4,
        max_wall_s_per_run=10,
        max_horizon=0.0005,
    )
    summaries = [baseline_summary, symbolic_summary]
    segments = baseline_segments + symbolic_segments
    breakdowns = baseline_breakdowns + symbolic_breakdowns
    stage3.write_outputs(tmp_path, summaries, segments, breakdowns)

    assert summaries and segments and breakdowns
    for name in [
        "symbolic_remainder_summary.csv",
        "symbolic_remainder_segments.csv",
        "symbolic_remainder_breakdown.csv",
        "symbolic_remainder_report.md",
    ]:
        assert (tmp_path / name).exists()
    report = (tmp_path / "symbolic_remainder_report.md").read_text(encoding="utf-8")
    assert "diagnostic-only" in report

    for csv_name in [
        "symbolic_remainder_summary.csv",
        "symbolic_remainder_segments.csv",
        "symbolic_remainder_breakdown.csv",
    ]:
        with (tmp_path / csv_name).open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                for value in row.values():
                    assert value.strip().lower() not in {"nan", "inf", "+inf", "-inf", "infinity", "-infinity"}
