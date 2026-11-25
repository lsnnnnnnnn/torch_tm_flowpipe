from __future__ import annotations
from fractions import Fraction
import pytest
import torch
from torch_tm_flowpipe import DenseRangePolicy, FlowstarNormalFlowpipeState, Interval, PolynomialODE
from torch_tm_flowpipe.batched_dense_tm import (
    DenseExecutionCounters,
    dense_picard_validate_step,
    sparse_tmvector_to_dense,
)


@pytest.mark.unit
def test_exact_decimal_torch_initialization_contains_exact_box() -> None:
    state = FlowstarNormalFlowpipeState.from_exact_decimal_box(
        [("11/10", "7/5"), ("47/20", "49/20")], 4
    )
    expected = ((Fraction(11, 10), Fraction(7, 5)), (Fraction(47, 20), Fraction(49, 20)))
    for center, scale, (lower, upper) in zip(
        state.center, state.scales, expected, strict=True
    ):
        center_q = Fraction.from_float(center)
        scale_q = Fraction.from_float(scale)
        assert center_q - scale_q <= lower
        assert upper <= center_q + scale_q
    assert state.diagnostics is not None
    assert state.diagnostics["initialization_contract"] == "exact_decimal_contract"
    assert all(
        row["contains_expected_exact"]
        for row in state.diagnostics["exact_decimal_containment_witness"]
    )


@pytest.mark.unit
def test_exact_decimal_compensation_is_consumed_by_actual_dense_picard_input() -> None:
    legacy = FlowstarNormalFlowpipeState.from_initial_box(
        [Interval(1.1, 1.4), Interval(2.35, 2.45)], 4
    )
    exact = FlowstarNormalFlowpipeState.from_exact_decimal_box(
        [("11/10", "7/5"), ("47/20", "49/20")], 4
    )
    counters = DenseExecutionCounters()
    legacy_input = sparse_tmvector_to_dense(
        legacy.normalized_initial_tm(4).extend_domain(Interval(0.0, 0.01)),
        order=4,
        device="cpu",
        dtype=torch.float64,
        counters=counters,
        segment_boundary=True,
        range_policy=DenseRangePolicy(),
        range_trace=[],
    )
    exact_input = sparse_tmvector_to_dense(
        exact.normalized_initial_tm(4).extend_domain(Interval(0.0, 0.01)),
        order=4,
        device="cpu",
        dtype=torch.float64,
        counters=counters,
        segment_boundary=True,
        range_policy=DenseRangePolicy(),
        range_trace=[],
    )
    assert not torch.equal(legacy_input.poly.coeffs, exact_input.poly.coeffs)

    ode = PolynomialODE.from_system_spec(
        {
            "state_names": ["x", "y"],
            "initial_box": [[1.1, 1.4], [2.35, 2.45]],
            "rhs": [
                {"terms": [{"coefficient": 1.0, "powers": [0, 1]}]},
                {"terms": [
                    {"coefficient": 1.0, "powers": [0, 1]},
                    {"coefficient": -1.0, "powers": [1, 0]},
                    {"coefficient": -1.0, "powers": [2, 1]},
                ]},
            ],
        }
    )
    common = dict(
        h=0.01,
        order=4,
        tau_index=2,
        target_remainder_radius=1e-4,
        cutoff_threshold=1e-10,
        max_validation_attempts=2,
        validation_eps=1e-12,
        validation_mode="flowstar_raw_remainder_compat",
    )
    legacy_step = dense_picard_validate_step(ode, legacy_input, **common)
    exact_step = dense_picard_validate_step(ode, exact_input, **common)
    assert legacy_step.status == exact_step.status == "validated"
    assert not torch.equal(legacy_step.segment_tm.poly.coeffs, exact_step.segment_tm.poly.coeffs)


@pytest.mark.unit
def test_exact_decimal_initialization_rejects_pre_rounded_float_inputs() -> None:
    with pytest.raises(TypeError, match="rejects binary floating-point"):
        FlowstarNormalFlowpipeState.from_exact_decimal_box([(1.1, "7/5")], 4)


@pytest.mark.unit
def test_torch_production_picard_observer_is_read_only() -> None:
    state = FlowstarNormalFlowpipeState.from_initial_box(
        [Interval(1.1, 1.4), Interval(2.35, 2.45)], 4
    )
    base = sparse_tmvector_to_dense(
        state.normalized_initial_tm(4).extend_domain(Interval(0.0, 0.01)),
        order=4,
        device="cpu",
        dtype=torch.float64,
        counters=DenseExecutionCounters(),
        segment_boundary=True,
        range_policy=DenseRangePolicy(),
        range_trace=[],
    )
    ode = PolynomialODE.from_system_spec(
        {
            "state_names": ["x", "y"],
            "initial_box": [[1.1, 1.4], [2.35, 2.45]],
            "rhs": [
                {"terms": [{"coefficient": 1.0, "powers": [0, 1]}]},
                {"terms": [
                    {"coefficient": 1.0, "powers": [0, 1]},
                    {"coefficient": -1.0, "powers": [1, 0]},
                    {"coefficient": -1.0, "powers": [2, 1]},
                ]},
            ],
        }
    )
    observed_iterations: list[int] = []
    common = dict(
        h=0.01,
        order=4,
        tau_index=2,
        target_remainder_radius=1e-4,
        cutoff_threshold=1e-10,
        max_validation_attempts=2,
        validation_eps=1e-12,
        validation_mode="flowstar_raw_remainder_compat",
    )
    observed = dense_picard_validate_step(
        ode,
        base,
        polynomial_observer=lambda iteration, _before, _after: observed_iterations.append(iteration),
        **common,
    )
    baseline = dense_picard_validate_step(ode, base, **common)
    assert observed_iterations == [1, 2, 3, 4]
    assert observed.status == baseline.status
    assert observed.trace == baseline.trace
    assert torch.equal(observed.segment_tm.poly.coeffs, baseline.segment_tm.poly.coeffs)
    assert torch.equal(observed.segment_tm.rem_lo, baseline.segment_tm.rem_lo)
    assert torch.equal(observed.segment_tm.rem_hi, baseline.segment_tm.rem_hi)
    assert torch.equal(observed.subset_margin, baseline.subset_margin)
