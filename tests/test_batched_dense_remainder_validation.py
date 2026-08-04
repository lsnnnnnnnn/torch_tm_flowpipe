import torch

from torch_tm_flowpipe import Interval, Polynomial, TaylorModel, TMVector
from torch_tm_flowpipe.batched_dense_tm import (
    BatchedTaylorModel,
    dense_picard_validate_step,
    sparse_tmvector_to_dense,
)


def _linear_base(h, order):
    domain = [Interval(-1.0, 1.0)]
    x0 = TMVector([TaylorModel(Polynomial({(1,): 1.0}, 1), Interval.zero(), domain, order=order)])
    return sparse_tmvector_to_dense(x0.extend_domain(Interval(0.0, h)), order=order)


def test_static_truncation_floor_remains_visible_during_refinement():
    base = _linear_base(0.05, order=1)

    def square_rhs(state):
        return state.mul_trunc(state)

    result = dense_picard_validate_step(
        square_rhs,
        base,
        h=0.05,
        order=1,
        tau_index=1,
        target_remainder_radius=0.2,
        cutoff_threshold=None,
        max_validation_attempts=5,
        validation_mode="target_remainder_refined",
    )
    rows = [row for row in result.trace if row["phase"] == "remainder_validation"]
    assert rows
    for row in rows:
        widths = row["remainder_ledger_widths"]
        static_width = sum(sum(component) for component in widths.get("polynomial_truncation", []))
        integration_width = sum(sum(component) for component in widths.get("integration_overflow", []))
        assert static_width > 0.0 or integration_width > 0.0


def test_nonfinite_rhs_fails_closed_without_endpoint():
    base = _linear_base(0.01, order=2)

    def nan_rhs(state):
        return BatchedTaylorModel.constants_like(float("nan"), state)

    result = dense_picard_validate_step(
        nan_rhs,
        base,
        h=0.01,
        order=2,
        tau_index=1,
        target_remainder_radius=1e-4,
        cutoff_threshold=1e-10,
    )
    assert result.status == "nonfinite"
    assert result.raw_endpoint is None


def test_rejected_dense_flowpipe_step_publishes_no_completed_endpoint():
    from torch_tm_flowpipe import flowpipe_step

    def square_ode(x, u=None):
        return TMVector([x[0] * x[0]])

    segment = flowpipe_step(
        square_ode,
        [Interval(-10.0, 10.0)],
        h=0.1,
        order=1,
        tm_backend="dense",
        validation_mode="target_remainder",
        target_remainder_radius=1e-8,
        cutoff_threshold=1e-10,
        max_validation_attempts=1,
    )
    assert segment.status == "failed"
    assert segment.endpoint_raw_tm is None
    assert segment.endpoint_tightened_tm is None
    assert segment.endpoint_semantics == "unpublished_rejected_step"


def test_one_rejected_batch_leaf_rejects_the_whole_dense_step():
    single = _linear_base(0.05, order=1)
    coeffs = single.poly.coeffs.repeat(2, 1, 1)
    coeffs[0] *= 0.01
    coeffs[1] *= 100.0
    base = BatchedTaylorModel(
        type(single.poly)(coeffs, single.poly.basis),
        single.rem_lo.repeat(2, 1),
        single.rem_hi.repeat(2, 1),
        single.domain_lo.repeat(2, 1),
        single.domain_hi.repeat(2, 1),
    )

    result = dense_picard_validate_step(
        lambda state: state.mul_trunc(state),
        base,
        h=0.05,
        order=1,
        tau_index=1,
        target_remainder_radius=0.2,
        cutoff_threshold=None,
    )

    assert not result.accepted
    assert result.raw_endpoint is None
    assert float(result.subset_margin[0, 0]) >= 0.0
    assert float(result.subset_margin[1, 0]) < 0.0
