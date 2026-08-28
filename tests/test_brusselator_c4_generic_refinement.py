from __future__ import annotations

import pytest
import torch

import torch_tm_flowpipe.batched_dense_tm as dense_module
from torch_tm_flowpipe import FlowstarNormalFlowpipeState, Interval
from torch_tm_flowpipe.batched_dense_tm import (
    DenseRangePolicy,
    FLOWSTAR_RAW_REMAINDER_GENERIC_REFINED_MODE,
    dense_picard_validate_step,
    sparse_tmvector_to_dense,
)
from torch_tm_flowpipe.ode_examples import brusselator_ode


LEGACY = "flowstar_raw_remainder_compat"
C4 = FLOWSTAR_RAW_REMAINDER_GENERIC_REFINED_MODE
COMMON = {
    "h": 0.02,
    "order": 6,
    "tau_index": 2,
    "target_remainder_radius": 1.0e-4,
    "cutoff_threshold": 1.0e-10,
    "max_validation_attempts": 2,
    "validation_eps": 1.0e-12,
}
FLOWSTAR_ACCEPTED_REMAINDER = (
    (-1.0256262340411214e-08, 8.3496722364602686e-09),
    (-8.5174143523546871e-09, 1.046213625838395e-08),
)


def _base():
    state = FlowstarNormalFlowpipeState.from_exact_decimal_box(
        (("1.48", "1.52"), ("2.98", "3.02")),
        6,
    )
    sparse = state.normalized_initial_tm(6).extend_domain(Interval(0.0, 0.02))
    policy = DenseRangePolicy(
        method="adaptive_subdivision",
        max_depth=1,
        max_leaves=4,
        split_vars=(0, 1),
        trigger="proactive_depth1_on_named_contexts",
        named_contexts=("polynomial_truncation",),
        variable_orders=((0, 1, 2), (1, 0, 2), (2, 0, 1)),
    )
    return sparse_tmvector_to_dense(
        sparse,
        order=6,
        device="cpu",
        dtype=torch.float64,
        range_policy=policy,
        range_trace=[],
    )


def _validation(step):
    return next(row for row in step.trace if row.get("phase") == "remainder_validation")


def _refinements(step):
    return [row for row in step.trace if row.get("phase") == "post_accept_refinement"]


@pytest.mark.unit
def test_generic_c4_first_acceptance_is_legacy_identical_then_refines_atomically() -> None:
    base = _base()
    legacy = dense_picard_validate_step(
        brusselator_ode, base, validation_mode=LEGACY, **COMMON
    )
    refined = dense_picard_validate_step(
        brusselator_ode, base, validation_mode=C4, **COMMON
    )
    assert legacy.status == refined.status == "validated"
    assert torch.equal(legacy.segment_tm.poly.coeffs, refined.segment_tm.poly.coeffs)
    legacy_first = _validation(legacy)
    refined_first = _validation(refined)
    for key in (
        "validation_status",
        "finite",
        "subset_result",
        "target_subset_result",
        "candidate_remainder_lo",
        "candidate_remainder_hi",
        "picard_image_remainder_lo",
        "picard_image_remainder_hi",
        "subset_margin",
        "raw_rhs_remainder_lo",
        "raw_rhs_remainder_hi",
        "poly_diff_range_lo",
        "poly_diff_range_hi",
    ):
        assert refined_first[key] == legacy_first[key]

    rows = _refinements(refined)
    assert len(rows) == 8
    assert all(row["committed"] for row in rows)
    assert rows[-1]["stop_reason"] == "stop_ratio"
    assert {row["generic_raw_rhs_evaluation"] for row in rows} == {"ordered_terms"}
    assert all(
        component["subset"]
        for row in rows
        for component in row["components"]
    )
    for previous, current in zip(rows, rows[1:]):
        assert current["input_remainder_lo"] == previous["retained_remainder_lo"]
        assert current["input_remainder_hi"] == previous["retained_remainder_hi"]
    assert all(row["validated_remainder_decomposition_contains_image"] for row in rows)


@pytest.mark.unit
def test_generic_c4_brusselator_binary64_snapshot_and_stock_direction() -> None:
    base = _base()
    legacy = dense_picard_validate_step(
        brusselator_ode, base, validation_mode=LEGACY, **COMMON
    )
    refined = dense_picard_validate_step(
        brusselator_ode, base, validation_mode=C4, **COMMON
    )
    assert [float(value).hex() for value in refined.segment_tm.rem_lo[0]] == [
        "-0x1.58aa609209c7bp-27",
        "-0x1.d4748bbc7cd95p-28",
    ]
    assert [float(value).hex() for value in refined.segment_tm.rem_hi[0]] == [
        "0x1.bcb368de134ecp-28",
        "0x1.6c320c0104a5fp-27",
    ]
    stock_lo = torch.tensor(
        [[pair[0] for pair in FLOWSTAR_ACCEPTED_REMAINDER]], dtype=torch.float64
    )
    stock_hi = torch.tensor(
        [[pair[1] for pair in FLOWSTAR_ACCEPTED_REMAINDER]], dtype=torch.float64
    )
    legacy_error = torch.sum(torch.abs(legacy.segment_tm.rem_lo - stock_lo)) + torch.sum(
        torch.abs(legacy.segment_tm.rem_hi - stock_hi)
    )
    refined_error = torch.sum(torch.abs(refined.segment_tm.rem_lo - stock_lo)) + torch.sum(
        torch.abs(refined.segment_tm.rem_hi - stock_hi)
    )
    assert float(refined_error) < 0.25 * float(legacy_error)
    assert bool(torch.all(refined.validated_remainder_decomposition.contains_image))


@pytest.mark.unit
def test_generic_c4_rejects_non_ordered_raw_override() -> None:
    with pytest.raises(ValueError, match="ordered-terms"):
        dense_picard_validate_step(
            brusselator_ode,
            _base(),
            validation_mode=C4,
            raw_rhs_evaluation_override="canonical_factorized_joint",
            **COMMON,
        )


@pytest.mark.unit
def test_generic_c4_failed_first_raw_self_map_is_not_rescued() -> None:
    result = dense_picard_validate_step(
        brusselator_ode,
        _base(),
        validation_mode=C4,
        **{**COMMON, "target_remainder_radius": 1.0e-5},
    )
    assert result.status == "failed"
    assert _validation(result)["subset_result"] is False
    assert _refinements(result) == []


@pytest.mark.unit
def test_generic_c4_refinement_exception_fails_closed_at_last_certified_vector(monkeypatch) -> None:
    baseline = dense_picard_validate_step(
        brusselator_ode, _base(), validation_mode=LEGACY, **COMMON
    )
    original = dense_module._dense_flowstar_raw_compat_image
    calls = 0

    def injected(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected refinement-only evaluation failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(dense_module, "_dense_flowstar_raw_compat_image", injected)
    result = dense_picard_validate_step(
        brusselator_ode, _base(), validation_mode=C4, **COMMON
    )
    assert result.status == "validated"
    assert torch.equal(result.segment_tm.poly.coeffs, baseline.segment_tm.poly.coeffs)
    assert torch.equal(result.segment_tm.rem_lo, baseline.segment_tm.rem_lo)
    assert torch.equal(result.segment_tm.rem_hi, baseline.segment_tm.rem_hi)
    rows = _refinements(result)
    assert len(rows) == 1
    assert rows[0]["committed"] is False
    assert rows[0]["stop_reason"] == "evaluation_failed_closed"
