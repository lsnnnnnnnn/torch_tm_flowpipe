from __future__ import annotations

import pytest

from experiments.run_brusselator_sr1000_parity import (
    INITIAL_DECIMAL,
    ORDER,
    _policy,
    _step,
)
from torch_tm_flowpipe import (
    DENSE_OBSERVER_FULL,
    DENSE_OBSERVER_LIGHTWEIGHT,
    DENSE_OBSERVER_NONE,
    FlowstarNormalFlowpipeState,
    accepted_boundary_sr_queue_sha256,
    tmvector_hashes,
)


def _snapshot(observer_mode: str):
    state = FlowstarNormalFlowpipeState.from_exact_decimal_box(INITIAL_DECIMAL, ORDER)
    current = state.normalized_initial_tm(ORDER)
    segment, diagnostics = _step(
        current,
        state,
        1,
        _policy(),
        validation_mode="flowstar_raw_remainder_compat_refined",
        lane_label=observer_mode,
        observer_mode=observer_mode,
    )
    assert segment.status == "validated"
    assert segment.endpoint_raw_tm is not None
    assert segment.reset_tm is not None
    assert segment.flowstar_normal_state is not None
    assert segment.flowstar_normal_state.symbolic_queue is not None
    refinements = [
        row for row in segment.backend_trace
        if row.get("phase") == "post_accept_refinement"
    ]
    counters = dict(segment.backend_counters or {})
    return {
        "endpoint": tmvector_hashes(segment.endpoint_raw_tm),
        "tube": tmvector_hashes(segment.tm),
        "reset": tmvector_hashes(segment.reset_tm),
        "queue": accepted_boundary_sr_queue_sha256(
            segment.flowstar_normal_state.symbolic_queue
        ),
        "center": tuple(segment.flowstar_normal_state.center),
        "scales": tuple(segment.flowstar_normal_state.scales),
        "candidate_remainder": segment.candidate_remainder,
        "picard_image_remainder": segment.picard_image_remainder,
        "refinement_count": counters["post_accept_replay_calls"],
        "committed_refinement_count": counters["post_accept_committed_replays"],
        "stop_ratio_count": counters["post_accept_stop_ratio_count"],
        "refinement_stop": refinements[-1].get("stop_reason") if refinements else None,
        "backend_trace_count": len(segment.backend_trace),
        "diagnostic_count": len(diagnostics),
    }


@pytest.mark.regression
def test_production_counter_and_evidence_observers_are_numerically_identical() -> None:
    production = _snapshot(DENSE_OBSERVER_NONE)
    counters = _snapshot(DENSE_OBSERVER_LIGHTWEIGHT)
    evidence = _snapshot(DENSE_OBSERVER_FULL)
    scientific = (
        "endpoint",
        "tube",
        "reset",
        "queue",
        "center",
        "scales",
        "candidate_remainder",
        "picard_image_remainder",
    )
    assert {key: production[key] for key in scientific} == {
        key: counters[key] for key in scientific
    } == {key: evidence[key] for key in scientific}
    assert production["backend_trace_count"] == 0
    assert production["diagnostic_count"] == 0
    assert counters["refinement_count"] == evidence["refinement_count"] == 8
    assert production["refinement_count"] == 8
    assert production["committed_refinement_count"] == 8
    assert production["stop_ratio_count"] == 1
    assert counters["refinement_stop"] == evidence["refinement_stop"] == "stop_ratio"
    assert counters["backend_trace_count"] < evidence["backend_trace_count"]


@pytest.mark.unit
def test_unknown_observer_mode_fails_closed() -> None:
    state = FlowstarNormalFlowpipeState.from_exact_decimal_box(INITIAL_DECIMAL, ORDER)
    segment, _ = _step(
        state.normalized_initial_tm(ORDER),
        state,
        1,
        _policy(),
        validation_mode="flowstar_raw_remainder_compat_refined",
        lane_label="invalid",
        observer_mode="silent_but_unregistered",
    )
    assert segment.status == "failed"
    assert "observer_mode" in segment.message
