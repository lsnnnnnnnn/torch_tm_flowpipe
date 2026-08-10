from __future__ import annotations

import torch

from torch_tm_flowpipe import (
    Interval,
    flowpipe_step_flowstar_style_adaptive,
    tmvector_hashes,
    verify_structured_publication,
)
from torch_tm_flowpipe.fixed_support_outward import OutwardIntervalTensor
from torch_tm_flowpipe.ode_examples import van_der_pol_ode
from torch_tm_flowpipe.structured_remainder import (
    StructuredRemainderState,
    materialize_structured_remainder,
    normal_interval_to_physical,
    structured_column_contributions,
)


def _step(current, state, h=0.005):
    return flowpipe_step_flowstar_style_adaptive(
        van_der_pol_ode,
        current,
        h=h,
        h_min=h,
        h_max=h,
        order=4,
        target_remainder_radius=1e-4,
        cutoff_threshold=1e-10,
        max_validation_attempts=2,
        validation_mode="flowstar_raw_remainder_compat",
        reset_mode="normalized_insertion_structured_remainder_k16",
        step_policy_mode="flowstar_compat",
        tm_backend="dense",
        flowstar_normal_state=state,
    )


def _assert_state_equal(left: StructuredRemainderState, right: StructuredRemainderState):
    for name in left.__dataclass_fields__:
        lhs = getattr(left, name)
        rhs = getattr(right, name)
        if isinstance(lhs, torch.Tensor):
            assert torch.equal(lhs, rhs), name
        else:
            assert lhs == rhs, name


def test_vdp_boundaries_1_2_and_9_publish_complete_structured_totals():
    current = [Interval(1.1, 1.4), Interval(2.35, 2.45)]
    normal_state = None
    observations = {}
    for boundary in range(1, 10):
        segment = _step(current, normal_state)
        assert segment.status == "validated", segment.message
        assert segment.reset_tm is not None
        assert segment.flowstar_normal_state is not None
        state = segment.flowstar_normal_state.structured_remainder_state
        assert isinstance(state, StructuredRemainderState)
        assert state.accepted_boundary_index == boundary
        assert segment.structured_state_after is state
        assert segment.structured_boundary_result.accepted.tolist() == [True]
        assert segment.endpoint_publication_mask.tolist() == [True]
        assert segment.tube_publication_mask.tolist() == [True]
        assert segment.flowstar_normal_stats["structured_total_self_map_containment"]
        assert segment.flowstar_normal_stats["structured_endpoint_in_tube"]
        assert segment.flowstar_normal_stats["structured_raw_picard_target_changed"] is False

        identities = [
            (
                int(state.source_boundary_index[0, slot]),
                int(state.source_id[0, slot]),
                int(state.source_occurrence_index[0, slot]),
            )
            for slot in range(state.capacity)
            if bool(state.active[0, slot])
        ]
        assert len(identities) == len(set(identities))
        for event in segment.structured_boundary_result.source_events:
            for active, category in zip(event.active_mask.tolist(), event.source_category):
                assert bool(category) == bool(active)

        scale = torch.tensor(
            [segment.flowstar_normal_state.scales], dtype=torch.float64
        )
        materialized = materialize_structured_remainder(state)
        endpoint_physical = normal_interval_to_physical(
            materialized.lo,
            materialized.hi,
            forward_scale=scale,
        )
        assert torch.all(segment.endpoint_total_remainder.lo <= endpoint_physical.lo)
        assert torch.all(segment.endpoint_total_remainder.hi >= endpoint_physical.hi)
        if boundary in {1, 2, 9}:
            observations[boundary] = segment
        current = segment.reset_tm
        normal_state = segment.flowstar_normal_state

    assert set(observations) == {1, 2, 9}
    assert observations[9].structured_state_after.active.sum().item() == 9

    # Removing one live endpoint column from the published interval must fail.
    segment = observations[9]
    state = segment.structured_state_after
    columns = structured_column_contributions(state)
    active_slots = torch.nonzero(state.active[0], as_tuple=False).flatten()
    omitted = int(active_slots[0])
    keep = state.active.clone()
    keep[0, omitted] = False
    mask = keep[..., None]
    remaining = OutwardIntervalTensor(
        torch.where(mask, columns.lo, torch.zeros_like(columns.lo)),
        torch.where(mask, columns.hi, torch.zeros_like(columns.hi)),
    ).sum(dim=1)
    scale = torch.tensor([segment.flowstar_normal_state.scales], dtype=torch.float64)
    remaining_physical = normal_interval_to_physical(
        remaining.lo, remaining.hi, forward_scale=scale
    )
    incomplete = segment.endpoint_ordinary_remainder.add(remaining_physical)
    gate = verify_structured_publication(
        segment.endpoint_ordinary_remainder.lo,
        segment.endpoint_ordinary_remainder.hi,
        segment.endpoint_total_structured_remainder.lo,
        segment.endpoint_total_structured_remainder.hi,
        incomplete.lo,
        incomplete.hi,
    )
    assert gate.tolist() == [False]


def test_rejected_attempt_preserves_exact_prestate_and_retry_lineage():
    current = [Interval(1.1, 1.4), Interval(2.35, 2.45)]
    normal_state = None
    for _ in range(2):
        accepted = _step(current, normal_state)
        assert accepted.status == "validated"
        current = accepted.reset_tm
        normal_state = accepted.flowstar_normal_state
    assert current is not None and normal_state is not None
    prestate = normal_state.structured_remainder_state
    assert isinstance(prestate, StructuredRemainderState)
    snapshot = StructuredRemainderState(
        **{
            name: value.clone() if isinstance(value, torch.Tensor) else value
            for name, value in prestate.__dict__.items()
        }
    )

    rejected = _step(current, normal_state, h=0.1)
    assert rejected.status == "failed"
    _assert_state_equal(prestate, snapshot)
    assert normal_state.structured_remainder_state is prestate

    retry_a = _step(current, normal_state)
    retry_b = _step(current, normal_state)
    assert retry_a.status == retry_b.status == "validated"
    _assert_state_equal(
        retry_a.flowstar_normal_state.structured_remainder_state,
        retry_b.flowstar_normal_state.structured_remainder_state,
    )
    assert tmvector_hashes(retry_a.reset_tm) == tmvector_hashes(retry_b.reset_tm)
