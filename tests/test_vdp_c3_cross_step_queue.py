from __future__ import annotations

from dataclasses import replace
from fractions import Fraction

import pytest
import torch

from torch_tm_flowpipe import (
    C3_CROSS_STEP_SYMBOLIC_QUEUE,
    FlowstarNormalFlowpipeState,
    Interval,
    NORMALIZED_INSERTION_DEPENDENCY_PRESERVING,
    flowpipe_step_flowstar_style_adaptive,
    load_terminal_checkpoint,
    save_terminal_checkpoint,
    tmvector_hashes,
)
from torch_tm_flowpipe.ode_examples import van_der_pol_ode
from torch_tm_flowpipe.symbolic_remainder import (
    FlowstarSymbolicRemainderQueue,
    c3_symbolic_queue_commit,
    c3_symbolic_queue_propagate,
    c3_symbolic_queue_sha256,
    validate_c3_symbolic_queue,
)
from torch_tm_flowpipe.terminal_checkpoint import SCHEMA_V5


def _c3_step(current, state, *, target: float = 1e-4):
    return flowpipe_step_flowstar_style_adaptive(
        van_der_pol_ode,
        current,
        h=0.01,
        h_min=0.01,
        h_max=0.01,
        order=4,
        target_remainder_radius=target,
        cutoff_threshold=1e-10,
        reset_mode=C3_CROSS_STEP_SYMBOLIC_QUEUE,
        flowstar_symbolic_queue_max_size=100,
        flowstar_normal_state=state,
    )


def _initial():
    state = FlowstarNormalFlowpipeState.from_exact_decimal_box(
        [("1.1", "1.4"), ("2.35", "2.45")],
        4,
    )
    return state.normalized_initial_tm(4), state


def _accept_once(
    state: FlowstarSymbolicRemainderQueue,
    current_j: tuple[Interval, ...],
    *,
    boundary: int,
    scales: tuple[float, ...] = (1.0, 1.0),
):
    reference = current_j[0]
    updated, updated_iv, propagated, _stats = c3_symbolic_queue_propagate(
        state,
        ((1.0, 0.0), (0.0, 1.0)),
        expected_boundary_index=boundary - 1,
        reference=reference,
    )
    assert all(value.contains(0.0) for value in propagated)
    return c3_symbolic_queue_commit(
        state,
        updated,
        updated_iv,
        current_j,
        scales=scales,
        accepted_boundary_index=boundary,
        reference=reference,
    )[0]


def test_c3_exact_fraction_operator_oracle_contains_linear_history_image():
    reference = Interval(-0.125, 0.25)
    state = FlowstarSymbolicRemainderQueue.empty_c3(2, 100, reference=reference)
    j0 = (Interval(-0.125, 0.25), Interval(-0.0625, 0.125))
    state = _accept_once(
        state,
        j0,
        boundary=1,
        scales=(2.0, 4.0),
    )
    _point, _interval, propagated, _stats = c3_symbolic_queue_propagate(
        state,
        ((2.0, 3.0), (5.0, 7.0)),
        expected_boundary_index=1,
        reference=reference,
    )
    phi = (
        (Fraction(2) * Fraction(1, 2), Fraction(3) * Fraction(1, 4)),
        (Fraction(5) * Fraction(1, 2), Fraction(7) * Fraction(1, 4)),
    )
    j_fraction = (
        (Fraction(-1, 8), Fraction(1, 4)),
        (Fraction(-1, 16), Fraction(1, 8)),
    )
    for row, actual in zip(phi, propagated):
        exact_lo = sum(weight * bounds[0] for weight, bounds in zip(row, j_fraction))
        exact_hi = sum(weight * bounds[1] for weight, bounds in zip(row, j_fraction))
        assert Fraction.from_float(float(actual.lo)) <= exact_lo
        assert Fraction.from_float(float(actual.hi)) >= exact_hi


def test_c3_stale_owner_generation_and_partial_update_fail_closed():
    state = FlowstarSymbolicRemainderQueue.empty_c3(2, 100, reference=Interval.zero())
    state = _accept_once(
        state,
        (Interval(-0.1, 0.1), Interval(-0.2, 0.2)),
        boundary=1,
    )
    with pytest.raises(ValueError, match="stale generation"):
        validate_c3_symbolic_queue(replace(state, generation=0))
    with pytest.raises(ValueError, match="stale accepted-boundary owner"):
        validate_c3_symbolic_queue(state, expected_boundary_index=0)
    with pytest.raises(ValueError, match="partial update"):
        validate_c3_symbolic_queue(replace(state, Phi_L=()))


def test_c3_retry_shadow_is_immutable_and_reset_is_atomic():
    reference = Interval(-0.1, 0.1)
    state = FlowstarSymbolicRemainderQueue.empty_c3(2, 2, reference=reference)
    state = _accept_once(
        state,
        (reference, Interval(-0.2, 0.2)),
        boundary=1,
    )
    before = c3_symbolic_queue_sha256(state)
    c3_symbolic_queue_propagate(
        state,
        ((0.5, 0.25), (-0.125, 0.75)),
        expected_boundary_index=1,
        reference=reference,
    )
    assert c3_symbolic_queue_sha256(state) == before
    reset = _accept_once(
        state,
        (Interval(-0.01, 0.01), Interval(-0.02, 0.02)),
        boundary=2,
    )
    assert reset.generation == 2
    assert reset.accepted_boundary_index == 2
    assert reset.reset_count == 1
    assert reset.J == reset.Phi_L == reset.Phi_L_iv == ()


def test_c3_rejected_step_rolls_back_the_accepted_queue_bit_exactly():
    current, normal_state = _initial()
    accepted = _c3_step(current, normal_state)
    assert accepted.status == "validated"
    queue = accepted.flowstar_normal_state.symbolic_queue
    before = c3_symbolic_queue_sha256(queue)
    rejected = flowpipe_step_flowstar_style_adaptive(
        van_der_pol_ode,
        accepted.reset_tm,
        h=0.01,
        h_min=0.01,
        h_max=0.01,
        order=4,
        target_remainder_radius=1e-30,
        cutoff_threshold=1e-10,
        max_validation_attempts=1,
        reset_mode=C3_CROSS_STEP_SYMBOLIC_QUEUE,
        flowstar_symbolic_queue_max_size=100,
        flowstar_normal_state=accepted.flowstar_normal_state,
    )
    assert rejected.status == "failed"
    assert c3_symbolic_queue_sha256(queue) == before


def test_c3_subnormal_owner_is_not_flushed_to_zero():
    tiny = torch.nextafter(
        torch.tensor(0.0, dtype=torch.float64),
        torch.tensor(torch.inf, dtype=torch.float64),
    )
    reference = Interval(tiny, tiny)
    state = FlowstarSymbolicRemainderQueue.empty_c3(2, 100, reference=reference)
    state = _accept_once(
        state,
        (Interval(tiny, tiny), Interval(-tiny, tiny)),
        boundary=1,
    )
    _point, _interval, propagated, _stats = c3_symbolic_queue_propagate(
        state,
        ((1.0, 0.0), (0.0, 1.0)),
        expected_boundary_index=1,
        reference=reference,
    )
    assert propagated[0].contains(tiny)
    assert float(propagated[0].hi) > 0.0


def test_c3_checkpoint_resume_is_bit_exact(tmp_path):
    current, state = _initial()
    first = _c3_step(current, state)
    assert first.status == "validated"
    second = _c3_step(first.reset_tm, first.flowstar_normal_state)
    assert second.status == "validated"
    uninterrupted = _c3_step(second.reset_tm, second.flowstar_normal_state)
    assert uninterrupted.status == "validated"

    checkpoint = tmp_path / "c3_checkpoint"
    manifest = save_terminal_checkpoint(
        checkpoint,
        current=second.reset_tm,
        normal_state=second.flowstar_normal_state,
        scheduler={"current_time": 0.02, "h_next": 0.01},
        contract={"order": 4, "cutoff": 1e-10, "queue": 100},
        provenance={"test": True},
    )
    assert manifest["schema"] == SCHEMA_V5
    loaded = load_terminal_checkpoint(
        checkpoint,
        expected_order=4,
        expected_dtype="float64",
    )
    resumed = _c3_step(loaded.current, loaded.normal_state)
    assert resumed.status == "validated"
    assert tmvector_hashes(uninterrupted.reset_tm) == tmvector_hashes(resumed.reset_tm)
    assert tmvector_hashes(uninterrupted.flowstar_normal_state.tmv_right) == tmvector_hashes(
        resumed.flowstar_normal_state.tmv_right
    )
    assert c3_symbolic_queue_sha256(uninterrupted.flowstar_normal_state.symbolic_queue) == c3_symbolic_queue_sha256(
        resumed.flowstar_normal_state.symbolic_queue
    )


def test_c3_off_preserves_frozen_c2_two_step_binary64_hashes():
    current, state = _initial()
    expected = (
        (
            "73571d915d6465897c4d820ce27e37107b0cd533a81942c8ac263adba6937c19",
            "ff55d684753c3b8144e270ba6f573358f8b1bc4bbe14dcbbf1cb57b84f371ef4",
        ),
        (
            "3c6b955985ea3b0610c32467626abc59e5c69c99a269ac482a7ded5c68a5c47a",
            "6a7e4190f2a6f269c8e7921333866d6a1ddefd424369afdcf675b3ef11f3a957",
        ),
    )
    for expected_pair in expected:
        segment = flowpipe_step_flowstar_style_adaptive(
            van_der_pol_ode,
            current,
            h=0.01,
            h_min=0.01,
            h_max=0.01,
            order=4,
            target_remainder_radius=1e-4,
            cutoff_threshold=1e-10,
            reset_mode=NORMALIZED_INSERTION_DEPENDENCY_PRESERVING,
            flowstar_normal_state=state,
        )
        assert segment.status == "validated"
        assert segment.flowstar_normal_state.symbolic_queue is None
        assert tmvector_hashes(segment.reset_tm)["tmvector_sha256"] == expected_pair[0]
        assert tmvector_hashes(segment.flowstar_normal_state.tmv_right)["tmvector_sha256"] == expected_pair[1]
        current = segment.reset_tm
        state = segment.flowstar_normal_state
