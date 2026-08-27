from __future__ import annotations

from dataclasses import replace
from fractions import Fraction

import pytest
import torch

from torch_tm_flowpipe import (
    FlowstarNormalFlowpipeState,
    GENERIC_ACCEPTED_BOUNDARY_SYMBOLIC_REMAINDER,
    Interval,
    Polynomial,
    TMVector,
    TaylorModel,
    accepted_boundary_sr_queue_commit,
    accepted_boundary_sr_queue_propagate,
    accepted_boundary_sr_queue_sha256,
    commit_accepted_boundary_sr,
    flowpipe_step_flowstar_style_adaptive,
    load_terminal_checkpoint,
    prepare_accepted_boundary_sr,
    save_terminal_checkpoint,
    validate_accepted_boundary_sr_queue,
)
from torch_tm_flowpipe.ode_examples import scalar_quadratic_ode
from torch_tm_flowpipe.symbolic_remainder import FlowstarSymbolicRemainderQueue


MATRICES = {
    1: ((-1.5,),),
    2: ((0.5, -0.75), (1.25, 0.125)),
    3: (
        (0.5, -0.75, 0.125),
        (1.25, 0.125, -0.5),
        (-0.25, 1.5, 0.75),
    ),
}


def _identity(dim: int) -> tuple[tuple[float, ...], ...]:
    return tuple(
        tuple(1.0 if row == column else 0.0 for column in range(dim))
        for row in range(dim)
    )


def _fraction_interval_sum(
    weights: tuple[Fraction, ...],
    columns: tuple[tuple[Fraction, Fraction], ...],
) -> tuple[Fraction, Fraction]:
    lo = Fraction(0)
    hi = Fraction(0)
    for weight, (column_lo, column_hi) in zip(weights, columns):
        products = (weight * column_lo, weight * column_hi)
        lo += min(products)
        hi += max(products)
    return lo, hi


def _linear_endpoint_map(matrix: tuple[tuple[float, ...], ...]) -> TMVector:
    dim = len(matrix)
    domain = [Interval(-1.0, 1.0) for _ in range(dim)]
    models: list[TaylorModel] = []
    for row in matrix:
        terms = {}
        for variable, coefficient in enumerate(row):
            if coefficient:
                exponent = tuple(1 if index == variable else 0 for index in range(dim))
                terms[exponent] = torch.tensor(coefficient, dtype=torch.float64)
        models.append(
            TaylorModel(
                Polynomial(terms, dim),
                Interval.zero(),
                domain,
                order=2,
            )
        )
    return TMVector(models)


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_generic_accepted_boundary_operator_contains_exact_fraction_image(dim: int) -> None:
    owners = tuple(
        Interval(-1.0 / (2 ** (index + 2)), 1.0 / (2 ** (index + 1)))
        for index in range(dim)
    )
    scales = tuple(float(2 ** (index + 1)) for index in range(dim))
    state = FlowstarSymbolicRemainderQueue.empty_accepted_boundary_sr(
        dim,
        8,
        reference=owners[0],
    )
    updated, updated_iv, propagated, _ = accepted_boundary_sr_queue_propagate(
        state,
        _identity(dim),
        expected_boundary_index=0,
        reference=owners[0],
    )
    assert all(value.contains(0.0) for value in propagated)
    state, _ = accepted_boundary_sr_queue_commit(
        state,
        updated,
        updated_iv,
        owners,
        scales=scales,
        accepted_boundary_index=1,
        reference=owners[0],
    )
    validate_accepted_boundary_sr_queue(state, expected_boundary_index=1)

    endpoint = _linear_endpoint_map(MATRICES[dim])
    right_map = TMVector.identity(endpoint.domain, order=2)
    prepared = prepare_accepted_boundary_sr(
        endpoint,
        right_map,
        domain=endpoint.domain,
        order=2,
        cutoff_threshold=None,
        queue_state=state,
        queue_capacity=8,
        previous_accepted_boundary_index=1,
        compose=_clone_composition,
        diagnostics={},
    )
    actual = prepared.propagated_history
    exact_columns = tuple(
        (
            Fraction(-1, 2 ** (index + 2)),
            Fraction(1, 2 ** (index + 1)),
        )
        for index in range(dim)
    )
    for row_index, interval in enumerate(actual):
        exact_weights = tuple(
            Fraction.from_float(MATRICES[dim][row_index][column_index])
            / Fraction.from_float(scales[column_index])
            for column_index in range(dim)
        )
        exact_lo, exact_hi = _fraction_interval_sum(exact_weights, exact_columns)
        assert Fraction.from_float(float(interval.lo)) <= exact_lo
        assert Fraction.from_float(float(interval.hi)) >= exact_hi
    committed = commit_accepted_boundary_sr(
        prepared,
        normalization_scales=tuple(1.0 for _ in range(dim)),
        cutoff_threshold=None,
    )
    assert committed.queue_after.owner_generations == (1, 2)


def _clone_composition(
    outer: TMVector,
    _inner: TMVector,
    _order: int,
    _cutoff: float | None,
    _domain: list[Interval] | tuple[Interval, ...],
    _diagnostics: dict[str, object],
) -> TMVector:
    return TMVector(model.clone() for model in outer)


def _one_dimensional_linear_map(coefficient: float, remainder: Interval) -> TMVector:
    domain = [Interval(-1.0, 1.0)]
    polynomial = Polynomial({(1,): torch.tensor(coefficient, dtype=torch.float64)}, 1)
    return TMVector([TaylorModel(polynomial, remainder, domain, order=2)])


def test_capacity_reset_materializes_history_once_then_reanchors_without_duplicate_owner() -> None:
    reference = Interval(-0.125, 0.25)
    state = FlowstarSymbolicRemainderQueue.empty_accepted_boundary_sr(
        1,
        2,
        reference=reference,
    )
    updated, updated_iv, _, _ = accepted_boundary_sr_queue_propagate(
        state,
        ((1.0,),),
        expected_boundary_index=0,
        reference=reference,
    )
    state, _ = accepted_boundary_sr_queue_commit(
        state,
        updated,
        updated_iv,
        (reference,),
        scales=(1.0,),
        accepted_boundary_index=1,
        reference=reference,
    )

    right_map = TMVector.identity([Interval(-1.0, 1.0)], order=2)
    prepared = prepare_accepted_boundary_sr(
        _one_dimensional_linear_map(2.0, Interval.zero()),
        right_map,
        domain=right_map.domain,
        order=2,
        cutoff_threshold=None,
        queue_state=state,
        queue_capacity=2,
        previous_accepted_boundary_index=1,
        compose=_clone_composition,
        diagnostics={},
    )
    expected_once = (Fraction(-1, 4), Fraction(1, 2))
    materialized = prepared.inserted[0].remainder
    assert Fraction.from_float(float(materialized.lo)) <= expected_once[0]
    assert Fraction.from_float(float(materialized.hi)) >= expected_once[1]
    assert prepared.current_owner[0].contains(0.0)

    committed = commit_accepted_boundary_sr(
        prepared,
        normalization_scales=(1.0,),
        cutoff_threshold=None,
    )
    assert committed.queue_after.reset_count == 1
    assert committed.queue_after.J == ()

    reanchored = prepare_accepted_boundary_sr(
        prepared.inserted,
        right_map,
        domain=right_map.domain,
        order=2,
        cutoff_threshold=None,
        queue_state=committed.queue_after,
        queue_capacity=2,
        previous_accepted_boundary_index=2,
        compose=_clone_composition,
        diagnostics={},
    )
    assert reanchored.composition_branch == "full_reanchor"
    assert reanchored.current_owner[0].lo.equal(materialized.lo)
    assert reanchored.current_owner[0].hi.equal(materialized.hi)
    assert reanchored.propagated_history[0].contains(0.0)
    assert float(reanchored.propagated_history[0].width()) < 1e-300

    recommitted = commit_accepted_boundary_sr(
        reanchored,
        normalization_scales=(1.0,),
        cutoff_threshold=None,
    )
    assert recommitted.queue_after.owner_generations == (3,)
    _, _, paid_once, _ = accepted_boundary_sr_queue_propagate(
        recommitted.queue_after,
        ((1.0,),),
        expected_boundary_index=3,
        reference=reference,
    )
    assert paid_once[0].contains(float(materialized.lo))
    assert paid_once[0].contains(float(materialized.hi))
    assert float(paid_once[0].width()) < 1.01


def test_generic_queue_checkpoint_resume_preserves_owner_and_generation(tmp_path) -> None:
    current_state = FlowstarNormalFlowpipeState.from_exact_decimal_box(
        [("-0.5", "0.75")],
        2,
    )
    current = current_state.normalized_initial_tm(2)
    reference = Interval(-0.125, 0.25)
    queue = FlowstarSymbolicRemainderQueue.empty_accepted_boundary_sr(
        1,
        5,
        reference=reference,
    )
    updated, updated_iv, _, _ = accepted_boundary_sr_queue_propagate(
        queue,
        ((1.0,),),
        expected_boundary_index=0,
        reference=reference,
    )
    queue, _ = accepted_boundary_sr_queue_commit(
        queue,
        updated,
        updated_iv,
        (reference,),
        scales=(1.0,),
        accepted_boundary_index=1,
        reference=reference,
    )
    current_state = replace(
        current_state,
        step_index=1,
        symbolic_queue=queue,
        symbolic_queue_max_size=5,
    )
    checkpoint = tmp_path / "generic_sr_checkpoint"
    save_terminal_checkpoint(
        checkpoint,
        current=current,
        normal_state=current_state,
        scheduler={"current_time": 0.1, "h_next": 0.1},
        contract={"plant": "generic-polynomial"},
        provenance={"test": True},
    )
    loaded = load_terminal_checkpoint(
        checkpoint,
        expected_order=2,
        expected_dtype="float64",
    )
    loaded_queue = loaded.normal_state.symbolic_queue
    assert loaded_queue is not None
    validate_accepted_boundary_sr_queue(loaded_queue, expected_boundary_index=1)
    assert accepted_boundary_sr_queue_sha256(loaded_queue) == (
        accepted_boundary_sr_queue_sha256(queue)
    )


def test_generic_flowpipe_mode_is_not_bound_to_vdp_order_cutoff_or_capacity() -> None:
    state = FlowstarNormalFlowpipeState.from_exact_decimal_box(
        [("-0.1", "0.1")],
        3,
    )
    current = state.normalized_initial_tm(3)
    for expected_boundary in (1, 2):
        segment = flowpipe_step_flowstar_style_adaptive(
            scalar_quadratic_ode,
            current,
            h=0.01,
            h_min=0.01,
            h_max=0.01,
            order=3,
            target_remainder_radius=1e-4,
            cutoff_threshold=1e-12,
            reset_mode=GENERIC_ACCEPTED_BOUNDARY_SYMBOLIC_REMAINDER,
            flowstar_symbolic_queue_max_size=2,
            flowstar_normal_state=state,
        )
        assert segment.status == "validated"
        assert segment.reset_tm is not None
        assert segment.flowstar_normal_state is not None
        queue = segment.flowstar_normal_state.symbolic_queue
        assert queue is not None
        assert queue.owner_schema == "accepted_boundary_sr_v1"
        assert queue.generation == expected_boundary
        current = segment.reset_tm
        state = segment.flowstar_normal_state
    assert queue.reset_count == 1
    assert queue.J == ()


@pytest.mark.parametrize(
    ("dtype", "coefficient", "error"),
    [
        (torch.float32, 1.0, ValueError),
        (torch.float64, torch.inf, FloatingPointError),
    ],
)
def test_generic_operator_fails_closed_outside_cpu_float64_finite_contract(
    dtype: torch.dtype,
    coefficient: float,
    error: type[Exception],
) -> None:
    domain = [
        Interval(
            torch.tensor(-1.0, dtype=dtype),
            torch.tensor(1.0, dtype=dtype),
        )
    ]
    endpoint = TMVector(
        [
            TaylorModel(
                Polynomial({(1,): torch.tensor(coefficient, dtype=dtype)}, 1),
                Interval.zero(dtype=dtype),
                domain,
                order=2,
            )
        ]
    )
    with pytest.raises(error):
        prepare_accepted_boundary_sr(
            endpoint,
            endpoint,
            domain=domain,
            order=2,
            cutoff_threshold=None,
            queue_state=None,
            queue_capacity=2,
            previous_accepted_boundary_index=0,
            compose=_clone_composition,
            diagnostics={},
        )


def test_generic_operator_rejects_stale_capacity_and_nonconstant_endpoint() -> None:
    endpoint = _one_dimensional_linear_map(1.0, Interval.zero())
    right_map = TMVector.identity(endpoint.domain, order=2)
    stale_capacity = FlowstarSymbolicRemainderQueue.empty_accepted_boundary_sr(
        1,
        3,
        reference=endpoint[0].remainder,
    )
    with pytest.raises(ValueError, match="capacity mismatch"):
        prepare_accepted_boundary_sr(
            endpoint,
            right_map,
            domain=endpoint.domain,
            order=2,
            cutoff_threshold=None,
            queue_state=stale_capacity,
            queue_capacity=4,
            previous_accepted_boundary_index=0,
            compose=_clone_composition,
            diagnostics={},
        )

    constant_endpoint = TMVector(
        [
            TaylorModel(
                endpoint[0].polynomial + Polynomial.constant(0.5, 1),
                Interval.zero(),
                endpoint.domain,
                order=2,
            )
        ]
    )
    with pytest.raises(ValueError, match="constant-free"):
        prepare_accepted_boundary_sr(
            constant_endpoint,
            right_map,
            domain=endpoint.domain,
            order=2,
            cutoff_threshold=None,
            queue_state=None,
            queue_capacity=4,
            previous_accepted_boundary_index=0,
            compose=_clone_composition,
            diagnostics={},
        )
