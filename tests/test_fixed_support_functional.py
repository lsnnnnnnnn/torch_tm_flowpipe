from __future__ import annotations

import math

import pytest
import torch

from torch_tm_flowpipe.fixed_support import (
    FixedSupportDescriptor,
    FixedSupportReachability,
    diffreach_vdp_polynomial_rhs,
    diffreach_vdp_tm_rhs,
    fixed_support_kernel_plan,
)
from torch_tm_flowpipe.fixed_support_functional import (
    fixed_support_functional_verify,
    initialize_fixed_support_functional_state,
    make_fixed_support_functional_chunk,
    prepare_fixed_support_vdp_functional_step,
)


def _partition(batch: int, device: torch.device | str = "cpu"):
    left = int(math.sqrt(batch))
    while batch % left:
        left -= 1
    split_x, split_y = batch // left, left
    x = torch.linspace(1.1, 1.4, split_x + 1, dtype=torch.float64, device=device)
    y = torch.linspace(2.35, 2.45, split_y + 1, dtype=torch.float64, device=device)
    lo = []
    hi = []
    for i in range(split_x):
        for j in range(split_y):
            lo.append(torch.stack((x[i], y[j])))
            hi.append(torch.stack((x[i + 1], y[j + 1])))
    return torch.stack(lo), torch.stack(hi)


def _object_solver() -> FixedSupportReachability:
    support = FixedSupportDescriptor.diffreach_restricted_quadratic(2)
    return FixedSupportReachability(
        support=support,
        state_dim=2,
        polynomial_rhs=diffreach_vdp_polynomial_rhs,
        tm_rhs=diffreach_vdp_tm_rhs,
        step_size=0.01,
        initial_remainder=0.01,
        polynomial_picard_iterations=2,
        remainder_rounds=10,
        symbolic_window_size=1000,
    )


def _assert_equal(batch: int, steps: int, device: torch.device | str = "cpu") -> None:
    initial_lo, initial_hi = _partition(batch, device)
    support = FixedSupportDescriptor.diffreach_restricted_quadratic(2)
    object_result = _object_solver().verify(initial_lo, initial_hi, steps=steps)
    functional = fixed_support_functional_verify(
        initial_lo,
        initial_hi,
        fixed_support_kernel_plan(support, device=device, dtype=torch.float64),
        step_size=0.01,
        steps=steps,
        trace=True,
    )
    assert functional.first_failure_step is None
    assert functional.validated_steps == steps
    assert functional.host_synchronizations == 1
    assert functional.device_transfers == 0
    assert torch.equal(object_result.endpoint_lo, functional.endpoint_lo)
    assert torch.equal(object_result.endpoint_hi, functional.endpoint_hi)
    assert torch.equal(object_result.tube_lo, functional.tube_lo)
    assert torch.equal(object_result.tube_hi, functional.tube_hi)
    assert torch.equal(
        object_result.initial_inclusion_masks, functional.initial_inclusion_masks
    )
    assert torch.equal(
        object_result.round_inclusion_masks, functional.round_inclusion_masks
    )
    state = functional.final_state
    assert torch.equal(object_result.final_model.polynomial.coeffs, state.model_coeffs)
    assert torch.equal(object_result.final_model.remainder.lo, state.model_rem_lo)
    assert torch.equal(object_result.final_model.remainder.hi, state.model_rem_hi)
    assert torch.equal(
        object_result.final_parameterization.polynomial.coeffs,
        state.parameter_coeffs,
    )
    assert torch.equal(
        object_result.final_parameterization.remainder.lo,
        state.parameter_rem_lo,
    )
    assert torch.equal(
        object_result.final_parameterization.remainder.hi,
        state.parameter_rem_hi,
    )
    symbolic = object_result.final_symbolic_state
    assert torch.equal(symbolic.phi_buffer, state.phi_buffer)
    assert torch.equal(symbolic.j_buffer.lo, state.j_lo)
    assert torch.equal(symbolic.j_buffer.hi, state.j_hi)
    assert torch.equal(symbolic.inverse_scale, state.inverse_scale)
    assert torch.all(state.queue_count == symbolic.count)
    assert functional.tube_lo is not None and functional.tube_hi is not None
    assert torch.equal(state.tube_hull_lo, functional.tube_lo.amin(dim=1))
    assert torch.equal(state.tube_hull_hi, functional.tube_hi.amax(dim=1))


@pytest.mark.parametrize(
    ("batch", "steps"),
    [
        (1, 1),
        (1, 2),
        (1, 10),
        (1, 100),
        (8, 1),
        (8, 2),
        (8, 10),
        (8, 100),
        (64, 1),
        (64, 2),
        (64, 10),
        (64, 100),
    ],
)
def test_functional_eager_is_bit_exact_with_object_eager_cpu(batch, steps):
    _assert_equal(batch, steps)


def test_functional_summary_and_trace_final_state_are_bit_exact():
    initial_lo, initial_hi = _partition(8)
    support = FixedSupportDescriptor.diffreach_restricted_quadratic(2)
    plan = fixed_support_kernel_plan(support, device="cpu", dtype=torch.float64)
    trace = fixed_support_functional_verify(
        initial_lo, initial_hi, plan, step_size=0.01, steps=10, trace=True
    )
    summary = fixed_support_functional_verify(
        initial_lo, initial_hi, plan, step_size=0.01, steps=10, trace=False
    )
    assert trace.endpoint_lo is not None and summary.endpoint_lo is None
    for field in trace.final_state.__dataclass_fields__:
        assert torch.equal(
            getattr(trace.final_state, field), getattr(summary.final_state, field)
        ), field


def test_functional_fail_closed_freezes_failed_batches_without_host_gate():
    initial_lo = torch.tensor([[1.1, 2.35], [10.0, 10.0]], dtype=torch.float64)
    initial_hi = torch.tensor([[1.4, 2.45], [11.0, 11.0]], dtype=torch.float64)
    support = FixedSupportDescriptor.diffreach_restricted_quadratic(2)
    plan = fixed_support_kernel_plan(support, device="cpu", dtype=torch.float64)
    initial_state, _ = initialize_fixed_support_functional_state(
        initial_lo, initial_hi, plan, queue_capacity=2
    )
    result = fixed_support_functional_verify(
        initial_lo,
        initial_hi,
        plan,
        step_size=0.01,
        steps=2,
        initial_remainder_radius=0.01,
        trace=True,
    )
    assert result.first_failure_step == 0
    assert result.validated_steps == 0
    assert torch.equal(result.final_state.active_mask, torch.tensor([True, False]))
    assert torch.equal(result.final_state.first_failure_index, torch.tensor([-1, 0]))
    assert torch.equal(
        result.final_state.model_coeffs[1], initial_state.model_coeffs[1]
    )
    assert torch.equal(
        result.final_state.parameter_coeffs[1], initial_state.parameter_coeffs[1]
    )
    assert torch.equal(result.final_state.phi_buffer[1], initial_state.phi_buffer[1])
    object_failure = _object_solver().verify(initial_lo, initial_hi, steps=2)
    assert object_failure.first_failure_step == result.first_failure_step
    assert object_failure.validated_steps == result.validated_steps


@pytest.mark.integration
def test_fullgraph_step_matches_eager_on_first_and_later_same_signature_inputs():
    initial_lo, initial_hi = _partition(8)
    support = FixedSupportDescriptor.diffreach_restricted_quadratic(2)
    plan = fixed_support_kernel_plan(support, device="cpu", dtype=torch.float64)
    initial_state, eager_step = prepare_fixed_support_vdp_functional_step(
        initial_lo, initial_hi, plan, step_size=0.01, steps=10
    )
    compiled_step = torch.compile(
        eager_step, backend="eager", fullgraph=True, dynamic=False
    )
    state = initial_state.tensors()
    for _ in range(8):
        expected = eager_step(state)
        actual = compiled_step(state)
        for expected_tensor, actual_tensor in zip(expected, actual):
            assert torch.equal(expected_tensor, actual_tensor)
        state = expected


@pytest.mark.integration
def test_fullgraph_chunk10_matches_ten_eager_steps():
    initial_lo, initial_hi = _partition(1)
    support = FixedSupportDescriptor.diffreach_restricted_quadratic(2)
    plan = fixed_support_kernel_plan(support, device="cpu", dtype=torch.float64)
    initial_state, eager_step = prepare_fixed_support_vdp_functional_step(
        initial_lo, initial_hi, plan, step_size=0.01, steps=10
    )
    expected = initial_state.tensors()
    for _ in range(10):
        expected = eager_step(expected)
    chunk10 = make_fixed_support_functional_chunk(eager_step, chunk_size=10)
    compiled_chunk10 = torch.compile(
        chunk10, backend="eager", fullgraph=True, dynamic=False
    )
    actual = compiled_chunk10(initial_state.tensors())
    for expected_tensor, actual_tensor in zip(expected, actual):
        assert torch.equal(expected_tensor, actual_tensor)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("batch", [1, 8, 64])
def test_functional_cuda_matches_object_cuda_for_multiple_batches(batch):
    _assert_equal(batch, 10, torch.device("cuda:0"))
