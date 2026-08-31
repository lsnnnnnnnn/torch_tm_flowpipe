from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from torch_tm_flowpipe import Interval, validate_accepted_boundary_sr_queue
from torch_tm_flowpipe.symbolic_remainder import (
    FlowstarSymbolicRemainderQueue,
    _add_interval_columns,
    _matmul_interval_matrix_col_scalar,
    _matmul_interval_matrix_scalar,
    _tensorized_interval_matrix_update_and_image,
)


def _interval(seed: int) -> Interval:
    center = ((seed * 37) % 29 - 14) / 8.0
    radius = ((seed * 17) % 7 + 1) / 64.0
    return Interval(center - radius, center + radius)


def _matrix(dim: int, seed: int) -> tuple[tuple[Interval, ...], ...]:
    return tuple(
        tuple(_interval(seed + row * dim + column) for column in range(dim))
        for row in range(dim)
    )


def _column(dim: int, seed: int) -> tuple[Interval, ...]:
    return tuple(_interval(seed + row) for row in range(dim))


def _bounds(values: object) -> list[tuple[torch.Tensor, torch.Tensor]]:
    flattened: list[Interval] = []

    def visit(value: object) -> None:
        if isinstance(value, Interval):
            flattened.append(value)
        elif isinstance(value, tuple):
            for child in value:
                visit(child)

    visit(values)
    return [(value.lo, value.hi) for value in flattened]


@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("owner_count", [0, 1, 7, 99])
def test_tensorized_owner_payload_is_bitwise_scalar_schedule_oracle(
    dim: int,
    owner_count: int,
) -> None:
    reference = Interval.zero()
    left = _matrix(dim, 1000)
    matrices = tuple(_matrix(dim, 2000 + 31 * owner) for owner in range(owner_count))
    columns = tuple(_column(dim, 3000 + 13 * owner) for owner in range(owner_count))

    scalar_matrices = tuple(
        _matmul_interval_matrix_scalar(left, matrix, reference)
        for matrix in matrices
    )
    scalar_image = tuple(Interval.zero() for _ in range(dim))
    for matrix, column in zip(scalar_matrices, columns):
        scalar_image = _add_interval_columns(
            scalar_image,
            _matmul_interval_matrix_col_scalar(matrix, column, reference),
        )

    packed_matrices, packed_image = _tensorized_interval_matrix_update_and_image(
        left,
        matrices,
        columns,
        reference,
    )
    expected = _bounds((scalar_matrices, scalar_image))
    actual = _bounds((packed_matrices, packed_image))
    assert len(actual) == len(expected)
    for (actual_lo, actual_hi), (expected_lo, expected_hi) in zip(actual, expected):
        assert torch.equal(actual_lo, expected_lo)
        assert torch.equal(actual_hi, expected_hi)


def test_tensorized_owner_payload_retains_subnormal_and_owner_addition_order() -> None:
    tiny = torch.nextafter(
        torch.tensor(0.0, dtype=torch.float64),
        torch.tensor(torch.inf, dtype=torch.float64),
    )
    reference = Interval.zero()
    left = ((Interval(1.0, 1.0),),)
    matrices = (((Interval(1.0, 1.0),),),) * 4
    columns = (
        (Interval(tiny, tiny),),
        (Interval(1e200, 1e200),),
        (Interval(-1e200, -1e200),),
        (Interval(tiny, tiny),),
    )
    scalar_matrices = tuple(
        _matmul_interval_matrix_scalar(left, matrix, reference)
        for matrix in matrices
    )
    scalar_image = (Interval.zero(),)
    for matrix, column in zip(scalar_matrices, columns):
        scalar_image = _add_interval_columns(
            scalar_image,
            _matmul_interval_matrix_col_scalar(matrix, column, reference),
        )
    packed_matrices, packed_image = _tensorized_interval_matrix_update_and_image(
        left,
        matrices,
        columns,
        reference,
    )
    assert _bounds(packed_matrices) == _bounds(scalar_matrices)
    assert torch.equal(packed_image[0].lo, scalar_image[0].lo)
    assert torch.equal(packed_image[0].hi, scalar_image[0].hi)


@pytest.mark.parametrize("payload", ["scalar", "phi", "owner"])
def test_packed_validation_fails_closed_after_mutable_tensor_tamper(payload: str) -> None:
    owner = Interval(-0.125, 0.25)
    queue = FlowstarSymbolicRemainderQueue(
        J=((owner,),),
        Phi_L=(((1.0,),),),
        scalars=(1.0,),
        max_size=8,
        Phi_L_iv=(((Interval(1.0, 1.0),),),),
        scalars_iv=(Interval(1.0, 1.0),),
        generation=1,
        accepted_boundary_index=1,
        owner_generations=(1,),
        owner_boundary_indices=(1,),
        owner_schema="accepted_boundary_sr_v1",
    )
    validate_accepted_boundary_sr_queue(queue, expected_boundary_index=1)
    if payload == "scalar":
        queue.scalars_iv[0].hi.fill_(torch.inf)
    elif payload == "phi":
        queue.Phi_L_iv[0][0][0].lo.fill_(torch.inf)
    else:
        queue.J[0][0].lo.fill_(-torch.inf)
    with pytest.raises(FloatingPointError):
        validate_accepted_boundary_sr_queue(queue, expected_boundary_index=1)


def test_packed_validation_rejects_nonfinite_point_payload() -> None:
    queue = FlowstarSymbolicRemainderQueue.empty_accepted_boundary_sr(1, 8)
    with pytest.raises(FloatingPointError, match="point scalar"):
        validate_accepted_boundary_sr_queue(replace(queue, scalars=(float("inf"),)))
