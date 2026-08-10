from __future__ import annotations

import pytest
import torch

from torch_tm_flowpipe.structured_remainder import (
    STRUCTURED_REMAINDER_CAPACITY,
    initialize_structured_remainder_state,
    materialize_structured_remainder,
    structured_quadratic_nonlinear_residual,
    structured_remainder_boundary_update,
)


DTYPE = torch.float64


def _identity(batch: int, dim: int) -> torch.Tensor:
    return torch.eye(dim, dtype=DTYPE).expand(batch, -1, -1).clone()


def _zero(batch: int, dim: int) -> torch.Tensor:
    return torch.zeros((batch, dim), dtype=DTYPE)


def _update(
    state,
    sources,
    *,
    linear=None,
    nonlinear=None,
    validated=None,
    boundary=0,
    affine=True,
):
    batch, dim = state.batch, state.state_dim
    linear = _identity(batch, dim) if linear is None else linear
    nonlinear = (_zero(batch, dim), _zero(batch, dim)) if nonlinear is None else nonlinear
    if validated is None:
        lo = _zero(batch, dim)
        hi = _zero(batch, dim)
        for source_lo, source_hi in sources.values():
            lo = lo + source_lo
            hi = hi + source_hi
        validated = (lo, hi)
    return structured_remainder_boundary_update(
        state,
        typed_sources=sources,
        validated_remainder_lo=validated[0],
        validated_remainder_hi=validated[1],
        linear_map_lo=linear,
        linear_map_hi=linear,
        nonlinear_residual_lo=nonlinear[0],
        nonlinear_residual_hi=nonlinear[1],
        normalization_scale=torch.ones((batch, dim), dtype=DTYPE),
        boundary_index=boundary,
        map_is_affine=affine,
    )


@pytest.mark.unit
def test_asymmetric_source_is_centered_once_and_conserved_without_double_count():
    state = initialize_structured_remainder_state(1, 1)
    source = (
        torch.tensor([[-1.0]], dtype=DTYPE),
        torch.tensor([[3.0]], dtype=DTYPE),
    )
    result = _update(state, {"polynomial_truncation": source})
    assert result.accepted.tolist() == [True]
    assert result.state.active.sum().item() == 1
    assert result.state.source_id[result.state.active].item() == 1
    assert result.state.ordinary_rem_lo.item() <= 1.0 <= result.state.ordinary_rem_hi.item()
    assert result.materialized_lo.item() <= -1.0
    assert result.materialized_hi.item() >= 3.0
    # The eligible source is absent from ordinary apart from its unique center.
    assert result.state.ordinary_rem_hi.item() - result.state.ordinary_rem_lo.item() < 1e-12


@pytest.mark.unit
def test_zero_and_ineligible_sources_stay_ordinary_without_symbol_slots():
    state = initialize_structured_remainder_state(1, 2)
    zero = _zero(1, 2)
    result = _update(state, {"cutoff": (zero, zero)})
    assert result.accepted.tolist() == [True]
    assert not result.state.active.any()
    materialized = materialize_structured_remainder(result.state)
    assert torch.all(materialized.lo <= 0)
    assert torch.all(materialized.hi >= 0)


@pytest.mark.unit
def test_multiple_sources_and_affine_scalar_propagation_are_conservative():
    state = initialize_structured_remainder_state(1, 1)
    first = _update(
        state,
        {
            "polynomial_truncation": (
                torch.tensor([[-1.0]], dtype=DTYPE),
                torch.tensor([[1.0]], dtype=DTYPE),
            ),
            "integration_overflow": (
                torch.tensor([[-0.25]], dtype=DTYPE),
                torch.tensor([[0.5]], dtype=DTYPE),
            ),
        },
    )
    assert first.state.active.sum().item() == 2
    linear = torch.tensor([[[2.0]]], dtype=DTYPE)
    second = _update(first.state, {}, linear=linear, boundary=1)
    assert second.accepted.tolist() == [True]
    assert second.materialized_lo.item() <= 2.0 * first.materialized_lo.item()
    assert second.materialized_hi.item() >= 2.0 * first.materialized_hi.item()


@pytest.mark.unit
def test_harmonic_rotation_propagates_two_state_column():
    state = initialize_structured_remainder_state(1, 2)
    source = (
        torch.tensor([[-1.0, -2.0]], dtype=DTYPE),
        torch.tensor([[1.0, 2.0]], dtype=DTYPE),
    )
    first = _update(state, {"polynomial_truncation": source})
    rotation = torch.tensor([[[0.0, 1.0], [-1.0, 0.0]]], dtype=DTYPE)
    second = _update(first.state, {}, linear=rotation, boundary=1)
    assert second.accepted.tolist() == [True]
    assert second.materialized_lo[0, 0] <= -2.0
    assert second.materialized_hi[0, 0] >= 2.0
    assert second.materialized_lo[0, 1] <= -1.0
    assert second.materialized_hi[0, 1] >= 1.0


@pytest.mark.property
def test_scalar_quadratic_and_cross_residual_bounds_analytic_corners():
    z_lo = torch.tensor([[-0.5, -0.25]], dtype=DTYPE)
    z_hi = torch.tensor([[0.75, 0.5]], dtype=DTYPE)
    quadratic = torch.zeros((1, 2, 2, 2), dtype=DTYPE)
    quadratic[0, 0, 0, 0] = 2.0  # 2*x^2, Riccati-style
    quadratic[0, 1, 0, 1] = 3.0  # 3*x*y, cross-correlated system
    residual = structured_quadratic_nonlinear_residual(quadratic, z_lo, z_hi)
    for x in (-0.5, 0.75):
        for y in (-0.25, 0.5):
            values = torch.tensor([2.0 * x * x, 3.0 * x * y], dtype=DTYPE)
            assert torch.all(residual.lo[0] <= values)
            assert torch.all(residual.hi[0] >= values)


@pytest.mark.unit
def test_nonlinear_map_requires_residual_and_materializes_it_once():
    state = initialize_structured_remainder_state(1, 1)
    missing = structured_remainder_boundary_update(
        state,
        typed_sources={},
        validated_remainder_lo=_zero(1, 1),
        validated_remainder_hi=_zero(1, 1),
        linear_map_lo=_identity(1, 1),
        linear_map_hi=_identity(1, 1),
        nonlinear_residual_lo=None,
        nonlinear_residual_hi=None,
        normalization_scale=torch.ones((1, 1), dtype=DTYPE),
        boundary_index=0,
        map_is_affine=False,
    )
    assert not missing.accepted.item()
    assert missing.failure_reason == "missing_structured_nonlinear_residual"
    nonlinear = (
        torch.tensor([[-0.2]], dtype=DTYPE),
        torch.tensor([[0.3]], dtype=DTYPE),
    )
    present = _update(state, {}, nonlinear=nonlinear, affine=False)
    assert present.accepted.item()
    assert present.materialized_lo.item() <= -0.2
    assert present.materialized_hi.item() >= 0.3


@pytest.mark.unit
def test_k1_oldest_eviction_materializes_before_overwrite():
    state = initialize_structured_remainder_state(1, 1, capacity=1)
    first_source = (
        torch.tensor([[-1.0]], dtype=DTYPE),
        torch.tensor([[1.0]], dtype=DTYPE),
    )
    first = _update(state, {"polynomial_truncation": first_source}, boundary=4)
    second_source = (
        torch.tensor([[-2.0]], dtype=DTYPE),
        torch.tensor([[2.0]], dtype=DTYPE),
    )
    second = _update(
        first.state,
        {"integration_overflow": second_source},
        boundary=5,
    )
    assert second.accepted.item()
    assert second.evicted_source_id.item() == 1
    assert second.evicted_age.item() == 4
    assert second.state.source_id[0, 0].item() == 2
    assert second.materialized_lo.item() <= -3.0
    assert second.materialized_hi.item() >= 3.0


@pytest.mark.property
def test_k16_order_repetition_and_batch_permutation_are_deterministic():
    batch, dim = 3, 2
    state = initialize_structured_remainder_state(batch, dim)
    trunc_lo = torch.tensor(
        [[-0.1, -0.2], [-0.3, -0.4], [-0.5, -0.6]], dtype=DTYPE
    )
    trunc_hi = -trunc_lo
    integ_lo = 0.5 * trunc_lo
    integ_hi = -integ_lo
    sources = {
        "integration_overflow": (integ_lo, integ_hi),
        "polynomial_truncation": (trunc_lo, trunc_hi),
    }
    first = _update(state, sources)
    repeated = _update(state, sources)
    for field in first.state.__dataclass_fields__:
        assert torch.equal(getattr(first.state, field), getattr(repeated.state, field))
    assert first.state.capacity == STRUCTURED_REMAINDER_CAPACITY
    assert first.state.source_id[0, :2].tolist() == [1, 2]

    permutation = torch.tensor([2, 0, 1])
    permuted_state = initialize_structured_remainder_state(batch, dim)
    permuted_sources = {
        name: (lo[permutation], hi[permutation])
        for name, (lo, hi) in sources.items()
    }
    permuted = _update(permuted_state, permuted_sources)
    inverse = torch.argsort(permutation)
    assert torch.equal(first.materialized_lo, permuted.materialized_lo[inverse])
    assert torch.equal(first.materialized_hi, permuted.materialized_hi[inverse])


@pytest.mark.unit
def test_k16_eviction_order_is_oldest_then_slot_and_conservative():
    state = initialize_structured_remainder_state(1, 1)
    for boundary in range(8):
        radius = float(boundary + 1) / 100.0
        state = _update(
            state,
            {
                "polynomial_truncation": (
                    torch.tensor([[-radius]], dtype=DTYPE),
                    torch.tensor([[radius]], dtype=DTYPE),
                ),
                "integration_overflow": (
                    torch.tensor([[-radius / 2]], dtype=DTYPE),
                    torch.tensor([[radius / 2]], dtype=DTYPE),
                ),
            },
            boundary=boundary,
        ).state
    assert state.active.sum().item() == 16
    ninth = _update(
        state,
        {
            "polynomial_truncation": (
                torch.tensor([[-0.1]], dtype=DTYPE),
                torch.tensor([[0.1]], dtype=DTYPE),
            )
        },
        boundary=8,
    )
    assert ninth.accepted.item()
    assert ninth.evicted_age.item() == 0
    assert ninth.evicted_source_id.item() == 1
    assert ninth.evicted_materialized_lo.item() <= -0.01
    assert ninth.evicted_materialized_hi.item() >= 0.01


@pytest.mark.unit
def test_nonfinite_domain_and_dimension_fail_closed_without_state_mutation():
    state = initialize_structured_remainder_state(1, 2)
    bad_dimension = structured_remainder_boundary_update(
        state,
        typed_sources={},
        validated_remainder_lo=torch.zeros((1, 1), dtype=DTYPE),
        validated_remainder_hi=torch.zeros((1, 1), dtype=DTYPE),
        linear_map_lo=_identity(1, 2),
        linear_map_hi=_identity(1, 2),
        nonlinear_residual_lo=_zero(1, 2),
        nonlinear_residual_hi=_zero(1, 2),
        normalization_scale=torch.ones((1, 2), dtype=DTYPE),
        boundary_index=0,
        map_is_affine=True,
    )
    assert bad_dimension.failure_reason == "dimension_mismatch"
    nonfinite = _update(
        state,
        {"polynomial_truncation": (_zero(1, 2), torch.full((1, 2), torch.inf))},
    )
    assert not nonfinite.accepted.any()
    assert nonfinite.failure_reason in {"nonfinite_input", "nonfinite_source:polynomial_truncation"}
    assert nonfinite.state is state
