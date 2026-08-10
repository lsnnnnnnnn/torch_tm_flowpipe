import pytest
import torch

from torch_tm_flowpipe import Interval, flowpipe_step_flowstar_style_adaptive
from torch_tm_flowpipe.batched_dense_tm import (
    REMAINDER_LEDGER_CATEGORIES,
    VALIDATED_REMAINDER_SOURCE_SCHEMA,
    VALIDATED_REMAINDER_SOURCE_SCHEMA_VERSION,
)
from torch_tm_flowpipe.ode_examples import van_der_pol_ode
from torch_tm_flowpipe.structured_remainder import (
    initialize_structured_remainder_state,
    materialize_structured_remainder,
    normal_interval_to_physical,
    physical_interval_to_normal,
    structured_remainder_boundary_update,
)


DTYPE = torch.float64


def _zeros(batch, dim):
    return torch.zeros((batch, dim), dtype=DTYPE)


def _schema(batch, dim, updates=None):
    updates = updates or {}
    return {
        category: updates.get(category, (_zeros(batch, dim), _zeros(batch, dim)))
        for category in REMAINDER_LEDGER_CATEGORIES
    }


def _dense_vdp(h):
    return flowpipe_step_flowstar_style_adaptive(
        van_der_pol_ode,
        [Interval(1.1, 1.4), Interval(2.35, 2.45)],
        h=h,
        h_min=h,
        h_max=h,
        order=4,
        target_remainder_radius=1e-4,
        cutoff_threshold=1e-10,
        max_validation_attempts=2,
        validation_mode="flowstar_raw_remainder_compat",
        reset_mode="normalized_insertion",
        step_policy_mode="flowstar_compat",
        tm_backend="dense",
    )


def test_accepted_and_failed_dense_steps_expose_tensor_native_canonical_ledgers():
    accepted = _dense_vdp(0.005)
    assert accepted.status == "validated"
    ledger = accepted.validated_remainder_ledger
    decomposition = accepted.validated_remainder_decomposition
    assert ledger.category_order == REMAINDER_LEDGER_CATEGORIES
    assert decomposition.source_schema == VALIDATED_REMAINDER_SOURCE_SCHEMA
    assert decomposition.source_schema_version == VALIDATED_REMAINDER_SOURCE_SCHEMA_VERSION
    assert decomposition.contains_image.tolist() == [True]
    for lo, hi in ledger.entries.values():
        assert isinstance(lo, torch.Tensor) and isinstance(hi, torch.Tensor)
        assert lo.device.type == hi.device.type == "cpu"
        assert lo.dtype == hi.dtype == DTYPE

    failed = _dense_vdp(0.1)
    assert failed.status == "failed"
    assert failed.validated_remainder_ledger.category_order == REMAINDER_LEDGER_CATEGORIES
    assert failed.validated_remainder_decomposition.contains_image.tolist() == [True]
    assert failed.endpoint_raw_tm is None


def test_physical_normal_roundtrip_uses_inverse_scale_and_handles_zero_scale():
    physical_lo = torch.tensor([[-1.0, -0.25]], dtype=DTYPE)
    physical_hi = torch.tensor([[2.0, 0.75]], dtype=DTYPE)
    scale = torch.tensor([[2.0, 0.5]], dtype=DTYPE)
    inverse = torch.tensor([[0.5, 2.0]], dtype=DTYPE)
    normal = physical_interval_to_normal(
        physical_lo,
        physical_hi,
        forward_scale=scale,
        inverse_scale=inverse,
    )
    reconstructed = normal_interval_to_physical(
        normal.lo,
        normal.hi,
        forward_scale=scale,
    )
    assert torch.all(reconstructed.lo <= physical_lo)
    assert torch.all(reconstructed.hi >= physical_hi)
    with pytest.raises(ValueError, match="inconsistent"):
        physical_interval_to_normal(
            physical_lo,
            physical_hi,
            forward_scale=scale,
            inverse_scale=torch.ones_like(scale),
        )

    zero_scale = torch.tensor([[0.0, 1.0]], dtype=DTYPE)
    zero_inverse = torch.ones_like(zero_scale)
    zero_source = _zeros(1, 2)
    normalized_zero = physical_interval_to_normal(
        zero_source,
        zero_source,
        forward_scale=zero_scale,
        inverse_scale=zero_inverse,
    )
    assert torch.equal(normalized_zero.lo, zero_source)
    nonzero = zero_source.clone()
    nonzero[0, 0] = 1.0
    with pytest.raises(ValueError, match="zero-scale"):
        physical_interval_to_normal(
            zero_source,
            nonzero,
            forward_scale=zero_scale,
            inverse_scale=zero_inverse,
        )


def test_boundary_insertion_reconstructs_physical_source_under_asymmetric_scales():
    state = initialize_structured_remainder_state(1, 2)
    source = (
        torch.tensor([[-0.4, -0.25]], dtype=DTYPE),
        torch.tensor([[0.8, 0.75]], dtype=DTYPE),
    )
    sources = _schema(1, 2, {"polynomial_truncation": source})
    scale = torch.tensor([[2.0, 0.5]], dtype=DTYPE)
    identity = torch.eye(2, dtype=DTYPE).unsqueeze(0)
    result = structured_remainder_boundary_update(
        state,
        typed_sources=sources,
        validated_remainder_lo=source[0],
        validated_remainder_hi=source[1],
        A_old_normal_to_new_normal_lo=identity,
        A_old_normal_to_new_normal_hi=identity,
        nonlinear_residual_lo=_zeros(1, 2),
        nonlinear_residual_hi=_zeros(1, 2),
        new_forward_scale=scale,
        boundary_index=0,
        map_is_affine=True,
    )
    assert result.accepted.tolist() == [True]
    assert torch.equal(result.state.inverse_scale, torch.tensor([[0.5, 2.0]], dtype=DTYPE))
    normal_total = materialize_structured_remainder(result.state)
    physical_total = normal_interval_to_physical(
        normal_total.lo,
        normal_total.hi,
        forward_scale=scale,
    )
    assert torch.all(physical_total.lo <= source[0])
    assert torch.all(physical_total.hi >= source[1])


def test_source_schema_unknown_and_missing_categories_fail_closed_without_mutation():
    state = initialize_structured_remainder_state(1, 1)
    sources = _schema(1, 1)
    identity = torch.ones((1, 1, 1), dtype=DTYPE)
    common = dict(
        validated_remainder_lo=_zeros(1, 1),
        validated_remainder_hi=_zeros(1, 1),
        A_old_normal_to_new_normal_lo=identity,
        A_old_normal_to_new_normal_hi=identity,
        nonlinear_residual_lo=_zeros(1, 1),
        nonlinear_residual_hi=_zeros(1, 1),
        new_forward_scale=torch.ones((1, 1), dtype=DTYPE),
        boundary_index=0,
        map_is_affine=True,
    )
    unknown = structured_remainder_boundary_update(
        state,
        typed_sources={**sources, "polynomial_truncaton": (_zeros(1, 1), _zeros(1, 1))},
        **common,
    )
    assert not unknown.accepted.item()
    assert unknown.state is state
    assert unknown.failure_reason.startswith("unknown_source_category")

    missing_sources = dict(sources)
    missing_sources.pop("cutoff")
    missing = structured_remainder_boundary_update(
        state,
        typed_sources=missing_sources,
        **common,
    )
    assert not missing.accepted.item()
    assert missing.state is state
    assert missing.failure_reason.startswith("missing_source_category")
