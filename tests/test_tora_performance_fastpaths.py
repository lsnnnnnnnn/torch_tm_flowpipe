from __future__ import annotations

import pytest
import torch

from torch_tm_flowpipe.batched_dense_tm import (
    BatchedMonomialBasis,
    BatchedPolynomial,
    _identity_dense_cover,
    _point_sin_cos_enclosure,
    _power_interval_bounds,
    dense_transient_ledger_suppressed,
    dense_validation_batch,
    validate_dense_subdivision_cover,
)
from torch_tm_flowpipe.tora_q3 import (
    build_tora_q3_initial_model,
    dense_tora_q3_dr_step,
)


@pytest.mark.regression
@pytest.mark.parametrize("batch", [1, 7, 48])
def test_identity_cover_by_construction_matches_independent_validator(batch: int) -> None:
    generator = torch.Generator().manual_seed(23 + batch)
    center = torch.randn((batch, 6), generator=generator, dtype=torch.float64)
    radius = torch.rand((batch, 6), generator=generator, dtype=torch.float64)
    domain_lo = center - radius
    domain_hi = center + radius
    coeffs = torch.zeros((batch, 5, 84), dtype=torch.float64)

    cover, report = _identity_dense_cover(coeffs, domain_lo, domain_hi)
    independent = validate_dense_subdivision_cover(cover, domain_lo, domain_hi)

    assert report["validation"] == "identity_cover_exact_by_construction"
    assert report["valid"] and independent["valid"]
    assert report["leaf_counts"] == independent["leaf_counts"]
    assert torch.equal(cover.lo, domain_lo)
    assert torch.equal(cover.hi, domain_hi)
    assert torch.equal(cover.owner, torch.arange(batch))


@pytest.mark.regression
def test_same_device_to_fastpaths_preserve_object_and_tensor_identity() -> None:
    basis = BatchedMonomialBasis.build(2, 2, "cpu")
    polynomial = BatchedPolynomial.zeros(3, 2, basis, dtype=torch.float64)
    assert polynomial.to("cpu") is polynomial

    model = build_tora_q3_initial_model(
        torch.full((48,), 9.8, dtype=torch.float64),
        torch.full((48,), 10.2, dtype=torch.float64),
    )
    assert model.to("cpu") is model


@pytest.mark.regression
def test_deferred_validation_fails_before_invalid_result_can_escape() -> None:
    lo = torch.tensor([-1.0], dtype=torch.float64)
    hi = torch.tensor([1.0], dtype=torch.float64)
    powers = torch.tensor([4], dtype=torch.long)
    with pytest.raises(ValueError, match="declared maximum"):
        with dense_validation_batch():
            _power_interval_bounds(lo, hi, powers, maximum_power=3)


@pytest.mark.regression
def test_phase_boundary_ledger_fastpath_is_bitwise_equal_on_cpu() -> None:
    lower = torch.full((48,), 9.8, dtype=torch.float64)
    upper = torch.full((48,), 10.2, dtype=torch.float64)
    base = build_tora_q3_initial_model(lower, upper)
    with torch.no_grad(), dense_validation_batch():
        strict = dense_tora_q3_dr_step(base, capture_trace=False)
    with (
        torch.no_grad(),
        dense_validation_batch(),
        dense_transient_ledger_suppressed(),
    ):
        optimized = dense_tora_q3_dr_step(base, capture_trace=False)
    pairs = (
        (strict.segment_tm.poly.coeffs, optimized.segment_tm.poly.coeffs),
        (strict.segment_tm.rem_lo, optimized.segment_tm.rem_lo),
        (strict.segment_tm.rem_hi, optimized.segment_tm.rem_hi),
        (strict.endpoint_lower, optimized.endpoint_lower),
        (strict.endpoint_upper, optimized.endpoint_upper),
        (strict.tube_lower, optimized.tube_lower),
        (strict.tube_upper, optimized.tube_upper),
    )
    assert all(torch.equal(left, right) for left, right in pairs)
    assert strict.segment_tm.ledger.entries.keys() == optimized.segment_tm.ledger.entries.keys()
    for category in strict.segment_tm.ledger.entries:
        assert all(
            torch.equal(left, right)
            for left, right in zip(
                strict.segment_tm.ledger.entries[category],
                optimized.segment_tm.ledger.entries[category],
                strict=True,
            )
        )


@pytest.mark.regression
def test_compiled_point_enclosure_soundly_falls_back_on_cpu() -> None:
    values = torch.linspace(-0.5, 0.5, 17, dtype=torch.float64)
    eager = _point_sin_cos_enclosure(values, backend="eager")
    fallback = _point_sin_cos_enclosure(values, backend="compiled")
    assert all(
        torch.equal(eager_value, fallback_value)
        for eager_value, fallback_value in zip(eager, fallback, strict=True)
    )
