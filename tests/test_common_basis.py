from __future__ import annotations

import pytest
import torch

from torch_tm_flowpipe import (
    AffineCoordinateBasis,
    IntervalPolynomialBatch,
    affine_common_basis_transform,
    evaluate_common_basis_point,
)


def _basis(names, center, scale, lo, hi, *, time_name="tau"):
    return AffineCoordinateBasis(
        tuple(names),
        torch.tensor(center, dtype=torch.float64),
        torch.tensor(scale, dtype=torch.float64),
        torch.tensor(lo, dtype=torch.float64),
        torch.tensor(hi, dtype=torch.float64),
        time_name=time_name,
        time_semantics="physical_local_time_[0,h]" if time_name is not None else None,
    )


def _coefficient_interval(result, exponent):
    slot = result.transformed.support.index(tuple(exponent))
    return result.transformed.coeff_lo[..., slot], result.transformed.coeff_hi[..., slot]


def test_affine_quadratic_transform_has_analytic_coefficients_and_explicit_time():
    source = _basis(
        ("x0", "tau"), [[2.0, 0.0]], [[3.0, 1.0]], [[-1.0, 0.0]], [[1.0, 0.1]]
    )
    target = _basis(
        ("x0", "tau"), [[5.0, 0.0]], [[2.0, 1.0]], [[-1.0, 0.0]], [[1.0, 0.1]]
    )
    polynomial = IntervalPolynomialBatch.from_point_coefficients(
        torch.tensor([[[1.0, 2.0, 3.0, 4.0]]], dtype=torch.float64),
        [(0, 0), (1, 0), (2, 0), (0, 1)],
    )
    result = affine_common_basis_transform(polynomial, source, target)
    expected = {(0, 0): 6.0, (1, 0): 16.0 / 3.0, (2, 0): 4.0 / 3.0, (0, 1): 4.0}
    for exponent, value in expected.items():
        lo, hi = _coefficient_interval(result, exponent)
        assert float(lo) <= value <= float(hi)
        assert float(hi - lo) < 1e-12
    assert result.time_treatment == "preserved tau: physical_local_time_[0,h]"
    assert result.coordinate_identity_known


def test_round_trip_contains_original_at_corresponding_physical_points():
    source = _basis(
        ("x0", "y0", "tau"),
        [[1.5, 2.0, 0.0]],
        [[0.2, 0.3, 1.0]],
        [[-1.0, -1.0, 0.0]],
        [[1.0, 1.0, 0.02]],
    )
    target = _basis(
        ("tau", "y0", "x0"),
        [[0.0, 2.02, 1.48]],
        [[1.0, 0.25, 0.18]],
        [[0.0, -1.0, -1.0]],
        [[0.02, 1.0, 1.0]],
    )
    polynomial = IntervalPolynomialBatch.from_point_coefficients(
        torch.tensor([[[1.0, 0.5, -0.25, 2.0], [-1.0, 0.3, 0.1, -0.4]]]),
        [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 1)],
    )
    transformed = affine_common_basis_transform(polynomial, source, target)
    round_trip = affine_common_basis_transform(transformed.transformed, target, source)
    source_point = torch.tensor([[0.2, -0.4, 0.01]], dtype=torch.float64)
    original_lo, original_hi = evaluate_common_basis_point(polynomial, source_point)
    round_lo, round_hi = evaluate_common_basis_point(round_trip.transformed, source_point)
    assert torch.all(round_lo <= original_lo)
    assert torch.all(round_hi >= original_hi)

    physical = source.center + source.scale * source_point
    target_point = torch.empty_like(physical)
    for index, name in enumerate(target.names):
        source_index = source.names.index(name)
        target_point[:, index] = (
            physical[:, source_index] - target.center[:, index]
        ) / target.scale[:, index]
    transformed_lo, transformed_hi = evaluate_common_basis_point(transformed.transformed, target_point)
    assert torch.all(transformed_lo <= original_lo)
    assert torch.all(transformed_hi >= original_hi)


def test_batch_state_support_retention_and_intervalized_discarded_terms():
    source = _basis(
        ("x0", "tau"),
        [[1.0, 0.0], [2.0, 0.0]],
        [[0.5, 1.0], [0.25, 1.0]],
        [[-1.0, 0.0], [-1.0, 0.0]],
        [[1.0, 0.1], [1.0, 0.1]],
    )
    target = _basis(
        ("x0", "tau"), [[0.0, 0.0]], [[1.0, 1.0]], [[0.5, 0.0]], [[2.5, 0.1]]
    )
    coefficients = torch.tensor(
        [
            [[1.0, 2.0, 3.0], [-1.0, 0.5, 0.25]],
            [[4.0, -2.0, 1.0], [0.0, 3.0, -0.5]],
        ],
        dtype=torch.float64,
    )
    polynomial = IntervalPolynomialBatch.from_point_coefficients(
        coefficients, [(0, 0), (1, 0), (2, 0)]
    )
    result = affine_common_basis_transform(
        polynomial, source, target, retain_support=[(0, 0), (1, 0)]
    )
    assert result.transformed.batch == 2
    assert result.transformed.states == 2
    assert result.retained.support == ((0, 0), (1, 0))
    assert torch.all(result.intervalized_discarded_lo <= result.intervalized_discarded_hi)
    assert torch.any(result.intervalized_discarded_hi - result.intervalized_discarded_lo > 0)


def test_unknown_nonconstant_coordinate_and_time_mismatch_fail_closed():
    source = _basis(
        ("x0", "clock0", "tau"),
        [[1.0, 0.0, 0.0]],
        [[0.2, 1.0, 1.0]],
        [[-1.0, -1.0, 0.0]],
        [[1.0, 1.0, 0.1]],
    )
    target = _basis(
        ("x0", "tau"), [[0.0, 0.0]], [[1.0, 1.0]], [[0.8, 0.0]], [[1.2, 0.1]]
    )
    nonconstant_clock = IntervalPolynomialBatch.from_point_coefficients(
        torch.tensor([[[1.0]]]), [(0, 1, 0)]
    )
    with pytest.raises(ValueError, match="unknown.*clock0"):
        affine_common_basis_transform(nonconstant_clock, source, target)

    zero_clock = IntervalPolynomialBatch.from_point_coefficients(
        torch.tensor([[[1.0, 2.0]]]), [(0, 0, 0), (1, 0, 0)]
    )
    result = affine_common_basis_transform(zero_clock, source, target)
    assert result.dropped_zero_variables == ("clock0",)

    wrong_time = AffineCoordinateBasis(
        target.names,
        target.center,
        target.scale,
        target.domain_lo,
        target.domain_hi,
        time_name="tau",
        time_semantics="normalized_[0,1]",
    )
    with pytest.raises(ValueError, match="local-time"):
        affine_common_basis_transform(zero_clock, source, wrong_time)
