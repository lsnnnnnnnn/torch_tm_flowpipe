from __future__ import annotations

import math

import mpmath
import pytest
import torch

from torch_tm_flowpipe.batched_dense_tm import (
    BatchedMonomialBasis,
    BatchedPolynomial,
    BatchedTaylorModel,
    _point_sin_cos_enclosure,
    sin_tm,
)


def affine_input(
    center: float,
    radius: float,
    *,
    remainder: tuple[float, float] = (0.0, 0.0),
    device: str = "cpu",
) -> BatchedTaylorModel:
    basis = BatchedMonomialBasis.build(2, 3, device)
    coefficients = torch.zeros(
        (1, 1, basis.num_terms), dtype=torch.float64, device=device
    )
    coefficients[..., basis.constant_index] = center
    coefficients[..., basis.term_index((0, 1))] = radius
    polynomial = BatchedPolynomial(coefficients, basis)
    domain_lo = torch.tensor([[0.0, -1.0]], dtype=torch.float64, device=device)
    domain_hi = torch.tensor([[0.1, 1.0]], dtype=torch.float64, device=device)
    rem_lo = torch.tensor([[remainder[0]]], dtype=torch.float64, device=device)
    rem_hi = torch.tensor([[remainder[1]]], dtype=torch.float64, device=device)
    return BatchedTaylorModel(
        polynomial, rem_lo, rem_hi, domain_lo, domain_hi
    )


@pytest.mark.parametrize("order", [0, 1, 2, 3])
@pytest.mark.parametrize(
    ("center", "radius", "remainder"),
    [
        (0.0, 0.0, (0.0, 0.0)),
        (0.0, 0.4, (0.0, 0.0)),
        (-0.35, 0.05, (0.0, 0.0)),
        (0.2, 0.15, (-0.01, 0.02)),
        (1.6, 0.2, (0.0, 0.0)),
    ],
)
def test_sin_tm_encloses_dense_oracle_grid(
    order: int,
    center: float,
    radius: float,
    remainder: tuple[float, float],
) -> None:
    model = affine_input(center, radius, remainder=remainder)
    result = sin_tm(model, order=order)
    lower, upper = result.range_bound(context="test_sine_output")
    input_lower = center - radius + remainder[0]
    input_upper = center + radius + remainder[1]
    samples = torch.linspace(
        input_lower, input_upper, 2001, dtype=torch.float64
    )
    exact = torch.sin(samples)
    assert lower.item() <= exact.min().item()
    assert upper.item() >= exact.max().item()
    assert result.ledger.entries["composition_overflow"][0].shape == (1, 1)


@pytest.mark.parametrize("value", [-8.0, -3.25, -0.4, 0.0, 1.6, 7.75, 8.0])
def test_point_sin_cos_enclosure_contains_high_precision_oracle(value: float) -> None:
    point = torch.tensor([[value]], dtype=torch.float64)
    sin_lo, sin_hi, cos_lo, cos_hi = _point_sin_cos_enclosure(point)
    with mpmath.workdps(100):
        exact_sin = float(mpmath.sin(mpmath.mpf(value)))
        exact_cos = float(mpmath.cos(mpmath.mpf(value)))
    assert sin_lo.item() <= exact_sin <= sin_hi.item()
    assert cos_lo.item() <= exact_cos <= cos_hi.item()


def test_sin_tm_interval_crossing_extremum_contains_one() -> None:
    result = sin_tm(affine_input(1.6, 0.2), order=3)
    lower, upper = result.range_bound(context="crossing_extremum")
    assert lower.item() <= math.sin(1.4)
    assert upper.item() >= 1.0


def test_sin_tm_polynomial_composition_routes_degree_overflow_to_remainder() -> None:
    model = affine_input(0.1, 0.4)
    coefficients = model.poly.coeffs.clone()
    coefficients[..., model.poly.basis.term_index((0, 2))] = 0.05
    nonlinear = BatchedTaylorModel(
        BatchedPolynomial(coefficients, model.poly.basis),
        model.rem_lo,
        model.rem_hi,
        model.domain_lo,
        model.domain_hi,
    )
    result = sin_tm(nonlinear, order=3)
    assert "polynomial_truncation" in result.ledger.entries
    assert "roundoff_safeguard" in result.ledger.entries
    assert torch.all(result.rem_lo <= result.rem_hi)


@pytest.mark.parametrize(
    ("center", "radius"),
    [(-0.35, 0.05), (-0.2, 0.2), (0.4, 0.1), (1.5, 0.25)],
)
def test_sin_tm_tora_domain_regressions(center: float, radius: float) -> None:
    result = sin_tm(affine_input(center, radius), order=2)
    lower, upper = result.range_bound(context="tora_x3_regression")
    samples = torch.linspace(center - radius, center + radius, 1001)
    assert lower.item() <= torch.sin(samples).min().item()
    assert upper.item() >= torch.sin(samples).max().item()


def test_sin_tm_wide_domain_fails_closed() -> None:
    with pytest.raises(ValueError, match="split the input domain or fail closed"):
        sin_tm(affine_input(0.0, 4.01), order=3)


def test_sin_tm_rejects_non_float64_formal_path() -> None:
    model = affine_input(0.0, 0.1)
    converted = BatchedTaylorModel(
        BatchedPolynomial(model.poly.coeffs.float(), model.poly.basis),
        model.rem_lo.float(),
        model.rem_hi.float(),
        model.domain_lo.float(),
        model.domain_hi.float(),
    )
    with pytest.raises(TypeError, match="requires float64"):
        sin_tm(converted)


@pytest.mark.cuda
def test_sin_tm_cpu_cuda_enclosure_parity() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    cpu = sin_tm(
        affine_input(-0.35, 0.05, remainder=(-1e-4, 2e-4)), order=3
    )
    gpu = sin_tm(
        affine_input(
            -0.35,
            0.05,
            remainder=(-1e-4, 2e-4),
            device="cuda",
        ),
        order=3,
    )
    cpu_lo, cpu_hi = cpu.range_bound(context="cpu_parity")
    gpu_lo, gpu_hi = gpu.range_bound(context="cuda_parity")
    assert torch.allclose(cpu_lo, gpu_lo.cpu(), rtol=0.0, atol=2e-14)
    assert torch.allclose(cpu_hi, gpu_hi.cpu(), rtol=0.0, atol=2e-14)
