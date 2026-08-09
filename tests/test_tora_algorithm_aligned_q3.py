from __future__ import annotations

import hashlib
import math
from pathlib import Path

import mpmath
import pytest
import torch

from torch_tm_flowpipe.batched_dense_tm import (
    BatchedMonomialBasis,
    BatchedPolynomial,
    BatchedTaylorModel,
    DenseRemainderLedger,
    sin_tm,
)
from torch_tm_flowpipe.tora_algorithm_aligned import (
    algorithm_aligned_q3_step,
    algorithm_aligned_tora_rhs,
    aligned_sin_tm,
)
from torch_tm_flowpipe.tora_q3 import (
    build_tora_q3_box_model,
    tora_b48_boxes,
)


ROOT = Path(__file__).resolve().parents[1]
ALIGNED_OUTPUT = (
    ROOT / "outputs/tora_q3_stage_parity_fused_20260809/algorithm_aligned"
)
FROZEN_BASELINE_HASHES = {
    "src/torch_tm_flowpipe/batched_dense_tm.py": (
        "7198489a4adcce07ad741a021da96e7e3ca4a033ba7d947d02edb88e141f1980"
    ),
    "src/torch_tm_flowpipe/tora_q3.py": (
        "e343b2a60c7bc3861f655f952b4333744e6ecd9929543a469a2f4a542f59b63b"
    ),
}


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
    domain_lo = torch.tensor([[0.0, -1.0]], dtype=torch.float64, device=device)
    domain_hi = torch.tensor([[0.1, 1.0]], dtype=torch.float64, device=device)
    rem_lo = torch.tensor([[remainder[0]]], dtype=torch.float64, device=device)
    rem_hi = torch.tensor([[remainder[1]]], dtype=torch.float64, device=device)
    return BatchedTaylorModel(
        BatchedPolynomial(coefficients, basis),
        rem_lo,
        rem_hi,
        domain_lo,
        domain_hi,
        DenseRemainderLedger.empty(),
    )


def evaluate_scalar_polynomial(
    model: BatchedTaylorModel, points: torch.Tensor
) -> torch.Tensor:
    monomials = torch.prod(
        points[:, None, :] ** model.poly.basis.exponents[None, :, :], dim=-1
    )
    return torch.einsum("bot,nt->nbo", model.poly.coeffs, monomials)


def one_leaf_model(device: str = "cpu") -> BatchedTaylorModel:
    lower, upper = tora_b48_boxes(device=device)
    control_lo = torch.tensor([9.8], dtype=torch.float64, device=device)
    control_hi = torch.tensor([10.2], dtype=torch.float64, device=device)
    return build_tora_q3_box_model(
        lower[:1], upper[:1], control_lo, control_hi, device=device
    )


@pytest.mark.unit
def test_aligned_sine_scalar_and_affine_retained_cases() -> None:
    scalar = aligned_sin_tm(affine_input(0.0, 0.0))
    assert torch.count_nonzero(scalar.poly.coeffs) == 0
    assert scalar.rem_lo.item() <= 0.0 <= scalar.rem_hi.item()

    affine = aligned_sin_tm(affine_input(0.0, 0.2))
    slot = affine.poly.basis.term_index((0, 1))
    assert affine.poly.coeffs[..., slot].item() == pytest.approx(
        0.2, rel=0.0, abs=5e-16
    )
    assert affine.poly.coeffs[..., affine.poly.basis.constant_index].item() == 0.0
    assert affine.rem_lo.item() < 0.0 < affine.rem_hi.item()


@pytest.mark.unit
@pytest.mark.parametrize(
    ("center", "radius", "remainder"),
    [
        (math.pi / 2.0, 0.2, (0.0, 0.0)),
        (-math.pi / 2.0, 0.2, (0.0, 0.0)),
        (-0.35, 0.08, (-0.01, 0.02)),
    ],
)
def test_aligned_sine_extrema_and_nonzero_remainder_contain_mpmath_oracle(
    center: float,
    radius: float,
    remainder: tuple[float, float],
) -> None:
    result = aligned_sin_tm(
        affine_input(center, radius, remainder=remainder)
    )
    lower, upper = result.range_bound(context="aligned_sine_mpmath_oracle")
    with mpmath.workdps(100):
        values = [
            mpmath.sin(
                mpmath.mpf(center)
                + mpmath.mpf(radius) * mpmath.mpf(-1 + 2 * index / 400)
                + mpmath.mpf(remainder[0])
                + (mpmath.mpf(remainder[1]) - mpmath.mpf(remainder[0]))
                * mpmath.mpf(index / 400)
            )
            for index in range(401)
        ]
    assert lower.item() <= float(min(values))
    assert upper.item() >= float(max(values))
    if center > 1.0:
        assert upper.item() >= 1.0
    if center < -1.0:
        assert lower.item() <= -1.0


@pytest.mark.unit
def test_aligned_sine_q3_degree_overflow_is_explicit_and_sound() -> None:
    model = affine_input(-0.3, 0.2, remainder=(-1e-3, 2e-3))
    coefficients = model.poly.coeffs.clone()
    coefficients[..., model.poly.basis.term_index((0, 3))] = 0.04
    nonlinear = BatchedTaylorModel(
        BatchedPolynomial(coefficients, model.poly.basis),
        model.rem_lo,
        model.rem_hi,
        model.domain_lo,
        model.domain_hi,
        DenseRemainderLedger.empty(),
    )
    result = aligned_sin_tm(nonlinear)
    assert "polynomial_truncation" in result.ledger.entries
    assert "roundoff_safeguard" in result.ledger.entries
    assert torch.all(result.rem_lo <= result.rem_hi)

    parameter = torch.linspace(-1.0, 1.0, 2001, dtype=torch.float64)
    points = torch.stack((torch.full_like(parameter, 0.05), parameter), dim=1)
    input_polynomial = evaluate_scalar_polynomial(nonlinear, points)[:, 0, 0]
    output_polynomial = evaluate_scalar_polynomial(result, points)[:, 0, 0]
    for input_remainder in nonlinear.rem_lo.item(), nonlinear.rem_hi.item():
        exact = torch.sin(input_polynomial + input_remainder)
        assert torch.all(exact >= output_polynomial + result.rem_lo.item())
        assert torch.all(exact <= output_polynomial + result.rem_hi.item())


@pytest.mark.regression
def test_aligned_remainder_routing_is_tighter_than_generic_sine() -> None:
    model = affine_input(-0.35, 0.08, remainder=(-0.01, 0.01))
    aligned = aligned_sin_tm(model)
    generic = sin_tm(model, order=2)
    aligned_width = aligned.rem_hi - aligned.rem_lo
    generic_width = generic.rem_hi - generic.rem_lo
    assert torch.all(aligned_width < generic_width)


@pytest.mark.property
def test_aligned_sine_random_sample_containment_sanity() -> None:
    torch.manual_seed(20260809)
    model = affine_input(-0.35, 0.12, remainder=(-0.015, 0.01))
    result = aligned_sin_tm(model)
    parameter = -1.0 + 2.0 * torch.rand(4096, dtype=torch.float64)
    points = torch.stack((0.1 * torch.rand_like(parameter), parameter), dim=1)
    input_remainder = model.rem_lo.item() + (
        model.rem_hi.item() - model.rem_lo.item()
    ) * torch.rand_like(parameter)
    input_value = evaluate_scalar_polynomial(model, points)[:, 0, 0]
    output_value = evaluate_scalar_polynomial(result, points)[:, 0, 0]
    exact = torch.sin(input_value + input_remainder)
    assert torch.all(exact >= output_value + result.rem_lo.item())
    assert torch.all(exact <= output_value + result.rem_hi.item())


@pytest.mark.integration
def test_algorithm_aligned_step_validates_and_holds_control_exactly() -> None:
    base = one_leaf_model()
    rhs = algorithm_aligned_tora_rhs(base)
    assert torch.count_nonzero(rhs.poly.coeffs[:, 4]) == 0
    assert torch.equal(rhs.rem_lo[:, 4], torch.zeros(1))
    assert torch.equal(rhs.rem_hi[:, 4], torch.zeros(1))

    step = algorithm_aligned_q3_step(base)
    assert step.accepted
    assert len(step.polynomial_trace) == 2
    assert len(step.round_trace) == 10
    assert torch.equal(step.segment_tm.rem_lo[:, 4], torch.zeros(1))
    assert torch.equal(step.segment_tm.rem_hi[:, 4], torch.zeros(1))


@pytest.mark.cuda
@pytest.mark.integration
def test_algorithm_aligned_cpu_cuda_each_contains_same_oracle() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    inputs = (-0.35, 0.08, (-0.002, 0.003))
    cpu = aligned_sin_tm(affine_input(*inputs[:2], remainder=inputs[2]))
    gpu = aligned_sin_tm(
        affine_input(*inputs[:2], remainder=inputs[2], device="cuda")
    )
    cpu_lo, cpu_hi = cpu.range_bound(context="aligned_cpu_containment")
    gpu_lo, gpu_hi = gpu.range_bound(context="aligned_cuda_containment")
    samples = torch.linspace(-0.432, -0.267, 2001, dtype=torch.float64)
    exact = torch.sin(samples)
    assert cpu_lo.item() <= exact.min().item() <= exact.max().item() <= cpu_hi.item()
    assert gpu_lo.item() <= exact.min().item() <= exact.max().item() <= gpu_hi.item()


@pytest.mark.cuda
def test_algorithm_aligned_compiled_contains_eager_or_falls_back_soundly() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    model = affine_input(-0.35, 0.08, remainder=(-0.002, 0.003), device="cuda")
    eager = aligned_sin_tm(model, point_enclosure_backend="eager")
    compiled = aligned_sin_tm(model, point_enclosure_backend="compiled")
    if not (
        torch.equal(eager.poly.coeffs, compiled.poly.coeffs)
        and torch.equal(eager.rem_lo, compiled.rem_lo)
        and torch.equal(eager.rem_hi, compiled.rem_hi)
    ):
        assert torch.all(compiled.rem_lo <= eager.rem_lo)
        assert torch.all(compiled.rem_hi >= eager.rem_hi)


@pytest.mark.regression
def test_algorithm_aligned_lane_does_not_modify_frozen_baseline_sources() -> None:
    for relative, expected in FROZEN_BASELINE_HASHES.items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected


@pytest.mark.regression
@pytest.mark.protocol
def test_algorithm_aligned_formal_artifact_passes_every_gate() -> None:
    import json

    summary = json.loads(
        (ALIGNED_OUTPUT / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["status"] == "PASS"
    assert summary["lane"] == "algorithm_aligned_q3"
    assert not summary["raw_paths_in_public_record"]
    assert [row["gate"] for row in summary["one_step"]["gates"]] == [
        "G0",
        "G1",
        "G2",
        "G3",
        "G4",
    ]
    assert all(
        row["status"] == "PASS" and row["accepted"]["equal"]
        for row in summary["one_step"]["gates"]
    )
    common = summary["common_control"]
    assert common["completed_segments"] == 200
    assert common["status"] == "PASS"
    assert common["accepted_status_equal_to_reference"]
    assert common["period_local_frozen_xiangru_input_restart"]
    assert not common["reference_outputs_used_as_native_inputs"]


@pytest.mark.unit
def test_algorithm_aligned_invalid_inputs_fail_closed() -> None:
    model = affine_input(0.0, 0.1)
    float32_model = BatchedTaylorModel(
        BatchedPolynomial(model.poly.coeffs.float(), model.poly.basis),
        model.rem_lo.float(),
        model.rem_hi.float(),
        model.domain_lo.float(),
        model.domain_hi.float(),
        DenseRemainderLedger.empty(),
    )
    with pytest.raises(TypeError, match="requires float64"):
        aligned_sin_tm(float32_model)
    with pytest.raises(ValueError, match="composition radius"):
        aligned_sin_tm(affine_input(0.0, 4.01))
    with pytest.raises(ValueError, match="K2 and ten rounds"):
        algorithm_aligned_q3_step(one_leaf_model(), remainder_rounds=9)
