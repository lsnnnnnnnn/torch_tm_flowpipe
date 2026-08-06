from __future__ import annotations

import math
from pathlib import Path
import subprocess

import pytest
import torch

from torch_tm_flowpipe.batched_dense_tm import (
    BatchedMonomialBasis,
    BatchedPolynomial,
    BatchedTaylorModel,
    DenseRemainderLedger,
    DenseRangePolicy,
)
from torch_tm_flowpipe.protocol.q3_audit import complete_total_degree_exponents
from torch_tm_flowpipe.tora_q3 import (
    build_tora_q3_box_model,
    build_tora_q3_initial_model,
    compose_tora_q3_tm,
    dense_tora_q3_dr_step,
    ToraQ3AffineCarry,
    tora_b48_boxes,
    tora_q3_rhs,
)


Q3_FINGERPRINT = "fa135259d41a68a73a6fc609880c4fd466bf2d53b2dddeba30298a484fa5e44d"


@pytest.mark.unit
def test_six_variable_complete_q3_basis_order_and_fingerprint() -> None:
    basis = BatchedMonomialBasis.build(6, 3, "cpu")
    actual = tuple(tuple(row) for row in basis.exponents.tolist())
    assert basis.num_terms == math.comb(9, 3) == 84
    assert actual == complete_total_degree_exponents(3, 6)
    assert len(set(actual)) == 84
    assert basis.fingerprint == Q3_FINGERPRINT
    assert actual[basis.term_index((1, 0, 0, 0, 0, 0))] == (1, 0, 0, 0, 0, 0)


@pytest.mark.unit
def test_b48_order_hull_adjacency_and_volume() -> None:
    lower, upper = tora_b48_boxes()
    assert lower.shape == upper.shape == (48, 4)
    assert torch.equal(lower.min(dim=0).values, torch.tensor([0.6, -0.7, -0.4, 0.5], dtype=torch.float64))
    assert torch.equal(upper.max(dim=0).values, torch.tensor([0.7, -0.6, -0.3, 0.6], dtype=torch.float64))
    widths = upper - lower
    assert torch.all(widths > 0.0)
    assert math.isclose(torch.prod(widths, dim=1).sum().item(), 1.0e-4, rel_tol=0.0, abs_tol=2e-19)
    # itertools.product order: x2 advances within each of the eight x1 slabs.
    assert torch.equal(lower[:6, 0], lower[0, 0].expand(6))
    assert torch.equal(lower[:6, 1], torch.linspace(-0.7, -0.6, 7, dtype=torch.float64)[:-1])
    assert torch.equal(lower[6, 1:], lower[0, 1:])
    assert lower[6, 0] == upper[0, 0]


@pytest.mark.unit
def test_six_variable_multiplication_and_time_integration_route_overflow() -> None:
    basis = BatchedMonomialBasis.build(6, 3, "cpu")
    domain_lo = torch.tensor([[0.0, -1.0, -1.0, -1.0, -1.0, -1.0]])
    domain_hi = torch.tensor([[0.1, 1.0, 1.0, 1.0, 1.0, 1.0]])
    cubic = BatchedPolynomial.zeros(1, 1, basis)
    cubic_coeffs = cubic.coeffs.clone()
    cubic_coeffs[..., basis.term_index((0, 3, 0, 0, 0, 0))] = 2.0
    cubic = BatchedPolynomial(cubic_coeffs, basis)
    linear = BatchedPolynomial.zeros(1, 1, basis)
    linear_coeffs = linear.coeffs.clone()
    linear_coeffs[..., basis.term_index((0, 1, 0, 0, 0, 0))] = 3.0
    linear = BatchedPolynomial(linear_coeffs, basis)
    product, product_lo, product_hi = cubic.mul_trunc(
        linear,
        return_truncation_bound=True,
        domain_lo=domain_lo,
        domain_hi=domain_hi,
    )
    assert torch.count_nonzero(product.coeffs) == 0
    assert product_lo.item() <= 0.0
    assert product_hi.item() >= 6.0

    time_cubic = BatchedPolynomial.zeros(1, 1, basis)
    time_coeffs = time_cubic.coeffs.clone()
    time_coeffs[..., basis.term_index((3, 0, 0, 0, 0, 0))] = 4.0
    time_cubic = BatchedPolynomial(time_coeffs, basis)
    integral, overflow_lo, overflow_hi = time_cubic.integrate(
        0,
        return_overflow_bound=True,
        domain_lo=domain_lo,
        domain_hi=domain_hi,
    )
    assert torch.count_nonzero(integral.coeffs) == 0
    assert overflow_lo.item() <= 0.0
    assert overflow_hi.item() >= 1.0e-4


def _fixed_control(batch: int, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.full((batch,), 9.8, dtype=torch.float64, device=device),
        torch.full((batch,), 10.2, dtype=torch.float64, device=device),
    )


@pytest.mark.unit
def test_affine_composition_materializes_local_spatial_coordinates() -> None:
    basis = BatchedMonomialBasis.build(6, 3, "cpu")
    coefficients = torch.zeros((1, 1, basis.num_terms), dtype=torch.float64)
    local_slot = basis.term_index((0, 0, 0, 0, 0, 1))
    global_slot = basis.term_index((0, 1, 0, 0, 0, 0))
    coefficients[:, :, local_slot] = 0.2
    domain_lower = torch.tensor(
        [[0.0, -1.0, -1.0, -1.0, -1.0, -1.0]], dtype=torch.float64
    )
    domain_upper = torch.tensor(
        [[0.1, 1.0, 1.0, 1.0, 1.0, 1.0]], dtype=torch.float64
    )
    zeros = torch.zeros((1, 1), dtype=torch.float64)
    local = BatchedTaylorModel(
        BatchedPolynomial(coefficients, basis),
        zeros,
        zeros.clone(),
        domain_lower,
        domain_upper,
        DenseRemainderLedger.empty(),
        DenseRangePolicy(),
    )
    linear = torch.zeros((1, 5, 5), dtype=torch.float64)
    linear[:, 4, 0] = 0.5
    carry_zeros = torch.zeros((1, 5), dtype=torch.float64)
    composed = compose_tora_q3_tm(
        local, ToraQ3AffineCarry(linear, carry_zeros, carry_zeros.clone())
    )
    assert composed.poly.coeffs[0, 0, global_slot] == pytest.approx(0.1)
    assert composed.poly.coeffs[0, 0, local_slot] == 0.0
    lower, upper = composed.range_bound(context="composition_regression")
    assert lower.item() <= -0.1
    assert upper.item() >= 0.1
    assert upper.item() - lower.item() < 0.200000000001


@pytest.mark.unit
def test_exact_time_endpoint_aggregates_equal_spatial_exponents_soundly() -> None:
    basis = BatchedMonomialBasis.build(6, 3, "cpu")
    coefficients = torch.zeros((1, 1, basis.num_terms), dtype=torch.float64)
    coefficients[:, :, basis.term_index((0, 1, 0, 0, 0, 0))] = 1.0
    coefficients[:, :, basis.term_index((1, 1, 0, 0, 0, 0))] = -10.0
    domain_lower = torch.tensor(
        [[0.0, -1.0, -1.0, -1.0, -1.0, -1.0]], dtype=torch.float64
    )
    domain_upper = torch.tensor(
        [[0.1, 1.0, 1.0, 1.0, 1.0, 1.0]], dtype=torch.float64
    )
    zeros = torch.zeros((1, 1), dtype=torch.float64)
    model = BatchedTaylorModel(
        BatchedPolynomial(coefficients, basis),
        zeros,
        zeros.clone(),
        domain_lower,
        domain_upper,
        DenseRemainderLedger.empty(),
        DenseRangePolicy(),
    )
    endpoint = model.endpoint(0, 0.1)
    lower, upper = endpoint.range_bound(context="endpoint_cancellation_regression")
    assert lower.item() <= 0.0 <= upper.item()
    assert upper.item() - lower.item() < 1.0e-12


@pytest.mark.integration
def test_tora_one_step_fixed_u_validates_and_separates_tube_from_endpoint() -> None:
    lo, hi = _fixed_control(48)
    step = dense_tora_q3_dr_step(build_tora_q3_initial_model(lo, hi))
    assert step.status == "validated"
    assert int(step.accepted_by_leaf.sum()) == 48
    assert step.endpoint_lower.shape == step.tube_lower.shape == (48, 5)
    assert torch.all(step.tube_lower <= step.endpoint_lower)
    assert torch.all(step.tube_upper >= step.endpoint_upper)
    assert torch.equal(step.segment_tm.rem_lo[:, 4], torch.zeros(48))
    assert torch.equal(step.segment_tm.rem_hi[:, 4], torch.zeros(48))


@pytest.mark.integration
def test_configurable_k2_and_k3_are_distinct_validated_picard_contracts() -> None:
    state_lo, state_hi = tora_b48_boxes()
    control_lo, control_hi = _fixed_control(1)
    base = build_tora_q3_box_model(
        state_lo[:1], state_hi[:1], control_lo, control_hi
    )
    k2 = dense_tora_q3_dr_step(base, polynomial_picard_rounds=2)
    k3 = dense_tora_q3_dr_step(base, polynomial_picard_rounds=3)
    assert k2.accepted and k3.accepted
    assert len(k2.polynomial_trace) == 2
    assert len(k3.polynomial_trace) == 3
    assert k2.polynomial_trace[-1]["coefficient_sha256"] != k3.polynomial_trace[-1]["coefficient_sha256"]
    assert k3.polynomial_trace[-1]["iteration"] == 3


@pytest.mark.integration
def test_b48_batch_matches_representative_per_leaf_cpu_runs() -> None:
    state_lo, state_hi = tora_b48_boxes()
    control_lo, control_hi = _fixed_control(48)
    batch = dense_tora_q3_dr_step(
        build_tora_q3_box_model(state_lo, state_hi, control_lo, control_hi)
    )
    assert batch.accepted
    for leaf in (0, 23, 47):
        individual = dense_tora_q3_dr_step(
            build_tora_q3_box_model(
                state_lo[leaf : leaf + 1],
                state_hi[leaf : leaf + 1],
                control_lo[leaf : leaf + 1],
                control_hi[leaf : leaf + 1],
            )
        )
        assert individual.accepted
        assert torch.equal(batch.accepted_by_leaf[leaf : leaf + 1], individual.accepted_by_leaf)
        assert torch.allclose(batch.endpoint_lower[leaf], individual.endpoint_lower[0], rtol=0.0, atol=2e-14)
        assert torch.allclose(batch.endpoint_upper[leaf], individual.endpoint_upper[0], rtol=0.0, atol=2e-14)
        assert torch.allclose(batch.tube_lower[leaf], individual.tube_lower[0], rtol=0.0, atol=2e-14)
        assert torch.allclose(batch.tube_upper[leaf], individual.tube_upper[0], rtol=0.0, atol=2e-14)


@pytest.mark.cuda
@pytest.mark.integration
def test_tora_one_leaf_cpu_cuda_enclosure_parity() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    state_lo, state_hi = tora_b48_boxes()
    control_lo, control_hi = _fixed_control(1)
    cpu = dense_tora_q3_dr_step(
        build_tora_q3_box_model(state_lo[:1], state_hi[:1], control_lo, control_hi)
    )
    gpu = dense_tora_q3_dr_step(
        build_tora_q3_box_model(
            state_lo[:1].cuda(), state_hi[:1].cuda(), control_lo.cuda(), control_hi.cuda(), device="cuda"
        )
    )
    assert cpu.accepted and gpu.accepted
    for cpu_value, gpu_value in (
        (cpu.endpoint_lower, gpu.endpoint_lower),
        (cpu.endpoint_upper, gpu.endpoint_upper),
        (cpu.tube_lower, gpu.tube_lower),
        (cpu.tube_upper, gpu.tube_upper),
    ):
        assert torch.allclose(cpu_value, gpu_value.cpu(), rtol=0.0, atol=2e-12)


@pytest.mark.unit
def test_tora_held_control_rhs_is_exact_zero() -> None:
    state_lo, state_hi = tora_b48_boxes()
    lo, hi = _fixed_control(1)
    rhs = tora_q3_rhs(build_tora_q3_box_model(state_lo[:1], state_hi[:1], lo, hi))
    assert torch.count_nonzero(rhs.poly.coeffs[:, 4]) == 0
    assert torch.equal(rhs.rem_lo[:, 4], torch.zeros(1))
    assert torch.equal(rhs.rem_hi[:, 4], torch.zeros(1))


@pytest.mark.unit
def test_portable_q3_unit_test_has_no_server_path_literal() -> None:
    source = (Path(__file__).with_name("test_xiangru_q3_matched_audit.py")).read_text(encoding="utf-8")
    server_prefix = "/" + "srv/local/"
    assert server_prefix not in source


@pytest.mark.unit
def test_no_new_private_binary_is_tracked_in_native_tora_deliverables() -> None:
    root = Path(__file__).resolve().parents[1]
    tracked = subprocess.run(
        ["git", "ls-files", "-z"], cwd=root, check=True, capture_output=True
    ).stdout.decode().split("\0")
    sensitive_suffixes = {".onnx", ".pt", ".pth", ".ckpt", ".safetensors"}
    native = [
        path for path in tracked
        if path.startswith("outputs/tora_q3_native_matched_20260806/")
        and Path(path).suffix.lower() in sensitive_suffixes
    ]
    assert native == []
