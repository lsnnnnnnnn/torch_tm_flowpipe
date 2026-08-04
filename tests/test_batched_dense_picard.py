import pytest
import torch

from torch_tm_flowpipe import Interval, Polynomial, TaylorModel, TMVector
from torch_tm_flowpipe.batched_dense_tm import (
    BatchedTaylorModel,
    dense_picard_validate_step,
    sparse_tmvector_to_dense,
)


def _base_ext(center=0.25, radius=0.1, h=0.05, order=4):
    uncertainty_domain = [Interval(-1.0, 1.0)]
    x0 = TMVector(
        [
            TaylorModel(
                Polynomial({(0,): center, (1,): radius}, 1),
                Interval.zero(),
                uncertainty_domain,
                order=order,
            )
        ]
    )
    return sparse_tmvector_to_dense(x0.extend_domain(Interval(0.0, h)), order=order)


def test_constant_ode_picard_is_exact_affine_local_time():
    h = 0.05
    base = _base_ext(h=h)

    def constant_rhs(state):
        return BatchedTaylorModel.constants_like(2.0, state)

    result = dense_picard_validate_step(
        constant_rhs,
        base,
        h=h,
        order=4,
        tau_index=1,
        target_remainder_radius=1e-4,
        cutoff_threshold=1e-10,
    )

    assert result.accepted
    assert result.contract.local_time_semantics == "physical_[0,h]"
    assert result.contract.time_scale == "integration_only"
    assert result.raw_endpoint is not None
    tau_slot = result.segment_tm.poly.basis.term_index((0, 1))
    assert torch.allclose(result.segment_tm.poly.coeffs[0, 0, tau_slot], torch.tensor(2.0, dtype=torch.float64))
    endpoint_lo, endpoint_hi = result.raw_endpoint.range_bound()
    assert float(endpoint_lo) <= 0.25 - 0.1 + 2.0 * h
    assert float(endpoint_hi) >= 0.25 + 0.1 + 2.0 * h


def test_quadratic_picard_contains_higher_local_time_terms_and_is_not_euler():
    base = _base_ext(center=0.1, radius=0.02, h=0.01, order=4)

    def quadratic_rhs(state):
        x = state.component(0)
        return x.mul_trunc(x).add(1.0)

    result = dense_picard_validate_step(
        quadratic_rhs,
        base,
        h=0.01,
        order=4,
        tau_index=1,
        target_remainder_radius=1e-3,
        cutoff_threshold=1e-12,
    )
    assert result.accepted
    tau2 = result.segment_tm.poly.basis.term_index((0, 2))
    assert abs(float(result.segment_tm.poly.coeffs[0, 0, tau2])) > 0.0
    assert len([row for row in result.trace if row["phase"] == "polynomial_picard"]) == 4


def test_affine_ode_picard_endpoint_contains_high_accuracy_solution():
    h = 0.02
    base = _base_ext(center=0.2, radius=0.01, h=h, order=4)

    def affine_rhs(state):
        return state.scale(0.5).add(1.0)

    result = dense_picard_validate_step(
        affine_rhs,
        base,
        h=h,
        order=4,
        tau_index=1,
        target_remainder_radius=1e-4,
        cutoff_threshold=1e-12,
    )
    assert result.accepted and result.raw_endpoint is not None
    lo, hi = result.raw_endpoint.range_bound()
    for x0 in (0.19, 0.21):
        exact = (x0 + 2.0) * torch.exp(torch.tensor(0.5 * h, dtype=torch.float64)) - 2.0
        assert float(lo) <= float(exact) <= float(hi)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_true_picard_validation_cpu_cuda_parity():
    base_cpu = _base_ext(center=0.2, radius=0.01, h=0.01, order=4)

    def affine_rhs(state):
        return state.scale(0.5).add(1.0)

    cpu = dense_picard_validate_step(
        affine_rhs,
        base_cpu,
        h=0.01,
        order=4,
        tau_index=1,
        target_remainder_radius=1e-4,
        cutoff_threshold=1e-12,
    )
    cuda = dense_picard_validate_step(
        affine_rhs,
        base_cpu.to("cuda"),
        h=0.01,
        order=4,
        tau_index=1,
        target_remainder_radius=1e-4,
        cutoff_threshold=1e-12,
    )

    assert cpu.accepted and cuda.accepted
    assert cuda.segment_tm.poly.coeffs.device.type == "cuda"
    assert torch.allclose(cpu.segment_tm.poly.coeffs, cuda.segment_tm.poly.coeffs.cpu(), atol=2e-14, rtol=2e-14)
    assert torch.allclose(cpu.segment_tm.rem_lo, cuda.segment_tm.rem_lo.cpu(), atol=2e-14, rtol=2e-14)
    assert torch.allclose(cpu.segment_tm.rem_hi, cuda.segment_tm.rem_hi.cpu(), atol=2e-14, rtol=2e-14)
