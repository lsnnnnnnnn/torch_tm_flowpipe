from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import pytest
import torch

from torch_tm_flowpipe.tora_algorithm_aligned import algorithm_aligned_q3_step
from torch_tm_flowpipe.batched_dense_tm import dense_validation_batch
from torch_tm_flowpipe.tora_fused_kernel import (
    _segmented_execute,
    compose_fused_tora_q3_step,
    fused_algorithm_aligned_q3_step,
    fused_finalize_kernel,
    fused_full_step_kernel,
    fused_kernel_status,
    fused_natural_range_kernel,
    fused_polynomial_picard_kernel,
    fused_remainder_initialize_kernel,
    fused_remainder_round_kernel,
    fused_tora_q3_boundary_from_model,
    run_fused_full_step,
    run_segmented_fused_step,
    tora_q3_kernel_metadata,
)
from torch_tm_flowpipe.tora_q3 import (
    build_tora_q3_box_model,
    identity_tora_q3_carry,
    tora_b48_boxes,
)


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixed_model(batch: int, device: str = "cpu", width: float = 0.4):
    lower, upper = tora_b48_boxes(device=device)
    repeats = (batch + 47) // 48
    lower = lower.repeat((repeats, 1))[:batch]
    upper = upper.repeat((repeats, 1))[:batch]
    control_lower = torch.full(
        (batch,), 10.0 - width / 2.0, dtype=torch.float64, device=device
    )
    control_upper = torch.full(
        (batch,), 10.0 + width / 2.0, dtype=torch.float64, device=device
    )
    return build_tora_q3_box_model(
        lower,
        upper,
        control_lower,
        control_upper,
        device=device,
    )


@pytest.mark.unit
def test_f1_natural_range_outwardly_contains_object_reference() -> None:
    base = fixed_model(3)
    metadata = tora_q3_kernel_metadata("cpu")
    lower, upper = fused_natural_range_kernel(
        base.poly.coeffs,
        base.domain_lo,
        base.domain_hi,
        metadata.exponents,
    )
    reference_lo, reference_hi = base.poly.range_bound(
        base.domain_lo, base.domain_hi
    )
    assert torch.all(lower <= reference_lo)
    assert torch.all(upper >= reference_hi)


@pytest.mark.integration
@pytest.mark.parametrize("batch", [1, 48])
def test_fused_eager_is_sound_outer_lane_with_exact_coefficients(batch: int) -> None:
    base = fixed_model(batch)
    fused = fused_algorithm_aligned_q3_step(base, backend="eager")
    reference = algorithm_aligned_q3_step(base, capture_trace=False)
    assert fused.accepted and reference.accepted
    assert torch.equal(
        fused.segment_tm.poly.coeffs, reference.segment_tm.poly.coeffs
    )
    for fused_lo, fused_hi, reference_lo, reference_hi in (
        (
            fused.segment_tm.rem_lo,
            fused.segment_tm.rem_hi,
            reference.segment_tm.rem_lo,
            reference.segment_tm.rem_hi,
        ),
        (
            fused.endpoint_lower,
            fused.endpoint_upper,
            reference.endpoint_lower,
            reference.endpoint_upper,
        ),
        (
            fused.tube_lower,
            fused.tube_upper,
            reference.tube_lower,
            reference.tube_upper,
        ),
    ):
        assert torch.all(fused_lo <= reference_lo)
        assert torch.all(fused_hi >= reference_hi)
    assert torch.equal(fused.segment_tm.rem_lo[:, 4], torch.zeros(batch))
    assert torch.equal(fused.segment_tm.rem_hi[:, 4], torch.zeros(batch))


@pytest.mark.regression
def test_segmented_eager_is_bitwise_equal_to_f5_eager() -> None:
    base = fixed_model(7)
    monolithic, monolithic_backend = run_fused_full_step(base, backend="eager")
    segmented, segmented_backend = run_segmented_fused_step(base, backend="eager")
    assert monolithic_backend == "eager"
    assert segmented_backend == "segmented_eager"
    assert all(
        torch.equal(left, right)
        for left, right in zip(monolithic, segmented, strict=True)
    )


@pytest.mark.property
def test_fused_b48_batch_matches_representative_individual_leaves() -> None:
    batch = fused_algorithm_aligned_q3_step(fixed_model(48), backend="eager")
    assert batch.accepted
    lower, upper = tora_b48_boxes()
    for leaf in (0, 23, 47):
        individual_base = build_tora_q3_box_model(
            lower[leaf : leaf + 1],
            upper[leaf : leaf + 1],
            torch.tensor([9.8], dtype=torch.float64),
            torch.tensor([10.2], dtype=torch.float64),
        )
        individual = fused_algorithm_aligned_q3_step(
            individual_base, backend="eager"
        )
        assert individual.accepted
        assert torch.equal(
            batch.segment_tm.poly.coeffs[leaf],
            individual.segment_tm.poly.coeffs[0],
        )
        assert torch.allclose(
            batch.endpoint_lower[leaf],
            individual.endpoint_lower[0],
            rtol=0.0,
            atol=2e-14,
        )
        assert torch.allclose(
            batch.endpoint_upper[leaf],
            individual.endpoint_upper[0],
            rtol=0.0,
            atol=2e-14,
        )


@pytest.mark.unit
def test_pure_tensor_boundary_has_no_object_or_host_sync_operations() -> None:
    functions = (
        fused_natural_range_kernel,
        fused_polynomial_picard_kernel,
        fused_remainder_initialize_kernel,
        fused_remainder_round_kernel,
        fused_finalize_kernel,
        fused_full_step_kernel,
        _segmented_execute,
    )
    forbidden = (
        ".item(",
        "bool(",
        ".cpu(",
        ".numpy(",
        ".tolist(",
        "json.",
        "hashlib.",
        "BatchedTaylorModel(",
        "DenseRemainderLedger(",
    )
    for function in functions:
        source = inspect.getsource(function)
        assert not any(token in source for token in forbidden), function.__name__


@pytest.mark.unit
def test_fused_invalid_signature_fails_closed_or_uses_sound_cpu_fallback() -> None:
    base = fixed_model(1)
    with pytest.raises(ValueError, match="backend"):
        run_segmented_fused_step(base, backend="invalid")
    output, selected = run_segmented_fused_step(base, backend="compiled")
    assert selected == "segmented_eager"
    assert bool(output[11].all())
    float32 = type(base)(
        type(base.poly)(base.poly.coeffs.float(), base.poly.basis),
        base.rem_lo.float(),
        base.rem_hi.float(),
        base.domain_lo.float(),
        base.domain_hi.float(),
    )
    with pytest.raises(TypeError, match="requires float64"):
        fused_algorithm_aligned_q3_step(float32)


@pytest.mark.integration
def test_batched_fail_closed_wrapper_consolidates_valid_acceptance() -> None:
    base = fixed_model(3)
    with dense_validation_batch():
        local = fused_algorithm_aligned_q3_step(
            base,
            backend="eager",
            batched_fail_closed=True,
        )
        physical = compose_fused_tora_q3_step(
            local,
            identity_tora_q3_carry(3, device="cpu"),
        )
    assert local.accepted
    assert physical.accepted


@pytest.mark.cuda
@pytest.mark.slow
@pytest.mark.integration
def test_segmented_compiled_is_outward_for_two_b48_inputs() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    with torch.no_grad():
        first = fixed_model(48, "cuda", width=0.4)
        compiled_first, selected = run_segmented_fused_step(
            first, backend="compiled"
        )
        eager_first, _ = run_segmented_fused_step(first, backend="eager")
        assert selected == "segmented_compiled_verified"

        second = fixed_model(48, "cuda", width=0.3)
        compiled_second, selected = run_segmented_fused_step(
            second, backend="compiled"
        )
        eager_second, _ = run_segmented_fused_step(second, backend="eager")
        assert selected == "segmented_compiled_verified"
    for compiled, eager in (
        (compiled_first, eager_first),
        (compiled_second, eager_second),
    ):
        assert torch.equal(compiled[0], eager[0])
        for lower_index, upper_index in ((1, 2), (3, 4), (5, 6)):
            assert torch.all(compiled[lower_index] <= eager[lower_index])
            assert torch.all(compiled[upper_index] >= eager[upper_index])
        for predicate_index in range(7, 12):
            assert torch.all(
                (~compiled[predicate_index]) | eager[predicate_index]
            )
    status = fused_kernel_status()
    assert status["segmented_verified_signatures"]
    verification = next(iter(status["segmented_verified_signatures"].values()))
    assert verification["fullgraph_stage_count"] == 4
    assert verification["per_round_host_decisions"] == 0
    assert verification["outward_contains_eager"]


@pytest.mark.cuda
@pytest.mark.integration
def test_fused_eager_cpu_cuda_predicates_are_equal() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    cpu, _ = run_segmented_fused_step(fixed_model(48), backend="eager")
    cuda, _ = run_segmented_fused_step(
        fixed_model(48, "cuda"), backend="eager"
    )
    for predicate_index in range(7, 12):
        assert torch.equal(cpu[predicate_index], cuda[predicate_index].cpu())


@pytest.mark.formal_run
@pytest.mark.protocol
def test_formal_fused_runtime_artifacts_are_current_and_pass() -> None:
    output = (
        ROOT
        / "outputs"
        / "tora_q3_stage_parity_fused_20260809"
        / "fused_kernel"
    )
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "PASS"
    assert summary["grad_enabled"] is False
    assert summary["common_control_t20"]["completed_segments_each"] == [200] * 5
    assert summary["common_control_t20"]["checksum_stable"] is True
    assert summary["one_step"]["runtime"]["repeat_count"] == 10
    assert summary["common_control_t20"]["runtime"]["repeat_count"] == 5
    assert summary["gates"]["P0_correctness_soundness"] == "PASS"
    for gate in ("P1_graph_breaks", "P2_program_sync", "P3_aten_to", "P4_b48_one_step", "P5_common_control_t20"):
        assert summary["gates"][gate]["status"].startswith("PASS")
    assert summary["gates"]["P5_common_control_t20"]["stretch_status"] == "MISS"
    for relative, expected in summary["source_sha256"].items():
        assert sha256(ROOT / relative) == expected

    dispatch = json.loads(
        (output / "program_dispatch_all_lanes.json").read_text(encoding="utf-8")
    )
    assert {
        lane: payload["program_issued_host_scalar_sync_count"]
        for lane, payload in dispatch["lanes"].items()
    } == {
        "algorithm_aligned_q3": 4,
        "baseline_native_k2": 4,
        "fused_segmented": 1,
    }
