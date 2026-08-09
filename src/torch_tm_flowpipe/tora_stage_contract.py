"""Offline stage observations for the frozen TORA-Q3 plant contract.

The functions in this module never call an external implementation.  They
replay the native Torch local step from explicit tensors and expose the
mathematical boundaries needed by the stage comparator.  Raw tensor trees are
intended for private evidence; public callers should retain only hashes and
aggregates.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Mapping

import torch

from .batched_dense_tm import (
    BatchedMonomialBasis,
    BatchedPolynomial,
    BatchedTaylorModel,
    DenseRemainderLedger,
    _interval_add,
    _interval_div_positive_integer,
    _point_sin_cos_enclosure,
    _positive_power_over_factorial,
    _sound_add_tm,
    _sound_mul_tm,
    _sound_scale_tm_interval,
    _subset_margin,
    dense_polynomial_picard,
    sin_tm,
)
from .tora_q3 import (
    ToraQ3AffineCarry,
    _endpoint_bounds,
    _zero_exact_held_remainder,
    compose_tora_q3_step,
    dense_tora_q3_dr_step,
    tora_q3_rhs,
)


STAGE_IDS = tuple(f"A{index}" for index in range(13))
SELECTED_SEGMENTS = (1, 2, 10, 40, 43, 44, 45)
REPLAY_POINTS: Mapping[str, tuple[int, ...]] = {
    "S0": (1,),
    "S1": (2,),
    "R1": (10,),
    "R2": (40,),
    "F0": (43, 44, 45),
}


@dataclass(frozen=True)
class TorchStageObservation:
    segment_index: int
    stages: Mapping[str, Any]
    local_step: Any
    physical_step: Any
    replay_equivalence: Mapping[str, Any]


def tensor_sha256(*values: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for value in values:
        tensor = value.detach().cpu().contiguous()
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def tensor_tree_to_lists(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): tensor_tree_to_lists(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [tensor_tree_to_lists(item) for item in value]
    if isinstance(value, list):
        return [tensor_tree_to_lists(item) for item in value]
    return value


def _interval_payload(lower: torch.Tensor, upper: torch.Tensor) -> dict[str, torch.Tensor]:
    return {"lower": lower, "upper": upper}


def _tm_payload(model: BatchedTaylorModel) -> dict[str, Any]:
    return {
        "polynomial": {"coefficients": model.poly.coeffs},
        "remainder": _interval_payload(model.rem_lo, model.rem_hi),
        "polynomial_range": _interval_payload(
            *model.poly.range_bound(model.domain_lo, model.domain_hi)
        ),
    }


def _maximum_abs(pairs: tuple[tuple[torch.Tensor, torch.Tensor], ...]) -> float:
    maxima = [torch.max(torch.abs(left - right)) for left, right in pairs]
    return float(torch.max(torch.stack(maxima)).detach().cpu())


def _bitwise(pairs: tuple[tuple[torch.Tensor, torch.Tensor], ...]) -> bool:
    return all(torch.equal(left, right) for left, right in pairs)


def _observe_sine(
    model: BatchedTaylorModel,
    *,
    order: int,
    point_enclosure_backend: str,
) -> tuple[BatchedTaylorModel, dict[str, Any]]:
    degree = int(order)
    constant = model.poly.coeffs[..., model.poly.basis.constant_index]
    sin_lo, sin_hi, cos_lo, cos_hi = _point_sin_cos_enclosure(
        constant,
        series_terms=32,
        maximum_abs_center=8.0,
        backend=point_enclosure_backend,
    )
    delta_coeffs = model.poly.coeffs.clone()
    delta_coeffs[..., model.poly.basis.constant_index] = 0.0
    delta = BatchedTaylorModel(
        BatchedPolynomial(delta_coeffs, model.poly.basis),
        model.rem_lo,
        model.rem_hi,
        model.domain_lo,
        model.domain_hi,
        model.ledger,
        model.range_policy,
        model.range_trace,
    )
    delta_lo, delta_hi = delta.range_bound(context="sine_delta")
    delta_radius = torch.maximum(torch.abs(delta_lo), torch.abs(delta_hi))
    constant_poly = BatchedPolynomial.constants(
        sin_lo + 0.5 * (sin_hi - sin_lo), model.poly.basis
    )
    constant_mid = constant_poly.coeffs[..., model.poly.basis.constant_index]
    constant_error_lo = torch.nextafter(
        sin_lo - constant_mid, torch.full_like(sin_lo, -torch.inf)
    )
    constant_error_hi = torch.nextafter(
        sin_hi - constant_mid, torch.full_like(sin_hi, torch.inf)
    )
    pre_tail = BatchedTaylorModel(
        constant_poly,
        constant_error_lo,
        constant_error_hi,
        model.domain_lo,
        model.domain_hi,
        DenseRemainderLedger.empty().add(
            "composition_overflow", constant_error_lo, constant_error_hi
        ),
        model.range_policy,
        model.range_trace,
    )
    powers = [delta]
    for _power in range(2, degree + 1):
        powers.append(_sound_mul_tm(powers[-1], delta))
    coefficient_intervals: list[tuple[torch.Tensor, torch.Tensor]] = []
    if degree >= 1:
        coefficient_intervals.append((cos_lo, cos_hi))
    if degree >= 2:
        coefficient_intervals.append(
            _interval_div_positive_integer(-sin_hi, -sin_lo, 2)
        )
    if degree >= 3:
        coefficient_intervals.append(
            _interval_div_positive_integer(-cos_hi, -cos_lo, 6)
        )
    for power_model, (coefficient_lo, coefficient_hi) in zip(
        powers[:degree], coefficient_intervals, strict=True
    ):
        pre_tail = _sound_add_tm(
            pre_tail,
            _sound_scale_tm_interval(
                power_model, coefficient_lo, coefficient_hi
            ),
        )
    tail_radius = _positive_power_over_factorial(delta_radius, degree + 1)
    result_lo, result_hi = _interval_add(
        pre_tail.rem_lo,
        pre_tail.rem_hi,
        -tail_radius,
        tail_radius,
    )
    replayed = BatchedTaylorModel(
        pre_tail.poly,
        result_lo,
        result_hi,
        model.domain_lo,
        model.domain_hi,
        pre_tail.ledger.add(
            "composition_overflow", -tail_radius, tail_radius
        ),
        model.range_policy,
        model.range_trace,
    )
    reference = sin_tm(
        model,
        order=degree,
        point_enclosure_backend=point_enclosure_backend,
    )
    pairs = (
        (replayed.poly.coeffs, reference.poly.coeffs),
        (replayed.rem_lo, reference.rem_lo),
        (replayed.rem_hi, reference.rem_hi),
    )
    return reference, {
        "input": _tm_payload(model),
        "constant": constant,
        "point_sine": _interval_payload(sin_lo, sin_hi),
        "point_cosine": _interval_payload(cos_lo, cos_hi),
        "delta_range": _interval_payload(delta_lo, delta_hi),
        "retained_polynomial": {"coefficients": pre_tail.poly.coeffs},
        "composition_overflow": _interval_payload(
            pre_tail.rem_lo, pre_tail.rem_hi
        ),
        "analytic_remainder": _interval_payload(-tail_radius, tail_radius),
        "output": _tm_payload(reference),
        "replay_equivalence": {
            "bitwise": _bitwise(pairs),
            "maximum_absolute_error": _maximum_abs(pairs),
        },
    }


def _observe_rhs(
    model: BatchedTaylorModel,
    *,
    sine_order: int,
    point_enclosure_backend: str,
) -> tuple[BatchedTaylorModel, dict[str, Any]]:
    sine, sine_observation = _observe_sine(
        model.component(2),
        order=sine_order,
        point_enclosure_backend=point_enclosure_backend,
    )
    result = tora_q3_rhs(
        model,
        sine_order=sine_order,
        point_enclosure_backend=point_enclosure_backend,
    )
    return result, {
        "sine": sine_observation,
        "sine_output_replayed": _tm_payload(sine),
        "rhs": _tm_payload(result),
    }


def model_and_carry_from_xiangru_record(
    record: Mapping[str, Any],
    *,
    device: torch.device | str,
) -> tuple[BatchedTaylorModel, ToraQ3AffineCarry]:
    stage = record.get("stage_contract")
    if not isinstance(stage, Mapping):
        raise ValueError("Xiangru record is missing stage_contract")
    a0 = stage.get("A0_normalized_input")
    if not isinstance(a0, Mapping):
        raise ValueError("Xiangru stage contract is missing A0")
    coefficients = torch.as_tensor(
        a0["polynomial"]["coefficients"],
        dtype=torch.float64,
        device=device,
    )
    rem_lo = torch.as_tensor(
        a0["remainder"]["lower"], dtype=torch.float64, device=device
    )
    rem_hi = torch.as_tensor(
        a0["remainder"]["upper"], dtype=torch.float64, device=device
    )
    if coefficients.ndim != 3 or coefficients.shape[1:] != (5, 84):
        raise ValueError("Xiangru A0 coefficient shape is not [B,5,84]")
    batch = coefficients.shape[0]
    basis = BatchedMonomialBasis.build(6, 3, str(torch.device(device)))
    domain_lo = torch.full(
        (batch, 6), -1.0, dtype=torch.float64, device=device
    )
    domain_hi = torch.ones_like(domain_lo)
    domain_lo[:, 0] = 0.0
    domain_hi[:, 0] = 0.1
    model = BatchedTaylorModel(
        BatchedPolynomial(coefficients, basis),
        rem_lo,
        rem_hi,
        domain_lo,
        domain_hi,
        DenseRemainderLedger.empty(),
    )
    normalized_map = record["normalization"]["normalized_map"]
    full_linear = torch.as_tensor(
        normalized_map["polynomial"]["L"],
        dtype=torch.float64,
        device=device,
    )
    if full_linear.shape != (batch, 5, 6):
        raise ValueError("Xiangru normalized-map L shape is not [B,5,6]")
    if torch.count_nonzero(full_linear[:, :, 0]).item() != 0:
        raise ValueError("Xiangru normalized-map carry contains local time")
    carry = ToraQ3AffineCarry(
        full_linear[:, :, 1:],
        torch.as_tensor(
            normalized_map["remainder"]["lower"],
            dtype=torch.float64,
            device=device,
        ),
        torch.as_tensor(
            normalized_map["remainder"]["upper"],
            dtype=torch.float64,
            device=device,
        ),
    )
    return model, carry


def model_from_xiangru_tm_payload(
    payload: Mapping[str, Any],
    *,
    device: torch.device | str,
) -> BatchedTaylorModel:
    """Reconstruct an observed Xiangru TM as a diagnostic Torch input."""
    coefficients = torch.as_tensor(
        payload["polynomial"]["coefficients"],
        dtype=torch.float64,
        device=device,
    )
    rem_lo = torch.as_tensor(
        payload["remainder"]["lower"], dtype=torch.float64, device=device
    )
    rem_hi = torch.as_tensor(
        payload["remainder"]["upper"], dtype=torch.float64, device=device
    )
    if coefficients.ndim != 3 or coefficients.shape[-1] != 84:
        raise ValueError("observed Q3 coefficients must have shape [B,D,84]")
    if rem_lo.shape != coefficients.shape[:2] or rem_hi.shape != rem_lo.shape:
        raise ValueError("observed remainder shape does not match coefficients")
    basis = BatchedMonomialBasis.build(6, 3, str(torch.device(device)))
    domain_lo = torch.full(
        (coefficients.shape[0], 6),
        -1.0,
        dtype=torch.float64,
        device=device,
    )
    domain_hi = torch.ones_like(domain_lo)
    domain_lo[:, 0] = 0.0
    domain_hi[:, 0] = 0.1
    return BatchedTaylorModel(
        BatchedPolynomial(coefficients, basis),
        rem_lo,
        rem_hi,
        domain_lo,
        domain_hi,
        DenseRemainderLedger.empty(),
    )


def observe_torch_sine_from_xiangru_payload(
    payload: Mapping[str, Any],
    *,
    device: torch.device | str,
    order: int = 2,
    point_enclosure_backend: str = "eager",
) -> Mapping[str, Any]:
    """Apply native Torch sine to an exact observed Xiangru TM input.

    This is a diagnostic counterfactual only.  It is deliberately separate
    from the formal propagation path.
    """
    model = model_from_xiangru_tm_payload(payload, device=device)
    _output, observation = _observe_sine(
        model,
        order=order,
        point_enclosure_backend=point_enclosure_backend,
    )
    return observation


def observe_torch_integration_from_xiangru_payload(
    payload: Mapping[str, Any],
    *,
    device: torch.device | str,
) -> Mapping[str, Any]:
    """Apply native Torch time integration to an exact Xiangru TM input."""
    model = model_from_xiangru_tm_payload(payload, device=device)
    output = model.integrate(0)
    return {"input": _tm_payload(model), "output": _tm_payload(output)}


def observe_torch_local_step(
    base: BatchedTaylorModel,
    carry: ToraQ3AffineCarry,
    *,
    segment_index: int,
    sine_order: int = 2,
    polynomial_picard_rounds: int = 2,
    remainder_rounds: int = 10,
    seed: float = 0.01,
    point_enclosure_backend: str = "eager",
) -> TorchStageObservation:
    if polynomial_picard_rounds != 2 or remainder_rounds != 10:
        raise ValueError("stage contract freezes K2 and ten remainder rounds")
    reference = dense_tora_q3_dr_step(
        base,
        sine_order=sine_order,
        polynomial_picard_rounds=polynomial_picard_rounds,
        remainder_rounds=remainder_rounds,
        seed=seed,
        capture_trace=True,
        point_enclosure_backend=point_enclosure_backend,
    )
    base_polynomial = base.without_remainder()
    polynomial_current = base_polynomial
    polynomial_rows: list[dict[str, Any]] = []
    for iteration in range(1, polynomial_picard_rounds + 1):
        rhs, rhs_observation = _observe_rhs(
            polynomial_current,
            sine_order=sine_order,
            point_enclosure_backend=point_enclosure_backend,
        )
        integrated = rhs.integrate(0)
        picard = base_polynomial.add(integrated)
        zeros = torch.zeros_like(picard.rem_lo)
        polynomial_current = BatchedTaylorModel(
            picard.poly,
            zeros,
            zeros.clone(),
            picard.domain_lo,
            picard.domain_hi,
            DenseRemainderLedger.empty(),
            picard.range_policy,
            picard.range_trace,
        ).apply_cutoff(None)
        polynomial_rows.append(
            {
                "iteration": iteration,
                "rhs": rhs_observation,
                "integration": _tm_payload(integrated),
                "candidate": _tm_payload(polynomial_current),
            }
        )

    canonical_candidate, _canonical_trace = dense_polynomial_picard(
        lambda value: tora_q3_rhs(
            value,
            sine_order=sine_order,
            point_enclosure_backend=point_enclosure_backend,
        ),
        base_polynomial,
        tau_index=0,
        order=3,
        iterations=polynomial_picard_rounds,
        cutoff_threshold=None,
        capture_trace=False,
    )
    polynomial_rows[-1]["candidate_replay_equivalence"] = {
        "bitwise": torch.equal(
            polynomial_current.poly.coeffs,
            canonical_candidate.poly.coeffs,
        ),
        "maximum_absolute_error": float(
            torch.max(
                torch.abs(
                    polynomial_current.poly.coeffs
                    - canonical_candidate.poly.coeffs
                )
            ).detach().cpu()
        ),
    }
    polynomial_current = canonical_candidate

    seed_vector = torch.tensor(
        [seed, seed, seed, seed, 0.0],
        dtype=base.poly.coeffs.dtype,
        device=base.poly.coeffs.device,
    ).view(1, 5).expand(base.poly.batch, -1)
    current = polynomial_current.with_remainder(-seed_vector, seed_vector)
    initial_rhs, initial_rhs_observation = _observe_rhs(
        current,
        sine_order=sine_order,
        point_enclosure_backend=point_enclosure_backend,
    )
    initial_integration = initial_rhs.integrate(0)
    initial_image = _zero_exact_held_remainder(
        base_polynomial.add(initial_integration)
    )
    initial_margin = _subset_margin(
        -seed_vector,
        seed_vector,
        initial_image.rem_lo,
        initial_image.rem_hi,
    )
    difference = BatchedPolynomial(
        initial_image.poly.coeffs - current.poly.coeffs,
        current.poly.basis,
    )
    roundoff_lo, roundoff_hi = difference.range_bound(
        base.domain_lo,
        base.domain_hi,
        policy=base.range_policy,
        context="tora_picard_roundoff",
        trace=base.range_trace,
    )
    roundoff_lo[:, 4] = 0.0
    roundoff_hi[:, 4] = 0.0

    remainder_rows: list[dict[str, Any]] = []
    for round_index in range(1, remainder_rounds + 1):
        rhs, rhs_observation = _observe_rhs(
            current,
            sine_order=sine_order,
            point_enclosure_backend=point_enclosure_backend,
        )
        integrated = rhs.integrate(0)
        image = _zero_exact_held_remainder(
            base_polynomial.add(integrated)
        )
        candidate_lo, candidate_hi = _interval_add(
            image.rem_lo,
            image.rem_hi,
            roundoff_lo,
            roundoff_hi,
        )
        candidate_lo[:, 4] = 0.0
        candidate_hi[:, 4] = 0.0
        margin = _subset_margin(
            current.rem_lo,
            current.rem_hi,
            candidate_lo,
            candidate_hi,
        )
        shrink = margin >= 0.0
        accepted_lo = torch.where(shrink, candidate_lo, current.rem_lo)
        accepted_hi = torch.where(shrink, candidate_hi, current.rem_hi)
        remainder_rows.append(
            {
                "round": round_index,
                "input_remainder": _interval_payload(
                    current.rem_lo, current.rem_hi
                ),
                "rhs": rhs_observation,
                "integration": _tm_payload(integrated),
                "image": _tm_payload(image),
                "candidate": _interval_payload(candidate_lo, candidate_hi),
                "accepted": _interval_payload(accepted_lo, accepted_hi),
                "subset_margin": margin,
                "shrink_mask": shrink,
            }
        )
        current = BatchedTaylorModel(
            image.poly,
            accepted_lo,
            accepted_hi,
            base.domain_lo,
            base.domain_hi,
            DenseRemainderLedger(
                {"picard_residual": (accepted_lo.clone(), accepted_hi.clone())}
            ),
            base.range_policy,
            base.range_trace,
        )

    local_tube = current.range_bound(context="tora_full_step_tube")
    local_endpoint = _endpoint_bounds(current, h=0.1)
    physical = compose_tora_q3_step(reference, carry)
    pairs = (
        (current.poly.coeffs, reference.segment_tm.poly.coeffs),
        (current.rem_lo, reference.segment_tm.rem_lo),
        (current.rem_hi, reference.segment_tm.rem_hi),
        (local_tube[0], reference.tube_lower),
        (local_tube[1], reference.tube_upper),
        (local_endpoint[0], reference.endpoint_lower),
        (local_endpoint[1], reference.endpoint_upper),
    )
    stages = {
        "A0": {
            "normalized_input": _tm_payload(base),
            "basis_exponents": base.poly.basis.exponents,
            "domain": _interval_payload(base.domain_lo, base.domain_hi),
            "carry": {
                "linear": carry.linear,
                "remainder": _interval_payload(
                    carry.remainder_lower, carry.remainder_upper
                ),
            },
        },
        "A1": {"base_polynomial_and_remainder": _tm_payload(base)},
        "A2_A5": polynomial_rows,
        "A6": {
            "polynomial_difference": {"coefficients": difference.coeffs},
            "range": _interval_payload(roundoff_lo, roundoff_hi),
        },
        "A7": {
            "seed": _interval_payload(-seed_vector, seed_vector),
            "rhs": initial_rhs_observation,
            "integration": _tm_payload(initial_integration),
            "image": _tm_payload(initial_image),
            "subset_margin": initial_margin,
        },
        "A8": remainder_rows,
        "A9": {
            "local_final": _tm_payload(current),
            "local_endpoint": _interval_payload(*local_endpoint),
            "local_tube": _interval_payload(*local_tube),
            "physical_coefficients": physical.segment_tm.poly.coeffs,
            "physical_remainder": _interval_payload(
                physical.segment_tm.rem_lo, physical.segment_tm.rem_hi
            ),
            "physical_endpoint": _interval_payload(
                physical.endpoint_lower, physical.endpoint_upper
            ),
            "physical_tube": _interval_payload(
                physical.tube_lower, physical.tube_upper
            ),
        },
        "A10": {
            "affine_carry": {
                "linear": carry.linear,
                "remainder": _interval_payload(
                    carry.remainder_lower, carry.remainder_upper
                ),
            }
        },
        "predicates": {
            "finite_ok_by_leaf": reference.finite_ok_by_leaf,
            "initial_subset_ok_by_leaf": reference.initial_subset_ok_by_leaf,
            "all_remainder_rounds_ok_by_leaf": reference.all_remainder_rounds_ok_by_leaf,
            "local_property_ok_by_leaf": reference.local_property_ok_by_leaf,
            "composed_property_ok_by_leaf": physical.composed_property_ok_by_leaf,
            "accepted_by_leaf": physical.accepted_by_leaf,
        },
    }
    equivalence = {
        "bitwise": _bitwise(pairs),
        "maximum_absolute_error": _maximum_abs(pairs),
        "reference_output_sha256": tensor_sha256(
            reference.segment_tm.poly.coeffs,
            reference.segment_tm.rem_lo,
            reference.segment_tm.rem_hi,
            reference.endpoint_lower,
            reference.endpoint_upper,
            reference.tube_lower,
            reference.tube_upper,
        ),
    }
    return TorchStageObservation(
        int(segment_index), stages, reference, physical, equivalence
    )


def validate_xiangru_stage_record(record: Mapping[str, Any]) -> None:
    if record.get("schema") != "xiangru_tora_q3_plant_segment_observation_v1":
        raise ValueError("unexpected Xiangru plant observation schema")
    segment = record.get("segment_index")
    if segment not in SELECTED_SEGMENTS:
        raise ValueError("record is not a selected replay segment")
    if record.get("leaf_id") != list(range(48)):
        raise ValueError("stage record does not use canonical B48 leaf order")
    stage = record.get("stage_contract")
    if not isinstance(stage, Mapping):
        raise ValueError("selected Xiangru record has no stage_contract")
    if stage.get("schema") != "xiangru_tora_q3_stage_contract_observation_v1":
        raise ValueError("unexpected Xiangru stage-contract schema")
    if len(stage.get("A2_A5_polynomial_picard", ())) != 2:
        raise ValueError("Xiangru stage contract does not contain K1 and K2")
    if len(stage.get("A8_remainder_rounds", ())) != 10:
        raise ValueError("Xiangru stage contract does not contain ten remainder rounds")


__all__ = [
    "REPLAY_POINTS",
    "SELECTED_SEGMENTS",
    "STAGE_IDS",
    "TorchStageObservation",
    "model_and_carry_from_xiangru_record",
    "model_from_xiangru_tm_payload",
    "observe_torch_integration_from_xiangru_payload",
    "observe_torch_local_step",
    "observe_torch_sine_from_xiangru_payload",
    "tensor_sha256",
    "tensor_tree_to_lists",
    "validate_xiangru_stage_record",
]
