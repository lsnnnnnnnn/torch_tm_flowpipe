"""Observation-only counterfactuals for TORA-Q3 stage attribution.

These helpers accept frozen stage tensors as diagnostic inputs.  They are not
called by the formal runner and never turn external outputs into a native
certificate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from .batched_dense_tm import (
    BatchedPolynomial,
    BatchedTaylorModel,
    DenseRemainderLedger,
    _interval_add,
    _subset_margin,
)
from .tora_q3 import (
    ToraQ3AffineCarry,
    _endpoint_bounds,
    _zero_exact_held_remainder,
    compose_tora_q3_tm,
    tora_q3_rhs,
)
from .tora_stage_contract import model_from_xiangru_tm_payload


@dataclass(frozen=True)
class CounterfactualRemainderResult:
    initial_image: BatchedTaylorModel
    initial_margin: torch.Tensor
    polynomial_difference: BatchedPolynomial
    roundoff_lower: torch.Tensor
    roundoff_upper: torch.Tensor
    rounds: tuple[Mapping[str, torch.Tensor], ...]
    final: BatchedTaylorModel
    endpoint_lower: torch.Tensor
    endpoint_upper: torch.Tensor
    tube_lower: torch.Tensor
    tube_upper: torch.Tensor


@dataclass(frozen=True)
class CounterfactualCompositionResult:
    physical: BatchedTaylorModel
    endpoint_lower: torch.Tensor
    endpoint_upper: torch.Tensor
    tube_lower: torch.Tensor
    tube_upper: torch.Tensor


def candidate_from_xiangru_coefficients(
    base: BatchedTaylorModel,
    coefficients: Any,
) -> BatchedTaylorModel:
    value = torch.as_tensor(
        coefficients,
        dtype=base.poly.coeffs.dtype,
        device=base.poly.coeffs.device,
    )
    if value.shape != base.poly.coeffs.shape:
        raise ValueError("counterfactual candidate coefficient shape mismatch")
    zeros = torch.zeros_like(base.rem_lo)
    return BatchedTaylorModel(
        BatchedPolynomial(value, base.poly.basis),
        zeros,
        zeros.clone(),
        base.domain_lo,
        base.domain_hi,
        DenseRemainderLedger.empty(),
        base.range_policy,
        base.range_trace,
    )


def _rhs_with_sine(
    state: BatchedTaylorModel,
    sine: BatchedTaylorModel,
) -> BatchedTaylorModel:
    if state.poly.out_dim != 5 or sine.poly.out_dim != 1:
        raise ValueError("counterfactual RHS shape mismatch")
    x1 = state.component(0)
    x2 = state.component(1)
    x4 = state.component(3)
    control = state.component(4)
    state2 = -x1 + sine.scale(0.1)
    state4 = control - 10.0
    held = BatchedTaylorModel.constants_like(0.0, control)
    return BatchedTaylorModel.concat((x2, state2, x4, state4, held))


def run_torch_remainder_counterfactual(
    base: BatchedTaylorModel,
    polynomial_candidate: BatchedTaylorModel,
    *,
    sine_overrides: Sequence[BatchedTaylorModel] | None = None,
    remainder_rounds: int = 10,
    seed: float = 0.01,
    point_enclosure_backend: str = "eager",
) -> CounterfactualRemainderResult:
    """Run the native Torch remainder phase from a supplied K2 candidate.

    When supplied, ``sine_overrides`` contains the observed initial sine and
    one observed sine for every remainder round.  The override is explicitly
    a diagnostic stage substitution.
    """
    if remainder_rounds != 10:
        raise ValueError("the counterfactual contract freezes ten rounds")
    if sine_overrides is not None and len(sine_overrides) != 11:
        raise ValueError("sine override contract requires initial plus ten rounds")
    if polynomial_candidate.poly.coeffs.shape != base.poly.coeffs.shape:
        raise ValueError("counterfactual K2 candidate shape mismatch")

    def rhs(value: BatchedTaylorModel, index: int) -> BatchedTaylorModel:
        if sine_overrides is None:
            return tora_q3_rhs(
                value,
                sine_order=2,
                point_enclosure_backend=point_enclosure_backend,
            )
        override = sine_overrides[index]
        if override.poly.batch != value.poly.batch:
            raise ValueError("sine override batch mismatch")
        return _rhs_with_sine(value, override)

    base_polynomial = base.without_remainder()
    seed_vector = torch.tensor(
        [seed, seed, seed, seed, 0.0],
        dtype=base.poly.coeffs.dtype,
        device=base.poly.coeffs.device,
    ).view(1, 5).expand(base.poly.batch, -1)
    current = polynomial_candidate.with_remainder(-seed_vector, seed_vector)
    initial_image = _zero_exact_held_remainder(
        base_polynomial.add(rhs(current, 0).integrate(0))
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
        context="tora_counterfactual_picard_roundoff",
        trace=base.range_trace,
    )
    roundoff_lo[:, 4] = 0.0
    roundoff_hi[:, 4] = 0.0

    rows: list[Mapping[str, torch.Tensor]] = []
    for round_index in range(1, remainder_rounds + 1):
        image = _zero_exact_held_remainder(
            base_polynomial.add(rhs(current, round_index).integrate(0))
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
        rows.append(
            {
                "candidate_lower": candidate_lo,
                "candidate_upper": candidate_hi,
                "accepted_lower": accepted_lo,
                "accepted_upper": accepted_hi,
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
                {
                    "picard_residual": (
                        accepted_lo.clone(),
                        accepted_hi.clone(),
                    )
                }
            ),
            base.range_policy,
            base.range_trace,
        )
    tube_lower, tube_upper = current.range_bound(
        context="tora_counterfactual_full_step_tube"
    )
    endpoint_lower, endpoint_upper = _endpoint_bounds(current, h=0.1)
    return CounterfactualRemainderResult(
        initial_image,
        initial_margin,
        difference,
        roundoff_lo,
        roundoff_hi,
        tuple(rows),
        current,
        endpoint_lower,
        endpoint_upper,
        tube_lower,
        tube_upper,
    )


def sine_overrides_from_xiangru_stage(
    stage: Mapping[str, Any],
    *,
    device: torch.device | str,
) -> tuple[BatchedTaylorModel, ...]:
    sources = [
        stage["A7_initial_remainder_image"],
        *stage["A8_remainder_rounds"],
    ]
    return tuple(
        model_from_xiangru_tm_payload(
            source["rhs"]["sine"]["output"], device=device
        )
        for source in sources
    )


def torch_range_of_xiangru_difference(
    base: BatchedTaylorModel,
    coefficients: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    value = torch.as_tensor(
        coefficients,
        dtype=base.poly.coeffs.dtype,
        device=base.poly.coeffs.device,
    )
    if value.shape != base.poly.coeffs.shape:
        raise ValueError("counterfactual difference coefficient shape mismatch")
    polynomial = BatchedPolynomial(value, base.poly.basis)
    return polynomial.range_bound(
        base.domain_lo,
        base.domain_hi,
        policy=base.range_policy,
        context="tora_counterfactual_same_polynomial_range",
        trace=base.range_trace,
    )


def compose_xiangru_local_with_torch(
    local_payload: Mapping[str, Any],
    carry: ToraQ3AffineCarry,
    *,
    device: torch.device | str,
) -> CounterfactualCompositionResult:
    local = model_from_xiangru_tm_payload(local_payload, device=device)
    physical = compose_tora_q3_tm(local, carry)
    tube_lower, tube_upper = physical.range_bound(
        context="tora_counterfactual_same_endpoint_tube"
    )
    endpoint_lower, endpoint_upper = _endpoint_bounds(physical, h=0.1)
    return CounterfactualCompositionResult(
        physical,
        endpoint_lower,
        endpoint_upper,
        tube_lower,
        tube_upper,
    )


__all__ = [
    "CounterfactualCompositionResult",
    "CounterfactualRemainderResult",
    "candidate_from_xiangru_coefficients",
    "compose_xiangru_local_with_torch",
    "run_torch_remainder_counterfactual",
    "sine_overrides_from_xiangru_stage",
    "torch_range_of_xiangru_difference",
]
