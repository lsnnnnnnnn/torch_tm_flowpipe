#!/usr/bin/env python3
"""One-leaf/one-step fixed-control diagnostic with private decimal/hex detail."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from torch_tm_flowpipe.batched_dense_tm import dense_polynomial_picard
from torch_tm_flowpipe.tora_q3 import (
    build_tora_q3_box_model,
    compose_tora_q3_step,
    dense_tora_q3_dr_step,
    identity_tora_q3_carry,
    normalize_tora_q3_boundary,
    tora_q3_boundary_from_model,
    tora_q3_rhs,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def values(tensor: torch.Tensor) -> Any:
    return tensor.detach().cpu().tolist()


def hex_values(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: hex_values(item) for key, item in value.items()}
    if isinstance(value, list):
        return [hex_values(item) for item in value]
    return float(value).hex()


def maximum_abs(left: Any, right: Any) -> float:
    a = torch.as_tensor(left, dtype=torch.float64)
    b = torch.as_tensor(right, dtype=torch.float64)
    if a.shape != b.shape:
        return float("inf")
    return float(torch.max(torch.abs(a - b)).item())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller-trace", type=Path, required=True)
    parser.add_argument("--xiangru-plant", type=Path, required=True)
    parser.add_argument("--private-detail", type=Path, required=True)
    parser.add_argument("--public-summary", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.set_default_dtype(torch.float64)
    controller = json.loads(
        args.controller_trace.read_text(encoding="utf-8")
    )["rows"][0]
    if args.xiangru_plant.suffix == ".json":
        xiangru = json.loads(args.xiangru_plant.read_text(encoding="utf-8"))
        xiangru_header = {
            "basis_exponents": xiangru["basis_exponents"]
        }
    else:
        with args.xiangru_plant.open(encoding="utf-8") as handle:
            xiangru_header = json.loads(next(handle))
            xiangru = json.loads(next(handle))
    state_lower = torch.tensor(
        [controller["pre_controller_state_box"]["lower"][0]],
        dtype=torch.float64,
        device=device,
    )
    state_upper = torch.tensor(
        [controller["pre_controller_state_box"]["upper"][0]],
        dtype=torch.float64,
        device=device,
    )
    control_lower = torch.tensor(
        [controller["u1_interval_installed_for_next_ten_segments"]["lower"][0][0]],
        dtype=torch.float64,
        device=device,
    )
    control_upper = torch.tensor(
        [controller["u1_interval_installed_for_next_ten_segments"]["upper"][0][0]],
        dtype=torch.float64,
        device=device,
    )
    initial = build_tora_q3_box_model(
        state_lower,
        state_upper,
        control_lower,
        control_upper,
        device=device,
    )
    local, carry = normalize_tora_q3_boundary(
        tora_q3_boundary_from_model(initial),
        identity_tora_q3_carry(1, device=device),
    )
    k1, _ = dense_polynomial_picard(
        tora_q3_rhs,
        local.without_remainder(),
        tau_index=0,
        order=3,
        iterations=1,
        cutoff_threshold=None,
        capture_trace=False,
    )
    k2, _ = dense_polynomial_picard(
        tora_q3_rhs,
        local.without_remainder(),
        tau_index=0,
        order=3,
        iterations=2,
        cutoff_threshold=None,
        capture_trace=False,
    )
    local_step = dense_tora_q3_dr_step(local)
    physical_step = compose_tora_q3_step(local_step, carry)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    basis = local.poly.basis
    center = local.poly.coeffs[..., basis.constant_index]
    scale = torch.zeros((1, 5), dtype=torch.float64, device=device)
    for state in range(5):
        exponent = [0] * 6
        exponent[state + 1] = 1
        scale[:, state] = local.poly.coeffs[
            :, state, basis.term_index(exponent)
        ]
    torch_detail = {
        "normalization_center": values(center),
        "normalization_scale": values(scale),
        "normalized_map_linear": values(carry.linear),
        "normalized_map_remainder_lower": values(carry.remainder_lower),
        "normalized_map_remainder_upper": values(carry.remainder_upper),
        "k1_coefficients": values(k1.poly.coeffs),
        "k2_coefficients": values(k2.poly.coeffs),
        "final_local_coefficients": values(local_step.segment_tm.poly.coeffs),
        "final_local_remainder_lower": values(local_step.segment_tm.rem_lo),
        "final_local_remainder_upper": values(local_step.segment_tm.rem_hi),
        "physical_coefficients": values(physical_step.segment_tm.poly.coeffs),
        "physical_remainder_lower": values(physical_step.segment_tm.rem_lo),
        "physical_remainder_upper": values(physical_step.segment_tm.rem_hi),
        "endpoint_lower": values(physical_step.endpoint_lower),
        "endpoint_upper": values(physical_step.endpoint_upper),
        "tube_lower": values(physical_step.tube_lower),
        "tube_upper": values(physical_step.tube_upper),
        "initial_margin": values(local_step.initial_margin),
        "ledger": physical_step.segment_tm.ledger.intervals(),
    }
    xiangru_detail = {
        "normalization_center": [xiangru["normalization"]["center"][0]],
        "normalization_scale": [xiangru["normalization"]["scale"][0]],
        "normalized_map_linear": [
            xiangru["normalization"]["normalized_map_linear"][0]
        ],
        "normalized_map_remainder_lower": [
            xiangru["normalization"]["normalized_map_remainder"]["lower"][0]
        ],
        "normalized_map_remainder_upper": [
            xiangru["normalization"]["normalized_map_remainder"]["upper"][0]
        ],
        "k1_coefficients": [xiangru["picard"]["polynomial_k1"]["coefficients"][0]],
        "k2_coefficients": [xiangru["picard"]["polynomial_k2"]["coefficients"][0]],
        "final_local_coefficients": [xiangru["picard"]["final_polynomial"]["coefficients"][0]],
        "final_local_remainder_lower": [xiangru["picard"]["final_remainder"]["lower"][0]],
        "final_local_remainder_upper": [xiangru["picard"]["final_remainder"]["upper"][0]],
        "physical_coefficients": [xiangru["polynomial_coefficient_vector"][0]],
        "physical_remainder_lower": [xiangru["interval_remainder"]["lower"][0]],
        "physical_remainder_upper": [xiangru["interval_remainder"]["upper"][0]],
        "endpoint_lower": [xiangru["endpoint"]["lower"][0]],
        "endpoint_upper": [xiangru["endpoint"]["upper"][0]],
        "tube_lower": [xiangru["tube"]["lower"][0]],
        "tube_upper": [xiangru["tube"]["upper"][0]],
    }
    comparisons = {
        name: maximum_abs(torch_detail[name], xiangru_detail[name])
        for name in xiangru_detail
    }
    private = {
        "schema": "tora_q3_one_leaf_one_step_private_detail_v1",
        "leaf_id": 0,
        "basis_exponents": xiangru_header["basis_exponents"],
        "torch_decimal": torch_detail,
        "torch_hexfloat": hex_values(torch_detail),
        "xiangru_decimal": xiangru_detail,
        "xiangru_hexfloat": hex_values(xiangru_detail),
        "maximum_absolute_differences": comparisons,
    }
    args.private_detail.parent.mkdir(parents=True, exist_ok=True)
    args.private_detail.write_text(
        json.dumps(private, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    public = {
        "schema": "tora_q3_one_leaf_one_step_summary_v1",
        "status": "PASS" if physical_step.accepted else "FAIL",
        "leaf_id": 0,
        "fixed_control_interval": {
            "lower": float(control_lower.item()),
            "upper": float(control_upper.item()),
        },
        "basis": {
            "slot_count": basis.num_terms,
            "fingerprint": basis.fingerprint,
            "identity_slot_permutation": list(range(basis.num_terms)),
        },
        "normalization_bijection": "identical diagonal initial normalization",
        "maximum_absolute_differences": comparisons,
        "validation": {
            "accepted": physical_step.accepted,
            "minimum_initial_margin": float(local_step.initial_margin.min().item()),
            "remainder_round_count": len(local_step.round_trace),
        },
        "torch_remainder_ledger": physical_step.segment_tm.ledger.intervals(),
        "private_decimal_hex_detail_sha256": sha256(args.private_detail),
        "source_hashes": {
            "controller_trace": sha256(args.controller_trace),
            "xiangru_plant_trace": sha256(args.xiangru_plant),
        },
    }
    args.public_summary.parent.mkdir(parents=True, exist_ok=True)
    args.public_summary.write_text(
        json.dumps(public, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": public["status"],
        "maximum_absolute_differences": comparisons,
    }))
    return 0 if physical_step.accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
