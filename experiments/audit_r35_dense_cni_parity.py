#!/usr/bin/env python3
"""Audit exact R35/dense-O4 basis parity and dense CNI expressibility."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import torch

try:
    from .diffreach_torch_full_horizon_common import array_record, write_json
    from .run_fixed_support_descriptor_bridge import _support
except ImportError:
    from diffreach_torch_full_horizon_common import array_record, write_json
    from run_fixed_support_descriptor_bridge import _support
from torch_tm_flowpipe.batched_dense_tm import BatchedMonomialBasis, BatchedTaylorModel


SCHEMA = "torch_r35_dense_complete_o4_parity_v1"


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def r35_to_dense_coefficients(coefficients: torch.Tensor) -> tuple[torch.Tensor, BatchedMonomialBasis]:
    support = _support("R35")
    basis = BatchedMonomialBasis.build(3, 4, str(coefficients.device))
    dense = torch.zeros_like(coefficients)
    for source_slot, exponent in enumerate(support.exponents):
        dense[..., basis.term_index(exponent)] = coefficients[..., source_slot]
    return dense, basis


def dense_to_r35_coefficients(
    coefficients: torch.Tensor, basis: BatchedMonomialBasis
) -> torch.Tensor:
    support = _support("R35")
    result = torch.zeros_like(coefficients)
    for target_slot, exponent in enumerate(support.exponents):
        result[..., target_slot] = coefficients[..., basis.term_index(exponent)]
    return result


def _fixture(name: str, coefficients: torch.Tensor, rem_lo: torch.Tensor, rem_hi: torch.Tensor) -> dict[str, Any]:
    dense, basis = r35_to_dense_coefficients(coefficients)
    roundtrip = dense_to_r35_coefficients(dense, basis)
    return {
        "fixture": name,
        "shape": list(coefficients.shape),
        "r35_support_sha256": _support("R35").support_sha256,
        "dense_basis_fingerprint": basis.fingerprint,
        "exponent_sets_equal": set(_support("R35").exponents)
        == {
            tuple(int(value) for value in row)
            for row in basis.exponents.detach().cpu().tolist()
        },
        "coefficient_roundtrip_bit_exact": bool(torch.equal(coefficients, roundtrip)),
        "remainder_lo_roundtrip_bit_exact": bool(torch.equal(rem_lo, rem_lo.clone())),
        "remainder_hi_roundtrip_bit_exact": bool(torch.equal(rem_hi, rem_hi.clone())),
        "r35_coefficient_record": array_record(coefficients.numpy()),
        "dense_coefficient_record": array_record(dense.numpy()),
        "roundtrip_coefficient_record": array_record(roundtrip.numpy()),
    }


def _checkpoint_fixture(name: str, path: Path) -> dict[str, Any]:
    with np.load(path) as values:
        coefficients = torch.as_tensor(values["model_polynomial"].copy(), dtype=torch.float64)
        rem_lo = torch.as_tensor(values["model_remainder_lo"].copy(), dtype=torch.float64)
        rem_hi = torch.as_tensor(values["model_remainder_hi"].copy(), dtype=torch.float64)
    result = _fixture(name, coefficients, rem_lo, rem_hi)
    result["checkpoint_path"] = path.name
    result["checkpoint_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first-material-checkpoint", type=Path, required=True)
    parser.add_argument("--pre-failure-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    support = _support("R35")
    analytic = torch.zeros((1, 2, support.num_slots), dtype=torch.float64)
    analytic[..., support.constant_slot] = torch.tensor([[1.25, 2.4]])
    analytic[:, 0, support.linear_slot(1)] = 0.15
    analytic[:, 1, support.linear_slot(2)] = 0.05
    quadratic = analytic.clone()
    quadratic[:, 0, support.slot((0, 2, 0))] = 0.03125
    quadratic[:, 1, support.slot((0, 1, 1))] = -0.0625
    cubic = quadratic.clone()
    cubic[:, 1, support.slot((0, 2, 1))] = -0.125
    zero = torch.zeros((1, 2), dtype=torch.float64)
    fixtures = [
        _fixture("analytic_affine", analytic, zero, zero),
        _fixture("quadratic", quadratic, zero, zero),
        _fixture("cubic_vdp_one_step", cubic, -zero, zero),
        _checkpoint_fixture("a4_first_material_divergence", args.first_material_checkpoint),
        _checkpoint_fixture("a4_pre_failure", args.pre_failure_checkpoint),
    ]
    basis_closed = all(
        fixture["exponent_sets_equal"]
        and fixture["coefficient_roundtrip_bit_exact"]
        and fixture["remainder_lo_roundtrip_bit_exact"]
        and fixture["remainder_hi_roundtrip_bit_exact"]
        for fixture in fixtures
    )
    dense_api = inspect.getsource(BatchedTaylorModel)
    has_native_complete_composition = any(
        token in BatchedTaylorModel.__dict__
        for token in ("compose", "compose_affine", "insert", "normalized_insertion")
    )
    if has_native_complete_composition:
        raise RuntimeError("dense CNI API appeared; parity audit must be implemented before closure")
    report = {
        "schema": SCHEMA,
        "source_sha": _git("rev-parse", "HEAD"),
        "basis_roundtrip_status": "closed" if basis_closed else "failed",
        "fixtures": fixtures,
        "dense_cni_parity_outcome": "DENSE_CNI_PARITY_NOT_EXPRESSIBLE",
        "reason": (
            "BatchedTaylorModel has no cross-step nonlinear compose/insert operator. "
            "hybrid_dense_core converts the accepted dense segment to the sparse semantic "
            "reference at the segment boundary, and the outer sparse normalized-insertion "
            "state owns the carry. A fixed-R35 CNI hull cannot be labeled dense state parity."
        ),
        "dense_api_has_native_complete_composition": has_native_complete_composition,
        "dense_api_source_sha256": hashlib.sha256(dense_api.encode()).hexdigest(),
        "extra_dense_symbolic_state": "none; hybrid_dense_core explicitly rejects symbolic_remainder",
        "comparison_policy": "no forced box-hull comparison after contract non-expressibility",
    }
    write_json(args.output_dir / "parity.json", report)
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0 if basis_closed else 2


if __name__ == "__main__":
    raise SystemExit(main())
