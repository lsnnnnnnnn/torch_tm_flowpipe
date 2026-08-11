#!/usr/bin/env python3
"""Apply preregistered CDR/CNI reciprocal substitutions to frozen prestates."""

from __future__ import annotations

import argparse
import csv
import hashlib
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
from torch_tm_flowpipe.fixed_support import (
    FixedSupportInterval,
    FixedSupportPolynomial,
    FixedSupportSymbolicRemainderState,
    FixedSupportTaylorModel,
    diffreach_vdp_polynomial_rhs,
    diffreach_vdp_tm_rhs,
    fixed_support_build_linear_tm,
    fixed_support_polynomial_picard,
    fixed_support_step_boxes,
    fixed_support_symbolic_step_linear,
)


SCHEMA = "torch_r35_a3_a4_same_prestate_substitution_v1"
SUBSTITUTIONS = (
    ("CDR_complete_carry", "CDR", 1e-12),
    ("CNI_complete_carry", "CNI", 0.0),
    ("CDR_reciprocal_with_epsilon", "CDR", 1e-12),
    ("CDR_reciprocal_without_epsilon", "CDR", 0.0),
    ("CNI_reciprocal_with_epsilon", "CNI", 1e-12),
    ("CNI_reciprocal_without_epsilon", "CNI", 0.0),
)


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def _restore(path: Path) -> tuple[Any, Any, Any]:
    support = _support("R35")
    with np.load(path) as values:
        tensor = lambda name, dtype=torch.float64: torch.as_tensor(values[name].copy(), dtype=dtype)
        model = FixedSupportTaylorModel(
            FixedSupportPolynomial(tensor("model_polynomial"), support),
            FixedSupportInterval(tensor("model_remainder_lo"), tensor("model_remainder_hi")),
        )
        parameterization = FixedSupportTaylorModel(
            FixedSupportPolynomial(tensor("parameterization_polynomial"), support),
            FixedSupportInterval(
                tensor("parameterization_remainder_lo"), tensor("parameterization_remainder_hi")
            ),
        )
        phi = tensor("symbolic_Phi")
        j_lo = tensor("symbolic_J_lo")
        j_hi = tensor("symbolic_J_hi")
        count = tensor("symbolic_count", dtype=torch.long)
        inverse = tensor("symbolic_inverse_scale")
    symbolic = FixedSupportSymbolicRemainderState(
        phi,
        FixedSupportInterval(j_lo, j_hi),
        count,
        int(phi.shape[1]),
        inverse,
        torch.arange(phi.shape[1], dtype=torch.long),
    )
    return model, parameterization, symbolic


def _prestate_sha(model: Any, parameterization: Any, symbolic: Any) -> str:
    digest = hashlib.sha256()
    for value in (
        model.polynomial.coeffs, model.remainder.lo, model.remainder.hi,
        parameterization.polynomial.coeffs, parameterization.remainder.lo,
        parameterization.remainder.hi, symbolic.phi_buffer, symbolic.j_buffer.lo,
        symbolic.j_buffer.hi, symbolic.count, symbolic.inverse_scale,
    ):
        digest.update(array_record(value.detach().cpu().numpy())["sha256"].encode())
    return digest.hexdigest()


def _cni(
    endpoint: FixedSupportTaylorModel,
    parameterization: FixedSupportTaylorModel,
    eval_lo: torch.Tensor,
    eval_hi: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, FixedSupportTaylorModel, Any, Any]:
    inserted = endpoint.compose_affine(parameterization, 0.0)
    support = inserted.polynomial.support
    center = inserted.polynomial.coeffs[..., support.constant_slot]
    coefficients = inserted.polynomial.coeffs.clone()
    coefficients[..., support.constant_slot] = 0.0
    centered = FixedSupportTaylorModel(
        FixedSupportPolynomial(coefficients, support), inserted.remainder, inserted.ledger
    )
    centered_range = centered.range(eval_lo, eval_hi)
    scale = torch.maximum(torch.abs(centered_range.lo), torch.abs(centered_range.hi))
    if epsilon:
        inverse = 1.0 / (scale + float(epsilon))
    else:
        inverse = torch.where(scale == 0.0, torch.ones_like(scale), 1.0 / scale)
    return center, scale, inverse, centered.scale(inverse), inserted, centered


def _one(
    *,
    label: str,
    family: str,
    epsilon: float,
    model: Any,
    parameterization: Any,
    symbolic: Any,
) -> dict[str, Any]:
    support = model.polynomial.support
    step_lo, step_hi, eval_lo, eval_hi = fixed_support_step_boxes(
        model.polynomial.batch, 2, 0.01, dtype=torch.float64, device="cpu"
    )
    endpoint_previous = model.evaluate_time(0.01)
    inserted = centered = None
    if family == "CDR":
        carry = fixed_support_symbolic_step_linear(
            parameterization, endpoint_previous, symbolic, eval_lo, eval_hi, epsilon=epsilon
        )
        center = endpoint_previous.polynomial.coeffs[..., support.constant_slot]
        scale = carry.scale
        inverse = carry.state.inverse_scale
        normalized = carry.normalized_parameterization
        symbolic_next = carry.state
    else:
        center, scale, inverse, normalized, inserted, centered = _cni(
            endpoint_previous, parameterization, eval_lo, eval_hi, epsilon
        )
        symbolic_next = symbolic
    new_x0 = fixed_support_build_linear_tm(center, scale, support)
    polynomial, _ = fixed_support_polynomial_picard(
        new_x0.polynomial,
        diffreach_vdp_polynomial_rhs,
        step_lo,
        step_hi,
        iterations=4,
    )
    target = torch.full((model.polynomial.batch, 2), 0.01, dtype=torch.float64)
    seed = FixedSupportTaylorModel(polynomial, FixedSupportInterval(-target, target))
    raw_image = new_x0.add(
        diffreach_vdp_tm_rhs(seed, step_lo, step_hi).integrate_time(step_lo, step_hi)
    )
    difference = raw_image.polynomial.sub(seed.polynomial).range(step_lo, step_hi)
    raw_remainder = raw_image.remainder.add(difference)
    mask = raw_remainder.subseteq_elem(seed.remainder)
    validated = FixedSupportTaylorModel(polynomial, raw_remainder, raw_image.ledger)
    composed = validated.compose_affine(normalized, 0.01)
    endpoint_lo = step_lo.clone()
    endpoint_lo[:, support.local_time_index] = 0.01
    endpoint = composed.range(endpoint_lo, step_hi)
    tube = composed.range(step_lo, step_hi)
    margin = torch.minimum(raw_remainder.lo + target, target - raw_remainder.hi)
    return {
        "label": label,
        "family": family,
        "epsilon": epsilon,
        "epsilon_hex": float(epsilon).hex(),
        "accepted": bool(torch.all(mask).item()),
        "mask_sha256": array_record(mask.numpy())["sha256"],
        "minimum_target_margin": float(margin.min().item()),
        "center_sha256": array_record(center.numpy())["sha256"],
        "scale_sha256": array_record(scale.numpy())["sha256"],
        "inverse_scale_sha256": array_record(inverse.numpy())["sha256"],
        "normalized_polynomial_sha256": array_record(normalized.polynomial.coeffs.numpy())["sha256"],
        "normalized_remainder_lo_sha256": array_record(normalized.remainder.lo.numpy())["sha256"],
        "normalized_remainder_hi_sha256": array_record(normalized.remainder.hi.numpy())["sha256"],
        "raw_remainder_width_max": float(raw_remainder.width.max().item()),
        "endpoint_width_max": float(endpoint.width.max().item()),
        "tube_width_max": float(tube.width.max().item()),
        "symbolic_post_count": int(symbolic_next.count.item()),
        "inserted_polynomial_sha256": None
        if inserted is None
        else array_record(inserted.polynomial.coeffs.numpy())["sha256"],
        "centered_inserted_remainder_width_max": None
        if centered is None
        else float(centered.remainder.width.max().item()),
        "no_state_commit": True,
    }


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, Any]] = []
    checkpoint_reports = []
    for checkpoint in args.checkpoint:
        model, parameterization, symbolic = _restore(checkpoint)
        prestate_sha = _prestate_sha(model, parameterization, symbolic)
        checkpoint_rows = [
            {
                "checkpoint": checkpoint.name,
                "prestate_sha256": prestate_sha,
                **_one(
                    label=label,
                    family=family,
                    epsilon=epsilon,
                    model=model,
                    parameterization=parameterization,
                    symbolic=symbolic,
                ),
            }
            for label, family, epsilon in SUBSTITUTIONS
        ]
        rows.extend(checkpoint_rows)
        by_label = {row["label"]: row for row in checkpoint_rows}
        cdr_relevant = (
            by_label["CDR_reciprocal_with_epsilon"]["mask_sha256"]
            != by_label["CDR_reciprocal_without_epsilon"]["mask_sha256"]
            or by_label["CDR_reciprocal_with_epsilon"]["accepted"]
            != by_label["CDR_reciprocal_without_epsilon"]["accepted"]
        )
        cni_relevant = (
            by_label["CNI_reciprocal_with_epsilon"]["mask_sha256"]
            != by_label["CNI_reciprocal_without_epsilon"]["mask_sha256"]
            or by_label["CNI_reciprocal_with_epsilon"]["accepted"]
            != by_label["CNI_reciprocal_without_epsilon"]["accepted"]
        )
        checkpoint_reports.append(
            {
                "checkpoint": checkpoint.name,
                "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "prestate_sha256": prestate_sha,
                "all_substitutions_used_identical_prestate": len(
                    {row["prestate_sha256"] for row in checkpoint_rows}
                )
                == 1,
                "cdr_epsilon_decision_relevant": cdr_relevant,
                "cni_epsilon_decision_relevant": cni_relevant,
                "canonical_duplicate_checks": {
                    "CDR": by_label["CDR_complete_carry"] == {
                        **by_label["CDR_reciprocal_with_epsilon"],
                        "label": "CDR_complete_carry",
                    },
                    "CNI": by_label["CNI_complete_carry"] == {
                        **by_label["CNI_reciprocal_without_epsilon"],
                        "label": "CNI_complete_carry",
                    },
                },
            }
        )
    csv_path = args.output_dir / "substitutions.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "schema": SCHEMA,
        "source_sha": _git("rev-parse", "HEAD"),
        "checkpoints": checkpoint_reports,
        "substitution_count": len(rows),
        "epsilon_decision_relevant_anywhere": any(
            report["cdr_epsilon_decision_relevant"] or report["cni_epsilon_decision_relevant"]
            for report in checkpoint_reports
        ),
        "epsilon_scope": "one-step same-prestate diagnostic only; no new long-horizon lane",
        "csv_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
