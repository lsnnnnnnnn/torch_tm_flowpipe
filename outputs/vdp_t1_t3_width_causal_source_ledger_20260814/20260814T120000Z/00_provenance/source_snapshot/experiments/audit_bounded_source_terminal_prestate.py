#!/usr/bin/env python3
"""Frozen candidate-terminal actual-consumer controls for bounded G1."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from torch_tm_flowpipe import DenseRangePolicy, PolynomialODE, TaylorModel, TMVector, load_terminal_checkpoint
from torch_tm_flowpipe.flowpipe import _normalized_tm_from_box, flowpipe_step_from_tm
from torch_tm_flowpipe.source_ledger import collapse_source_polynomial

sys.path.insert(0, str(ROOT / "experiments"))
from run_vdp_dense_backend import load_contract


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def tm_digest(tmv: TMVector) -> str:
    return digest([
        [
            [list(exp), float(coef.detach().cpu()).hex()]
            for exp, coef in sorted(model.polynomial.terms.items())
        ]
        for model in tmv
    ])


def policy() -> DenseRangePolicy:
    return DenseRangePolicy(
        method="adaptive_subdivision",
        max_depth=1,
        max_leaves=4,
        split_vars=(0, 1),
        trigger="proactive_depth1_on_named_contexts",
        named_contexts=("polynomial_truncation",),
    )


def ordinary(tmv: TMVector) -> TMVector:
    if tmv.n_vars != 4:
        raise ValueError("frozen G1 terminal input must have four boundary variables")
    models = []
    for model in tmv:
        collapsed = collapse_source_polynomial(model.polynomial, model.domain, (2, 3))
        models.append(TaylorModel(
            collapsed.retained,
            model.remainder + collapsed.collapsed,
            model.domain,
            order=model.order,
            truncation_range_split=model.truncation_range_split,
        ))
    return TMVector(models)


def tamper(tmv: TMVector) -> TMVector:
    models = []
    changed = False
    for model in tmv:
        terms = dict(model.polynomial.terms)
        for exp in terms:
            if exp[2] or exp[3]:
                terms[exp] = terms[exp] * 1.01
                changed = True
                break
        models.append(TaylorModel(
            type(model.polynomial)(terms, model.n_vars),
            model.remainder,
            model.domain,
            order=model.order,
            truncation_range_split=model.truncation_range_split,
        ))
    if not changed:
        raise AssertionError("terminal payload has no source coefficient")
    return TMVector(models)


def consume(ode: PolynomialODE, tmv: TMVector, h: float) -> dict[str, Any]:
    seg = flowpipe_step_from_tm(
        ode,
        tmv,
        h,
        4,
        max_validation_attempts=2,
        validation_eps=1e-12,
        validation_mode="flowstar_raw_remainder_compat",
        target_remainder_radius=1e-4,
        cutoff_threshold=1e-10,
        tm_backend="dense",
        dense_range_policy=policy(),
    )
    return {
        "input_sha256": tm_digest(tmv),
        "status": seg.status,
        "message": seg.message,
        "raw_picard_image": seg.picard_image_remainder,
        "subset_margin": seg.subset_margin,
        "result_sha256": digest({
            "status": seg.status,
            "raw": seg.picard_image_remainder,
            "margin": seg.subset_margin,
        }),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checkpoint = load_terminal_checkpoint(args.checkpoint, expected_order=4, expected_dtype="float64")
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    h = float(reference["attempted_h"])
    contract = load_contract()
    ode = PolynomialODE.from_system_spec(contract["canonical_system_spec"])
    candidate = checkpoint.current
    ordinary_only = ordinary(candidate)
    legacy_rebox = _normalized_tm_from_box(candidate.range_box(), 4)
    payload_changed = tamper(candidate)
    rows = {
        "candidate_source_payload": consume(ode, candidate, h),
        "ordinary_only_same_affine_source_set": consume(ode, ordinary_only, h),
        "legacy_box_reparameterization_same_component_box": consume(ode, legacy_rebox, h),
        "payload_tamper_plus_1_percent": consume(ode, payload_changed, h),
    }
    metadata_reference = {**reference, "audit_metadata_only": "tampered"}
    metadata_result = consume(ode, candidate, h)
    rows["metadata_tamper"] = metadata_result
    if rows["candidate_source_payload"]["result_sha256"] != metadata_result["result_sha256"]:
        raise AssertionError("terminal metadata tamper altered the consumer")
    if rows["candidate_source_payload"]["result_sha256"] == rows["payload_tamper_plus_1_percent"]["result_sha256"]:
        raise AssertionError("terminal payload tamper did not alter the consumer")
    result = {
        "schema": "bounded_source_terminal_same_prestate_audit_v1",
        "time": checkpoint.scheduler["current_time"],
        "h": h,
        "checkpoint_schema": checkpoint.manifest["schema"],
        "checkpoint_full_sha256": checkpoint.manifest["full_checkpoint_sha256"],
        "actual_source_variables": [2, 3],
        "candidate_reference_result_sha256": digest({
            "status": reference["status"],
            "raw": reference["picard_image_remainder"],
            "margin": reference["subset_margin"],
        }),
        "metadata_tamper_reference_sha256": digest(metadata_reference),
        "metadata_tamper_preserved_actual_consumer": True,
        "payload_tamper_changed_actual_consumer": True,
        "flowstar_operator": {
            "status": "UNAVAILABLE",
            "reason": "candidate checkpoint has no lossless Flow* Phi_L/J Symbolic_Remainder object",
        },
        "rows": rows,
        "interpretation": (
            "The frozen prestate is the candidate's own last accepted affine source input. "
            "Ordinary-only and legacy box controls are constructed from that same represented component box."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({name: row["status"] for name, row in rows.items()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
