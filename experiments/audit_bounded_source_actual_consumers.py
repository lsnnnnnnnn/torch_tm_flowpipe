#!/usr/bin/env python3
"""Same-prestate actual-consumer audit for the bounded G1 source ledger."""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from torch_tm_flowpipe import (
    DenseRangePolicy,
    FlowstarNormalFlowpipeState,
    Interval,
    PolynomialODE,
    TaylorModel,
    TMVector,
)
from torch_tm_flowpipe.batched_dense_tm import REMAINDER_LEDGER_CATEGORIES
from torch_tm_flowpipe.flowpipe import (
    FlowpipeSegment,
    _flowstar_bounded_source_ledger_transition,
    _initialize_bounded_source_normal_state,
    flowpipe_step_flowstar_style_adaptive,
    flowpipe_step_from_tm,
)
from torch_tm_flowpipe.source_ledger import collapse_source_polynomial, metadata_tamper

sys.path.insert(0, str(ROOT / "experiments"))
from run_vdp_dense_backend import load_contract


CANDIDATE = "normalized_insertion_bounded_source_ledger_o4_g1"
CHECKPOINTS = {
    1: "step_1_to_2",
    99: "before_T_1",
    299: "before_T_3",
    631: "before_T_6_32",
}


def frozen_range_policy() -> DenseRangePolicy:
    return DenseRangePolicy(
        method="adaptive_subdivision",
        max_depth=1,
        max_leaves=4,
        split_vars=(0, 1),
        trigger="proactive_depth1_on_named_contexts",
        named_contexts=("polynomial_truncation",),
    )


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def tm_hash(tmv: TMVector) -> str:
    return canonical_hash(
        [
            {
                "terms": [
                    [list(exp), float(coef.detach().cpu()).hex()]
                    for exp, coef in sorted(model.polynomial.terms.items())
                ],
                "rem_lo": float(model.remainder.lo.detach().cpu()).hex(),
                "rem_hi": float(model.remainder.hi.detach().cpu()).hex(),
            }
            for model in tmv
        ]
    )


def extend_tmv(tmv: TMVector | None, count: int) -> TMVector | None:
    if tmv is None:
        return None
    out = tmv
    template = tmv.domain[0]
    for _ in range(count):
        out = out.extend_domain(
            Interval(
                torch.as_tensor(-1.0, dtype=template.lo.dtype, device=template.lo.device),
                torch.as_tensor(1.0, dtype=template.lo.dtype, device=template.lo.device),
            )
        )
    return out


def extend_segment(seg: FlowpipeSegment, count: int) -> FlowpipeSegment:
    return replace(
        seg,
        tm=extend_tmv(seg.tm, count),
        final_tm=extend_tmv(seg.final_tm, count),
        endpoint_raw_tm=extend_tmv(seg.endpoint_raw_tm, count),
        endpoint_tightened_tm=extend_tmv(seg.endpoint_tightened_tm, count),
        reset_tm=None,
        flowstar_normal_state=None,
        flowstar_normal_stats=None,
    )


def ordinary_materialization(source_reset: TMVector, base_dim: int) -> TMVector:
    models: list[TaylorModel] = []
    source_indices = tuple(range(base_dim, 2 * base_dim))
    for model in source_reset:
        collapsed = collapse_source_polynomial(model.polynomial, model.domain, source_indices)
        models.append(
            TaylorModel(
                collapsed.retained,
                model.remainder + collapsed.collapsed,
                model.domain,
                order=model.order,
                truncation_range_split=model.truncation_range_split,
            )
        )
    return TMVector(models)


def payload_tamper(source_reset: TMVector, base_dim: int) -> TMVector:
    models: list[TaylorModel] = []
    changed = False
    for model in source_reset:
        terms = dict(model.polynomial.terms)
        for exponent in sorted(terms):
            if any(exponent[index] for index in range(base_dim, 2 * base_dim)):
                terms[exponent] = terms[exponent] * 1.01
                changed = True
                break
        models.append(
            TaylorModel(
                type(model.polynomial)(terms, model.n_vars),
                model.remainder,
                model.domain,
                order=model.order,
                truncation_range_split=model.truncation_range_split,
            )
        )
    if not changed:
        raise AssertionError("source payload tamper found no live source coefficient")
    return TMVector(models)


def consume(ode: PolynomialODE, tmv: TMVector, h: float) -> dict[str, Any]:
    diagnostics: list[dict[str, Any]] = []
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
        dense_range_policy=frozen_range_policy(),
        diagnostics=diagnostics,
    )
    endpoint = seg.endpoint_raw_tm.range_box() if seg.endpoint_raw_tm is not None else []
    return {
        "status": seg.status,
        "message": seg.message,
        "input_sha256": tm_hash(tmv),
        "input_n_vars": tmv.n_vars,
        "input_active_variables": sorted(tmv.active_variables()),
        "candidate_remainder": seg.candidate_remainder,
        "raw_picard_image": seg.picard_image_remainder,
        "subset_margin": seg.subset_margin,
        "endpoint": [
            {
                "lo": float(iv.lo.detach().cpu()),
                "hi": float(iv.hi.detach().cpu()),
                "width": float(iv.width().detach().cpu()),
            }
            for iv in endpoint
        ],
        "segment_widths": [float(iv.width().detach().cpu()) for iv in seg.tm.range_box()],
        "result_sha256": canonical_hash(
            {
                "status": seg.status,
                "candidate": seg.candidate_remainder,
                "image": seg.picard_image_remainder,
                "margin": seg.subset_margin,
                "endpoint": [
                    [float(iv.lo.detach().cpu()).hex(), float(iv.hi.detach().cpu()).hex()]
                    for iv in endpoint
                ],
            }
        ),
    }


def audit_boundary(
    ode: PolynomialODE,
    seg: FlowpipeSegment,
    previous_state: FlowstarNormalFlowpipeState,
    label: str,
) -> dict[str, Any]:
    dim = len(previous_state.center)
    extended_state = _initialize_bounded_source_normal_state(previous_state, 4)
    assert extended_state.bounded_source_ledger_state is not None
    source_state = replace(
        extended_state.bounded_source_ledger_state,
        accepted_boundary_index=previous_state.step_index,
        generation=previous_state.step_index,
    )
    extended_state = replace(extended_state, bounded_source_ledger_state=source_state)
    extended_seg = extend_segment(seg, dim)
    source_reset, source_post, source_stats = _flowstar_bounded_source_ledger_transition(
        extended_seg,
        extended_state,
        4,
        cutoff_threshold=1e-10,
        right_map_range_mode="standard",
        right_map_center_mode="constant",
    )
    ordinary = ordinary_materialization(source_reset, dim)
    tampered = payload_tamper(source_reset, dim)
    assert source_post.bounded_source_ledger_state is not None
    metadata_state = replace(
        source_post,
        bounded_source_ledger_state=metadata_tamper(
            source_post.bounded_source_ledger_state,
            "same-prestate-consumer-audit",
        ),
    )
    metadata_reset = metadata_state.normalized_initial_tm(4)

    consumers = {
        "legacy_normalized_insertion": consume(ode, seg.reset_tm, 0.01),
        "ordinary_only_same_source_set": consume(ode, ordinary, 0.01),
        "bounded_source_ledger": consume(ode, source_reset, 0.01),
        "payload_tamper_plus_1_percent": consume(ode, tampered, 0.01),
        "metadata_tamper": consume(ode, metadata_reset, 0.01),
        "flowstar_operator": {
            "status": "UNAVAILABLE",
            "reason": (
                "the Torch accepted prestate does not serialize Flow* Phi_L/J and ordinary remainder "
                "as a lossless Flow* Symbolic_Remainder object; a lossy adapter is forbidden"
            ),
        },
    }
    source_consumer = consumers["bounded_source_ledger"]
    metadata_consumer = consumers["metadata_tamper"]
    tamper_consumer = consumers["payload_tamper_plus_1_percent"]
    if source_consumer["result_sha256"] != metadata_consumer["result_sha256"]:
        raise AssertionError(f"metadata changed actual consumer at {label}")
    if source_consumer["result_sha256"] == tamper_consumer["result_sha256"]:
        raise AssertionError(f"payload did not change actual consumer at {label}")
    return {
        "label": label,
        "accepted_boundary_index": previous_state.step_index + 1,
        "boundary_h": seg.h,
        "boundary_endpoint_sha256": tm_hash(seg.final_tm),
        "complete_ledger_categories": list(REMAINDER_LEDGER_CATEGORIES),
        "complete_ledger_contains_image": bool(
            torch.all(seg.validated_remainder_decomposition.contains_image)
        ),
        "source_reset_sha256": tm_hash(source_reset),
        "ordinary_reset_sha256": tm_hash(ordinary),
        "ordinary_materialization_contains_same_affine_source_set": True,
        "source_live_count": source_post.bounded_source_ledger_state.live_source_count,
        "source_fingerprint": source_post.bounded_source_ledger_state.fingerprint,
        "source_structured_width_mass": source_stats["source_ledger_structured_width_mass"],
        "source_ordinary_width_mass": source_stats["source_ledger_ordinary_width_mass"],
        "source_first_consumer_field": source_stats["source_ledger_first_consumer_field"],
        "tamper_payload_changed_consumer": True,
        "tamper_metadata_preserved_consumer": True,
        "consumers": consumers,
    }


def flatten(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        for mode, consumer in row["consumers"].items():
            out.append(
                {
                    "label": row["label"],
                    "boundary_index": row["accepted_boundary_index"],
                    "mode": mode,
                    "status": consumer.get("status", ""),
                    "input_sha256": consumer.get("input_sha256", ""),
                    "result_sha256": consumer.get("result_sha256", ""),
                    "raw_picard_image": json.dumps(consumer.get("raw_picard_image", ""), sort_keys=True),
                    "subset_margin": json.dumps(consumer.get("subset_margin", ""), sort_keys=True),
                    "endpoint": json.dumps(consumer.get("endpoint", ""), sort_keys=True),
                    "reason": consumer.get("reason", ""),
                }
            )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    contract = load_contract()
    ode = PolynomialODE.from_system_spec(contract["canonical_system_spec"])
    initial = [Interval(*bounds) for bounds in contract["initial_box"]]
    normal_state = FlowstarNormalFlowpipeState.from_initial_box(initial, 4)
    current = normal_state.normalized_initial_tm(4)
    results: list[dict[str, Any]] = []
    started = time.perf_counter()
    for accepted_index in range(1, max(CHECKPOINTS) + 1):
        before = normal_state
        seg = flowpipe_step_flowstar_style_adaptive(
            ode,
            current,
            h=0.01,
            h_min=0.01,
            h_max=0.01,
            order=4,
            target_remainder_radius=1e-4,
            cutoff_threshold=1e-10,
            max_validation_attempts=2,
            validation_eps=1e-12,
            validation_mode="flowstar_raw_remainder_compat",
            reset_mode="normalized_insertion",
            step_policy_mode="flowstar_compat",
            flowstar_normal_state=normal_state,
            tm_backend="dense",
            dense_range_policy=frozen_range_policy(),
        )
        if seg.status != "validated" or seg.reset_tm is None or seg.flowstar_normal_state is None:
            raise RuntimeError(f"legacy prefix failed at {accepted_index}: {seg.message}")
        if accepted_index in CHECKPOINTS:
            results.append(audit_boundary(ode, seg, before, CHECKPOINTS[accepted_index]))
        current = seg.reset_tm
        normal_state = seg.flowstar_normal_state

    summary = {
        "schema": "bounded_source_same_prestate_actual_consumer_audit_v1",
        "candidate": CANDIDATE,
        "fixed_h": 0.01,
        "order": 4,
        "cutoff": 1e-10,
        "target_remainder_radius": 1e-4,
        "validation_mode": "flowstar_raw_remainder_compat",
        "legacy_prefix_accepted_steps": max(CHECKPOINTS),
        "runtime_s": time.perf_counter() - started,
        "all_complete_ledgers_contain_image": all(row["complete_ledger_contains_image"] for row in results),
        "all_payload_tampers_changed_consumer": all(row["tamper_payload_changed_consumer"] for row in results),
        "all_metadata_tampers_preserved_consumer": all(row["tamper_metadata_preserved_consumer"] for row in results),
        "first_causally_active_field": "affine_source_coefficient_in_next_dense_picard_input",
        "flowstar_operator_status": "UNAVAILABLE_LOSSLESS_FLOWSTAR_STATE_NOT_SERIALIZED",
        "rows": results,
    }
    (args.output_dir / "consumer_audit.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    flat = flatten(results)
    with (args.output_dir / "consumer_audit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    print(json.dumps({key: summary[key] for key in (
        "legacy_prefix_accepted_steps",
        "runtime_s",
        "all_complete_ledgers_contain_image",
        "all_payload_tampers_changed_consumer",
        "all_metadata_tampers_preserved_consumer",
    )}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
