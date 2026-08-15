#!/usr/bin/env python3
"""Resolve G1 ordinary owners and run real next-Picard owner interventions."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch

from torch_tm_flowpipe import (
    DenseRangePolicy,
    FlowstarNormalFlowpipeState,
    Interval,
    PolynomialODE,
    TaylorModel,
    TMVector,
)
from torch_tm_flowpipe.flowpipe import (
    _interval_magnitude,
    _tmvector_constant_part,
    _tmvector_rm_constants,
    _tmvector_without_remainder,
    flowpipe_step_flowstar_style_adaptive,
    flowpipe_step_from_tm,
    insert_ctrunc_normal_like,
)
from torch_tm_flowpipe.g2_shared_column import owner_rows, polynomial_payload_sha256
from torch_tm_flowpipe.polynomial import Polynomial
from torch_tm_flowpipe.source_ledger import (
    affine_lift_interval,
    collapse_source_polynomial,
    metadata_tamper,
)

sys.path.insert(0, str(ROOT / "experiments"))
from run_vdp_dense_backend import load_contract


CANDIDATE = "normalized_insertion_bounded_source_ledger_o4_g1"
POSITIONS = {
    1: "step_1_to_2",
    99: "before_T1",
    299: "before_T3",
    631: "before_T6p32",
    632: "at_T6p32_boundary_mass_2p1933445893",
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
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def tm_hash(tmv: TMVector) -> str:
    return canonical_hash(
        [
            {
                "terms": [
                    [list(exp), float(coef.detach().cpu()).hex()]
                    for exp, coef in sorted(model.polynomial.terms.items())
                ],
                "remainder": [
                    float(model.remainder.lo.detach().cpu()).hex(),
                    float(model.remainder.hi.detach().cpu()).hex(),
                ],
            }
            for model in tmv
        ]
    )


def consume(
    ode: PolynomialODE,
    tmv: TMVector,
    *,
    h: float = 0.01,
    nonconsumer_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    segment = flowpipe_step_from_tm(
        ode,
        tmv,
        float(h),
        4,
        max_validation_attempts=2,
        validation_eps=1e-12,
        validation_mode="flowstar_raw_remainder_compat",
        target_remainder_radius=1e-4,
        cutoff_threshold=1e-10,
        tm_backend="dense",
        dense_range_policy=frozen_range_policy(),
    )
    endpoint = segment.endpoint_raw_tm.range_box() if segment.endpoint_raw_tm is not None else []
    tube = segment.tm.range_box()
    consumer_output = {
        "status": segment.status,
        "message": segment.message,
        "raw_picard_image": segment.picard_image_remainder,
        "candidate_remainder": segment.candidate_remainder,
        "subset_margin": segment.subset_margin,
        "endpoint_raw": [
            [float(interval.lo.detach().cpu()).hex(), float(interval.hi.detach().cpu()).hex()]
            for interval in endpoint
        ],
        "segment_tube_raw": [
            [float(interval.lo.detach().cpu()).hex(), float(interval.hi.detach().cpu()).hex()]
            for interval in tube
        ],
    }
    result = {
        "status": segment.status,
        "message": segment.message,
        "input_sha256": tm_hash(tmv),
        "input_n_vars": tmv.n_vars,
        "input_active_variables": sorted(tmv.active_variables()),
        "input_term_count": sum(len(model.polynomial.terms) for model in tmv),
        "raw_picard_image": segment.picard_image_remainder,
        "candidate_remainder": segment.candidate_remainder,
        "subset_margin": segment.subset_margin,
        "endpoint_raw": [
            {
                "lo_hex": float(interval.lo.detach().cpu()).hex(),
                "hi_hex": float(interval.hi.detach().cpu()).hex(),
                "width": float(interval.width().detach().cpu()),
            }
            for interval in endpoint
        ],
        "segment_tube_raw": [
            {
                "lo_hex": float(interval.lo.detach().cpu()).hex(),
                "hi_hex": float(interval.hi.detach().cpu()).hex(),
                "width": float(interval.width().detach().cpu()),
            }
            for interval in tube
        ],
        "nonconsumer_metadata": dict(nonconsumer_metadata or {}),
        "consumer_output_sha256": canonical_hash(consumer_output),
    }
    return result


def extend_polynomial(poly: Polynomial, count: int) -> Polynomial:
    return poly.extend_vars(count)


def build_reset(
    base_models: Sequence[TaylorModel],
    *,
    center: Sequence[float],
    old_source_polys: Sequence[Polynomial] | None,
    ordinary_lift: Any | None,
    fresh_lift: Any,
) -> TMVector:
    dim = len(base_models)
    n_vars = 3 * dim
    domain = [Interval(-1.0, 1.0) for _ in range(n_vars)]
    scales: list[float] = []
    for model in base_models:
        magnitude = _interval_magnitude(model.range_box())
        scale = 0.0 if magnitude is None or magnitude == 0.0 else float(magnitude)
        if scale > 0.0:
            for _ in range(8):
                scale = math.nextafter(scale, math.inf)
        scales.append(scale)
    models: list[TaylorModel] = []
    for component, (base, scale) in enumerate(zip(base_models, scales)):
        physical_center = float(center[component])
        if ordinary_lift is not None:
            physical_center += float(ordinary_lift.midpoint[0, component].detach().cpu())
        poly = Polynomial.constant(physical_center, n_vars)
        if scale:
            poly = poly + Polynomial.variable(component, n_vars) * scale
        if old_source_polys is not None:
            poly = poly + extend_polynomial(old_source_polys[component], dim)
        elif ordinary_lift is not None:
            radius = float(ordinary_lift.radius[0, component].detach().cpu())
            if radius:
                poly = poly + Polynomial.variable(dim + component, n_vars) * radius
        fresh_radius = float(fresh_lift.radius[0, component].detach().cpu())
        if fresh_radius:
            poly = poly + Polynomial.variable(2 * dim + component, n_vars) * fresh_radius
        models.append(TaylorModel(poly, Interval.zero(), domain, order=4))
    return TMVector(models)


def tamper_first_source(tmv: TMVector, source_indices: Sequence[int]) -> TMVector:
    changed = False
    models: list[TaylorModel] = []
    for model in tmv:
        terms = dict(model.polynomial.terms)
        for exponent in sorted(terms):
            if any(exponent[index] for index in source_indices):
                terms[exponent] = terms[exponent] * 2.0
                changed = True
                break
        models.append(TaylorModel(Polynomial(terms, model.n_vars), model.remainder, model.domain, order=4))
    if not changed:
        raise RuntimeError("owner payload tamper found no source coefficient")
    return TMVector(models)


def max_source_coefficient(tmv: TMVector, source_indices: Sequence[int]) -> float:
    return max(
        (
            abs(float(coefficient.detach().cpu()))
            for model in tmv
            for exponent, coefficient in model.polynomial.terms.items()
            if any(exponent[index] for index in source_indices)
        ),
        default=0.0,
    )


def variants(previous: FlowstarNormalFlowpipeState, segment: Any) -> tuple[dict[str, TMVector], dict[str, Any]]:
    source_state = previous.bounded_source_ledger_state
    if source_state is None or segment.validated_remainder_decomposition is None:
        raise ValueError("G1 owner intervention needs actual accepted source state and ledger")
    dim = source_state.state_dim
    inner = list(previous.tmv_right)
    inner.extend(
        TaylorModel.variable(dim + index, previous.domain, order=4)
        for index in range(dim)
    )
    diagnostics: dict[str, Any] = {}
    inserted = insert_ctrunc_normal_like(
        _tmvector_without_remainder(_tmvector_rm_constants(segment.final_tm)),
        TMVector(inner),
        4,
        1e-10,
        previous.domain,
        diagnostics,
    )
    base_preserve_old: list[TaylorModel] = []
    base_share_ordinary: list[TaylorModel] = []
    old_source_polys: list[Polynomial] = []
    retired_rows: list[Mapping[str, Any]] = []
    carried_rows: list[dict[str, Any]] = []
    ordinary_lo: list[torch.Tensor] = []
    ordinary_hi: list[torch.Tensor] = []
    for component, model in enumerate(inserted):
        collapse = collapse_source_polynomial(model.polynomial, previous.domain, source_state.source_indices)
        source_partition = {
            exponent: coefficient
            for exponent, coefficient in model.polynomial.terms.items()
            if any(exponent[index] for index in source_state.source_indices)
        }
        source_poly = Polynomial(source_partition, model.n_vars)
        old_source_polys.append(source_poly)
        base_preserve_old.append(
            TaylorModel(collapse.retained, model.remainder, previous.domain, order=4)
        )
        base_share_ordinary.append(
            TaylorModel(collapse.retained, collapse.collapsed, previous.domain, order=4)
        )
        ordinary_lo.append(model.remainder.lo)
        ordinary_hi.append(model.remainder.hi)
        retired_rows.extend(
            owner_rows(
                model.polynomial,
                previous.domain,
                component=component,
                oldest_indices=source_state.source_indices,
                current_indices=(),
                oldest_source_ids=source_state.source_ids,
            )
        )
        lo = float(model.remainder.lo.detach().cpu())
        hi = float(model.remainder.hi.detach().cpu())
        carried_rows.append(
            {
                "category": "ordinary_parameterization_composition_remainder",
                "component": component,
                "lo_hex": lo.hex(),
                "hi_hex": hi.hex(),
                "width": hi - lo,
                "support_sha256": hashlib.sha256(
                    f"ordinary:{component}:{lo.hex()}:{hi.hex()}".encode("utf-8")
                ).hexdigest(),
                "containment_witness": "actual_inserted_TaylorModel_remainder",
            }
        )
    decomposition = segment.validated_remainder_decomposition
    fresh_lift = affine_lift_interval(decomposition.decomposition_lo, decomposition.decomposition_hi)
    ordinary_lift = affine_lift_interval(
        torch.stack(ordinary_lo)[None, :],
        torch.stack(ordinary_hi)[None, :],
    )
    old_center = _tmvector_constant_part(segment.final_tm)
    fresh_center = [
        float(old_center[index]) + float(fresh_lift.midpoint[0, index].detach().cpu())
        for index in range(dim)
    ]
    preserve_old = build_reset(
        base_preserve_old,
        center=fresh_center,
        old_source_polys=old_source_polys,
        ordinary_lift=None,
        fresh_lift=fresh_lift,
    )
    share_ordinary = build_reset(
        base_share_ordinary,
        center=fresh_center,
        old_source_polys=None,
        ordinary_lift=ordinary_lift,
        fresh_lift=fresh_lift,
    )
    owner_record = {
        "retired_source_owners": retired_rows,
        "carried_ordinary_owners": carried_rows,
        "insertion_owners": diagnostics.get("_insertion_owner_rows", []),
        "retained_old_source_payload_sha256": polynomial_payload_sha256(old_source_polys),
        "fresh_lift": fresh_lift.as_dict(),
        "ordinary_lift": ordinary_lift.as_dict(),
        "interactions_additive": False,
    }
    return {
        "g1_actual": segment.reset_tm,
        "share_retired_old_source_polynomial": preserve_old,
        "share_cumulative_ordinary_parameterization": share_ordinary,
    }, owner_record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    contract = load_contract()
    ode = PolynomialODE.from_system_spec(contract["canonical_system_spec"])
    normal = FlowstarNormalFlowpipeState.from_exact_decimal_box(
        [("1.1", "1.4"), ("2.35", "2.45")], 4
    ).with_bounded_source_g1(4)
    current = normal.normalized_initial_tm(4)
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for accepted_step in range(1, max(POSITIONS) + 1):
        before = normal
        segment = flowpipe_step_flowstar_style_adaptive(
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
            reset_mode=CANDIDATE,
            step_policy_mode="flowstar_compat",
            flowstar_normal_state=normal,
            tm_backend="dense",
            dense_range_policy=frozen_range_policy(),
        )
        if segment.status != "validated" or segment.reset_tm is None or segment.flowstar_normal_state is None:
            raise RuntimeError(f"fresh G1 prefix failed at {accepted_step}: {segment.message}")
        if accepted_step in POSITIONS:
            candidate_variants, owners = variants(before, segment)
            consumers: dict[str, Any] = {}
            control_status: dict[str, Any] = {}
            for name, reset in candidate_variants.items():
                consumers[name] = consume(ode, reset)
                if name != "g1_actual":
                    source_indices = (2, 3) if name == "share_cumulative_ordinary_parameterization" else (2, 3)
                    maximum_source = max_source_coefficient(reset, source_indices)
                    if maximum_source <= 1e-10:
                        control_status[name] = {
                            "payload_control": "NOT_APPLICABLE_OWNER_ABSENT_OR_BELOW_FROZEN_CUTOFF",
                            "maximum_source_coefficient": maximum_source,
                            "frozen_cutoff": 1e-10,
                            "reason": (
                                "the selected owner has no live coefficient above the preregistered cutoff; "
                                "no artificial source term was introduced"
                            ),
                        }
                        continue
                    try:
                        tampered = tamper_first_source(reset, source_indices)
                    except RuntimeError:
                        control_status[name] = {
                            "payload_control": "NOT_APPLICABLE_OWNER_ABSENT",
                            "reason": (
                                "the selected owner has no live coefficient at this boundary; "
                                "no artificial source term was introduced"
                            ),
                        }
                        continue
                    consumers[f"{name}__payload_tamper_x2"] = consume(ode, tampered)
                    consumers[f"{name}__metadata_tamper"] = consume(
                        ode,
                        reset,
                        nonconsumer_metadata={
                            "tamper": "owner_label_changed_without_modifying_TMVector",
                            "owner_label": f"{name}:metadata-only-control",
                        },
                    )
                    if consumers[name]["consumer_output_sha256"] == consumers[f"{name}__payload_tamper_x2"]["consumer_output_sha256"]:
                        raise RuntimeError(f"payload tamper did not change consumer: {name}")
                    if consumers[name]["consumer_output_sha256"] != consumers[f"{name}__metadata_tamper"]["consumer_output_sha256"]:
                        raise RuntimeError(f"metadata tamper changed consumer: {name}")
                    control_status[name] = {
                        "payload_control": "PASS_CHANGED_REAL_CONSUMER_OUTPUT",
                        "metadata_control": "PASS_PRESERVED_REAL_CONSUMER_OUTPUT",
                    }
            actual_state = segment.flowstar_normal_state
            if actual_state is None or actual_state.bounded_source_ledger_state is None:
                raise RuntimeError("G1 actual control is missing accepted source metadata")
            actual_metadata_state = replace(
                actual_state,
                bounded_source_ledger_state=metadata_tamper(
                    actual_state.bounded_source_ledger_state,
                    "actual-consumer-negative-control",
                ),
            )
            actual_metadata_reset = actual_metadata_state.normalized_initial_tm(4)
            if tm_hash(actual_metadata_reset) != tm_hash(candidate_variants["g1_actual"]):
                raise RuntimeError("G1 metadata-only tamper changed the actual polynomial input")
            consumers["g1_actual__metadata_tamper"] = consume(
                ode,
                actual_metadata_reset,
                nonconsumer_metadata={
                    "pre_fingerprint": actual_state.bounded_source_ledger_state.fingerprint,
                    "post_fingerprint": actual_metadata_state.bounded_source_ledger_state.fingerprint,
                    "tamper": "source_lineage_metadata_only",
                },
            )
            if (
                consumers["g1_actual"]["consumer_output_sha256"]
                != consumers["g1_actual__metadata_tamper"]["consumer_output_sha256"]
            ):
                raise RuntimeError("actual G1 metadata tamper changed the real consumer")
            rows.append(
                {
                    "label": POSITIONS[accepted_step],
                    "accepted_boundary_step": accepted_step,
                    "time": accepted_step * 0.01,
                    "time_hex": float(accepted_step * 0.01).hex(),
                    "owners": owners,
                    "consumers": consumers,
                    "control_status": control_status,
                }
            )
        current = segment.reset_tm
        normal = segment.flowstar_normal_state

    result = {
        "schema": "g1_owner_resolved_actual_consumer_interventions_v1",
        "candidate": CANDIDATE,
        "initialization": "exact_decimal_contract",
        "fixed_h": 0.01,
        "positions": list(POSITIONS.values()),
        "pre_registered_interventions": [
            "share_retired_old_source_polynomial",
            "share_cumulative_ordinary_parameterization",
        ],
        "selection_rule": "all two recoverable major ordinary owners; no magnitude or result selection",
        "rows": rows,
        "runtime_s": time.perf_counter() - started,
    }
    (output / "owner_interventions.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "PASS", "positions": len(rows), "runtime_s": result["runtime_s"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
