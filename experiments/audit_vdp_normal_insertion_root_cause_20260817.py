#!/usr/bin/env python3
"""Gate-A causal audit for repeated ordinary-remainder insertion in VDP.

The script follows one fresh exact-decimal legacy trajectory.  At six accepted
boundaries it freezes the real endpoint/right-map prestate, runs the preregistered
D/H/D-P/D-one/H-P insertion cells, and feeds every resulting reset to the real
next dense Picard/validator consumer.  Exact binary64 inputs are lifted to
``Fraction`` polynomials; exact multivariate Bernstein coefficients then certify
the D/H enclosures without samples or a Flow* runtime oracle.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import subprocess
import sys
import time
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(ROOT / "experiments"))

import run_vdp_dense_backend as authoritative
from torch_tm_flowpipe import (
    DenseRangePolicy,
    FlowstarNormalFlowpipeState,
    Interval,
    PolynomialODE,
    TaylorModel,
    TMVector,
    flowpipe_step_flowstar_style_adaptive,
    insert_ctrunc_normal_dependency_preserving,
    insert_ctrunc_normal_like,
    save_terminal_checkpoint,
    taylor_model_mul_breakdown,
    tmvector_hashes,
)
from torch_tm_flowpipe.flowpipe import (
    _compose_term_with_inner,
    _flowstar_normalized_insertion_transition,
    _interval_magnitude,
    _normalized_tm_from_center_scale,
    _poly_interval_with_split,
    _scale_tmvector_components,
    _tm_max_degree,
    _tmvector_constant_part,
    _tmvector_rm_constants,
    _tmvector_with_order,
    _tmvector_without_remainder,
)


ORDER = 4
CUTOFF = 1e-10
H = 0.01
TARGET = 1e-4
REGISTERED_D_ONE_PATH = (1, (2, 1))
CHECKPOINT_STEPS = {
    1: "step_1_to_2",
    2: "first_nonzero_step_2_to_3",
    99: "before_T1",
    299: "before_T3",
    631: "before_T6p32",
    632: "terminal_pre",
}
CELLS = ("D", "H", "D-P", "D-one", "H-P")
TRUE_SEMANTIC_CELLS = {"D", "H"}

RationalInterval = tuple[Fraction, Fraction]
RationalPolynomial = dict[tuple[int, ...], Fraction]


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_bytes(value))


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
    return float(value)


def _interval_payload(value: Interval) -> dict[str, Any]:
    lo = _float(value.lo)
    hi = _float(value.hi)
    return {"lo": lo, "hi": hi, "lo_hex": lo.hex(), "hi_hex": hi.hex(), "width": hi - lo}


def _tmvector_payload(value: TMVector) -> dict[str, Any]:
    return {
        "components": [
            {
                "terms": [
                    {"exponent": list(exponent), "coefficient_hex": _float(coefficient).hex()}
                    for exponent, coefficient in sorted(model.polynomial.terms.items())
                ],
                "remainder": _interval_payload(model.remainder),
                "order": model.order,
                "truncation_range_split": model.truncation_range_split,
            }
            for model in value
        ],
        "domain": [_interval_payload(interval) for interval in value.domain],
        "n_vars": value.n_vars,
        "dtype": str(value[0].remainder.lo.dtype),
        "device": str(value[0].remainder.lo.device),
    }


def _tmvector_sha(value: TMVector) -> str:
    return _sha(_tmvector_payload(value))


def _box_payload(value: Sequence[Interval] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    names = ("x", "y")
    return {
        names[index] if index < len(names) else f"state_{index}": _interval_payload(interval)
        for index, interval in enumerate(value)
    }


def _q(value: Any) -> Fraction:
    return Fraction.from_float(_float(value))


def _q_text(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def _q_interval_payload(value: RationalInterval) -> dict[str, Any]:
    lower = float(value[0])
    upper = float(value[1])
    lower_out = math.nextafter(lower, -math.inf)
    upper_out = math.nextafter(upper, math.inf)
    return {
        "exact_lower": _q_text(value[0]),
        "exact_upper": _q_text(value[1]),
        "directed_binary64_lower_hex": lower_out.hex(),
        "directed_binary64_upper_hex": upper_out.hex(),
        "directed_binary64_width": upper_out - lower_out,
    }


def _iadd(left: RationalInterval, right: RationalInterval) -> RationalInterval:
    return left[0] + right[0], left[1] + right[1]


def _imul(left: RationalInterval, right: RationalInterval) -> RationalInterval:
    products = tuple(a * b for a in left for b in right)
    return min(products), max(products)


def _poly_add(left: RationalPolynomial, right: RationalPolynomial) -> RationalPolynomial:
    result = dict(left)
    for exponent, coefficient in right.items():
        result[exponent] = result.get(exponent, Fraction(0)) + coefficient
        if result[exponent] == 0:
            del result[exponent]
    return result


def _poly_mul(left: RationalPolynomial, right: RationalPolynomial) -> RationalPolynomial:
    result: RationalPolynomial = {}
    for left_exp, left_coefficient in left.items():
        for right_exp, right_coefficient in right.items():
            exponent = tuple(a + b for a, b in zip(left_exp, right_exp))
            result[exponent] = result.get(exponent, Fraction(0)) + left_coefficient * right_coefficient
    return {exponent: coefficient for exponent, coefficient in result.items() if coefficient}


def _poly_pow(value: RationalPolynomial, power: int) -> RationalPolynomial:
    variables = len(next(iter(value)))
    result = {(0,) * variables: Fraction(1)}
    for _ in range(power):
        result = _poly_mul(result, value)
    return result


def _bernstein_range(
    value: RationalPolynomial,
    domain: Sequence[RationalInterval],
) -> RationalInterval:
    """Exact tensor-product Bernstein enclosure after an affine box map."""

    variables = len(domain)
    if not value:
        return Fraction(0), Fraction(0)
    unit_power: RationalPolynomial = {}
    for exponent, coefficient in value.items():
        expanded = {(0,) * variables: coefficient}
        for variable, power in enumerate(exponent):
            lower, upper = domain[variable]
            width = upper - lower
            factor: RationalPolynomial = {}
            for unit_power_value in range(power + 1):
                factor_exponent = [0] * variables
                factor_exponent[variable] = unit_power_value
                factor[tuple(factor_exponent)] = (
                    Fraction(math.comb(power, unit_power_value))
                    * lower ** (power - unit_power_value)
                    * width ** unit_power_value
                )
            expanded = _poly_mul(expanded, factor)
        unit_power = _poly_add(unit_power, expanded)
    degrees = [
        max((exponent[variable] for exponent in unit_power), default=0)
        for variable in range(variables)
    ]
    bounds: list[Fraction] = []
    for beta in itertools.product(*(range(degree + 1) for degree in degrees)):
        coefficient = Fraction(0)
        for alpha, power_coefficient in unit_power.items():
            if any(a > b for a, b in zip(alpha, beta)):
                continue
            weight = Fraction(1)
            for a, b, degree in zip(alpha, beta, degrees):
                weight *= Fraction(math.comb(b, a), math.comb(degree, a))
            coefficient += power_coefficient * weight
        bounds.append(coefficient)
    return min(bounds), max(bounds)


def _augmented_inner(
    inner: TMVector,
    use_remainders: Sequence[bool],
) -> list[RationalPolynomial]:
    base_variables = inner.n_vars
    remainder_variables = len(inner)
    result: list[RationalPolynomial] = []
    for remainder_index, (model, use_remainder) in enumerate(zip(inner, use_remainders)):
        polynomial = {
            tuple(exponent) + (0,) * remainder_variables: _q(coefficient)
            for exponent, coefficient in model.polynomial.terms.items()
        }
        if use_remainder:
            exponent = [0] * (base_variables + remainder_variables)
            exponent[base_variables + remainder_index] = 1
            polynomial[tuple(exponent)] = Fraction(1)
        result.append(polynomial)
    return result


def _exact_cell_composition(
    outer: TaylorModel,
    inner: TMVector,
    cell: str,
    component_index: int,
) -> RationalPolynomial:
    variables = inner.n_vars + len(inner)
    exact: RationalPolynomial = {}
    for outer_exponent, outer_coefficient in outer.polynomial.terms.items():
        use_remainders = [cell in TRUE_SEMANTIC_CELLS] * len(inner)
        if cell in {"D-P", "H-P"}:
            use_remainders = [False] * len(inner)
        elif cell == "D-one":
            registered = (component_index, tuple(outer_exponent)) == REGISTERED_D_ONE_PATH
            use_remainders = [registered] * len(inner)
        augmented = _augmented_inner(inner, use_remainders)
        term = {(0,) * variables: _q(outer_coefficient)}
        for inner_polynomial, power in zip(augmented, outer_exponent):
            term = _poly_mul(term, _poly_pow(inner_polynomial, int(power)))
        exact = _poly_add(exact, term)
    return exact


def _exact_oracle(
    outer: TaylorModel,
    inner: TMVector,
    result: TaylorModel,
    cell: str,
    component_index: int,
) -> dict[str, Any]:
    exact = _exact_cell_composition(outer, inner, cell, component_index)
    represented = {
        tuple(exponent) + (0,) * len(inner): -_q(coefficient)
        for exponent, coefficient in result.polynomial.terms.items()
    }
    residual = _poly_add(exact, represented)
    domain: list[RationalInterval] = [
        (_q(interval.lo), _q(interval.hi)) for interval in inner.domain
    ] + [
        (_q(model.remainder.lo), _q(model.remainder.hi)) for model in inner
    ]
    residual_range = _bernstein_range(residual, domain)
    residual_range = (
        residual_range[0] + _q(outer.remainder.lo),
        residual_range[1] + _q(outer.remainder.hi),
    )
    represented_polynomial = {
        tuple(exponent): _q(coefficient)
        for exponent, coefficient in result.polynomial.terms.items()
    }
    polynomial_domain = [(_q(interval.lo), _q(interval.hi)) for interval in inner.domain]
    polynomial_range = _bernstein_range(represented_polynomial, polynomial_domain)
    output_lower = _q(result.remainder.lo)
    output_upper = _q(result.remainder.hi)
    contained = output_lower <= residual_range[0] and output_upper >= residual_range[1]
    return {
        "method": "exact_binary64_rationals_tensor_product_bernstein",
        "directed_rounding": "exact Fraction bounds converted outward with nextafter only for display",
        "polynomial_image": _q_interval_payload(polynomial_range),
        "semantic_residual_enclosure": _q_interval_payload(residual_range),
        "production_remainder": _interval_payload(result.remainder),
        "production_remainder_contains_exact_bernstein_enclosure": contained,
        "true_original_semantics": cell in TRUE_SEMANTIC_CELLS,
        "control_semantics": (
            "inner ordinary remainder zeroed"
            if cell in {"D-P", "H-P"}
            else "only preregistered nonlinear path consumes inner ordinary remainder"
            if cell == "D-one"
            else "original"
        ),
        "exact_residual_term_count": len(residual),
    }


def _one_path_insert(
    outer: TMVector,
    inner: TMVector,
    diagnostics: dict[str, Any],
) -> TMVector:
    out_domain = list(inner.domain)
    polynomial_inner = _tmvector_without_remainder(inner)
    models: list[TaylorModel] = []
    rows: list[dict[str, Any]] = []
    for component_index, outer_model in enumerate(outer):
        inner_degree = max((_tm_max_degree(TMVector([model])) for model in inner), default=0)
        work_order = max(ORDER, outer_model.polynomial.degree() * max(1, inner_degree))
        full_work = _tmvector_with_order(inner, work_order)
        polynomial_work = _tmvector_with_order(polynomial_inner, work_order)
        accumulator = TaylorModel.zero(out_domain, order=work_order)
        for exponent, coefficient in outer_model.polynomial.terms.items():
            selected = (component_index, tuple(exponent)) == REGISTERED_D_ONE_PATH
            selected_inner = full_work if selected else polynomial_work
            term = _compose_term_with_inner(
                coefficient,
                tuple(exponent),
                selected_inner,
                work_order=work_order,
                domain=out_domain,
            )
            rows.append(
                {
                    "component_index": component_index,
                    "outer_exponent": list(exponent),
                    "registered_path": selected,
                    "consumes_inner_ordinary_remainder": selected,
                    "term_remainder": _interval_payload(term.remainder),
                }
            )
            accumulator = accumulator + term
        kept, dropped = accumulator.polynomial.truncate(ORDER)
        truncation = _poly_interval_with_split(dropped, out_domain, accumulator.truncation_range_split)
        cutoff_kept, cutoff = kept.cutoff(CUTOFF, out_domain)
        models.append(
            TaylorModel(
                cutoff_kept,
                accumulator.remainder + outer_model.remainder + truncation + cutoff,
                out_domain,
                order=ORDER,
            )
        )
    diagnostics["registered_path"] = {
        "component_index": REGISTERED_D_ONE_PATH[0],
        "outer_exponent": list(REGISTERED_D_ONE_PATH[1]),
        "preregistered_before_checkpoint_observation": True,
    }
    diagnostics["path_rows"] = rows
    diagnostics["registered_path_present"] = any(row["registered_path"] for row in rows)
    return TMVector(models)


def _direct_stage_ledger(outer: TMVector, inner: TMVector) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    inner_degree = max((_tm_max_degree(TMVector([model])) for model in inner), default=0)
    for component_index, outer_model in enumerate(outer):
        work_order = max(ORDER, outer_model.polynomial.degree() * max(1, inner_degree))
        inner_work = _tmvector_with_order(inner, work_order)
        for exponent, coefficient in outer_model.polynomial.terms.items():
            term = TaylorModel.constant(coefficient, inner.domain, order=work_order)
            for variable_index, power in enumerate(exponent):
                for occurrence in range(int(power)):
                    breakdown = taylor_model_mul_breakdown(term, inner_work[variable_index], work_order)
                    term = term * inner_work[variable_index]
                    rows.append(
                        {
                            "component_index": component_index,
                            "outer_exponent": list(exponent),
                            "variable_index": variable_index,
                            "power_occurrence": occurrence,
                            "inner_remainder_nonzero": not (
                                _float(inner_work[variable_index].remainder.lo) == 0.0
                                and _float(inner_work[variable_index].remainder.hi) == 0.0
                            ),
                            **breakdown,
                            "result_remainder": _interval_payload(term.remainder),
                        }
                    )
    return rows


def _transition_from_inserted(
    segment: Any,
    previous_state: FlowstarNormalFlowpipeState,
    inserted: TMVector,
    cell: str,
) -> tuple[TMVector, FlowstarNormalFlowpipeState, dict[str, Any]]:
    center = _tmvector_constant_part(segment.final_tm)
    inserted_box = inserted.range_box()
    scales = [float(_interval_magnitude(interval) or 0.0) for interval in inserted_box]
    inverse = [1.0 if scale == 0.0 else 1.0 / scale for scale in scales]
    right = _scale_tmvector_components(inserted, inverse).apply_cutoff(CUTOFF)
    reset = _normalized_tm_from_center_scale(center, scales, ORDER, template_domain=previous_state.domain)
    state = FlowstarNormalFlowpipeState(
        tmv_pre=segment.tm,
        tmv_right=right,
        domain=list(previous_state.domain),
        center=center,
        scales=scales,
        step_index=previous_state.step_index + 1,
        diagnostics={"reset_mode": cell, "diagnostic_control": cell not in TRUE_SEMANTIC_CELLS},
    )
    return reset, state, {
        "inserted_box": _box_payload(inserted_box),
        "center": center,
        "scales": scales,
        "right_map_sha256": _tmvector_sha(right),
        "reset_sha256": _tmvector_sha(reset),
    }


def _consumer(
    ode: PolynomialODE,
    reset: TMVector,
    state: FlowstarNormalFlowpipeState,
    policy: DenseRangePolicy,
) -> dict[str, Any]:
    diagnostics: list[dict[str, Any]] = []
    segment = flowpipe_step_flowstar_style_adaptive(
        ode,
        reset,
        h=H,
        h_min=H,
        h_max=H,
        order=ORDER,
        target_remainder_radius=TARGET,
        cutoff_threshold=CUTOFF,
        max_validation_attempts=2,
        validation_eps=1e-12,
        validation_mode="flowstar_raw_remainder_compat",
        reset_mode="normalized_insertion",
        step_policy_mode="flowstar_compat",
        flowstar_normal_state=state,
        tm_backend="dense",
        dense_device="cpu",
        dense_dtype=torch.float64,
        dense_range_policy=policy,
        diagnostics=diagnostics,
    )
    ledger_rows = [
        row for row in (segment.backend_trace or []) if row.get("phase") == "remainder_validation"
    ]
    return {
        "status": segment.status,
        "message": segment.message,
        "accepted": segment.status == "validated" and segment.endpoint_raw_tm is not None,
        "candidate_remainder": segment.candidate_remainder,
        "picard_image_remainder": segment.picard_image_remainder,
        "subset_margin": segment.subset_margin,
        "validator_ledger": ledger_rows[-1] if ledger_rows else None,
        "segment_width": _box_payload(segment.tm.range_box()),
        "endpoint_width": _box_payload(
            segment.endpoint_raw_tm.range_box() if segment.endpoint_raw_tm is not None else None
        ),
    }


def _channel_widths(cell: Mapping[str, Any]) -> dict[str, float | None]:
    consumer = cell["consumer"]
    result: dict[str, float | None] = {}
    for prefix, source in (("segment", consumer.get("segment_width")), ("endpoint", consumer.get("endpoint_width"))):
        for component in ("x", "y"):
            result[f"{prefix}_{component}"] = (
                None if source is None else float(source[component]["width"])
            )
    return result


def _deltas(cells: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    widths = {name: _channel_widths(value) for name, value in cells.items()}
    pairs = {"D-(D-P)": ("D", "D-P"), "H-(H-P)": ("H", "H-P"), "D-H": ("D", "H")}
    result: dict[str, Any] = {"consumer_widths": widths}
    for label, (left, right) in pairs.items():
        result[label] = {
            channel: (
                None
                if widths[left][channel] is None or widths[right][channel] is None
                else widths[left][channel] - widths[right][channel]
            )
            for channel in widths[left]
        }
        result[label]["insertion_remainder"] = [
            cells[left]["insertion_output"][index]["remainder"]["width"]
            - cells[right]["insertion_output"][index]["remainder"]["width"]
            for index in range(2)
        ]
    return result


def _audit_boundary(
    *,
    label: str,
    accepted_step: int,
    time_value: float,
    segment: Any,
    previous_state: FlowstarNormalFlowpipeState,
    ode: PolynomialODE,
    policy: DenseRangePolicy,
    output_dir: Path,
) -> dict[str, Any]:
    if segment.endpoint_raw_tm is None:
        raise RuntimeError("accepted boundary has no raw endpoint")
    outer = _tmvector_rm_constants(segment.endpoint_raw_tm)
    inner = previous_state.tmv_right
    input_payload = {
        "outer_endpoint_tm_sha256": _tmvector_sha(segment.endpoint_raw_tm),
        "outer_without_constants_sha256": _tmvector_sha(outer),
        "tmv_right_sha256": _tmvector_sha(inner),
        "domain_sha256": _sha([_interval_payload(interval) for interval in previous_state.domain]),
        "order": ORDER,
        "cutoff": CUTOFF,
        "dtype": str(inner[0].remainder.lo.dtype),
        "device": str(inner[0].remainder.lo.device),
        "outer_endpoint": _tmvector_payload(segment.endpoint_raw_tm),
        "tmv_right": _tmvector_payload(inner),
        "input_ordinary_remainders": [
            _interval_payload(model.remainder) for model in inner
        ],
    }
    shared_hash = _sha(input_payload)
    direct_stages = _direct_stage_ledger(outer, inner)
    polynomial_inner = _tmvector_without_remainder(inner)
    cells: dict[str, dict[str, Any]] = {}
    for cell in CELLS:
        diagnostics: dict[str, Any] = {}
        if cell == "D":
            inserted = insert_ctrunc_normal_like(outer, inner, ORDER, CUTOFF, inner.domain, diagnostics)
        elif cell == "H":
            inserted = insert_ctrunc_normal_dependency_preserving(
                outer, inner, ORDER, CUTOFF, inner.domain, diagnostics
            )
        elif cell == "D-P":
            inserted = insert_ctrunc_normal_like(
                outer, polynomial_inner, ORDER, CUTOFF, inner.domain, diagnostics
            )
        elif cell == "H-P":
            inserted = insert_ctrunc_normal_dependency_preserving(
                outer, polynomial_inner, ORDER, CUTOFF, inner.domain, diagnostics
            )
        else:
            inserted = _one_path_insert(outer, inner, diagnostics)
        assert isinstance(inserted, TMVector)
        if cell == "D":
            reset, state, transition_diagnostics = _flowstar_normalized_insertion_transition(
                segment,
                previous_state,
                ORDER,
                cutoff_threshold=CUTOFF,
            )
        elif cell == "H":
            reset, state, transition_diagnostics = _flowstar_normalized_insertion_transition(
                segment,
                previous_state,
                ORDER,
                cutoff_threshold=CUTOFF,
                dependency_preserving_insertion=True,
            )
        else:
            reset, state, transition_diagnostics = _transition_from_inserted(
                segment, previous_state, inserted, cell
            )
        insertion_output = [
            {
                "polynomial_term_count": len(model.polynomial.terms),
                "polynomial_sha256": _sha(
                    [
                        [list(exponent), _float(coefficient).hex()]
                        for exponent, coefficient in sorted(model.polynomial.terms.items())
                    ]
                ),
                "remainder": _interval_payload(model.remainder),
                "range": _interval_payload(model.range_box()),
                "oracle": _exact_oracle(outer[index], inner, model, cell, index),
            }
            for index, model in enumerate(inserted)
        ]
        if cell in TRUE_SEMANTIC_CELLS and not all(
            row["oracle"]["production_remainder_contains_exact_bernstein_enclosure"]
            for row in insertion_output
        ):
            raise RuntimeError(f"{label}/{cell} failed exact rational containment")
        cells[cell] = {
            "eligibility": "sound_production_semantics" if cell in TRUE_SEMANTIC_CELLS else "counterfactual_diagnostic_only",
            "same_input_sha256": shared_hash,
            "same_input_byte_identity": True,
            "canonical_variable_order": [0, 1] if cell in {"H", "H-P"} else None,
            "insertion_output": insertion_output,
            "reset_sha256": _tmvector_sha(reset),
            "right_map_sha256": _tmvector_sha(state.tmv_right),
            "transition_diagnostics": transition_diagnostics,
            "insertion_diagnostics": diagnostics,
            "consumer": _consumer(ode, reset, state, policy),
        }
    direct_reset_sha = _tmvector_sha(segment.reset_tm)
    direct_state_sha = _tmvector_sha(segment.flowstar_normal_state.tmv_right)
    if cells["D"]["reset_sha256"] != direct_reset_sha or cells["D"]["right_map_sha256"] != direct_state_sha:
        raise RuntimeError("replayed direct cell differs from the actual accepted legacy transition")
    checkpoint_dir = output_dir / "checkpoints" / label
    manifest = save_terminal_checkpoint(
        checkpoint_dir,
        current=segment.reset_tm,
        normal_state=segment.flowstar_normal_state,
        scheduler={
            "current_time": time_value,
            "h_next": H,
            "accepted_segment_count": accepted_step,
            "checkpoint_label": label,
        },
        contract={
            "ode": "x'=y, y'=y-x-x^2*y",
            "initial_box_exact_decimal": [["1.1", "1.4"], ["2.35", "2.45"]],
            "order": ORDER,
            "cutoff": CUTOFF,
            "fixed_h": H,
            "target_remainder_radius": TARGET,
        },
        provenance={
            "branch": _git("branch", "--show-current"),
            "commit": _git("rev-parse", "HEAD"),
            "tracked_diff_sha256": hashlib.sha256(
                subprocess.run(["git", "diff", "HEAD", "--binary"], cwd=ROOT, check=True, capture_output=True).stdout
            ).hexdigest(),
            "dtype": "float64",
            "device": "cpu",
        },
    )
    result = {
        "label": label,
        "accepted_step": accepted_step,
        "time": time_value,
        "same_prestate": input_payload,
        "same_prestate_sha256": shared_hash,
        "direct_stage_ledger": direct_stages,
        "direct_nonlinear_path_remainder_consumptions": sum(
            row["inner_remainder_nonzero"]
            for row in direct_stages
            if sum(row["outer_exponent"]) >= 2
        ),
        "horner_factorized_multiplications": cells["H"]["insertion_diagnostics"].get(
            "insertion_factorized_multiplication_count"
        ),
        "cells": cells,
        "deltas": _deltas(cells),
        "checkpoint": {
            "relative_path": str(checkpoint_dir.relative_to(output_dir)),
            "manifest": manifest,
            "tmvector_hashes": tmvector_hashes(segment.reset_tm),
        },
        "actual_legacy_replay_byte_identical": True,
    }
    _write_json(output_dir / "boundaries" / f"{label}.json", result)
    return result


def run(output_dir: Path) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    contract = authoritative.load_contract()
    ode = PolynomialODE.from_system_spec(contract["canonical_system_spec"])
    policy = DenseRangePolicy(
        method="adaptive_subdivision",
        max_depth=1,
        max_leaves=4,
        split_vars=(0, 1),
        trigger="proactive_depth1_on_named_contexts",
        named_contexts=("polynomial_truncation",),
        variable_orders=((0, 1, 2), (1, 0, 2), (2, 0, 1)),
    )
    state = FlowstarNormalFlowpipeState.from_exact_decimal_box(
        [("1.1", "1.4"), ("2.35", "2.45")], ORDER
    )
    current = state.normalized_initial_tm(ORDER)
    current_time = 0.0
    boundaries: list[dict[str, Any]] = []
    terminal_consumer: dict[str, Any] | None = None
    started = time.perf_counter()
    for accepted_step in range(1, max(CHECKPOINT_STEPS) + 2):
        previous_state = state
        diagnostics: list[dict[str, Any]] = []
        segment = flowpipe_step_flowstar_style_adaptive(
            ode,
            current,
            h=H,
            h_min=H,
            h_max=H,
            order=ORDER,
            target_remainder_radius=TARGET,
            cutoff_threshold=CUTOFF,
            max_validation_attempts=2,
            validation_eps=1e-12,
            validation_mode="flowstar_raw_remainder_compat",
            reset_mode="normalized_insertion",
            step_policy_mode="flowstar_compat",
            flowstar_normal_state=state,
            tm_backend="dense",
            dense_device="cpu",
            dense_dtype=torch.float64,
            dense_range_policy=policy,
            diagnostics=diagnostics,
        )
        accepted = segment.status == "validated" and segment.reset_tm is not None and segment.flowstar_normal_state is not None
        if not accepted:
            terminal_consumer = {
                "attempted_after_accepted_steps": accepted_step - 1,
                "t_before": current_time,
                "status": segment.status,
                "message": segment.message,
                "candidate_remainder": segment.candidate_remainder,
                "picard_image_remainder": segment.picard_image_remainder,
                "subset_margin": segment.subset_margin,
            }
            break
        current_time = accepted_step * H
        if accepted_step in CHECKPOINT_STEPS:
            boundaries.append(
                _audit_boundary(
                    label=CHECKPOINT_STEPS[accepted_step],
                    accepted_step=accepted_step,
                    time_value=current_time,
                    segment=segment,
                    previous_state=previous_state,
                    ode=ode,
                    policy=policy,
                    output_dir=output_dir,
                )
            )
        current = segment.reset_tm
        state = segment.flowstar_normal_state
    if len(boundaries) != len(CHECKPOINT_STEPS):
        raise RuntimeError(f"captured {len(boundaries)} of {len(CHECKPOINT_STEPS)} required boundaries")
    all_same = all(
        len({row["same_input_sha256"] for row in boundary["cells"].values()}) == 1
        for boundary in boundaries
    )
    all_sound = all(
        all(
            component["oracle"]["production_remainder_contains_exact_bernstein_enclosure"]
            for cell in ("D", "H")
            for component in boundary["cells"][cell]["insertion_output"]
        )
        for boundary in boundaries
    )
    boundary_by_label = {boundary["label"]: boundary for boundary in boundaries}
    zero_remainder_control = (
        boundary_by_label["step_1_to_2"]["direct_nonlinear_path_remainder_consumptions"] == 0
    )
    first_nonzero_repetition = (
        boundary_by_label["first_nonzero_step_2_to_3"][
            "direct_nonlinear_path_remainder_consumptions"
        ]
        > 1
    )
    repeated_after_first_nonzero = all(
        boundary["direct_nonlinear_path_remainder_consumptions"] > 1
        for boundary in boundaries
        if boundary["accepted_step"] >= 2
    )
    summary = {
        "schema": "vdp_normal_insertion_gate_a_v2",
        "branch": _git("branch", "--show-current"),
        "base_tip": "e47ce68c61e73fc38f17fab3037d6cfe1877f3fd",
        "contract": {
            "ode": "x'=y, y'=y-x-x^2*y",
            "initial_box_exact_decimal": [["1.1", "1.4"], ["2.35", "2.45"]],
            "order": ORDER,
            "cutoff": CUTOFF,
            "fixed_h": H,
            "target": TARGET,
            "dtype": "float64",
            "device": "cpu",
        },
        "registered_d_one_path": {
            "component_index": REGISTERED_D_ONE_PATH[0],
            "outer_exponent": list(REGISTERED_D_ONE_PATH[1]),
        },
        "checkpoint_count": len(boundaries),
        "all_cells_byte_identical_inputs": all_same,
        "D_and_H_exact_rational_bernstein_containment_all_checkpoints": all_sound,
        "step_1_to_2_zero_inner_remainder_negative_control": zero_remainder_control,
        "first_nonzero_repeated_insertion_at_step_2_to_3": first_nonzero_repetition,
        "direct_repeated_nonlinear_remainder_consumption_after_first_nonzero": repeated_after_first_nonzero,
        "direct_repeated_nonlinear_remainder_consumption_constructively_reproduced": (
            zero_remainder_control and first_nonzero_repetition and repeated_after_first_nonzero
        ),
        "H1_gate_a_mechanism_pass": (
            all_same
            and all_sound
            and zero_remainder_control
            and first_nonzero_repetition
            and repeated_after_first_nonzero
        ),
        "terminal_next_consumer": terminal_consumer,
        "elapsed_s": time.perf_counter() - started,
        "boundaries": [
            {
                "label": boundary["label"],
                "accepted_step": boundary["accepted_step"],
                "time": boundary["time"],
                "same_prestate_sha256": boundary["same_prestate_sha256"],
                "direct_nonlinear_path_remainder_consumptions": boundary["direct_nonlinear_path_remainder_consumptions"],
                "horner_factorized_multiplications": boundary["horner_factorized_multiplications"],
                "deltas": boundary["deltas"],
                "relative_evidence": f"boundaries/{boundary['label']}.json",
            }
            for boundary in boundaries
        ],
    }
    _write_json(output_dir / "summary.json", summary)
    print(json.dumps({key: summary[key] for key in (
        "checkpoint_count",
        "all_cells_byte_identical_inputs",
        "D_and_H_exact_rational_bernstein_containment_all_checkpoints",
        "direct_repeated_nonlinear_remainder_consumption_constructively_reproduced",
        "H1_gate_a_mechanism_pass",
        "elapsed_s",
    )}, sort_keys=True))
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
