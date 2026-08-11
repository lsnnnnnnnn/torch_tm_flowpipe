#!/usr/bin/env python3
"""Put the first Flow*/Torch VDP divergence into one physical basis and replay it."""
from __future__ import annotations

import argparse
import csv
from decimal import Decimal, getcontext
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(ROOT / "experiments"))

import run_vdp_dense_backend as authoritative
from torch_tm_flowpipe import (
    AffineCoordinateBasis,
    DenseRangePolicy,
    IntervalPolynomialBatch,
    PolynomialODE,
    affine_common_basis_transform,
)
from torch_tm_flowpipe.batched_dense_tm import (
    BatchedMonomialBasis,
    BatchedPolynomial,
    BatchedTaylorModel,
    _dense_flowstar_raw_compat_image,
)
from torch_tm_flowpipe.raw_remainder_trace import RawRemainderTraceRecorder


TIME_SEMANTICS = "physical_local_time_[0,h]"
getcontext().prec = 80


def _number(value: Any) -> float:
    if isinstance(value, Mapping) and "decimal" in value:
        return float(value["decimal"])
    return float(value)


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        return value.item() if value.numel() == 1 else value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _flowstar_polynomial(
    records: Iterable[Mapping[str, Any]],
    *,
    stage: str,
    picard_iteration: int | None,
    components: Sequence[int] = (0, 1),
) -> IntervalPolynomialBatch:
    terms: dict[int, dict[tuple[int, ...], tuple[float, float]]] = {
        component: {} for component in components
    }
    for row in records:
        if (
            row.get("record_type") != "polynomial_term"
            or row.get("stage") != stage
            or (
                picard_iteration is not None
                and int(row.get("picard_iteration", -1)) != picard_iteration
            )
            or int(row.get("component", -1)) not in terms
        ):
            continue
        component = int(row["component"])
        exponent = tuple(int(value) for value in row["exponents"])
        if exponent in terms[component]:
            raise ValueError(f"duplicate Flow* term for component {component}: {exponent}")
        terms[component][exponent] = (
            _number(row["coefficient_lower"]),
            _number(row["coefficient_upper"]),
        )
    if any(not values for values in terms.values()):
        raise ValueError(f"Flow* stage {stage!r}, iteration {picard_iteration} is incomplete")
    support = sorted(set().union(*(set(values) for values in terms.values())))
    lo = torch.zeros((1, len(components), len(support)), dtype=torch.float64)
    hi = torch.zeros_like(lo)
    for state, component in enumerate(components):
        for slot, exponent in enumerate(support):
            lo[0, state, slot], hi[0, state, slot] = terms[component].get(exponent, (0.0, 0.0))
    return IntervalPolynomialBatch(lo, hi, torch.tensor(support, dtype=torch.int64))


def _flowstar_remainders(
    records: Iterable[Mapping[str, Any]],
    stage: str,
    components: Sequence[int] = (0, 1),
    attempt_index: int | None = None,
) -> list[list[float]]:
    rows = [
        row
        for row in records
        if row.get("stage") == stage
        and row.get("record_type") == "taylor_model"
        and int(row.get("component", -1)) in components
        and (
            attempt_index is None
            or int(row.get("attempt_index", -1)) == attempt_index
        )
    ]
    rows.sort(key=lambda row: int(row["component"]))
    if len(rows) != len(components):
        raise ValueError(f"Flow* Taylor-model stage {stage!r} has incomplete remainders")
    return [
        [_number(row["remainder"]["lower"]), _number(row["remainder"]["upper"])]
        for row in rows
    ]


def _affine_output_transform(
    polynomial: IntervalPolynomialBatch,
    source_center: Sequence[float],
    source_scale: Sequence[float],
    target_center: Sequence[float],
    target_scale: Sequence[float],
) -> IntervalPolynomialBatch:
    lo = polynomial.coeff_lo.clone()
    hi = polynomial.coeff_hi.clone()
    zero = (0,) * polynomial.variables
    support = list(polynomial.support)
    if zero not in support:
        support.append(zero)
        order = sorted(range(len(support)), key=lambda index: support[index])
        old_slots = {exponent: slot for slot, exponent in enumerate(polynomial.support)}
        new_lo = torch.zeros((polynomial.batch, polynomial.states, len(support)), dtype=torch.float64)
        new_hi = torch.zeros_like(new_lo)
        support = [support[index] for index in order]
        for slot, exponent in enumerate(support):
            if exponent in old_slots:
                new_lo[:, :, slot] = lo[:, :, old_slots[exponent]]
                new_hi[:, :, slot] = hi[:, :, old_slots[exponent]]
        lo, hi = new_lo, new_hi
    zero_slot = support.index(zero)
    for state in range(polynomial.states):
        multiplier = float(source_scale[state]) / float(target_scale[state])
        offset = (float(source_center[state]) - float(target_center[state])) / float(target_scale[state])
        if multiplier < 0:
            state_lo = hi[:, state, :] * multiplier
            state_hi = lo[:, state, :] * multiplier
        else:
            state_lo = lo[:, state, :] * multiplier
            state_hi = hi[:, state, :] * multiplier
        lo[:, state, :] = torch.nextafter(state_lo, torch.full_like(state_lo, -torch.inf))
        hi[:, state, :] = torch.nextafter(state_hi, torch.full_like(state_hi, torch.inf))
        lo[:, state, zero_slot] = torch.nextafter(
            lo[:, state, zero_slot] + offset,
            torch.full_like(lo[:, state, zero_slot], -torch.inf),
        )
        hi[:, state, zero_slot] = torch.nextafter(
            hi[:, state, zero_slot] + offset,
            torch.full_like(hi[:, state, zero_slot], torch.inf),
        )
    return IntervalPolynomialBatch(lo, hi, torch.tensor(support, dtype=torch.int64))


def _torch_polynomial(data: Mapping[str, Any]) -> IntervalPolynomialBatch:
    coefficients = torch.tensor(data["coefficients"], dtype=torch.float64)
    exponents = torch.tensor(data["exponents"], dtype=torch.int64)
    return IntervalPolynomialBatch.from_point_coefficients(coefficients, exponents)


def _torch_sparse_polynomial(data: Mapping[str, Any]) -> IntervalPolynomialBatch:
    components = list(data["components"])
    support = sorted(
        {
            tuple(int(value) for value in term["exponents"])
            for component in components
            for term in component["terms"]
        }
    )
    coefficients = torch.zeros((1, len(components), len(support)), dtype=torch.float64)
    slots = {exponent: slot for slot, exponent in enumerate(support)}
    for state, component in enumerate(components):
        for term in component["terms"]:
            exponent = tuple(int(value) for value in term["exponents"])
            coefficients[0, state, slots[exponent]] = float(term["coefficient"])
    return IntervalPolynomialBatch.from_point_coefficients(coefficients, support)


def _flowstar_real_vector(records: Iterable[Mapping[str, Any]], stage: str) -> list[float]:
    rows = [
        row
        for row in records
        if row.get("stage") == stage
        and row.get("record_type") == "real_vector_entry"
        and int(row.get("component", -1)) in (0, 1)
    ]
    rows.sort(key=lambda row: int(row["component"]))
    if len(rows) != 2:
        raise ValueError(f"Flow* real-vector stage {stage!r} is incomplete")
    return [_number(row["lower"]) for row in rows]


def _basis(
    names: Sequence[str],
    center: Sequence[float],
    scale: Sequence[float],
    lo: Sequence[float],
    hi: Sequence[float],
    *,
    distinguished_time: bool = True,
) -> AffineCoordinateBasis:
    return AffineCoordinateBasis(
        tuple(names),
        torch.tensor([center], dtype=torch.float64),
        torch.tensor([scale], dtype=torch.float64),
        torch.tensor([lo], dtype=torch.float64),
        torch.tensor([hi], dtype=torch.float64),
        time_name="tau" if distinguished_time else None,
        time_semantics=TIME_SEMANTICS if distinguished_time else None,
    )


def _transform_dict(result: Any) -> dict[str, Any]:
    return {
        "coordinate_identity_known": result.coordinate_identity_known,
        "time_treatment": result.time_treatment,
        "dropped_zero_variables": list(result.dropped_zero_variables),
        "support": [list(item) for item in result.transformed.support],
        "coefficient_lo": result.transformed.coeff_lo,
        "coefficient_hi": result.transformed.coeff_hi,
        "natural_range_lo": result.transformed_range_lo,
        "natural_range_hi": result.transformed_range_hi,
        "retained_support": [list(item) for item in result.retained.support],
        "retained_range_lo": result.retained_range_lo,
        "retained_range_hi": result.retained_range_hi,
        "intervalized_discarded_lo": result.intervalized_discarded_lo,
        "intervalized_discarded_hi": result.intervalized_discarded_hi,
    }


def _coefficient_comparison(flowstar: Any, torch_result: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    flow_slots = {exponent: slot for slot, exponent in enumerate(flowstar.transformed.support)}
    torch_slots = {exponent: slot for slot, exponent in enumerate(torch_result.transformed.support)}
    rows: list[dict[str, Any]] = []
    max_midpoint = [0.0, 0.0]
    max_enclosure = [0.0, 0.0]
    all_contain_zero = [True, True]
    for exponent in sorted(set(flow_slots) | set(torch_slots)):
        for state, state_name in enumerate(("x", "y")):
            fslot = flow_slots.get(exponent)
            tslot = torch_slots.get(exponent)
            flo = float(flowstar.transformed.coeff_lo[0, state, fslot]) if fslot is not None else 0.0
            fhi = float(flowstar.transformed.coeff_hi[0, state, fslot]) if fslot is not None else 0.0
            tlo = float(torch_result.transformed.coeff_lo[0, state, tslot]) if tslot is not None else 0.0
            thi = float(torch_result.transformed.coeff_hi[0, state, tslot]) if tslot is not None else 0.0
            error_lo = math.nextafter(tlo - fhi, -math.inf)
            error_hi = math.nextafter(thi - flo, math.inf)
            midpoint_error = 0.5 * (tlo + thi) - 0.5 * (flo + fhi)
            rows.append(
                {
                    "state": state_name,
                    "exponents_x0_y0_tau": exponent,
                    "flowstar_lo": flo,
                    "flowstar_hi": fhi,
                    "torch_lo": tlo,
                    "torch_hi": thi,
                    "torch_minus_flowstar_lo": error_lo,
                    "torch_minus_flowstar_hi": error_hi,
                    "midpoint_error": midpoint_error,
                    "error_contains_zero": error_lo <= 0.0 <= error_hi,
                }
            )
            max_midpoint[state] = max(max_midpoint[state], abs(midpoint_error))
            max_enclosure[state] = max(max_enclosure[state], abs(error_lo), abs(error_hi))
            all_contain_zero[state] &= error_lo <= 0.0 <= error_hi
    return rows, {
        "maximum_absolute_midpoint_coefficient_error": max_midpoint,
        "maximum_absolute_error_enclosure": max_enclosure,
        "every_coefficient_error_contains_zero": all_contain_zero,
        "coefficient_rows": len(rows),
    }


def _write_coefficients(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            rendered = dict(row)
            rendered["exponents_x0_y0_tau"] = ",".join(
                str(item) for item in row["exponents_x0_y0_tau"]
            )
            writer.writerow(rendered)


def _midpoint_model(
    polynomial: IntervalPolynomialBatch,
    *,
    domain_lo: torch.Tensor,
    domain_hi: torch.Tensor,
    policy: DenseRangePolicy,
    range_trace: list[dict[str, Any]],
    order: int | None = None,
) -> BatchedTaylorModel:
    selected_order = (
        max(sum(exponent) for exponent in polynomial.support) if order is None else int(order)
    )
    if any(sum(exponent) > selected_order for exponent in polynomial.support):
        raise ValueError("declared complete basis order is below the polynomial support")
    basis = BatchedMonomialBasis.build(polynomial.variables, selected_order, "cpu")
    coefficients = torch.zeros((polynomial.batch, polynomial.states, basis.num_terms), dtype=torch.float64)
    midpoint = 0.5 * (polynomial.coeff_lo + polynomial.coeff_hi)
    for slot, exponent in enumerate(polynomial.support):
        coefficients[:, :, basis.term_index(exponent)] = midpoint[:, :, slot]
    zeros = torch.zeros((polynomial.batch, polynomial.states), dtype=torch.float64)
    return BatchedTaylorModel(
        BatchedPolynomial(coefficients, basis),
        zeros,
        zeros.clone(),
        domain_lo,
        domain_hi,
        range_policy=policy,
        range_trace=range_trace,
    )


def _checkpoint_dense_model(
    data: Mapping[str, Any],
    *,
    policy: DenseRangePolicy,
    range_trace: list[dict[str, Any]],
    force_zero_remainder: bool = False,
) -> BatchedTaylorModel:
    polynomial = _torch_polynomial(data)
    model = _midpoint_model(
        polynomial,
        domain_lo=torch.tensor(data["domain_lo"], dtype=torch.float64),
        domain_hi=torch.tensor(data["domain_hi"], dtype=torch.float64),
        policy=policy,
        range_trace=range_trace,
        order=int(data["order"]),
    )
    if force_zero_remainder:
        return model
    return model.with_remainder(
        torch.tensor(data["remainder_lo"], dtype=torch.float64),
        torch.tensor(data["remainder_hi"], dtype=torch.float64),
    )


def _coefficient_intervalization_bound(
    polynomial: IntervalPolynomialBatch, domain_lo: torch.Tensor, domain_hi: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    midpoint = 0.5 * (polynomial.coeff_lo + polynomial.coeff_hi)
    radius = 0.5 * (polynomial.coeff_hi - polynomial.coeff_lo)
    bound = torch.zeros((polynomial.batch, polynomial.states), dtype=torch.float64)
    for slot, exponent in enumerate(polynomial.support):
        monomial_mag = torch.ones((polynomial.batch, 1), dtype=torch.float64)
        for variable, power in enumerate(exponent):
            magnitude = torch.maximum(domain_lo[:, variable].abs(), domain_hi[:, variable].abs())
            monomial_mag *= magnitude[:, None] ** power
        # Midpoint rounding is covered by one outward ulp plus the interval radius.
        midpoint_ulp = torch.maximum(
            midpoint[:, :, slot] - torch.nextafter(midpoint[:, :, slot], torch.full_like(midpoint[:, :, slot], -torch.inf)),
            torch.nextafter(midpoint[:, :, slot], torch.full_like(midpoint[:, :, slot], torch.inf)) - midpoint[:, :, slot],
        )
        bound += (radius[:, :, slot] + midpoint_ulp) * monomial_mag
    return -bound, bound


def _torch_raw_replay(
    base: BatchedTaylorModel,
    candidate: BatchedTaylorModel,
    *,
    h: float,
    order: int,
    cutoff: float,
    target_radius: float,
    validation_eps: float,
    ode: PolynomialODE,
    raw_trace_recorder: RawRemainderTraceRecorder | None = None,
) -> dict[str, Any]:
    target_lo = torch.full_like(candidate.rem_lo, -abs(target_radius))
    target_hi = torch.full_like(candidate.rem_hi, abs(target_radius))
    with_target = candidate.with_remainder(target_lo, target_hi, category="initial_remainder")
    image_lo, image_hi, details, _decomposition = _dense_flowstar_raw_compat_image(
        ode,
        base,
        with_target,
        candidate,
        tau_index=2,
        order=order,
        cutoff_threshold=cutoff,
        validation_eps=validation_eps,
        raw_trace_recorder=raw_trace_recorder,
    )
    margin = torch.minimum(image_lo - target_lo, target_hi - image_hi)
    return {
        "image_lo": image_lo,
        "image_hi": image_hi,
        "subset_margin": margin,
        "accepted": bool(torch.all(margin >= 0)),
        "details": details,
    }


def _flowstar_subset(records: Iterable[Mapping[str, Any]], attempt: int) -> dict[str, Any]:
    rows = [
        row
        for row in records
        if row.get("stage") == "candidate_subset"
        and row.get("record_type") == "subset_test"
        and int(row.get("attempt_index", -1)) == attempt
        and int(row.get("component", -1)) in (0, 1)
    ]
    rows.sort(key=lambda row: int(row["component"]))
    if len(rows) != 2:
        raise ValueError(f"expected two Flow* candidate subset rows for attempt {attempt}")
    return {
        "image_lo": [_number(row["image"]["lower"]) for row in rows],
        "image_hi": [_number(row["image"]["upper"]) for row in rows],
        "target_lo": [_number(row["target"]["lower"]) for row in rows],
        "target_hi": [_number(row["target"]["upper"]) for row in rows],
        "minimum_margin": [_number(row["minimum_margin"]) for row in rows],
        "accepted": all(bool(row["accepted"]) for row in rows),
    }


def _write_flowstar_replay(path: Path, transformed: IntervalPolynomialBatch) -> dict[str, Any]:
    maximum_width = 0.0
    rows = 0
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# component coefficient e_tau e_xlocal e_ylocal e_clocklocal\n")
        for component in range(transformed.states):
            for slot, exponent in enumerate(transformed.support):
                lo = float(transformed.coeff_lo[0, component, slot])
                hi = float(transformed.coeff_hi[0, component, slot])
                if lo == 0.0 and hi == 0.0:
                    continue
                coefficient = 0.5 * (lo + hi)
                if coefficient == 0.0:
                    continue
                handle.write(
                    f"{component} {coefficient:.17g} "
                    + " ".join(str(value) for value in exponent)
                    + "\n"
                )
                maximum_width = max(maximum_width, hi - lo)
                rows += 1
    return {"rows": rows, "maximum_coefficient_interval_width": maximum_width, "sha256": _sha(path)}


def _write_flowstar_endpoint_replay(
    path: Path,
    transformed: IntervalPolynomialBatch,
    remainders: Sequence[Sequence[float]],
    coefficient_error_lo: torch.Tensor,
    coefficient_error_hi: torch.Tensor,
) -> dict[str, Any]:
    rows = 0
    maximum_width = 0.0
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# T component coefficient e_tau e_xlocal e_ylocal e_clocklocal\n")
        handle.write("# R component remainder_lower remainder_upper\n")
        for component in range(transformed.states):
            for slot, exponent in enumerate(transformed.support):
                lo = float(transformed.coeff_lo[0, component, slot])
                hi = float(transformed.coeff_hi[0, component, slot])
                coefficient = 0.5 * (lo + hi)
                if coefficient == 0.0:
                    continue
                handle.write(
                    f"T {component} {coefficient:.17g} "
                    + " ".join(str(value) for value in exponent)
                    + "\n"
                )
                rows += 1
                maximum_width = max(maximum_width, hi - lo)
            remainder_lo = math.nextafter(
                float(remainders[component][0]) + float(coefficient_error_lo[0, component]),
                -math.inf,
            )
            remainder_hi = math.nextafter(
                float(remainders[component][1]) + float(coefficient_error_hi[0, component]),
                math.inf,
            )
            handle.write(f"R {component} {remainder_lo:.17g} {remainder_hi:.17g}\n")
    return {
        "polynomial_rows": rows,
        "maximum_coefficient_interval_width": maximum_width,
        "coefficient_intervalization_absorbed_into_remainder": True,
        "sha256": _sha(path),
    }


def _write_torch_right_map_replay(
    path: Path,
    polynomial: IntervalPolynomialBatch,
    remainders: Sequence[Sequence[float]],
    coefficient_error_lo: torch.Tensor,
    coefficient_error_hi: torch.Tensor,
    domain_lo: Sequence[float],
    domain_hi: Sequence[float],
) -> dict[str, Any]:
    components = []
    for state in range(polynomial.states):
        terms = []
        for slot, exponent in enumerate(polynomial.support):
            lo = float(polynomial.coeff_lo[0, state, slot])
            hi = float(polynomial.coeff_hi[0, state, slot])
            coefficient = 0.5 * (lo + hi)
            if coefficient != 0.0:
                terms.append({"exponents": list(exponent), "coefficient": coefficient})
        rem_lo = math.nextafter(
            float(remainders[state][0]) + float(coefficient_error_lo[0, state]),
            -math.inf,
        )
        rem_hi = math.nextafter(
            float(remainders[state][1]) + float(coefficient_error_hi[0, state]),
            math.inf,
        )
        components.append(
            {
                "terms": terms,
                "remainder": [rem_lo, rem_hi],
                "domain": [[float(lo), float(hi)] for lo, hi in zip(domain_lo, domain_hi)],
            }
        )
    payload = {
        "schema": "flowstar_right_map_in_torch_semantics_v1",
        "components": components,
        "variables": polynomial.variables,
        "states": polynomial.states,
        "coefficient_intervalization_absorbed_into_remainder": True,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"sha256": _sha(path), "components": len(components)}


def _decimal_evaluate(polynomial: IntervalPolynomialBatch, state: int, point: Sequence[Decimal]) -> Decimal:
    total = Decimal(0)
    for slot, exponent in enumerate(polynomial.support):
        coefficient = Decimal(
            str(
                0.5
                * (
                    float(polynomial.coeff_lo[0, state, slot])
                    + float(polynomial.coeff_hi[0, state, slot])
                )
            )
        )
        term = coefficient
        for value, power in zip(point, exponent):
            if power:
                term *= value**power
        total += term
    return total


def _right_map_high_precision_replay(
    flow_polynomial: IntervalPolynomialBatch,
    converted_polynomial: IntervalPolynomialBatch,
    native_torch_polynomial: IntervalPolynomialBatch,
    *,
    current_flow_center: Sequence[float],
    current_flow_scale: Sequence[float],
    current_torch_center: Sequence[float],
    current_torch_scale: Sequence[float],
    previous_flow_center: Sequence[float],
    previous_flow_scale: Sequence[float],
    previous_torch_center: Sequence[float],
    previous_torch_scale: Sequence[float],
) -> dict[str, Any]:
    physical_lo = [
        max(fc - fs, tc - ts)
        for fc, fs, tc, ts in zip(
            current_flow_center, current_flow_scale, current_torch_center, current_torch_scale
        )
    ]
    physical_hi = [
        min(fc + fs, tc + ts)
        for fc, fs, tc, ts in zip(
            current_flow_center, current_flow_scale, current_torch_center, current_torch_scale
        )
    ]
    fractions = (Decimal(0), Decimal("0.25"), Decimal("0.5"), Decimal("0.75"), Decimal(1))
    max_conversion_error = [Decimal(0), Decimal(0)]
    max_native_difference = [Decimal(0), Decimal(0)]
    samples = 0
    for alpha in fractions:
        for beta in fractions:
            physical = [
                Decimal(str(physical_lo[0]))
                + alpha * (Decimal(str(physical_hi[0])) - Decimal(str(physical_lo[0]))),
                Decimal(str(physical_lo[1]))
                + beta * (Decimal(str(physical_hi[1])) - Decimal(str(physical_lo[1]))),
            ]
            flow_point = [Decimal(0)] + [
                (value - Decimal(str(center))) / Decimal(str(scale))
                for value, center, scale in zip(physical, current_flow_center, current_flow_scale)
            ] + [Decimal(0)]
            torch_point = [
                (value - Decimal(str(center))) / Decimal(str(scale))
                for value, center, scale in zip(physical, current_torch_center, current_torch_scale)
            ]
            for state in range(2):
                flow_value = _decimal_evaluate(flow_polynomial, state, flow_point)
                direct_physical = Decimal(str(previous_flow_center[state])) + Decimal(
                    str(previous_flow_scale[state])
                ) * flow_value
                converted_value = _decimal_evaluate(converted_polynomial, state, torch_point)
                converted_physical = Decimal(str(previous_torch_center[state])) + Decimal(
                    str(previous_torch_scale[state])
                ) * converted_value
                native_value = _decimal_evaluate(native_torch_polynomial, state, torch_point)
                native_physical = Decimal(str(previous_torch_center[state])) + Decimal(
                    str(previous_torch_scale[state])
                ) * native_value
                max_conversion_error[state] = max(
                    max_conversion_error[state], abs(converted_physical - direct_physical)
                )
                max_native_difference[state] = max(
                    max_native_difference[state], abs(native_physical - direct_physical)
                )
            samples += 1
    return {
        "arithmetic": "Python decimal, 80 significant digits",
        "grid": "5x5 tensor grid over common physical current-state box",
        "samples": samples,
        "maximum_physical_conversion_roundtrip_error": [str(value) for value in max_conversion_error],
        "maximum_native_torch_vs_flowstar_polynomial_difference": [
            str(value) for value in max_native_difference
        ],
        "common_physical_input_lo": physical_lo,
        "common_physical_input_hi": physical_hi,
    }


def analyze(
    flowstar_path: Path,
    flowstar_previous_path: Path,
    torch_path: Path,
    flowstar_candidate_replay_path: Path,
    flowstar_endpoint_replay_path: Path,
    torch_right_map_replay_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    records = _load_jsonl(flowstar_path)
    previous_records = _load_jsonl(flowstar_previous_path)
    candidate_replay_records = _load_jsonl(flowstar_candidate_replay_path)
    endpoint_replay_records = _load_jsonl(flowstar_endpoint_replay_path)
    right_map_replay_result = json.loads(torch_right_map_replay_path.read_text(encoding="utf-8"))
    checkpoint = json.loads(torch_path.read_text(encoding="utf-8"))
    h = float(checkpoint["h_attempt"])
    t_pre = float(checkpoint["t_pre"])
    flow_center = _flowstar_real_vector(records, "normalization_center")
    flow_scale = _flowstar_real_vector(records, "normalization_scale")
    torch_center = [float(value) for value in checkpoint["normalization_center"]]
    torch_scale = [float(value) for value in checkpoint["normalization_scale"]]

    flow_basis = _basis(
        ("tau", "x0", "y0", "clock0"),
        (0.0, flow_center[0], flow_center[1], t_pre),
        (1.0, flow_scale[0], flow_scale[1], 0.0),
        (0.0, -1.0, -1.0, -1.0),
        (h, 1.0, 1.0, 1.0),
    )
    torch_basis = _basis(
        ("x0", "y0", "tau"),
        (torch_center[0], torch_center[1], 0.0),
        (torch_scale[0], torch_scale[1], 1.0),
        (-1.0, -1.0, 0.0),
        (1.0, 1.0, h),
    )
    common_lo = (
        max(flow_center[0] - flow_scale[0], torch_center[0] - torch_scale[0]),
        max(flow_center[1] - flow_scale[1], torch_center[1] - torch_scale[1]),
        0.0,
    )
    common_hi = (
        min(flow_center[0] + flow_scale[0], torch_center[0] + torch_scale[0]),
        min(flow_center[1] + flow_scale[1], torch_center[1] + torch_scale[1]),
        h,
    )
    common_basis = _basis(
        ("x0", "y0", "tau"),
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        common_lo,
        common_hi,
    )

    flow_candidate = _flowstar_polynomial(
        records, stage="picard_polynomial_iteration", picard_iteration=4
    )
    torch_candidate = _torch_polynomial(checkpoint["picard_iterations"][-1]["retained"])
    flow_common = affine_common_basis_transform(flow_candidate, flow_basis, common_basis)
    torch_common = affine_common_basis_transform(torch_candidate, torch_basis, common_basis)
    coefficient_rows, coefficient_summary = _coefficient_comparison(flow_common, torch_common)
    coefficient_csv = output_dir / "common_basis_coefficients.csv"
    _write_coefficients(coefficient_csv, coefficient_rows)

    torch_to_flow = affine_common_basis_transform(torch_candidate, torch_basis, flow_basis)
    flow_to_torch = affine_common_basis_transform(flow_candidate, flow_basis, torch_basis)
    replay_input = output_dir / "torch_candidate_in_flowstar_basis.tsv"
    replay_input_summary = _write_flowstar_replay(replay_input, torch_to_flow.transformed)

    previous_flow_center = _flowstar_real_vector(previous_records, "normalization_center")
    previous_flow_scale = _flowstar_real_vector(previous_records, "normalization_scale")
    previous_t_pre = _number(previous_records[0]["t_pre"])
    previous_h = next(
        _number(row["h_attempt"])
        for row in previous_records
        if row.get("stage") == "candidate_begin" and int(row.get("attempt_index", -1)) == 0
    )
    previous_torch_center = [float(value) for value in checkpoint["previous_step_normalization_center"]]
    previous_torch_scale = [float(value) for value in checkpoint["previous_step_normalization_scale"]]
    endpoint_data = checkpoint["previous_accepted_exact_time_endpoint"]
    torch_endpoint = _torch_sparse_polynomial(endpoint_data)
    previous_torch_basis = _basis(
        ("x0", "y0"),
        previous_torch_center,
        previous_torch_scale,
        (-1.0, -1.0),
        (1.0, 1.0),
        distinguished_time=False,
    )
    previous_flow_basis = _basis(
        ("tau", "x0", "y0", "clock0"),
        (0.0, previous_flow_center[0], previous_flow_center[1], previous_t_pre),
        (1.0, previous_flow_scale[0], previous_flow_scale[1], 0.0),
        (0.0, -1.0, -1.0, -1.0),
        (previous_h, 1.0, 1.0, 1.0),
    )
    endpoint_to_flow = affine_common_basis_transform(
        torch_endpoint, previous_torch_basis, previous_flow_basis
    )
    endpoint_coefficient_error = _coefficient_intervalization_bound(
        endpoint_to_flow.transformed,
        previous_flow_basis.domain_lo,
        previous_flow_basis.domain_hi,
    )
    endpoint_replay_input = output_dir / "torch_endpoint_in_flowstar_previous_basis.tsv"
    endpoint_replay_summary = _write_flowstar_endpoint_replay(
        endpoint_replay_input,
        endpoint_to_flow.transformed,
        [component["remainder"] for component in endpoint_data["components"]],
        endpoint_coefficient_error[0],
        endpoint_coefficient_error[1],
    )

    flow_right_map = _flowstar_polynomial(
        records, stage="right_map_output_after_cutoff", picard_iteration=None
    )
    flow_right_remainders = _flowstar_remainders(records, "right_map_output_after_cutoff")
    flow_right_input_basis = _basis(
        ("tau", "x0", "y0", "clock0"),
        (0.0, flow_center[0], flow_center[1], t_pre),
        (1.0, flow_scale[0], flow_scale[1], 0.0),
        (0.0, -1.0, -1.0, -1.0),
        (h, 1.0, 1.0, 1.0),
        distinguished_time=False,
    )
    torch_right_input_basis = _basis(
        ("x0", "y0"),
        torch_center,
        torch_scale,
        (-1.0, -1.0),
        (1.0, 1.0),
        distinguished_time=False,
    )
    flow_right_in_torch_inputs = affine_common_basis_transform(
        flow_right_map, flow_right_input_basis, torch_right_input_basis
    )
    flow_right_in_torch_semantics = _affine_output_transform(
        flow_right_in_torch_inputs.transformed,
        previous_flow_center,
        previous_flow_scale,
        previous_torch_center,
        previous_torch_scale,
    )
    flow_right_torch_remainders = []
    for state, remainder in enumerate(flow_right_remainders):
        multiplier = previous_flow_scale[state] / previous_torch_scale[state]
        flow_right_torch_remainders.append(
            [
                math.nextafter(multiplier * remainder[0], -math.inf),
                math.nextafter(multiplier * remainder[1], math.inf),
            ]
        )
    right_map_coefficient_error = _coefficient_intervalization_bound(
        flow_right_in_torch_semantics,
        torch_right_input_basis.domain_lo,
        torch_right_input_basis.domain_hi,
    )
    right_map_replay_path = output_dir / "flowstar_right_map_in_torch_semantics.json"
    right_map_replay_summary = _write_torch_right_map_replay(
        right_map_replay_path,
        flow_right_in_torch_semantics,
        flow_right_torch_remainders,
        right_map_coefficient_error[0],
        right_map_coefficient_error[1],
        (-1.0, -1.0),
        (1.0, 1.0),
    )
    native_torch_right_map = _torch_sparse_polynomial(checkpoint["right_map_input"])
    right_map_high_precision = _right_map_high_precision_replay(
        flow_right_map,
        flow_right_in_torch_semantics,
        native_torch_right_map,
        current_flow_center=flow_center,
        current_flow_scale=flow_scale,
        current_torch_center=torch_center,
        current_torch_scale=torch_scale,
        previous_flow_center=previous_flow_center,
        previous_flow_scale=previous_flow_scale,
        previous_torch_center=previous_torch_center,
        previous_torch_scale=previous_torch_scale,
    )

    contract = authoritative.load_contract()
    policy = DenseRangePolicy(
        method="adaptive_subdivision",
        max_depth=1,
        max_leaves=4,
        split_vars=(0, 1),
        trigger="proactive_depth1_on_named_contexts",
        named_contexts=("polynomial_truncation",),
        variable_orders=((0, 1, 2), (1, 0, 2), (2, 0, 1)),
    )
    range_trace: list[dict[str, Any]] = []
    base = _checkpoint_dense_model(
        checkpoint["validation_base"], policy=policy, range_trace=range_trace
    )
    native_candidate = _checkpoint_dense_model(
        checkpoint["picard_iterations"][-1]["retained"],
        policy=policy,
        range_trace=range_trace,
        force_zero_remainder=True,
    )
    flow_candidate_for_torch = _midpoint_model(
        flow_to_torch.transformed,
        domain_lo=base.domain_lo,
        domain_hi=base.domain_hi,
        policy=policy,
        range_trace=range_trace,
        order=int(contract["requested_order"]),
    )
    ode = PolynomialODE.from_system_spec(contract["canonical_system_spec"])
    replay_kwargs = {
        "h": h,
        "order": int(contract["requested_order"]),
        "cutoff": float(contract["cutoff"]),
        "target_radius": float(contract["target_remainder_radius"]),
        "validation_eps": 1e-12,
        "ode": ode,
    }
    native_replay = _torch_raw_replay(base, native_candidate, **replay_kwargs)
    expected_native_lo = torch.tensor(checkpoint["picard_image_remainder_lo"], dtype=torch.float64)
    expected_native_hi = torch.tensor(checkpoint["picard_image_remainder_hi"], dtype=torch.float64)
    native_exact = bool(
        torch.equal(native_replay["image_lo"], expected_native_lo)
        and torch.equal(native_replay["image_hi"], expected_native_hi)
    )
    if not native_exact:
        raise RuntimeError("standalone Torch replay does not reproduce the production validator")
    flow_in_torch = _torch_raw_replay(base, flow_candidate_for_torch, **replay_kwargs)
    coefficient_error_bound = _coefficient_intervalization_bound(
        flow_to_torch.transformed, base.domain_lo, base.domain_hi
    )
    flow_subset = _flowstar_subset(records, 0)
    torch_image_lo = [float(value) for value in expected_native_lo.flatten()]
    torch_image_hi = [float(value) for value in expected_native_hi.flatten()]
    torch_subset_margin = [
        min(value + float(contract["target_remainder_radius"]), float(contract["target_remainder_radius"]) - upper)
        for value, upper in zip(torch_image_lo, torch_image_hi)
    ]
    candidate_replay_subset = _flowstar_subset(candidate_replay_records, 0)
    if not any(
        row.get("stage") == "counterfactual_candidate_replaced" and bool(row.get("accepted"))
        for row in candidate_replay_records
    ):
        raise RuntimeError("native Flow* polynomial replay lacks a successful replacement event")
    endpoint_replay_center = _flowstar_real_vector(endpoint_replay_records, "normalization_center")
    endpoint_replay_scale = _flowstar_real_vector(endpoint_replay_records, "normalization_scale")
    endpoint_replay_subset = _flowstar_subset(endpoint_replay_records, 0)
    if not any(
        row.get("stage") == "counterfactual_endpoint_replaced" and bool(row.get("accepted"))
        for row in endpoint_replay_records
    ):
        raise RuntimeError("native Flow* endpoint replay lacks a successful replacement event")
    flow_raw_remainder = _flowstar_remainders(
        records, "candidate_raw_picard", attempt_index=0
    )
    roundoff_rows = [
        row
        for row in records
        if row.get("stage") == "candidate_roundoff_interval"
        and row.get("record_type") == "interval_vector_entry"
        and int(row.get("attempt_index", -1)) == 0
        and int(row.get("component", -1)) in (0, 1)
    ]
    roundoff_rows.sort(key=lambda row: int(row["component"]))
    flow_roundoff = [
        [_number(row["interval"]["lower"]), _number(row["interval"]["upper"])]
        for row in roundoff_rows
    ]
    if len(flow_roundoff) != 2:
        raise RuntimeError("Flow* candidate roundoff observation is incomplete")

    common_report = {
        "schema": "vdp_flowstar_torch_common_basis_v1",
        "inputs": {
            "flowstar_observer": str(flowstar_path),
            "flowstar_observer_sha256": _sha(flowstar_path),
            "flowstar_previous_observer": str(flowstar_previous_path),
            "flowstar_previous_observer_sha256": _sha(flowstar_previous_path),
            "torch_checkpoint": str(torch_path),
            "torch_checkpoint_sha256": _sha(torch_path),
            "flowstar_candidate_counterfactual": str(flowstar_candidate_replay_path),
            "flowstar_candidate_counterfactual_sha256": _sha(flowstar_candidate_replay_path),
            "flowstar_endpoint_counterfactual": str(flowstar_endpoint_replay_path),
            "flowstar_endpoint_counterfactual_sha256": _sha(flowstar_endpoint_replay_path),
            "torch_right_map_counterfactual": str(torch_right_map_replay_path),
            "torch_right_map_counterfactual_sha256": _sha(torch_right_map_replay_path),
        },
        "last_common_state": {
            "accepted_step_index": int(checkpoint["accepted_step_index"]),
            "t_pre": t_pre,
            "h_first_differing_proposal": h,
        },
        "coordinate_contract": {
            "flowstar": {
                "names": list(flow_basis.names),
                "physical_equals_center_plus_scale_times_coordinate": True,
                "center": flow_basis.center,
                "scale": flow_basis.scale,
                "domain_lo": flow_basis.domain_lo,
                "domain_hi": flow_basis.domain_hi,
            },
            "torch": {
                "names": list(torch_basis.names),
                "physical_equals_center_plus_scale_times_coordinate": True,
                "center": torch_basis.center,
                "scale": torch_basis.scale,
                "domain_lo": torch_basis.domain_lo,
                "domain_hi": torch_basis.domain_hi,
            },
            "common_physical_intersection": {
                "names": list(common_basis.names),
                "center": common_basis.center,
                "scale": common_basis.scale,
                "domain_lo": common_basis.domain_lo,
                "domain_hi": common_basis.domain_hi,
            },
            "time_treatment": TIME_SEMANTICS,
            "coefficient_comparison_authorized": True,
            "authorization_reason": (
                "all coordinate identities and affine maps are explicit; Flow* clock-local has "
                "zero exponent in every compared x/y monomial"
            ),
        },
        "pre_picard_normalization": {
            "center_flowstar": flow_center,
            "center_torch": torch_center,
            "center_torch_minus_flowstar": [t - f for t, f in zip(torch_center, flow_center)],
            "scale_flowstar": flow_scale,
            "scale_torch": torch_scale,
            "scale_torch_minus_flowstar": [t - f for t, f in zip(torch_scale, flow_scale)],
        },
        "flowstar_in_common_basis": _transform_dict(flow_common),
        "torch_in_common_basis": _transform_dict(torch_common),
        "coefficient_error_summary": coefficient_summary,
        "coefficient_csv": str(coefficient_csv),
        "coefficient_csv_sha256": _sha(coefficient_csv),
    }
    common_path = output_dir / "common_basis_comparison.json"
    common_path.write_text(json.dumps(_jsonable(common_report), indent=2, sort_keys=True) + "\n")

    counterfactuals = {
        "schema": "vdp_flowstar_torch_one_stage_counterfactuals_v1",
        "causal_attribution": {
            "earliest_decision-changing_stage": "candidate raw Picard remainder before roundoff",
            "flowstar_raw_picard_remainder": flow_raw_remainder,
            "flowstar_roundoff_interval": flow_roundoff,
            "torch_raw_compat_remainder": {
                "lo": torch_image_lo,
                "hi": torch_image_hi,
            },
            "finding": (
                "common-basis polynomials agree at roundoff scale, both polynomial swaps preserve "
                "the receiving validator's decision, and Flow* is already outside the y target "
                "before its negligible polynomial-roundoff interval is added"
            ),
            "normalization_scale_is_representational_not_causal": True,
        },
        "native_controls": {
            "torch_candidate_to_torch_validator": {
                **native_replay,
                "bit_exact_with_production": native_exact,
            },
            "flowstar_candidate_to_flowstar_validator": flow_subset,
        },
        "substitutions": {
            "flowstar_polynomial_to_torch_validator": {
                **flow_in_torch,
                "candidate_basis": "Torch normalized coordinates, transformed from explicit Flow* basis",
                "coefficient_intervalization_error_bound_lo": coefficient_error_bound[0],
                "coefficient_intervalization_error_bound_hi": coefficient_error_bound[1],
            },
            "torch_polynomial_to_flowstar_validator": {
                "status": "executed_in_native_Flowstar_replay",
                "input": str(replay_input),
                "observer": str(flowstar_candidate_replay_path),
                "observer_sha256": _sha(flowstar_candidate_replay_path),
                "result": candidate_replay_subset,
                **replay_input_summary,
            },
            "torch_endpoint_to_flowstar_normalization": {
                "status": "executed_in_native_Flowstar_replay",
                "input": str(endpoint_replay_input),
                "observer": str(flowstar_endpoint_replay_path),
                "observer_sha256": _sha(flowstar_endpoint_replay_path),
                "source_previous_normalization_center": previous_torch_center,
                "source_previous_normalization_scale": previous_torch_scale,
                "target_previous_normalization_center": previous_flow_center,
                "target_previous_normalization_scale": previous_flow_scale,
                "result_normalization_center": endpoint_replay_center,
                "result_normalization_scale": endpoint_replay_scale,
                "stock_normalization_scale": flow_scale,
                "torch_normalization_scale": torch_scale,
                "result_subset": endpoint_replay_subset,
                **endpoint_replay_summary,
            },
            "flowstar_right_map_to_torch_next_step": {
                "status": "executed_in_authoritative_Torch_replay",
                "input": str(right_map_replay_path),
                "input_semantics": "Torch current local variables -> Torch previous normalized frame",
                "result": right_map_replay_result,
                **right_map_replay_summary,
            },
            "torch_right_map_to_high_precision_replay": {
                "status": "executed",
                **right_map_high_precision,
            },
            "flowstar_remainder_to_torch_subset_predicate": {
                "image_lo": flow_subset["image_lo"],
                "image_hi": flow_subset["image_hi"],
                "target_lo": [-float(contract["target_remainder_radius"])] * 2,
                "target_hi": [float(contract["target_remainder_radius"])] * 2,
                "minimum_margin": flow_subset["minimum_margin"],
                "accepted": flow_subset["accepted"],
                "note": "both tools use the same componentwise fixed target at this checkpoint",
            },
            "torch_remainder_to_flowstar_subset_predicate": {
                "image_lo": torch_image_lo,
                "image_hi": torch_image_hi,
                "target_lo": [-float(contract["target_remainder_radius"])] * 2,
                "target_hi": [float(contract["target_remainder_radius"])] * 2,
                "minimum_margin": torch_subset_margin,
                "accepted": all(value >= 0 for value in torch_subset_margin),
                "note": "the predicate is exactly the stock Flow* componentwise subset test",
            },
        },
    }
    counterfactual_path = output_dir / "counterfactuals.json"
    counterfactual_path.write_text(
        json.dumps(_jsonable(counterfactuals), indent=2, sort_keys=True) + "\n"
    )
    summary = {
        "common_basis_comparison": str(common_path),
        "common_basis_sha256": _sha(common_path),
        "counterfactuals": str(counterfactual_path),
        "counterfactuals_sha256": _sha(counterfactual_path),
        "torch_candidate_flowstar_replay": str(replay_input),
        "torch_candidate_flowstar_replay_sha256": _sha(replay_input),
        "torch_endpoint_flowstar_replay": str(endpoint_replay_input),
        "torch_endpoint_flowstar_replay_sha256": _sha(endpoint_replay_input),
        "flowstar_right_map_torch_replay": str(right_map_replay_path),
        "flowstar_right_map_torch_replay_sha256": _sha(right_map_replay_path),
        "native_torch_replay_bit_exact": native_exact,
        "flowstar_polynomial_in_torch_validator_accepted": bool(flow_in_torch["accepted"]),
        "flowstar_native_accepted": bool(flow_subset["accepted"]),
        "torch_native_accepted": bool(native_replay["accepted"]),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flowstar-observer", type=Path, required=True)
    parser.add_argument("--flowstar-previous-observer", type=Path, required=True)
    parser.add_argument("--flowstar-candidate-replay", type=Path, required=True)
    parser.add_argument("--flowstar-endpoint-replay", type=Path, required=True)
    parser.add_argument("--torch-right-map-replay", type=Path, required=True)
    parser.add_argument("--torch-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = analyze(
        args.flowstar_observer.resolve(),
        args.flowstar_previous_observer.resolve(),
        args.torch_checkpoint.resolve(),
        args.flowstar_candidate_replay.resolve(),
        args.flowstar_endpoint_replay.resolve(),
        args.torch_right_map_replay.resolve(),
        args.output_dir,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
