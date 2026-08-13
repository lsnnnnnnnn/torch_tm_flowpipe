"""Pure derivations for the Flow*--Torch source/carry audit.

The routines in this module deliberately derive scientific labels from raw
bounds and runtime fields.  They do not accept a requested verdict and do not
use a subprocess exit code as a scientific result.
"""

from __future__ import annotations

from decimal import Decimal, localcontext
from fractions import Fraction
import json
import math
from typing import Any, Mapping, Sequence


FLOWSTAR_CHANNELS: dict[str, tuple[str, str]] = {
    "endpoint_x": ("flowstar_tau_h_endpoint_x_lo", "flowstar_tau_h_endpoint_x_hi"),
    "endpoint_y": ("flowstar_tau_h_endpoint_y_lo", "flowstar_tau_h_endpoint_y_hi"),
    "segment_tube_x": ("flowstar_full_step_tube_x_lo", "flowstar_full_step_tube_x_hi"),
    "segment_tube_y": ("flowstar_full_step_tube_y_lo", "flowstar_full_step_tube_y_hi"),
}

TORCH_CHANNELS: dict[str, tuple[str, str]] = {
    "endpoint_x": ("endpoint_x_lo", "endpoint_x_hi"),
    "endpoint_y": ("endpoint_y_lo", "endpoint_y_hi"),
    "segment_tube_x": ("segment_x_lo", "segment_x_hi"),
    "segment_tube_y": ("segment_y_lo", "segment_y_hi"),
}

EXPECTED_RATIOS = {
    1: {
        "endpoint_x": 0.9992883176901923,
        "endpoint_y": 0.9894036064925323,
        "segment_tube_x": 1.0000057168777448,
        "segment_tube_y": 1.000144318604266,
    },
    100: {
        "endpoint_x": 1.1061157688329692,
        "endpoint_y": 1.0243353113289309,
        "segment_tube_x": 1.1011186386824114,
        "segment_tube_y": 1.0743583953482423,
    },
    300: {
        "endpoint_x": 1.352002801307621,
        "endpoint_y": 1.431895840052862,
        "segment_tube_x": 1.29779101738795,
        "segment_tube_y": 1.3746227052696502,
    },
    632: {
        "endpoint_x": 5.987318478989109,
        "endpoint_y": 13.000127268946539,
        "segment_tube_x": 5.282654101161162,
        "segment_tube_y": 11.500892201495919,
    },
}


def finite_float(raw: str, *, field: str) -> float:
    if raw == "":
        raise ValueError(f"missing numeric field: {field}")
    value = float(raw)
    if not math.isfinite(value):
        raise ValueError(f"non-finite numeric field: {field}")
    return value


def accepted_flowstar_rows(rows: Sequence[Mapping[str, str]]) -> list[Mapping[str, str]]:
    accepted: list[Mapping[str, str]] = []
    failed = False
    for row_number, row in enumerate(rows, start=2):
        status = row.get("status", "")
        is_accepted = status == "accepted" and row.get("accepted") == "true"
        if is_accepted:
            if failed:
                raise ValueError("Flow* accepted a row after its first failure")
            accepted.append(row)
        elif status in {"rejected", "failed"}:
            failed = True
        else:
            raise ValueError(f"unknown Flow* status at CSV row {row_number}: {status!r}")
    for index, row in enumerate(accepted):
        if int(row.get("accepted_step_index", "-1")) != index:
            raise ValueError(f"non-contiguous Flow* accepted step index at {index}")
        if int(row.get("attempt_index_within_step", "-1")) != 1:
            raise ValueError(f"fixed schedule has a duplicate/misaligned attempt at {index}")
        expected_before = index * 0.01
        expected_after = (index + 1) * 0.01
        if finite_float(row.get("t_before", ""), field="t_before") != expected_before:
            raise ValueError(f"Flow* t_before mismatch at accepted step {index}")
        if finite_float(row.get("t_after", ""), field="t_after") != expected_after:
            raise ValueError(f"Flow* t_after mismatch at accepted step {index}")
        for lo_field, hi_field in FLOWSTAR_CHANNELS.values():
            lo = finite_float(row.get(lo_field, ""), field=lo_field)
            hi = finite_float(row.get(hi_field, ""), field=hi_field)
            if lo > hi:
                raise ValueError(f"inverted Flow* interval at accepted step {index}: {lo_field}")
    return accepted


def accepted_torch_rows(rows: Sequence[Mapping[str, str]]) -> list[Mapping[str, str]]:
    accepted: list[Mapping[str, str]] = []
    failed = False
    for row in rows:
        status = row.get("status", "")
        if status == "accepted":
            if failed:
                raise ValueError("Torch accepted a row after its first failure")
            accepted.append(row)
        elif status == "rejected":
            failed = True
        else:
            raise ValueError(f"unknown Torch status: {status!r}")
    for index, row in enumerate(accepted):
        if int(row.get("carry_step_index", "-1")) != index + 1:
            raise ValueError(f"non-contiguous Torch accepted step at {index + 1}")
        if int(row.get("segment_index", "-1")) != index:
            raise ValueError(f"non-contiguous Torch segment index at {index}")
        for lo_field, hi_field in TORCH_CHANNELS.values():
            lo = finite_float(row.get(lo_field, ""), field=lo_field)
            hi = finite_float(row.get(hi_field, ""), field=hi_field)
            if lo > hi:
                raise ValueError(f"inverted Torch interval at step {index + 1}: {lo_field}")
    return accepted


def interval_record(lo_raw: str, hi_raw: str) -> dict[str, Any]:
    """Return independently recomputed decimal and binary64 widths."""

    with localcontext() as context:
        context.prec = 80
        lo_decimal = Decimal(lo_raw)
        hi_decimal = Decimal(hi_raw)
        width_decimal = hi_decimal - lo_decimal
    if width_decimal < 0:
        raise ValueError("inverted decimal interval")
    lo = finite_float(lo_raw, field="lower")
    hi = finite_float(hi_raw, field="upper")
    width = hi - lo
    if width < 0 or not math.isfinite(width):
        raise ValueError("invalid binary64 width")
    smallest_normal = float.fromhex("0x1p-1022")
    return {
        "lower": lo,
        "upper": hi,
        "width": width,
        "lower_17e": f"{lo:.17e}",
        "upper_17e": f"{hi:.17e}",
        "width_17e": f"{width:.17e}",
        "lower_hex": lo.hex(),
        "upper_hex": hi.hex(),
        "width_hex": width.hex(),
        "lower_decimal": str(lo_decimal),
        "upper_decimal": str(hi_decimal),
        "width_decimal": str(width_decimal),
        "exact_zero": width_decimal == 0,
        "binary64_zero": width == 0.0,
        "subnormal": 0.0 < width < smallest_normal,
        "below_1e_16": width < 1e-16,
        "below_1e_12": width < 1e-12,
        "below_1e_9": width < 1e-9,
    }


def derive_width_minima(
    accepted: Sequence[Mapping[str, str]], *, context_radius: int = 20
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    minima: list[dict[str, Any]] = []
    contexts: list[dict[str, Any]] = []
    for channel, (lo_field, hi_field) in FLOWSTAR_CHANNELS.items():
        records = [interval_record(row[lo_field], row[hi_field]) for row in accepted]
        minimum_index = min(range(len(records)), key=lambda index: records[index]["width"])
        counts = {
            "exact_zero_count": sum(bool(record["exact_zero"]) for record in records),
            "binary64_zero_count": sum(bool(record["binary64_zero"]) for record in records),
            "subnormal_count": sum(bool(record["subnormal"]) for record in records),
            "below_1e_16_count": sum(bool(record["below_1e_16"]) for record in records),
            "below_1e_12_count": sum(bool(record["below_1e_12"]) for record in records),
            "below_1e_9_count": sum(bool(record["below_1e_9"]) for record in records),
        }
        minimum = {
            "channel": channel,
            "step": minimum_index + 1,
            "time": finite_float(accepted[minimum_index]["t_after"], field="t_after"),
            **records[minimum_index],
            **counts,
        }
        minima.append(minimum)
        start = max(0, minimum_index - context_radius)
        stop = min(len(accepted), minimum_index + context_radius + 1)
        for index in range(start, stop):
            contexts.append(
                {
                    "channel": channel,
                    "minimum_step": minimum_index + 1,
                    "relative_step": index - minimum_index,
                    "step": index + 1,
                    "time": finite_float(accepted[index]["t_after"], field="t_after"),
                    **records[index],
                }
            )
    return minima, contexts


def checkpoint_reproduction(
    flow_rows: Sequence[Mapping[str, str]],
    torch_rows: Sequence[Mapping[str, str]],
    *,
    ratio_tolerance: float = 5e-13,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(flow_rows) < max(EXPECTED_RATIOS) or len(torch_rows) < max(EXPECTED_RATIOS):
        raise ValueError("checkpoint reproduction inputs end before step 632")
    output: list[dict[str, Any]] = []
    deviations: list[float] = []
    for step, expected_channels in EXPECTED_RATIOS.items():
        flow = flow_rows[step - 1]
        torch = torch_rows[step - 1]
        for channel, expected in expected_channels.items():
            flo, fhi = FLOWSTAR_CHANNELS[channel]
            tlo, thi = TORCH_CHANNELS[channel]
            flow_width = finite_float(flow[fhi], field=fhi) - finite_float(flow[flo], field=flo)
            torch_width = finite_float(torch[thi], field=thi) - finite_float(torch[tlo], field=tlo)
            ratio = torch_width / flow_width
            deviation = abs(ratio - expected)
            deviations.append(deviation)
            output.append(
                {
                    "step": step,
                    "time": step * 0.01,
                    "channel": channel,
                    "flowstar_width": flow_width,
                    "torch_width": torch_width,
                    "absolute_excess": torch_width - flow_width,
                    "ratio": ratio,
                    "goal_ratio": expected,
                    "absolute_ratio_deviation": deviation,
                    "within_tolerance": deviation <= ratio_tolerance,
                }
            )
    maximum = max(deviations)
    reproduced = maximum <= ratio_tolerance
    verdict = {
        "status": (
            "BASELINE_CONCLUSIONS_REPRODUCED"
            if reproduced
            else "BASELINE_NOT_REPRODUCIBLE_STOP"
        ),
        "ratio_tolerance": ratio_tolerance,
        "maximum_absolute_ratio_deviation": maximum,
        "tolerance_basis": (
            "The replay and historical artifacts use identical decimal bounds; the tolerance "
            "only permits binary64 parsing/arithmetic ordering at the final ratio."
        ),
        "all_checkpoints_within_tolerance": reproduced,
    }
    return output, verdict


def _window_growth_change(widths: Sequence[float], radius: int = 20) -> dict[str, Any]:
    growth = [widths[index] - widths[index - 1] for index in range(1, len(widths))]
    candidates: list[tuple[float, int, float, float]] = []
    for index in range(radius, len(growth) - radius):
        before = sum(growth[index - radius : index]) / radius
        after = sum(growth[index : index + radius]) / radius
        candidates.append((after - before, index + 1, before, after))
    change, boundary, before, after = max(candidates)
    return {
        "window_radius_steps": radius,
        "largest_mean_growth_change_boundary_step": boundary + 1,
        "largest_mean_growth_change_time": (boundary + 1) * 0.01,
        "mean_growth_before": before,
        "mean_growth_after": after,
        "mean_growth_change": change,
    }


def growth_and_ratio_analysis(
    flow_rows: Sequence[Mapping[str, str]],
    torch_rows: Sequence[Mapping[str, str]],
    minima: Sequence[Mapping[str, Any]],
    *,
    guarded_ratio_threshold: float = 1e-9,
) -> dict[str, Any]:
    minima_by_channel = {str(row["channel"]): row for row in minima}
    channels: dict[str, Any] = {}
    for channel in FLOWSTAR_CHANNELS:
        flo, fhi = FLOWSTAR_CHANNELS[channel]
        tlo, thi = TORCH_CHANNELS[channel]
        flow_widths = [finite_float(row[fhi], field=fhi) - finite_float(row[flo], field=flo) for row in flow_rows[: len(torch_rows)]]
        torch_widths = [finite_float(row[thi], field=thi) - finite_float(row[tlo], field=tlo) for row in torch_rows]
        excess = [torch - flow for flow, torch in zip(flow_widths, torch_widths)]
        guarded = [
            (torch / flow if flow >= guarded_ratio_threshold else None)
            for flow, torch in zip(flow_widths, torch_widths)
        ]
        minimum_index = int(minima_by_channel[channel]["step"]) - 1
        prior_start = max(1, minimum_index - 20)
        prior_torch_growth = [
            torch_widths[index] - torch_widths[index - 1]
            for index in range(prior_start, minimum_index + 1)
        ]
        prior_excess_growth = [
            excess[index] - excess[index - 1]
            for index in range(prior_start, minimum_index + 1)
        ]
        channels[channel] = {
            "minimum_step": minimum_index + 1,
            "minimum_time": (minimum_index + 1) * 0.01,
            "flowstar_width_at_minimum": flow_widths[minimum_index],
            "torch_width_at_minimum": torch_widths[minimum_index],
            "absolute_excess_at_minimum": excess[minimum_index],
            "guarded_ratio_at_minimum": guarded[minimum_index],
            "torch_width_increased_every_prior_20_steps": all(value > 0 for value in prior_torch_growth),
            "absolute_excess_increased_every_prior_20_steps": all(value > 0 for value in prior_excess_growth),
            "ratio_was_guarded_at_minimum": guarded[minimum_index] is not None,
            **_window_growth_change(torch_widths),
        }
    minimum_times = sorted({float(value["minimum_time"]) for value in channels.values()})
    return {
        "guarded_ratio_threshold": guarded_ratio_threshold,
        "channels": channels,
        "minimum_times_are_identical": len(minimum_times) == 1,
        "distinct_minimum_times": minimum_times,
        "causal_interpretation": (
            "The tools run independently, so a Flow* projection minimum cannot alter Torch. "
            "The minima change the displayed ratio denominator; absolute widths and excess are "
            "the decision-relevant comparison."
        ),
    }


def runtime_feature_summary(
    flow_rows: Sequence[Mapping[str, str]],
    flow_metadata: Mapping[str, str],
    torch_summary: Mapping[str, Any],
) -> dict[str, Any]:
    queue_sizes = [int(row["symbolic_J_size"]) for row in flow_rows if row.get("symbolic_J_size", "")]
    active_after_first = any(size > 1 for size in queue_sizes[1:])
    return {
        "flowstar_symbolic_remainder_enabled": flow_metadata.get("symbolic_remainder_enabled") == "true",
        "flowstar_symbolic_queue_observed_active_after_first_step": active_after_first,
        "flowstar_symbolic_queue_max_observed_size": max(queue_sizes),
        "flowstar_expression_picard_observed": all(
            "Picard_ctrunc_normal" in row.get("flowstar_internal_intermediate_ranges_source_path", "")
            for row in flow_rows
        ),
        "flowstar_horner_normal_insertion_source_enabled": True,
        "flowstar_qr_preconditioning_observed": False,
        "flowstar_shrink_wrapping_observed": False,
        "flowstar_invariant_remainder_contraction_observed": False,
        "flowstar_range_outward_rounding": "MPFR_RNDD lower / MPFR_RNDU upper",
        "torch_reset_mode": torch_summary.get("reset_mode"),
        "torch_legacy_symbolic_queue_enabled": False,
        "torch_direct_monomial_insertion_source_enabled": torch_summary.get("reset_mode") == "normalized_insertion",
    }


def _fraction_interval_mul(
    left: tuple[Fraction, Fraction], right: tuple[Fraction, Fraction]
) -> tuple[Fraction, Fraction]:
    products = [a * b for a in left for b in right]
    return min(products), max(products)


def _fraction_interval_add(
    left: tuple[Fraction, Fraction], right: tuple[Fraction, Fraction]
) -> tuple[Fraction, Fraction]:
    return left[0] + right[0], left[1] + right[1]


def _fraction_record(value: tuple[Fraction, Fraction]) -> dict[str, str]:
    return {
        "lower": str(value[0]),
        "upper": str(value[1]),
        "width": str(value[1] - value[0]),
    }


def exact_semantics_micro_oracles() -> list[dict[str, Any]]:
    """Exact-rational dependency fixtures; these evaluate no float64 code."""

    delta = Fraction(1, 10)
    error = (-delta, delta)
    one = (Fraction(1), Fraction(1))
    x = (Fraction(1) - delta, Fraction(1) + delta)
    y = x

    affine_shared = error
    affine_intervalized = error

    quadratic_shared = (Fraction(0), delta * delta)
    z_squared = _fraction_interval_mul(x, x)
    minus_two_z = (-2 * x[1], -2 * x[0])
    quadratic_intervalized = _fraction_interval_add(
        _fraction_interval_add(z_squared, minus_two_z), one
    )

    cubic_intervalized = _fraction_interval_mul(_fraction_interval_mul(x, x), y)
    # x=1+e and y=1-e: f(e)=1+e-e^2-e^3.  Its derivative is positive on
    # [-1/10,1/10], so the exact extrema are the two endpoints.
    def cubic_shared_value(e: Fraction) -> Fraction:
        return Fraction(1) + e - e * e - e * e * e

    cubic_shared = (cubic_shared_value(-delta), cubic_shared_value(delta))

    fixtures = [
        ("affine_exact_carry", affine_shared, affine_intervalized),
        ("quadratic_shared_error_cancellation", quadratic_shared, quadratic_intervalized),
        ("cubic_x2y_shared_source_interaction", cubic_shared, cubic_intervalized),
    ]
    output: list[dict[str, Any]] = []
    for name, shared, intervalized in fixtures:
        contains = intervalized[0] <= shared[0] and intervalized[1] >= shared[1]
        output.append(
            {
                "fixture": name,
                "arithmetic": "exact_rational",
                "shared_source_range": _fraction_record(shared),
                "independently_intervalized_range": _fraction_record(intervalized),
                "intervalized_contains_shared": contains,
                "width_excess": str(
                    (intervalized[1] - intervalized[0]) - (shared[1] - shared[0])
                ),
            }
        )
    return output


def source_semantics_map_is_closed(rows: Sequence[Mapping[str, Any]]) -> bool:
    """Check that the minimum benchmark-to-output semantic path is present."""

    required = {
        "benchmark/model entry",
        "fixed-step reach loop",
        "cross-step carry decomposition",
        "normal polynomial composition",
        "TM multiplication remainder",
        "Picard/validator",
        "endpoint/tube range extraction",
        "serialization/parser/join",
    }
    stages = {str(row.get("mathematical_stage", "")) for row in rows}
    complete_rows = all(
        bool(row.get("flowstar_source"))
        and bool(row.get("torch_source"))
        and bool(row.get("dependency_consequence"))
        for row in rows
    )
    return required <= stages and complete_rows and any(
        bool(row.get("first_unequal")) for row in rows
    )


def derive_same_prestate_gate(
    *, coefficient_export: str, symbolic_queue_exported: bool, import_path_available: bool
) -> dict[str, Any]:
    lossless = (
        coefficient_export == "binary_exact"
        and symbolic_queue_exported
        and import_path_available
    )
    return {
        "status": (
            "SAME_PRESTATE_LOSSLESS_BRIDGE_AVAILABLE"
            if lossless
            else "SAME_PRESTATE_LOSSLESS_BRIDGE_NOT_AVAILABLE"
        ),
        "lossless_full_prestate_bridge": lossless,
        "coefficient_export": coefficient_export,
        "symbolic_queue_exported": symbolic_queue_exported,
        "import_path_available": import_path_available,
    }


def derive_package_verification(
    audit_verification: Mapping[str, Any], high_precision_summary: Mapping[str, Any]
) -> dict[str, Any]:
    """Aggregate scientific outputs without consulting subprocess exit codes."""

    outcome = audit_verification.get("outcome")
    if not isinstance(outcome, Mapping):
        raise ValueError("audit outcome object missing")
    statuses = outcome.get("statuses")
    if not isinstance(statuses, list) or not statuses:
        raise ValueError("audit statuses missing")
    falsification = high_precision_summary.get("falsification_result")
    proof_status = high_precision_summary.get("proof_status")
    if not isinstance(falsification, str) or not isinstance(proof_status, str):
        raise ValueError("high-precision result fields missing")
    return {
        "schema": "flowstar_torch_source_carry_package_verification_v1",
        "scientific_statuses": list(statuses),
        "width_classification": audit_verification.get("width_classification"),
        "same_prestate_gate": audit_verification.get("same_prestate_gate"),
        "candidate": audit_verification.get("candidate"),
        "numerical_falsification_result": falsification,
        "numerical_proof_status": proof_status,
        "scientific_outcome_uses_process_exit_code": False,
    }


def derive_scientific_outcome(
    *,
    baseline_verdict: Mapping[str, Any],
    minima: Sequence[Mapping[str, Any]],
    runtime_features: Mapping[str, Any],
    source_map_closed: bool,
    lossless_full_prestate_bridge: bool,
    independent_candidate_oracle_closed: bool,
    flowstar_soundness_gate_closed: bool,
) -> dict[str, Any]:
    minima_positive = all(
        int(row["exact_zero_count"]) == 0
        and int(row["below_1e_9_count"]) == 0
        and float(row["width"]) >= 1e-9
        for row in minima
    )
    runtime_delta = all(
        bool(runtime_features.get(key))
        for key in (
            "flowstar_symbolic_queue_observed_active_after_first_step",
            "flowstar_horner_normal_insertion_source_enabled",
            "torch_direct_monomial_insertion_source_enabled",
        )
    )
    source_localized = bool(source_map_closed and runtime_delta)
    candidate_authorized = bool(
        source_localized
        and lossless_full_prestate_bridge
        and independent_candidate_oracle_closed
        and flowstar_soundness_gate_closed
    )
    statuses = [str(baseline_verdict["status"])]
    statuses.append(
        "FLOWSTAR_WIDTH_IS_POSITIVE_NEAR_ZERO"
        if minima_positive
        else "FLOWSTAR_OUTPUT_PIPELINE_ARTIFACT"
    )
    statuses.append(
        "SOURCE_LEVEL_DEPENDENCY_LOSS_LOCALIZED"
        if source_localized
        else "SOURCE_LEVEL_DELTA_UNRESOLVED"
    )
    statuses.append("SOUND_CARRY_CANDIDATE_L1" if candidate_authorized else "NO_FIX_AUTHORIZED")
    return {
        "statuses": statuses,
        "zero_width_classification": (
            "Z0_POSITIVE_WIDTH_ONLY_LOOKS_ZERO" if minima_positive else "Z5_UNRESOLVED"
        ),
        "source_localized": source_localized,
        "candidate_authorized": candidate_authorized,
        "gates": {
            "lossless_full_prestate_bridge": lossless_full_prestate_bridge,
            "independent_candidate_oracle_closed": independent_candidate_oracle_closed,
            "flowstar_soundness_gate_closed": flowstar_soundness_gate_closed,
        },
    }


def parse_json_cell(raw: str, *, field: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON field {field}") from error
