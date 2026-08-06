"""Fail-closed helpers for the Xiangru-Q3 matched-contract audit.

The functions in this module deliberately separate extraction/reporting from
the numerical solvers.  They do not adapt one model to another and they never
interpolate interval enclosures.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any


MATCH_STATES = frozenset({True, False, "unknown", "not_applicable"})

REQUIRED_CONTRACT_FIELDS = (
    "dynamics",
    "state_order",
    "coordinate_transform",
    "plant_parameters",
    "controller",
    "initial_set",
    "property_set",
    "target_horizon",
    "order_semantics",
    "local_time_interval",
    "step_policy",
    "remainder_policy",
    "cutoff_truncation",
    "picard_validation",
    "transition_lifecycle",
    "interval_backend",
    "output_semantics",
    "included_stages",
    "device_threads",
    "success_predicate",
)

REQUIRED_FIELD_METADATA = (
    "value",
    "matched",
    "source_file",
    "source_line",
    "evidence",
    "reason",
)


def contract_field(
    value: Any,
    matched: bool | str,
    source_file: str,
    source_line: str | int,
    evidence: str,
    reason: str,
) -> dict[str, Any]:
    """Build one fully evidenced contract field."""
    if matched not in MATCH_STATES:
        raise ValueError(f"invalid matched state: {matched!r}")
    return {
        "value": value,
        "matched": matched,
        "source_file": source_file,
        "source_line": str(source_line),
        "evidence": evidence,
        "reason": reason,
    }


def validate_contract(contract: Mapping[str, Any]) -> list[str]:
    """Return schema errors; missing/unknown comparison evidence fails closed."""
    errors: list[str] = []
    fields = contract.get("fields")
    if not isinstance(fields, Mapping):
        return ["fields must be an object"]
    for name in REQUIRED_CONTRACT_FIELDS:
        field = fields.get(name)
        if not isinstance(field, Mapping):
            errors.append(f"missing contract field: {name}")
            continue
        for metadata in REQUIRED_FIELD_METADATA:
            if metadata not in field:
                errors.append(f"{name} missing metadata: {metadata}")
        if field.get("matched") not in MATCH_STATES:
            errors.append(f"{name} has invalid matched state")
        for evidence_name in ("source_file", "source_line", "evidence", "reason"):
            if not str(field.get(evidence_name, "")).strip():
                errors.append(f"{name} has empty {evidence_name}")
    return errors


def formal_match_decision(contracts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Authorize a formal comparison only when every field is explicitly true."""
    blockers: list[dict[str, Any]] = []
    for contract in contracts:
        tool = str(contract.get("tool", "unknown"))
        schema_errors = validate_contract(contract)
        for error in schema_errors:
            blockers.append({"tool": tool, "field": "schema", "reason": error})
        fields = contract.get("fields", {})
        if not isinstance(fields, Mapping):
            continue
        for name in REQUIRED_CONTRACT_FIELDS:
            field = fields.get(name)
            if not isinstance(field, Mapping):
                continue
            if field.get("matched") is not True:
                blockers.append(
                    {
                        "tool": tool,
                        "field": name,
                        "matched": field.get("matched", "unknown"),
                        "reason": str(field.get("reason", "missing reason")),
                    }
                )
    return {
        "formal_comparison_authorized": not blockers,
        "blockers": blockers,
    }


def complete_total_degree_exponents(degree: int, variables: int) -> tuple[tuple[int, ...], ...]:
    """Return the mathematical dense support used by complete-Qq."""
    if degree < 0 or variables <= 0:
        raise ValueError("degree must be nonnegative and variables must be positive")
    exponents: list[tuple[int, ...]] = []

    def visit(position: int, remaining: int, current: list[int]) -> None:
        if position == variables - 1:
            exponents.append(tuple([*current, remaining]))
            return
        for value in range(remaining + 1):
            visit(position + 1, remaining - value, [*current, value])

    for total in range(degree + 1):
        visit(0, total, [])
    return tuple(sorted(exponents, key=lambda exponent: (sum(exponent), exponent)))


def total_degree_retained(exponent: Sequence[int], order: int) -> bool:
    """The common retention predicate independently checked in both sources."""
    if any(value < 0 for value in exponent):
        raise ValueError("exponents must be nonnegative")
    return sum(exponent) <= order


def tagged_enclosure(kind: str, time_lo: float, time_hi: float, bounds: Sequence[Sequence[float]]) -> dict[str, Any]:
    """Create an enclosure while preventing endpoint/tube type collisions."""
    if kind not in {"endpoint", "tube"}:
        raise ValueError("kind must be endpoint or tube")
    if kind == "endpoint" and not math.isclose(time_lo, time_hi, rel_tol=0.0, abs_tol=0.0):
        raise ValueError("endpoint enclosure must have a single physical time")
    if kind == "tube" and not time_lo < time_hi:
        raise ValueError("tube enclosure must span a nonempty interval")
    normalized = [[float(pair[0]), float(pair[1])] for pair in bounds]
    if any(lower > upper for lower, upper in normalized):
        raise ValueError("invalid interval bounds")
    return {"kind": kind, "time_lo": float(time_lo), "time_hi": float(time_hi), "bounds": normalized}


def map_coordinates(bounds: Sequence[Sequence[float]], source_order: Sequence[str], target_order: Sequence[str]) -> list[list[float]]:
    """Reorder coordinates only when the mapping is a named bijection."""
    if len(bounds) != len(source_order) or len(set(source_order)) != len(source_order):
        raise ValueError("source coordinate map is not bijective")
    if set(source_order) != set(target_order) or len(set(target_order)) != len(target_order):
        raise ValueError("source and target coordinates differ")
    by_name = {name: [float(bounds[index][0]), float(bounds[index][1])] for index, name in enumerate(source_order)}
    return [by_name[name] for name in target_order]


def align_exact_endpoint_times(left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]]) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    """Align only bit-identical reported times; ordinary interpolation is forbidden."""
    right_by_time = {float(row["time"]): row for row in right}
    return [(row, right_by_time[float(row["time"])]) for row in left if float(row["time"]) in right_by_time]


def reject_formal_interpolation(*_: Any, **__: Any) -> None:
    raise ValueError("formal interval endpoint interpolation is prohibited")


def width_ratio(numerator_bounds: Sequence[float], denominator_bounds: Sequence[float]) -> float | None:
    """Return a width ratio, using None for a zero-width denominator."""
    numerator = float(numerator_bounds[1]) - float(numerator_bounds[0])
    denominator = float(denominator_bounds[1]) - float(denominator_bounds[0])
    if numerator < 0 or denominator < 0:
        raise ValueError("invalid interval")
    if denominator == 0.0:
        return None
    return numerator / denominator


def horizon_row(tool: str, requested: float, reached: float, status: str) -> dict[str, Any]:
    """Keep failed horizons as explicit rows rather than silently dropping them."""
    completed = status == "completed" and reached >= requested
    return {
        "tool": tool,
        "requested_horizon": float(requested),
        "reached_horizon": float(reached),
        "status": status,
        "completed_requested_horizon": completed,
        "target_horizon_tightness": "available" if completed else "N/A",
    }


def parse_xiangru_runtime(payload: Mapping[str, Any], policy: str, method: str) -> dict[str, Any]:
    """Extract only measured upstream stages; unavailable stages stay unavailable."""
    timing = payload["cells"][policy][method]["timing"]
    controls = payload["controls"]
    return {
        "environment_setup_seconds": "unavailable",
        "compile_or_graph_construction_seconds": float(timing["implementation_compile_and_warm_seconds_excluded"]),
        "model_checkpoint_loading_seconds": "unavailable",
        "warm_up_seconds": "included_with_compile_graph_construction",
        "plant_propagation_seconds": float(timing["default_dynamics_seconds"]) + float(timing["retry_dynamics_seconds"]),
        "nn_bound_and_controller_update_seconds": float(timing["controller_seconds"]),
        "validation_scheduler_seconds": float(timing["validation_seconds_excluded"]),
        "serialization_io_seconds": "unavailable",
        "solver_seconds_excluding_validation": float(timing["solver_wall_seconds_excluding_validation"]),
        "total_end_to_end_seconds_including_validation": float(timing["total_wall_seconds_including_validation"]),
        "device": controls["device"],
        "dtype": controls["dtype"],
        "scope": "native closed-loop implementation runtime; compile/warm is separately excluded",
    }


def deterministic_json_bytes(value: Any) -> bytes:
    """Canonical serialization used by audit hashes and tests."""
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
