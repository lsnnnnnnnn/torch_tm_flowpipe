from __future__ import annotations

import math
from enum import Enum
from typing import Any, Mapping


SCHEMA_VERSION = "2.1.0"
RUNTIME_BOUNDARY_VERSION = "total_configuration_v2"


class BoundSemantics(str, Enum):
    RAW_ENDPOINT = "raw_endpoint"
    TIGHTENED_ENDPOINT = "tightened_endpoint"
    SEGMENT_BOX = "segment_box"
    TUBE_BOX = "tube_box"


class BoundKind(str, Enum):
    ENDPOINT = "endpoint"
    ACCEPTED_SEGMENT = "accepted_segment"
    FULL_TUBE = "full_tube"


class RefinementSemantics(str, Enum):
    RAW = "raw"
    TIGHTENED = "tightened"


class FailureCategory(str, Enum):
    COMPLETED = "completed"
    VALIDATION_REJECTED = "validation_rejected"
    NONFINITE = "nonfinite"
    TIMEOUT = "timeout"
    PROCESS_ERROR = "process_error"
    COMPILE_ERROR = "compile_error"
    MISSING_DEPENDENCY = "missing_dependency"
    TRAJECTORY_SANITY_FAILED = "trajectory_sanity_failed"
    ANALYTIC_CONTAINMENT_FAILED = "analytic_containment_failed"
    SCHEMA_INVALID = "schema_invalid"
    INCOMPLETE_UNKNOWN = "incomplete_unknown"


class Applicability(str, Enum):
    REQUIRED = "required"
    NOT_APPLICABLE = "not_applicable"


RUNTIME_FIELDS = (
    "cold_total_configuration_time_s",
    "steady_total_configuration_time_s",
    "engine_internal_time_s",
    "compile_or_jit_time_s",
    "posthoc_validation_time_s",
    "plot_report_time_s",
)

IDENTITY_FIELDS = (
    "tool",
    "variant",
    "system",
    "h",
    "requested_horizon",
    "requested_order",
    "effective_order",
    "effective_degree",
    "basis_id",
    "remainder_policy",
    "step_policy",
    "bound_semantics",
    "bound_kind",
    "refinement_semantics",
    "endpoint_exporter_semantics",
    "runtime_boundary_version",
)

REQUIRED_OBSERVATION_FIELDS = (
    *IDENTITY_FIELDS,
    "successful_horizon",
    "completed_requested_horizon",
    "last_valid_step",
    "failure_step",
    "failure_category",
    "failure_message",
    "backend_class",
    "backend_sha",
    "backend_dirty",
    "backend_primary_eligible",
    "execution_route",
    "primary_comparable",
    "runtime_repetitions",
    "all_required_repetitions_present",
    *RUNTIME_FIELDS,
)


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def normalize_observation(row: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize without inferring requested horizon or validation outcomes."""
    normalized = dict(row)
    normalized.setdefault("schema_version", SCHEMA_VERSION)
    normalized.setdefault("failure_message", "")
    normalized.setdefault("last_valid_step", "")
    normalized.setdefault("failure_step", "")
    normalized.setdefault("requested_order", "")
    normalized.setdefault("effective_order", "")
    normalized.setdefault("effective_degree", "")
    normalized.setdefault("basis_id", "")
    normalized.setdefault("remainder_policy", "")
    normalized.setdefault("step_policy", "")
    errors = validate_observation(normalized)
    if errors:
        raise ValueError("; ".join(errors))
    return normalized


def validate_observation(row: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    missing = [
        field
        for field in REQUIRED_OBSERVATION_FIELDS
        if field not in row
    ]
    if missing:
        errors.append(f"missing required fields: {', '.join(missing)}")
        return errors

    requested = _finite_number(row.get("requested_horizon"))
    successful = _finite_number(row.get("successful_horizon"))
    if requested is None or requested < 0:
        errors.append("requested_horizon must be finite and nonnegative")
    if successful is None or successful < 0:
        errors.append("successful_horizon must be finite and nonnegative")
    if (
        requested is not None
        and successful is not None
        and successful > requested + 1e-12 * max(1.0, abs(requested))
    ):
        errors.append("successful_horizon exceeds requested_horizon")

    try:
        FailureCategory(str(row.get("failure_category")))
    except ValueError:
        errors.append("failure_category is not in the canonical taxonomy")
    try:
        bound_semantics = BoundSemantics(str(row.get("bound_semantics")))
    except ValueError:
        errors.append("bound_semantics is invalid")
        bound_semantics = None
    try:
        bound_kind = BoundKind(str(row.get("bound_kind")))
    except ValueError:
        errors.append("bound_kind is invalid")
        bound_kind = None
    try:
        refinement = RefinementSemantics(
            str(row.get("refinement_semantics"))
        )
    except ValueError:
        errors.append("refinement_semantics is invalid")
        refinement = None
    expected_dimensions = {
        BoundSemantics.RAW_ENDPOINT: (
            BoundKind.ENDPOINT,
            RefinementSemantics.RAW,
        ),
        BoundSemantics.TIGHTENED_ENDPOINT: (
            BoundKind.ENDPOINT,
            RefinementSemantics.TIGHTENED,
        ),
        BoundSemantics.SEGMENT_BOX: (
            BoundKind.ACCEPTED_SEGMENT,
            RefinementSemantics.RAW,
        ),
        BoundSemantics.TUBE_BOX: (
            BoundKind.FULL_TUBE,
            RefinementSemantics.RAW,
        ),
    }
    if (
        bound_semantics is not None
        and bound_kind is not None
        and refinement is not None
        and (bound_kind, refinement) != expected_dimensions[bound_semantics]
    ):
        errors.append(
            "bound_semantics collides with bound_kind/refinement_semantics"
        )

    for field in ("backend_dirty", "backend_primary_eligible"):
        if type(row.get(field)) is not bool:
            errors.append(f"{field} must be an explicit boolean")
    if not str(row.get("backend_class", "")).strip():
        errors.append("backend_class must be explicit")
    if not str(row.get("backend_sha", "")).strip():
        errors.append("backend_sha must be explicit")
    if not str(row.get("execution_route", "")).strip():
        errors.append("execution_route must be explicit")
    if not str(row.get("endpoint_exporter_semantics", "")).strip():
        errors.append("endpoint_exporter_semantics must be explicit")

    if row.get("runtime_boundary_version") != RUNTIME_BOUNDARY_VERSION:
        errors.append(
            f"runtime_boundary_version must be {RUNTIME_BOUNDARY_VERSION}"
        )
    for field in RUNTIME_FIELDS:
        number = _finite_number(row.get(field))
        if number is None or number < 0:
            errors.append(f"{field} must be finite and nonnegative")

    if (
        row.get("completed_requested_horizon") is True
        and requested is not None
        and successful is not None
        and not math.isclose(requested, successful, rel_tol=1e-12, abs_tol=1e-12)
    ):
        errors.append(
            "completed_requested_horizon=True but successful horizon differs"
        )
    return errors
