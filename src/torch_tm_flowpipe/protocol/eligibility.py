from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .schema import (
    Applicability,
    BoundSemantics,
    FailureCategory,
    RUNTIME_BOUNDARY_VERSION,
)


@dataclass(frozen=True)
class EligibilityDecision:
    eligible: bool
    reasons: tuple[str, ...]


def strict_required_true(value: Any) -> bool:
    """Accept only an actual boolean True, never truthy status text or integers."""
    return type(value) is bool and value is True


def _finite_positive(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number > 0.0


def evaluate_primary_eligibility(
    row: Mapping[str, Any],
    *,
    required_repetitions: int = 10,
    required_validation_fields: Sequence[str] = (
        "native_validation_passed",
        "analytic_containment_passed",
        "trajectory_sanity_passed",
    ),
) -> EligibilityDecision:
    reasons: list[str] = []
    for field in (
        "completed_requested_horizon",
        "all_required_repetitions_present",
        "primary_comparable",
    ):
        if not strict_required_true(row.get(field)):
            reasons.append(f"{field}_not_explicit_true")
    for field in required_validation_fields:
        applicability = row.get(
            f"{field.removesuffix('_passed')}_applicability",
            Applicability.REQUIRED.value,
        )
        if applicability == Applicability.NOT_APPLICABLE.value:
            continue
        if applicability != Applicability.REQUIRED.value:
            reasons.append(f"{field}_applicability_invalid")
        elif not strict_required_true(row.get(field)):
            reasons.append(f"{field}_not_explicit_true")

    try:
        repetitions = int(row.get("runtime_repetitions", 0))
    except (TypeError, ValueError):
        repetitions = 0
    if repetitions < required_repetitions:
        reasons.append("insufficient_runtime_repetitions")

    if row.get("bound_semantics") != BoundSemantics.RAW_ENDPOINT.value:
        reasons.append("bound_semantics_not_raw_endpoint")
    if row.get("runtime_boundary_version") != RUNTIME_BOUNDARY_VERSION:
        reasons.append("runtime_boundary_version_mismatch")
    if not _finite_positive(row.get("steady_total_configuration_time_s")):
        reasons.append("steady_total_configuration_time_not_positive_finite")
    if row.get("failure_category") != FailureCategory.COMPLETED.value:
        reasons.append("configuration_not_completed")

    try:
        requested = float(row.get("requested_horizon"))
        successful = float(row.get("successful_horizon"))
    except (TypeError, ValueError):
        reasons.append("horizon_not_numeric")
    else:
        if not (
            math.isfinite(requested)
            and math.isfinite(successful)
            and math.isclose(
                requested,
                successful,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ):
            reasons.append("requested_horizon_not_completed")

    return EligibilityDecision(not reasons, tuple(reasons))


def partition_and_recompute_pareto(
    rows: Sequence[Mapping[str, Any]],
    *,
    required_repetitions: int = 10,
    required_validation_fields: Sequence[str] = (
        "native_validation_passed",
        "analytic_containment_passed",
        "trajectory_sanity_passed",
    ),
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from .pareto import recompute_pareto

    eligible: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        decision = evaluate_primary_eligibility(
            row,
            required_repetitions=required_repetitions,
            required_validation_fields=required_validation_fields,
        )
        row["primary_numerical_eligible"] = decision.eligible
        row["exclusion_reason"] = ";".join(decision.reasons)
        if decision.eligible:
            row["excluded_from_authoritative"] = False
            eligible.append(row)
        else:
            row["excluded_from_authoritative"] = True
            row["width_runtime_pareto"] = False
            excluded.append(row)
    recompute_pareto(eligible)
    return eligible, excluded
