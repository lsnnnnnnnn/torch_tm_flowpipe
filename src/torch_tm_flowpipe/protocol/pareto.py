from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Mapping, MutableMapping, Sequence


def _positive_finite(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("boolean is not a Pareto objective")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError("Pareto objectives must be positive and finite")
    return number


def recompute_pareto(
    rows: Sequence[MutableMapping[str, Any]],
) -> None:
    """Recompute dominance only on the already-eligible input rows."""
    groups: dict[tuple[str, float], list[MutableMapping[str, Any]]] = (
        defaultdict(list)
    )
    for row in rows:
        if row.get("primary_numerical_eligible") is not True:
            raise ValueError("Pareto input contains an ineligible row")
        evaluation_time = float(row["evaluation_time"])
        if not math.isfinite(evaluation_time):
            raise ValueError("evaluation_time must be finite")
        groups[
            (
                str(row["system"]),
                round(evaluation_time, 12),
            )
        ].append(row)

    for group in groups.values():
        objectives = {
            id(row): (
                _positive_finite(row["width_at_evaluation_time"]),
                _positive_finite(row["steady_total_configuration_time_s"]),
            )
            for row in group
        }
        for candidate in group:
            candidate_width, candidate_runtime = objectives[id(candidate)]
            candidate["width_runtime_pareto"] = not any(
                other is not candidate
                and other_width <= candidate_width
                and other_runtime <= candidate_runtime
                and (
                    other_width < candidate_width
                    or other_runtime < candidate_runtime
                )
                for other in group
                for other_width, other_runtime in [objectives[id(other)]]
            )


def independent_pareto_keys(
    rows: Sequence[Mapping[str, Any]],
) -> set[tuple[str, str, str, str]]:
    """Small pure helper for consumers that need stable frontier identities."""
    copies = [dict(row) for row in rows]
    recompute_pareto(copies)
    return {
        (
            str(row["tool"]),
            str(row["variant"]),
            str(row["system"]),
            str(row["h"]),
        )
        for row in copies
        if row["width_runtime_pareto"] is True
    }
