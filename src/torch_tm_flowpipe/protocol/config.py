from __future__ import annotations

from typing import Any, Mapping

from .provenance import canonical_config_identity
from .schema import (
    BoundKind,
    BoundSemantics,
    RefinementSemantics,
    RUNTIME_BOUNDARY_VERSION,
)


def configuration_semantics(tool: str, variant: str) -> dict[str, Any]:
    order: int | str = ""
    for token in variant.replace("-", "_").split("_"):
        if token.startswith("order") and token[5:].isdigit():
            order = int(token[5:])
            break
    if tool == "diffreach":
        return {
            "requested_order": "not_applicable",
            "effective_order": "restricted_quasiquadratic_round5",
            "effective_degree": "restricted_quasiquadratic_round5",
            "basis_id": "restricted_quasiquadratic_symbolic_window100",
            "remainder_policy": "frr_round5_stop_ratio_0.95",
            "step_policy": "fixed",
        }
    if tool == "flowstar":
        order = order or 4
        return {
            "requested_order": order,
            "effective_order": order,
            "effective_degree": order,
            "basis_id": f"complete_total_degree_{order}",
            "remainder_policy": "validated_interval_candidate",
            "step_policy": "fixed",
        }
    order = order or 2
    return {
        "requested_order": order,
        "effective_order": order,
        "effective_degree": order,
        "basis_id": f"complete_total_degree_{order}",
        "remainder_policy": "validated_interval",
        "step_policy": "fixed",
    }


def expected_configuration_rows(
    benchmark: Mapping[str, Any],
    profile: Mapping[str, Any],
) -> list[dict[str, Any]]:
    source = str(profile["configuration_source"])
    if source not in {"smoke", "multi_step"}:
        raise ValueError(f"unsupported configuration source: {source}")
    rows: list[dict[str, Any]] = []
    for tool, tool_profile in profile["tools"].items():
        for variant in tool_profile["variants"]:
            for system, configurations in benchmark[source].items():
                selected = (
                    [configurations]
                    if source == "smoke"
                    else configurations
                )
                for configuration in selected:
                    row = {
                        "tool": str(tool),
                        "variant": str(variant),
                        "system": str(system),
                        "h": float(configuration["h"]),
                        "requested_horizon": float(
                            configuration["horizon"]
                        ),
                        "bound_semantics": (
                            BoundSemantics.RAW_ENDPOINT.value
                        ),
                        "bound_kind": BoundKind.ENDPOINT.value,
                        "refinement_semantics": (
                            RefinementSemantics.RAW.value
                        ),
                        "endpoint_exporter_semantics": (
                            "raw_endpoint_at_requested_horizon"
                        ),
                        "runtime_boundary_version": (
                            RUNTIME_BOUNDARY_VERSION
                        ),
                        **configuration_semantics(
                            str(tool), str(variant)
                        ),
                    }
                    row["config_id"] = canonical_config_identity(row)
                    rows.append(row)
    return sorted(
        rows,
        key=lambda row: (
            row["tool"],
            row["variant"],
            row["system"],
            row["h"],
            row["requested_horizon"],
        ),
    )
