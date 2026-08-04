#!/usr/bin/env python3
"""Analyze current-run Flowstar/Torch one-step exports without field fallback."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


CASES = {
    "scalar_affine_o4_h001": (
        "flowstar_scalar_affine_o4.json",
        "torch_scalar_affine_normalized_flowstar_compat_o4.json",
    ),
    "scalar_quadratic_o4_h001": (
        "flowstar_scalar_quadratic_o4.json",
        "torch_scalar_quadratic_normalized_flowstar_compat_o4.json",
    ),
    "harmonic_oscillator_o4_h001": (
        "flowstar_harmonic_oscillator_o4.json",
        "torch_harmonic_oscillator_normalized_flowstar_compat_o4.json",
    ),
    "van_der_pol_o4_h0005": (
        "flowstar_van_der_pol_official_o4.json",
        "torch_van_der_pol_normalized_flowstar_compat_o4.json",
    ),
    "van_der_pol_o2_h0001_sensitivity": (
        "flowstar_van_der_pol_h0001_o2.json",
        "torch_van_der_pol_h0001_normalized_flowstar_compat_o2.json",
    ),
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_exponent(exponents: Sequence[int], roles: Sequence[str]) -> tuple[int, ...]:
    if len(exponents) != len(roles):
        raise ValueError("exponent/role dimension mismatch")
    state = [index for index, role in enumerate(roles) if role == "state_generator"]
    local = [index for index, role in enumerate(roles) if role == "local_time"]
    other = [
        index
        for index, role in enumerate(roles)
        if role not in {"state_generator", "local_time"}
    ]
    if len(local) != 1:
        raise ValueError("one-step trace must have exactly one local-time variable")
    return tuple(int(exponents[index]) for index in [*state, *other, *local])


def canonical_terms(record: Mapping[str, Any]) -> list[dict[tuple[int, ...], float]]:
    roles = record["variable_roles"]
    states = record["enclosures"]["last_segment"]["states"]
    return [
        {
            canonical_exponent(term["exponents"], roles): float(term["coefficient"])
            for term in state["polynomial_terms"]
        }
        for state in states
    ]


def support_sha256(terms: Sequence[Mapping[tuple[int, ...], float]]) -> str:
    payload = [sorted([list(exponent) for exponent in state]) for state in terms]
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def box_contains(outer: Sequence[Sequence[float]], inner: Sequence[Sequence[float]]) -> bool:
    return all(
        float(out[0]) <= float(inside[0]) and float(out[1]) >= float(inside[1])
        for out, inside in zip(outer, inner)
    )


def max_box_difference(
    left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]
) -> float:
    return max(
        abs(float(a) - float(b))
        for left_state, right_state in zip(left, right)
        for a, b in zip(left_state, right_state)
    )


def _rhs(system_definition: Mapping[str, Any], state: Sequence[float]) -> list[float]:
    values: list[float] = []
    for polynomial in system_definition["equations"]:
        value = 0.0
        for term in polynomial["terms"]:
            product = float(term["coefficient"])
            for coordinate, exponent in zip(state, term["powers"]):
                product *= float(coordinate) ** int(exponent)
            value += product
        values.append(value)
    return values


def rk4_trajectory(
    system_definition: Mapping[str, Any], initial: Sequence[float], h: float, *, steps: int = 1000
) -> list[list[float]]:
    state = [float(value) for value in initial]
    trajectory = [state.copy()]
    dt = float(h) / steps
    for _ in range(steps):
        k1 = _rhs(system_definition, state)
        k2 = _rhs(system_definition, [x + 0.5 * dt * k for x, k in zip(state, k1)])
        k3 = _rhs(system_definition, [x + 0.5 * dt * k for x, k in zip(state, k2)])
        k4 = _rhs(system_definition, [x + dt * k for x, k in zip(state, k3)])
        state = [
            x + dt * (a + 2.0 * b + 2.0 * c + d) / 6.0
            for x, a, b, c, d in zip(state, k1, k2, k3, k4)
        ]
        trajectory.append(state.copy())
    return trajectory


def trajectory_sanity(record: Mapping[str, Any]) -> dict[str, Any]:
    initial_box = record["system_definition"]["initial_domain"]
    corners = list(itertools.product(*[(float(lo), float(hi)) for lo, hi in initial_box]))
    endpoint = record["enclosures"]["endpoint_raw"]["box"]
    segment = record["enclosures"]["last_segment"]["box"]
    endpoint_violations: list[dict[str, Any]] = []
    segment_violations: list[dict[str, Any]] = []
    for corner in corners:
        trajectory = rk4_trajectory(record["system_definition"], corner, float(record["h"]))
        for state_index, value in enumerate(trajectory[-1]):
            if not (
                float(endpoint[state_index][0]) - 1.0e-12
                <= value
                <= float(endpoint[state_index][1]) + 1.0e-12
            ):
                endpoint_violations.append(
                    {"initial": list(corner), "state": state_index, "value": value}
                )
        for sample_index in range(0, len(trajectory), 100):
            for state_index, value in enumerate(trajectory[sample_index]):
                if not (
                    float(segment[state_index][0]) - 1.0e-12
                    <= value
                    <= float(segment[state_index][1]) + 1.0e-12
                ):
                    segment_violations.append(
                        {
                            "initial": list(corner),
                            "sample_index": sample_index,
                            "state": state_index,
                            "value": value,
                        }
                    )
    return {
        "method": "deterministic_RK4_1000_substeps_at_initial_box_corners",
        "formal_proof": False,
        "corner_count": len(corners),
        "endpoint_violations": endpoint_violations,
        "segment_violations": segment_violations,
        "passed": not endpoint_violations and not segment_violations,
    }


def _availability(value: Any) -> str:
    if isinstance(value, Mapping) and value.get("availability") == "unavailable":
        return "unavailable"
    return "available"


def analyze_pair(flowstar: Mapping[str, Any], torch: Mapping[str, Any]) -> dict[str, Any]:
    flow_terms = canonical_terms(flowstar)
    torch_terms = canonical_terms(torch)
    support_equal_by_state = [set(left) == set(right) for left, right in zip(flow_terms, torch_terms)]
    coefficient_differences: list[dict[str, Any]] = []
    for state, (left, right) in enumerate(zip(flow_terms, torch_terms)):
        for exponent in sorted(set(left) & set(right)):
            coefficient_differences.append(
                {
                    "state": state,
                    "canonical_exponent": list(exponent),
                    "flowstar": left[exponent],
                    "torch": right[exponent],
                    "absolute_difference": abs(left[exponent] - right[exponent]),
                }
            )
    flow_raw = flowstar["enclosures"]["endpoint_raw"]["box"]
    torch_raw = torch["enclosures"]["endpoint_raw"]["box"]
    flow_segment = flowstar["enclosures"]["last_segment"]["box"]
    torch_segment = torch["enclosures"]["last_segment"]["box"]
    flow_remainders = [
        state["independent_interval_remainder"]
        for state in flowstar["enclosures"]["last_segment"]["states"]
    ]
    torch_remainders = [
        state["independent_interval_remainder"]
        for state in torch["enclosures"]["last_segment"]["states"]
    ]
    collapsed = flowstar["enclosures"]["endpoint_collapsed"]["box"]
    collapsed_available = _availability(collapsed) == "available"
    flow_picard_available = bool(flowstar.get("validation_trace"))
    torch_picard_available = bool(torch.get("validation_trace"))
    return {
        "system_equal": flowstar["system"] == torch["system"],
        "h_equal": float(flowstar["h"]) == float(torch["h"]),
        "state_count_equal": len(flow_terms) == len(torch_terms),
        "source_coordinate_contract": {
            "flowstar_roles": flowstar["variable_roles"],
            "torch_roles": torch["variable_roles"],
            "canonical_order": ["state_generators_in_state_order", "local_time"],
            "flowstar_domains_canonical": [
                flowstar["domains"][index]
                for index in [
                    *[
                        i
                        for i, role in enumerate(flowstar["variable_roles"])
                        if role == "state_generator"
                    ],
                    *[
                        i
                        for i, role in enumerate(flowstar["variable_roles"])
                        if role == "local_time"
                    ],
                ]
            ],
            "torch_domains_canonical": [
                torch["domains"][index]
                for index in [
                    *[
                        i
                        for i, role in enumerate(torch["variable_roles"])
                        if role == "state_generator"
                    ],
                    *[
                        i
                        for i, role in enumerate(torch["variable_roles"])
                        if role == "local_time"
                    ],
                ]
            ],
            "domains_equal_after_permutation": sorted(flowstar["domains"]) == sorted(torch["domains"]),
            "torch_source_coordinates": torch["native_metadata"].get("source_coordinates"),
        },
        "acceptance": {
            "flowstar": flowstar["outcome"],
            "torch": torch["outcome"],
            "match": flowstar["outcome"]["status"] == torch["outcome"]["status"],
        },
        "effective_support": {
            "flowstar_sha256": support_sha256(flow_terms),
            "torch_sha256": support_sha256(torch_terms),
            "equal_by_state_after_variable_permutation": support_equal_by_state,
            "all_equal": all(support_equal_by_state),
        },
        "coefficients": {
            "comparison_is_diagnostic_without_preregistered_tolerance": True,
            "max_absolute_difference": max(
                (row["absolute_difference"] for row in coefficient_differences),
                default=0.0,
            ),
            "rows": coefficient_differences,
        },
        "independent_remainders": {
            "flowstar": flow_remainders,
            "torch": torch_remainders,
            "max_absolute_endpoint_difference": max_box_difference(
                flow_remainders, torch_remainders
            ),
            "first_observable_output_difference": "independent_interval_remainder",
        },
        "picard_trace": {
            "flowstar_available": flow_picard_available,
            "torch_available": torch_picard_available,
            "field_parity_testable": flow_picard_available and torch_picard_available,
        },
        "enclosures": {
            "flowstar_raw_endpoint_inside_flowstar_last_segment": box_contains(flow_segment, flow_raw),
            "torch_raw_endpoint_inside_torch_last_segment": box_contains(torch_segment, torch_raw),
            "raw_endpoint_max_absolute_difference": max_box_difference(flow_raw, torch_raw),
            "flowstar_raw_contains_torch_raw": box_contains(flow_raw, torch_raw),
            "torch_raw_contains_flowstar_raw": box_contains(torch_raw, flow_raw),
            "flowstar_collapsed_available": collapsed_available,
            "flowstar_collapsed_contains_flowstar_raw": (
                box_contains(collapsed, flow_raw) if collapsed_available else None
            ),
            "flowstar_raw_collapsed_separated": collapsed is not flow_raw,
        },
        "trajectory_sanity": {
            "flowstar": trajectory_sanity(flowstar),
            "torch": trajectory_sanity(torch),
            "interpretation": "zero violations is a sanity check, not a formal enclosure proof",
        },
        "first_contract_blocker": (
            None
            if flow_picard_available and torch_picard_available
            else "flowstar.validation_trace.picard_iteration[0]_unavailable"
        ),
    }


def run(run_dir: Path, output: Path) -> dict[str, Any]:
    trace_dir = run_dir / "one_step_trace"
    cases: dict[str, Any] = {}
    for name, (flow_name, torch_name) in CASES.items():
        flow_path = trace_dir / flow_name
        torch_path = trace_dir / torch_name
        flowstar = json.loads(flow_path.read_text(encoding="utf-8"))
        torch = json.loads(torch_path.read_text(encoding="utf-8"))
        cases[name] = {
            "inputs": {
                "flowstar": str(flow_path.relative_to(run_dir)),
                "flowstar_sha256": sha256_file(flow_path),
                "torch": str(torch_path.relative_to(run_dir)),
                "torch_sha256": sha256_file(torch_path),
            },
            **analyze_pair(flowstar, torch),
        }
    result = {
        "schema_version": "one-step-parity-evidence-1.0.0",
        "run_id": run_dir.name,
        "cases": cases,
        "gate_passed": False,
        "blocker": (
            "stock Flowstar exporter does not expose source/Picard iteration, discarded-term, "
            "and candidate self-map fields; output-only agreement cannot establish field parity"
        ),
        "headline_use_allowed": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.run_dir.resolve(), args.output.resolve())
    print(json.dumps({"gate_passed": result["gate_passed"], "cases": len(result["cases"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
