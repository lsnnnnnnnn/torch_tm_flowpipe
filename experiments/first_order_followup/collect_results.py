#!/usr/bin/env python3
"""Merge tool outputs and enforce common validation semantics."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from scipy.integrate import solve_ivp

HERE = Path(__file__).resolve().parent
BASELINE_EXPERIMENT = HERE.parent / "first_order_three_way"
if str(BASELINE_EXPERIMENT) not in sys.path:
    sys.path.insert(0, str(BASELINE_EXPERIMENT))

from common import evaluate_rhs, exact_endpoint, load_spec

RAW_FIELDS = [
    "tool", "protocol", "system", "mode", "basis", "h", "horizon",
    "state_index", "step_index", "time", "interval_kind", "lower", "upper",
    "width", "local_construction_basis", "local_construction_order",
    "carried_basis", "carried_max_degree", "projection_method", "reset_method",
    "validator", "numerical_backend", "native_validation_passed",
    "exact_reference_contained", "sampled_trajectory_contained",
    "directed_rounding_or_mpfr", "floating_point_enclosure_candidate",
    "validation_failed", "validation_attempts", "retained_coefficients",
    "discarded_candidates", "python_orchestration_time_s", "compile_time_s",
    "first_call_time_s", "steady_step_time_s", "number_of_steps",
    "number_of_retained_coefficients", "number_of_discarded_candidates",
    "successful_horizon", "message",
]

SUMMARY_FIELDS = [
    "tool", "protocol", "system", "mode", "basis", "h", "horizon",
    "completed_steps", "requested_steps", "successful_horizon",
    "native_validation_passed", "exact_reference_checks",
    "exact_reference_violations", "sample_checks", "sample_violations",
    "validation_failed", "final_endpoint_width_max",
    "local_construction_basis", "local_construction_order", "carried_basis",
    "carried_max_degree", "projection_method", "reset_method", "validator",
    "numerical_backend", "python_orchestration_time_s", "compile_time_s",
    "first_call_time_s", "steady_step_time_s",
    "number_of_retained_coefficients", "number_of_discarded_candidates",
    "message",
]


def _truth(value: Any) -> bool | None:
    if value in (True, "True", "true", "1", 1):
        return True
    if value in (False, "False", "false", "0", 0):
        return False
    return None


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(
    path: Path, rows: Iterable[Mapping[str, Any]], fields: list[str]
) -> None:
    rows = list(rows)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def _group_key(row: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(
        str(row.get(field, ""))
        for field in ("tool", "protocol", "system", "mode", "basis", "h", "horizon")
    )


def _vdp_solution(
    initial: tuple[float, float],
    horizon: float,
):
    def rhs(_time: float, state: np.ndarray) -> list[float]:
        x, y = state
        return [y, y - x - x * x * y]

    return solve_ivp(
        rhs,
        (0.0, horizon),
        initial,
        method="DOP853",
        rtol=1e-12,
        atol=1e-14,
        dense_output=True,
        max_step=0.001,
    )


def apply_vdp_samples(
    rows: list[dict[str, Any]],
    spec: Mapping[str, Any],
    *,
    tolerance: float,
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("system") == "van_der_pol":
            grouped[_group_key(row)].append(row)
    initial_box = spec["systems"]["van_der_pol"]["initial_box"]
    grids = [
        np.linspace(float(lower), float(upper), 3)
        for lower, upper in initial_box
    ]
    initials = [
        (float(x), float(y))
        for x in grids[0]
        for y in grids[1]
    ]
    for key, group in grouped.items():
        by_interval = {
            (
                row["interval_kind"],
                int(row["step_index"]),
                int(row["state_index"]),
            ): row
            for row in group
        }
        endpoint_steps = sorted(
            {
                int(row["step_index"])
                for row in group
                if row["interval_kind"] == "endpoint"
            }
        )
        if not endpoint_steps:
            continue
        h = float(group[0]["h"])
        max_time = max(float(row["time"]) for row in group)
        solutions = [_vdp_solution(initial, max_time) for initial in initials]
        violations = []
        checked = 0
        for step in endpoint_steps:
            time_value = step * h
            if step == 0:
                sample_times = [0.0]
            else:
                sample_times = [time_value]
            for solution, initial in zip(solutions, initials):
                for sample_time in sample_times:
                    state = solution.sol(sample_time)
                    for state_index, value in enumerate(state):
                        row = by_interval.get(("endpoint", step, state_index))
                        if row is None:
                            continue
                        checked += 1
                        if (
                            value < float(row["lower"]) - tolerance
                            or value > float(row["upper"]) + tolerance
                        ):
                            violations.append(
                                {
                                    "step": step,
                                    "time": sample_time,
                                    "initial": initial,
                                    "state": state_index,
                                    "value": float(value),
                                    "interval": [
                                        float(row["lower"]),
                                        float(row["upper"]),
                                    ],
                                }
                            )
            if step == 0:
                continue
            tube_times = np.linspace((step - 1) * h, step * h, 4)
            for solution, initial in zip(solutions, initials):
                for sample_time in tube_times:
                    state = solution.sol(sample_time)
                    for state_index, value in enumerate(state):
                        row = by_interval.get(("tube", step, state_index))
                        if row is None:
                            continue
                        checked += 1
                        if (
                            value < float(row["lower"]) - tolerance
                            or value > float(row["upper"]) + tolerance
                        ):
                            violations.append(
                                {
                                    "step": step,
                                    "time": float(sample_time),
                                    "initial": initial,
                                    "state": state_index,
                                    "value": float(value),
                                    "interval": [
                                        float(row["lower"]),
                                        float(row["upper"]),
                                    ],
                                }
                            )
        passed = not violations
        for row in group:
            row["sampled_trajectory_contained"] = passed
            if not passed:
                row["validation_failed"] = True
                row["message"] = (
                    str(row.get("message") or "")
                    + f"; {len(violations)}/{checked} deterministic samples violated"
                ).strip("; ")
        checks.append(
            {
                "group": list(key),
                "check": "deterministic_high_accuracy_trajectory_containment",
                "checked": checked,
                "violations": len(violations),
                "proof": False,
                "details": violations[:50],
            }
        )
    return checks


def enforce_exact_checks(
    rows: list[dict[str, Any]],
    spec: Mapping[str, Any],
    *,
    tolerance: float,
) -> list[dict[str, Any]]:
    checks = []
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("interval_kind") == "endpoint":
            grouped[_group_key(row)].append(row)
    for key, group in grouped.items():
        system_name = group[0]["system"]
        system = spec["systems"][system_name]
        violations = []
        checked = 0
        for row in group:
            expected = exact_endpoint(
                system_name, float(row["time"]), system["initial_box"]
            )
            if expected is None:
                continue
            state = int(row["state_index"])
            exact_lower, exact_upper = expected[state]
            checked += 1
            contained = (
                float(row["lower"]) <= exact_lower + tolerance
                and float(row["upper"]) >= exact_upper - tolerance
            )
            row["exact_reference_contained"] = contained
            if not contained:
                row["validation_failed"] = True
                violations.append(
                    {
                        "step": int(row["step_index"]),
                        "time": float(row["time"]),
                        "state": state,
                        "expected": [exact_lower, exact_upper],
                        "reported": [float(row["lower"]), float(row["upper"])],
                    }
                )
        if checked:
            checks.append(
                {
                    "group": list(key),
                    "check": "analytic_exact_endpoint_containment",
                    "checked": checked,
                    "violations": len(violations),
                    "details": violations[:50],
                }
            )
    return checks


def summarize(
    rows: list[dict[str, Any]],
    exact_checks: list[dict[str, Any]],
    sample_checks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_group_key(row)].append(row)
    exact_by_key = {tuple(item["group"]): item for item in exact_checks}
    sample_by_key = {tuple(item["group"]): item for item in sample_checks}
    summaries = []
    for key, group in grouped.items():
        endpoints = [row for row in group if row["interval_kind"] == "endpoint"]
        completed_steps = max((int(row["step_index"]) for row in endpoints), default=0)
        requested_steps = int(float(group[0].get("number_of_steps") or 0))
        horizon = float(group[0]["horizon"])
        successful_horizon = max(
            (
                float(row["time"])
                for row in endpoints
                if _truth(row.get("validation_failed")) is not True
            ),
            default=0.0,
        )
        final_rows = [
            row for row in endpoints if int(row["step_index"]) == completed_steps
        ]
        exact = exact_by_key.get(key, {})
        samples = sample_by_key.get(key, {})
        validation_failed = (
            completed_steps < requested_steps
            or int(exact.get("violations", 0)) > 0
            or int(samples.get("violations", 0)) > 0
        )
        first = group[0]
        summaries.append(
            {
                **{field: first.get(field, "") for field in SUMMARY_FIELDS},
                "completed_steps": completed_steps,
                "requested_steps": requested_steps,
                "successful_horizon": successful_horizon,
                "native_validation_passed": completed_steps == requested_steps,
                "exact_reference_checks": exact.get("checked", 0),
                "exact_reference_violations": exact.get("violations", 0),
                "sample_checks": samples.get("checked", 0),
                "sample_violations": samples.get("violations", 0),
                "validation_failed": validation_failed,
                "final_endpoint_width_max": max(
                    (float(row["width"]) for row in final_rows), default=math.nan
                ),
                "message": first.get("message", ""),
            }
        )
    return sorted(
        summaries,
        key=lambda row: (
            row["system"], row["tool"], row["protocol"], row["mode"], row["basis"]
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    spec = load_spec(HERE / "benchmark_spec.yaml")
    source_paths = [
        output / "torch_raw_results.csv",
        output / "diffreach_raw_results.csv",
        output / "flowstar_raw_results.csv",
    ]
    rows: list[dict[str, Any]] = []
    for path in source_paths:
        if path.is_file():
            rows.extend(_read_csv(path))
    if not rows:
        raise SystemExit("no tool raw results found")
    for row in rows:
        for field in RAW_FIELDS:
            row.setdefault(field, "")
    tolerance = float(spec["numerical_containment_tolerance"])
    exact_checks = enforce_exact_checks(rows, spec, tolerance=tolerance)
    sample_checks = apply_vdp_samples(rows, spec, tolerance=tolerance)
    flowstar_audit_path = output / "logs" / "flowstar_audit" / "flowstar_correctness.json"
    flowstar_audit = (
        json.loads(flowstar_audit_path.read_text(encoding="utf-8"))
        if flowstar_audit_path.is_file()
        else {}
    )
    summaries = summarize(rows, exact_checks, sample_checks)
    _write_csv(output / "raw_results.csv", rows, RAW_FIELDS)
    _write_csv(output / "run_summary.csv", summaries, SUMMARY_FIELDS)
    correctness = {
        "all_exact_checks_passed": all(
            int(item["violations"]) == 0 for item in exact_checks
        ),
        "all_sample_checks_passed": all(
            int(item["violations"]) == 0 for item in sample_checks
        ),
        "sample_checks_are_formal_proof": False,
        "exact_reference_checks": sum(int(item["checked"]) for item in exact_checks),
        "exact_reference_violations": sum(
            int(item["violations"]) for item in exact_checks
        ),
        "sample_checks": sum(int(item["checked"]) for item in sample_checks),
        "sample_violations": sum(
            int(item["violations"]) for item in sample_checks
        ),
        "checks": exact_checks + sample_checks,
        "flowstar_extraction_audit": flowstar_audit,
        "no_successful_row_with_exact_violation": all(
            not (
                _truth(row.get("exact_reference_contained")) is False
                and _truth(row.get("validation_failed")) is not True
            )
            for row in rows
        ),
    }
    (output / "correctness_checks.json").write_text(
        json.dumps(correctness, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"Collected {len(rows)} rows; exact violations="
        f"{correctness['exact_reference_violations']}; sample violations="
        f"{correctness['sample_violations']}",
        flush=True,
    )
    if not correctness["all_exact_checks_passed"]:
        raise SystemExit("analytic exact-reference containment gate failed")
    if not correctness["all_sample_checks_passed"]:
        raise SystemExit("deterministic trajectory bug-catching gate failed")
    if not correctness["no_successful_row_with_exact_violation"]:
        raise SystemExit("a successful row violates an analytic exact reference")


if __name__ == "__main__":
    main()
