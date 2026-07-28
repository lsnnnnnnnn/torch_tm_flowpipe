#!/usr/bin/env python3
"""Collect repaired outputs and enforce semantic and correctness gates."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from scipy.integrate import solve_ivp

from common import (
    FAILURE_CATEGORIES,
    PROTOCOL_BOX,
    PROTOCOL_NATIVE,
    PROTOCOL_RAW,
    PROTOCOL_STRESS,
    PROTOCOL_TUBE,
    RAW_FIELDS,
    load_spec,
    manifest_digest,
    read_csv,
    sha256_manifest,
    write_json,
)

HERE = Path(__file__).resolve().parent


def _float(row: Mapping[str, str], key: str) -> float:
    return float(row[key])


def _bool(value: str) -> bool:
    return value.strip().lower() == "true"


def _write_table(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    records = list(rows)
    fields = sorted({key for row in records for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or ["status"])
        writer.writeheader()
        writer.writerows(records)


def _rhs(system: Mapping[str, Any], _: float, state: np.ndarray) -> np.ndarray:
    values = []
    for polynomial in system["rhs"]:
        total = 0.0
        for term in polynomial["terms"]:
            value = float(term["coefficient"])
            for coordinate, exponent in zip(state, term["powers"]):
                value *= float(coordinate) ** int(exponent)
            total += value
        values.append(total)
    return np.asarray(values)


def _sample_points(box: list[list[float]]) -> list[tuple[float, ...]]:
    axes = [
        (float(bounds[0]), 0.5 * (float(bounds[0]) + float(bounds[1])), float(bounds[1]))
        for bounds in box
    ]
    return list(product(*axes))


def _trajectory_cache(
    spec: Mapping[str, Any], system_name: str, horizon: float
) -> list[Any]:
    system = spec["systems"][system_name]
    solutions = []
    for point in _sample_points(system["initial_box"]):
        solutions.append(
            solve_ivp(
                lambda t, x: _rhs(system, t, x),
                (0.0, horizon),
                np.asarray(point, dtype=np.float64),
                method="DOP853",
                rtol=1e-12,
                atol=1e-14,
                dense_output=True,
            ).sol
        )
    return solutions


def _annotate_trajectories(
    spec: Mapping[str, Any], rows: list[dict[str, str]]
) -> dict[str, int]:
    caches: dict[tuple[str, float], list[Any]] = {}
    passed = failed = not_applicable = 0
    tolerance = float(spec["trajectory_tolerance"])
    for row in rows:
        if row["interval_kind"] not in {"tube", "endpoint_raw", "endpoint_tightened"}:
            row["sampled_trajectory_status"] = "not_applicable"
            not_applicable += 1
            continue
        if row["lower"] == "" or row["upper"] == "":
            row["sampled_trajectory_status"] = "not_checked"
            not_applicable += 1
            continue
        system_name = row["system"]
        horizon = _float(row, "requested_horizon")
        key = (system_name, horizon)
        if key not in caches:
            caches[key] = _trajectory_cache(spec, system_name, horizon)
        absolute_time = _float(row, "absolute_time")
        h = _float(row, "h")
        if row["interval_kind"] == "tube":
            times = np.linspace(max(0.0, absolute_time - h), absolute_time, 5)
        else:
            times = np.asarray([absolute_time])
        state = int(row["state_index"])
        values = np.concatenate(
            [np.asarray(solution(times))[state].reshape(-1) for solution in caches[key]]
        )
        lower, upper = _float(row, "lower"), _float(row, "upper")
        contains = bool(
            np.min(values) >= lower - tolerance
            and np.max(values) <= upper + tolerance
        )
        row["sampled_trajectory_status"] = "passed" if contains else "failed"
        if contains:
            passed += 1
        else:
            failed += 1
    return {"passed": passed, "failed": failed, "not_applicable": not_applicable}


def _endpoint_tube_violations(rows: list[dict[str, str]]) -> list[str]:
    tubes: dict[tuple[str, str, str, str, str, str], tuple[float, float]] = {}
    endpoints: list[dict[str, str]] = []
    for row in rows:
        key = (
            row["tool"],
            row["tool_variant"],
            row["system"],
            row["h"],
            row["absolute_time"],
            row["state_index"],
        )
        if row["protocol"] == PROTOCOL_TUBE and row["interval_kind"] == "tube":
            tubes[key] = (_float(row, "lower"), _float(row, "upper"))
        if row["protocol"] == PROTOCOL_RAW and row["interval_kind"] == "endpoint_raw":
            endpoints.append(row)
    violations = []
    for row in endpoints:
        key = (
            row["tool"],
            row["tool_variant"],
            row["system"],
            row["h"],
            row["absolute_time"],
            row["state_index"],
        )
        tube = tubes.get(key)
        if tube is None:
            if "diagnostic" not in row["tool_variant"]:
                violations.append(f"missing tube for {key}")
            continue
        if _float(row, "lower") < tube[0] - 1e-15 or _float(row, "upper") > tube[1] + 1e-15:
            violations.append(f"endpoint outside tube for {key}")
    return violations


def _late_point_violations(rows: list[dict[str, str]]) -> list[str]:
    by_run: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_run[row["run_id"]].append(row)
    violations = []
    for run_id, group in by_run.items():
        failures = [
            int(row["step_index"])
            for row in group
            if row["interval_kind"] == "failure"
        ]
        if not failures:
            continue
        first = min(failures)
        if any(
            int(row["step_index"]) >= first
            and row["interval_kind"] not in {"failure"}
            for row in group
        ):
            violations.append(run_id)
    return violations


def _one_step_summary(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    return [
        {
            key: row[key]
            for key in (
                "tool",
                "tool_variant",
                "protocol",
                "system",
                "h",
                "state_index",
                "interval_kind",
                "lower",
                "upper",
                "width",
                "exact_lower",
                "exact_upper",
                "lower_error",
                "upper_error",
                "inflation_ratio",
                "analytic_reference_status",
                "sampled_trajectory_status",
                "endpoint_semantics",
            )
        }
        for row in rows
        if row["protocol"] in {PROTOCOL_TUBE, PROTOCOL_RAW}
        and row["interval_kind"] in {"tube", "endpoint_raw", "endpoint_tightened"}
    ]


def _common_time_summary(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        if row["protocol"] != PROTOCOL_BOX or row["interval_kind"] != "endpoint_raw":
            continue
        result.append(
            {
                key: row[key]
                for key in (
                    "tool",
                    "tool_variant",
                    "system",
                    "h",
                    "requested_horizon",
                    "step_index",
                    "absolute_time",
                    "state_index",
                    "lower",
                    "upper",
                    "width",
                    "analytic_reference_status",
                    "sampled_trajectory_status",
                )
            }
        )
    return result


def _failure_summary(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    by_run: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_run[row["run_id"]].append(row)
    result = []
    for run_id, group in by_run.items():
        endpoints = [
            row
            for row in group
            if row["interval_kind"] in {"endpoint_raw", "tube"}
        ]
        failures = [row for row in group if row["interval_kind"] == "failure"]
        first = group[0]
        result.append(
            {
                "run_id": run_id,
                "tool": first["tool"],
                "tool_variant": first["tool_variant"],
                "protocol": first["protocol"],
                "system": first["system"],
                "h": first["h"],
                "requested_horizon": first["requested_horizon"],
                "successful_horizon": max(
                    (_float(row, "absolute_time") for row in endpoints), default=0.0
                ),
                "failure_category": failures[0]["failure_category"] if failures else "",
                "failure_message": failures[0]["failure_message"] if failures else "",
            }
        )
    return result


def _claim_rows() -> list[dict[str, str]]:
    return [
        {
            "old_claim": "Torch is tightest on Riccati",
            "evidence_originally_used": "tightened Torch endpoint versus Flow*/DiffReach raw endpoints",
            "confounder": "endpoint postprocessing mismatch",
            "status": "invalid",
            "corrected_wording": "Torch tightening is supplemental; raw endpoint comparisons are required",
            "supporting_new_artifact": "torch_endpoint_audit.csv",
        },
        {
            "old_claim": "Flow* fails around t=0.08",
            "evidence_originally_used": "fixed order-2/fixed-candidate wrapper",
            "confounder": "constrained stress configuration and generic failure code",
            "status": "invalid",
            "corrected_wording": "that configuration fails; stock Flow* reaches T=10 under its original settings",
            "supporting_new_artifact": "flowstar_original_parity.csv",
        },
        {
            "old_claim": "Flow* order 2 is less capable",
            "evidence_originally_used": "deliberate low-order stress",
            "confounder": "different minimum legal bases and resource settings",
            "status": "invalid",
            "corrected_wording": "order-2 fixed stress is diagnostic, not general capability",
            "supporting_new_artifact": "flowstar_parameter_sensitivity.csv",
        },
        {
            "old_claim": "DiffReach is tighter on harmonic",
            "evidence_originally_used": "common-box widths",
            "confounder": "tool-specific local bases remain unmatched",
            "status": "unresolved",
            "corrected_wording": "report matched raw semantics and configuration caveats only",
            "supporting_new_artifact": "corrected_one_step_summary.csv",
        },
        {
            "old_claim": "DiffReach is faster",
            "evidence_originally_used": "mixed compile/JIT/steady totals",
            "confounder": "incomparable one-time and steady costs",
            "status": "invalid",
            "corrected_wording": "compile, JIT, warmup, and steady costs are separate",
            "supporting_new_artifact": "raw_results.csv",
        },
        {
            "old_claim": "common-box comparison is fair",
            "evidence_originally_used": "same external boxes",
            "confounder": "box reset hides native dependency-preservation behavior",
            "status": "corrected",
            "corrected_wording": "common-box controls carry representation but is not a native-method ranking",
            "supporting_new_artifact": "corrected_common_time_summary.csv",
        },
        {
            "old_claim": "all correctness gates passed",
            "evidence_originally_used": "candidate reinjection rows",
            "confounder": "stock Flow* analytic violation was excluded by postprocessing",
            "status": "invalid",
            "corrected_wording": "stock Riccati exact-reference checks fail; Outcome B applies",
            "supporting_new_artifact": "correctness_checks.json",
        },
        {
            "old_claim": "Flow* refinement was unvalidated",
            "evidence_originally_used": "code inspection and preliminary Riccati anomaly",
            "confounder": "no full-Picard post-refinement recheck had been executed",
            "status": "confirmed",
            "corrected_wording": "the remainder-only refined image fails a regenerated full-Picard inclusion check",
            "supporting_new_artifact": "flowstar_refinement_trace.csv",
        },
    ]


def collect(spec: Mapping[str, Any], output: Path, strict: bool) -> dict[str, Any]:
    rows: list[dict[str, str]] = []
    for filename in (
        "torch_endpoint_audit.csv",
        "diffreach_endpoint_audit.csv",
        "flowstar_audit.csv",
    ):
        rows.extend(read_csv(output / filename))
    trajectory_counts = _annotate_trajectories(spec, rows)

    failures = [row for row in rows if row["interval_kind"] == "failure"]
    exact_rows = [
        row
        for row in rows
        if row["analytic_reference_status"] in {"passed", "failed"}
        and row["interval_kind"] in {"tube", "endpoint_raw", "endpoint_tightened"}
    ]
    stock_exact_failures = [
        row
        for row in exact_rows
        if row["tool"] == "flowstar"
        and row["tool_variant"] == "flowstar_stock"
        and row["analytic_reference_status"] == "failed"
    ]
    non_flowstar_exact_failures = [
        row
        for row in exact_rows
        if row["tool"] != "flowstar"
        and row["analytic_reference_status"] == "failed"
    ]
    non_flowstar_trajectory_failures = [
        row
        for row in rows
        if row["tool"] != "flowstar"
        and row["sampled_trajectory_status"] == "failed"
    ]
    finite_violations = [
        row["run_id"]
        for row in rows
        if row["lower"] != ""
        and (
            not math.isfinite(_float(row, "lower"))
            or not math.isfinite(_float(row, "upper"))
        )
    ]
    endpoint_tube = _endpoint_tube_violations(rows)
    late_points = _late_point_violations(rows)
    failure_category_violations = [
        row["run_id"]
        for row in failures
        if row["failure_category"] not in FAILURE_CATEGORIES
    ]
    stock_overwrites = [
        row["run_id"]
        for row in rows
        if row["tool_variant"] == "flowstar_stock"
        and _bool(row["remainder_overwrite_applied"])
    ]
    primary_tightening = [
        row["run_id"]
        for row in rows
        if row["tool"] == "torch_tm_flowpipe"
        and row["interval_kind"] == "endpoint_raw"
        and _bool(row["endpoint_tightening_applied"])
    ]
    parity = json.loads(
        (output / "flowstar_original_parity_summary.json").read_text(encoding="utf-8")
    )
    before = json.loads(
        (output / "frozen_manifest_before.json").read_text(encoding="utf-8")
    )
    frozen_root = Path(spec["repositories"]["torch"]) / spec["frozen_result"]
    after_manifest = sha256_manifest(frozen_root)
    after_digest = manifest_digest(after_manifest)
    before_digest = before["manifest_digest"]
    write_json(
        output / "frozen_manifest_after.json",
        {"manifest": after_manifest, "manifest_digest": after_digest},
    )

    checks = {
        "schema_complete": {
            "passed": all(list(row) == RAW_FIELDS for row in rows),
            "violations": 0 if all(list(row) == RAW_FIELDS for row in rows) else 1,
        },
        "frozen_artifact_unchanged": {
            "passed": before_digest == after_digest,
            "violations": 0 if before_digest == after_digest else 1,
            "before": before_digest,
            "after": after_digest,
        },
        "flowstar_stock_has_no_overwrite": {
            "passed": not stock_overwrites,
            "violations": len(stock_overwrites),
        },
        "primary_torch_uses_raw_endpoint": {
            "passed": not primary_tightening,
            "violations": len(primary_tightening),
        },
        "flowstar_original_parity": {
            "passed": bool(parity["passed"]),
            "violations": 0 if parity["passed"] else 1,
        },
        "torch_and_diffreach_exact_references": {
            "passed": not non_flowstar_exact_failures,
            "violations": len(non_flowstar_exact_failures),
        },
        "flowstar_stock_exact_references": {
            "passed": not stock_exact_failures,
            "violations": len(stock_exact_failures),
        },
        "endpoint_contained_in_tube": {
            "passed": not endpoint_tube,
            "violations": len(endpoint_tube),
            "details": endpoint_tube[:20],
        },
        "sampled_trajectories": {
            "passed": trajectory_counts["failed"] == 0,
            "violations": trajectory_counts["failed"],
            "counts": trajectory_counts,
        },
        "torch_and_diffreach_sampled_trajectories": {
            "passed": not non_flowstar_trajectory_failures,
            "violations": len(non_flowstar_trajectory_failures),
        },
        "finite_arithmetic": {
            "passed": not finite_violations,
            "violations": len(finite_violations),
        },
        "failure_categories_populated": {
            "passed": not failure_category_violations,
            "violations": len(failure_category_violations),
        },
        "no_points_after_failure": {
            "passed": not late_points,
            "violations": len(late_points),
        },
    }
    torch_diff_valid = all(
        checks[name]["passed"]
        for name in (
            "torch_and_diffreach_exact_references",
            "endpoint_contained_in_tube",
            "torch_and_diffreach_sampled_trajectories",
            "finite_arithmetic",
        )
    )
    if (
        checks["flowstar_original_parity"]["passed"]
        and checks["flowstar_stock_exact_references"]["passed"]
        and torch_diff_valid
    ):
        outcome = "A"
    elif torch_diff_valid and checks["flowstar_original_parity"]["passed"]:
        outcome = "B"
    else:
        outcome = "C"
    result = {
        "outcome": outcome,
        "all_gates_passed": all(item["passed"] for item in checks.values()),
        "gates": checks,
        "counts": {
            "raw_rows": len(rows),
            "exact_reference_rows": len(exact_rows),
            "exact_reference_passed": sum(
                row["analytic_reference_status"] == "passed" for row in exact_rows
            ),
            "exact_reference_failed": sum(
                row["analytic_reference_status"] == "failed" for row in exact_rows
            ),
            "flowstar_stock_exact_failures": len(stock_exact_failures),
            "failure_rows": len(failures),
            "trajectory_checks_passed": trajectory_counts["passed"],
            "trajectory_checks_failed": trajectory_counts["failed"],
            "torch_diffreach_trajectory_checks_failed": len(
                non_flowstar_trajectory_failures
            ),
        },
    }
    with (output / "raw_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RAW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    _write_table(output / "corrected_one_step_summary.csv", _one_step_summary(rows))
    _write_table(output / "corrected_common_time_summary.csv", _common_time_summary(rows))
    _write_table(output / "corrected_failure_horizon_summary.csv", _failure_summary(rows))
    claims = _claim_rows()
    _write_table(output / "claim_audit.csv", claims)
    write_json(output / "correctness_checks.json", result)
    if strict and outcome == "C":
        raise SystemExit("corrected comparison gates require Outcome C")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", default=str(HERE / "benchmark_spec.yaml"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    result = collect(
        load_spec(args.spec), Path(args.output_dir).resolve(), args.strict
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
