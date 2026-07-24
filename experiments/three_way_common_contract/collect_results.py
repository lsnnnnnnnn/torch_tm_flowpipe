#!/usr/bin/env python3
"""Collect adapter output, enforce correctness gates, and build comparison CSVs."""
from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.integrate import solve_ivp

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from common import (
    PROTOCOL_A,
    PROTOCOL_B,
    PROTOCOL_C,
    RAW_FIELDS,
    RUN_FIELDS,
    configuration_key,
    deterministic_initial_points,
    evaluate_rhs,
    git_sha,
    iter_configurations,
    load_spec,
    read_csv,
    read_json,
    write_csv,
    write_json,
)

PRIMARY_VARIANTS = {
    "torch_tm_flowpipe": "complete_total_degree_order_1",
    "diffreach": "affine_flag",
    "flowstar": "minimum_supported_fixed_order_2",
}


def _float(value: Any, default: float = math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _is_number(value: Any) -> bool:
    return math.isfinite(_float(value))


def _close(a: Any, b: Any, tolerance: float = 1e-10) -> bool:
    return math.isclose(
        _float(a), _float(b), rel_tol=0.0, abs_tol=tolerance
    )


def _git_tracked_clean(path: str | Path) -> bool:
    result = subprocess.run(
        ["git", "-C", str(path), "diff", "--quiet"],
        check=False,
    )
    cached = subprocess.run(
        ["git", "-C", str(path), "diff", "--cached", "--quiet"],
        check=False,
    )
    return result.returncode == 0 and cached.returncode == 0


def _gate(checks: int, violations: Sequence[Any], **extra: Any) -> dict[str, Any]:
    return {
        "checks": int(checks),
        "violations": len(violations),
        "examples": list(violations[:20]),
        "passed": len(violations) == 0,
        **extra,
    }


def _expected_run_keys(
    spec: Mapping[str, Any], *, smoke: bool
) -> set[tuple[str, str, str, str, float, float]]:
    expected: set[tuple[str, str, str, str, float, float]] = set()
    for config in iter_configurations(spec, smoke=smoke):
        for tool, variant in PRIMARY_VARIANTS.items():
            expected.add(
                configuration_key(
                    tool,
                    variant,
                    config["protocol"],
                    config["system"],
                    config["h"],
                    config["horizon"],
                )
            )
        if config["protocol"] == PROTOCOL_C:
            expected.add(
                configuration_key(
                    "diffreach",
                    "default_restricted_quasi_quadratic",
                    config["protocol"],
                    config["system"],
                    config["h"],
                    config["horizon"],
                )
            )
    return expected


def _actual_run_key(run: Mapping[str, Any]) -> tuple[str, str, str, str, float, float]:
    return configuration_key(
        str(run["tool"]),
        str(run["tool_variant"]),
        str(run["protocol"]),
        str(run["system"]),
        _float(run["h"]),
        _float(run["horizon"]),
    )


def _point_gate(
    spec: Mapping[str, Any],
    output: Path,
) -> dict[str, Any]:
    violations: list[dict[str, Any]] = []
    checks = 0
    for filename in (
        "torch_point_evaluations.json",
        "diffreach_point_evaluations.json",
        "flowstar_point_evaluations.json",
    ):
        path = output / filename
        if not path.exists():
            violations.append({"missing": filename})
            continue
        document = read_json(path)
        actual = {
            (item["system"], int(item["point_index"])): item
            for item in document["values"]
        }
        for system_name, system in spec["systems"].items():
            for point_index, point in enumerate(system["point_checks"]):
                expected = [
                    float(value)
                    for value in evaluate_rhs(
                        list(map(float, point)), system
                    )
                ]
                item = actual.get((system_name, point_index))
                if item is None:
                    violations.append(
                        {
                            "tool_file": filename,
                            "system": system_name,
                            "point_index": point_index,
                            "reason": "missing",
                        }
                    )
                    continue
                if len(item["value"]) != len(expected):
                    violations.append(
                        {
                            "tool_file": filename,
                            "system": system_name,
                            "point_index": point_index,
                            "reason": "dimension",
                        }
                    )
                    continue
                for state, (got, wanted) in enumerate(
                    zip(item["value"], expected)
                ):
                    checks += 1
                    if not _close(got, wanted, 1e-12):
                        violations.append(
                            {
                                "tool_file": filename,
                                "system": system_name,
                                "point_index": point_index,
                                "state": state,
                                "actual": got,
                                "expected": wanted,
                            }
                        )
                    intervals = item.get("intervals")
                    if intervals is not None and not (
                        intervals[state][0] <= wanted <= intervals[state][1]
                    ):
                        violations.append(
                            {
                                "tool_file": filename,
                                "system": system_name,
                                "point_index": point_index,
                                "state": state,
                                "reason": "Flow* point interval misses expected",
                            }
                        )
    return _gate(checks, violations)


def _initial_contract_gate(
    spec: Mapping[str, Any],
    runs: Sequence[Mapping[str, Any]],
    rows_by_run: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    checks = 0
    violations: list[dict[str, Any]] = []
    for run in runs:
        system = spec["systems"][run["system"]]
        initial = [
            row
            for row in rows_by_run[run["run_id"]]
            if int(row["step_index"]) == 0
            and row["interval_kind"] == "endpoint"
        ]
        checks += len(system["state_names"]) * 3
        if len(initial) != len(system["state_names"]):
            violations.append(
                {"run_id": run["run_id"], "reason": "initial row count"}
            )
            continue
        initial.sort(key=lambda row: int(row["state_index"]))
        for state, row in enumerate(initial):
            expected = system["initial_box"][state]
            if (
                int(row["state_index"]) != state
                or row["state_name"] != system["state_names"][state]
                or not _close(row["lower"], expected[0])
                or not _close(row["upper"], expected[1])
            ):
                violations.append(
                    {
                        "run_id": run["run_id"],
                        "state": state,
                        "actual": [
                            row["state_index"],
                            row["state_name"],
                            row["lower"],
                            row["upper"],
                        ],
                        "expected": [
                            state,
                            system["state_names"][state],
                            *expected,
                        ],
                    }
                )
    return _gate(checks, violations)


def _endpoint_tube_gate(
    rows_by_run: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    checks = 0
    violations: list[dict[str, Any]] = []
    for run_id, rows in rows_by_run.items():
        index = {
            (
                int(row["step_index"]),
                int(row["state_index"]),
                row["interval_kind"],
            ): row
            for row in rows
            if _is_number(row["lower"]) and int(row["step_index"]) > 0
        }
        endpoints = [
            row
            for row in rows
            if row["interval_kind"] == "endpoint"
            and int(row["step_index"]) > 0
            and row["row_status"] == "validated"
        ]
        for endpoint in endpoints:
            key = (
                int(endpoint["step_index"]),
                int(endpoint["state_index"]),
                "tube",
            )
            tube = index.get(key)
            checks += 1
            if tube is None:
                violations.append(
                    {
                        "run_id": run_id,
                        "step": key[0],
                        "state": key[1],
                        "reason": "missing tube",
                    }
                )
            elif not (
                _float(tube["lower"]) <= _float(endpoint["lower"]) + 1e-12
                and _float(tube["upper"]) >= _float(endpoint["upper"]) - 1e-12
            ):
                violations.append(
                    {
                        "run_id": run_id,
                        "step": key[0],
                        "state": key[1],
                        "endpoint": [endpoint["lower"], endpoint["upper"]],
                        "tube": [tube["lower"], tube["upper"]],
                    }
                )
    return _gate(checks, violations)


def _analytic_gate(
    rows: Sequence[Mapping[str, Any]],
    *,
    protocol: str | None = None,
    system: str | None = None,
    endpoints_only: bool = False,
) -> dict[str, Any]:
    selected = [
        row
        for row in rows
        if row["system"] in {"riccati", "harmonic"}
        and int(row["step_index"]) > 0
        and row["interval_kind"] in {"endpoint", "tube"}
        and (protocol is None or row["protocol"] == protocol)
        and (system is None or row["system"] == system)
        and (not endpoints_only or row["interval_kind"] == "endpoint")
        and row["native_validation_status"] == "validated"
    ]
    violations = [
        {
            "run_id": row["run_id"],
            "step": row["step_index"],
            "state": row["state_index"],
            "kind": row["interval_kind"],
            "actual": [row["lower"], row["upper"]],
            "exact": [row["exact_lower"], row["exact_upper"]],
        }
        for row in selected
        if not (
            _is_number(row["exact_lower"])
            and _float(row["lower"]) <= _float(row["exact_lower"]) + 1e-12
            and _float(row["upper"]) >= _float(row["exact_upper"]) - 1e-12
        )
    ]
    return _gate(len(selected), violations)


def _native_status_gate(
    runs: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    violations: list[dict[str, Any]] = []
    checks = 0
    for run in runs:
        checks += 1
        success = run["run_status"] == "success"
        if success and (
            run["native_validation_status"] != "validated"
            or int(run["completed_steps"]) != int(run["requested_steps"])
        ):
            violations.append(
                {
                    "run_id": run["run_id"],
                    "reason": "successful run has incomplete/failed native status",
                }
            )
    for row in rows:
        if row["interval_kind"] == "failure_marker":
            continue
        checks += 1
        if row["row_status"] == "validated" and row[
            "native_validation_status"
        ] not in {"validated", "initial_set"}:
            violations.append(
                {
                    "run_id": row["run_id"],
                    "step": row["step_index"],
                    "reason": "validated row has failed native status",
                }
            )
    return _gate(checks, violations)


def _flowstar_gate(
    spec: Mapping[str, Any],
    runs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    expected_workaround = spec["flowstar"]["extraction_workaround"]
    selected = [run for run in runs if run["tool"] == "flowstar"]
    violations: list[dict[str, Any]] = []
    for run in selected:
        guard = run.get("minimum_order_guard")
        if run["extraction_workaround"] != expected_workaround:
            violations.append(
                {"run_id": run["run_id"], "reason": "workaround label"}
            )
        if guard != {
            "order1_supported": False,
            "order2_supported": True,
        }:
            violations.append(
                {
                    "run_id": run["run_id"],
                    "reason": "minimum order guard",
                    "actual": guard,
                }
            )
        if str(run["local_order"]) != "2":
            violations.append(
                {"run_id": run["run_id"], "reason": "not order 2"}
            )
    return _gate(len(selected) * 3, violations)


def _diffreach_gate(
    output: Path,
    runs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    provenance_path = output / "diffreach_upstream_provenance.json"
    violations: list[dict[str, Any]] = []
    checks = 0
    if not provenance_path.exists():
        return _gate(0, [{"reason": "missing provenance"}])
    provenance = read_json(provenance_path)
    for key in (
        "upstream_step_source_file",
        "upstream_picard_source_file",
        "upstream_taylor_model_source_file",
    ):
        checks += 1
        if not str(provenance.get(key, "")).startswith(
            "/srv/local/shengenli/DiffReach/src/"
        ):
            violations.append({"reason": key, "actual": provenance.get(key)})
    for key in (
        "upstream_step_callable_identity",
        "adapter_calls_saved_upstream_step_directly",
    ):
        checks += 1
        if provenance.get(key) is not True:
            violations.append({"reason": key})
    checks += 1
    if int(provenance.get("total_upstream_step_trace_invocations", 0)) <= 0:
        violations.append({"reason": "no upstream trace invocation"})
    primary = [
        run
        for run in runs
        if run["tool"] == "diffreach" and run["tool_variant"] == "affine_flag"
    ]
    for run in primary:
        checks += 1
        if int(run.get("upstream_step_trace_invocations", 0)) <= 0:
            violations.append(
                {
                    "run_id": run["run_id"],
                    "reason": "primary run did not trace upstream step",
                }
            )
    return _gate(
        checks,
        violations,
        primary_upstream_runs=len(primary),
        provenance=provenance,
    )


def _trajectory_reference(
    system: Mapping[str, Any],
    initial_point: Sequence[float],
    times: np.ndarray,
) -> np.ndarray:
    def rhs(_: float, state: np.ndarray) -> np.ndarray:
        return np.asarray(
            evaluate_rhs(list(state), system), dtype=np.float64
        )

    if times[-1] == 0:
        return np.asarray(initial_point, dtype=np.float64)[None, :]
    result = solve_ivp(
        rhs,
        (0.0, float(times[-1])),
        np.asarray(initial_point, dtype=np.float64),
        method="DOP853",
        t_eval=times,
        rtol=1e-12,
        atol=1e-14,
        max_step=max(float(times[-1]) / 1000.0, 1e-5),
    )
    if not result.success:
        raise RuntimeError(result.message)
    return result.y.T


def _trajectory_gate(
    spec: Mapping[str, Any],
    runs: Sequence[Mapping[str, Any]],
    rows_by_run: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    checks = 0
    violations: list[dict[str, Any]] = []
    tolerance = float(spec["trajectory_tolerance"])
    substeps = int(spec["trajectory_substeps_per_segment"])
    cache: dict[
        tuple[str, float, float],
        tuple[np.ndarray, list[np.ndarray]],
    ] = {}
    for run in runs:
        completed = int(run["completed_steps"])
        if completed <= 0:
            continue
        system_name = run["system"]
        h = _float(run["h"])
        final_time = completed * h
        cache_key = (system_name, round(h, 12), round(final_time, 12))
        if cache_key not in cache:
            times = np.asarray(
                sorted(
                    {
                        round((step - 1 + part / substeps) * h, 14)
                        for step in range(1, completed + 1)
                        for part in range(substeps + 1)
                    }
                ),
                dtype=np.float64,
            )
            trajectories = [
                _trajectory_reference(
                    spec["systems"][system_name], point, times
                )
                for point in deterministic_initial_points(
                    spec["systems"][system_name]["initial_box"]
                )
            ]
            cache[cache_key] = times, trajectories
        times, trajectories = cache[cache_key]
        lookup = {
            round(float(value), 14): index
            for index, value in enumerate(times)
        }
        for row in rows_by_run[run["run_id"]]:
            if (
                row["row_status"] != "validated"
                or row["interval_kind"] not in {"endpoint", "tube"}
                or int(row["step_index"]) <= 0
            ):
                continue
            state = int(row["state_index"])
            step_index = int(row["step_index"])
            if row["interval_kind"] == "endpoint":
                sample_times = [round(step_index * h, 14)]
            else:
                sample_times = [
                    round(
                        (step_index - 1 + part / substeps) * h,
                        14,
                    )
                    for part in range(substeps + 1)
                ]
            for trajectory_index, trajectory in enumerate(trajectories):
                for sample_time in sample_times:
                    checks += 1
                    value = float(trajectory[lookup[sample_time], state])
                    if not (
                        value >= _float(row["lower"]) - tolerance
                        and value <= _float(row["upper"]) + tolerance
                    ):
                        violations.append(
                            {
                                "run_id": run["run_id"],
                                "step": row["step_index"],
                                "kind": row["interval_kind"],
                                "state": state,
                                "trajectory": trajectory_index,
                                "sample_time": sample_time,
                                "sample": value,
                                "interval": [row["lower"], row["upper"]],
                            }
                        )
    return _gate(
        checks,
        violations,
        method="SciPy DOP853 rtol=1e-12 atol=1e-14 deterministic corners/center/face-centers",
        proves_soundness=False,
    )


def _one_step_summary(
    spec: Mapping[str, Any],
    runs: Sequence[Mapping[str, Any]],
    rows_by_run: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for run in runs:
        if run["protocol"] != PROTOCOL_A:
            continue
        endpoint_by_state = {
            int(row["state_index"]): row
            for row in rows_by_run[run["run_id"]]
            if row["interval_kind"] == "endpoint"
            and int(row["step_index"]) == 1
        }
        for state, state_name in enumerate(
            spec["systems"][run["system"]]["state_names"]
        ):
            row = endpoint_by_state.get(state)
            output.append(
                {
                    "tool": run["tool"],
                    "tool_variant": run["tool_variant"],
                    "system": run["system"],
                    "h": run["h"],
                    "state_index": state,
                    "state_name": state_name,
                    "status": (
                        "validated" if row is not None else "validation_failed"
                    ),
                    "lower": "" if row is None else row["lower"],
                    "upper": "" if row is None else row["upper"],
                    "width": "" if row is None else row["width"],
                    "exact_width": "" if row is None else row["exact_width"],
                    "exact_inflation_ratio": (
                        "" if row is None else row["exact_inflation_ratio"]
                    ),
                    "native_validation_status": run[
                        "native_validation_status"
                    ],
                    "local_order": run["local_order"],
                    "local_retained_basis": run["local_retained_basis"],
                    "interval_remainder_width": (
                        "" if row is None else row["interval_remainder_width"]
                    ),
                    "polynomial_width": (
                        "" if row is None else row["polynomial_width"]
                    ),
                    "steady_runtime_per_step_s": run[
                        "steady_runtime_per_step_s"
                    ],
                    "first_failure_time": run["first_failure_time"],
                }
            )
    return output


def _common_time_summary(
    spec: Mapping[str, Any],
    runs: Sequence[Mapping[str, Any]],
    rows_by_run: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    primary_runs = [
        run
        for run in runs
        if run["protocol"] == PROTOCOL_B
        and PRIMARY_VARIANTS.get(run["tool"]) == run["tool_variant"]
    ]
    for run in primary_runs:
        system = spec["systems"][run["system"]]
        endpoints = {
            (round(_float(row["time"]), 12), int(row["state_index"])): row
            for row in rows_by_run[run["run_id"]]
            if row["interval_kind"] == "endpoint"
            and int(row["step_index"]) > 0
            and row["row_status"] == "validated"
        }
        for checkpoint in spec["common_time_checkpoints"][run["system"]]:
            checkpoint = float(checkpoint)
            if checkpoint > _float(run["horizon"]) + 1e-12:
                continue
            for state, state_name in enumerate(system["state_names"]):
                row = endpoints.get((round(checkpoint, 12), state))
                output.append(
                    {
                        "tool": run["tool"],
                        "tool_variant": run["tool_variant"],
                        "system": run["system"],
                        "h": run["h"],
                        "checkpoint": checkpoint,
                        "state_index": state,
                        "state_name": state_name,
                        "status": (
                            "validated" if row else "validation_failed"
                        ),
                        "lower": "" if row is None else row["lower"],
                        "upper": "" if row is None else row["upper"],
                        "width": "" if row is None else row["width"],
                        "exact_width": (
                            "" if row is None else row["exact_width"]
                        ),
                        "exact_inflation_ratio": (
                            ""
                            if row is None
                            else row["exact_inflation_ratio"]
                        ),
                        "first_failure_time": run["first_failure_time"],
                    }
                )
    return output


def _failure_summary(
    spec: Mapping[str, Any],
    runs: Sequence[Mapping[str, Any]],
    rows_by_run: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for run in runs:
        if run["protocol"] not in {PROTOCOL_B, PROTOCOL_C}:
            continue
        endpoints = [
            row
            for row in rows_by_run[run["run_id"]]
            if row["interval_kind"] == "endpoint"
            and int(row["step_index"]) > 0
            and row["row_status"] == "validated"
        ]
        by_state: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
        for row in endpoints:
            by_state[int(row["state_index"])].append(row)
        for state, state_name in enumerate(
            spec["systems"][run["system"]]["state_names"]
        ):
            state_rows = sorted(
                by_state[state], key=lambda row: _float(row["time"])
            )
            final = state_rows[-1] if state_rows else None
            failure_horizon = (
                _float(run["first_failure_time"])
                if _is_number(run["first_failure_time"])
                else _float(run["horizon"])
            )
            output.append(
                {
                    "tool": run["tool"],
                    "tool_variant": run["tool_variant"],
                    "protocol": run["protocol"],
                    "system": run["system"],
                    "h": run["h"],
                    "horizon": run["horizon"],
                    "state_index": state,
                    "state_name": state_name,
                    "run_status": run["run_status"],
                    "first_failure_time": run["first_failure_time"],
                    "failure_horizon_or_censor": failure_horizon,
                    "censored_at_requested_horizon": (
                        run["run_status"] == "success"
                    ),
                    "final_valid_time": (
                        "" if final is None else final["time"]
                    ),
                    "width_at_own_final_valid_step": (
                        "" if final is None else final["width"]
                    ),
                }
            )
    return output


def _runtime_summary(
    runs: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    fields = [
        "tool",
        "tool_variant",
        "protocol",
        "system",
        "h",
        "horizon",
        "run_status",
        "build_time_s",
        "jit_compile_time_s",
        "first_execution_time_s",
        "steady_runtime_per_step_s",
        "orchestration_time_s",
        "completed_steps",
    ]
    return [{field: run.get(field, "") for field in fields} for run in runs]


def _semantics_summary(
    runs: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    seen: set[tuple[Any, ...]] = set()
    output: list[dict[str, Any]] = []
    fields = [
        "tool",
        "tool_variant",
        "protocol",
        "local_order",
        "local_retained_basis",
        "carried_representation",
        "reset_policy",
        "validator",
        "extraction_workaround",
    ]
    for run in runs:
        row = {field: run.get(field, "") for field in fields}
        key = tuple(row[field] for field in fields)
        if key not in seen:
            output.append(row)
            seen.add(key)
    return output


def _write_table(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    write_csv(path, rows, fields)


def collect(
    spec: Mapping[str, Any],
    output: Path,
    *,
    smoke: bool,
    strict: bool,
) -> dict[str, Any]:
    raw_rows: list[dict[str, Any]] = []
    runs: list[dict[str, Any]] = []
    for tool in ("torch", "diffreach", "flowstar"):
        raw_rows.extend(read_csv(output / f"{tool}_raw_results.csv"))
        runs.extend(read_json(output / f"{tool}_runs.json"))
    rows_by_run: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        rows_by_run[row["run_id"]].append(row)

    expected = _expected_run_keys(spec, smoke=smoke)
    actual = {_actual_run_key(run) for run in runs}
    coverage_violations = [
        {"missing": list(item)} for item in sorted(expected - actual)
    ] + [{"unexpected": list(item)} for item in sorted(actual - expected)]
    coverage = _gate(
        len(expected),
        coverage_violations,
        expected_runs=len(expected),
        actual_runs=len(actual),
    )
    checks = {
        "configuration_coverage": coverage,
        "identical_ode_point_evaluations": _point_gate(spec, output),
        "identical_initial_boxes_and_state_order": _initial_contract_gate(
            spec, runs, rows_by_run
        ),
        "endpoint_vs_whole_tube_extraction": _endpoint_tube_gate(rows_by_run),
        "riccati_one_step_exact_containment": _analytic_gate(
            raw_rows,
            protocol=PROTOCOL_A,
            system="riccati",
            endpoints_only=True,
        ),
        "harmonic_one_step_exact_containment": _analytic_gate(
            raw_rows,
            protocol=PROTOCOL_A,
            system="harmonic",
            endpoints_only=True,
        ),
        "all_analytic_intervals_contained": _analytic_gate(raw_rows),
        "deterministic_high_accuracy_trajectory_sanity": _trajectory_gate(
            spec, runs, rows_by_run
        ),
        "native_validation_status_consistency": _native_status_gate(
            runs, raw_rows
        ),
        "flowstar_order_and_extraction_workaround": _flowstar_gate(
            spec, runs
        ),
        "diffreach_real_upstream_operations": _diffreach_gate(output, runs),
    }
    all_passed = all(item["passed"] for item in checks.values())
    correctness = {
        "title": spec["title"],
        "smoke": smoke,
        "all_gates_passed": all_passed,
        "gates": checks,
        "caveat": (
            "Trajectory sampling is a deterministic sanity check and does not "
            "prove soundness; analytic containment and native validation are "
            "separate gates."
        ),
    }

    write_csv(output / "raw_results.csv", raw_rows, RAW_FIELDS)
    write_csv(output / "run_summary.csv", runs, RUN_FIELDS)
    _write_table(
        output / "one_step_summary.csv",
        _one_step_summary(spec, runs, rows_by_run),
    )
    _write_table(
        output / "common_time_summary.csv",
        _common_time_summary(spec, runs, rows_by_run),
    )
    _write_table(
        output / "failure_horizon_summary.csv",
        _failure_summary(spec, runs, rows_by_run),
    )
    _write_table(
        output / "runtime_summary.csv", _runtime_summary(runs)
    )
    _write_table(
        output / "semantics_summary.csv", _semantics_summary(runs)
    )
    write_json(output / "correctness_checks.json", correctness)

    environment = {
        "title": spec["title"],
        "platform": platform.platform(),
        "collector_python": sys.version,
        "tool_paths": spec["tool_paths"],
        "git_shas": {
            name: git_sha(path)
            for name, path in spec["tool_paths"].items()
        },
        "tracked_worktrees_clean": {
            "diffreach": _git_tracked_clean(spec["tool_paths"]["diffreach"]),
            "flowstar": _git_tracked_clean(spec["tool_paths"]["flowstar"]),
        },
        "adapter_environments": {
            "torch": read_json(output / "torch_point_evaluations.json"),
            "diffreach": read_json(
                output / "diffreach_point_evaluations.json"
            ),
            "flowstar": read_json(
                output / "flowstar_point_evaluations.json"
            ),
        },
        "diffreach_upstream_provenance": read_json(
            output / "diffreach_upstream_provenance.json"
        ),
    }
    write_json(output / "environment.json", environment)
    if strict and not all_passed:
        failed = [name for name, value in checks.items() if not value["passed"]]
        raise SystemExit("correctness gates failed: " + ", ".join(failed))
    return correctness


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", default=str(HERE / "benchmark_spec.yaml"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    result = collect(
        load_spec(args.spec),
        Path(args.output_dir).resolve(),
        smoke=args.smoke,
        strict=args.strict,
    )
    print(
        json.dumps(
            {
                name: value["violations"]
                for name, value in result["gates"].items()
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
