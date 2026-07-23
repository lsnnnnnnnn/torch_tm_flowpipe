#!/usr/bin/env python3
"""Merge adapter output, enforce sanity checks, and compute long-form summaries."""
from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from common import (
    RAW_FIELDS,
    STATUS_VALUES,
    exact_endpoint,
    finite_number,
    load_spec,
    output_dir_from_args,
    read_csv,
    read_json,
    write_csv,
    write_json,
)

SUMMARY_FIELDS = [
    "run_id", "tool", "protocol", "system", "h", "horizon", "state_index",
    "status", "validation_status", "final_lower", "final_upper",
    "final_endpoint_width", "maximum_tube_width", "sum_final_widths",
    "maximum_final_width", "box_volume", "log_box_volume", "exact_width",
    "exact_inflation_ratio", "first_failure_time", "successful_horizon",
    "build_time_s", "warmup_time_s", "steady_runtime_median_s",
    "steady_runtime_iqr_s", "retained_basis", "effective_max_degree",
    "truncate_to_affine", "nonzero_Lt", "dtype", "device", "git_commit",
    "environment", "message",
]


def _float(value: Any, default: float = math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _group(rows: Iterable[Mapping[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(dict(row))
    return grouped


def _reference_map(rows: Iterable[Mapping[str, Any]]) -> dict[tuple[Any, ...], tuple[float, float]]:
    result: dict[tuple[Any, ...], tuple[float, float]] = {}
    for row in rows:
        key = (
            row["system"],
            round(_float(row["h"]), 14),
            round(_float(row["horizon"]), 14),
            int(row["step_index"]),
            int(row["state_index"]),
        )
        result[key] = (_float(row["lower"]), _float(row["upper"]))
    return result


def _mark_violation(
    rows: list[dict[str, Any]],
    validation_status: str,
    message: str,
    *,
    failure_time: float,
) -> None:
    for row in rows:
        row["status"] = "sample_violation"
        row["validation_status"] = validation_status
        row["first_failure_time"] = failure_time
        row["successful_horizon"] = max(0.0, failure_time - _float(row["h"]))
        row["message"] = message


def _exact_checks(
    grouped: dict[str, list[dict[str, Any]]],
    references: Mapping[tuple[Any, ...], tuple[float, float]],
    tolerance: float,
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for run_id, rows in grouped.items():
        if not rows or rows[0]["status"] != "certified_ok":
            continue
        violations = 0
        violation_times: list[float] = []
        checked = 0
        for row in rows:
            if row["interval_kind"] != "endpoint" or not finite_number(row["lower"]):
                continue
            key = (
                row["system"],
                round(_float(row["h"]), 14),
                round(_float(row["horizon"]), 14),
                int(row["step_index"]),
                int(row["state_index"]),
            )
            if key not in references:
                continue
            checked += 1
            exact_lo, exact_hi = references[key]
            if _float(row["lower"]) > exact_lo + tolerance or _float(row["upper"]) < exact_hi - tolerance:
                violations += 1
                violation_times.append(_float(row["time"]))
        if violations:
            _mark_violation(
                rows,
                "exact_reference_violation",
                f"{violations}/{checked} exact endpoint hull checks failed",
                failure_time=min(violation_times),
            )
        checks.append(
            {
                "run_id": run_id,
                "check": "exact_endpoint_containment",
                "checked": checked,
                "violations": violations,
                "tolerance": tolerance,
            }
        )
    return checks


def _trajectory_checks(
    grouped: dict[str, list[dict[str, Any]]],
    trajectories: list[dict[str, str]],
    tolerance: float,
) -> list[dict[str, Any]]:
    by_config: dict[tuple[Any, ...], list[dict[str, str]]] = defaultdict(list)
    for row in trajectories:
        by_config[
            (
                row["system"],
                round(_float(row["h"]), 14),
                round(_float(row["horizon"]), 14),
            )
        ].append(row)
    checks: list[dict[str, Any]] = []
    for run_id, rows in grouped.items():
        if not rows or rows[0]["status"] != "certified_ok":
            continue
        h, horizon = _float(rows[0]["h"]), _float(rows[0]["horizon"])
        config_key = (rows[0]["system"], round(h, 14), round(horizon, 14))
        tube_map = {
            (int(row["step_index"]), int(row["state_index"])): (_float(row["lower"]), _float(row["upper"]))
            for row in rows
            if row["interval_kind"] == "tube" and finite_number(row["lower"])
        }
        checked = violations = 0
        violation_times: list[float] = []
        for sample in by_config.get(config_key, []):
            time_value = _float(sample["time"])
            if time_value <= 0.0:
                continue
            step_index = min(int(round(horizon / h)), max(1, int(math.ceil(time_value / h - 1.0e-12))))
            key = (step_index, int(sample["state_index"]))
            if key not in tube_map:
                continue
            checked += 1
            lower, upper = tube_map[key]
            value = _float(sample["value"])
            if value < lower - tolerance or value > upper + tolerance:
                violations += 1
                violation_times.append(time_value)
        if violations:
            _mark_violation(
                rows,
                "sample_violation",
                f"{violations}/{checked} simulated tube samples fell outside",
                failure_time=min(violation_times),
            )
        checks.append(
            {
                "run_id": run_id,
                "check": "sampled_trajectory_tube_containment",
                "checked": checked,
                "violations": violations,
                "tolerance": tolerance,
                "proof": False,
            }
        )
    return checks


def _summary(grouped: Mapping[str, list[dict[str, Any]]], spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for run_id, rows in grouped.items():
        if not rows:
            continue
        first = rows[0]
        state_count = len(spec["systems"][first["system"]]["state_names"])
        endpoint_by_state: dict[int, list[dict[str, Any]]] = defaultdict(list)
        tube_by_state: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            if row["interval_kind"] == "endpoint" and finite_number(row["lower"]):
                endpoint_by_state[int(row["state_index"])].append(row)
            elif row["interval_kind"] == "tube" and finite_number(row["lower"]):
                tube_by_state[int(row["state_index"])].append(row)
        finals: dict[int, dict[str, Any]] = {}
        for state_index, state_rows in endpoint_by_state.items():
            finals[state_index] = max(state_rows, key=lambda row: int(row["step_index"]))
        final_widths = [_float(finals[i]["width"]) for i in range(state_count) if i in finals]
        all_finite = len(final_widths) == state_count and all(math.isfinite(width) for width in final_widths)
        volume = math.prod(final_widths) if all_finite else math.nan
        log_volume = math.log(volume) if math.isfinite(volume) and volume > 0 else math.nan
        sum_widths = sum(final_widths) if all_finite else math.nan
        max_width = max(final_widths) if all_finite else math.nan
        exact = exact_endpoint(
            first["system"],
            _float(first["horizon"]),
            spec["systems"][first["system"]]["initial_box"],
        )
        for state_index in range(state_count):
            final = finals.get(state_index)
            final_width = _float(final["width"]) if final else math.nan
            exact_width = (
                exact[state_index][1] - exact[state_index][0]
                if exact is not None and state_index < len(exact)
                else math.nan
            )
            ratio = final_width / exact_width if exact_width > 0 and math.isfinite(final_width) else math.nan
            tube_width = max(
                (_float(row["width"]) for row in tube_by_state.get(state_index, [])),
                default=math.nan,
            )
            summary = {field: first.get(field, "") for field in SUMMARY_FIELDS}
            summary.update(
                run_id=run_id,
                state_index=state_index,
                final_lower=final["lower"] if final else "",
                final_upper=final["upper"] if final else "",
                final_endpoint_width=final_width if math.isfinite(final_width) else "",
                maximum_tube_width=tube_width if math.isfinite(tube_width) else "",
                sum_final_widths=sum_widths if math.isfinite(sum_widths) else "",
                maximum_final_width=max_width if math.isfinite(max_width) else "",
                box_volume=volume if math.isfinite(volume) else "",
                log_box_volume=log_volume if math.isfinite(log_volume) else "",
                exact_width=exact_width if math.isfinite(exact_width) else "",
                exact_inflation_ratio=ratio if math.isfinite(ratio) else "",
            )
            summaries.append(summary)
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", default=str(HERE / "benchmark_spec.yaml"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    spec = load_spec(args.spec)
    output_dir = output_dir_from_args(args.output_dir)
    suffix = "smoke" if args.smoke else "full"
    adapter_paths = [
        output_dir / f"{tool}_raw_{suffix}.csv"
        for tool in ("torch", "flowstar", "diffreach")
    ]
    missing = [str(path) for path in adapter_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing adapter output: {missing}")
    rows: list[dict[str, Any]] = []
    for path in adapter_paths:
        rows.extend(read_csv(path))
    for index, row in enumerate(rows):
        missing_fields = [field for field in RAW_FIELDS if field not in row]
        if missing_fields:
            raise ValueError(f"row {index} missing fields: {missing_fields}")
        if row["status"] not in STATUS_VALUES:
            raise ValueError(f"row {index} has unknown status {row['status']!r}")
    grouped = _group(rows, "run_id")
    references = _reference_map(read_csv(output_dir / "references.csv"))
    tolerance = float(spec["numerical_containment_tolerance"])
    checks = _exact_checks(grouped, references, tolerance)
    checks.extend(
        _trajectory_checks(grouped, read_csv(output_dir / "trajectories.csv"), tolerance)
    )
    rows = [row for run_rows in grouped.values() for row in run_rows]
    summaries = _summary(grouped, spec)
    write_csv(output_dir / "raw_results.csv", rows, RAW_FIELDS)
    write_csv(output_dir / "run_summary.csv", summaries, SUMMARY_FIELDS)
    write_json(
        output_dir / "correctness_checks.json",
        {
            "checks": checks,
            "all_exact_checks_passed": all(
                check["violations"] == 0
                for check in checks
                if check["check"] == "exact_endpoint_containment"
            ),
            "all_sample_checks_passed": all(
                check["violations"] == 0
                for check in checks
                if check["check"] == "sampled_trajectory_tube_containment"
            ),
            "sample_checks_are_formal_proof": False,
        },
    )
    for run_id, run_rows in grouped.items():
        metadata_path = output_dir / "per_run" / f"{run_id}.json"
        if metadata_path.exists():
            metadata = read_json(metadata_path)
            metadata["post_collection_status"] = run_rows[0]["status"]
            metadata["post_collection_validation_status"] = run_rows[0]["validation_status"]
            metadata["correctness_checks"] = [
                check for check in checks if check["run_id"] == run_id
            ]
            write_json(metadata_path, metadata)
    print(f"collected {len(rows)} rows and {len(summaries)} summary rows into {output_dir}")


if __name__ == "__main__":
    main()
