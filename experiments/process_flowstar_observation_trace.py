#!/usr/bin/env python3
"""Validate, split, and equivalence-check stock Flow* observation trace output."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
from pathlib import Path
from typing import Any, Iterable, Sequence


SOURCE_COMMIT = "b85a3211748cb77b736fe4ad42ee02d8d2b81148"
REQUIRED_COMMON = {
    "tool",
    "source_commit",
    "run_id",
    "accepted_step_index",
    "attempt_index",
    "retry_index",
    "t_pre",
    "h_attempt",
    "accepted",
    "rejection_reason",
    "state_component",
    "stage",
}
STAGES = {
    "step_pre_state",
    "raw_picard_image",
    "truncation_cutoff",
    "insertion_input",
    "insertion_output",
    "right_map_input",
    "right_map_output",
    "normalized_reset_input",
    "normalized_reset_output",
    "next_step_pre_state",
    "acceptance_predicate",
}
TIME_RE = re.compile(r"time = ([0-9.]+),\s*step = ([0-9.]+),\s*order = ([0-9]+)")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _num(value: float) -> dict[str, str]:
    value = float(value)
    return {"decimal": format(value, ".17g"), "hex": value.hex()}


def _value(encoded: dict[str, str]) -> float:
    decimal = float(encoded["decimal"])
    hexadecimal = float.fromhex(encoded["hex"])
    if decimal != hexadecimal:
        raise ValueError(f"decimal/hex do not round-trip identically: {encoded}")
    return decimal


def _jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def _schedule(stdout: Path) -> list[dict[str, Any]]:
    rows = []
    for index, match in enumerate(TIME_RE.finditer(stdout.read_text(encoding="utf-8", errors="replace"))):
        rows.append(
            {
                "accepted_step_index": index,
                "t_printed": match.group(1),
                "h_printed": match.group(2),
                "order_printed": match.group(3),
            }
        )
    return rows


def _derived_attempts(accepted_h: list[float], *, run_id: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    t_pre = 0.0
    attempt_index = 0
    previous_h: float | None = None
    for step, final_h in enumerate(accepted_h):
        proposed = 0.1 if previous_h is None else min(previous_h * 1.1, 0.1)
        candidates = [proposed]
        while not math.isclose(candidates[-1], final_h, rel_tol=2e-15, abs_tol=2e-15):
            next_h = candidates[-1] * 0.5
            if next_h < final_h - 2e-15:
                raise ValueError(f"accepted h is not on stock half-step schedule at step {step}: {final_h}")
            candidates.append(next_h)
            if len(candidates) > 16:
                raise ValueError("unbounded derived retry chain")
        for retry, candidate in enumerate(candidates):
            accepted = retry == len(candidates) - 1
            rows.append(
                {
                    "tool": "flowstar",
                    "source_commit": SOURCE_COMMIT,
                    "run_id": run_id,
                    "accepted_step_index": step,
                    "attempt_index": attempt_index,
                    "retry_index": retry,
                    "t_pre": _num(t_pre),
                    "h_attempt": _num(candidate),
                    "accepted": accepted,
                    "rejection_reason": "" if accepted else "derived from exact stock half-step scheduler and final accepted h",
                    "state_component": -1,
                    "stage": "scheduler",
                    "derivation": "reconstructed_from_previous_accepted_h_times_1.1_and_repeated_half_step",
                }
            )
            attempt_index += 1
        t_pre += final_h
        previous_h = final_h
    return rows


def run(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    transitions: list[dict[str, Any]] = []
    terms: list[dict[str, Any]] = []
    remainders: list[dict[str, Any]] = []
    accepted_h_by_step: dict[int, float] = {}
    seen_terms: set[tuple[Any, ...]] = set()
    seen_stages: set[str] = set()
    observed_run_ids: set[str] = set()
    raw_records = 0
    with args.raw_trace.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"empty trace record at line {line_number}")
            row = json.loads(line)
            raw_records += 1
            observed_run_ids.add(str(row.get("run_id", "")))
            missing = REQUIRED_COMMON - row.keys()
            if missing:
                raise ValueError(f"line {line_number} missing required fields: {sorted(missing)}")
            _value(row["t_pre"])
            h = _value(row["h_attempt"])
            record_type = row.get("record_type")
            if record_type == "transition":
                seen_stages.add(row["stage"])
                for key in ("polynomial_range", "remainder", "self_map_candidate_box", "self_map_image"):
                    interval = row.get(key)
                    if interval is not None and _value(interval["lower"]) > _value(interval["upper"]):
                        raise ValueError(f"invalid interval at line {line_number}: {key}")
                transitions.append(row)
            elif record_type == "polynomial_term":
                exponent = row.get("exponent_tuple")
                if not isinstance(exponent, list) or any(not isinstance(value, int) or value < 0 for value in exponent):
                    raise ValueError(f"noncanonical exponent at line {line_number}")
                if sum(exponent) != row.get("degree"):
                    raise ValueError(f"degree/exponent mismatch at line {line_number}")
                key = (
                    row["accepted_step_index"],
                    row["attempt_index"],
                    row["stage"],
                    row["state_component"],
                    tuple(exponent),
                )
                if key in seen_terms:
                    raise ValueError(f"duplicate coefficient/support record at line {line_number}: {key}")
                seen_terms.add(key)
                terms.append(row)
            elif record_type == "remainder":
                interval = row["interval"]
                if _value(interval["lower"]) > _value(interval["upper"]):
                    raise ValueError(f"invalid remainder interval at line {line_number}")
                remainders.append(row)
            elif record_type == "acceptance_attempt":
                if row["accepted"]:
                    accepted_h_by_step[int(row["accepted_step_index"])] = h
            else:
                raise ValueError(f"unknown record type at line {line_number}: {record_type}")

    missing_stages = STAGES - seen_stages
    if missing_stages:
        raise ValueError(f"trace is missing lifecycle stages: {sorted(missing_stages)}")
    if len(observed_run_ids) != 1 or "" in observed_run_ids:
        raise ValueError(f"trace must contain exactly one nonempty run_id: {sorted(observed_run_ids)}")
    accepted_h = [accepted_h_by_step[index] for index in range(len(accepted_h_by_step))]
    attempts = _derived_attempts(accepted_h, run_id=next(iter(observed_run_ids)))

    stock_schedule = _schedule(args.stock_stdout)
    instrumented_schedule = _schedule(args.instrumented_stdout)
    plot_pairs = [(args.stock_plot_x, args.instrumented_plot_x), (args.stock_plot_y, args.instrumented_plot_y)]
    equivalence = {
        "schema": "flowstar_observation_equivalence_v1",
        "stock_commit": SOURCE_COMMIT,
        "instrumented_label": "stock Flow* + observation-only instrumentation",
        "stock_exit_code": int(args.stock_exit.read_text().strip()),
        "instrumented_exit_code": int(args.instrumented_exit.read_text().strip()),
        "schedule_equal_as_printed": stock_schedule == instrumented_schedule,
        "stock_segment_count": len(stock_schedule),
        "instrumented_segment_count": len(instrumented_schedule),
        "completion_equal": bool(stock_schedule and instrumented_schedule and stock_schedule[-1] == instrumented_schedule[-1]),
        "plot_files": [
            {
                "stock": str(stock),
                "instrumented": str(instrumented),
                "stock_sha256": _sha256(stock),
                "instrumented_sha256": _sha256(instrumented),
                "byte_equal": stock.read_bytes() == instrumented.read_bytes(),
            }
            for stock, instrumented in plot_pairs
        ],
    }
    equivalence["passed"] = bool(
        equivalence["stock_exit_code"] == 0
        and equivalence["instrumented_exit_code"] == 0
        and equivalence["schedule_equal_as_printed"]
        and equivalence["completion_equal"]
        and all(item["byte_equal"] for item in equivalence["plot_files"])
    )
    if not equivalence["passed"]:
        raise ValueError("stock/instrumented observational-equivalence gate failed")

    _jsonl(output_dir / "transitions.jsonl", transitions)
    _jsonl(output_dir / "polynomial_terms.jsonl", terms)
    _jsonl(output_dir / "remainders.jsonl", remainders)
    unavailable = [
        {
            **{key: attempts[0][key] for key in REQUIRED_COMMON},
            "record_type": "discarded_term",
            "availability": False,
            "discarded_term": None,
            "reason": "stock scheduler observation point does not export per-term cutoff/truncation objects; diagnostic probe is separate and not called stock",
        }
    ]
    _jsonl(output_dir / "discarded_terms.jsonl", unavailable)
    with (output_dir / "acceptance_attempts.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "tool", "source_commit", "run_id", "accepted_step_index", "attempt_index", "retry_index",
            "t_pre_decimal", "t_pre_hex", "h_attempt_decimal", "h_attempt_hex", "accepted",
            "rejection_reason", "state_component", "stage", "derivation",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in attempts:
            writer.writerow(
                {
                    **{key: row[key] for key in fieldnames if key in row},
                    "t_pre_decimal": row["t_pre"]["decimal"],
                    "t_pre_hex": row["t_pre"]["hex"],
                    "h_attempt_decimal": row["h_attempt"]["decimal"],
                    "h_attempt_hex": row["h_attempt"]["hex"],
                }
            )

    schema = {
        "schema": "vdp_transition_trace_schema_v1",
        "required_common_fields": sorted(REQUIRED_COMMON),
        "required_stages": sorted(STAGES),
        "number_encoding": {"decimal": "binary64 max_digits10 string", "hex": "C/Python hexfloat string"},
        "null_policy": "unavailable non-equivalent fields are null and explained; substitution is forbidden",
    }
    metadata = {
        "schema": "vdp_flowstar_observation_run_v1",
        "tool": "stock Flow* + observation-only instrumentation",
        "source_commit": SOURCE_COMMIT,
        "raw_record_count": raw_records,
        "transition_count": len(transitions),
        "polynomial_term_count": len(terms),
        "remainder_count": len(remainders),
        "accepted_segment_count": len(accepted_h),
        "derived_attempt_count": len(attempts),
        "observational_equivalence": equivalence,
        "instrumentation_limit": "inner rejected Picard attempts and partitioned dropped terms are not exported by the scheduler hook; retry rows are explicitly marked derived and the separate probe lane is non-stock",
    }
    (output_dir / "trace_schema.json").write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for source, name in [
        (args.stock_plot_x, "stock_vanderpol_t_x.plt"),
        (args.stock_plot_y, "stock_vanderpol_t_y.plt"),
        (args.instrumented_plot_x, "instrumented_vanderpol_t_x.plt"),
        (args.instrumented_plot_y, "instrumented_vanderpol_t_y.plt"),
    ]:
        shutil.copyfile(source, output_dir / name)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "raw_trace", "stock_stdout", "instrumented_stdout", "stock_exit", "instrumented_exit",
        "stock_plot_x", "stock_plot_y", "instrumented_plot_x", "instrumented_plot_y", "output_dir",
    ):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
