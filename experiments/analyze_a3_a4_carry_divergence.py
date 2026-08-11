#!/usr/bin/env python3
"""Locate the complete A3/CDR versus A4/CNI carry divergence ledger."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

try:
    from .diffreach_torch_full_horizon_common import write_json
except ImportError:
    from diffreach_torch_full_horizon_common import write_json


SCHEMA = "torch_r35_a3_a4_divergence_ledger_v1"


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a3-b1", type=Path, required=True)
    parser.add_argument("--a3-b64", type=Path, required=True)
    parser.add_argument("--a4-b1", type=Path, required=True)
    parser.add_argument("--a4-b64", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _trace(path: Path) -> list[dict[str, Any]]:
    with (path / "state_trace.jsonl").open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _metrics(path: Path) -> list[dict[str, str]]:
    with (path / "metrics.csv").open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _field_first(a3: list[dict[str, Any]], a4: list[dict[str, Any]], field: str) -> int | None:
    for index, (left, right) in enumerate(zip(a3, a4), start=1):
        left_field = left["fields"].get(field)
        right_field = right["fields"].get(field)
        if left_field is None or right_field is None:
            continue
        if left_field["sha256"] != right_field["sha256"]:
            return index
    return None


def _first_of(a3: list[dict[str, Any]], a4: list[dict[str, Any]], fields: tuple[str, ...]) -> dict[str, Any] | None:
    values = [(field, _field_first(a3, a4, field)) for field in fields]
    values = [(field, step) for field, step in values if step is not None]
    if not values:
        return None
    step = min(value for _, value in values)
    return {"step": step, "fields": [field for field, value in values if value == step]}


def _one_batch(a3_dir: Path, a4_dir: Path, batch: int) -> dict[str, Any]:
    a3 = _trace(a3_dir)
    a4 = _trace(a4_dir)
    a3_metrics = _metrics(a3_dir)
    a4_metrics = _metrics(a4_dir)
    coefficient_fields = tuple(
        field
        for field in a3[0]["fields"]
        if field.endswith("polynomial") or field.startswith("polynomial_picard_")
    )
    remainder_fields = tuple(
        field
        for field in a3[0]["fields"]
        if "remainder_lo" in field or "remainder_hi" in field
    )
    margin_relation = []
    for left, right in zip(a3_metrics, a4_metrics):
        delta = float(right["minimum_target_margin"]) - float(left["minimum_target_margin"])
        margin_relation.append(0 if delta == 0.0 else (1 if delta > 0.0 else -1))
    baseline_relation = next((value for value in margin_relation if value), 0)
    ordering_change = next(
        (
            index
            for index, value in enumerate(margin_relation, start=1)
            if value and baseline_relation and value != baseline_relation
        ),
        None,
    )
    a4_failure_step = _json(a4_dir / "summary.json")["first_failure"]
    failure_number = None if a4_failure_step is None else int(a4_failure_step["step"])
    last_ten = []
    if failure_number is not None:
        start = max(1, failure_number - 10)
        for step in range(start, failure_number + 1):
            left = a3_metrics[step - 1]
            right = a4_metrics[step - 1]
            last_ten.append(
                {
                    "step": step,
                    "time": float(right["time"]),
                    "a3_margin": float(left["minimum_target_margin"]),
                    "a4_margin": float(right["minimum_target_margin"]),
                    "a3_scale_max": float(left["scale_max"]),
                    "a4_scale_max": float(right["scale_max"]),
                    "a3_endpoint_width_max": float(left["endpoint_width_max"]),
                    "a4_endpoint_width_max": float(right["endpoint_width_max"]),
                    "a3_composition_ledger_width_max": float(left["composition_ledger_width_max"]),
                    "a4_composition_ledger_width_max": float(right["composition_ledger_width_max"]),
                    "a4_decision": right["decision"],
                }
            )
    return {
        "batch": batch,
        "common_attempted_steps": min(len(a3), len(a4)),
        "first_coefficient_bit_divergence": _first_of(a3, a4, coefficient_fields),
        "first_remainder_divergence": _first_of(a3, a4, remainder_fields),
        "first_physical_endpoint_divergence": _first_of(a3, a4, ("endpoint_lo", "endpoint_hi")),
        "first_tube_divergence": _first_of(a3, a4, ("tube_lo", "tube_hi")),
        "first_scale_divergence": _first_of(a3, a4, ("scale", "inverse_scale")),
        "first_carry_after_step_divergence": _first_of(
            a3,
            a4,
            (
                "normalized_parameterization_polynomial",
                "normalized_parameterization_remainder_lo",
                "normalized_parameterization_remainder_hi",
            ),
        ),
        "first_margin_ordering_change_step": ordering_change,
        "a4_failure_step": failure_number,
        "a4_failure_time": None if failure_number is None else (failure_number - 1) * 0.01,
        "failure_window": last_ten,
    }


def main() -> int:
    args = _args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    summaries = {
        "A3_B1": _json(args.a3_b1 / "summary.json"),
        "A3_B64": _json(args.a3_b64 / "summary.json"),
        "A4_B1": _json(args.a4_b1 / "summary.json"),
        "A4_B64": _json(args.a4_b64 / "summary.json"),
    }
    if any(summary.get("reproduction_status") != "reproduced" for summary in summaries.values()):
        raise RuntimeError("BRIDGE_REPRODUCTION_STOP")
    expected = {
        "A3_B1": (1000, None), "A3_B64": (1000, None),
        "A4_B1": (319, 320), "A4_B64": (333, 334),
    }
    for name, (completed, failure) in expected.items():
        summary = summaries[name]
        actual_failure = summary.get("first_failure")
        actual_failure_step = None if actual_failure is None else actual_failure["step"]
        if summary["completed_steps"] != completed or actual_failure_step != failure:
            raise RuntimeError("BRIDGE_REPRODUCTION_STOP")
    report = {
        "schema": SCHEMA,
        "reproduction_status": "A3_A4_FROZEN_RESULTS_REPRODUCED",
        "contract": {
            "support": "R35", "picard": 4, "validator": "VRAW",
            "h": 0.01, "target": 0.01, "cutoff": None,
        },
        "cells": {
            name: {
                "completed_steps": value["completed_steps"],
                "validated_horizon": value["validated_horizon"],
                "failure_step": None
                if value["first_failure"] is None
                else value["first_failure"]["step"],
            }
            for name, value in summaries.items()
        },
        "divergence": [
            _one_batch(args.a3_b1, args.a4_b1, 1),
            _one_batch(args.a3_b64, args.a4_b64, 64),
        ],
        "qualification": (
            "ordinary-float64 empirical ablation; A3 completion is not evidence that CDR is sounder"
        ),
    }
    write_json(args.output_dir / "divergence_ledger.json", report)
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
