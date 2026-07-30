#!/usr/bin/env python3
"""Run the Flow* analytic, refinement-trace, and original-parity gates."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from common import (
    analytic_contained,
    load_spec,
    validate_record,
    write_csv,
    write_json,
)
from export_flowstar_segment import export_segment

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
REPAIR = HERE.parent / "three_way_comparison_repair"


def _load_module(name: str, path: Path):
    module_spec = importlib.util.spec_from_file_location(name, path)
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[name] = module
    module_spec.loader.exec_module(module)
    return module


def _repair_runner():
    previous = sys.modules.get("common")
    repair_common = _load_module("common", REPAIR / "common.py")
    try:
        runner = _load_module("_deep_study_repair_flowstar", REPAIR / "run_flowstar_audit.py")
    finally:
        if previous is None:
            sys.modules.pop("common", None)
        else:
            sys.modules["common"] = previous
    runner._repair_common = repair_common
    return runner


def run_original_parity_gate(spec: dict[str, Any], output: Path) -> dict[str, Any]:
    runner = _repair_runner()
    repair_spec = runner.load_spec(REPAIR / "benchmark_spec.yaml")
    repair_spec["repositories"].update(
        {
            "flowstar_original": spec["repositories"]["flowstar_original"],
            "flowstar_audit": spec["repositories"]["flowstar_audit"],
        }
    )
    _, summary = runner.run_original_parity(repair_spec, output)
    executable = (
        output
        / "logs"
        / "flowstar_original_parity"
        / "generated_identical"
        / "parity"
    )
    environment = os.environ.copy()
    environment["FLOWSTAR_AUDIT_CACHE_LEAF_TRUNCATION"] = "1"
    environment["FLOWSTAR_AUDIT_REVALIDATE_REFINEMENT"] = "0"
    corrected = subprocess.run(
        [str(executable)],
        cwd=executable.parent,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=float(spec["timeout_s"]),
    )
    corrected_log = executable.parent / "root_cause.stdout.txt"
    corrected_log.write_text(corrected.stdout, encoding="utf-8")
    schedule = runner._parse_schedule(corrected.stdout)
    original_log = (
        output
        / "logs"
        / "flowstar_original_parity"
        / "original"
        / "stdout.txt"
    )
    original_schedule = runner._parse_schedule(
        original_log.read_text(encoding="utf-8")
    )
    root_reached = bool(
        corrected.returncode == 0
        and schedule
        and abs(schedule[-1]["time"] - 10.0) <= 5e-7
    )
    root_schedule_agreement = (
        len(schedule) == len(original_schedule)
        and all(
            abs(left["time"] - right["time"]) <= 5e-7
            and abs(left["step_size"] - right["step_size"]) <= 5e-7
            and left["order"] == right["order"]
            for left, right in zip(schedule, original_schedule)
        )
    )
    return {
        **summary,
        "root_cause_variant_reached_horizon_10": root_reached,
        "root_cause_segments": len(schedule),
        "root_cause_schedule_agreement": root_schedule_agreement,
        "root_cause_schedule_note": (
            "The correctness patch changes accepted adaptive refinements, so its "
            "native adaptive schedule may differ. The unchanged generated stock "
            "harness is the schedule-preservation gate."
        ),
        "root_cause_log": str(corrected_log),
        "passed": bool(summary["passed"] and root_reached),
    }


def run_correctness(
    spec: dict[str, Any], output: Path, *, smoke: bool
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    variants = list(spec["flowstar"]["audit_variants"])
    orders = [2] if smoke else list(map(int, spec["flowstar"]["orders"]))
    for system_name in ("riccati", "harmonic"):
        hs = [0.01] if smoke else list(map(float, spec["one_step"][system_name]))
        for h in hs:
            for order in orders:
                for variant in variants:
                    tag = f"{variant}_{system_name}_h{h:g}_o{order}"
                    try:
                        record = export_segment(
                            spec,
                            system_name=system_name,
                            h=h,
                            order=order,
                            variant=variant,
                            work_dir=output / "logs" / "flowstar_correctness" / tag,
                        )
                        representation = validate_record(record)
                        analytic = analytic_contained(
                            system_name,
                            spec["systems"][system_name]["initial_box"],
                            h,
                            record["raw_endpoint_box"],
                            float(spec["containment_tolerance"]),
                        )
                        row = {
                            "tool": "flowstar",
                            "variant": variant,
                            "system": system_name,
                            "h": h,
                            "order": order,
                            "native_validation_passed": record[
                                "native_validation_passed"
                            ],
                            "analytic_reference_contained": analytic,
                            "endpoint_vs_tube_violations": representation[
                                "endpoint_vs_tube_violations"
                            ],
                            "export_round_trip_passed": representation["passed"],
                            "all_values_finite": all(
                                all(map(lambda value: abs(value) < float("inf"), box))
                                for box in record["raw_endpoint_box"]
                                + record["whole_tube_box"]
                            ),
                            "status": "success",
                            "failure_category": "",
                            "message": "",
                        }
                        traces.append(
                            {
                                "variant": variant,
                                "system": system_name,
                                "h": h,
                                "order": order,
                                "trace": record["validation_trace"],
                            }
                        )
                    except Exception as exc:
                        message = f"{type(exc).__name__}: {exc}"
                        rejected = (
                            "reason=first_picard_inclusion_failed" in message
                            or "reason=cached_proposal_non_subset" in message
                        )
                        row = {
                            "tool": "flowstar",
                            "variant": variant,
                            "system": system_name,
                            "h": h,
                            "order": order,
                            "native_validation_passed": False,
                            "analytic_reference_contained": False,
                            "endpoint_vs_tube_violations": "",
                            "export_round_trip_passed": False,
                            "all_values_finite": False,
                            "status": (
                                "configuration_rejected"
                                if rejected
                                else "failed"
                            ),
                            "failure_category": (
                                "native_configuration_rejected"
                                if rejected
                                else "wrapper_or_validation_failure"
                            ),
                            "message": message,
                        }
                    rows.append(row)
                    print(
                        f"Flow* correctness {variant} {system_name} "
                        f"h={h:g} o={order}: {row['status']} "
                        f"analytic={row['analytic_reference_contained']}",
                        flush=True,
                    )
    primary_variants = {
        "flowstar_full_picard_revalidated",
        "flowstar_root_cause_patch",
    }
    primary = [row for row in rows if row["variant"] in primary_variants]
    stock = [row for row in rows if row["variant"] == "flowstar_stock"]
    primary_successes = [
        row for row in primary if row["status"] == "success"
    ]
    required_reference = [
        row
        for row in primary
        if row["system"] == "riccati"
        and math.isclose(float(row["h"]), 0.01)
        and int(row["order"]) == 2
    ]
    counts = {
        "total_rows": len(rows),
        "primary_rows": len(primary),
        "primary_native_configuration_rejections": sum(
            row["status"] == "configuration_rejected" for row in primary
        ),
        "primary_unexpected_failures": sum(
            row["status"] == "failed" for row in primary
        ),
        "required_reference_failures": sum(
            row["status"] != "success"
            or row["analytic_reference_contained"] is not True
            or not bool(row["native_validation_passed"])
            or not bool(row["export_round_trip_passed"])
            for row in required_reference
        ),
        "primary_analytic_violations": sum(
            row["analytic_reference_contained"] is not True
            for row in primary_successes
        ),
        "primary_endpoint_tube_violations": sum(
            int(row["endpoint_vs_tube_violations"] or 0)
            for row in primary_successes
        ),
        "primary_export_failures": sum(
            not bool(row["export_round_trip_passed"]) for row in primary
            if row["status"] == "success"
        ),
        "stock_analytic_violations": sum(
            row["analytic_reference_contained"] is not True for row in stock
        ),
    }
    counts["passed"] = not any(
        counts[key]
        for key in (
            "primary_unexpected_failures",
            "required_reference_failures",
            "primary_analytic_violations",
            "primary_endpoint_tube_violations",
            "primary_export_failures",
        )
    )
    return rows, traces, counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--spec", default=str(REPO_ROOT / "benchmarks" / "canonical.yaml")
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-parity", action="store_true")
    args = parser.parse_args()
    spec = load_spec(args.spec)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows, traces, counts = run_correctness(spec, output, smoke=args.smoke)
    parity = (
        {"skipped": True, "passed": True}
        if args.skip_parity
        else run_original_parity_gate(spec, output)
    )
    if args.skip_parity:
        adaptive_trajectory = {"skipped": True, "passed": True}
    else:
        from flowstar_adaptive_trajectory_audit import run_adaptive_audit

        adaptive_trace = run_adaptive_audit(
            spec,
            output / "flowstar_adaptive_trajectory_audit",
            parity_summary=parity,
            parity_output=output,
        )
        adaptive_trajectory = {
            "passed": adaptive_trace["passed"],
            "classification": adaptive_trace["classification"],
            "first_failure": adaptive_trace["first_failure"],
            "repair": adaptive_trace["repair"],
        }
    summary = {
        "analytic_counts": counts,
        "original_parity": parity,
        "adaptive_trajectory": adaptive_trajectory,
        "passed": bool(
            counts["passed"]
            and parity["passed"]
            and adaptive_trajectory["passed"]
        ),
    }
    write_csv(output / "flowstar_correctness.csv", rows)
    write_json(output / "flowstar_correctness_traces.json", traces)
    write_json(output / "flowstar_correctness_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
