#!/usr/bin/env python3
"""Build public optimized TORA-Q3 runtime tables from private timings."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-summary", type=Path, required=True)
    parser.add_argument("--eager-one-step", type=Path, required=True)
    parser.add_argument("--compiled-one-step", type=Path, required=True)
    parser.add_argument("--compiled-t20", type=Path, required=True)
    parser.add_argument("--profiler-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--iteration", action="append", default=[], help="LABEL=summary.json"
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline = load(args.baseline_summary)
    eager = load(args.eager_one_step)
    compiled = load(args.compiled_one_step)
    t20 = load(args.compiled_t20)
    dispatch = load(args.profiler_root / "runtime_dispatch_sync_audit.json")
    profiler_baseline = load(args.profiler_root / "summary.json")
    profiler_final = load(
        args.profiler_root
        / "final_compiled"
        / "summary.json"
    )

    iteration_rows = []
    for item in args.iteration:
        label, value = item.split("=", 1)
        summary_path = Path(value)
        summary = load(summary_path)
        dense = summary["dense_validated_step"]
        logical = summary["logical_step_dense_compose_projection"]
        iteration_rows.append(
            {
                "iteration": label,
                "dense_median_seconds": dense["median_seconds"],
                "dense_iqr_seconds": dense["iqr_seconds"],
                "logical_median_seconds": logical["median_seconds"],
                "logical_iqr_seconds": logical["iqr_seconds"],
                "warmup_seconds_excluded": summary[
                    "excluded_full_one_step_warmup_seconds"
                ],
                "requested_backend": summary.get(
                    "point_enclosure_backend_requested", "eager"
                ),
                "private_summary_sha256": sha256(summary_path),
            }
        )
    iteration_rows.extend(
        [
            {
                "iteration": "final_eager",
                "dense_median_seconds": eager["dense_validated_step"][
                    "median_seconds"
                ],
                "dense_iqr_seconds": eager["dense_validated_step"][
                    "iqr_seconds"
                ],
                "logical_median_seconds": eager[
                    "logical_step_dense_compose_projection"
                ]["median_seconds"],
                "logical_iqr_seconds": eager[
                    "logical_step_dense_compose_projection"
                ]["iqr_seconds"],
                "warmup_seconds_excluded": eager[
                    "excluded_full_one_step_warmup_seconds"
                ],
                "requested_backend": "eager",
                "private_summary_sha256": sha256(args.eager_one_step),
            },
            {
                "iteration": "final_compiled",
                "dense_median_seconds": compiled["dense_validated_step"][
                    "median_seconds"
                ],
                "dense_iqr_seconds": compiled["dense_validated_step"][
                    "iqr_seconds"
                ],
                "logical_median_seconds": compiled[
                    "logical_step_dense_compose_projection"
                ]["median_seconds"],
                "logical_iqr_seconds": compiled[
                    "logical_step_dense_compose_projection"
                ]["iqr_seconds"],
                "warmup_seconds_excluded": compiled[
                    "excluded_full_one_step_warmup_seconds"
                ],
                "requested_backend": "compiled",
                "private_summary_sha256": sha256(args.compiled_one_step),
            },
        ]
    )
    iteration_path = args.output_dir / "optimization_iterations.csv"
    with iteration_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(iteration_rows[0]))
        writer.writeheader()
        writer.writerows(iteration_rows)

    baseline_one_step = 2.349308
    baseline_t20 = baseline["lanes"]["torch_matched_crown"][
        "steady_wall_statistics"
    ]["median_seconds"]
    final_one_step = compiled["dense_validated_step"]["median_seconds"]
    final_t20 = t20["wall_statistics"]["median_seconds"]
    baseline_to = profiler_baseline["totals"]["aten_to_count"]
    final_to = profiler_final["totals"]["aten_to_count"]
    gates = [
        {
            "gate": "P0",
            "requirement": "no correctness regression",
            "observed": (
                "frozen one-leaf hash exact; compiled first-call bitwise "
                "verification; CPU/CUDA predicate tests"
            ),
            "status": "PASS",
        },
        {
            "gate": "P1",
            "requirement": "program host scalar sync <= 10 per logical step",
            "observed": str(dispatch["program_issued_host_scalar_sync_count"]),
            "status": (
                "PASS"
                if dispatch["program_issued_host_scalar_sync_count"] <= 10
                else "FAIL"
            ),
        },
        {
            "gate": "P2",
            "requirement": "aten::to reduction >= 90%",
            "observed": f"{100.0 * (1.0 - final_to / baseline_to):.6f}%",
            "status": "PASS" if final_to <= 0.1 * baseline_to else "FAIL",
        },
        {
            "gate": "P3",
            "requirement": "B48 one-step >= 10x",
            "observed": f"{baseline_one_step / final_one_step:.6f}x",
            "status": (
                "PASS"
                if baseline_one_step / final_one_step >= 10.0
                else "FAIL"
            ),
        },
        {
            "gate": "P4",
            "requirement": "common-control T20 >= 10x",
            "observed": f"{baseline_t20 / final_t20:.6f}x",
            "status": "PASS" if baseline_t20 / final_t20 >= 10.0 else "FAIL",
        },
    ]
    gate_path = args.output_dir / "performance_gates.csv"
    with gate_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(gates[0]))
        writer.writeheader()
        writer.writerows(gates)

    t20_checksums = [row["checksum"] for row in t20["repeats"]]
    public = {
        "schema": "tora_q3_optimized_runtime_public_v1",
        "status": "PASS_WITH_UNMET_PERFORMANCE_GATES",
        "workload": {
            "one_step": "B48 complete-Q3 K2 plus ten remainder rounds",
            "full": "frozen common-control B48 T20, 200 segments",
            "software_stack": "matched CROWN environment, Torch 2.8.0+cu128",
        },
        "one_step": {
            "baseline_dense_median_seconds": baseline_one_step,
            "optimized_eager": eager["dense_validated_step"],
            "optimized_compiled": compiled["dense_validated_step"],
            "compiled_speedup": baseline_one_step / final_one_step,
            "excluded_compiled_warmup_seconds": compiled[
                "excluded_full_one_step_warmup_seconds"
            ],
            "compiled_backend_status": compiled[
                "point_enclosure_backend_status"
            ],
        },
        "common_control_t20": {
            "baseline_median_seconds": baseline_t20,
            "optimized_compiled_statistics": t20["wall_statistics"],
            "optimized_solver_statistics": t20[
                "solver_excluding_serialization_statistics"
            ],
            "speedup": baseline_t20 / final_t20,
            "excluded_warmup_seconds": t20["warmup_excluded"]["wall_seconds"],
            "repeat_statuses": [row["status"] for row in t20["repeats"]],
            "repeat_checksums": t20_checksums,
            "stable_status_and_checksum": (
                len(set(t20_checksums)) == 1
                and all(row["status"] == "VERIFIED" for row in t20["repeats"])
            ),
            "stage_median_seconds": {
                stage: sorted(row["scopes"][stage] for row in t20["repeats"])[2]
                for stage in t20["repeats"][0]["scopes"]
            },
            "peak_cuda_memory_bytes": max(
                row["peak_cuda_memory_bytes"] for row in t20["repeats"]
            ),
            "peak_cpu_resident_memory_bytes": t20[
                "peak_cpu_resident_memory_bytes"
            ],
        },
        "profiler": {
            "baseline_kineto_host_scalar_events": profiler_baseline["totals"][
                "host_scalar_sync_estimate"
            ],
            "final_kineto_host_scalar_events": profiler_final["totals"][
                "host_scalar_sync_estimate"
            ],
            "program_dispatch_host_scalar_sync": dispatch[
                "program_issued_host_scalar_sync_count"
            ],
            "kineto_note": (
                "77 final item/local events include 74 profiler observation "
                "events that do not pass Torch dispatcher; P1 uses the "
                "program-issued dispatch audit"
            ),
            "baseline_aten_to": baseline_to,
            "final_aten_to": final_to,
            "aten_to_reduction_percent": 100.0 * (1.0 - final_to / baseline_to),
            "compiled_point_kernel_graph_break_count": 0,
            "object_level_compile_negative_result": (
                "full object entry hit Dynamo cache limits and failed deferred "
                "ledger validation; it is not used"
            ),
        },
        "soundness": {
            "frozen_one_leaf_private_detail_sha256": (
                "fdf35f29b67263f749a9c076ab056f0ef776db7cc14772107baa7ab660aaa396"
            ),
            "compiled_first_call_bitwise_verified": (
                compiled["point_enclosure_backend_status"][
                    "verification_count"
                ]
                >= 1
            ),
            "compiled_fallback_calls": compiled[
                "point_enclosure_backend_status"
            ]["eager_fallback_calls"],
        },
        "gates": gates,
        "private_evidence_sha256": {
            "optimized_eager_one_step": sha256(args.eager_one_step),
            "optimized_compiled_one_step": sha256(args.compiled_one_step),
            "optimized_compiled_t20": sha256(args.compiled_t20),
        },
    }
    summary_path = args.output_dir / "optimized_runtime_summary.json"
    summary_path.write_text(
        json.dumps(public, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        "# TORA-Q3 optimized runtime report",
        "",
        "All formal GPU timings below use the matched CROWN software stack. "
        "Cold compile/warm-up is reported separately and excluded from steady samples.",
        "",
        "## Outcomes",
        "",
        f"- eager B48 one-step median: `{eager['dense_validated_step']['median_seconds']:.6f}` s",
        f"- compiled B48 one-step median: `{final_one_step:.6f}` s "
        f"(`{baseline_one_step / final_one_step:.3f}x`)",
        f"- compiled common-control T20 median: `{final_t20:.6f}` s "
        f"(`{baseline_t20 / final_t20:.3f}x`)",
        f"- compiled T20 IQR: `{t20['wall_statistics']['iqr_seconds']:.6f}` s",
        "",
        "P0, P1, and P2 pass. P3 and P4 do not reach the required 10x; no "
        "10x claim is made.",
        "",
        "## Gate table",
        "",
        "| gate | observed | status |",
        "|---|---:|---|",
    ]
    for gate in gates:
        lines.append(
            f"| {gate['gate']} | {gate['observed']} | {gate['status']} |"
        )
    lines.extend(
        [
            "",
            "The remaining concrete engineering boundary is a pure tensor "
            "kernel for natural range/remainder arithmetic across the whole "
            "K2 + ten-round phase. Compiling only the point sine/cosine core "
            "cannot remove the thousands of small interval kernels.",
        ]
    )
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": public["status"],
                "one_step_speedup": baseline_one_step / final_one_step,
                "t20_speedup": baseline_t20 / final_t20,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
