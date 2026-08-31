#!/usr/bin/env python3
"""Run clean-SHA C4 reference/optimized correctness and CPU performance gates."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]


WORKER = r'''
import json
import os
from pathlib import Path
import resource
import sys
import tempfile
import time

root = Path(sys.argv[1]).resolve()
workload = sys.argv[2]
steps = int(sys.argv[3])
repeats = int(sys.argv[4])
sys.path.insert(0, str(root / "src"))
sys.path.insert(0, str(root))

import torch
torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

from experiments.profile_c4_reference_solver import (
    _run_brusselator_steps,
    _snapshot,
    _vdp_initial,
)
from torch_tm_flowpipe import (
    C3_CROSS_STEP_SYMBOLIC_QUEUE,
    DENSE_OBSERVER_NONE,
    DenseRangePolicy,
    FLOWSTAR_RAW_REMAINDER_REFINED_MODE,
    flowpipe_step_flowstar_style_adaptive,
    save_terminal_checkpoint,
)

def run_vdp_prefix(count):
    # Bind the accepted C3 evidence policy explicitly.  The frozen numerical
    # reference predates the config wrapper and its old profile helper used a
    # natural range policy, which is not the authoritative C3 T10 contract.
    ode, current, state = _vdp_initial()
    policy = DenseRangePolicy(
        method="adaptive_subdivision",
        max_depth=1,
        max_leaves=4,
        split_vars=(0, 1),
        trigger="proactive_depth1_on_named_contexts",
        named_contexts=("polynomial_truncation",),
        variable_orders=((0, 1, 2), (1, 0, 2), (2, 0, 1)),
    )
    segment = None
    for _ in range(count):
        segment = flowpipe_step_flowstar_style_adaptive(
            ode,
            current,
            h=0.01,
            h_min=0.01,
            h_max=0.01,
            order=4,
            target_remainder_radius=1e-4,
            cutoff_threshold=1e-10,
            max_validation_attempts=2,
            validation_eps=1e-12,
            validation_mode=FLOWSTAR_RAW_REMAINDER_REFINED_MODE,
            reset_mode=C3_CROSS_STEP_SYMBOLIC_QUEUE,
            step_policy_mode="flowstar_compat",
            flowstar_normal_state=state,
            flowstar_symbolic_queue_max_size=100,
            right_map_center_mode="constant",
            right_map_range_mode="standard",
            tm_backend="dense",
            dense_device="cpu",
            dense_dtype=torch.float64,
            dense_range_policy=policy,
            dense_observer_mode=DENSE_OBSERVER_NONE,
        )
        if (
            segment.status != "validated"
            or segment.reset_tm is None
            or segment.flowstar_normal_state is None
        ):
            raise RuntimeError(f"VDP reference rejected fixed prefix: {segment.message}")
        current = segment.reset_tm
        state = segment.flowstar_normal_state
    return current, state, segment, count

def peak_rss_bytes():
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024

def range_hex(tmv):
    return [
        {"lo_hex": float(value.lo.detach().cpu()).hex(), "hi_hex": float(value.hi.detach().cpu()).hex()}
        for value in tmv.range_box()
    ]

results = []
for repeat in range(repeats):
    started = time.perf_counter()
    if workload == "brusselator":
        current, state, segment, accepted = _run_brusselator_steps(
            steps,
            DENSE_OBSERVER_NONE,
        )
    elif workload == "vdp_prefix":
        current, state, segment, accepted = run_vdp_prefix(steps)
    else:
        raise ValueError(f"unknown workload: {workload}")
    wall_s = time.perf_counter() - started
    solver_peak_rss_bytes = peak_rss_bytes()
    snapshot_started = time.perf_counter()
    snapshot = _snapshot(segment)
    snapshot["endpoint_range_hex"] = range_hex(segment.endpoint_raw_tm)
    snapshot["tube_range_hex"] = range_hex(segment.tm)
    snapshot["reset_range_hex"] = range_hex(segment.reset_tm)
    snapshot_s = time.perf_counter() - snapshot_started
    serialization_started = time.perf_counter()
    json.dumps(snapshot, sort_keys=True, separators=(",", ":"), allow_nan=False)
    serialization_s = time.perf_counter() - serialization_started
    with tempfile.TemporaryDirectory(prefix="c4_perf_checkpoint_") as temporary:
        checkpoint_started = time.perf_counter()
        manifest = save_terminal_checkpoint(
            Path(temporary) / "terminal",
            current=current,
            normal_state=state,
            scheduler={"workload": workload, "steps": steps},
            contract={"performance_gate": 1, "workload": workload, "steps": steps},
            provenance={"producer": "run_c4_performance_gate.py"},
        )
        checkpoint_export_s = time.perf_counter() - checkpoint_started
    results.append({
        "repeat": repeat,
        "wall_s": wall_s,
        "solver_peak_rss_bytes": solver_peak_rss_bytes,
        "accepted_steps": accepted,
        "rejected_steps": 0,
        "snapshot": snapshot,
        "snapshot_construction_s": snapshot_s,
        "serialization_s": serialization_s,
        "checkpoint_export_s": checkpoint_export_s,
        "checkpoint_sha256": manifest["full_checkpoint_sha256"],
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "torch_num_threads": torch.get_num_threads(),
    })
print(json.dumps({"workload": workload, "steps": steps, "results": results}, sort_keys=True, allow_nan=False))
'''


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _validate_scientific_root(root: Path, expected_sha: str, label: str) -> dict[str, Any]:
    actual = _git(root, "rev-parse", "HEAD")
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    if actual != expected_sha:
        raise ValueError(f"{label} scientific root SHA mismatch: {actual} != {expected_sha}")
    if status:
        raise ValueError(f"{label} scientific root is dirty")
    return {"label": label, "root": str(root), "sha": actual, "clean": True}


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    fields: Sequence[str] | None = None,
) -> None:
    selected = list(fields or sorted({str(key) for row in rows for key in row}))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=selected, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _child_environment(root: Path) -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        {
            "PYTHONPATH": os.pathsep.join((str(root / "src"), str(root))),
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    return env


def _measure(
    root: Path,
    workload: str,
    steps: int,
    repeats: int,
) -> dict[str, Any]:
    completed = subprocess.run(
        [sys.executable, "-c", WORKER, str(root), workload, str(steps), str(repeats)],
        cwd=root,
        env=_child_environment(root),
        check=True,
        capture_output=True,
        text=True,
    )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"{workload} worker returned no JSON")
    return json.loads(lines[-1])


def _iqr(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    quartiles = statistics.quantiles(values, n=4, method="inclusive")
    return float(quartiles[2] - quartiles[0])


def _result_rows(
    payload: Mapping[str, Any],
    *,
    variant: str,
    scientific_sha: str,
) -> list[dict[str, Any]]:
    results = list(payload["results"])
    walls = [float(result["wall_s"]) for result in results]
    median = statistics.median(walls)
    iqr = _iqr(walls)
    rows = []
    for result in results:
        rows.append(
            {
                "workload": payload["workload"],
                "variant": variant,
                "scientific_sha": scientific_sha,
                "steps": int(payload["steps"]),
                "repeat": int(result["repeat"]),
                "wall_s": float(result["wall_s"]),
                "median_wall_s": median,
                "iqr_wall_s": iqr,
                "accepted_steps": int(result["accepted_steps"]),
                "rejected_steps": int(result["rejected_steps"]),
                "solver_peak_rss_bytes": int(result["solver_peak_rss_bytes"]),
                "checkpoint_export_s": float(result["checkpoint_export_s"]),
                "snapshot_construction_s": float(result["snapshot_construction_s"]),
                "serialization_s": float(result["serialization_s"]),
                "checkpoint_sha256": result["checkpoint_sha256"],
                "snapshot_sha256": hashlib.sha256(
                    json.dumps(
                        result["snapshot"],
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    ).encode("utf-8")
                ).hexdigest(),
                "cpu_affinity": ";".join(str(value) for value in result["cpu_affinity"]),
                "observer_mode": "production_no_observer",
                "timer_scope": "solver_only_excludes_snapshot_serialization_checkpoint",
            }
        )
    return rows


def _median(rows: Sequence[Mapping[str, Any]], workload: str, variant: str, steps: int) -> float:
    values = [
        float(row["wall_s"])
        for row in rows
        if row["workload"] == workload
        and row["variant"] == variant
        and int(row["steps"]) == int(steps)
    ]
    if not values:
        raise ValueError(f"missing runtime rows for {workload}/{variant}/{steps}")
    return statistics.median(values)


def _max_rss(rows: Sequence[Mapping[str, Any]], variant: str) -> int:
    values = [
        int(row["solver_peak_rss_bytes"])
        for row in rows
        if row["variant"] == variant
    ]
    return max(values) if values else 0


def _run_vdp_case(root: Path, output_dir: Path, horizon: str, fixed: bool) -> dict[str, Any]:
    command = [
        sys.executable,
        str(root / "experiments" / "run_vdp_dense_backend.py"),
        "--output-dir",
        str(output_dir),
        "--tm-backend",
        "dense",
        "--device",
        "cpu",
        "--horizon",
        horizon,
        "--initialization-contract",
        "exact_decimal_contract",
        "--reset-mode",
        "normalized_insertion_dependency_preserving_c3_sr100",
        "--validation-mode",
        "flowstar_raw_remainder_compat_factorized_joint_closure_refined",
        "--trace-flush-every",
        "0",
        "--dense-range-method",
        "adaptive_subdivision",
        "--dense-range-trigger",
        "proactive_depth1_on_named_contexts",
        "--dense-range-max-depth",
        "1",
        "--dense-range-max-leaves",
        "4",
        "--dense-range-split-vars",
        "0,1",
        "--dense-range-contexts",
        "polynomial_truncation",
        "--wall-cap-s",
        "3600",
    ]
    if fixed:
        command.extend(("--fixed-step", "0.01"))
    subprocess.run(
        command,
        cwd=root,
        env=_child_environment(root),
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))


def _scientific_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "status",
        "accepted_steps",
        "rejected_attempts",
        "completed_horizon",
        "completed_requested_horizon",
        "requested_horizon",
        "requested_order",
        "target_remainder_radius",
        "cutoff",
        "reset_mode",
        "validation_mode",
        "raw_endpoint",
        "full_tube",
        "last_segment",
        "endpoint_repair_used",
        "endpoint_tightening_used",
    )
    return {field: summary.get(field) for field in fields}


def _csv_scientific_sha(
    path: Path,
    *,
    fields: Sequence[str] | None = None,
) -> tuple[str, tuple[str, ...]]:
    excluded = {
        "stage_runtime_s",
        "dense_kernel_s",
        "host_to_device_s",
        "device_to_host_s",
    }
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        selected = tuple(
            field
            for field in (fields or tuple(reader.fieldnames or ()))
            if field not in excluded
        )
        rows = [
            {key: row.get(key, "") for key in selected}
            for row in reader
        ]
    return (
        hashlib.sha256(
            json.dumps(rows, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
                "utf-8"
            )
        ).hexdigest(),
        selected,
    )


def _vdp_regression(root: Path) -> dict[str, Any]:
    baseline_root = (
        root
        / "artifacts"
        / "runs"
        / "vdp_c3_cross_step_causal_closure_20260827"
        / "raw"
    )
    with tempfile.TemporaryDirectory(prefix="c4_vdp_regression_") as temporary:
        run_root = Path(temporary)
        current_native = _run_vdp_case(root, run_root / "native", "10", False)
        native_baseline_dir = baseline_root / "native" / "torch_c3"
        baseline_native = json.loads(
            (native_baseline_dir / "summary.json").read_text(encoding="utf-8")
        )
        native_summary_equal = _scientific_summary(current_native) == _scientific_summary(
            baseline_native
        )
        baseline_native_segments_sha, baseline_fields = _csv_scientific_sha(
            native_baseline_dir / "segments.csv"
        )
        native_segments_sha, current_projected_fields = _csv_scientific_sha(
            run_root / "native" / "segments.csv",
            fields=baseline_fields,
        )
        fixed_results: dict[str, Any] = {}
        for label, horizon in (("T1", "1"), ("T3", "3"), ("T6p32", "6.32")):
            current = _run_vdp_case(root, run_root / label, horizon, True)
            baseline = json.loads(
                (baseline_root / "fixed" / "torch_c3" / label / "summary.json").read_text(
                    encoding="utf-8"
                )
            )
            fixed_results[label] = {
                "horizon": horizon,
                "current": _scientific_summary(current),
                "baseline": _scientific_summary(baseline),
                "exact": _scientific_summary(current) == _scientific_summary(baseline),
            }
    passed = (
        native_summary_equal
        and native_segments_sha == baseline_native_segments_sha
        and all(value["exact"] for value in fixed_results.values())
    )
    return {
        "schema": "torch_tm_flowpipe.c4_vdp_zero_regression/1",
        "status": "VDP_NATIVE_T10_ZERO_REGRESSION_PASSED" if passed else "VDP_REGRESSION_FAILED",
        "passed": passed,
        "native": {
            "current": _scientific_summary(current_native),
            "baseline": _scientific_summary(baseline_native),
            "summary_exact": native_summary_equal,
            "segments_scientific_sha256": native_segments_sha,
            "baseline_segments_scientific_sha256": baseline_native_segments_sha,
            "segments_exact": native_segments_sha == baseline_native_segments_sha,
            "historical_scientific_columns_compared": len(baseline_fields),
            "current_projection_columns": len(current_projected_fields),
            "current_extra_observer_columns_ignored": True,
        },
        "fixed_snapshots": fixed_results,
    }


def _historical_brusselator_final(root: Path) -> dict[str, Any]:
    path = (
        root
        / "artifacts"
        / "runs"
        / "brusselator_live_range_c5_20260828"
        / "raw"
        / "c4_baseline"
        / "segments.csv"
    )
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    final = rows[-1]
    return {
        "step": int(final["step"]),
        "t_after": float(final["t_after"]),
        "endpoint_range_hex": [
            {"lo_hex": final["endpoint_x_lo_hex"], "hi_hex": final["endpoint_x_hi_hex"]},
            {"lo_hex": final["endpoint_y_lo_hex"], "hi_hex": final["endpoint_y_hi_hex"]},
        ],
        "tube_range_hex": [
            {"lo_hex": final["tube_x_lo_hex"], "hi_hex": final["tube_x_hi_hex"]},
            {"lo_hex": final["tube_y_lo_hex"], "hi_hex": final["tube_y_hi_hex"]},
        ],
        "queue_sha256": final["queue_hash"],
        "queue_generation": int(final["queue_generation"]),
        "queue_reset_count": int(final["queue_reset_count"]),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    reference_root = args.reference_root.resolve()
    optimized_root = args.optimized_root.resolve()
    provenance = {
        "reference": _validate_scientific_root(
            reference_root, args.reference_sha, "reference"
        ),
        "optimized": _validate_scientific_root(
            optimized_root, args.optimized_sha, "optimized"
        ),
    }
    os.sched_setaffinity(0, {int(args.cpu)})
    affinity = sorted(os.sched_getaffinity(0))
    prefix_rows: list[dict[str, Any]] = []
    measurements: dict[str, Any] = {}
    cases = (
        ("brusselator", args.prefix100_steps, args.prefix100_repeats),
        ("brusselator", args.prefix300_steps, args.prefix300_repeats),
        ("vdp_prefix", args.vdp_prefix_steps, args.vdp_prefix_repeats),
    )
    for workload, steps, repeats in cases:
        for variant, root, sha in (
            ("reference", reference_root, args.reference_sha),
            ("optimized", optimized_root, args.optimized_sha),
        ):
            payload = _measure(root, workload, steps, repeats)
            measurements[f"{workload}:{steps}:{variant}"] = payload
            prefix_rows.extend(
                _result_rows(payload, variant=variant, scientific_sha=sha)
            )
    _write_csv(output_dir / "prefix_runtime_matrix.csv", prefix_rows)

    full_rows: list[dict[str, Any]] = []
    full_payloads: dict[str, Any] = {}
    if args.run_full:
        for variant, root, sha in (
            ("reference", reference_root, args.reference_sha),
            ("optimized", optimized_root, args.optimized_sha),
        ):
            payload = _measure(root, "brusselator", 1000, 1)
            full_payloads[variant] = payload
            full_rows.extend(_result_rows(payload, variant=variant, scientific_sha=sha))
    else:
        full_rows.append(
            {
                "workload": "brusselator",
                "variant": "not_run",
                "scientific_sha": "",
                "steps": 1000,
                "repeat": -1,
                "wall_s": "",
                "median_wall_s": "",
                "iqr_wall_s": "",
                "accepted_steps": "",
                "rejected_steps": "",
                "solver_peak_rss_bytes": "",
                "checkpoint_export_s": "",
                "snapshot_construction_s": "",
                "serialization_s": "",
                "checkpoint_sha256": "",
                "snapshot_sha256": "",
                "cpu_affinity": ";".join(str(value) for value in affinity),
                "observer_mode": "production_no_observer",
                "timer_scope": "not_run",
            }
        )
    _write_csv(output_dir / "full_runtime_matrix.csv", full_rows)

    speedup100 = _median(
        prefix_rows, "brusselator", "reference", args.prefix100_steps
    ) / _median(prefix_rows, "brusselator", "optimized", args.prefix100_steps)
    speedup300 = _median(
        prefix_rows, "brusselator", "reference", args.prefix300_steps
    ) / _median(prefix_rows, "brusselator", "optimized", args.prefix300_steps)
    prefix_gate = speedup100 >= 2.0 and speedup300 >= 2.0

    brusselator_regression: dict[str, Any]
    full_speed_gate = False
    memory_gate = False
    full_exact = False
    full_speedup: float | None = None
    if args.run_full:
        reference_result = full_payloads["reference"]["results"][0]
        optimized_result = full_payloads["optimized"]["results"][0]
        reference_wall = float(reference_result["wall_s"])
        optimized_wall = float(optimized_result["wall_s"])
        full_speedup = reference_wall / optimized_wall
        full_speed_gate = optimized_wall <= 0.5 * reference_wall
        reference_rss = int(reference_result["solver_peak_rss_bytes"])
        optimized_rss = int(optimized_result["solver_peak_rss_bytes"])
        memory_gate = optimized_rss <= 1.5 * reference_rss
        full_exact = (
            reference_result["snapshot"] == optimized_result["snapshot"]
            and reference_result["checkpoint_sha256"]
            == optimized_result["checkpoint_sha256"]
            and int(reference_result["accepted_steps"])
            == int(optimized_result["accepted_steps"])
            == 1000
        )
        historical = _historical_brusselator_final(optimized_root)
        historical_exact = (
            optimized_result["snapshot"]["endpoint_range_hex"]
            == historical["endpoint_range_hex"]
            and optimized_result["snapshot"]["tube_range_hex"]
            == historical["tube_range_hex"]
            and optimized_result["snapshot"]["queue_sha256"]
            == historical["queue_sha256"]
            and optimized_result["snapshot"]["queue_generation"]
            == historical["queue_generation"]
        )
        brusselator_regression = {
            "schema": "torch_tm_flowpipe.c4_brusselator_zero_regression/1",
            "status": (
                "BRUSSELATOR_T20_ZERO_REGRESSION_PASSED"
                if full_exact and historical_exact
                else "BRUSSELATOR_REGRESSION_FAILED"
            ),
            "passed": full_exact and historical_exact,
            "reference_vs_optimized_exact": full_exact,
            "historical_final_snapshot_exact": historical_exact,
            "accepted_steps": int(optimized_result["accepted_steps"]),
            "rejected_steps": int(optimized_result["rejected_steps"]),
            "completed_horizon": 20.0,
            "reference": reference_result,
            "optimized": optimized_result,
            "historical": historical,
        }
    else:
        brusselator_regression = {
            "schema": "torch_tm_flowpipe.c4_brusselator_zero_regression/1",
            "status": "NOT_RUN",
            "passed": False,
        }
    _write_json(output_dir / "BRUSSELATOR_REGRESSION.json", brusselator_regression)

    vdp_regression = (
        _vdp_regression(optimized_root)
        if args.run_vdp_regression
        else {
            "schema": "torch_tm_flowpipe.c4_vdp_zero_regression/1",
            "status": "NOT_RUN",
            "passed": False,
        }
    )
    _write_json(output_dir / "VDP_REGRESSION.json", vdp_regression)

    correctness_gate = bool(brusselator_regression["passed"] and vdp_regression["passed"])
    fully_measured = bool(args.run_full and args.run_vdp_regression)
    if fully_measured and correctness_gate and prefix_gate and full_speed_gate and memory_gate:
        status = "SEMANTICS_PRESERVING_OPTIMIZATION_CORRECT__PRODUCTION_SPEED_GATE_PASSED"
    elif fully_measured and correctness_gate:
        status = "SEMANTICS_PRESERVING_OPTIMIZATION_CORRECT__PRODUCTION_SPEED_GATE_FAILED"
    elif fully_measured:
        status = "PERFORMANCE_OPTIMIZATION_SEMANTICS_REGRESSION_STOP"
    else:
        status = "PERFORMANCE_GATE_INCOMPLETE"
    result = {
        "schema": "torch_tm_flowpipe.c4_performance_gate/1",
        "status": status,
        "fully_measured": fully_measured,
        "scientific_roots": provenance,
        "cpu_affinity": affinity,
        "observer_mode": "production_no_observer",
        "timer_scope": "solver_only_excludes_snapshot_serialization_checkpoint",
        "brusselator_prefix_100_speedup": speedup100,
        "brusselator_prefix_300_speedup": speedup300,
        "prefix_speed_gate_passed": prefix_gate,
        "brusselator_full_speedup": full_speedup,
        "full_speed_gate_passed": full_speed_gate,
        "memory_gate_passed": memory_gate,
        "b1_exact": full_exact,
        "vdp_regression_passed": bool(vdp_regression["passed"]),
        "brusselator_regression_passed": bool(brusselator_regression["passed"]),
        "correctness_gate_passed": correctness_gate,
        "reference_full_peak_rss_bytes": _max_rss(full_rows, "reference"),
        "optimized_full_peak_rss_bytes": _max_rss(full_rows, "optimized"),
        "no_second_optimization_stacked": True,
    }
    _write_json(output_dir / "optimization_result.json", result)
    _write_json(output_dir / "performance_gate_provenance.json", provenance)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--optimized-root", type=Path, required=True)
    parser.add_argument("--reference-sha", required=True)
    parser.add_argument("--optimized-sha", required=True)
    parser.add_argument("--cpu", type=int, default=0)
    parser.add_argument("--prefix100-steps", type=int, default=100)
    parser.add_argument("--prefix100-repeats", type=int, default=5)
    parser.add_argument("--prefix300-steps", type=int, default=300)
    parser.add_argument("--prefix300-repeats", type=int, default=3)
    parser.add_argument("--vdp-prefix-steps", type=int, default=20)
    parser.add_argument("--vdp-prefix-repeats", type=int, default=3)
    parser.add_argument("--run-full", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--run-vdp-regression",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    args = parser.parse_args(argv)
    for name in (
        "prefix100_steps",
        "prefix100_repeats",
        "prefix300_steps",
        "prefix300_repeats",
        "vdp_prefix_steps",
        "vdp_prefix_repeats",
    ):
        if int(getattr(args, name)) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    result = run(parse_args(argv))
    print(result["status"])
    return 0 if result["status"] != "PERFORMANCE_OPTIMIZATION_SEMANTICS_REGRESSION_STOP" else 1


if __name__ == "__main__":
    raise SystemExit(main())
