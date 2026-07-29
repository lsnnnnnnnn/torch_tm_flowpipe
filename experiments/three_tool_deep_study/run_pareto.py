#!/usr/bin/env python3
"""Repeat selected full configurations and construct native Pareto tables."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import resource
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from common import analytic_contained, load_spec, write_csv, write_json

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]


def _float(value: Any, default: float = math.nan) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _multi_cases(spec: Mapping[str, Any], smoke: bool):
    for system_name, configurations in spec["multi_step"].items():
        selected = [spec["smoke"][system_name]] if smoke else configurations
        for configuration in selected:
            yield system_name, float(configuration["h"]), float(
                configuration["horizon"]
            )


def _acceleration_case(
    spec: Mapping[str, Any],
) -> tuple[str, float, float]:
    """Return one nonlinear, cross-term-active full configuration."""
    configuration = spec["multi_step"]["coupled_quadratic"][0]
    return (
        "coupled_quadratic",
        float(configuration["h"]),
        float(configuration["horizon"]),
    )


def _projected_affine_box_reset(
    endpoint: Any,
    *,
    project_to_basis: Any,
    affine_reset: Any,
    stage: str,
    iteration: int,
) -> tuple[Any, int]:
    """Project a possibly nonlinear endpoint before the affine-only reset."""
    affine_endpoint, discarded = project_to_basis(
        endpoint,
        "B1",
        tau_index=None,
        stage=stage,
        iteration=iteration,
    )
    current, _ = affine_reset(affine_endpoint, method="box")
    return current, len(discarded)


def run_torch_repetitions(
    spec: Mapping[str, Any], output: Path, *, smoke: bool
) -> dict[str, Any]:
    import torch

    src = REPO_ROOT / "src"
    followup = HERE.parent / "first_order_followup"
    for candidate in (src, followup):
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
    from export_torch_segment import rhs_from_spec
    from torch_basis import (
        affine_reset,
        normalized_initial_tm,
        project_to_basis,
    )
    from torch_tm_flowpipe import flowpipe_step_from_tm

    repetitions = 1 if smoke else int(spec["runtime"]["repetitions"])
    orders = [2] if smoke else [2, 4]
    rows: list[dict[str, Any]] = []
    for system_name, h, horizon in _multi_cases(spec, smoke):
        system = spec["systems"][system_name]
        rhs = rhs_from_spec(system)
        steps = round(horizon / h)
        for order in orders:
            for repetition in range(repetitions):
                current = normalized_initial_tm(
                    system["initial_box"], order=order
                )
                timings: list[float] = []
                completed = 0
                reset_discarded_total = 0
                endpoint_box: list[list[float]] = []
                failure = ""
                for step in range(1, steps + 1):
                    started = time.perf_counter()
                    segment = flowpipe_step_from_tm(
                        rhs, current, h, order
                    )
                    timings.append(time.perf_counter() - started)
                    if (
                        segment.status != "validated"
                        or segment.endpoint_raw_tm is None
                    ):
                        failure = segment.message or "validation failure"
                        break
                    endpoint = segment.endpoint_raw_tm
                    endpoint_box = [
                        [
                            float(interval.lo.detach().cpu()),
                            float(interval.hi.detach().cpu()),
                        ]
                        for interval in endpoint.range_box()
                    ]
                    current, discarded_count = _projected_affine_box_reset(
                        endpoint,
                        project_to_basis=project_to_basis,
                        affine_reset=affine_reset,
                        stage="pareto_affine_reset_projection",
                        iteration=step,
                    )
                    reset_discarded_total += discarded_count
                    completed = step
                exact = (
                    analytic_contained(
                        system_name,
                        system["initial_box"],
                        completed * h,
                        endpoint_box,
                    )
                    if endpoint_box
                    else None
                )
                rows.append(
                    {
                        "tool": "torch_tm_flowpipe",
                        "variant": f"order{order}_affine_reset_selected",
                        "system": system_name,
                        "h": h,
                        "requested_horizon": horizon,
                        "evaluation_time": completed * h,
                        "repetition": repetition,
                        "requested_steps": steps,
                        "completed_steps": completed,
                        "successful_horizon": completed * h,
                        "width_at_evaluation_time": max(
                            (box[1] - box[0] for box in endpoint_box),
                            default=math.nan,
                        ),
                        "native_validation_passed": completed == steps,
                        "reset_discarded_term_count": (
                            reset_discarded_total
                        ),
                        "analytic_reference_contained": exact,
                        "compile_or_jit_time_s": 0.0,
                        "first_step_time_s": (
                            timings[0] if timings else math.nan
                        ),
                        "steady_full_configuration_time_s": sum(timings),
                        "steady_step_time_s": (
                            statistics.median(timings[1:] or timings)
                            if timings
                            else math.nan
                        ),
                        "peak_process_rss_kib": resource.getrusage(
                            resource.RUSAGE_SELF
                        ).ru_maxrss,
                        "device": "cpu",
                        "dtype": "float64",
                        "failure_category": (
                            "" if completed == steps else "validation_failure"
                        ),
                        "message": failure,
                    }
                )
    write_csv(output / "pareto_repetitions_torch.csv", rows)

    acceleration_rows: list[dict[str, Any]] = []
    acceleration_system, acceleration_h, acceleration_horizon = (
        _acceleration_case(spec)
    )
    acceleration_variant = (
        "order2_affine_reset_selected"
        if smoke
        else "order4_affine_reset_selected"
    )
    for row in rows:
        if (
            row["variant"] == acceleration_variant
            and row["system"] == acceleration_system
            and math.isclose(float(row["h"]), acceleration_h)
        ):
            acceleration_rows.append(
                {
                    **row,
                    "backend": "torch_cpu",
                    "backend_status": "available",
                    "acceleration_scope": (
                        "secondary_native_hardware_throughput;"
                        "same_full_configuration"
                    ),
                    "algorithmic_hardware_fair_comparison": False,
                }
            )

    if not smoke:
        if torch.cuda.is_available():
            system = spec["systems"][acceleration_system]
            rhs = rhs_from_spec(system)
            order = 4
            steps = round(acceleration_horizon / acceleration_h)
            device = torch.device("cuda:0")

            # Initialize the CUDA context and run one unreported solver step so
            # context startup is not charged to the repeated configurations.
            warm_current = normalized_initial_tm(
                system["initial_box"],
                order=order,
                dtype=torch.float64,
                device=device,
            )
            torch.cuda.synchronize(device)
            warm_started = time.perf_counter()
            warm_segment = flowpipe_step_from_tm(
                rhs, warm_current, acceleration_h, order
            )
            torch.cuda.synchronize(device)
            warmup_time = time.perf_counter() - warm_started
            if warm_segment.status != "validated":
                raise RuntimeError(
                    "Torch CUDA acceleration warmup failed: "
                    f"{warm_segment.message}"
                )

            for repetition in range(repetitions):
                current = normalized_initial_tm(
                    system["initial_box"],
                    order=order,
                    dtype=torch.float64,
                    device=device,
                )
                timings: list[float] = []
                endpoint_box: list[list[float]] = []
                completed = 0
                reset_discarded_total = 0
                failure = ""
                torch.cuda.reset_peak_memory_stats(device)
                for step in range(1, steps + 1):
                    torch.cuda.synchronize(device)
                    started = time.perf_counter()
                    segment = flowpipe_step_from_tm(
                        rhs, current, acceleration_h, order
                    )
                    torch.cuda.synchronize(device)
                    timings.append(time.perf_counter() - started)
                    if (
                        segment.status != "validated"
                        or segment.endpoint_raw_tm is None
                    ):
                        failure = (
                            segment.message or "CUDA validation failure"
                        )
                        break
                    endpoint = segment.endpoint_raw_tm
                    endpoint_box = [
                        [
                            float(interval.lo.detach().cpu()),
                            float(interval.hi.detach().cpu()),
                        ]
                        for interval in endpoint.range_box()
                    ]
                    current, discarded_count = _projected_affine_box_reset(
                        endpoint,
                        project_to_basis=project_to_basis,
                        affine_reset=affine_reset,
                        stage="pareto_cuda_affine_reset_projection",
                        iteration=step,
                    )
                    reset_discarded_total += discarded_count
                    completed = step
                acceleration_rows.append(
                    {
                        "tool": "torch_tm_flowpipe",
                        "variant": "order4_affine_reset_selected",
                        "system": acceleration_system,
                        "h": acceleration_h,
                        "requested_horizon": acceleration_horizon,
                        "evaluation_time": completed * acceleration_h,
                        "repetition": repetition,
                        "requested_steps": steps,
                        "completed_steps": completed,
                        "successful_horizon": completed * acceleration_h,
                        "width_at_evaluation_time": max(
                            (
                                bounds[1] - bounds[0]
                                for bounds in endpoint_box
                            ),
                            default=math.nan,
                        ),
                        "native_validation_passed": completed == steps,
                        "reset_discarded_term_count": (
                            reset_discarded_total
                        ),
                        "analytic_reference_contained": "",
                        "compile_or_jit_time_s": (
                            warmup_time if repetition == 0 else 0.0
                        ),
                        "first_step_time_s": (
                            timings[0] if timings else math.nan
                        ),
                        "steady_full_configuration_time_s": sum(timings),
                        "steady_step_time_s": (
                            statistics.median(timings[1:] or timings)
                            if timings
                            else math.nan
                        ),
                        "peak_process_rss_kib": "",
                        "peak_device_memory_bytes": (
                            torch.cuda.max_memory_allocated(device)
                        ),
                        "device": str(device),
                        "backend": "torch_cuda",
                        "backend_status": "available",
                        "dtype": "float64",
                        "failure_category": (
                            ""
                            if completed == steps
                            else "validation_failure"
                        ),
                        "message": failure,
                        "acceleration_scope": (
                            "secondary_native_hardware_throughput;"
                            "same_full_configuration"
                        ),
                        "algorithmic_hardware_fair_comparison": False,
                    }
                )
        else:
            acceleration_rows.append(
                {
                    "tool": "torch_tm_flowpipe",
                    "variant": "order4_affine_reset_selected",
                    "system": acceleration_system,
                    "h": acceleration_h,
                    "requested_horizon": acceleration_horizon,
                    "device": "cuda",
                    "backend": "torch_cuda",
                    "backend_status": "unavailable",
                    "message": "torch.cuda.is_available() is false",
                    "acceleration_scope": (
                        "secondary_native_hardware_throughput"
                    ),
                    "algorithmic_hardware_fair_comparison": False,
                }
            )
    write_csv(output / "acceleration_torch.csv", acceleration_rows)
    return {
        "rows": len(rows),
        "repetitions": repetitions,
        "acceleration_rows": len(acceleration_rows),
    }


def run_diffreach_repetitions(
    spec: Mapping[str, Any], output: Path, *, smoke: bool
) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp

    from export_diffreach_segment import (
        _initial_carry,
        _rhs,
        dr_settings,
        reachability,
    )

    jax.config.update("jax_enable_x64", True)
    repetitions = 1 if smoke else int(spec["runtime"]["repetitions"])
    rows: list[dict[str, Any]] = []
    for system_name, h, horizon in _multi_cases(spec, smoke):
        system = spec["systems"][system_name]
        steps = round(horizon / h)
        window = max(map(int, spec["diffreach"]["symbolic_windows"]))
        rounds = max(map(int, spec["diffreach"]["frr_rounds"]))
        dr_settings.update_config(
            {
                "TRUNCATE_TO_AFFINE": False,
                "BOUND_TIME_STEP": True,
                "DEBUG_LOG": False,
            }
        )
        core = reachability.CT_Dyn_Reach(
            rhs=_rhs(system),
            state_dim=len(system["state_names"]),
            nn_dyn=False,
            step_size=h,
            init_remainder=float(spec["diffreach"]["init_remainder"]),
            frr_rounds=rounds,
            frr_stop_ratio=float(spec["diffreach"]["frr_stop_ratio"]),
            sr_window_size=window,
        )
        core.step_boxes = reachability._make_step_boxes(
            1,
            len(system["state_names"]),
            h,
            dtype=jnp.float64,
        )
        carry = _initial_carry(system, min(window, steps))
        compiled = jax.jit(
            lambda initial: jax.lax.scan(
                core.step_once, initial, None, length=steps
            )
        )
        started = time.perf_counter()
        warm = jax.tree.map(
            lambda value: value.block_until_ready()
            if hasattr(value, "block_until_ready")
            else value,
            compiled(carry),
        )
        compile_first = time.perf_counter() - started
        del warm
        for repetition in range(repetitions):
            started = time.perf_counter()
            result = jax.tree.map(
                lambda value: value.block_until_ready()
                if hasattr(value, "block_until_ready")
                else value,
                compiled(carry),
            )
            elapsed = time.perf_counter() - started
            _, (_, his, contraction) = result
            uppers = np.asarray(his)[:, 0, :]
            contractions = np.asarray(contraction)
            valid = np.all(
                contractions,
                axis=tuple(range(1, contractions.ndim)),
            )
            finite = np.all(np.isfinite(uppers), axis=1)
            bad = np.flatnonzero(~(valid & finite))
            completed = int(bad[0]) if bad.size else steps
            endpoint: list[list[float]] = []
            if completed:
                los = np.asarray(result[1][0])[:, 0, :]
                endpoint = [
                    [float(lo), float(hi)]
                    for lo, hi in zip(
                        los[completed - 1], uppers[completed - 1]
                    )
                ]
            exact = (
                analytic_contained(
                    system_name,
                    system["initial_box"],
                    completed * h,
                    endpoint,
                )
                if endpoint
                else None
            )
            rows.append(
                {
                    "tool": "diffreach",
                    "variant": "restricted_quasi_window100_round5_selected",
                    "system": system_name,
                    "h": h,
                    "requested_horizon": horizon,
                    "evaluation_time": completed * h,
                    "repetition": repetition,
                    "requested_steps": steps,
                    "completed_steps": completed,
                    "successful_horizon": completed * h,
                    "width_at_evaluation_time": max(
                        (box[1] - box[0] for box in endpoint),
                        default=math.nan,
                    ),
                    "native_validation_passed": completed == steps,
                    "analytic_reference_contained": exact,
                    "compile_or_jit_time_s": (
                        compile_first if repetition == 0 else 0.0
                    ),
                    "first_step_time_s": "",
                    "steady_full_configuration_time_s": elapsed,
                    "steady_step_time_s": elapsed / max(steps, 1),
                    "peak_process_rss_kib": resource.getrusage(
                        resource.RUSAGE_SELF
                    ).ru_maxrss,
                    "device": "cpu",
                    "dtype": "float64",
                    "failure_category": (
                        "" if completed == steps else "validation_failure"
                    ),
                    "message": "",
                }
            )
    write_csv(output / "pareto_repetitions_diffreach.csv", rows)

    acceleration_rows: list[dict[str, Any]] = []
    acceleration_system, acceleration_h, acceleration_horizon = (
        _acceleration_case(spec)
    )
    for row in rows:
        if (
            row["system"] == acceleration_system
            and math.isclose(float(row["h"]), acceleration_h)
        ):
            acceleration_rows.append(
                {
                    **row,
                    "backend": "jax_cpu",
                    "backend_status": "available",
                    "acceleration_scope": (
                        "secondary_native_hardware_throughput;"
                        "same_full_configuration"
                    ),
                    "algorithmic_hardware_fair_comparison": False,
                }
            )

    if not smoke:
        gpu_devices = [
            device for device in jax.devices() if device.platform == "gpu"
        ]
        if gpu_devices:
            device = gpu_devices[0]
            system = spec["systems"][acceleration_system]
            steps = round(acceleration_horizon / acceleration_h)
            window = max(map(int, spec["diffreach"]["symbolic_windows"]))
            rounds = max(map(int, spec["diffreach"]["frr_rounds"]))
            with jax.default_device(device):
                dr_settings.update_config(
                    {
                        "TRUNCATE_TO_AFFINE": False,
                        "BOUND_TIME_STEP": True,
                        "DEBUG_LOG": False,
                    }
                )
                core = reachability.CT_Dyn_Reach(
                    rhs=_rhs(system),
                    state_dim=len(system["state_names"]),
                    nn_dyn=False,
                    step_size=acceleration_h,
                    init_remainder=float(
                        spec["diffreach"]["init_remainder"]
                    ),
                    frr_rounds=rounds,
                    frr_stop_ratio=float(
                        spec["diffreach"]["frr_stop_ratio"]
                    ),
                    sr_window_size=window,
                )
                core.step_boxes = reachability._make_step_boxes(
                    1,
                    len(system["state_names"]),
                    acceleration_h,
                    dtype=jnp.float64,
                )
                carry = jax.device_put(
                    _initial_carry(system, min(window, steps)), device
                )
                compiled = jax.jit(
                    lambda initial: jax.lax.scan(
                        core.step_once, initial, None, length=steps
                    ),
                    device=device,
                )
                started = time.perf_counter()
                warm = jax.tree.map(
                    lambda value: value.block_until_ready()
                    if hasattr(value, "block_until_ready")
                    else value,
                    compiled(carry),
                )
                compile_first = time.perf_counter() - started
                del warm
                for repetition in range(repetitions):
                    started = time.perf_counter()
                    result = jax.tree.map(
                        lambda value: value.block_until_ready()
                        if hasattr(value, "block_until_ready")
                        else value,
                        compiled(carry),
                    )
                    elapsed = time.perf_counter() - started
                    los = np.asarray(result[1][0])[:, 0, :]
                    uppers = np.asarray(result[1][1])[:, 0, :]
                    contractions = np.asarray(result[1][2])
                    valid = np.all(
                        contractions,
                        axis=tuple(range(1, contractions.ndim)),
                    )
                    finite = np.all(np.isfinite(uppers), axis=1)
                    bad = np.flatnonzero(~(valid & finite))
                    completed = int(bad[0]) if bad.size else steps
                    endpoint = (
                        [
                            [float(lo), float(hi)]
                            for lo, hi in zip(
                                los[completed - 1],
                                uppers[completed - 1],
                            )
                        ]
                        if completed
                        else []
                    )
                    acceleration_rows.append(
                        {
                            "tool": "diffreach",
                            "variant": (
                                "restricted_quasi_window100_round5_selected"
                            ),
                            "system": acceleration_system,
                            "h": acceleration_h,
                            "requested_horizon": acceleration_horizon,
                            "evaluation_time": completed * acceleration_h,
                            "repetition": repetition,
                            "requested_steps": steps,
                            "completed_steps": completed,
                            "successful_horizon": (
                                completed * acceleration_h
                            ),
                            "width_at_evaluation_time": max(
                                (
                                    bounds[1] - bounds[0]
                                    for bounds in endpoint
                                ),
                                default=math.nan,
                            ),
                            "native_validation_passed": (
                                completed == steps
                            ),
                            "analytic_reference_contained": "",
                            "compile_or_jit_time_s": (
                                compile_first
                                if repetition == 0
                                else 0.0
                            ),
                            "first_step_time_s": "",
                            "steady_full_configuration_time_s": elapsed,
                            "steady_step_time_s": elapsed / max(steps, 1),
                            "peak_process_rss_kib": resource.getrusage(
                                resource.RUSAGE_SELF
                            ).ru_maxrss,
                            "device": str(device),
                            "backend": "jax_cuda",
                            "backend_status": "available",
                            "dtype": "float64",
                            "failure_category": (
                                ""
                                if completed == steps
                                else "validation_failure"
                            ),
                            "message": "",
                            "acceleration_scope": (
                                "secondary_native_hardware_throughput;"
                                "same_full_configuration"
                            ),
                            "algorithmic_hardware_fair_comparison": False,
                        }
                    )
        else:
            acceleration_rows.append(
                {
                    "tool": "diffreach",
                    "variant": (
                        "restricted_quasi_window100_round5_selected"
                    ),
                    "system": acceleration_system,
                    "h": acceleration_h,
                    "requested_horizon": acceleration_horizon,
                    "device": "cuda",
                    "backend": "jax_cuda",
                    "backend_status": "unavailable",
                    "message": (
                        "installed JAX/JAXlib exposes no GPU device"
                    ),
                    "acceleration_scope": (
                        "secondary_native_hardware_throughput"
                    ),
                    "algorithmic_hardware_fair_comparison": False,
                }
            )
    write_csv(
        output / "acceleration_diffreach.csv", acceleration_rows
    )
    return {
        "rows": len(rows),
        "repetitions": repetitions,
        "acceleration_rows": len(acceleration_rows),
    }


def run_flowstar_repetitions(
    spec: Mapping[str, Any], output: Path, *, smoke: bool
) -> dict[str, Any]:
    from run_native import _load_flowstar_repair

    runner = _load_flowstar_repair()
    repetitions = 1 if smoke else int(spec["runtime"]["repetitions"])
    rows: list[dict[str, Any]] = []
    environment = os.environ.copy()
    environment["FLOWSTAR_AUDIT_CACHE_LEAF_TRUNCATION"] = "1"
    environment["FLOWSTAR_AUDIT_REVALIDATE_REFINEMENT"] = "0"
    for system_name, h, horizon in _multi_cases(spec, smoke):
        candidate = float(
            spec["flowstar"]["candidate_remainder"][system_name]
        )
        _, run, _ = runner.run_fixed_case(
            spec,
            output,
            system_name=system_name,
            protocol=runner.PROTOCOL_NATIVE,
            h=h,
            horizon=horizon,
            order=4,
            candidate=candidate,
            cutoff=float(spec["flowstar"]["cutoff"]),
            variant="root_cause_order4_selected",
        )
        executable = Path(run["source"]).with_suffix("")
        for repetition in range(repetitions):
            started = time.perf_counter()
            process = subprocess.run(
                [str(executable)],
                cwd=executable.parent,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
                timeout=float(spec["timeout_s"]),
            )
            elapsed = time.perf_counter() - started
            parsed = runner._parse_fixed(process.stdout)
            endpoint_rows = [
                row
                for row in parsed["rows"]
                if row["kind"] == "endpoint_raw"
            ]
            completed = max(
                (int(row["step"]) for row in endpoint_rows), default=0
            )
            final_rows = [
                row
                for row in endpoint_rows
                if int(row["step"]) == completed
            ]
            endpoint = [
                [float(row["lower"]), float(row["upper"])]
                for row in sorted(final_rows, key=lambda item: item["state"])
            ]
            exact = (
                analytic_contained(
                    system_name,
                    spec["systems"][system_name]["initial_box"],
                    completed * h,
                    endpoint,
                )
                if endpoint
                else None
            )
            rows.append(
                {
                    "tool": "flowstar",
                    "variant": "root_cause_order4_selected",
                    "system": system_name,
                    "h": h,
                    "requested_horizon": horizon,
                    "evaluation_time": completed * h,
                    "repetition": repetition,
                    "requested_steps": round(horizon / h),
                    "completed_steps": completed,
                    "successful_horizon": completed * h,
                    "width_at_evaluation_time": max(
                        (box[1] - box[0] for box in endpoint),
                        default=math.nan,
                    ),
                    "native_validation_passed": (
                        process.returncode == 0
                        and completed == round(horizon / h)
                    ),
                    "analytic_reference_contained": exact,
                    "compile_or_jit_time_s": (
                        run["build_time_s"] if repetition == 0 else 0.0
                    ),
                    "first_step_time_s": (
                        parsed["steps"][0]["seconds"]
                        if parsed["steps"]
                        else ""
                    ),
                    "steady_full_configuration_time_s": elapsed,
                    "steady_step_time_s": (
                        statistics.median(
                            [
                                item["seconds"]
                                for item in parsed["steps"][1:]
                            ]
                            or [
                                item["seconds"]
                                for item in parsed["steps"]
                            ]
                        )
                        if parsed["steps"]
                        else math.nan
                    ),
                    "peak_process_rss_kib": "",
                    "device": "cpu",
                    "dtype": "MPFR_interval_53_bit",
                    "failure_category": (
                        ""
                        if process.returncode == 0
                        else "native_process_failure"
                    ),
                    "message": process.stderr[-1000:],
                }
            )
    write_csv(output / "pareto_repetitions_flowstar.csv", rows)
    acceleration_system, acceleration_h, _ = _acceleration_case(spec)
    acceleration_rows = [
        {
            **row,
            "backend": "flowstar_cpu",
            "backend_status": "available",
            "acceleration_scope": (
                "secondary_native_hardware_throughput;"
                "same_full_configuration"
            ),
            "algorithmic_hardware_fair_comparison": False,
        }
        for row in rows
        if row["system"] == acceleration_system
        and math.isclose(float(row["h"]), acceleration_h)
    ]
    write_csv(
        output / "acceleration_flowstar.csv", acceleration_rows
    )
    return {
        "rows": len(rows),
        "repetitions": repetitions,
        "acceleration_rows": len(acceleration_rows),
    }


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _native_widths(output: Path) -> dict[tuple[str, ...], tuple[float, float]]:
    selected: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for tool in ("torch", "diffreach", "flowstar"):
        for row in _read_csv(output / f"native_{tool}.csv"):
            if row.get("interval_kind") != "endpoint_raw":
                continue
            key = tuple(
                str(row.get(field, ""))
                for field in ("tool", "variant", "system", "h")
            )
            selected[key].append(row)
    results: dict[tuple[str, ...], tuple[float, float]] = {}
    for key, rows in selected.items():
        final_step = max(_float(row.get("step_index"), 0.0) for row in rows)
        final = [
            row
            for row in rows
            if _float(row.get("step_index"), 0.0) == final_step
        ]
        results[key] = (
            max((_float(row.get("width")) for row in final), default=math.nan),
            max((_float(row.get("time")) for row in final), default=math.nan),
        )
    return results


def _runtime(summary: Mapping[str, Any]) -> float:
    for field in (
        "total_runtime_s",
        "execution_time_s",
        "after_jit_call_s",
    ):
        value = _float(summary.get(field))
        if math.isfinite(value):
            return value
    step = _float(summary.get("steady_step_time_s"))
    steps = _float(summary.get("completed_steps"))
    return step * steps


def _pareto_flags(rows: list[dict[str, Any]]) -> None:
    # Native practical configurations do not share exact bases/backends.
    # Pareto dominance is therefore meaningful only among configurations of
    # the same tool at the same system and absolute evaluation time.
    groups: dict[tuple[str, str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        evaluation_time = _float(row.get("evaluation_time"))
        if not math.isfinite(evaluation_time):
            continue
        groups[
            (
                str(row["tool"]),
                str(row["system"]),
                round(evaluation_time, 12),
            )
        ].append(row)
    for group in groups.values():
        for candidate in group:
            candidate["width_runtime_pareto"] = not any(
                other is not candidate
                and _float(other["width_at_evaluation_time"])
                <= _float(candidate["width_at_evaluation_time"])
                and _float(other["steady_full_configuration_time_s"])
                <= _float(candidate["steady_full_configuration_time_s"])
                and (
                    _float(other["width_at_evaluation_time"])
                    < _float(candidate["width_at_evaluation_time"])
                    or _float(other["steady_full_configuration_time_s"])
                    < _float(candidate["steady_full_configuration_time_s"])
                )
                for other in group
            )


def _collect_acceleration(output: Path) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    for tool in ("torch", "diffreach", "flowstar"):
        observations.extend(
            _read_csv(output / f"acceleration_{tool}.csv")
        )
    unavailable = [
        row
        for row in observations
        if str(row.get("backend_status", "")).lower() != "available"
    ]
    available = [
        row
        for row in observations
        if str(row.get("backend_status", "")).lower() == "available"
    ]
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in available:
        grouped[
            tuple(
                str(row.get(field, ""))
                for field in (
                    "tool",
                    "backend",
                    "variant",
                    "system",
                    "h",
                    "requested_horizon",
                )
            )
        ].append(row)

    summaries: list[dict[str, Any]] = []
    for key, values in sorted(grouped.items()):
        runtimes = [
            _float(row.get("steady_full_configuration_time_s"))
            for row in values
        ]
        widths = [
            _float(row.get("width_at_evaluation_time"))
            for row in values
        ]
        horizons = [
            _float(row.get("successful_horizon")) for row in values
        ]
        summaries.append(
            {
                "tool": key[0],
                "backend": key[1],
                "backend_status": "available",
                "variant": key[2],
                "system": key[3],
                "h": key[4],
                "requested_horizon": key[5],
                "runtime_repetitions": len(values),
                "median_full_configuration_time_s": statistics.median(
                    runtimes
                ),
                "runtime_min_s": min(runtimes),
                "runtime_max_s": max(runtimes),
                "median_width_at_evaluation_time": statistics.median(
                    widths
                ),
                "median_successful_horizon": statistics.median(horizons),
                "all_native_validations_passed": all(
                    str(row.get("native_validation_passed", "")).lower()
                    == "true"
                    for row in values
                ),
                "compile_or_context_time_s": max(
                    _float(row.get("compile_or_jit_time_s"), 0.0)
                    for row in values
                ),
                "algorithmic_hardware_fair_comparison": False,
                "interpretation": (
                    "implementation/hardware throughput only"
                ),
                "message": "",
            }
        )
    for row in unavailable:
        summaries.append(
            {
                "tool": row.get("tool", ""),
                "backend": row.get("backend", ""),
                "backend_status": row.get(
                    "backend_status", "unavailable"
                ),
                "variant": row.get("variant", ""),
                "system": row.get("system", ""),
                "h": row.get("h", ""),
                "requested_horizon": row.get(
                    "requested_horizon", ""
                ),
                "runtime_repetitions": 0,
                "median_full_configuration_time_s": "",
                "runtime_min_s": "",
                "runtime_max_s": "",
                "median_width_at_evaluation_time": "",
                "median_successful_horizon": "",
                "all_native_validations_passed": "",
                "compile_or_context_time_s": "",
                "algorithmic_hardware_fair_comparison": False,
                "interpretation": (
                    "implementation/hardware capability unavailable"
                ),
                "message": row.get("message", ""),
            }
        )

    cpu_by_tool = {
        str(row["tool"]): _float(
            row.get("median_full_configuration_time_s")
        )
        for row in summaries
        if str(row.get("backend", "")).endswith("_cpu")
        and row.get("backend_status") == "available"
    }
    for row in summaries:
        cpu = cpu_by_tool.get(str(row.get("tool", "")), math.nan)
        runtime = _float(row.get("median_full_configuration_time_s"))
        row["speedup_vs_same_tool_cpu"] = (
            cpu / runtime
            if math.isfinite(cpu)
            and math.isfinite(runtime)
            and runtime > 0
            else ""
        )
    write_csv(output / "acceleration_summary.csv", summaries)
    return summaries


def collect(output: Path) -> dict[str, Any]:
    native_widths = _native_widths(output)
    rows: list[dict[str, Any]] = []
    for tool in ("torch", "diffreach", "flowstar"):
        path = output / f"native_{tool}_summary.json"
        if not path.exists():
            continue
        for summary in json.loads(path.read_text(encoding="utf-8")):
            key = (
                str(summary.get("tool", "")),
                str(summary.get("variant", "")),
                str(summary.get("system", "")),
                str(summary.get("h", "")),
            )
            width, evaluation_time = native_widths.get(
                key, (math.nan, math.nan)
            )
            rows.append(
                {
                    "tool": key[0],
                    "variant": key[1],
                    "system": key[2],
                    "h": key[3],
                    "evaluation_time": evaluation_time,
                    "requested_horizon": summary.get(
                        "horizon",
                        summary.get("requested_horizon", ""),
                    ),
                    "successful_horizon": summary.get(
                        "successful_horizon",
                        _float(summary.get("completed_steps"))
                        * _float(summary.get("h")),
                    ),
                    "width_at_evaluation_time": width,
                    "steady_full_configuration_time_s": _runtime(summary),
                    "compile_or_jit_time_s": summary.get(
                        "jit_compile_and_first_call_s",
                        summary.get("build_time_s", 0.0),
                    ),
                    "runtime_repetitions": 1,
                    "runtime_statistic": "single_native_sweep",
                    "native_validation_passed": summary.get(
                        "native_validation_passed",
                        summary.get("status") == "success",
                    ),
                    "memory_kib": "",
                    "basis": summary.get("basis", ""),
                    "carry_or_preconditioning": summary.get(
                        "carry", summary.get("preconditioning", "")
                    ),
                }
            )
    repetition_rows: list[dict[str, Any]] = []
    for tool in ("torch", "diffreach", "flowstar"):
        repetition_rows.extend(
            _read_csv(output / f"pareto_repetitions_{tool}.csv")
        )
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in repetition_rows:
        grouped[
            tuple(
                str(row.get(field, ""))
                for field in ("tool", "variant", "system", "h")
            )
        ].append(row)
    runtime_rows: list[dict[str, Any]] = []
    for key, values in grouped.items():
        successful = [
            row
            for row in values
            if str(row.get("native_validation_passed", "")).lower()
            == "true"
        ]
        source = successful or values
        runtimes = [
            _float(row.get("steady_full_configuration_time_s"))
            for row in source
        ]
        widths = [
            _float(row.get("width_at_evaluation_time")) for row in source
        ]
        horizons = [_float(row.get("successful_horizon")) for row in source]
        compile_costs = [
            _float(row.get("compile_or_jit_time_s"), 0.0)
            for row in source
        ]
        row = {
            "tool": key[0],
            "variant": key[1],
            "system": key[2],
            "h": key[3],
            "evaluation_time": statistics.median(horizons),
            "requested_horizon": source[0].get("requested_horizon", ""),
            "successful_horizon": statistics.median(horizons),
            "width_at_evaluation_time": statistics.median(widths),
            "steady_full_configuration_time_s": statistics.median(
                runtimes
            ),
            "runtime_min_s": min(runtimes),
            "runtime_max_s": max(runtimes),
            "compile_or_jit_time_s": max(compile_costs),
            "runtime_repetitions": len(values),
            "runtime_statistic": "median_full_configuration",
            "native_validation_passed": len(successful) == len(values),
            "memory_kib": max(
                (
                    _float(item.get("peak_process_rss_kib"), 0.0)
                    for item in source
                ),
                default=0.0,
            ),
            "basis": "",
            "carry_or_preconditioning": "selected_native_practical",
        }
        rows.append(row)
        runtime_rows.append(row)
    _pareto_flags(rows)
    write_csv(output / "native_pareto_summary.csv", rows)
    write_csv(output / "runtime_summary.csv", runtime_rows)
    acceleration = _collect_acceleration(output)
    result = {
        "pareto_rows": len(rows),
        "repeated_configuration_rows": len(runtime_rows),
        "repetition_observations": len(repetition_rows),
        "acceleration_rows": len(acceleration),
        "available_acceleration_rows": sum(
            row.get("backend_status") == "available"
            for row in acceleration
        ),
        "all_selected_have_ten_repetitions": all(
            int(row["runtime_repetitions"]) >= 10 for row in runtime_rows
        )
        if runtime_rows
        else False,
    }
    write_json(output / "pareto_checks.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", default=str(HERE / "benchmark_spec.yaml"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--tool",
        choices=["torch", "diffreach", "flowstar", "collect"],
        required=True,
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    spec = load_spec(args.spec)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.tool == "torch":
        result = run_torch_repetitions(spec, output, smoke=args.smoke)
    elif args.tool == "diffreach":
        result = run_diffreach_repetitions(
            spec, output, smoke=args.smoke
        )
    elif args.tool == "flowstar":
        result = run_flowstar_repetitions(
            spec, output, smoke=args.smoke
        )
    else:
        result = collect(output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
