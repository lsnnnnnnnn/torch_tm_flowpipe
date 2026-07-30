#!/usr/bin/env python3
"""Repeat selected full configurations and construct native Pareto tables."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
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
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from torch_tm_flowpipe.protocol.eligibility import (
    partition_and_recompute_pareto,
)
from torch_tm_flowpipe.protocol.config import configuration_semantics
from torch_tm_flowpipe.protocol.carry import projected_affine_box_reset
from torch_tm_flowpipe.protocol.provenance import canonical_config_identity
from torch_tm_flowpipe.protocol.schema import (
    Applicability,
    BoundSemantics,
    FailureCategory,
    RUNTIME_BOUNDARY_VERSION,
)
from torch_tm_flowpipe.protocol.runtime import measure_configuration_step


def _float(value: Any, default: float = math.nan) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _maximum_measured_memory(values: Sequence[Any]) -> float | str:
    """Return a measured positive peak, never a fabricated zero."""
    measured = [
        value
        for item in values
        if math.isfinite(value := _float(item)) and value > 0.0
    ]
    return max(measured) if measured else "unavailable"


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
            for repetition in range(repetitions + 1):
                current = normalized_initial_tm(
                    system["initial_box"], order=order
                )
                total_timings: list[float] = []
                engine_timings: list[float] = []
                completed = 0
                reset_discarded_total = 0
                endpoint_box: list[list[float]] = []
                failure = ""
                for step in range(1, steps + 1):
                    def complete_segment(segment: Any):
                        if (
                            segment.status != "validated"
                            or segment.endpoint_raw_tm is None
                        ):
                            return None
                        endpoint = segment.endpoint_raw_tm
                        box = [
                            [
                                float(interval.lo.detach().cpu()),
                                float(interval.hi.detach().cpu()),
                            ]
                            for interval in endpoint.range_box()
                        ]
                        carried, discarded_count = (
                            projected_affine_box_reset(
                                endpoint,
                                project_to_basis=project_to_basis,
                                affine_reset=affine_reset,
                                stage=(
                                    "pareto_affine_reset_projection"
                                ),
                                iteration=step,
                            )
                        )
                        return box, carried, discarded_count

                    timing = measure_configuration_step(
                        lambda: flowpipe_step_from_tm(
                            rhs, current, h, order
                        ),
                        complete_segment,
                    )
                    segment = timing.engine_result
                    engine_timings.append(timing.engine_seconds)
                    total_timings.append(timing.total_seconds)
                    if (
                        segment.status != "validated"
                        or segment.endpoint_raw_tm is None
                    ):
                        failure = segment.message or "validation failure"
                        break
                    if timing.completion_result is None:
                        failure = "validated segment lacked completion data"
                        break
                    endpoint_box, current, discarded_count = (
                        timing.completion_result
                    )
                    reset_discarded_total += discarded_count
                    completed = step
                validation_started = time.perf_counter()
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
                posthoc_validation_s = (
                    time.perf_counter() - validation_started
                )
                total_configuration_s = sum(total_timings)
                rows.append(
                    {
                        "tool": "torch_tm_flowpipe",
                        "variant": f"order{order}_affine_reset_selected",
                        "system": system_name,
                        "h": h,
                        "requested_horizon": horizon,
                        "evaluation_time": completed * h,
                        "repetition": repetition,
                        "measurement_phase": (
                            "cold" if repetition == 0 else "steady"
                        ),
                        "requested_steps": steps,
                        "completed_steps": completed,
                        "successful_horizon": completed * h,
                        "width_at_evaluation_time": max(
                            (box[1] - box[0] for box in endpoint_box),
                            default=math.nan,
                        ),
                        "native_validation_passed": completed == steps,
                        "completed_requested_horizon": completed == steps,
                        "trajectory_sanity_passed": (
                            completed == steps
                            and bool(endpoint_box)
                            and all(
                                math.isfinite(bound)
                                for box in endpoint_box
                                for bound in box
                            )
                            and all(
                                box[0] <= box[1] for box in endpoint_box
                            )
                        ),
                        "reset_discarded_term_count": (
                            reset_discarded_total
                        ),
                        "analytic_reference_contained": exact,
                        "analytic_containment_passed": exact is True,
                        "analytic_containment_applicability": (
                            Applicability.REQUIRED.value
                            if exact is not None
                            else Applicability.NOT_APPLICABLE.value
                        ),
                        "compile_or_jit_time_s": 0.0,
                        "first_step_time_s": (
                            total_timings[0]
                            if total_timings
                            else math.nan
                        ),
                        "cold_total_configuration_time_s": (
                            total_configuration_s
                            if repetition == 0
                            else 0.0
                        ),
                        "steady_total_configuration_time_s": (
                            total_configuration_s
                        ),
                        "steady_full_configuration_time_s": (
                            total_configuration_s
                        ),
                        "engine_internal_time_s": sum(engine_timings),
                        "posthoc_validation_time_s": posthoc_validation_s,
                        "plot_report_time_s": 0.0,
                        "runtime_boundary_version": (
                            RUNTIME_BOUNDARY_VERSION
                        ),
                        "steady_step_time_s": (
                            statistics.median(
                                total_timings[1:] or total_timings
                            )
                            if total_timings
                            else math.nan
                        ),
                        "peak_process_rss_kib": "unavailable",
                        "memory_measurement": "unavailable",
                        "memory_unavailable_reason": (
                            "configuration did not run in an isolated "
                            "measurement subprocess"
                        ),
                        "device": "cpu",
                        "dtype": "float64",
                        "failure_category": (
                            "completed"
                            if completed == steps
                            else "validation_rejected"
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

            for repetition in range(repetitions + 1):
                current = normalized_initial_tm(
                    system["initial_box"],
                    order=order,
                    dtype=torch.float64,
                    device=device,
                )
                total_timings: list[float] = []
                engine_timings: list[float] = []
                endpoint_box: list[list[float]] = []
                completed = 0
                reset_discarded_total = 0
                failure = ""
                torch.cuda.reset_peak_memory_stats(device)
                for step in range(1, steps + 1):
                    total_started = time.perf_counter()
                    torch.cuda.synchronize(device)
                    engine_started = time.perf_counter()
                    segment = flowpipe_step_from_tm(
                        rhs, current, acceleration_h, order
                    )
                    torch.cuda.synchronize(device)
                    engine_timings.append(
                        time.perf_counter() - engine_started
                    )
                    if (
                        segment.status != "validated"
                        or segment.endpoint_raw_tm is None
                    ):
                        failure = (
                            segment.message or "CUDA validation failure"
                        )
                        total_timings.append(
                            time.perf_counter() - total_started
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
                    current, discarded_count = projected_affine_box_reset(
                        endpoint,
                        project_to_basis=project_to_basis,
                        affine_reset=affine_reset,
                        stage="pareto_cuda_affine_reset_projection",
                        iteration=step,
                    )
                    reset_discarded_total += discarded_count
                    completed = step
                    torch.cuda.synchronize(device)
                    total_timings.append(
                        time.perf_counter() - total_started
                    )
                total_configuration_s = sum(total_timings)
                acceleration_rows.append(
                    {
                        "tool": "torch_tm_flowpipe",
                        "variant": "order4_affine_reset_selected",
                        "system": acceleration_system,
                        "h": acceleration_h,
                        "requested_horizon": acceleration_horizon,
                        "evaluation_time": completed * acceleration_h,
                        "repetition": repetition,
                        "measurement_phase": (
                            "cold" if repetition == 0 else "steady"
                        ),
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
                        "completed_requested_horizon": completed == steps,
                        "trajectory_sanity_passed": (
                            completed == steps
                            and bool(endpoint_box)
                            and all(
                                math.isfinite(bound)
                                for box in endpoint_box
                                for bound in box
                            )
                            and all(
                                box[0] <= box[1] for box in endpoint_box
                            )
                        ),
                        "reset_discarded_term_count": (
                            reset_discarded_total
                        ),
                        "analytic_reference_contained": "",
                        "analytic_containment_passed": "",
                        "analytic_containment_applicability": (
                            Applicability.NOT_APPLICABLE.value
                        ),
                        "compile_or_jit_time_s": (
                            warmup_time if repetition == 0 else 0.0
                        ),
                        "first_step_time_s": (
                            total_timings[0]
                            if total_timings
                            else math.nan
                        ),
                        "cold_total_configuration_time_s": (
                            total_configuration_s
                            if repetition == 0
                            else 0.0
                        ),
                        "steady_total_configuration_time_s": (
                            total_configuration_s
                        ),
                        "steady_full_configuration_time_s": (
                            total_configuration_s
                        ),
                        "engine_internal_time_s": sum(engine_timings),
                        "posthoc_validation_time_s": 0.0,
                        "plot_report_time_s": 0.0,
                        "runtime_boundary_version": (
                            RUNTIME_BOUNDARY_VERSION
                        ),
                        "steady_step_time_s": (
                            statistics.median(
                                total_timings[1:] or total_timings
                            )
                            if total_timings
                            else math.nan
                        ),
                        "peak_process_rss_kib": "unavailable",
                        "peak_device_memory_bytes": (
                            torch.cuda.max_memory_allocated(device)
                        ),
                        "device": str(device),
                        "backend": "torch_cuda",
                        "backend_status": "available",
                        "dtype": "float64",
                        "failure_category": (
                            "completed"
                            if completed == steps
                            else "validation_rejected"
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
        for repetition in range(repetitions + 1):
            total_started = time.perf_counter()
            engine_started = time.perf_counter()
            result = jax.tree.map(
                lambda value: value.block_until_ready()
                if hasattr(value, "block_until_ready")
                else value,
                compiled(carry),
            )
            engine_elapsed = time.perf_counter() - engine_started
            _, (los_raw, his, contraction) = result
            los = np.asarray(los_raw)[:, 0, :]
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
                endpoint = [
                    [float(lo), float(hi)]
                    for lo, hi in zip(
                        los[completed - 1], uppers[completed - 1]
                    )
                ]
            total_configuration_s = (
                time.perf_counter() - total_started
            )
            validation_started = time.perf_counter()
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
            posthoc_validation_s = (
                time.perf_counter() - validation_started
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
                    "measurement_phase": (
                        "cold" if repetition == 0 else "steady"
                    ),
                    "requested_steps": steps,
                    "completed_steps": completed,
                    "successful_horizon": completed * h,
                    "width_at_evaluation_time": max(
                        (box[1] - box[0] for box in endpoint),
                        default=math.nan,
                    ),
                    "native_validation_passed": completed == steps,
                    "completed_requested_horizon": completed == steps,
                    "trajectory_sanity_passed": (
                        completed == steps
                        and bool(endpoint)
                        and bool(np.all(np.isfinite(los)))
                        and bool(np.all(np.isfinite(uppers)))
                        and bool(np.all(los <= uppers))
                    ),
                    "analytic_reference_contained": exact,
                    "analytic_containment_passed": exact is True,
                    "analytic_containment_applicability": (
                        Applicability.REQUIRED.value
                        if exact is not None
                        else Applicability.NOT_APPLICABLE.value
                    ),
                    "compile_or_jit_time_s": (
                        compile_first if repetition == 0 else 0.0
                    ),
                    "first_step_time_s": "",
                    "cold_total_configuration_time_s": (
                        total_configuration_s
                        if repetition == 0
                        else 0.0
                    ),
                    "steady_total_configuration_time_s": (
                        total_configuration_s
                    ),
                    "steady_full_configuration_time_s": (
                        total_configuration_s
                    ),
                    "engine_internal_time_s": engine_elapsed,
                    "posthoc_validation_time_s": posthoc_validation_s,
                    "plot_report_time_s": 0.0,
                    "runtime_boundary_version": (
                        RUNTIME_BOUNDARY_VERSION
                    ),
                    "steady_step_time_s": (
                        total_configuration_s / max(steps, 1)
                    ),
                    "peak_process_rss_kib": "unavailable",
                    "memory_measurement": "unavailable",
                    "memory_unavailable_reason": (
                        "configuration did not run in an isolated "
                        "measurement subprocess"
                    ),
                    "device": "cpu",
                    "dtype": "float64",
                    "failure_category": (
                        "completed"
                        if completed == steps
                        else "validation_rejected"
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
                for repetition in range(repetitions + 1):
                    total_started = time.perf_counter()
                    engine_started = time.perf_counter()
                    result = jax.tree.map(
                        lambda value: value.block_until_ready()
                        if hasattr(value, "block_until_ready")
                        else value,
                        compiled(carry),
                    )
                    engine_elapsed = (
                        time.perf_counter() - engine_started
                    )
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
                    total_configuration_s = (
                        time.perf_counter() - total_started
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
                            "measurement_phase": (
                                "cold" if repetition == 0 else "steady"
                            ),
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
                            "completed_requested_horizon": (
                                completed == steps
                            ),
                            "trajectory_sanity_passed": (
                                completed == steps
                                and bool(endpoint)
                                and bool(np.all(np.isfinite(los)))
                                and bool(np.all(np.isfinite(uppers)))
                                and bool(np.all(los <= uppers))
                            ),
                            "analytic_reference_contained": "",
                            "analytic_containment_passed": "",
                            "analytic_containment_applicability": (
                                Applicability.NOT_APPLICABLE.value
                            ),
                            "compile_or_jit_time_s": (
                                compile_first
                                if repetition == 0
                                else 0.0
                            ),
                            "first_step_time_s": "",
                            "cold_total_configuration_time_s": (
                                total_configuration_s
                                if repetition == 0
                                else 0.0
                            ),
                            "steady_total_configuration_time_s": (
                                total_configuration_s
                            ),
                            "steady_full_configuration_time_s": (
                                total_configuration_s
                            ),
                            "engine_internal_time_s": engine_elapsed,
                            "posthoc_validation_time_s": 0.0,
                            "plot_report_time_s": 0.0,
                            "runtime_boundary_version": (
                                RUNTIME_BOUNDARY_VERSION
                            ),
                            "steady_step_time_s": (
                                total_configuration_s
                                / max(steps, 1)
                            ),
                            "peak_process_rss_kib": "unavailable",
                            "memory_measurement": "unavailable",
                            "memory_unavailable_reason": (
                                "configuration did not run in an isolated "
                                "measurement subprocess"
                            ),
                            "device": str(device),
                            "backend": "jax_cuda",
                            "backend_status": "available",
                            "dtype": "float64",
                            "failure_category": (
                                "completed"
                                if completed == steps
                                else "validation_rejected"
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
        for repetition in range(repetitions + 1):
            total_started = time.perf_counter()
            engine_started = time.perf_counter()
            process = subprocess.run(
                [str(executable)],
                cwd=executable.parent,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
                timeout=float(spec["timeout_s"]),
            )
            engine_elapsed = time.perf_counter() - engine_started
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
            total_configuration_s = time.perf_counter() - total_started
            validation_started = time.perf_counter()
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
            posthoc_validation_s = time.perf_counter() - validation_started
            requested_steps = round(horizon / h)
            completed_requested_horizon = (
                process.returncode == 0 and completed == requested_steps
            )
            if process.returncode != 0:
                failure_category = FailureCategory.PROCESS_ERROR.value
            elif not completed_requested_horizon:
                failure_category = FailureCategory.INCOMPLETE_UNKNOWN.value
            elif exact is False:
                failure_category = (
                    FailureCategory.ANALYTIC_CONTAINMENT_FAILED.value
                )
            else:
                failure_category = FailureCategory.COMPLETED.value
            rows.append(
                {
                    "tool": "flowstar",
                    "variant": "root_cause_order4_selected",
                    "system": system_name,
                    "h": h,
                    "requested_horizon": horizon,
                    "evaluation_time": completed * h,
                    "repetition": repetition,
                    "measurement_phase": (
                        "cold" if repetition == 0 else "steady"
                    ),
                    "requested_steps": requested_steps,
                    "completed_steps": completed,
                    "successful_horizon": completed * h,
                    "completed_requested_horizon": (
                        completed_requested_horizon
                    ),
                    "width_at_evaluation_time": max(
                        (box[1] - box[0] for box in endpoint),
                        default=math.nan,
                    ),
                    "native_validation_passed": completed_requested_horizon,
                    "trajectory_sanity_passed": (
                        completed_requested_horizon
                        and len(endpoint_rows)
                        == requested_steps * len(
                            spec["systems"][system_name]["initial_box"]
                        )
                        and all(
                            math.isfinite(float(row["lower"]))
                            and math.isfinite(float(row["upper"]))
                            and float(row["lower"]) <= float(row["upper"])
                            for row in endpoint_rows
                        )
                    ),
                    "analytic_reference_contained": exact,
                    "analytic_containment_passed": exact is True,
                    "analytic_containment_applicability": (
                        Applicability.REQUIRED.value
                        if exact is not None
                        else Applicability.NOT_APPLICABLE.value
                    ),
                    "compile_or_jit_time_s": (
                        run["build_time_s"] if repetition == 0 else 0.0
                    ),
                    "first_step_time_s": (
                        parsed["steps"][0]["seconds"]
                        if parsed["steps"]
                        else ""
                    ),
                    "cold_total_configuration_time_s": (
                        total_configuration_s if repetition == 0 else 0.0
                    ),
                    "steady_total_configuration_time_s": (
                        total_configuration_s
                    ),
                    "steady_full_configuration_time_s": (
                        total_configuration_s
                    ),
                    "engine_internal_time_s": engine_elapsed,
                    "posthoc_validation_time_s": posthoc_validation_s,
                    "plot_report_time_s": 0.0,
                    "runtime_boundary_version": (
                        RUNTIME_BOUNDARY_VERSION
                    ),
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
                    "peak_process_rss_kib": "unavailable",
                    "memory_measurement_reason": (
                        "repetitions share a parent process; isolated "
                        "configuration peak was not measured"
                    ),
                    "device": "cpu",
                    "dtype": "MPFR_interval_53_bit",
                    "failure_category": failure_category,
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
        and row.get("measurement_phase", "steady") == "steady"
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


def _explicit_true(value: Any) -> bool:
    if type(value) is bool:
        return value
    return isinstance(value, str) and value.strip().lower() == "true"


def _finite_values(rows: Sequence[Mapping[str, Any]], field: str) -> list[float]:
    values = [_float(row.get(field)) for row in rows]
    return [value for value in values if math.isfinite(value)]


def collect(
    output: Path, spec: Mapping[str, Any], *, smoke: bool = False
) -> dict[str, Any]:
    native_widths = _native_widths(output)
    exploratory_rows: list[dict[str, Any]] = []
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
            exploratory_rows.append(
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
                    "primary_numerical_eligible": False,
                    "excluded_from_authoritative": True,
                    "exclusion_reason": (
                        "exploratory_single_sweep;"
                        "insufficient_runtime_repetitions"
                    ),
                    "width_runtime_pareto": False,
                    "native_validation_passed": summary.get(
                        "native_validation_passed",
                        summary.get("status") == "success",
                    ),
                    "memory_kib": "unavailable",
                    "basis": summary.get("basis", ""),
                    "carry_or_preconditioning": summary.get(
                        "carry", summary.get("preconditioning", "")
                    ),
                }
            )
    write_csv(output / "native_pareto_exploratory.csv", exploratory_rows)
    write_csv(output / "EXPLORATORY.csv", exploratory_rows)

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
                for field in (
                    "tool",
                    "variant",
                    "system",
                    "h",
                    "requested_horizon",
                )
            )
        ].append(row)
    expected_repetitions = (
        1 if smoke else int(spec["runtime"]["repetitions"])
    )
    runtime_rows: list[dict[str, Any]] = []
    for key, all_values in sorted(grouped.items()):
        cold = [
            row
            for row in all_values
            if row.get("measurement_phase") == "cold"
        ]
        values = [
            row
            for row in all_values
            if row.get("measurement_phase", "steady") == "steady"
        ]
        repetition_indices = {
            int(_float(row.get("repetition"), -1.0)) for row in values
        }
        all_required_repetitions_present = (
            len(values) == expected_repetitions
            and repetition_indices
            == set(range(1, expected_repetitions + 1))
        )
        runtimes = _finite_values(
            values, "steady_total_configuration_time_s"
        )
        widths = _finite_values(values, "width_at_evaluation_time")
        horizons = _finite_values(values, "successful_horizon")
        engine_times = _finite_values(values, "engine_internal_time_s")
        validation_times = _finite_values(
            values, "posthoc_validation_time_s"
        )
        plot_times = _finite_values(values, "plot_report_time_s")
        compile_costs = _finite_values(
            all_values, "compile_or_jit_time_s"
        )
        cold_times = _finite_values(
            cold, "cold_total_configuration_time_s"
        )
        native_passed = bool(values) and all(
            _explicit_true(row.get("native_validation_passed"))
            for row in values
        )
        analytic_not_applicable = bool(values) and all(
            row.get("analytic_containment_applicability")
            == Applicability.NOT_APPLICABLE.value
            for row in values
        )
        analytic_passed = bool(values) and all(
            _explicit_true(row.get("analytic_containment_passed"))
            or row.get("analytic_containment_applicability")
            == Applicability.NOT_APPLICABLE.value
            for row in values
        )
        trajectory_passed = bool(values) and all(
            _explicit_true(row.get("trajectory_sanity_passed"))
            for row in values
        )
        completed = bool(values) and all(
            _explicit_true(row.get("completed_requested_horizon"))
            for row in values
        )
        failure_categories = [
            str(row.get("failure_category", ""))
            for row in values
            if str(row.get("failure_category", ""))
            not in ("", FailureCategory.COMPLETED.value)
        ]
        if not completed:
            failure_category = (
                failure_categories[0]
                if failure_categories
                else FailureCategory.INCOMPLETE_UNKNOWN.value
            )
        elif not analytic_passed:
            failure_category = (
                FailureCategory.ANALYTIC_CONTAINMENT_FAILED.value
            )
        elif not trajectory_passed:
            failure_category = (
                FailureCategory.TRAJECTORY_SANITY_FAILED.value
            )
        elif not native_passed:
            failure_category = FailureCategory.VALIDATION_REJECTED.value
        else:
            failure_category = FailureCategory.COMPLETED.value
        successful_horizon = (
            statistics.median(horizons) if horizons else math.nan
        )
        steady_runtime = (
            statistics.median(runtimes) if runtimes else math.nan
        )
        completed_steps = [
            int(_float(item.get("completed_steps"), 0.0))
            for item in values
        ]
        requested_steps = [
            int(_float(item.get("requested_steps"), 0.0))
            for item in values
        ]
        row = {
            "tool": key[0],
            "variant": key[1],
            "system": key[2],
            "h": float(key[3]),
            "evaluation_time": successful_horizon,
            "requested_horizon": float(key[4]),
            "successful_horizon": successful_horizon,
            "completed_requested_horizon": completed,
            "last_valid_step": (
                min(completed_steps) if completed_steps else 0
            ),
            "failure_step": (
                min(completed_steps) + 1
                if completed_steps and not completed
                else ""
            ),
            "failure_category": failure_category,
            "failure_message": "; ".join(
                sorted(
                    {
                        str(item.get("message", "")).strip()
                        for item in values
                        if str(item.get("message", "")).strip()
                    }
                )
            ),
            "width_at_evaluation_time": (
                statistics.median(widths) if widths else math.nan
            ),
            "cold_total_configuration_time_s": (
                max(cold_times) if cold_times else 0.0
            ),
            "steady_total_configuration_time_s": steady_runtime,
            "steady_full_configuration_time_s": steady_runtime,
            "engine_internal_time_s": (
                statistics.median(engine_times)
                if engine_times
                else 0.0
            ),
            "compile_or_jit_time_s": (
                max(compile_costs) if compile_costs else 0.0
            ),
            "posthoc_validation_time_s": (
                statistics.median(validation_times)
                if validation_times
                else 0.0
            ),
            "plot_report_time_s": (
                statistics.median(plot_times) if plot_times else 0.0
            ),
            "runtime_boundary_version": RUNTIME_BOUNDARY_VERSION,
            "runtime_min_s": min(runtimes) if runtimes else math.nan,
            "runtime_max_s": max(runtimes) if runtimes else math.nan,
            "runtime_repetitions": len(values),
            "runtime_statistic": "median_full_configuration",
            "all_required_repetitions_present": (
                all_required_repetitions_present
            ),
            "native_validation_passed": native_passed,
            "analytic_containment_passed": analytic_passed,
            "analytic_containment_applicability": (
                Applicability.NOT_APPLICABLE.value
                if analytic_not_applicable
                else Applicability.REQUIRED.value
            ),
            "trajectory_sanity_passed": trajectory_passed,
            "bound_semantics": BoundSemantics.RAW_ENDPOINT.value,
            "primary_comparable": True,
            "memory_kib": "unavailable",
            "memory_measurement_reason": (
                "configurations were not launched in isolated "
                "peak-memory subprocesses"
            ),
            "requested_steps": (
                max(requested_steps) if requested_steps else 0
            ),
            "carry_or_preconditioning": "selected_native_practical",
            **configuration_semantics(key[0], key[1]),
        }
        row["config_id"] = canonical_config_identity(row)
        runtime_rows.append(row)

    primary_rows, excluded_rows = partition_and_recompute_pareto(
        runtime_rows,
        required_repetitions=int(spec["runtime"]["repetitions"]),
    )
    write_csv(output / "native_pareto_summary.csv", primary_rows)
    write_csv(output / "native_pareto_excluded.csv", excluded_rows)
    write_csv(output / "runtime_summary.csv", runtime_rows)
    acceleration = _collect_acceleration(output)
    result = {
        "pareto_rows": len(primary_rows),
        "excluded_repeated_configuration_rows": len(excluded_rows),
        "exploratory_single_sweep_rows": len(exploratory_rows),
        "repeated_configuration_rows": len(runtime_rows),
        "repetition_observations": len(repetition_rows),
        "acceleration_rows": len(acceleration),
        "available_acceleration_rows": sum(
            row.get("backend_status") == "available"
            for row in acceleration
        ),
        "all_selected_have_ten_repetitions": all(
            int(row["runtime_repetitions"])
            >= int(spec["runtime"]["repetitions"])
            and row["all_required_repetitions_present"] is True
            for row in runtime_rows
        )
        if runtime_rows
        else False,
    }
    write_json(output / "pareto_checks.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--spec", default=str(REPO_ROOT / "benchmarks" / "canonical.yaml")
    )
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
        result = collect(output, spec, smoke=args.smoke)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
