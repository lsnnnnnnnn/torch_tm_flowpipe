#!/usr/bin/env python3
"""Run Torch TM under one-step, common-box, and native-carry contracts."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
SRC_ROOT = REPO_ROOT / "src"
for candidate in (SRC_ROOT, HERE):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import torch

from torch_tm_flowpipe import Interval, TMVector, flowpipe_step, flowpipe_step_from_tm

from common import (
    PROTOCOL_A,
    PROTOCOL_B,
    PROTOCOL_C,
    RAW_FIELDS,
    RUN_FIELDS,
    base_run,
    copy_runtime_fields,
    evaluate_rhs,
    exact_interval_for_row,
    git_sha,
    iter_configurations,
    load_spec,
    make_row,
    median,
    write_csv,
    write_json,
)

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)


def _rhs(system_spec: Mapping[str, Any]):
    def rhs(state: TMVector, control: TMVector | None = None) -> TMVector:
        del control
        return TMVector(evaluate_rhs(list(state), system_spec))

    return rhs


def _finite_box(box: Sequence[Interval]) -> bool:
    return all(interval.is_finite() for interval in box)


def _bounds(interval: Interval) -> tuple[float, float]:
    return interval.to_tuple()


def _component_metrics(model: Any) -> tuple[float, float]:
    polynomial_range = model.polynomial.evaluate_interval(model.domain)
    return (
        float(polynomial_range.width().detach().cpu()),
        float(model.remainder.width().detach().cpu()),
    )


def _append_initial_rows(
    rows: list[dict[str, Any]],
    run: Mapping[str, Any],
    system: Mapping[str, Any],
) -> None:
    for state_index, (state_name, bounds) in enumerate(
        zip(system["state_names"], system["initial_box"])
    ):
        lower, upper = map(float, bounds)
        rows.append(
            make_row(
                run,
                state_index=state_index,
                state_name=state_name,
                step_index=0,
                time_value=0.0,
                interval_kind="endpoint",
                lower=lower,
                upper=upper,
                exact=(lower, upper),
                polynomial_width=upper - lower,
                interval_remainder_width=0.0,
                row_status="validated",
                native_validation_status="initial_set",
            )
        )


def _append_segment_rows(
    rows: list[dict[str, Any]],
    run: Mapping[str, Any],
    system_name: str,
    system: Mapping[str, Any],
    segment: Any,
    step_index: int,
    h: float,
) -> bool:
    time_value = step_index * h
    analytic_ok = True
    for interval_kind, tm_vector in (
        ("endpoint", segment.final_tm),
        ("tube", segment.tm),
    ):
        exact_boxes = exact_interval_for_row(
            system_name,
            interval_kind,
            time_value,
            h,
            system["initial_box"],
        )
        for state_index, (state_name, model, interval) in enumerate(
            zip(system["state_names"], tm_vector, tm_vector.range_box())
        ):
            lower, upper = _bounds(interval)
            exact = None if exact_boxes is None else exact_boxes[state_index]
            contained = (
                True
                if exact is None
                else lower <= exact[0] + 1e-12 and upper >= exact[1] - 1e-12
            )
            analytic_ok = analytic_ok and contained
            polynomial_width, remainder_width = _component_metrics(model)
            rows.append(
                make_row(
                    run,
                    state_index=state_index,
                    state_name=state_name,
                    step_index=step_index,
                    time_value=time_value,
                    interval_kind=interval_kind,
                    lower=lower,
                    upper=upper,
                    exact=exact,
                    polynomial_width=polynomial_width,
                    interval_remainder_width=remainder_width,
                    row_status=(
                        "validated" if contained else "analytic_reference_violation"
                    ),
                    native_validation_status="validated",
                    message=(
                        ""
                        if contained
                        else "analytic exact interval is not contained"
                    ),
                )
            )
    return analytic_ok


def _support(segments: Sequence[Any]) -> str:
    supports: set[tuple[int, ...]] = set()
    for segment in segments:
        for vector in (segment.tm, segment.final_tm):
            for model in vector:
                supports.update(tuple(map(int, exponent)) for exponent in model.polynomial.terms)
    return json.dumps([list(item) for item in sorted(supports)])


def run_configuration(
    spec: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    system_name = str(config["system"])
    system = spec["systems"][system_name]
    protocol = str(config["protocol"])
    h = float(config["h"])
    steps = int(config["steps"])
    settings = spec["torch"]
    if protocol == PROTOCOL_A:
        carried_representation = "none_one_segment"
        reset_policy = "not_applicable"
    elif protocol == PROTOCOL_B:
        carried_representation = "componentwise_axis_aligned_box"
        reset_policy = "endpoint_box_exact_no_inflation"
    else:
        carried_representation = "dependency_preserving_taylor_model"
        reset_policy = "none_native_dependency_carry"
    run = base_run(
        tool="torch_tm_flowpipe",
        tool_variant="complete_total_degree_order_1",
        config=config,
        local_order=1,
        local_retained_basis="complete_total_degree_1(local_time,state_generators)",
        carried_representation=carried_representation,
        reset_policy=reset_policy,
        validator="torch_native_picard_growth",
        dtype="float64",
        device="cpu",
        tool_git_sha=git_sha(REPO_ROOT),
        adapter_git_sha=git_sha(REPO_ROOT),
    )
    run["build_time_s"] = 0.0
    run["jit_compile_time_s"] = 0.0
    rows: list[dict[str, Any]] = []
    _append_initial_rows(rows, run, system)
    rhs = _rhs(system)
    kwargs = {
        "validation_mode": str(settings["validation_mode"]),
        "max_validation_attempts": int(settings["max_validation_attempts"]),
        "cutoff_threshold": settings.get("cutoff"),
        "symbolic_remainder": False,
        "max_symbolic_remainders": 0,
    }
    current_box = [
        Interval(float(lower), float(upper))
        for lower, upper in system["initial_box"]
    ]
    current_tm = TMVector.identity(current_box, order=1)
    segments: list[Any] = []
    step_times: list[float] = []
    validation_attempts = 0
    failure_message = ""
    run_started = time.perf_counter()
    for step_index in range(1, steps + 1):
        started = time.perf_counter()
        if protocol == PROTOCOL_C:
            segment = flowpipe_step_from_tm(rhs, current_tm, h, 1, **kwargs)
        else:
            segment = flowpipe_step(rhs, current_box, h, 1, **kwargs)
        elapsed = time.perf_counter() - started
        step_times.append(elapsed)
        segments.append(segment)
        validation_attempts += int(segment.validation_attempts)
        endpoint_finite = _finite_box(segment.final_tm.range_box())
        if segment.status != "validated" or not endpoint_finite:
            failure_message = segment.message or "native Torch validation failed"
            run.update(
                run_status="validation_failed",
                row_status="validation_failed",
                native_validation_status="failed",
                first_failure_time=step_index * h,
                successful_horizon=(step_index - 1) * h,
                completed_steps=step_index - 1,
                message=failure_message,
            )
            for state_index, state_name in enumerate(system["state_names"]):
                rows.append(
                    make_row(
                        run,
                        state_index=state_index,
                        state_name=state_name,
                        step_index=step_index,
                        time_value=step_index * h,
                        interval_kind="failure_marker",
                        lower="",
                        upper="",
                        row_status="validation_failed",
                        native_validation_status="failed",
                        message=failure_message,
                    )
                )
            break
        analytic_ok = _append_segment_rows(
            rows, run, system_name, system, segment, step_index, h
        )
        if not analytic_ok:
            failure_message = "analytic exact interval is not contained"
            run.update(
                run_status="analytic_reference_violation",
                row_status="analytic_reference_violation",
                native_validation_status="validated_but_reference_failed",
                first_failure_time=step_index * h,
                successful_horizon=(step_index - 1) * h,
                completed_steps=step_index - 1,
                message=failure_message,
            )
            break
        run["completed_steps"] = step_index
        run["successful_horizon"] = step_index * h
        if protocol == PROTOCOL_B:
            current_box = list(segment.final_tm.range_box())
        elif protocol == PROTOCOL_C:
            current_tm = segment.final_tm
    else:
        run.update(
            run_status="success",
            row_status="validated",
            native_validation_status="validated",
            completed_steps=steps,
            successful_horizon=float(config["horizon"]),
        )
    run["orchestration_time_s"] = time.perf_counter() - run_started
    run["first_execution_time_s"] = step_times[0] if step_times else math.nan
    run["steady_runtime_per_step_s"] = median(step_times[1:] or step_times)
    # One-step configurations need true repeated eager measurements.
    if protocol == PROTOCOL_A and run["run_status"] == "success":
        repeated: list[float] = []
        for _ in range(int(spec["steady_repetitions"])):
            started = time.perf_counter()
            repeat = flowpipe_step(rhs, current_box, h, 1, **kwargs)
            if repeat.status != "validated":
                raise RuntimeError("Torch repeated one-step timing run failed validation")
            repeated.append(time.perf_counter() - started)
        run["steady_runtime_per_step_s"] = median(repeated)
        run["timing_repetitions_s"] = repeated
    run["validation_attempts"] = validation_attempts
    run["measured_polynomial_support"] = _support(segments)
    copy_runtime_fields(run, rows)
    return rows, run


def point_evaluations(spec: Mapping[str, Any]) -> dict[str, Any]:
    values: list[dict[str, Any]] = []
    for system_name, system in spec["systems"].items():
        for point_index, point in enumerate(system["point_checks"]):
            state = [
                torch.tensor(float(value), dtype=torch.float64, device="cpu")
                for value in point
            ]
            result = evaluate_rhs(state, system)
            values.append(
                {
                    "system": system_name,
                    "point_index": point_index,
                    "point": list(map(float, point)),
                    "value": [float(item.detach().cpu()) for item in result],
                }
            )
    return {
        "tool": "torch_tm_flowpipe",
        "dtype": "float64",
        "device": "cpu",
        "values": values,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", default=str(HERE / "benchmark_spec.yaml"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--protocols", nargs="*")
    parser.add_argument("--systems", nargs="*")
    args = parser.parse_args()
    spec = load_spec(args.spec)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    runs: list[dict[str, Any]] = []
    for config in iter_configurations(
        spec,
        smoke=args.smoke,
        protocols=args.protocols,
        systems=args.systems,
    ):
        config_rows, run = run_configuration(spec, config)
        rows.extend(config_rows)
        runs.append(run)
        print(
            f"Torch {config['protocol']} {config['system']} "
            f"h={config['h']:g} T={config['horizon']:g}: "
            f"{run['completed_steps']}/{run['requested_steps']} {run['run_status']}",
            flush=True,
        )
    write_csv(output / "torch_raw_results.csv", rows, RAW_FIELDS)
    write_csv(output / "torch_runs.csv", runs, RUN_FIELDS)
    write_json(output / "torch_runs.json", runs)
    write_json(output / "torch_point_evaluations.json", point_evaluations(spec))


if __name__ == "__main__":
    main()
