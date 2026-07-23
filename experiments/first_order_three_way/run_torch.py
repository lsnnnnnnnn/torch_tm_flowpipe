#!/usr/bin/env python3
"""Run the canonical benchmark with torch_tm_flowpipe."""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any, Mapping

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
SRC_ROOT = REPO_ROOT / "src"
for candidate in (SRC_ROOT, HERE):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import torch

from torch_tm_flowpipe import Interval, TMVector, flowpipe_step, flowpipe_step_from_tm

from common import (
    configuration_timeout,
    interval_row,
    iter_configurations,
    load_spec,
    median_iqr,
    output_dir_from_args,
    raw_run_template,
    utc_timestamp,
    write_csv,
    write_json,
    evaluate_rhs,
    git_sha,
)

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)


def _ode(system_spec: Mapping[str, Any]):
    def rhs(x: TMVector, u: TMVector | None = None) -> TMVector:
        del u
        return TMVector(evaluate_rhs(list(x), system_spec))

    return rhs


def _finite_box(box: list[Interval]) -> bool:
    return all(
        math.isfinite(float(iv.lo.detach().cpu()))
        and math.isfinite(float(iv.hi.detach().cpu()))
        for iv in box
    )


def _propagate(
    system_spec: Mapping[str, Any],
    *,
    h: float,
    steps: int,
    mode: str,
    settings: Mapping[str, Any],
) -> list[Any]:
    rhs = _ode(system_spec)
    initial = [Interval(float(lo), float(hi)) for lo, hi in system_spec["initial_box"]]
    segments: list[Any] = []
    kwargs = {
        "validation_mode": str(settings["validation_mode"]),
        "max_validation_attempts": int(settings["max_validation_attempts"]),
        "cutoff_threshold": settings.get("cutoff"),
        "symbolic_remainder": False,
        "max_symbolic_remainders": 0,
    }
    if mode == "dependency_preserving":
        current = TMVector.identity(initial, order=1)
        for _ in range(steps):
            segment = flowpipe_step_from_tm(rhs, current, h, 1, **kwargs)
            segments.append(segment)
            if segment.status != "validated" or not _finite_box(segment.final_tm.range_box()):
                break
            current = segment.final_tm
        return segments

    current_box = initial
    for _ in range(steps):
        segment = flowpipe_step(rhs, current_box, h, 1, **kwargs)
        segments.append(segment)
        if segment.status != "validated" or not _finite_box(segment.final_tm.range_box()):
            break
        current_box = [iv.inflate(1.0e-9) for iv in segment.final_tm.range_box()]
    return segments


def _support(segments: list[Any]) -> dict[str, Any]:
    supports: set[tuple[int, ...]] = set()
    degree = 0
    for segment in segments:
        for tmv in (segment.tm, segment.final_tm):
            for model in tmv:
                supports.update(tuple(exp) for exp in model.polynomial.terms)
                degree = max(degree, model.polynomial.degree())
    return {
        "effective_max_degree": degree,
        "nonzero_exponent_support": [list(exp) for exp in sorted(supports)],
        "all_models_order_one": all(
            segment.order == 1
            and all(model.polynomial.degree() <= 1 for model in segment.tm)
            and all(model.polynomial.degree() <= 1 for model in segment.final_tm)
            for segment in segments
        ),
    }


def _rows_and_metadata(
    segments: list[Any],
    *,
    spec: Mapping[str, Any],
    config: Mapping[str, Any],
    protocol: str,
    mode: str,
    warmup_s: float,
    timings: list[float],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    system_name = str(config["system"])
    system = spec["systems"][system_name]
    h, horizon, steps = float(config["h"]), float(config["horizon"]), int(config["steps"])
    support = _support(segments)
    first_bad = next(
        (
            index
            for index, segment in enumerate(segments, start=1)
            if segment.status != "validated" or not _finite_box(segment.final_tm.range_box())
        ),
        None,
    )
    complete = len(segments) == steps and first_bad is None
    failure_time = "" if complete else float((first_bad or len(segments) + 1) * h)
    successful_horizon = horizon if complete else max(0.0, float((first_bad or 1) - 1) * h)
    median_s, iqr_s = median_iqr(timings)
    retained_basis = "complete_total_degree_1(local_time,initial_generators)"
    run = raw_run_template(
        tool="torch_tm_flowpipe",
        protocol=protocol,
        system=system_name,
        h=h,
        horizon=horizon,
        requested_order_label="order=1",
        retained_basis=retained_basis,
        effective_max_degree=support["effective_max_degree"],
        truncate_to_affine=True,
        nonzero_lt=False,
        dependency_mode=mode,
        symbolic_remainder_size=0,
        cutoff=spec["torch"].get("cutoff"),
        dtype="float64",
        device="cpu",
        git_commit=git_sha(REPO_ROOT),
        environment="py11",
    )
    run.update(
        status="certified_ok" if complete else "validation_failed",
        validation_status="validated" if complete else "failed",
        first_failure_time=failure_time,
        successful_horizon=successful_horizon,
        warmup_time_s=warmup_s,
        steady_runtime_median_s=median_s,
        steady_runtime_iqr_s=iqr_s,
        validation_attempts=sum(int(segment.validation_attempts) for segment in segments),
        message="" if complete else (segments[first_bad - 1].message if first_bad else "incomplete run"),
    )
    rows: list[dict[str, Any]] = []
    for state_index, (lo, hi) in enumerate(system["initial_box"]):
        rows.append(
            interval_row(
                run=run,
                state_index=state_index,
                step_index=0,
                time_value=0.0,
                interval_kind="endpoint",
                lower=float(lo),
                upper=float(hi),
            )
        )
    for step_index, segment in enumerate(segments, start=1):
        if segment.status != "validated" or not _finite_box(segment.final_tm.range_box()):
            for state_index in range(len(system["state_names"])):
                rows.append(
                    interval_row(
                        run=run,
                        state_index=state_index,
                        step_index=step_index,
                        time_value=step_index * h,
                        interval_kind="failure_marker",
                        lower="",
                        upper="",
                    )
                )
            break
        for interval_kind, box in (
            ("endpoint", segment.final_tm.range_box()),
            ("tube", segment.tm.range_box()),
        ):
            for state_index, iv in enumerate(box):
                rows.append(
                    interval_row(
                        run=run,
                        state_index=state_index,
                        step_index=step_index,
                        time_value=step_index * h,
                        interval_kind=interval_kind,
                        lower=float(iv.lo.detach().cpu()),
                        upper=float(iv.hi.detach().cpu()),
                    )
                )
    metadata = {
        **run,
        "requested_steps": steps,
        "completed_segments": sum(1 for segment in segments if segment.status == "validated"),
        "support": support,
        "validation_attempts_per_segment": [int(segment.validation_attempts) for segment in segments],
        "segment_statuses": [segment.status for segment in segments],
        "segment_messages": [segment.message for segment in segments],
        "timing_repetitions_s": timings,
        "adaptive_step": False,
        "adaptive_order": False,
        "rescue": False,
        "symbolic_remainder": False,
        "interval_semantics": {
            "endpoint": "tau=h after local-time substitution",
            "tube": "whole validated segment over tau in [0,h]",
        },
    }
    return rows, metadata


def run_configuration(
    spec: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    mode: str,
) -> tuple[list[Any], float, list[float]]:
    system = spec["systems"][config["system"]]
    started = time.perf_counter()
    segments = _propagate(
        system,
        h=float(config["h"]),
        steps=int(config["steps"]),
        mode=mode,
        settings=spec["torch"],
    )
    warmup_s = time.perf_counter() - started
    timings: list[float] = []
    for _ in range(int(spec["steady_repetitions"])):
        started = time.perf_counter()
        _propagate(
            system,
            h=float(config["h"]),
            steps=int(config["steps"]),
            mode=mode,
            settings=spec["torch"],
        )
        timings.append(time.perf_counter() - started)
    return segments, warmup_s, timings


def _failure_output(
    spec: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    mode: str,
    protocol: str,
    status: str,
    message: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    retained_basis = "complete_total_degree_1(local_time,initial_generators)"
    run = raw_run_template(
        tool="torch_tm_flowpipe",
        protocol=protocol,
        system=str(config["system"]),
        h=float(config["h"]),
        horizon=float(config["horizon"]),
        requested_order_label="order=1",
        retained_basis=retained_basis,
        effective_max_degree="",
        truncate_to_affine=True,
        nonzero_lt=False,
        dependency_mode=mode,
        symbolic_remainder_size=0,
        cutoff=spec["torch"].get("cutoff"),
        dtype="float64",
        device="cpu",
        git_commit=git_sha(REPO_ROOT),
        environment="py11",
    )
    run.update(
        status=status,
        validation_status="timeout" if status == "timeout" else "exception",
        first_failure_time=0.0,
        successful_horizon=0.0,
        message=message,
    )
    rows = [
        interval_row(
            run=run,
            state_index=state_index,
            step_index=0,
            time_value=0.0,
            interval_kind="failure_marker",
            lower="",
            upper="",
        )
        for state_index in range(
            len(spec["systems"][str(config["system"])]["state_names"])
        )
    ]
    return rows, {**run, "requested_steps": int(config["steps"])}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", default=str(HERE / "benchmark_spec.yaml"))
    parser.add_argument("--output-dir")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--systems", nargs="*")
    args = parser.parse_args()
    spec = load_spec(args.spec)
    output_dir = output_dir_from_args(args.output_dir)
    all_rows: list[dict[str, Any]] = []
    metadata_dir = output_dir / "per_run"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    for config in iter_configurations(spec, smoke=args.smoke, systems=args.systems):
        for mode, protocols in (
            ("dependency_preserving", ("native_first_order_setting", "strict_common_affine")),
            ("range_only", ("supplementary_native_representations",)),
        ):
            error: Exception | None = None
            try:
                with configuration_timeout(float(spec["timeout_s"])):
                    segments, warmup_s, timings = run_configuration(spec, config, mode=mode)
            except Exception as exc:
                error = exc
            for protocol in protocols:
                if error is None:
                    rows, metadata = _rows_and_metadata(
                        segments,
                        spec=spec,
                        config=config,
                        protocol=protocol,
                        mode=mode,
                        warmup_s=warmup_s,
                        timings=timings,
                    )
                else:
                    rows, metadata = _failure_output(
                        spec,
                        config,
                        mode=mode,
                        protocol=protocol,
                        status="timeout" if isinstance(error, TimeoutError) else "numerical_error",
                        message=f"{type(error).__name__}: {error}",
                    )
                all_rows.extend(rows)
                write_json(metadata_dir / f"{metadata['run_id']}.json", metadata)
            print(
                f"torch {mode} {config['system']} h={config['h']} T={config['horizon']} "
                + (
                    f"segments={len(segments)}/{config['steps']}"
                    if error is None
                    else f"status={metadata['status']}"
                ),
                flush=True,
            )
    suffix = "smoke" if args.smoke else "full"
    write_csv(output_dir / f"torch_raw_{suffix}.csv", all_rows)
    write_json(
        output_dir / f"torch_manifest_{suffix}.json",
        {
            "timestamp": utc_timestamp(),
            "rows": len(all_rows),
            "configurations": list(iter_configurations(spec, smoke=args.smoke, systems=args.systems)),
        },
    )
    print(output_dir)


if __name__ == "__main__":
    main()
