#!/usr/bin/env python3
"""Run Torch with raw and fixed-time-tightened endpoints kept separate."""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
for candidate in (REPO_ROOT / "src", HERE):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import torch

from torch_tm_flowpipe import Interval, TMVector, flowpipe_step, flowpipe_step_from_tm

from common import (
    PROTOCOL_BOX,
    PROTOCOL_NATIVE,
    PROTOCOL_RAW,
    PROTOCOL_STRESS,
    PROTOCOL_TUBE,
    exact_steps,
    git_sha,
    load_spec,
    make_row,
    reference_for_row,
    write_csv,
    write_json,
)

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)


def _power(value: Any, exponent: int) -> Any:
    result: Any = 1.0
    for _ in range(exponent):
        result = result * value
    return result


def _rhs(system: Mapping[str, Any]):
    def rhs(state: TMVector, control: TMVector | None = None) -> TMVector:
        del control
        outputs = []
        for polynomial in system["rhs"]:
            value: Any = 0.0
            for term in polynomial["terms"]:
                product: Any = float(term["coefficient"])
                for coordinate, exponent in zip(state, term["powers"]):
                    product = product * _power(coordinate, int(exponent))
                value = value + product
            outputs.append(value)
        return TMVector(outputs)

    return rhs


def _metrics(model: Any) -> tuple[float, float]:
    polynomial = model.polynomial.evaluate_interval(model.domain)
    return (
        float(polynomial.width().detach().cpu()),
        float(model.remainder.width().detach().cpu()),
    )


def _bounds(vector: TMVector) -> list[tuple[float, float]]:
    return [interval.to_tuple() for interval in vector.range_box()]


def _append_vector(
    rows: list[dict[str, Any]],
    *,
    spec: Mapping[str, Any],
    system_name: str,
    variant: str,
    protocol: str,
    h: float,
    horizon: float,
    step_index: int,
    vector: TMVector,
    interval_kind: str,
    local_basis: str,
    carried_representation: str,
    endpoint_tightening_applied: bool,
    endpoint_semantics: str,
    native_status: str,
    runtime_s: float,
) -> bool:
    system = spec["systems"][system_name]
    absolute_time = step_index * h
    exact_boxes = reference_for_row(
        system_name,
        interval_kind,
        absolute_time,
        h,
        system["initial_box"],
    )
    valid = True
    for state_index, (model, interval) in enumerate(zip(vector, vector.range_box())):
        lower, upper = interval.to_tuple()
        exact = None if exact_boxes is None else exact_boxes[state_index]
        contains = (
            True
            if exact is None
            else lower <= exact[0] + float(spec["containment_tolerance"])
            and upper >= exact[1] - float(spec["containment_tolerance"])
        )
        valid = valid and contains and math.isfinite(lower) and math.isfinite(upper)
        polynomial_width, remainder_width = _metrics(model)
        rows.append(
            make_row(
                tool="torch_tm_flowpipe",
                variant=variant,
                protocol=protocol,
                system=system_name,
                h=h,
                horizon=horizon,
                step_index=step_index,
                absolute_time=absolute_time,
                state_index=state_index,
                interval_kind=interval_kind,
                lower=lower,
                upper=upper,
                exact=exact,
                native_validation_status=native_status,
                analytic_reference_status=(
                    "not_available"
                    if exact is None
                    else ("passed" if contains else "failed")
                ),
                local_order=1,
                local_basis=local_basis,
                carried_representation=carried_representation,
                step_policy=f"fixed_{h:.17g}",
                cutoff="",
                remainder_overwrite_applied=False,
                endpoint_tightening_applied=endpoint_tightening_applied,
                endpoint_semantics=endpoint_semantics,
                polynomial_width=polynomial_width,
                remainder_width=remainder_width,
                build_time_s=0.0,
                warmup_time_s=runtime_s if step_index == 1 else "",
                steady_runtime_s=runtime_s,
                dtype="float64",
                device="cpu",
                repository_sha=git_sha(REPO_ROOT),
                environment="py11",
            )
        )
    return valid


def _failure_rows(
    rows: list[dict[str, Any]],
    *,
    spec: Mapping[str, Any],
    system_name: str,
    variant: str,
    protocol: str,
    h: float,
    horizon: float,
    step_index: int,
    message: str,
) -> None:
    category = (
        "nonfinite_remainder" if "finite" in message.lower() else "fixed_step_validation_failed"
    )
    for state_index, _ in enumerate(spec["systems"][system_name]["state_names"]):
        rows.append(
            make_row(
                tool="torch_tm_flowpipe",
                variant=variant,
                protocol=protocol,
                system=system_name,
                h=h,
                horizon=horizon,
                step_index=step_index,
                absolute_time=step_index * h,
                state_index=state_index,
                interval_kind="failure",
                lower="",
                upper="",
                exact=None,
                native_validation_status="failed",
                analytic_reference_status="not_checked",
                failure_category=category,
                failure_message=message,
                local_order=1,
                local_basis="complete_total_degree_1",
                remainder_overwrite_applied=False,
                endpoint_tightening_applied=False,
                endpoint_semantics="not_available",
                dtype="float64",
                device="cpu",
                repository_sha=git_sha(REPO_ROOT),
                environment="py11",
            )
        )


def run_case(
    spec: Mapping[str, Any],
    *,
    system_name: str,
    protocol: str,
    h: float,
    horizon: float,
    native_carry: str = "raw",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    settings = spec["torch"]
    system = spec["systems"][system_name]
    rhs = _rhs(system)
    steps = exact_steps(h, horizon)
    current_box = [Interval(*bounds) for bounds in system["initial_box"]]
    current_tm = TMVector.identity(current_box, order=1)
    if protocol == PROTOCOL_BOX:
        carry = "componentwise_box_from_raw_endpoint"
        variant = "torch_order1_common_box_raw"
    elif protocol in {PROTOCOL_NATIVE, PROTOCOL_STRESS}:
        carry = f"dependency_preserving_{native_carry}_endpoint_tm"
        variant = f"torch_order1_native_{native_carry}_carry"
    else:
        carry = "none_one_step"
        variant = "torch_order1"
    rows: list[dict[str, Any]] = []
    step_times: list[float] = []
    status = "success"
    message = ""
    for step_index in range(1, steps + 1):
        started = time.perf_counter()
        if protocol in {PROTOCOL_NATIVE, PROTOCOL_STRESS}:
            segment = flowpipe_step_from_tm(
                rhs,
                current_tm,
                h,
                1,
                validation_mode=str(settings["validation_mode"]),
                max_validation_attempts=int(settings["max_validation_attempts"]),
                cutoff_threshold=settings.get("cutoff"),
            )
        else:
            segment = flowpipe_step(
                rhs,
                current_box,
                h,
                1,
                validation_mode=str(settings["validation_mode"]),
                max_validation_attempts=int(settings["max_validation_attempts"]),
                cutoff_threshold=settings.get("cutoff"),
            )
        elapsed = time.perf_counter() - started
        step_times.append(elapsed)
        assert segment.endpoint_raw_tm is not None
        assert segment.endpoint_tightened_tm is not None
        if segment.status != "validated":
            status = "failed"
            message = segment.message or "Torch validation failed"
            _failure_rows(
                rows,
                spec=spec,
                system_name=system_name,
                variant=variant,
                protocol=protocol,
                h=h,
                horizon=horizon,
                step_index=step_index,
                message=message,
            )
            break
        if protocol == PROTOCOL_TUBE:
            _append_vector(
                rows,
                spec=spec,
                system_name=system_name,
                variant=variant,
                protocol=protocol,
                h=h,
                horizon=horizon,
                step_index=step_index,
                vector=segment.tm,
                interval_kind="tube",
                local_basis="complete_total_degree_1(local_time,state_generators)",
                carried_representation=carry,
                endpoint_tightening_applied=False,
                endpoint_semantics="whole_segment_tau_in_[0,h]",
                native_status="validated",
                runtime_s=elapsed,
            )
        else:
            _append_vector(
                rows,
                spec=spec,
                system_name=system_name,
                variant=variant,
                protocol=protocol,
                h=h,
                horizon=horizon,
                step_index=step_index,
                vector=segment.endpoint_raw_tm,
                interval_kind="endpoint_raw",
                local_basis="complete_total_degree_1(state_generators)",
                carried_representation=carry,
                endpoint_tightening_applied=False,
                endpoint_semantics="raw_substitution_tau_equals_h",
                native_status="validated",
                runtime_s=elapsed,
            )
            # Always retain the supplemental tightened object in the endpoint
            # audit.  Collector rankings explicitly exclude this interval kind.
            _append_vector(
                rows,
                spec=spec,
                system_name=system_name,
                variant=variant,
                protocol=protocol,
                h=h,
                horizon=horizon,
                step_index=step_index,
                vector=segment.endpoint_tightened_tm,
                interval_kind="endpoint_tightened",
                local_basis="complete_total_degree_1(state_generators)",
                carried_representation=carry,
                endpoint_tightening_applied=segment.endpoint_tightening_applied,
                endpoint_semantics=segment.endpoint_semantics,
                native_status="validated",
                runtime_s=elapsed,
            )
        raw_box = _bounds(segment.endpoint_raw_tm)
        if protocol == PROTOCOL_BOX:
            current_box = [Interval(lower, upper) for lower, upper in raw_box]
        elif protocol in {PROTOCOL_NATIVE, PROTOCOL_STRESS}:
            current_tm = (
                segment.endpoint_raw_tm
                if native_carry == "raw"
                else segment.endpoint_tightened_tm
            )
    return rows, {
        "tool": "torch_tm_flowpipe",
        "variant": variant,
        "protocol": protocol,
        "system": system_name,
        "h": h,
        "requested_horizon": horizon,
        "requested_steps": steps,
        "completed_steps": len(
            {
                row["step_index"]
                for row in rows
                if row["interval_kind"] in {"tube", "endpoint_raw"}
            }
        ),
        "status": status,
        "message": message,
        "step_times_s": step_times,
        "endpoint_carry": native_carry,
    }


def _cases(spec: Mapping[str, Any], smoke: bool):
    for system_name, benchmark in spec["benchmarks"].items():
        one_steps = [float(benchmark["smoke"]["h"])] if smoke else [
            float(value) for value in benchmark["one_step_h"]
        ]
        for h in one_steps:
            yield system_name, PROTOCOL_TUBE, h, h, "raw"
            yield system_name, PROTOCOL_RAW, h, h, "raw"
        multi = (
            [benchmark["smoke"]]
            if smoke
            else benchmark["multi_step"]
        )
        for config in multi:
            h, horizon = float(config["h"]), float(config["horizon"])
            yield system_name, PROTOCOL_BOX, h, horizon, "raw"
            yield system_name, PROTOCOL_NATIVE, h, horizon, "raw"
            yield system_name, PROTOCOL_NATIVE, h, horizon, "tightened"
            yield system_name, PROTOCOL_STRESS, h, horizon, "raw"
    if not smoke:
        # Minimal historical-anomaly horizons not otherwise present in the
        # corrected benchmark matrix.
        yield "riccati", PROTOCOL_BOX, 0.01, 0.1, "raw"
        yield "harmonic", PROTOCOL_BOX, 0.01, 1.0, "raw"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", default=str(HERE / "benchmark_spec.yaml"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    spec = load_spec(args.spec)
    output = Path(args.output_dir).resolve()
    rows: list[dict[str, Any]] = []
    runs: list[dict[str, Any]] = []
    for system, protocol, h, horizon, carry in _cases(spec, args.smoke):
        case_rows, run = run_case(
            spec,
            system_name=system,
            protocol=protocol,
            h=h,
            horizon=horizon,
            native_carry=carry,
        )
        rows.extend(case_rows)
        runs.append(run)
        print(
            f"Torch {run['variant']} {protocol} {system} h={h:g} "
            f"T={horizon:g}: {run['status']}",
            flush=True,
        )
    write_csv(output / "torch_endpoint_audit.csv", rows)
    write_json(output / "torch_runs.json", runs)


if __name__ == "__main__":
    main()
