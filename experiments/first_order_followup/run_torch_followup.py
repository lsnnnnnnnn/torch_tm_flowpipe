#!/usr/bin/env python3
"""Run Torch dependency forensics and the B1/B_DR/B2 ablation."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
SRC_ROOT = REPO_ROOT / "src"
BASELINE_EXPERIMENT = HERE.parent / "first_order_three_way"
for path in (HERE, SRC_ROOT, BASELINE_EXPERIMENT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import torch

from common import evaluate_rhs, exact_endpoint, git_sha, load_spec
from torch_basis import (
    affine_reset,
    diagnose_tm,
    finite_basis_step_from_tm,
    harmonic_exact_affine,
    normalized_initial_tm,
    retained_dictionary,
)
from torch_tm_flowpipe import Interval, TMVector, flowpipe_step, flowpipe_step_from_tm

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

RAW_FIELDS = [
    "tool", "protocol", "system", "mode", "basis", "h", "horizon", "state_index",
    "step_index", "time", "interval_kind", "lower", "upper", "width",
    "local_construction_basis", "local_construction_order", "carried_basis",
    "carried_max_degree", "projection_method", "reset_method", "validator",
    "numerical_backend", "native_validation_passed",
    "exact_reference_contained", "sampled_trajectory_contained",
    "directed_rounding_or_mpfr", "floating_point_enclosure_candidate",
    "validation_failed", "validation_attempts", "retained_coefficients",
    "discarded_candidates", "python_orchestration_time_s", "compile_time_s",
    "first_call_time_s", "steady_step_time_s", "number_of_steps",
    "number_of_retained_coefficients", "number_of_discarded_candidates",
    "successful_horizon", "message",
]


def _rhs(system: Mapping[str, Any]):
    def rhs(state: TMVector, control: TMVector | None = None) -> TMVector:
        del control
        return TMVector(evaluate_rhs(list(state), system))

    return rhs


def _float(value: Any) -> float:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    return float(value)


def _box(tm: TMVector) -> list[tuple[float, float]]:
    return [interval.to_tuple() for interval in tm.range_box()]


def _contains_exact(
    system_name: str,
    system: Mapping[str, Any],
    time_value: float,
    box: Sequence[tuple[float, float]],
    *,
    tolerance: float = 1e-10,
) -> bool | None:
    exact = exact_endpoint(system_name, time_value, system["initial_box"])
    if exact is None:
        return None
    return all(
        lower <= exact_lower + tolerance and upper >= exact_upper - tolerance
        for (lower, upper), (exact_lower, exact_upper) in zip(box, exact)
    )


def _row(
    *,
    metadata: Mapping[str, Any],
    state_index: int,
    step_index: int,
    time_value: float,
    interval_kind: str,
    bounds: tuple[float, float],
) -> dict[str, Any]:
    lower, upper = bounds
    row = {field: metadata.get(field, "") for field in RAW_FIELDS}
    row.update(
        state_index=state_index,
        step_index=step_index,
        time=time_value,
        interval_kind=interval_kind,
        lower=lower,
        upper=upper,
        width=upper - lower,
    )
    return row


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=RAW_FIELDS, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _diagnostic_record(
    *,
    protocol: str,
    system: str,
    mode: str,
    basis: str,
    step: int,
    h: float,
    horizon: float,
    segment: Any,
    reset_stats: Mapping[str, Any] | None,
    discarded: Sequence[Any],
    validation_trace: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    endpoint = diagnose_tm(segment.final_tm, tau_index=None)
    tube = diagnose_tm(segment.tm, tau_index=segment.tau_index)
    generator = torch.zeros(
        (len(endpoint), segment.final_tm.n_vars), dtype=torch.float64
    )
    for state, model in enumerate(segment.final_tm):
        for exponent, coefficient in model.polynomial.terms.items():
            if sum(exponent) == 1:
                variable = next(
                    index for index, power in enumerate(exponent) if power
                )
                generator[state, variable] = coefficient * model.domain[
                    variable
                ].radius()
    singular_values = torch.linalg.svdvals(generator)
    positive = singular_values[singular_values > 0]
    condition_surrogate = (
        float("inf")
        if positive.numel() < min(generator.shape)
        else float((positive.max() / positive.min()).detach().cpu())
    )
    return {
        "protocol": protocol,
        "system": system,
        "mode": mode,
        "basis": basis,
        "horizon": horizon,
        "step_index": step,
        "time": step * h,
        "h": h,
        "status": segment.status,
        "validation_attempts": segment.validation_attempts,
        "message": segment.message,
        "tau_index": segment.tau_index,
        "local_time_added": segment.tau_index is not None,
        "local_time_removed_from_final": (
            segment.tau_index is not None
            and segment.final_tm.n_vars + 1 == segment.tm.n_vars
        ),
        "endpoint": endpoint,
        "tube": tube,
        "candidate_remainder_before_validation": (
            (segment.selective_term_stats or {}).get(
                "candidate_remainders_before_validation",
                {
                    key: value
                    for key, value in (validation_trace[0] if validation_trace else {}).items()
                    if key.startswith("remainder_")
                },
            )
        ),
        "validated_remainder": (segment.selective_term_stats or {}).get(
            "validated_remainders",
            {
                "segment": [model.remainder.to_tuple() for model in segment.tm],
                "last_validation_attempt": {
                    key: value
                    for key, value in (
                        validation_trace[-1] if validation_trace else {}
                    ).items()
                    if key.startswith("remainder_")
                },
            },
        ),
        "validation_trace": list(validation_trace),
        "discarded_terms": [
            {
                "stage": item.stage,
                "iteration": item.iteration,
                "state_index": item.state_index,
                "exponent": list(item.exponent),
                "coefficient": item.coefficient,
                "range_lower": item.range_lower,
                "range_upper": item.range_upper,
                "range_width": item.range_width,
            }
            for item in discarded
        ],
        "discarded_range_width_sum": sum(item.range_width for item in discarded),
        "affine_generator_condition_surrogate": condition_surrogate,
        "reset": dict(reset_stats or {}),
    }


def run_basis(
    *,
    system_name: str,
    system: Mapping[str, Any],
    basis: str,
    h: float,
    horizon: float,
    carry_policy: str = "affine_box",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    steps = round(horizon / h)
    rhs = _rhs(system)
    current = normalized_initial_tm(system["initial_box"], order=2)
    rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    step_times: list[float] = []
    exact_checks = 0
    exact_violations = 0
    discarded_count = 0
    retained_count = 0
    first_call = math.nan
    run_started = time.perf_counter()
    completed = 0
    failure_message = ""
    if carry_policy not in {"affine_box", "complete"}:
        raise ValueError("carry_policy must be 'affine_box' or 'complete'")
    if carry_policy == "complete" and basis != "B2":
        raise ValueError("complete carry is only defined for B2")
    protocol = (
        "complete_degree_two_reference"
        if carry_policy == "complete"
        else ("matched_affine_carry" if basis == "B1" else "matched_basis_ablation")
    )
    metadata = {
        "tool": "torch_tm_flowpipe",
        "protocol": protocol,
        "system": system_name,
        "mode": (
            "finite_dictionary_complete_carry"
            if carry_policy == "complete"
            else "finite_dictionary_affine_box_reset"
        ),
        "basis": basis,
        "h": h,
        "horizon": horizon,
        "local_construction_basis": basis,
        "local_construction_order": 2,
        "carried_basis": (
            "complete_total_degree_2"
            if carry_policy == "complete"
            else "constant+affine_state_generators+independent_interval"
        ),
        "carried_max_degree": 2 if carry_policy == "complete" else 1,
        "projection_method": "termwise_interval_to_independent_remainder",
        "reset_method": (
            "none_dependency_preserving"
            if carry_policy == "complete"
            else "affine_recenter_rescale_box"
        ),
        "validator": "torch_picard_growth",
        "numerical_backend": "torch_float64_cpu",
        "directed_rounding_or_mpfr": "torch_nextafter_outward",
        "floating_point_enclosure_candidate": True,
        "number_of_steps": steps,
        "compile_time_s": 0.0,
    }
    for state_index, bounds in enumerate(system["initial_box"]):
        rows.append(
            _row(
                metadata=metadata,
                state_index=state_index,
                step_index=0,
                time_value=0.0,
                interval_kind="endpoint",
                bounds=tuple(map(float, bounds)),
            )
        )
    for step in range(1, steps + 1):
        started = time.perf_counter()
        validation_trace: list[dict[str, Any]] = []
        segment, discarded = finite_basis_step_from_tm(
            rhs,
            current,
            h,
            basis,
            picard_iterations=2,
            max_validation_attempts=20,
            diagnostics=validation_trace,
        )
        elapsed = time.perf_counter() - started
        if step == 1:
            first_call = elapsed
        step_times.append(elapsed)
        discarded_count += len(discarded)
        retained_count += sum(len(model.polynomial.terms) for model in segment.tm)
        if segment.status != "validated" or not all(
            interval.is_finite() for interval in segment.final_tm.range_box()
        ):
            failure_message = segment.message or "native validation failed"
            diagnostics.append(
                _diagnostic_record(
                    protocol=protocol,
                    system=system_name,
                    mode=metadata["mode"],
                    basis=basis,
                    step=step,
                    h=h,
                    horizon=horizon,
                    segment=segment,
                    reset_stats=None,
                    discarded=discarded,
                    validation_trace=validation_trace,
                )
            )
            break
        endpoint_box = _box(segment.final_tm)
        exact_ok = _contains_exact(system_name, system, step * h, endpoint_box)
        if exact_ok is not None:
            exact_checks += len(endpoint_box)
            if not exact_ok:
                exact_violations += 1
                failure_message = "analytic exact endpoint reference violated"
        for interval_kind, tm in (("endpoint", segment.final_tm), ("tube", segment.tm)):
            for state_index, bounds in enumerate(_box(tm)):
                row_metadata = {
                    **metadata,
                    "native_validation_passed": True,
                    "exact_reference_contained": (
                        "" if exact_ok is None else bool(exact_ok)
                    ),
                    "sampled_trajectory_contained": "",
                    "validation_failed": False,
                    "validation_attempts": segment.validation_attempts,
                    "retained_coefficients": sum(
                        len(model.polynomial.terms) for model in segment.tm
                    ),
                    "discarded_candidates": len(discarded),
                    "message": failure_message,
                }
                rows.append(
                    _row(
                        metadata=row_metadata,
                        state_index=state_index,
                        step_index=step,
                        time_value=step * h,
                        interval_kind=interval_kind,
                        bounds=bounds,
                    )
                )
        if carry_policy == "complete":
            current = segment.final_tm
            reset_stats = {"method": "none_dependency_preserving"}
        else:
            reset_box = [
                interval.to_tuple() for interval in segment.final_tm.range_box()
            ]
            current = normalized_initial_tm(reset_box, order=2)
            reset_stats = {
                "method": "box",
                "output_box": reset_box,
                "generator_condition_number": (
                    max((upper - lower) for lower, upper in reset_box)
                    / min((upper - lower) for lower, upper in reset_box)
                    if all(upper > lower for lower, upper in reset_box)
                    else math.inf
                ),
            }
        diagnostics.append(
            _diagnostic_record(
                protocol=protocol,
                system=system_name,
                mode=metadata["mode"],
                basis=basis,
                step=step,
                h=h,
                horizon=horizon,
                segment=segment,
                reset_stats=reset_stats,
                discarded=discarded,
                validation_trace=validation_trace,
            )
        )
        completed = step
        if exact_ok is False:
            break
    orchestration = time.perf_counter() - run_started
    steady = statistics.median(step_times[1:] or step_times) if step_times else math.nan
    summary = {
        **metadata,
        "h": h,
        "horizon": horizon,
        "requested_steps": steps,
        "completed_steps": completed,
        "native_validation_passed": completed == steps,
        "exact_reference_checks": exact_checks,
        "exact_reference_violations": exact_violations,
        "exact_reference_contained": exact_violations == 0 if exact_checks else None,
        "sampled_trajectory_contained": None,
        "validation_failed": completed != steps or exact_violations > 0,
        "first_failure_time": "" if completed == steps else (completed + 1) * h,
        "successful_horizon": completed * h,
        "retained_coefficients_total": retained_count,
        "discarded_candidates_total": discarded_count,
        "retained_dictionary": [
            list(exponent)
            for exponent in retained_dictionary(basis, len(system["state_names"]) + 1, tau_index=len(system["state_names"]))
        ],
        "python_orchestration_time_s": orchestration,
        "first_call_time_s": first_call,
        "steady_step_time_s": steady,
        "message": failure_message,
    }
    for row in rows:
        row.update(
            python_orchestration_time_s=orchestration,
            first_call_time_s=first_call,
            steady_step_time_s=steady,
            number_of_retained_coefficients=retained_count,
            number_of_discarded_candidates=discarded_count,
            successful_horizon=completed * h,
        )
    return rows, diagnostics, summary


def run_harmonic_dependency_audit(
    *,
    system: Mapping[str, Any],
    h: float = 0.01,
    horizon: float = 10.0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    steps = round(horizon / h)
    rhs = _rhs(system)
    all_rows: list[dict[str, Any]] = []
    all_diagnostics: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for mode in ("dependency_preserving", "range_only", "affine_box", "qr"):
        mode_row_start = len(all_rows)
        current = (
            TMVector.identity(
                [Interval(*bounds) for bounds in system["initial_box"]], order=1
            )
            if mode in {"dependency_preserving", "range_only"}
            else normalized_initial_tm(system["initial_box"], order=1)
        )
        current_box = [Interval(*bounds) for bounds in system["initial_box"]]
        completed = 0
        exact_violations = 0
        step_times: list[float] = []
        first_call = math.nan
        for step in range(1, steps + 1):
            started = time.perf_counter()
            validation_trace: list[dict[str, Any]] = []
            if mode == "range_only":
                segment = flowpipe_step(
                    rhs,
                    current_box,
                    h,
                    1,
                    diagnostics=validation_trace,
                    diagnostics_segment_index=step,
                    diagnostics_context={"mode": mode},
                )
            else:
                segment = flowpipe_step_from_tm(
                    rhs,
                    current,
                    h,
                    1,
                    diagnostics=validation_trace,
                    diagnostics_segment_index=step,
                    diagnostics_context={"mode": mode},
                )
            elapsed = time.perf_counter() - started
            if step == 1:
                first_call = elapsed
            step_times.append(elapsed)
            if segment.status != "validated":
                break
            endpoint_box = _box(segment.final_tm)
            exact_ok = bool(
                _contains_exact("harmonic", system, step * h, endpoint_box)
            )
            if not exact_ok:
                exact_violations += 1
            metadata = {
                "tool": "torch_tm_flowpipe",
                "protocol": "torch_dependency_forensics",
                "system": "harmonic",
                "mode": mode,
                "basis": "B1",
                "h": h,
                "horizon": horizon,
                "local_construction_basis": "complete_total_degree_1",
                "local_construction_order": 1,
                "carried_basis": "affine",
                "carried_max_degree": 1,
                "projection_method": (
                    "none" if mode == "dependency_preserving"
                    else "endpoint_box_to_affine_generators"
                ),
                "reset_method": mode,
                "validator": "torch_picard_growth",
                "numerical_backend": "torch_float64_cpu",
                "native_validation_passed": True,
                "exact_reference_contained": exact_ok,
                "sampled_trajectory_contained": "",
                "directed_rounding_or_mpfr": "torch_nextafter_outward",
                "floating_point_enclosure_candidate": True,
                "validation_failed": not exact_ok,
                "validation_attempts": segment.validation_attempts,
                "retained_coefficients": sum(
                    len(model.polynomial.terms) for model in segment.tm
                ),
                "discarded_candidates": "",
                "compile_time_s": 0.0,
                "number_of_steps": steps,
                "message": "" if exact_ok else "analytic exact endpoint reference violated",
            }
            for interval_kind, tm in (("endpoint", segment.final_tm), ("tube", segment.tm)):
                for state_index, bounds in enumerate(_box(tm)):
                    all_rows.append(
                        _row(
                            metadata=metadata,
                            state_index=state_index,
                            step_index=step,
                            time_value=step * h,
                            interval_kind=interval_kind,
                            bounds=bounds,
                        )
                    )
            reset_stats = None
            if mode == "dependency_preserving":
                current = segment.final_tm
            elif mode == "range_only":
                current_box = [
                    interval.inflate(1e-9)
                    for interval in segment.final_tm.range_box()
                ]
            else:
                current, reset_stats = affine_reset(
                    segment.final_tm, method="box" if mode == "affine_box" else "qr"
                )
            all_diagnostics.append(
                _diagnostic_record(
                    protocol="torch_dependency_forensics",
                    system="harmonic",
                    mode=mode,
                    basis="B1",
                    step=step,
                    h=h,
                    horizon=horizon,
                    segment=segment,
                    reset_stats=reset_stats,
                    discarded=[],
                    validation_trace=validation_trace,
                )
            )
            completed = step
            if not exact_ok:
                break
        mode_summary = {
                "tool": "torch_tm_flowpipe",
                "protocol": "torch_dependency_forensics",
                "system": "harmonic",
                "mode": mode,
                "h": h,
                "horizon": horizon,
                "requested_steps": steps,
                "completed_steps": completed,
                "successful_horizon": completed * h,
                "native_validation_passed": completed == steps,
                "exact_reference_checks": completed * 2,
                "exact_reference_violations": exact_violations,
                "first_call_time_s": first_call,
                "steady_step_time_s": (
                    statistics.median(step_times[1:] or step_times)
                    if step_times else math.nan
                ),
            }
        summaries.append(mode_summary)
        for row in all_rows[mode_row_start:]:
            row.update(
                python_orchestration_time_s=sum(step_times),
                first_call_time_s=first_call,
                steady_step_time_s=mode_summary["steady_step_time_s"],
                number_of_retained_coefficients="",
                number_of_discarded_candidates="",
                successful_horizon=completed * h,
            )

    for step in range(1, steps + 1):
        oracle = harmonic_exact_affine(system["initial_box"], step * h)
        for state_index, bounds in enumerate(_box(oracle)):
            all_rows.append(
                _row(
                    metadata={
                        "tool": "analytic_oracle",
                        "protocol": "torch_dependency_forensics",
                        "system": "harmonic",
                        "mode": "exact_rotation",
                        "basis": "affine_exact",
                        "h": h,
                        "horizon": horizon,
                        "local_construction_basis": "rotation_matrix",
                        "local_construction_order": "exact",
                        "carried_basis": "affine",
                        "carried_max_degree": 1,
                        "projection_method": "none",
                        "reset_method": "none",
                        "validator": "analytic",
                        "numerical_backend": "torch_float64_cpu",
                        "native_validation_passed": True,
                        "exact_reference_contained": True,
                        "directed_rounding_or_mpfr": "not_applicable",
                        "floating_point_enclosure_candidate": False,
                        "validation_failed": False,
                        "number_of_steps": steps,
                    },
                    state_index=state_index,
                    step_index=step,
                    time_value=step * h,
                    interval_kind="endpoint",
                    bounds=bounds,
                )
            )
    summaries.append(
        {
            "tool": "analytic_oracle",
            "protocol": "torch_dependency_forensics",
            "system": "harmonic",
            "mode": "exact_rotation",
            "h": h,
            "horizon": horizon,
            "requested_steps": steps,
            "completed_steps": steps,
            "successful_horizon": horizon,
            "native_validation_passed": True,
            "exact_reference_checks": steps * 2,
            "exact_reference_violations": 0,
        }
    )
    return all_rows, all_diagnostics, summaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    spec = load_spec(HERE / "benchmark_spec.yaml")
    rows, diagnostics, summaries = run_harmonic_dependency_audit(
        system=spec["systems"]["harmonic"],
        horizon=0.1 if args.smoke else 10.0,
    )
    basis_configs = {
        "riccati": (0.01, 0.1 if args.smoke else 1.0),
        "harmonic": (0.01, 0.1 if args.smoke else 10.0),
        "van_der_pol": (0.005, 0.02 if args.smoke else 2.0),
    }
    for system_name, (h, horizon) in basis_configs.items():
        for basis in ("B1", "B_DR", "B2"):
            basis_rows, basis_diagnostics, summary = run_basis(
                system_name=system_name,
                system=spec["systems"][system_name],
                basis=basis,
                h=h,
                horizon=horizon,
            )
            rows.extend(basis_rows)
            diagnostics.extend(basis_diagnostics)
            summaries.append(summary)
            print(
                f"Torch {system_name} {basis}: "
                f"{summary['completed_steps']}/{summary['requested_steps']} steps",
                flush=True,
            )
        reference_rows, reference_diagnostics, reference_summary = run_basis(
            system_name=system_name,
            system=spec["systems"][system_name],
            basis="B2",
            h=h,
            horizon=horizon,
            carry_policy="complete",
        )
        rows.extend(reference_rows)
        diagnostics.extend(reference_diagnostics)
        summaries.append(reference_summary)
        print(
            f"Torch {system_name} complete-degree-two reference: "
            f"{reference_summary['completed_steps']}/"
            f"{reference_summary['requested_steps']} steps",
            flush=True,
        )
    _write_csv(output / "torch_raw_results.csv", rows)
    _write_json(output / "torch_diagnostics.json", diagnostics)
    _write_json(
        output / "torch_summary.json",
        {
            "git_commit": git_sha(REPO_ROOT),
            "dtype": "float64",
            "device": "cpu",
            "summaries": summaries,
        },
    )


if __name__ == "__main__":
    main()
