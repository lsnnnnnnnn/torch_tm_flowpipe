#!/usr/bin/env python3
"""Run the experimental strict-affine DiffReach protocol."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Mapping

HERE = Path(__file__).resolve().parent
BASELINE_EXPERIMENT = HERE.parent / "first_order_three_way"
DIFFREACH_ROOT = Path("/srv/local/shengenli/DiffReach")
for path in (HERE, BASELINE_EXPERIMENT, DIFFREACH_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp
import numpy as np

from common import exact_endpoint, git_sha, load_spec
from diffreach_adapter import StrictAffineDiffReachPlantCore
from run_diffreach import DiffReachPlantCore, _rhs
import src.settings as dr_settings

FIELDS = [
    "tool", "protocol", "system", "mode", "basis", "h", "horizon",
    "state_index", "step_index", "time",
    "interval_kind", "lower", "upper", "width", "local_construction_basis",
    "local_construction_order", "carried_basis", "carried_max_degree",
    "projection_method", "reset_method", "validator", "numerical_backend",
    "native_validation_passed", "exact_reference_contained",
    "sampled_trajectory_contained", "directed_rounding_or_mpfr",
    "floating_point_enclosure_candidate", "validation_failed",
    "python_orchestration_time_s", "compile_time_s", "first_call_time_s",
    "steady_step_time_s", "number_of_steps", "number_of_retained_coefficients",
    "number_of_discarded_candidates", "successful_horizon", "message",
]


def _sync(tree: Any) -> Any:
    return jax.tree.map(
        lambda value: value.block_until_ready()
        if hasattr(value, "block_until_ready")
        else value,
        tree,
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=FIELDS, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def run_configuration(
    *,
    spec: Mapping[str, Any],
    system_name: str,
    h: float,
    horizon: float,
    protocol: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    system = spec["systems"][system_name]
    steps = round(horizon / h)
    old_config = copy.deepcopy(dr_settings.CONFIG)
    started = time.perf_counter()
    try:
        dr_settings.update_config(
            {
                # The local polynomial kernel retains its stock restricted
                # quasi-quadratic Lt basis.  Projection happens explicitly only
                # after native Picard validation.
                "TRUNCATE_TO_AFFINE": False,
                "FP64_IN_CROWN": True,
                "BOUND_TIME_STEP": True,
                "DEBUG_LOG": False,
            }
        )
        core_class = (
            StrictAffineDiffReachPlantCore
            if protocol == "matched_affine_carry"
            else DiffReachPlantCore
        )
        core = core_class(
            _rhs(system),
            dimension=len(system["state_names"]),
            h=h,
            init_remainder=float(spec["diffreach"]["init_remainder"]),
            frr_rounds=int(spec["diffreach"]["frr_rounds"]),
            frr_stop_ratio=float(spec["diffreach"]["frr_stop_ratio"]),
            symbolic_window=int(spec["diffreach"]["symbolic_remainder_window"]),
        )
        lower = jnp.asarray(
            [[bounds[0] for bounds in system["initial_box"]]], dtype=jnp.float64
        )
        upper = jnp.asarray(
            [[bounds[1] for bounds in system["initial_box"]]], dtype=jnp.float64
        )
        compiled = jax.jit(lambda lo, hi: core.verify(lo, hi, steps))
        first_started = time.perf_counter()
        result = _sync(compiled(lower, upper))
        first_call_s = time.perf_counter() - first_started
        steady_samples = []
        for _ in range(5):
            sample_started = time.perf_counter()
            _sync(compiled(lower, upper))
            steady_samples.append(time.perf_counter() - sample_started)
    finally:
        dr_settings.CONFIG.clear()
        dr_settings.CONFIG.update(old_config)
    orchestration_s = time.perf_counter() - started

    times, endpoint_lo, endpoint_hi, tube_lo, tube_hi, final_tm, contraction = result
    times = np.asarray(times)
    endpoint_lo = np.asarray(endpoint_lo[0])
    endpoint_hi = np.asarray(endpoint_hi[0])
    tube_lo = np.asarray(tube_lo[0])
    tube_hi = np.asarray(tube_hi[0])
    contraction = np.asarray(contraction)
    contraction_by_step = np.all(
        contraction, axis=tuple(range(1, contraction.ndim))
    )
    finite_by_step = np.all(np.isfinite(endpoint_lo[1:]), axis=1) & np.all(
        np.isfinite(endpoint_hi[1:]), axis=1
    )
    valid_by_step = contraction_by_step & finite_by_step
    bad = np.flatnonzero(~valid_by_step)
    completed = int(bad[0]) if bad.size else steps
    exact_checks = 0
    exact_violations = 0
    rows: list[dict[str, Any]] = []
    metadata = {
        "tool": (
            "diffreach_experimental_strict_affine"
            if protocol == "matched_affine_carry"
            else "diffreach_restricted_quasiquadratic"
        ),
        "protocol": protocol,
        "system": system_name,
        "mode": (
            "strict_affine_projection"
            if protocol == "matched_affine_carry"
            else "restricted_quasiquadratic_native_carry"
        ),
        "basis": (
            "B1_carry"
            if protocol == "matched_affine_carry"
            else "restricted_quasiquadratic_not_complete_degree_2"
        ),
        "h": h,
        "horizon": horizon,
        "local_construction_basis": "restricted_quasi_quadratic_{1,z,t^2,t*z}",
        "local_construction_order": "restricted_quasi_quadratic",
        "carried_basis": (
            "constant+affine_state_generators+independent_interval"
            if protocol == "matched_affine_carry"
            else "restricted_{1,z,t^2,t*z}"
        ),
        "carried_max_degree": (
            1 if protocol == "matched_affine_carry" else "restricted_quasiquadratic"
        ),
        "projection_method": (
            "Lt_termwise_range_midpoint_to_constant_residual_to_remainder"
            if protocol == "matched_affine_carry" else "none"
        ),
        "reset_method": "stock_symbolic_linear_normalization",
        "validator": "DiffReach_remainder_picard",
        "numerical_backend": f"jax_{jax.default_backend()}_float64",
        "directed_rounding_or_mpfr": "jax_nextafter_projection_only",
        "floating_point_enclosure_candidate": True,
        "python_orchestration_time_s": orchestration_s,
        "compile_time_s": first_call_s,
        "first_call_time_s": first_call_s,
        "steady_step_time_s": statistics.median(steady_samples) / steps,
        "number_of_steps": steps,
        "number_of_retained_coefficients": int(
            np.count_nonzero(np.asarray(final_tm.P.c))
            + np.count_nonzero(np.asarray(final_tm.P.L))
        ),
        "number_of_discarded_candidates": int(
            steps * len(system["state_names"]) * (len(system["state_names"]) + 1)
        ),
        "successful_horizon": completed * h,
        "message": "" if completed == steps else "native Picard contraction failed",
    }
    for step in range(completed + 1):
        endpoint_box = list(
            zip(endpoint_lo[step].tolist(), endpoint_hi[step].tolist())
        )
        exact = exact_endpoint(system_name, float(times[step]), system["initial_box"])
        exact_ok: bool | str = ""
        if exact is not None:
            exact_checks += len(exact)
            exact_ok = all(
                lower <= exact_lower + 1e-10 and upper >= exact_upper - 1e-10
                for (lower, upper), (exact_lower, exact_upper)
                in zip(endpoint_box, exact)
            )
            if not exact_ok:
                exact_violations += 1
        for state, (lower_bound, upper_bound) in enumerate(endpoint_box):
            rows.append(
                {
                    **metadata,
                    "state_index": state,
                    "step_index": step,
                    "time": float(times[step]),
                    "interval_kind": "endpoint",
                    "lower": lower_bound,
                    "upper": upper_bound,
                    "width": upper_bound - lower_bound,
                    "native_validation_passed": bool(
                        step == 0 or valid_by_step[step - 1]
                    ),
                    "exact_reference_contained": exact_ok,
                    "sampled_trajectory_contained": "",
                    "validation_failed": exact_ok is False,
                }
            )
        if step > 0:
            for state in range(len(system["state_names"])):
                lower_bound = float(tube_lo[step - 1, state])
                upper_bound = float(tube_hi[step - 1, state])
                rows.append(
                    {
                        **metadata,
                        "state_index": state,
                        "step_index": step,
                        "time": float(times[step]),
                        "interval_kind": "tube",
                        "lower": lower_bound,
                        "upper": upper_bound,
                        "width": upper_bound - lower_bound,
                        "native_validation_passed": bool(valid_by_step[step - 1]),
                        "exact_reference_contained": "",
                        "sampled_trajectory_contained": "",
                        "validation_failed": False,
                    }
                )
    summary = {
        **metadata,
        "h": h,
        "horizon": horizon,
        "requested_steps": steps,
        "completed_steps": completed,
        "native_validation_passed": completed == steps,
        "exact_reference_checks": exact_checks,
        "exact_reference_violations": exact_violations,
        "exact_reference_contained": (
            exact_violations == 0 if exact_checks else None
        ),
        "zero_final_Lt_support": not bool(np.any(np.asarray(final_tm.P.Lt))),
        "is_complete_total_degree_2": False,
        "steady_call_samples_s": steady_samples,
    }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    spec = load_spec(HERE / "benchmark_spec.yaml")
    configs = {
        "riccati": (0.01, 0.1 if args.smoke else 1.0),
        "harmonic": (0.01, 0.1 if args.smoke else 10.0),
        "van_der_pol": (0.005, 0.02 if args.smoke else 2.0),
    }
    all_rows = []
    summaries = []
    for protocol in ("matched_affine_carry", "complete_degree_two_reference"):
        for system_name, (h, horizon) in configs.items():
            rows, summary = run_configuration(
                spec=spec,
                system_name=system_name,
                h=h,
                horizon=horizon,
                protocol=protocol,
            )
            all_rows.extend(rows)
            summaries.append(summary)
            print(
                f"DiffReach {protocol} {system_name}: "
                f"{summary['completed_steps']}/{summary['requested_steps']} steps",
                flush=True,
            )
    _write_csv(output / "diffreach_raw_results.csv", all_rows)
    (output / "diffreach_summary.json").write_text(
        json.dumps(
            {
                "diffreach_git_commit": git_sha(DIFFREACH_ROOT),
                "summaries": summaries,
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
