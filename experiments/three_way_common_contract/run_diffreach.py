#!/usr/bin/env python3
"""Run the real upstream DiffReach plant reachability implementation."""
from __future__ import annotations

import argparse
import copy
import inspect
import json
import math
import sys
import time
import types
from pathlib import Path
from typing import Any, Mapping, Sequence

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DIFFREACH_ROOT = Path("/srv/local/shengenli/DiffReach")
for candidate in (HERE, DIFFREACH_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp
import numpy as np

# The analytic plant class imports optional neural-bound modules at module load
# time even though the plant path never calls them.  Keep those names fail-fast
# and otherwise import the real upstream reachability module unchanged.
OPTIONAL_IMPORT_SHIM_USED = False
try:
    import jax_verify  # type: ignore  # noqa: F401
except ModuleNotFoundError:
    OPTIONAL_IMPORT_SHIM_USED = True
    jax_verify_stub = types.ModuleType("jax_verify")

    class _UnavailableIntervalBound:
        def __init__(self, *_: Any, **__: Any) -> None:
            raise RuntimeError("jax_verify is unavailable and is not used by analytic plant reachability")

    def _unavailable_jax_verify(*_: Any, **__: Any) -> Any:
        raise RuntimeError("jax_verify is unavailable and is not used by analytic plant reachability")

    jax_verify_stub.IntervalBound = _UnavailableIntervalBound
    jax_verify_stub.backward_crown_bound_propagation = _unavailable_jax_verify
    sys.modules["jax_verify"] = jax_verify_stub
    crown_stub = types.ModuleType("src.crown_wrapper")
    crown_stub.crown = _unavailable_jax_verify
    sys.modules["src.crown_wrapper"] = crown_stub

import src.reachability as upstream_reachability
import src.settings as dr_settings
from src.picard import remainder_picard
from src.symbolic_remainder import init_symbolic_state
from src.taylor_model import QuadTM

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
    file_sha256,
    git_sha,
    iter_configurations,
    load_spec,
    make_row,
    median,
    write_csv,
    write_json,
)

CT_DYN_REACH = upstream_reachability.CT_Dyn_Reach
UPSTREAM_STEP_ONCE = CT_DYN_REACH.step_once
UPSTREAM_BUILD_LINEAR_TM = upstream_reachability.build_linear_tm
UPSTREAM_STEP_TRACE_INVOCATIONS = 0


def _build_linear_tm_float64(center: Any, scale: Any, dtype: Any = jnp.float64) -> Any:
    """Preserve the upstream constructor while overriding its float32 default."""
    del dtype
    return UPSTREAM_BUILD_LINEAR_TM(center, scale, dtype=jnp.float64)


# CT_Dyn_Reach.step_once resolves this module global.  The repository hard-codes
# the helper's default to float32; the experiment contract requires x64.
upstream_reachability.build_linear_tm = _build_linear_tm_float64


def _rhs(system: Mapping[str, Any]):
    def rhs(state: Any) -> Any:
        # This upstream interpreter handles reshape + concatenate, while the
        # installed JAX keeps stack as a primitive it does not register.
        outputs = evaluate_rhs(
            [state[index] for index in range(state.shape[0])], system
        )
        return jnp.concatenate(
            [jnp.reshape(output, (1,)) for output in outputs], axis=0
        )

    return rhs


def _sync(tree: Any) -> Any:
    return jax.tree.map(
        lambda value: (
            value.block_until_ready()
            if hasattr(value, "block_until_ready")
            else value
        ),
        tree,
    )


def _make_core(
    system: Mapping[str, Any],
    settings: Mapping[str, Any],
    h: float,
) -> Any:
    dimension = len(system["state_names"])
    core = CT_DYN_REACH(
        rhs=_rhs(system),
        state_dim=dimension,
        nn_dyn=False,
        step_size=h,
        init_remainder=float(settings["init_remainder"]),
        frr_rounds=int(settings["frr_rounds"]),
        frr_stop_ratio=float(settings["frr_stop_ratio"]),
        sr_window_size=int(settings["symbolic_remainder_window"]),
    )
    core.step_boxes = upstream_reachability._make_step_boxes(
        B=1, D=dimension, h=h, dtype=jnp.float64
    )
    return core


def _initial_carry(
    lower: Any,
    upper: Any,
    *,
    dimension: int,
    symbolic_window: int,
) -> tuple[Any, Any, Any]:
    center = 0.5 * (lower + upper)
    scale = 0.5 * (upper - lower)
    x_tm = UPSTREAM_BUILD_LINEAR_TM(center, scale, dtype=jnp.float64)
    parameterization = upstream_reachability.identity_parameterization(
        1, dimension, dimension + 1, dtype=jnp.float64
    )
    symbolic_state = init_symbolic_state(
        1, dimension, M=symbolic_window, dtype=jnp.float64
    )
    return x_tm, parameterization, symbolic_state


def _upstream_step_with_metrics(
    core: Any,
    carry: tuple[Any, Any, Any],
) -> tuple[tuple[Any, Any, Any], tuple[Any, ...]]:
    global UPSTREAM_STEP_TRACE_INVOCATIONS
    UPSTREAM_STEP_TRACE_INVOCATIONS += 1
    next_carry, (_, _, contraction) = UPSTREAM_STEP_ONCE(core, carry, None)
    local_tm, parameterization, _ = next_carry
    step_lo, step_hi, _, _ = core.step_boxes
    composed = local_tm.compose_affine(parameterization, core.step_size)
    endpoint_lo_domain = jnp.concatenate(
        [step_hi[:, :1], step_lo[:, 1:]], axis=1
    )
    endpoint = composed.eval_interval(endpoint_lo_domain, step_hi)
    tube = composed.eval_interval(step_lo, step_hi)
    endpoint_poly = composed.P.eval_interval(endpoint_lo_domain, step_hi)
    tube_poly = composed.P.eval_interval(step_lo, step_hi)
    remainder_width = composed.R.hi - composed.R.lo
    counts = jnp.stack(
        [
            jnp.count_nonzero(composed.P.c),
            jnp.count_nonzero(composed.P.L),
            jnp.count_nonzero(composed.P.Lt),
        ]
    )
    return next_carry, (
        endpoint.lo,
        endpoint.hi,
        tube.lo,
        tube.hi,
        endpoint_poly.hi - endpoint_poly.lo,
        tube_poly.hi - tube_poly.lo,
        remainder_width,
        contraction,
        counts,
    )


def _compile_native(core: Any, carry: tuple[Any, ...]) -> tuple[Any, float]:
    jitted = jax.jit(lambda value: _upstream_step_with_metrics(core, value))
    started = time.perf_counter()
    compiled = jitted.lower(carry).compile()
    return compiled, time.perf_counter() - started


def _compile_box(
    core: Any,
    lower: Any,
    upper: Any,
) -> tuple[Any, float]:
    dimension = int(lower.shape[-1])

    def one_box_segment(lo: Any, hi: Any) -> tuple[Any, ...]:
        carry = _initial_carry(
            lo, hi, dimension=dimension, symbolic_window=1
        )
        _, outputs = _upstream_step_with_metrics(core, carry)
        return outputs

    jitted = jax.jit(one_box_segment)
    started = time.perf_counter()
    compiled = jitted.lower(lower, upper).compile()
    return compiled, time.perf_counter() - started


def _numpy_outputs(outputs: Sequence[Any]) -> tuple[np.ndarray, ...]:
    synced = _sync(outputs)
    return tuple(np.asarray(value) for value in synced)


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


def _append_outputs(
    rows: list[dict[str, Any]],
    run: Mapping[str, Any],
    system_name: str,
    system: Mapping[str, Any],
    outputs: tuple[np.ndarray, ...],
    step_index: int,
    h: float,
) -> bool:
    (
        endpoint_lo,
        endpoint_hi,
        tube_lo,
        tube_hi,
        endpoint_poly_width,
        tube_poly_width,
        remainder_width,
        _,
        _,
    ) = outputs
    time_value = step_index * h
    analytic_ok = True
    for interval_kind, lowers, uppers, polynomial_widths in (
        ("endpoint", endpoint_lo[0], endpoint_hi[0], endpoint_poly_width[0]),
        ("tube", tube_lo[0], tube_hi[0], tube_poly_width[0]),
    ):
        exact_boxes = exact_interval_for_row(
            system_name,
            interval_kind,
            time_value,
            h,
            system["initial_box"],
        )
        for state_index, state_name in enumerate(system["state_names"]):
            lower = float(lowers[state_index])
            upper = float(uppers[state_index])
            exact = None if exact_boxes is None else exact_boxes[state_index]
            contained = (
                True
                if exact is None
                else lower <= exact[0] + 1e-12 and upper >= exact[1] - 1e-12
            )
            analytic_ok = analytic_ok and contained
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
                    polynomial_width=float(polynomial_widths[state_index]),
                    interval_remainder_width=float(remainder_width[0, state_index]),
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


def run_configuration(
    spec: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    affine_flag: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    system_name = str(config["system"])
    system = spec["systems"][system_name]
    protocol = str(config["protocol"])
    h = float(config["h"])
    steps = int(config["steps"])
    settings = spec["diffreach"]
    variant = (
        "affine_flag"
        if affine_flag
        else "default_restricted_quasi_quadratic"
    )
    if protocol == PROTOCOL_A:
        carried_representation = "none_one_segment"
        reset_policy = "not_applicable"
    elif protocol == PROTOCOL_B:
        carried_representation = "componentwise_axis_aligned_box"
        reset_policy = "endpoint_box_exact_no_inflation"
    else:
        carried_representation = (
            "upstream_normalized_affine_symbolic_carry"
            if affine_flag
            else "upstream_restricted_quasi_quadratic_symbolic_carry"
        )
        reset_policy = "upstream_symbolic_linear_normalization"
    local_basis = (
        "stock_affine_flag_final_{1,t,z};transient_{t^2,t*z}"
        if affine_flag
        else "stock_restricted_quasi_quadratic_{1,t,z,t^2,t*z}"
    )
    local_order: str = (
        "affine_flag_with_transient_quasi_quadratic"
        if affine_flag
        else "restricted_quasi_quadratic"
    )
    run = base_run(
        tool="diffreach",
        tool_variant=variant,
        config=config,
        local_order=local_order,
        local_retained_basis=local_basis,
        carried_representation=carried_representation,
        reset_policy=reset_policy,
        validator="DiffReach_upstream_remainder_picard_initial_contraction",
        dtype="jax_x64",
        device=jax.default_backend(),
        tool_git_sha=git_sha(DIFFREACH_ROOT),
        adapter_git_sha=git_sha(REPO_ROOT),
    )
    rows: list[dict[str, Any]] = []
    _append_initial_rows(rows, run, system)
    lower = jnp.asarray(
        [[float(bounds[0]) for bounds in system["initial_box"]]],
        dtype=jnp.float64,
    )
    upper = jnp.asarray(
        [[float(bounds[1]) for bounds in system["initial_box"]]],
        dtype=jnp.float64,
    )
    old_config = copy.deepcopy(dr_settings.CONFIG)
    step_times: list[float] = []
    support_counts = np.zeros(3, dtype=np.int64)
    failure_message = ""
    trace_count_before = UPSTREAM_STEP_TRACE_INVOCATIONS
    run_started = time.perf_counter()
    try:
        dr_settings.update_config(
            {
                "TRUNCATE_TO_AFFINE": affine_flag,
                "FP64_IN_CROWN": True,
                "BOUND_TIME_STEP": False,
                "DEBUG_LOG": False,
            }
        )
        core = _make_core(system, settings, h)
        if protocol == PROTOCOL_C:
            carry = _initial_carry(
                lower,
                upper,
                dimension=len(system["state_names"]),
                symbolic_window=min(
                    int(settings["symbolic_remainder_window"]), steps
                ),
            )
            compiled, compile_time = _compile_native(core, carry)
        else:
            compiled, compile_time = _compile_box(core, lower, upper)
        run["jit_compile_time_s"] = compile_time
        for step_index in range(1, steps + 1):
            started = time.perf_counter()
            if protocol == PROTOCOL_C:
                carry, raw_outputs = compiled(carry)
            else:
                raw_outputs = compiled(lower, upper)
            outputs = _numpy_outputs(raw_outputs)
            elapsed = time.perf_counter() - started
            step_times.append(elapsed)
            support_counts = np.maximum(support_counts, outputs[-1])
            contraction = bool(np.all(outputs[-2]))
            finite = bool(
                np.all(np.isfinite(outputs[0]))
                and np.all(np.isfinite(outputs[1]))
            )
            if not contraction or not finite:
                failure_message = (
                    "upstream Picard initial contraction failed"
                    if not contraction
                    else "non-finite DiffReach endpoint"
                )
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
            analytic_ok = _append_outputs(
                rows, run, system_name, system, outputs, step_index, h
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
                lower = jnp.asarray(outputs[0], dtype=jnp.float64)
                upper = jnp.asarray(outputs[1], dtype=jnp.float64)
        else:
            run.update(
                run_status="success",
                row_status="validated",
                native_validation_status="validated",
                completed_steps=steps,
                successful_horizon=float(config["horizon"]),
            )
        if protocol == PROTOCOL_A and run["run_status"] == "success":
            repetitions: list[float] = []
            for _ in range(int(spec["steady_repetitions"])):
                started = time.perf_counter()
                _sync(compiled(lower, upper))
                repetitions.append(time.perf_counter() - started)
            run["timing_repetitions_s"] = repetitions
            run["steady_runtime_per_step_s"] = median(repetitions)
    finally:
        dr_settings.CONFIG.clear()
        dr_settings.CONFIG.update(old_config)
    run["orchestration_time_s"] = time.perf_counter() - run_started
    run["first_execution_time_s"] = step_times[0] if step_times else math.nan
    if protocol != PROTOCOL_A or "steady_runtime_per_step_s" not in run:
        run["steady_runtime_per_step_s"] = median(step_times[1:] or step_times)
    run["validation_attempts"] = int(run["completed_steps"])
    run["measured_polynomial_support"] = json.dumps(
        {
            "nonzero_constant_coefficients": int(support_counts[0]),
            "nonzero_linear_coefficients": int(support_counts[1]),
            "nonzero_Lt_coefficients": int(support_counts[2]),
        },
        sort_keys=True,
    )
    run["upstream_step_trace_invocations"] = (
        UPSTREAM_STEP_TRACE_INVOCATIONS - trace_count_before
    )
    run["float64_constructor_override"] = True
    copy_runtime_fields(run, rows)
    return rows, run


def point_evaluations(spec: Mapping[str, Any]) -> dict[str, Any]:
    values: list[dict[str, Any]] = []
    for system_name, system in spec["systems"].items():
        rhs = _rhs(system)
        for point_index, point in enumerate(system["point_checks"]):
            state = jnp.asarray(point, dtype=jnp.float64)
            result = np.asarray(_sync(rhs(state)))
            values.append(
                {
                    "system": system_name,
                    "point_index": point_index,
                    "point": list(map(float, point)),
                    "value": result.tolist(),
                }
            )
    return {
        "tool": "diffreach",
        "dtype": "jax_x64",
        "device": jax.default_backend(),
        "values": values,
    }


def provenance() -> dict[str, Any]:
    step_file = Path(inspect.getsourcefile(UPSTREAM_STEP_ONCE) or "")
    picard_file = Path(inspect.getsourcefile(remainder_picard) or "")
    tm_file = Path(inspect.getsourcefile(QuadTM) or "")
    step_lines = inspect.getsourcelines(UPSTREAM_STEP_ONCE)
    return {
        "diffreach_root": str(DIFFREACH_ROOT),
        "diffreach_git_sha": git_sha(DIFFREACH_ROOT),
        "upstream_class": f"{CT_DYN_REACH.__module__}.{CT_DYN_REACH.__qualname__}",
        "upstream_step_callable_identity": (
            UPSTREAM_STEP_ONCE is CT_DYN_REACH.step_once
        ),
        "upstream_step_source_file": str(step_file),
        "upstream_step_source_line": step_lines[1],
        "upstream_step_file_sha256": file_sha256(step_file),
        "upstream_picard_callable": (
            f"{remainder_picard.__module__}.{remainder_picard.__qualname__}"
        ),
        "upstream_picard_source_file": str(picard_file),
        "upstream_picard_file_sha256": file_sha256(picard_file),
        "upstream_taylor_model_class": f"{QuadTM.__module__}.{QuadTM.__qualname__}",
        "upstream_taylor_model_source_file": str(tm_file),
        "upstream_taylor_model_file_sha256": file_sha256(tm_file),
        "adapter_calls_saved_upstream_step_directly": True,
        "optional_jax_verify_import_shim_used": OPTIONAL_IMPORT_SHIM_USED,
        "optional_shim_behavior": "raises_if_neural_bound_path_is_called",
        "float64_constructor_override": (
            "upstream build_linear_tm called with explicit jnp.float64"
        ),
        "external_repository_modified": False,
        "jax_version": jax.__version__,
        "jax_x64_enabled": bool(jax.config.jax_enable_x64),
        "jax_backend": jax.default_backend(),
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
        variants = (
            (True, False)
            if config["protocol"] == PROTOCOL_C
            else (True,)
        )
        for affine_flag in variants:
            config_rows, run = run_configuration(
                spec, config, affine_flag=affine_flag
            )
            rows.extend(config_rows)
            runs.append(run)
            print(
                f"DiffReach {run['tool_variant']} {config['protocol']} "
                f"{config['system']} h={config['h']:g} T={config['horizon']:g}: "
                f"{run['completed_steps']}/{run['requested_steps']} "
                f"{run['run_status']}",
                flush=True,
            )
    write_csv(output / "diffreach_raw_results.csv", rows, RAW_FIELDS)
    write_csv(output / "diffreach_runs.csv", runs, RUN_FIELDS)
    write_json(output / "diffreach_runs.json", runs)
    write_json(
        output / "diffreach_point_evaluations.json", point_evaluations(spec)
    )
    details = provenance()
    details["total_upstream_step_trace_invocations"] = (
        UPSTREAM_STEP_TRACE_INVOCATIONS
    )
    write_json(output / "diffreach_upstream_provenance.json", details)


if __name__ == "__main__":
    main()
