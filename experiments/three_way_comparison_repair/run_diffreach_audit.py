#!/usr/bin/env python3
"""Audit the unchanged upstream DiffReach endpoint and carry semantics."""
from __future__ import annotations

import argparse
import inspect
import math
import sys
import time
import types
from pathlib import Path
from typing import Any, Mapping

HERE = Path(__file__).resolve().parent
DIFFREACH_ROOT = Path("/srv/local/shengenli/DiffReach")
for candidate in (HERE, DIFFREACH_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp
import numpy as np

OPTIONAL_IMPORT_SHIM_USED = False
try:
    import jax_verify  # type: ignore  # noqa: F401
except ModuleNotFoundError:
    OPTIONAL_IMPORT_SHIM_USED = True
    stub = types.ModuleType("jax_verify")

    class _Unavailable:
        def __init__(self, *_: Any, **__: Any) -> None:
            raise RuntimeError("unused neural-bound dependency is unavailable")

    def _unavailable(*_: Any, **__: Any) -> Any:
        raise RuntimeError("unused neural-bound dependency is unavailable")

    stub.IntervalBound = _Unavailable
    stub.backward_crown_bound_propagation = _unavailable
    sys.modules["jax_verify"] = stub
    crown_stub = types.ModuleType("src.crown_wrapper")
    crown_stub.crown = _unavailable
    sys.modules["src.crown_wrapper"] = crown_stub

import src.reachability as reachability
import src.settings as dr_settings
from src.symbolic_remainder import init_symbolic_state

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

ORIGINAL_BUILD_LINEAR_TM = reachability.build_linear_tm
ORIGINAL_STEP_ONCE = reachability.CT_Dyn_Reach.step_once


def _build_linear_tm_x64(center: Any, scale: Any, dtype: Any = jnp.float64):
    del dtype
    return ORIGINAL_BUILD_LINEAR_TM(center, scale, dtype=jnp.float64)


# This is the only arithmetic override: upstream hard-codes float32 through the
# default constructor even when the enclosing run is x64.
reachability.build_linear_tm = _build_linear_tm_x64


def _power(value: Any, exponent: int) -> Any:
    result: Any = 1.0
    for _ in range(exponent):
        result = result * value
    return result


def _rhs(system: Mapping[str, Any]):
    def rhs(state: Any) -> Any:
        outputs = []
        for polynomial in system["rhs"]:
            value: Any = 0.0
            for term in polynomial["terms"]:
                product: Any = float(term["coefficient"])
                for coordinate, exponent in zip(state, term["powers"]):
                    product = product * _power(coordinate, int(exponent))
                value = value + product
            outputs.append(jnp.reshape(value, (1,)))
        return jnp.concatenate(outputs, axis=0)

    return rhs


def _core(spec: Mapping[str, Any], system: Mapping[str, Any], h: float):
    settings = spec["diffreach"]
    dimension = len(system["state_names"])
    core = reachability.CT_Dyn_Reach(
        rhs=_rhs(system),
        state_dim=dimension,
        nn_dyn=False,
        step_size=h,
        init_remainder=float(settings["init_remainder"]),
        frr_rounds=int(settings["frr_rounds"]),
        frr_stop_ratio=float(settings["frr_stop_ratio"]),
        sr_window_size=int(settings["symbolic_remainder_window"]),
    )
    core.step_boxes = reachability._make_step_boxes(
        1, dimension, h, dtype=jnp.float64
    )
    return core


def _initial_carry(
    lower: Any, upper: Any, *, dimension: int, symbolic_window: int
):
    center = 0.5 * (lower + upper)
    scale = 0.5 * (upper - lower)
    local = ORIGINAL_BUILD_LINEAR_TM(center, scale, dtype=jnp.float64)
    parameterization = reachability.identity_parameterization(
        1, dimension, dimension + 1, dtype=jnp.float64
    )
    symbolic = init_symbolic_state(
        1, dimension, M=symbolic_window, dtype=jnp.float64
    )
    return local, parameterization, symbolic


def _advance(core: Any, carry: Any):
    next_carry, (_, _, contraction) = ORIGINAL_STEP_ONCE(core, carry, None)
    local_tm, parameterization, _ = next_carry
    step_lo, step_hi, _, _ = core.step_boxes
    composed = local_tm.compose_affine(parameterization, core.step_size)
    endpoint_lo_domain = jnp.concatenate(
        [step_hi[:, :1], step_lo[:, 1:]], axis=1
    )
    endpoint = composed.eval_interval(endpoint_lo_domain, step_hi)
    endpoint_poly = composed.P.eval_interval(endpoint_lo_domain, step_hi)
    tube = composed.eval_interval(step_lo, step_hi)
    tube_poly = composed.P.eval_interval(step_lo, step_hi)
    return next_carry, (
        endpoint,
        endpoint_poly,
        tube,
        tube_poly,
        composed.R,
        contraction,
    )


def _sync(value: Any) -> Any:
    return jax.tree.map(
        lambda item: item.block_until_ready()
        if hasattr(item, "block_until_ready")
        else item,
        value,
    )


def _append(
    rows: list[dict[str, Any]],
    *,
    spec: Mapping[str, Any],
    system_name: str,
    variant: str,
    protocol: str,
    h: float,
    horizon: float,
    step_index: int,
    kind: str,
    interval: Any,
    polynomial: Any,
    remainder: Any,
    affine: bool,
    carry_description: str,
    elapsed: float,
    build_time: float,
) -> bool:
    system = spec["systems"][system_name]
    absolute_time = step_index * h
    exact_boxes = reference_for_row(
        system_name, kind, absolute_time, h, system["initial_box"]
    )
    lo = np.asarray(interval.lo)[0]
    hi = np.asarray(interval.hi)[0]
    poly_lo = np.asarray(polynomial.lo)[0]
    poly_hi = np.asarray(polynomial.hi)[0]
    rem_lo = np.asarray(remainder.lo)[0]
    rem_hi = np.asarray(remainder.hi)[0]
    all_valid = True
    for state_index in range(len(system["state_names"])):
        lower, upper = float(lo[state_index]), float(hi[state_index])
        exact = None if exact_boxes is None else exact_boxes[state_index]
        contains = (
            True
            if exact is None
            else lower <= exact[0] + float(spec["containment_tolerance"])
            and upper >= exact[1] - float(spec["containment_tolerance"])
        )
        all_valid = (
            all_valid
            and contains
            and math.isfinite(lower)
            and math.isfinite(upper)
        )
        rows.append(
            make_row(
                tool="diffreach",
                variant=variant,
                protocol=protocol,
                system=system_name,
                h=h,
                horizon=horizon,
                step_index=step_index,
                absolute_time=absolute_time,
                state_index=state_index,
                interval_kind=kind,
                lower=lower,
                upper=upper,
                exact=exact,
                native_validation_status="initial_picard_inclusion_passed",
                analytic_reference_status=(
                    "not_available"
                    if exact is None
                    else ("passed" if contains else "failed")
                ),
                local_order="affine" if affine else "restricted_quasi_quadratic",
                local_basis=(
                    "constant_plus_linear"
                    if affine
                    else "constant_linear_t2_and_t_times_generators"
                ),
                carried_representation=carry_description,
                step_policy=f"fixed_{h:.17g}",
                cutoff="upstream_operation_specific",
                native_returned_remainder=(
                    float(rem_hi[state_index]) - float(rem_lo[state_index])
                ),
                postprocessed_remainder="",
                remainder_overwrite_applied=False,
                endpoint_tightening_applied=False,
                endpoint_semantics=(
                    "whole_segment_tau_in_[0,h]"
                    if kind == "tube"
                    else "raw_substitution_tau_equals_h"
                ),
                polynomial_width=(
                    float(poly_hi[state_index]) - float(poly_lo[state_index])
                ),
                remainder_width=(
                    float(rem_hi[state_index]) - float(rem_lo[state_index])
                ),
                build_time_s=build_time,
                warmup_time_s=elapsed if step_index == 1 else "",
                steady_runtime_s=elapsed,
                dtype="jax_float64",
                device=jax.default_backend(),
                repository_sha=git_sha(DIFFREACH_ROOT),
                environment="diffreach312",
            )
        )
    return all_valid


def run_case(
    spec: Mapping[str, Any],
    *,
    system_name: str,
    protocol: str,
    h: float,
    horizon: float,
    affine: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    system = spec["systems"][system_name]
    steps = exact_steps(h, horizon)
    variant = (
        "diffreach_affine"
        if affine
        else "diffreach_restricted_quasi_quadratic"
    )
    if protocol == PROTOCOL_BOX:
        carry_description = "componentwise_box_from_raw_endpoint"
    elif protocol in {PROTOCOL_NATIVE, PROTOCOL_STRESS}:
        carry_description = "upstream_symbolic_normalized_representation"
    else:
        carry_description = "none_one_step"
    lower = jnp.asarray(
        [[bounds[0] for bounds in system["initial_box"]]], dtype=jnp.float64
    )
    upper = jnp.asarray(
        [[bounds[1] for bounds in system["initial_box"]]], dtype=jnp.float64
    )
    dimension = len(system["state_names"])
    core = _core(spec, system, h)
    carry = _initial_carry(
        lower,
        upper,
        dimension=dimension,
        symbolic_window=min(
            int(spec["diffreach"]["symbolic_remainder_window"]), steps
        ),
    )
    old_config = dict(dr_settings.CONFIG)
    rows: list[dict[str, Any]] = []
    step_times: list[float] = []
    status = "success"
    message = ""
    try:
        dr_settings.update_config(
            {
                "TRUNCATE_TO_AFFINE": affine,
                "FP64_IN_CROWN": True,
                "BOUND_TIME_STEP": False,
                "DEBUG_LOG": False,
            }
        )
        # Warm/compile exactly the operation used in the loop.
        if protocol in {PROTOCOL_NATIVE, PROTOCOL_STRESS}:
            compiled = jax.jit(lambda value: _advance(core, value))
            started = time.perf_counter()
            compiled.lower(carry).compile()
            build_time = time.perf_counter() - started
        else:
            def box_step(lo: Any, hi: Any):
                initial = _initial_carry(
                    lo, hi, dimension=dimension, symbolic_window=1
                )
                return _advance(core, initial)[1]

            compiled = jax.jit(box_step)
            started = time.perf_counter()
            compiled.lower(lower, upper).compile()
            build_time = time.perf_counter() - started
        for step_index in range(1, steps + 1):
            started = time.perf_counter()
            if protocol in {PROTOCOL_NATIVE, PROTOCOL_STRESS}:
                carry, outputs = _sync(compiled(carry))
            else:
                outputs = _sync(compiled(lower, upper))
            elapsed = time.perf_counter() - started
            step_times.append(elapsed)
            endpoint, endpoint_poly, tube, tube_poly, remainder, contraction = outputs
            if not bool(np.all(np.asarray(contraction))):
                status = "failed"
                message = "upstream initial Picard inclusion failed"
                for state_index in range(dimension):
                    rows.append(
                        make_row(
                            tool="diffreach",
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
                            failure_category="first_picard_inclusion_failed",
                            failure_message=message,
                            local_order="affine" if affine else "restricted_quasi_quadratic",
                            carried_representation=carry_description,
                            endpoint_semantics="not_available",
                            dtype="jax_float64",
                            device=jax.default_backend(),
                            repository_sha=git_sha(DIFFREACH_ROOT),
                            environment="diffreach312",
                        )
                    )
                break
            if protocol == PROTOCOL_TUBE:
                _append(
                    rows,
                    spec=spec,
                    system_name=system_name,
                    variant=variant,
                    protocol=protocol,
                    h=h,
                    horizon=horizon,
                    step_index=step_index,
                    kind="tube",
                    interval=tube,
                    polynomial=tube_poly,
                    remainder=remainder,
                    affine=affine,
                    carry_description=carry_description,
                    elapsed=elapsed,
                    build_time=build_time,
                )
            else:
                _append(
                    rows,
                    spec=spec,
                    system_name=system_name,
                    variant=variant,
                    protocol=protocol,
                    h=h,
                    horizon=horizon,
                    step_index=step_index,
                    kind="endpoint_raw",
                    interval=endpoint,
                    polynomial=endpoint_poly,
                    remainder=remainder,
                    affine=affine,
                    carry_description=carry_description,
                    elapsed=elapsed,
                    build_time=build_time,
                )
            if protocol == PROTOCOL_BOX:
                lower, upper = endpoint.lo, endpoint.hi
    finally:
        dr_settings.CONFIG.clear()
        dr_settings.CONFIG.update(old_config)
    return rows, {
        "tool": "diffreach",
        "variant": variant,
        "protocol": protocol,
        "system": system_name,
        "h": h,
        "requested_horizon": horizon,
        "requested_steps": steps,
        "status": status,
        "message": message,
        "jit_compile_time_s": build_time,
        "step_times_s": step_times,
    }


def _cases(spec: Mapping[str, Any], smoke: bool):
    for system_name, benchmark in spec["benchmarks"].items():
        one_steps = [float(benchmark["smoke"]["h"])] if smoke else [
            float(value) for value in benchmark["one_step_h"]
        ]
        for h in one_steps:
            for affine in (True, False):
                yield system_name, PROTOCOL_TUBE, h, h, affine
                yield system_name, PROTOCOL_RAW, h, h, affine
        multi = [benchmark["smoke"]] if smoke else benchmark["multi_step"]
        for config in multi:
            h, horizon = float(config["h"]), float(config["horizon"])
            yield system_name, PROTOCOL_BOX, h, horizon, True
            for affine in (True, False):
                yield system_name, PROTOCOL_NATIVE, h, horizon, affine
            yield system_name, PROTOCOL_STRESS, h, horizon, True
    if not smoke:
        yield "riccati", PROTOCOL_BOX, 0.01, 0.1, True
        yield "harmonic", PROTOCOL_BOX, 0.01, 1.0, True


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
    for system, protocol, h, horizon, affine in _cases(spec, args.smoke):
        case_rows, run = run_case(
            spec,
            system_name=system,
            protocol=protocol,
            h=h,
            horizon=horizon,
            affine=affine,
        )
        rows.extend(case_rows)
        runs.append(run)
        print(
            f"DiffReach {run['variant']} {protocol} {system} "
            f"h={h:g} T={horizon:g}: {run['status']}",
            flush=True,
        )
    write_csv(output / "diffreach_endpoint_audit.csv", rows)
    write_json(output / "diffreach_runs.json", runs)
    write_json(
        output / "diffreach_provenance.json",
        {
            "repository_sha": git_sha(DIFFREACH_ROOT),
            "upstream_step_file": inspect.getsourcefile(ORIGINAL_STEP_ONCE),
            "upstream_step_line": inspect.getsourcelines(ORIGINAL_STEP_ONCE)[1],
            "adapter_calls_saved_upstream_step": True,
            "float64_constructor_override": True,
            "other_arithmetic_monkey_patches": False,
            "jax_version": jax.__version__,
            "jax_x64_enabled": bool(jax.config.read("jax_enable_x64")),
            "optional_import_shim_used": OPTIONAL_IMPORT_SHIM_USED,
        },
    )


if __name__ == "__main__":
    main()
