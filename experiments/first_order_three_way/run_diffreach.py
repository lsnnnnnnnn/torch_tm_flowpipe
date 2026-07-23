#!/usr/bin/env python3
"""Run DiffReach through a read-only, plant-only adapter.

This intentionally imports only DiffReach's analytic Taylor-model path.  Its
published project dependency set is currently internally inconsistent
(`jax2onnx` and `immrax` require incompatible Equinox versions), so the
benchmark uses a dedicated Python 3.12 environment with CPU JAX and does not
import controller/CROWN code.
"""
from __future__ import annotations

import argparse
import copy
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

DIFFREACH_ROOT = Path(os.environ.get("DIFFREACH_ROOT", "/srv/local/shengenli/DiffReach")).resolve()
if str(DIFFREACH_ROOT) not in sys.path:
    sys.path.insert(0, str(DIFFREACH_ROOT))

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp
import numpy as np

import src.settings as dr_settings
from src.interval import Interval
from src.picard import remainder_picard
from src.polynomial import QuadPoly
from src.rhs_eval import build_auto_rhs_analytic
from src.symbolic_remainder import init_symbolic_state, symbolic_step_linear
from src.taylor_model import QuadTM

from common import (
    configuration_timeout,
    evaluate_rhs,
    git_sha,
    interval_row,
    iter_configurations,
    load_spec,
    median_iqr,
    output_dir_from_args,
    raw_run_template,
    utc_timestamp,
    write_csv,
    write_json,
)


def build_linear_tm(c: jax.Array, scale: jax.Array) -> QuadTM:
    batch, dimension = c.shape
    variables = dimension + 1
    polynomial = QuadPoly.zeros(batch, dimension, variables, dtype=c.dtype)
    polynomial.c = c
    indices = jnp.arange(dimension)
    polynomial.L = polynomial.L.at[:, indices, indices + 1].set(scale)
    return QuadTM.from_poly(polynomial)


def identity_parameterization(batch: int, dimension: int, dtype: Any) -> QuadTM:
    variables = dimension + 1
    polynomial = QuadPoly.zeros(batch, dimension, variables, dtype=dtype)
    indices = jnp.arange(dimension)
    polynomial.L = polynomial.L.at[:, indices, indices + 1].set(1.0)
    return QuadTM.from_poly(polynomial)


def step_boxes(batch: int, dimension: int, h: float, dtype: Any) -> tuple[jax.Array, ...]:
    zero = jnp.zeros((batch, 1), dtype=dtype)
    h_column = jnp.full((batch, 1), h, dtype=dtype)
    ones = jnp.ones((batch, dimension), dtype=dtype)
    return (
        jnp.concatenate([zero, -ones], axis=1),
        jnp.concatenate([h_column, ones], axis=1),
        jnp.concatenate([zero, -ones], axis=1),
        jnp.concatenate([zero, ones], axis=1),
    )


class DiffReachPlantCore:
    def __init__(
        self,
        rhs: Any,
        *,
        dimension: int,
        h: float,
        init_remainder: float,
        frr_rounds: int,
        frr_stop_ratio: float,
        symbolic_window: int,
    ) -> None:
        self.dimension = int(dimension)
        self.variables = self.dimension + 1
        self.h = float(h)
        self.init_remainder = float(init_remainder)
        self.frr_rounds = int(frr_rounds)
        self.frr_stop_ratio = float(frr_stop_ratio)
        self.symbolic_window = int(symbolic_window)
        self.rhs_poly, self.rhs_tm = build_auto_rhs_analytic(rhs, D=self.dimension, V=self.variables)

    def step_once(self, carry: tuple[Any, ...], unused: Any) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
        del unused
        x_tm, parameterization, symbolic_state, step_lo, step_hi, eval_lo, eval_hi = carry
        endpoint_tm = x_tm.evaluate_time(self.h)
        center = endpoint_tm.P.c
        scale, normalized, symbolic_next = symbolic_step_linear(
            parameterization, endpoint_tm, symbolic_state, eval_lo, eval_hi
        )
        new_x0 = build_linear_tm(center, scale)
        base = new_x0.P
        poly1 = base.add(self.rhs_poly(base, step_lo, step_hi).integrate_time_trunc())
        poly2 = base.add(self.rhs_poly(poly1, step_lo, step_hi).integrate_time_trunc())
        polynomial_tm = QuadTM.from_poly(poly2)
        epsilon = jnp.broadcast_to(
            jnp.asarray(self.init_remainder, dtype=center.dtype),
            center.shape,
        )
        seeded = QuadTM(
            polynomial_tm.P,
            Interval(polynomial_tm.R.lo - epsilon, polynomial_tm.R.hi + epsilon),
        )
        x_next, contraction = remainder_picard(
            self.rhs_tm,
            new_x0,
            seeded,
            self.h,
            step_lo,
            step_hi,
            rounds=self.frr_rounds,
            stop_ratio=self.frr_stop_ratio,
        )
        if dr_settings.CONFIG["TRUNCATE_TO_AFFINE"]:
            x_next = x_next.truncate_to_affine(step_lo, step_hi)
        composed = x_next.compose_affine(normalized, self.h)
        endpoint_lo = jnp.concatenate([step_hi[:, :1], step_lo[:, 1:]], axis=1)
        endpoint = composed.eval_interval(endpoint_lo, step_hi)
        tube = composed.eval_interval(step_lo, step_hi)
        next_carry = (
            x_next,
            normalized,
            symbolic_next,
            step_lo,
            step_hi,
            eval_lo,
            eval_hi,
        )
        return next_carry, (endpoint.lo, endpoint.hi, tube.lo, tube.hi, contraction)

    def verify(self, lower: jax.Array, upper: jax.Array, steps: int) -> tuple[Any, ...]:
        batch, dimension = lower.shape
        dtype = lower.dtype
        boxes = step_boxes(batch, dimension, self.h, dtype)
        parameterization = identity_parameterization(batch, dimension, dtype)
        initial = build_linear_tm(0.5 * (lower + upper), 0.5 * (upper - lower))
        symbolic = init_symbolic_state(
            batch, dimension, M=min(self.symbolic_window, steps), dtype=dtype
        )
        initial_interval = initial.eval_interval(boxes[0], boxes[1])
        carry = (initial, parameterization, symbolic, *boxes)
        final_carry, outputs = jax.lax.scan(self.step_once, carry, None, length=steps)
        endpoint_lo, endpoint_hi, tube_lo, tube_hi, contraction = outputs
        return (
            jnp.arange(steps + 1, dtype=dtype) * self.h,
            jnp.concatenate([initial_interval.lo[:, None, :], endpoint_lo.transpose((1, 0, 2))], axis=1),
            jnp.concatenate([initial_interval.hi[:, None, :], endpoint_hi.transpose((1, 0, 2))], axis=1),
            tube_lo.transpose((1, 0, 2)),
            tube_hi.transpose((1, 0, 2)),
            final_carry[0],
            contraction,
        )


def _sync(tree: Any) -> Any:
    return jax.tree.map(
        lambda value: value.block_until_ready() if hasattr(value, "block_until_ready") else value,
        tree,
    )


def _rhs(system: Mapping[str, Any]):
    def rhs(value: jax.Array) -> jax.Array:
        # DiffReach's interpreter handles reshape + concatenate.  On recent JAX,
        # jnp.stack remains a dedicated primitive that this repository does not
        # register, while its own example was written against a JAX release that
        # lowered stack to these two primitives.
        outputs = evaluate_rhs([value[index] for index in range(value.shape[0])], system)
        return jnp.concatenate([jnp.reshape(output, (1,)) for output in outputs], axis=0)

    return rhs


def _support(poly: QuadPoly, *, threshold: float = 0.0) -> dict[str, Any]:
    c = np.asarray(poly.c)
    linear = np.asarray(poly.L)
    lt = np.asarray(poly.Lt)
    c_support = np.argwhere(np.abs(c) > threshold).tolist()
    linear_support = np.argwhere(np.abs(linear) > threshold).tolist()
    lt_support = np.argwhere(np.abs(lt) > threshold).tolist()
    return {
        "nonzero_c_support": c_support,
        "nonzero_L_support": linear_support,
        "nonzero_Lt_support": lt_support,
        "nonzero_Lt": bool(lt_support),
        "effective_max_degree": 2 if lt_support else (1 if linear_support else 0),
    }


def diagnose_support(
    core: DiffReachPlantCore,
    system: Mapping[str, Any],
    lower: jax.Array,
    upper: jax.Array,
    final_tm: QuadTM,
) -> dict[str, Any]:
    boxes = step_boxes(1, core.dimension, core.h, lower.dtype)
    base = build_linear_tm(0.5 * (lower + upper), 0.5 * (upper - lower)).P
    after_dynamics = core.rhs_poly(base, boxes[0], boxes[1])
    after_integration = after_dynamics.integrate_time_trunc()
    return {
        "initial_model": _support(base),
        "after_dynamics_evaluation": _support(after_dynamics),
        "after_time_integration": _support(after_integration),
        "final_flowpipe_segment": _support(final_tm.P),
        "basis_mapping": {
            "c": "constant",
            "L[...,0]": "local_time",
            "L[...,j>=1]": "normalized initial/state generator",
            "Lt[...,0]": "local_time^2",
            "Lt[...,j>=1]": "local_time*generator_j",
        },
    }


def _unsupported_strict_rows(
    *,
    spec: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    run = raw_run_template(
        tool="diffreach",
        protocol="strict_common_affine",
        system=str(config["system"]),
        h=float(config["h"]),
        horizon=float(config["horizon"]),
        requested_order_label="strict affine projection requested",
        retained_basis="unsupported: no tested sound projection of all Lt terms into an independent remainder",
        effective_max_degree="",
        truncate_to_affine=True,
        nonzero_lt="",
        dependency_mode="unsupported",
        symbolic_remainder_size=int(spec["diffreach"]["symbolic_remainder_window"]),
        cutoff=None,
        dtype="float64",
        device="cpu",
        git_commit=git_sha(DIFFREACH_ROOT),
        environment="diffreach312",
    )
    run.update(
        status="unsupported_order",
        validation_status="unsupported",
        first_failure_time=0.0,
        successful_horizon=0.0,
        message=(
            "DiffReach's native affine flag creates Lt terms during integration and its final "
            "projection embeds their interval radius into reused L generators; this benchmark "
            "does not treat that as a proved common-basis remainder projection."
        ),
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
        for state_index in range(len(spec["systems"][config["system"]]["state_names"]))
    ]
    return rows, {**run, "requested_steps": int(config["steps"])}


def run_native(
    *,
    spec: Mapping[str, Any],
    config: Mapping[str, Any],
    truncate_to_affine: bool,
    protocol: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    system_name = str(config["system"])
    system = spec["systems"][system_name]
    h, horizon, steps = float(config["h"]), float(config["horizon"]), int(config["steps"])
    old_config = copy.deepcopy(dr_settings.CONFIG)
    try:
        dr_settings.update_config(
            {
                "TRUNCATE_TO_AFFINE": bool(truncate_to_affine),
                "FP64_IN_CROWN": True,
                "BOUND_TIME_STEP": bool(spec["diffreach"]["bound_time_step"]),
                "DEBUG_LOG": False,
            }
        )
        core = DiffReachPlantCore(
            _rhs(system),
            dimension=len(system["state_names"]),
            h=h,
            init_remainder=float(spec["diffreach"]["init_remainder"]),
            frr_rounds=int(spec["diffreach"]["frr_rounds"]),
            frr_stop_ratio=float(spec["diffreach"]["frr_stop_ratio"]),
            symbolic_window=int(spec["diffreach"]["symbolic_remainder_window"]),
        )
        lower = jnp.asarray([[bounds[0] for bounds in system["initial_box"]]], dtype=jnp.float64)
        upper = jnp.asarray([[bounds[1] for bounds in system["initial_box"]]], dtype=jnp.float64)
        compiled = jax.jit(lambda lo, hi: core.verify(lo, hi, steps))
        started = time.perf_counter()
        result = _sync(compiled(lower, upper))
        warmup_s = time.perf_counter() - started
        timings: list[float] = []
        for _ in range(int(spec["steady_repetitions"])):
            started = time.perf_counter()
            _sync(compiled(lower, upper))
            timings.append(time.perf_counter() - started)
        times, endpoint_lower, endpoint_upper, tube_lower, tube_upper, final_tm, contraction = result
        support = diagnose_support(core, system, lower, upper, final_tm)
    finally:
        dr_settings.CONFIG.clear()
        dr_settings.CONFIG.update(old_config)

    times_np = np.asarray(times)
    endpoint_lo_np, endpoint_hi_np = np.asarray(endpoint_lower[0]), np.asarray(endpoint_upper[0])
    tube_lo_np, tube_hi_np = np.asarray(tube_lower[0]), np.asarray(tube_upper[0])
    contraction_np = np.asarray(contraction)
    finite_by_step = np.all(np.isfinite(endpoint_lo_np[1:]), axis=1) & np.all(
        np.isfinite(endpoint_hi_np[1:]), axis=1
    )
    contraction_by_step = np.all(contraction_np, axis=tuple(range(1, contraction_np.ndim)))
    bad_indices = np.flatnonzero(~finite_by_step | ~contraction_by_step)
    first_bad = int(bad_indices[0] + 1) if bad_indices.size else None
    complete = first_bad is None
    if complete:
        status = "certified_ok"
        validation_status = "picard_contraction_passed"
    elif not finite_by_step[first_bad - 1]:
        status = "numerical_error"
        validation_status = "nonfinite"
    else:
        status = "contraction_failed"
        validation_status = "picard_contraction_failed"
    successful_horizon = horizon if complete else float((first_bad - 1) * h)
    final_support = support["final_flowpipe_segment"]
    integration_support = support["after_time_integration"]
    nonzero_lt = bool(final_support["nonzero_Lt"])
    effective_degree = max(
        int(final_support["effective_max_degree"]),
        int(integration_support["effective_max_degree"]),
    )
    retained_basis = (
        "affine_final_after_Lt_range_embedding; transient_{t^2,t*generator}"
        if truncate_to_affine
        else "restricted_quasi_quadratic_{1,z,t^2,t*z}"
    )
    median_s, iqr_s = median_iqr(timings)
    run = raw_run_template(
        tool="diffreach",
        protocol=protocol,
        system=system_name,
        h=h,
        horizon=horizon,
        requested_order_label=(
            "TRUNCATE_TO_AFFINE=True" if truncate_to_affine else "TRUNCATE_TO_AFFINE=False"
        ),
        retained_basis=retained_basis,
        effective_max_degree=effective_degree,
        truncate_to_affine=truncate_to_affine,
        nonzero_lt=nonzero_lt,
        dependency_mode="normalized_symbolic_remainder_window",
        symbolic_remainder_size=int(spec["diffreach"]["symbolic_remainder_window"]),
        cutoff=None,
        dtype="float64",
        device=f"jax_{jax.default_backend()}",
        git_commit=git_sha(DIFFREACH_ROOT),
        environment="diffreach312",
    )
    run.update(
        status=status,
        validation_status=validation_status,
        first_failure_time="" if complete else float(first_bad * h),
        successful_horizon=successful_horizon,
        warmup_time_s=warmup_s,
        steady_runtime_median_s=median_s,
        steady_runtime_iqr_s=iqr_s,
        validation_attempts=int(spec["diffreach"]["frr_rounds"]) * min(steps, first_bad or steps),
        message="" if complete else validation_status,
    )
    rows: list[dict[str, Any]] = []
    max_success_step = steps if complete else first_bad - 1
    for step_index in range(max_success_step + 1):
        for state_index in range(len(system["state_names"])):
            rows.append(
                interval_row(
                    run=run,
                    state_index=state_index,
                    step_index=step_index,
                    time_value=float(times_np[step_index]),
                    interval_kind="endpoint",
                    lower=float(endpoint_lo_np[step_index, state_index]),
                    upper=float(endpoint_hi_np[step_index, state_index]),
                )
            )
            if step_index > 0:
                rows.append(
                    interval_row(
                        run=run,
                        state_index=state_index,
                        step_index=step_index,
                        time_value=float(times_np[step_index]),
                        interval_kind="tube",
                        lower=float(tube_lo_np[step_index - 1, state_index]),
                        upper=float(tube_hi_np[step_index - 1, state_index]),
                    )
                )
    if not complete:
        for state_index in range(len(system["state_names"])):
            rows.append(
                interval_row(
                    run=run,
                    state_index=state_index,
                    step_index=first_bad,
                    time_value=first_bad * h,
                    interval_kind="failure_marker",
                    lower="",
                    upper="",
                )
            )
    metadata = {
        **run,
        "requested_steps": steps,
        "completed_certified_steps": max_success_step,
        "support_diagnostic": support,
        "contraction_by_step": contraction_by_step.tolist(),
        "finite_by_step": finite_by_step.tolist(),
        "timing_repetitions_s": timings,
        "jax_version": jax.__version__,
        "jax_x64_enabled": bool(jax.config.x64_enabled),
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
        "global_settings_restored_after_run": True,
        "partitions": 1,
        "interval_semantics": {
            "endpoint": "composed segment evaluated with local time fixed at h",
            "tube": "same composed segment evaluated over local time [0,h]",
        },
    }
    return rows, metadata


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
    for config in iter_configurations(spec, smoke=args.smoke, systems=args.systems):
        for truncate, protocol in (
            (True, "native_first_order_setting"),
            (False, "supplementary_native_representations"),
        ):
            try:
                with configuration_timeout(float(spec["timeout_s"])):
                    rows, metadata = run_native(
                        spec=spec,
                        config=config,
                        truncate_to_affine=truncate,
                        protocol=protocol,
                    )
            except Exception as exc:
                run = raw_run_template(
                    tool="diffreach",
                    protocol=protocol,
                    system=str(config["system"]),
                    h=float(config["h"]),
                    horizon=float(config["horizon"]),
                    requested_order_label=f"TRUNCATE_TO_AFFINE={truncate}",
                    retained_basis="execution_failed_before_support_measurement",
                    effective_max_degree="",
                    truncate_to_affine=truncate,
                    nonzero_lt="",
                    dependency_mode="normalized_symbolic_remainder_window",
                    symbolic_remainder_size=int(spec["diffreach"]["symbolic_remainder_window"]),
                    cutoff=None,
                    dtype="float64",
                    device="jax_cpu",
                    git_commit=git_sha(DIFFREACH_ROOT),
                    environment="diffreach312",
                )
                run.update(
                    status="timeout" if isinstance(exc, TimeoutError) else "numerical_error",
                    validation_status="timeout" if isinstance(exc, TimeoutError) else "exception",
                    first_failure_time=0.0,
                    successful_horizon=0.0,
                    message=f"{type(exc).__name__}: {exc}",
                )
                rows = [
                    interval_row(
                        run=run,
                        state_index=index,
                        step_index=0,
                        time_value=0.0,
                        interval_kind="failure_marker",
                        lower="",
                        upper="",
                    )
                    for index in range(len(spec["systems"][config["system"]]["state_names"]))
                ]
                metadata = {**run, "requested_steps": int(config["steps"])}
            all_rows.extend(rows)
            write_json(output_dir / "per_run" / f"{metadata['run_id']}.json", metadata)
            print(
                f"diffreach affine={truncate} {config['system']} h={config['h']} "
                f"T={config['horizon']} status={metadata['status']}",
                flush=True,
            )
        strict_rows, strict_metadata = _unsupported_strict_rows(spec=spec, config=config)
        all_rows.extend(strict_rows)
        write_json(output_dir / "per_run" / f"{strict_metadata['run_id']}.json", strict_metadata)
    suffix = "smoke" if args.smoke else "full"
    write_csv(output_dir / f"diffreach_raw_{suffix}.csv", all_rows)
    write_json(
        output_dir / f"diffreach_manifest_{suffix}.json",
        {
            "timestamp": utc_timestamp(),
            "rows": len(all_rows),
            "environment": "diffreach312",
            "diffreach_root": str(DIFFREACH_ROOT),
        },
    )
    print(output_dir)


if __name__ == "__main__":
    main()
