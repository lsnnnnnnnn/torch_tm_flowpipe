#!/usr/bin/env python3
"""Export one upstream DiffReach segment to the common read-only representation."""
from __future__ import annotations

import argparse
import os
import sys
import time
import types
from pathlib import Path
from typing import Any, Mapping

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DIFFREACH_ROOT = Path(
    os.environ.get("DIFFREACH_ROOT", REPO_ROOT.parent / "DiffReach")
).resolve()
for candidate in (HERE, DIFFREACH_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp
import numpy as np

try:
    import jax_verify  # type: ignore  # noqa: F401
    OPTIONAL_IMPORT_SHIM_USED = False
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
    canonical_record,
    deterministic_points,
    git_sha,
    load_spec,
    unavailable,
    write_json,
)

ORIGINAL_BUILD_LINEAR_TM = reachability.build_linear_tm
ORIGINAL_STEP_ONCE = reachability.CT_Dyn_Reach.step_once


def _build_linear_tm_x64(center: Any, scale: Any, dtype: Any = jnp.float64):
    del dtype
    return ORIGINAL_BUILD_LINEAR_TM(center, scale, dtype=jnp.float64)


reachability.build_linear_tm = _build_linear_tm_x64


def _power(value: Any, exponent: int) -> Any:
    result: Any = 1.0
    for _ in range(int(exponent)):
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
    dimension = len(system["state_names"])
    rounds = spec["diffreach"]["frr_rounds"]
    if isinstance(rounds, list):
        rounds = max(rounds)
    windows = spec["diffreach"]["symbolic_windows"]
    if isinstance(windows, list):
        windows = max(windows)
    core = reachability.CT_Dyn_Reach(
        rhs=_rhs(system),
        state_dim=dimension,
        nn_dyn=False,
        step_size=float(h),
        init_remainder=float(spec["diffreach"]["init_remainder"]),
        frr_rounds=int(rounds),
        frr_stop_ratio=float(spec["diffreach"]["frr_stop_ratio"]),
        sr_window_size=int(windows),
    )
    core.step_boxes = reachability._make_step_boxes(
        1, dimension, float(h), dtype=jnp.float64
    )
    return core


def _initial_carry(system: Mapping[str, Any], symbolic_window: int):
    lower = jnp.asarray([[box[0] for box in system["initial_box"]]], dtype=jnp.float64)
    upper = jnp.asarray([[box[1] for box in system["initial_box"]]], dtype=jnp.float64)
    center = 0.5 * (lower + upper)
    scale = 0.5 * (upper - lower)
    local = ORIGINAL_BUILD_LINEAR_TM(center, scale, dtype=jnp.float64)
    dimension = lower.shape[1]
    parameterization = reachability.identity_parameterization(
        1, dimension, dimension + 1, dtype=jnp.float64
    )
    symbolic = init_symbolic_state(
        1, dimension, M=int(symbolic_window), dtype=jnp.float64
    )
    return local, parameterization, symbolic


def _as_list(value: Any) -> list[Any]:
    return np.asarray(value).tolist()


def _poly_states(poly: Any) -> list[dict[str, Any]]:
    c = np.asarray(poly.c)[0]
    linear = np.asarray(poly.L)[0]
    time_linear = np.asarray(poly.Lt)[0]
    states: list[dict[str, Any]] = []
    variables = linear.shape[1]
    for state_index in range(c.shape[0]):
        terms: list[dict[str, Any]] = []
        zero = [0] * variables
        if float(c[state_index]) != 0.0:
            terms.append({"exponents": zero, "coefficient": float(c[state_index])})
        for variable in range(variables):
            coefficient = float(linear[state_index, variable])
            if coefficient != 0.0:
                exponent = [0] * variables
                exponent[variable] = 1
                terms.append({"exponents": exponent, "coefficient": coefficient})
            coefficient = float(time_linear[state_index, variable])
            if coefficient != 0.0:
                exponent = [0] * variables
                exponent[0] += 1
                exponent[variable] += 1
                terms.append({"exponents": exponent, "coefficient": coefficient})
        states.append({"polynomial_terms": terms})
    return states


def _states_with_remainder(model: Any) -> list[dict[str, Any]]:
    states = _poly_states(model.P)
    remainder_lo = np.asarray(model.R.lo)[0]
    remainder_hi = np.asarray(model.R.hi)[0]
    for index, state in enumerate(states):
        state["independent_interval_remainder"] = [
            float(remainder_lo[index]),
            float(remainder_hi[index]),
        ]
        state["native_structured_symbolic_remainder"] = unavailable(
            "DiffReach symbolic carry is stored in native metadata, not as a "
            "single lossless per-state interval object"
        )
    return states


def _endpoint_states(model: Any, h: float) -> list[dict[str, Any]]:
    c = np.asarray(model.P.c)[0]
    linear = np.asarray(model.P.L)[0]
    time_linear = np.asarray(model.P.Lt)[0]
    remainder_lo = np.asarray(model.R.lo)[0]
    remainder_hi = np.asarray(model.R.hi)[0]
    dimension = c.shape[0]
    states: list[dict[str, Any]] = []
    for state_index in range(dimension):
        constant = (
            float(c[state_index])
            + float(linear[state_index, 0]) * h
            + float(time_linear[state_index, 0]) * h * h
        )
        terms = [{"exponents": [0] * dimension, "coefficient": constant}]
        for generator in range(dimension):
            coefficient = (
                float(linear[state_index, generator + 1])
                + float(time_linear[state_index, generator + 1]) * h
            )
            if coefficient != 0.0:
                exponent = [0] * dimension
                exponent[generator] = 1
                terms.append({"exponents": exponent, "coefficient": coefficient})
        states.append(
            {
                "polynomial_terms": terms,
                "independent_interval_remainder": [
                    float(remainder_lo[state_index]),
                    float(remainder_hi[state_index]),
                ],
                "native_structured_symbolic_remainder": unavailable(
                    "DiffReach symbolic carry is stored in native metadata, "
                    "not as a single lossless per-state interval object"
                ),
            }
        )
    return states


def _native_samples(model: Any, domains: list[list[float]]) -> list[dict[str, Any]]:
    c = np.asarray(model.P.c)[0]
    linear = np.asarray(model.P.L)[0]
    time_linear = np.asarray(model.P.Lt)[0]
    samples = []
    for point in deterministic_points(domains, limit=16):
        z = np.asarray(point)
        values = c + linear @ z + float(z[0]) * (time_linear @ z)
        samples.append({"point": point, "polynomial_values": values.tolist()})
    return samples


def export_segment(
    spec: Mapping[str, Any],
    *,
    system_name: str,
    h: float,
    affine: bool,
) -> dict[str, Any]:
    system = spec["systems"][system_name]
    setup_started = time.perf_counter()
    dr_settings.update_config(
        {
            "TRUNCATE_TO_AFFINE": bool(affine),
            "BOUND_TIME_STEP": True,
            "DEBUG_LOG": False,
        }
    )
    windows = spec["diffreach"]["symbolic_windows"]
    symbolic_window = max(windows) if isinstance(windows, list) else int(windows)
    core = _core(spec, system, h)
    carry = _initial_carry(system, symbolic_window)
    setup_s = time.perf_counter() - setup_started
    propagation_started = time.perf_counter()
    next_carry, (_, _, contraction) = ORIGINAL_STEP_ONCE(core, carry, None)
    local_tm, parameterization, symbolic = next_carry
    composed = local_tm.compose_affine(parameterization, core.step_size)
    step_lo, step_hi, _, _ = core.step_boxes
    endpoint_lo = jnp.concatenate([step_hi[:, :1], step_lo[:, 1:]], axis=1)
    tube = composed.eval_interval(step_lo, step_hi)
    endpoint = composed.eval_interval(endpoint_lo, step_hi)
    domains = [[0.0, float(h)]] + [[-1.0, 1.0]] * len(system["state_names"])
    contraction_array = np.asarray(contraction)
    propagation_s = time.perf_counter() - propagation_started
    validation_passed = bool(np.all(contraction_array))
    export_started = time.perf_counter()
    states = _states_with_remainder(composed)
    endpoint_states = _endpoint_states(composed, h)
    raw_endpoint_box = [
        [float(lo), float(hi)]
        for lo, hi in zip(
            np.asarray(endpoint.lo)[0], np.asarray(endpoint.hi)[0]
        )
    ]
    tube_box = [
        [float(lo), float(hi)]
        for lo, hi in zip(np.asarray(tube.lo)[0], np.asarray(tube.hi)[0])
    ]
    native_samples = _native_samples(composed, domains)
    export_s = time.perf_counter() - export_started
    basis_name = (
        "B1_affine" if affine else "B_DR_restricted_quasi_quadratic"
    )
    record = canonical_record(
        tool="diffreach",
        variant="upstream_affine_flag" if affine else "upstream_restricted_quasi_quadratic",
        system=system_name,
        h=h,
        variable_names=["tau", *[f"xi_{name}" for name in system["state_names"]]],
        variable_roles=["local_time", *["state_generator"] * len(system["state_names"])],
        domains=domains,
        states=states,
        raw_endpoint=endpoint_states,
        raw_endpoint_box=raw_endpoint_box,
        tube_box=tube_box,
        validation_trace=[
            {
                "upstream_step_once": True,
                "initial_picard_contraction": contraction_array.tolist(),
                "frr_rounds": int(core.frr_rounds),
                "frr_stop_ratio": float(core.frr_stop_ratio),
            }
        ],
        reset_metadata={
            "reset": "upstream_symbolic_linear_normalization",
            "preconditioning": "diagonal_scale",
            "symbolic_window": symbolic_window,
        },
        native_metadata={
            "status": "validated" if validation_passed else "failed",
            "affine_flag": bool(affine),
            "basis": (
                "{1,tau,xi_i}" if affine
                else "{1,tau,xi_i,tau^2,tau*xi_i}"
            ),
            "dtype": str(composed.P.c.dtype),
            "jax_x64_enabled": bool(jax.config.jax_enable_x64),
            "optional_jax_verify_shim_used": OPTIONAL_IMPORT_SHIM_USED,
            "directed_rounding_or_mpfr": False,
            "floating_point_enclosure_candidate": True,
            "local_tm": {
                "c": _as_list(local_tm.P.c),
                "L": _as_list(local_tm.P.L),
                "Lt": _as_list(local_tm.P.Lt),
                "remainder_lo": _as_list(local_tm.R.lo),
                "remainder_hi": _as_list(local_tm.R.hi),
            },
            "parameterization": {
                "c": _as_list(parameterization.P.c),
                "L": _as_list(parameterization.P.L),
                "Lt": _as_list(parameterization.P.Lt),
                "remainder_lo": _as_list(parameterization.R.lo),
                "remainder_hi": _as_list(parameterization.R.hi),
            },
            "native_point_samples": native_samples,
        },
        system_definition={
            "name": system_name,
            "state_names": list(system["state_names"]),
            "equations": system["rhs"],
            "initial_domain": system["initial_box"],
        },
        accepted_step=h if validation_passed else None,
        outcome={
            "status": "success" if validation_passed else "rejection",
            "category": (
                "" if validation_passed else "picard_contraction_rejected"
            ),
            "reason": (
                ""
                if validation_passed
                else "upstream step_once contraction predicate was false"
            ),
            "requested_horizon_reached": validation_passed,
        },
        execution_metadata={
            "backend": "jax",
            "dtype": str(composed.P.c.dtype),
            "device": str(jax.devices()[0]),
            "repository_commit": git_sha(
                spec["repositories"]["diffreach"]
            ),
            "runtime": {
                "setup_s": setup_s,
                "propagation_s": propagation_s,
                "export_s": export_s,
            },
        },
        basis_metadata={
            "name": basis_name,
            "requested_order": unavailable(
                "DiffReach selects a restricted basis rather than a total order"
            ),
            "native_order": 1 if affine else 2,
            "coefficient_representation": (
                "DiffReach dense c/L/Lt arrays mapped losslessly to sparse terms"
            ),
        },
    )
    record["native_validation_passed"] = validation_passed
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--spec", default=str(REPO_ROOT / "benchmarks" / "canonical.yaml")
    )
    parser.add_argument("--system", default="coupled_quadratic")
    parser.add_argument("--h", type=float, default=0.01)
    parser.add_argument("--affine", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    record = export_segment(
        load_spec(args.spec),
        system_name=args.system,
        h=args.h,
        affine=args.affine,
    )
    write_json(args.output, record)
    print(args.output)


if __name__ == "__main__":
    main()
