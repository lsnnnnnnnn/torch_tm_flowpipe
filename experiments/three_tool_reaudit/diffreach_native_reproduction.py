#!/usr/bin/env python3
"""Reproduce upstream DiffReach VDP while enforcing fail-closed completion.

The upstream launcher reports a Picard contraction rate but continues through
all scan iterations when the rate is below one.  This wrapper preserves the
native computation and records that returned horizon separately from the last
validated horizon.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import types
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def effective_support(dimension: int) -> list[list[int]]:
    """Return DiffReach's explicit {1,z_i,t*z_i} exponent support."""
    variables = dimension + 1
    support: set[tuple[int, ...]] = {tuple([0] * variables)}
    for variable in range(variables):
        exponent = [0] * variables
        exponent[variable] = 1
        support.add(tuple(exponent))
        time_product = exponent.copy()
        time_product[0] += 1
        support.add(tuple(time_product))
    return [list(item) for item in sorted(support)]


def support_sha256(dimension: int) -> str:
    payload = json.dumps(
        effective_support(dimension), separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def classify_completion(
    *,
    contraction: np.ndarray,
    lowers: np.ndarray,
    uppers: np.ndarray,
    step_size: float,
) -> dict[str, Any]:
    """Classify the first unusable step; returned scan length is not success."""
    if contraction.ndim < 1:
        raise ValueError("contraction trace must have a step dimension")
    n_steps = int(contraction.shape[0])
    if lowers.shape != uppers.shape or lowers.ndim != 3:
        raise ValueError("lowers/uppers must have identical [B,T,D] shapes")
    if lowers.shape[1] != n_steps + 1:
        raise ValueError("interval trace and contraction trace disagree")

    contraction_by_step = np.all(contraction, axis=tuple(range(1, contraction.ndim)))
    finite_by_step = np.all(
        np.isfinite(lowers) & np.isfinite(uppers) & (lowers <= uppers),
        axis=(0, 2),
    )[1:]
    bad_contraction = np.flatnonzero(~contraction_by_step)
    bad_finite = np.flatnonzero(~finite_by_step)
    failures: list[tuple[int, str]] = []
    if bad_contraction.size:
        failures.append((int(bad_contraction[0]), "picard_contraction_rejected"))
    if bad_finite.size:
        failures.append((int(bad_finite[0]), "nonfinite_or_invalid_interval"))
    if failures:
        first_index, category = min(failures)
        return {
            "validation_status": "validation_rejected",
            "failure_category": category,
            "first_failed_step_index": first_index,
            "first_failed_step_number": first_index + 1,
            "completed_horizon": first_index * float(step_size),
            "requested_horizon_reached": False,
            "upstream_scan_returned_horizon": n_steps * float(step_size),
        }
    return {
        "validation_status": "completed",
        "failure_category": None,
        "first_failed_step_index": None,
        "first_failed_step_number": None,
        "completed_horizon": n_steps * float(step_size),
        "requested_horizon_reached": True,
        "upstream_scan_returned_horizon": n_steps * float(step_size),
    }


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), *args], text=True
    ).strip()


def run_native(
    *,
    diffreach_root: Path,
    horizon: float,
    output: Path,
    steady_runs: int,
    rhs_route: str,
) -> dict[str, Any]:
    diffreach_root = diffreach_root.resolve()
    if str(diffreach_root) not in sys.path:
        sys.path.insert(0, str(diffreach_root))

    import jax

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    import jax.numpy as jnp
    import yaml

    optional_import_shim_used = False
    try:
        import jax_verify  # type: ignore  # noqa: F401
    except ModuleNotFoundError:
        # The analytic plant path never calls CROWN.  Keep that path runnable
        # without silently pretending the missing controller dependency exists.
        optional_import_shim_used = True
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

    from src.reachability import CT_Dyn_Reach
    from src.utils.box_set import prepare_initial_sets

    config_path = diffreach_root / "config" / "ct_dyn" / "van_der_pol.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    step_size = float(config["step_size"])
    requested_steps = horizon / step_size
    n_steps = int(round(requested_steps))
    if abs(requested_steps - n_steps) > 1e-12:
        raise ValueError("horizon must be an integer multiple of native step_size")

    initial_set = jnp.asarray(config["initial_set"])
    x0_lo, x0_hi = prepare_initial_sets(
        initial_set,
        {int(key): int(value) for key, value in config["splits"].items()},
        int(config["state_dim"]),
    )
    dynamics_path = diffreach_root / str(config["dynamics"])
    dynamics_spec = importlib.util.spec_from_file_location(
        "diffreach_native_vdp_dynamics", dynamics_path
    )
    if dynamics_spec is None or dynamics_spec.loader is None:
        raise ImportError(f"cannot load dynamics: {dynamics_path}")
    dynamics_module = importlib.util.module_from_spec(dynamics_spec)
    dynamics_spec.loader.exec_module(dynamics_module)
    if rhs_route == "upstream-model":
        rhs = dynamics_module.dynamics
    elif rhs_route == "canonical-polynomial-adapter":
        # Mathematically identical ODE expressed only with primitives supported
        # by the current quadratic RHS interpreter.  This is a matched adapter,
        # never labeled as the upstream native launcher route.
        def rhs(state: Any) -> Any:
            x1, x2 = state[0], state[1]
            return jnp.concatenate(
                [
                    jnp.reshape(x2, (1,)),
                    jnp.reshape((1.0 - x1 * x1) * x2 - x1, (1,)),
                ],
                axis=0,
            )
    else:  # pragma: no cover - argparse constrains this
        raise ValueError(f"unknown rhs route: {rhs_route}")
    reach = CT_Dyn_Reach(
        rhs=rhs,
        state_dim=int(config["state_dim"]),
        nn_dyn=False,
        step_size=step_size,
        init_remainder=float(config["init_remainder"]),
        frr_rounds=int(config["frr_rounds"]),
        frr_stop_ratio=float(config["frr_stop_ratio"]),
        sr_window_size=min(int(config["sr_window_size"]), n_steps),
    )
    verify = jax.jit(reach.verify, static_argnames=("n_total_steps",))

    def execute() -> tuple[Any, ...]:
        started = time.perf_counter()
        times, lowers, uppers, final_tm, contraction = verify(
            x0_lo, x0_hi, n_steps
        )
        jax.block_until_ready(uppers)
        return (
            time.perf_counter() - started,
            times,
            lowers,
            uppers,
            final_tm,
            contraction,
        )

    try:
        cold = execute()
    except Exception as error:
        result = {
            "schema_version": "diffreach-native-reproduction-1.0.0",
            "backend": "diffreach-native",
            "lane": "native_reproduction",
            "run_authority": "authoritative",
            "command": sys.argv,
            "repository": {
                "path": str(diffreach_root),
                "remote": _git(diffreach_root, "remote", "get-url", "origin"),
                "branch": _git(diffreach_root, "rev-parse", "--abbrev-ref", "HEAD"),
                "sha": _git(diffreach_root, "rev-parse", "HEAD"),
                "dirty": bool(_git(diffreach_root, "status", "--porcelain")),
            },
            "config": {
                "path": str(config_path),
                "sha256": sha256_file(config_path),
                "dynamics_path": str(dynamics_path),
                "dynamics_sha256": sha256_file(dynamics_path),
                "rhs_route": rhs_route,
                "requested_horizon": float(horizon),
                "step_size": step_size,
                "n_steps": n_steps,
                "partitions": int(x0_lo.shape[0]),
            },
            "execution": {
                "jax_version": jax.__version__,
                "jax_enable_x64": bool(jax.config.jax_enable_x64),
                "devices": [str(device) for device in jax.devices()],
                "optional_jax_verify_shim_used": optional_import_shim_used,
            },
            "completion": {
                "validation_status": "unsupported_configuration",
                "failure_category": "unsupported_native_rhs_primitive",
                "completed_horizon": 0.0,
                "requested_horizon_reached": False,
                "upstream_scan_returned_horizon": 0.0,
            },
            "failure": {
                "exception_type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
                "fail_closed": True,
            },
            "soundness": {
                "level": "unknown",
                "reason": "native route did not produce an enclosure",
            },
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        return result
    steady: list[tuple[Any, ...]] = [execute() for _ in range(steady_runs)]
    selected = steady[-1] if steady else cold
    wall_s, times, lowers, uppers, final_tm, contraction = selected
    del final_tm
    lowers_np = np.asarray(lowers)
    uppers_np = np.asarray(uppers)
    contraction_np = np.asarray(contraction, dtype=bool)
    classification = classify_completion(
        contraction=contraction_np,
        lowers=lowers_np,
        uppers=uppers_np,
        step_size=step_size,
    )
    per_step = np.all(
        contraction_np, axis=tuple(range(1, contraction_np.ndim))
    )
    support = effective_support(int(config["state_dim"]))
    result = {
        "schema_version": "diffreach-native-reproduction-1.0.0",
        "backend": (
            "diffreach-native"
            if rhs_route == "upstream-model"
            else "diffreach-canonical-adapter"
        ),
        "lane": (
            "native_reproduction"
            if rhs_route == "upstream-model"
            else "matched_plant_backend"
        ),
        "run_authority": "authoritative",
        "command": sys.argv,
        "repository": {
            "path": str(diffreach_root),
            "remote": _git(diffreach_root, "remote", "get-url", "origin"),
            "branch": _git(diffreach_root, "rev-parse", "--abbrev-ref", "HEAD"),
            "sha": _git(diffreach_root, "rev-parse", "HEAD"),
            "dirty": bool(_git(diffreach_root, "status", "--porcelain")),
        },
        "config": {
            "path": str(config_path),
            "sha256": sha256_file(config_path),
            "dynamics_path": str(dynamics_path),
            "dynamics_sha256": sha256_file(dynamics_path),
            "initial_set": config["initial_set"],
            "splits": config["splits"],
            "partitions": int(x0_lo.shape[0]),
            "step_size": step_size,
            "requested_horizon": float(horizon),
            "n_steps": n_steps,
            "init_remainder": float(config["init_remainder"]),
            "frr_rounds": int(config["frr_rounds"]),
            "frr_stop_ratio": float(config["frr_stop_ratio"]),
            "symbolic_window": min(int(config["sr_window_size"]), n_steps),
            "rhs_route": rhs_route,
        },
        "execution": {
            "jax_version": jax.__version__,
            "jax_enable_x64": bool(jax.config.jax_enable_x64),
            "actual_model_dtype": str(lowers.dtype),
            "devices": [str(device) for device in jax.devices()],
            "optional_jax_verify_shim_used": optional_import_shim_used,
            "cold_compile_and_execute_s": float(cold[0]),
            "steady_execute_s": [float(item[0]) for item in steady],
            "selected_execute_s": float(wall_s),
            "thread_count": os.environ.get("OMP_NUM_THREADS", "unspecified"),
        },
        "basis": {
            "name": "restricted_quasi_quadratic",
            "variable_order": ["tau", "xi_x", "xi_y"],
            "effective_support": support,
            "effective_support_sha256": support_sha256(
                int(config["state_dim"])
            ),
        },
        "soundness": {
            "level": "unknown",
            "directed_rounding_or_mpfr": False,
            "reason": "native JAX operations use round-to-nearest; no complete outward-rounding proof was found",
        },
        "completion": classification,
        "native_behavior": {
            "verify_returned_all_requested_scan_steps": int(np.asarray(times).size)
            == n_steps + 1,
            "contraction_shape": list(contraction_np.shape),
            "contraction_true": int(np.count_nonzero(contraction_np)),
            "contraction_total": int(contraction_np.size),
            "contraction_rate": float(np.mean(contraction_np)),
            "steps_all_dimensions_contracted": int(np.count_nonzero(per_step)),
            "first_failed_step_mask": (
                contraction_np[int(classification["first_failed_step_index"])].tolist()
                if classification["first_failed_step_index"] is not None
                else None
            ),
            "warning_is_not_promoted_to_process_failure_upstream": True,
            "wrapper_fail_closed": True,
        },
        "output": {
            "endpoint_box": [
                [float(value) for value in pair]
                for pair in zip(
                    np.min(lowers_np[:, -1, :], axis=0),
                    np.max(uppers_np[:, -1, :], axis=0),
                )
            ],
            "raw_endpoint_supported": False,
            "last_segment_supported": False,
            "full_tube_boxes_returned": True,
            "note": "verify returns interval boxes, not lossless native raw endpoint Taylor models",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--diffreach-root",
        type=Path,
        default=Path(os.environ.get("DIFFREACH_ROOT", REPO_ROOT.parent / "DiffReach")),
    )
    parser.add_argument("--horizon", type=float, required=True)
    parser.add_argument("--steady-runs", type=int, default=1)
    parser.add_argument(
        "--rhs-route",
        choices=("upstream-model", "canonical-polynomial-adapter"),
        default="upstream-model",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_native(
        diffreach_root=args.diffreach_root,
        horizon=args.horizon,
        output=args.output,
        steady_runs=args.steady_runs,
        rhs_route=args.rhs_route,
    )
    print(json.dumps(result["completion"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
