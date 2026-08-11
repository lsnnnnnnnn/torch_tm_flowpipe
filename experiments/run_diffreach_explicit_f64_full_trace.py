#!/usr/bin/env python3
"""Run patched pinned DiffReach directly with explicit-f64 full-horizon tracing."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
import types
from typing import Any

from diffreach_torch_full_horizon_common import (
    DIFFREACH_SOURCE_SHA,
    PARTITION_SHA256,
    SCHEMA,
    SUPPORT_SHA256,
    array_record,
    capture_npz,
    parse_capture_steps,
    partition_arrays,
    records_for_fields,
    write_json,
    write_jsonl_row,
)


def _git(root: Path, *args: str, binary: bool = False) -> str | bytes:
    result = subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=not binary
    )
    return result.stdout


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _install_optional_import_shim() -> bool:
    try:
        import jax_verify  # type: ignore  # noqa: F401

        return False
    except ModuleNotFoundError:
        stub = types.ModuleType("jax_verify")

        class _Unavailable:
            def __init__(self, *_: Any, **__: Any) -> None:
                raise RuntimeError("unused neural dependency is unavailable")

        def unavailable(*_: Any, **__: Any) -> Any:
            raise RuntimeError("unused neural dependency is unavailable")

        stub.IntervalBound = _Unavailable
        stub.backward_crown_bound_propagation = unavailable
        sys.modules["jax_verify"] = stub
        crown_stub = types.ModuleType("src.crown_wrapper")
        crown_stub.crown = unavailable
        sys.modules["src.crown_wrapper"] = crown_stub
        return True


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diffreach-root", type=Path, required=True)
    parser.add_argument("--observer-patch", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--step-size", type=float, default=0.01)
    parser.add_argument("--symbolic-window", type=int, default=1000)
    parser.add_argument("--capture-steps", default="1,2,62,63,999,1000")
    return parser.parse_args()


def main() -> int:
    args = _args()
    source = args.diffreach_root.resolve()
    patch_path = args.observer_patch.resolve()
    if args.steps <= 0 or args.step_size != float.fromhex("0x1.47ae147ae147bp-7"):
        raise ValueError("the frozen lane requires positive steps and binary64 h=0.01")
    if args.symbolic_window != 1000:
        raise ValueError("the frozen lane requires symbolic window 1000")
    commit = str(_git(source, "rev-parse", "HEAD")).strip()
    if commit != DIFFREACH_SOURCE_SHA:
        raise RuntimeError(f"DiffReach source commit mismatch: {commit}")
    status = str(_git(source, "status", "--short")).splitlines()
    if status != [" M src/picard.py", " M src/reachability.py"]:
        raise RuntimeError(f"unexpected patched DiffReach worktree state: {status}")
    actual_diff = _git(source, "diff", "--binary", binary=True)
    actual_patch_sha = hashlib.sha256(actual_diff).hexdigest()
    expected_patch_sha = _file_sha256(patch_path)
    if actual_patch_sha != expected_patch_sha:
        raise RuntimeError(
            f"DiffReach observer patch mismatch: {actual_patch_sha} != {expected_patch_sha}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=False)
    capture_dir = args.output_dir / "captures"
    capture_dir.mkdir()
    captures = parse_capture_steps(args.capture_steps)
    if any(step > args.steps for step in captures):
        raise ValueError("capture step exceeds requested steps")

    os.environ.setdefault("JAX_ENABLE_X64", "true")
    os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
    sys.path.insert(0, str(source))
    import jax

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    import jax.numpy as jnp
    import numpy as np

    optional_shim = _install_optional_import_shim()
    import src.reachability as reachability
    import src.settings as settings
    from models.dynamics.ct_dyn.van_der_pol import dynamics
    from src.symbolic_remainder import init_symbolic_state

    settings.update_config(
        {"TRUNCATE_TO_AFFINE": False, "BOUND_TIME_STEP": True, "DEBUG_LOG": False}
    )
    initial_lo_np, initial_hi_np = partition_arrays()
    initial_lo = jnp.asarray(initial_lo_np, dtype=jnp.float64)
    initial_hi = jnp.asarray(initial_hi_np, dtype=jnp.float64)
    core = reachability.CT_Dyn_Reach(
        rhs=dynamics,
        state_dim=2,
        nn_dyn=False,
        step_size=args.step_size,
        init_remainder=0.01,
        frr_rounds=10,
        frr_stop_ratio=0.95,
        sr_window_size=args.symbolic_window,
    )
    core.step_boxes = reachability._make_step_boxes(
        B=64, D=2, h=args.step_size, dtype=jnp.float64
    )
    center = 0.5 * (initial_lo + initial_hi)
    scale = 0.5 * (initial_hi - initial_lo)
    model = reachability.build_linear_tm(center, scale, dtype=jnp.float64)
    parameterization = reachability.identity_parameterization(
        64, 2, 3, dtype=jnp.float64
    )
    symbolic = init_symbolic_state(
        64, 2, M=min(args.symbolic_window, args.steps), dtype=jnp.float64
    )
    initial_carry = (model, parameterization, symbolic)

    native_step = jax.jit(core.step_once)
    observed_step = jax.jit(core.step_once_observed)
    native_carry, native_output = native_step(initial_carry, None)
    observed_carry, observed_output, _ = observed_step(initial_carry, None)
    jax.block_until_ready(observed_carry)
    native_leaves = jax.tree.leaves((native_carry, native_output))
    observed_leaves = jax.tree.leaves((observed_carry, observed_output))
    inertness_equal = len(native_leaves) == len(observed_leaves) and all(
        np.array_equal(np.asarray(left), np.asarray(right))
        for left, right in zip(native_leaves, observed_leaves)
    )
    if not inertness_equal:
        raise RuntimeError("DiffReach observer changed native step outputs")

    write_json(
        args.output_dir / "command.json",
        {
            "schema": SCHEMA,
            "tool": "pinned_diffreach_explicit_f64",
            "argv": sys.argv,
            "cwd": str(Path.cwd()),
            "python": platform.python_version(),
            "jax": jax.__version__,
            "devices": [str(device) for device in jax.devices()],
            "source_sha": commit,
            "observer_patch_sha256": expected_patch_sha,
            "partition_sha256": PARTITION_SHA256,
            "support_sha256": SUPPORT_SHA256,
        },
    )
    trace_path = args.output_dir / "trace.jsonl"
    prefix_lo = np.full((2,), np.inf, dtype=np.float64)
    prefix_hi = np.full((2,), -np.inf, dtype=np.float64)
    all_initial = True
    later_pass = 0
    later_total = 0
    endpoint_los: list[np.ndarray] = []
    endpoint_his: list[np.ndarray] = []
    tube_los: list[np.ndarray] = []
    tube_his: list[np.ndarray] = []
    carry = initial_carry
    started = time.perf_counter()
    with trace_path.open("w", encoding="utf-8") as trace_handle:
        for zero_index in range(args.steps):
            next_carry, output, trace = observed_step(carry, None)
            jax.block_until_ready(next_carry)
            fields = {name: np.asarray(value) for name, value in trace.items()}
            prefix_lo = np.minimum(prefix_lo, fields["tube_lo"].min(axis=0))
            prefix_hi = np.maximum(prefix_hi, fields["tube_hi"].max(axis=0))
            fields["prefix_tube_hull_lo"] = prefix_lo.copy()
            fields["prefix_tube_hull_hi"] = prefix_hi.copy()
            step_number = zero_index + 1
            write_jsonl_row(
                trace_handle,
                {
                    "schema": SCHEMA,
                    "tool": "pinned_diffreach_explicit_f64",
                    "step": step_number,
                    "time": step_number * args.step_size,
                    "time_hex": float(step_number * args.step_size).hex(),
                    "fields": records_for_fields(fields),
                },
            )
            if step_number in captures:
                capture_npz(capture_dir / f"step_{step_number:04d}.npz", fields)
            endpoint_los.append(fields["endpoint_lo"].copy())
            endpoint_his.append(fields["endpoint_hi"].copy())
            tube_los.append(fields["tube_lo"].copy())
            tube_his.append(fields["tube_hi"].copy())
            initial_mask = fields["initial_inclusion_mask"]
            initial_ok = bool(np.all(initial_mask))
            all_initial = all_initial and initial_ok
            later_pass += int(fields["round_masks"].sum())
            later_total += int(fields["round_masks"].size)
            if not initial_ok:
                raise RuntimeError(f"initial DR-RP inclusion failed at step {step_number}")
            carry = next_carry
    runtime_s = time.perf_counter() - started
    bounds_path = args.output_dir / "bounds.npz"
    np.savez_compressed(
        bounds_path,
        endpoint_lo=np.stack(endpoint_los),
        endpoint_hi=np.stack(endpoint_his),
        tube_lo=np.stack(tube_los),
        tube_hi=np.stack(tube_his),
    )

    final_trace = fields
    summary = {
        "schema": SCHEMA,
        "tool": "pinned_diffreach_explicit_f64",
        "source_sha": commit,
        "observer_patch_sha256": expected_patch_sha,
        "worktree_changes": status,
        "partition_sha256": PARTITION_SHA256,
        "support_sha256": SUPPORT_SHA256,
        "dtype": "float64",
        "device": "cpu",
        "jax_x64_enabled": bool(jax.config.x64_enabled),
        "steps": args.steps,
        "step_size": args.step_size,
        "step_size_hex": args.step_size.hex(),
        "validated_horizon": args.steps * args.step_size,
        "validated_horizon_hex": float(args.steps * args.step_size).hex(),
        "all_initial_masks_true": all_initial,
        "later_mask_pass_count": later_pass,
        "later_mask_total_count": later_total,
        "queue_count_final": int(np.asarray(final_trace["post_queue_count"])),
        "final_endpoint_hull_lo": final_trace["endpoint_lo"].min(axis=0).tolist(),
        "final_endpoint_hull_hi": final_trace["endpoint_hi"].max(axis=0).tolist(),
        "final_prefix_tube_hull_lo": prefix_lo.tolist(),
        "final_prefix_tube_hull_hi": prefix_hi.tolist(),
        "trace_sha256": _file_sha256(trace_path),
        "initial_partition_array_records": {
            "lo": array_record(initial_lo_np),
            "hi": array_record(initial_hi_np),
        },
        "runtime_with_hashing_and_capture_s": runtime_s,
        "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
        "observer_kind": "minimal_patch_on_native_CT_Dyn_Reach.step_once_operators",
        "observer_inertness_bit_exact": inertness_equal,
        "optional_jax_verify_import_shim_used": optional_shim,
        "undeclared_fallback_or_repair": False,
        "completion_status": "completed",
    }
    write_json(args.output_dir / "summary.json", summary)
    artifacts = [
        args.output_dir / "command.json",
        trace_path,
        bounds_path,
        args.output_dir / "summary.json",
    ]
    artifacts.extend(sorted(capture_dir.glob("*.npz")))
    write_json(
        args.output_dir / "artifact_manifest.json",
        {
            "schema": SCHEMA,
            "files": [
                {
                    "path": path.relative_to(args.output_dir).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": _file_sha256(path),
                }
                for path in artifacts
            ],
        },
    )
    print(json.dumps(summary, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
