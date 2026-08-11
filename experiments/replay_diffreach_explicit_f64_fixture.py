#!/usr/bin/env python3
"""Recreate the frozen one-step DR7 fixture with pinned DiffReach operators."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types
from typing import Any, Mapping, Sequence


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _compare(expected: Mapping[str, Any], actual: Mapping[str, Any]) -> list[str]:
    fields = (
        "poly1_slots",
        "poly2_slots",
        "initial_inclusion_mask",
        "round_masks",
        "round_accepted_lo",
        "round_accepted_hi",
        "tube_lo",
        "tube_hi",
        "endpoint_lo",
        "endpoint_hi",
    )
    return [field for field in fields if expected.get(field) != actual.get(field)]


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    source = args.diffreach_root.resolve()
    fixture_path = args.fixture.resolve()
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=source,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != args.source_commit:
        raise RuntimeError("DiffReach explicit-f64 source commit mismatch")
    sys.path.insert(0, str(source))

    import jax

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    import jax.numpy as jnp
    import numpy as np

    try:
        import jax_verify  # type: ignore  # noqa: F401
        optional_shim = False
    except ModuleNotFoundError:
        optional_shim = True
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

    import src.reachability as reachability
    import src.settings as settings
    from models.dynamics.ct_dyn.van_der_pol import dynamics

    settings.update_config(
        {"TRUNCATE_TO_AFFINE": False, "BOUND_TIME_STEP": True, "DEBUG_LOG": False}
    )
    expected = json.loads(fixture_path.read_text(encoding="utf-8"))
    h = float(expected["h"])
    centers = jnp.asarray([[1.25, 2.4], [-0.3, 0.7]], dtype=jnp.float64)
    scales = jnp.asarray([[0.15, 0.05], [0.2, 0.1]], dtype=jnp.float64)
    step_lo, step_hi, _, _ = reachability._make_step_boxes(
        2, 2, h, dtype=jnp.float64
    )
    core = reachability.CT_Dyn_Reach(
        rhs=dynamics,
        state_dim=2,
        nn_dyn=False,
        step_size=h,
        init_remainder=0.01,
        frr_rounds=10,
        frr_stop_ratio=0.95,
        sr_window_size=1000,
    )
    new_x0 = reachability.build_linear_tm(centers, scales, dtype=jnp.float64)
    base_poly = new_x0.P
    poly1 = base_poly.add(
        core.rhs_poly_fn(base_poly, step_lo, step_hi).integrate_time_trunc()
    )
    poly2 = base_poly.add(
        core.rhs_poly_fn(poly1, step_lo, step_hi).integrate_time_trunc()
    )

    def slots(poly: Any) -> list[Any]:
        return np.concatenate(
            (
                np.asarray(poly.c)[..., None],
                np.asarray(poly.L),
                np.asarray(poly.Lt),
            ),
            axis=-1,
        ).tolist()

    current = reachability.init_remainder_abs(
        reachability.QuadTM.from_poly(poly2), 0.01
    )
    first_rhs = core.rhs_tm_fn(current, step_lo, step_hi)
    first_delta = first_rhs.integrate_time(h, step_lo, step_hi)
    first_next = new_x0.add(first_delta)
    initial_mask = np.asarray(first_next.R.subseteq_elem(current.R)).tolist()
    roundoff = first_next.P.sub(current.P).eval_interval(step_lo, step_hi)
    masks = []
    accepted_lo = []
    accepted_hi = []
    for _ in range(10):
        rhs_tm = core.rhs_tm_fn(current, step_lo, step_hi)
        delta = rhs_tm.integrate_time(h, step_lo, step_hi)
        next_model = new_x0.add(delta)
        next_remainder = next_model.R.add(roundoff)
        mask = next_remainder.subseteq_elem(current.R)
        next_model.R.lo = jnp.where(mask, next_remainder.lo, current.R.lo)
        next_model.R.hi = jnp.where(mask, next_remainder.hi, current.R.hi)
        current = next_model
        masks.append(np.asarray(mask).tolist())
        accepted_lo.append(np.asarray(current.R.lo).tolist())
        accepted_hi.append(np.asarray(current.R.hi).tolist())
    endpoint_lo = jnp.concatenate([step_hi[:, :1], step_lo[:, 1:]], axis=1)
    tube = current.eval_interval(step_lo, step_hi)
    endpoint = current.eval_interval(endpoint_lo, step_hi)
    actual = {
        "schema": expected["schema"],
        "source_sha": commit,
        "dtype": str(current.P.c.dtype),
        "h": h,
        "poly1_slots": slots(poly1),
        "poly2_slots": slots(poly2),
        "initial_inclusion_mask": initial_mask,
        "round_masks": masks,
        "round_accepted_lo": accepted_lo,
        "round_accepted_hi": accepted_hi,
        "tube_lo": np.asarray(tube.lo).tolist(),
        "tube_hi": np.asarray(tube.hi).tolist(),
        "endpoint_lo": np.asarray(endpoint.lo).tolist(),
        "endpoint_hi": np.asarray(endpoint.hi).tolist(),
    }
    replay_path = output / "replayed_fixture.json"
    replay_path.write_text(
        json.dumps(actual, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    mismatches = _compare(expected, actual)
    report = {
        "schema": "diffreach_dr7_explicit_f64_fixture_replay_v1",
        "outcome": (
            "DIFFREACH_EXPLICIT_F64_FIXTURE_REPRODUCED"
            if not mismatches
            else "DIFFREACH_EXPLICIT_F64_FIXTURE_MISMATCH"
        ),
        "source_commit": commit,
        "jax_version": jax.__version__,
        "jax_x64_enabled": bool(jax.config.x64_enabled),
        "devices": [str(device) for device in jax.devices()],
        "optional_jax_verify_shim_used": optional_shim,
        "fixture_sha256": _sha(fixture_path),
        "replay_sha256": _sha(replay_path),
        "mismatched_fields": mismatches,
        "compared_fields": 10,
        "scope": "pinned DiffReach one-step operators with every builder explicitly float64",
    }
    (output / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if mismatches:
        raise RuntimeError(f"explicit-f64 fixture mismatch: {mismatches}")
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diffreach-root", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    print(json.dumps(run(parse_args(argv)), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
