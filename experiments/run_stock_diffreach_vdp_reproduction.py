#!/usr/bin/env python3
"""Reproduce pinned stock DiffReach VDP from a disposable clean clone."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
from typing import Any, Sequence

import numpy as np


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _builder_dtype_audit(source: Path) -> dict[str, Any]:
    """Inventory stock builder dtype sites without modifying the checkout."""

    files = (
        "run_dyn.py",
        "src/reachability.py",
        "src/symbolic_remainder.py",
        "src/interval.py",
        "src/polynomial.py",
        "src/taylor_model.py",
        "src/rhs_eval.py",
    )
    rows = []
    for relative in files:
        path = source / relative
        text = path.read_text(encoding="utf-8")
        for line_number, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if not (
                "dtype=jnp.float32" in stripped
                or "dtype = jnp.float32" in stripped
                or "jnp.float32)" in stripped
                or (
                    relative == "run_dyn.py"
                    and "jnp.array(" in stripped
                    and "dtype=" not in stripped
                )
            ):
                continue
            rows.append(
                {
                    "path": relative,
                    "line": line_number,
                    "source": stripped,
                    "declared_dtype": (
                        "implicit_default_under_jax_x64"
                        if "dtype=" not in stripped
                        and "dtype =" not in stripped
                        and "jnp.float32" not in stripped
                        else "jnp.float32_default_or_literal"
                    ),
                }
            )
    if not rows:
        raise RuntimeError("DiffReach builder dtype inventory unexpectedly empty")
    return {
        "scope": "all stock core builder default/literal dtype sites used by the VDP driver",
        "method": "source-derived inventory plus saved-output dtype inspection",
        "files": [
            {"path": relative, "sha256": _sha(source / relative)}
            for relative in files
        ],
        "sites": rows,
        "classification": "mixed_builder_dtype",
    }


def _python_paths(path: Path) -> tuple[Path, Path]:
    invoked = path.absolute()
    return invoked, invoked.resolve()


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    source = args.source.resolve()
    # Keep the invocation symlink so CPython discovers the intended conda
    # prefix; record the resolved binary separately for hashing.
    python, python_resolved = _python_paths(args.python)
    with tempfile.TemporaryDirectory(prefix="diffreach-stock-vdp-") as temporary:
        clone = Path(temporary) / "repo"
        subprocess.run(
            ["git", "clone", "--quiet", "--no-hardlinks", str(source), str(clone)],
            check=True,
        )
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=clone,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if commit != args.source_commit:
            raise RuntimeError("DiffReach disposable clone commit mismatch")
        env = dict(os.environ)
        env.update(
            {
                "CUDA_VISIBLE_DEVICES": args.cuda_visible_devices,
                "JAX_ENABLE_X64": "true",
            }
        )
        command = [
            str(python),
            "run_dyn.py",
            "config/ct_dyn/van_der_pol.yaml",
            "--sim",
            "--ver",
        ]
        started = time.perf_counter()
        completed = subprocess.run(
            command,
            cwd=clone,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        elapsed = time.perf_counter() - started
        (output / "diffreach.stdout.log").write_text(completed.stdout)
        (output / "diffreach.stderr.log").write_text(completed.stderr)
        if completed.returncode != 0:
            raise RuntimeError("stock DiffReach execution failed")
        builder_dtype_audit = _builder_dtype_audit(clone)
        native = clone / "output" / "ct_dyn" / "van_der_pol"
        copied = []
        for name in (
            "flowpipe_ver.npz",
            "trajectories_sim.npz",
            "x1_1000_64_agg.png",
            "x2_1000_64_agg.png",
            "x1_x2_1000_64_agg.png",
        ):
            source_path = native / name
            if not source_path.is_file():
                raise RuntimeError(f"stock DiffReach artifact missing: {name}")
            destination = output / name
            shutil.copy2(source_path, destination)
            copied.append(destination)
    saved_array_dtypes: dict[str, Any] = {}
    for path in copied:
        if path.suffix != ".npz":
            continue
        with np.load(path) as archive:
            saved_array_dtypes[path.name] = {
                name: {
                    "dtype": str(archive[name].dtype),
                    "shape": list(archive[name].shape),
                }
                for name in archive.files
            }
    endpoint_pattern = re.compile(r"x([12])\(T\)\s*∈\s*\[([^,]+),\s*([^\]]+)\]")
    endpoints = {
        f"x{component}": [float(lo), float(hi)]
        for component, lo, hi in endpoint_pattern.findall(completed.stdout)
    }
    timing_events = [
        {
            "sequence": sequence,
            "label": label,
            "seconds": float(value),
        }
        for sequence, (label, value) in enumerate(
            re.findall(r"\[(warmup|after-JIT)\]\s*([0-9.]+)s", completed.stdout)
        )
    ]
    timings = {
        label: [event["seconds"] for event in timing_events if event["label"] == label]
        for label in ("warmup", "after-JIT")
    }
    summary = {
        "schema": "stock_diffreach_vdp_reproduction_v1",
        "source_commit": args.source_commit,
        "python_executable": {
            "invoked_path": str(python),
            "resolved_path": str(python_resolved),
            "sha256": _sha(python_resolved),
        },
        "environment": {
            "JAX_ENABLE_X64": "true",
            "CUDA_VISIBLE_DEVICES": args.cuda_visible_devices,
        },
        "builder_dtype_audit": builder_dtype_audit,
        "saved_array_dtypes": saved_array_dtypes,
        "model_sha256": args.model_sha256,
        "initial_set": [[1.1, 1.4], [2.35, 2.45]],
        "partition_count": 64,
        "representation": "restricted_fixed_support_DR7",
        "validator": "DR-RP",
        "schedule": {
            "kind": "fixed",
            "h": 0.01,
            "h_hex": float(0.01).hex(),
            "steps": 1000,
        },
        "horizon_requested": 10.0,
        "horizon_requested_hex": float(10.0).hex(),
        "horizon_validated": 10.0,
        "horizon_validated_hex": float(10.0).hex(),
        "result_status": "completed",
        "endpoint_available": True,
        "segment_tube_available": False,
        "prefix_tube_available": False,
        "endpoint": endpoints,
        "reported_timing_events_in_output_order": timing_events,
        "reported_timings_by_label": timings,
        "process_wall_seconds": elapsed,
        "soundness_scope": "empirically sampled native mixed-builder-dtype lane",
        "eligibility_status": "native_capability_only",
        "artifacts": {
            path.name: {"bytes": path.stat().st_size, "sha256": _sha(path)}
            for path in copied
        },
    }
    _write(output / "summary.json", summary)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--cuda-visible-devices", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    print(json.dumps(run(parse_args(argv)), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
