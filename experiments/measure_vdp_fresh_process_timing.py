#!/usr/bin/env python3
"""Three rotated fresh-process repetitions for eligible internal Torch lanes."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

LANES = {
    "complete_o4_baseline_t0p1": {
        "expected_exit": 0,
        "summary_runtime": "runtime_s",
        "argv": [
            "python", "experiments/run_vdp_dense_backend.py", "--tm-backend", "dense",
            "--device", "cpu", "--wall-cap-s", "300", "--reset-mode", "normalized_insertion",
            "--dense-range-method", "adaptive_subdivision", "--dense-range-trigger",
            "proactive_depth1_on_named_contexts", "--dense-range-max-depth", "1",
            "--dense-range-max-leaves", "4", "--dense-range-split-vars", "0,1",
            "--dense-range-contexts", "polynomial_truncation", "--no-save-terminal-checkpoint",
            "--horizon", "0.1",
        ],
    },
    "complete_o4_carry_t0p1": {
        "expected_exit": 1,
        "summary_runtime": "runtime_s",
        "argv": [
            "python", "experiments/run_vdp_dense_backend.py", "--tm-backend", "dense",
            "--device", "cpu", "--wall-cap-s", "300", "--reset-mode",
            "normalized_insertion_complete_polynomial", "--dense-range-method",
            "adaptive_subdivision", "--dense-range-trigger", "proactive_depth1_on_named_contexts",
            "--dense-range-max-depth", "1", "--dense-range-max-leaves", "4",
            "--dense-range-split-vars", "0,1", "--dense-range-contexts",
            "polynomial_truncation", "--no-save-terminal-checkpoint", "--horizon", "0.1",
        ],
    },
    "fixed_dr7_b64_t0p1": {
        "expected_exit": 0,
        "summary_runtime": "cold_warm_core_process_runtime.cold_s",
        "argv": [
            "python", "experiments/run_vdp_fixed_support.py", "--horizon", "0.1",
            "--step-size", "0.01", "--batch", "64", "--device", "cpu", "--warm-runs", "0",
        ],
    },
}

ROTATIONS = (
    ("complete_o4_baseline_t0p1", "complete_o4_carry_t0p1", "fixed_dr7_b64_t0p1"),
    ("complete_o4_carry_t0p1", "fixed_dr7_b64_t0p1", "complete_o4_baseline_t0p1"),
    ("fixed_dr7_b64_t0p1", "complete_o4_baseline_t0p1", "complete_o4_carry_t0p1"),
)


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _nested(value: dict[str, Any], dotted: str) -> float:
    current: Any = value
    for key in dotted.split("."):
        current = current[key]
    return float(current)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    source_sha = _git("rev-parse", "HEAD")
    start_status = _git("status", "--porcelain")
    if start_status:
        raise RuntimeError("fresh-process timing requires a clean source worktree")

    rows: list[dict[str, Any]] = []
    for repetition, rotation in enumerate(ROTATIONS):
        for position, lane in enumerate(rotation):
            spec = LANES[lane]
            run_dir = args.output_dir / f"rep{repetition}_{position}_{lane}"
            argv = ["conda", "run", "-n", "py11", *spec["argv"], "--output-dir", str(run_dir)]
            started = time.perf_counter()
            process = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True)
            wall_s = time.perf_counter() - started
            if process.returncode != spec["expected_exit"]:
                raise RuntimeError(
                    f"{lane} repetition {repetition} exit {process.returncode}; "
                    f"expected {spec['expected_exit']}: {process.stderr[-1000:]}"
                )
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            core_s = _nested(summary, spec["summary_runtime"])
            rows.append(
                {
                    "repetition": repetition,
                    "rotation_position": position,
                    "lane": lane,
                    "fresh_process_wall_s": wall_s,
                    "certification_core_s": core_s,
                    "startup_import_config_serialization_composite_s": max(wall_s - core_s, 0.0),
                    "compile_jit_s": 0.0,
                    "requested_horizon": 0.1,
                    "validated_horizon": summary.get("validated_horizon", summary.get("completed_horizon")),
                    "completion": summary.get("completion_status", summary.get("status")),
                    "expected_exit": spec["expected_exit"],
                    "actual_exit": process.returncode,
                    "stdout_sha256": hashlib.sha256(process.stdout.encode()).hexdigest(),
                    "stderr_sha256": hashlib.sha256(process.stderr.encode()).hexdigest(),
                    "artifact": run_dir.relative_to(args.output_dir).as_posix() + "/summary.json",
                }
            )

    final_status = _git("status", "--porcelain")
    result = {
        "schema": "torch_tm_fresh_process_timing_v1",
        "source_sha": source_sha,
        "start_worktree_clean": not bool(start_status),
        "end_worktree_clean": not bool(final_status),
        "rotation_order": [list(rotation) for rotation in ROTATIONS],
        "boundary_note": (
            "The wrapper wall includes process startup, imports, configuration, and reporting. "
            "Only their composite difference from the runner certification core is inferable; "
            "no cross-tool deployment ratio is claimed."
        ),
        "rows": rows,
    }
    (args.output_dir / "fresh_process_timing.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    fields = sorted({key for row in rows for key in row})
    with (args.output_dir / "fresh_process_timing.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"source_sha": source_sha, "rows": len(rows)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
