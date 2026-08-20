#!/usr/bin/env python3
"""Run the fresh clean-detached CPU matrix for VDP C2."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "experiments/run_vdp_dense_backend.py"
C1 = "flowstar_raw_remainder_compat_factorized_joint_closure"
C2 = "flowstar_raw_remainder_compat_factorized_joint_closure_refined"
EMPTY_DIFF_SHA256 = hashlib.sha256(b"").hexdigest()
LANES = {
    "legacy": ("normalized_insertion", "flowstar_raw_remainder_compat"),
    "production_c1_candidate": ("normalized_insertion_dependency_preserving", C1),
    "production_c2_candidate": ("normalized_insertion_dependency_preserving", C2),
}
SCENARIOS = {
    "step1": (0.01, 0.01),
    "fixed_T1": (1.0, 0.01),
    "fixed_T3": (3.0, 0.01),
    "fixed_T6p32": (6.32, 0.01),
    "native_T10": (10.0, None),
}


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def run(output_dir: Path, *, wall_cap_s: float) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("scientific matrix output directory must be new or empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    status = _git("status", "--porcelain")
    diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    symbolic = subprocess.run(
        ["git", "symbolic-ref", "-q", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if status or hashlib.sha256(diff).hexdigest() != EMPTY_DIFF_SHA256:
        raise ValueError("scientific matrix requires a clean worktree")
    if symbolic.returncode == 0:
        raise ValueError("scientific matrix requires detached HEAD")
    head = _git("rev-parse", "HEAD")

    completed = []
    for scenario, (horizon, fixed_step) in SCENARIOS.items():
        for lane, (reset_mode, validation_mode) in LANES.items():
            destination = output_dir / scenario / lane
            command = [
                sys.executable,
                str(RUNNER),
                "--output-dir",
                str(destination),
                "--tm-backend",
                "dense",
                "--device",
                "cpu",
                "--horizon",
                format(horizon, ".17g"),
                "--initialization-contract",
                "exact_decimal_contract",
                "--reset-mode",
                reset_mode,
                "--validation-mode",
                validation_mode,
                "--trace-flush-every",
                "50",
                "--dense-range-method",
                "adaptive_subdivision",
                "--dense-range-trigger",
                "proactive_depth1_on_named_contexts",
                "--dense-range-max-depth",
                "1",
                "--dense-range-max-leaves",
                "4",
                "--dense-range-split-vars",
                "0,1",
                "--dense-range-contexts",
                "polynomial_truncation",
                "--wall-cap-s",
                format(float(wall_cap_s), ".17g"),
            ]
            if fixed_step is not None:
                command.extend(("--fixed-step", format(fixed_step, ".17g")))
            if scenario == "native_T10":
                command.append("--save-terminal-checkpoint")
            subprocess.run(command, cwd=ROOT, check=True)
            summary = _load(destination / "summary.json")
            if summary["commit"] != head or summary["worktree_dirty"] is not False:
                raise ValueError(f"run provenance mismatch: {scenario}/{lane}")
            if summary["tracked_diff_sha256"] != EMPTY_DIFF_SHA256:
                raise ValueError(f"run tracked diff is nonempty: {scenario}/{lane}")
            completed.append(
                {
                    "scenario": scenario,
                    "lane": lane,
                    "status": summary["status"],
                    "completed_horizon": summary["completed_horizon"],
                    "runtime_s": summary["runtime_s"],
                }
            )
    for device in ("cpu", "cuda"):
        destination = output_dir / "consistency_T0p1" / device
        command = [
            sys.executable,
            str(RUNNER),
            "--output-dir",
            str(destination),
            "--tm-backend",
            "dense",
            "--device",
            device,
            "--horizon",
            "0.1",
            "--fixed-step",
            "0.01",
            "--initialization-contract",
            "exact_decimal_contract",
            "--reset-mode",
            "normalized_insertion_dependency_preserving",
            "--validation-mode",
            C2,
            "--trace-flush-every",
            "0",
            "--dense-range-method",
            "adaptive_subdivision",
            "--dense-range-trigger",
            "proactive_depth1_on_named_contexts",
            "--dense-range-max-depth",
            "1",
            "--dense-range-max-leaves",
            "4",
            "--dense-range-split-vars",
            "0,1",
            "--dense-range-contexts",
            "polynomial_truncation",
            "--wall-cap-s",
            format(float(wall_cap_s), ".17g"),
        ]
        subprocess.run(command, cwd=ROOT, check=True)
        summary = _load(destination / "summary.json")
        if summary["commit"] != head or summary["worktree_dirty"] is not False:
            raise ValueError(f"consistency run provenance mismatch: {device}")
        completed.append(
            {
                "scenario": "consistency_T0p1",
                "lane": device,
                "status": summary["status"],
                "completed_horizon": summary["completed_horizon"],
                "runtime_s": summary["runtime_s"],
                "peak_rss_bytes": summary["peak_rss_bytes"],
            }
        )
    result = {
        "schema": "vdp_c2_fresh_cpu_matrix_run_v1",
        "scientific_sha": head,
        "clean_detached": True,
        "runs": completed,
    }
    (output_dir / "run_index.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--wall-cap-s", type=float, default=3600.0)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(run(args.output_dir, wall_cap_s=args.wall_cap_s), sort_keys=True))
