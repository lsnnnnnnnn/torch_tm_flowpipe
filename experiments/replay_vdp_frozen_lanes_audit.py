#!/usr/bin/env python3
"""Replay the 14 H1 lanes at their numerical-source commit with raw logs."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_COMMIT = "a1fb3527bb7c12ce23aa2fb49d66f6380c463c90"
STABLE_FIELDS = (
    "attempted_h",
    "t_before",
    "accepted",
    "status",
    "validation_rejection_reason",
    "candidate_hashes",
    "candidate_remainder",
    "picard_image_remainder",
    "subset_margin",
    "backend_lane",
    "fallback_count",
    "endpoint_repair_used",
    "current_hashes",
    "normalized_current_hashes",
    "contract_sha256",
    "checkpoint_full_sha256",
)


def _load_runner(source: Path):
    experiments = source / "experiments"
    sys.path.insert(0, str(experiments))
    spec = importlib.util.spec_from_file_location(
        "frozen_lane_source_runner",
        experiments / "run_vdp_later_terminal_factorized_range.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load frozen lane source runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _orders_arg(orders: Sequence[Sequence[int]]) -> str:
    return ";".join(",".join(str(index) for index in order) for order in orders)


def _expected_dir(phase: str, lane: str) -> Path:
    return (
        ROOT
        / "outputs/vdp_later_terminal_factorized_range/evidence_package/raw"
        / ("attribution" if phase == "attribution" else "terminal_ab")
        / lane
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-worktree", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    source = args.source_worktree.resolve()
    output = args.output_root.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing non-empty output root: {output}")
    output.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=source, check=True, capture_output=True, text=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=source, check=True, capture_output=True, text=True
    ).stdout
    if commit != EXPECTED_COMMIT or status:
        raise RuntimeError(f"frozen source is not clean {EXPECTED_COMMIT}: commit={commit}, status={status!r}")
    runner = _load_runner(source)
    checkpoint = runner.CHECKPOINT
    phases = (
        ("attribution", runner.ATTRIBUTION_LANES),
        ("terminal_ab", runner.TERMINAL_AB_LANES),
    )
    rows = []
    for phase, lanes in phases:
        for lane in lanes:
            lane_dir = output / phase / lane.name
            artifact_dir = lane_dir / "artifacts"
            command = [
                sys.executable,
                str(source / "experiments/replay_vdp_terminal_range.py"),
                "--checkpoint",
                str(checkpoint),
                "--output-dir",
                str(artifact_dir),
                "--range-method",
                lane.method,
                "--subdivision-depth",
                str(lane.depth),
                "--max-leaves",
                str(lane.leaves),
                "--split-vars",
                "0,1",
                "--named-contexts",
                ",".join(lane.contexts),
                "--variable-orders",
                _orders_arg(lane.orders),
                "--device",
                "cpu",
            ]
            started_at = datetime.now(timezone.utc)
            started = time.perf_counter()
            completed = subprocess.run(command, cwd=source, capture_output=True)
            runtime_s = time.perf_counter() - started
            lane_dir.mkdir(parents=True, exist_ok=True)
            (lane_dir / "stdout.log").write_bytes(completed.stdout)
            (lane_dir / "stderr.log").write_bytes(completed.stderr)
            (lane_dir / "exit_code.txt").write_text(f"{completed.returncode}\n", encoding="utf-8")
            execution_completed = completed.returncode in {0, 1} and (artifact_dir / "summary.json").is_file()
            if not execution_completed:
                raise RuntimeError(
                    f"lane {phase}/{lane.name} failed ({completed.returncode}): "
                    f"{completed.stderr.decode('utf-8', errors='replace')}"
                )
            actual = _read(artifact_dir / "summary.json")
            expected = _read(_expected_dir(phase, lane.name) / "summary.json")
            differences = [field for field in STABLE_FIELDS if actual.get(field) != expected.get(field)]
            capture = {
                "schema": "torch_tm_flowpipe_frozen_lane_replay_v1",
                "phase": phase,
                "lane": lane.name,
                "source_commit": commit,
                "source_worktree_clean": not bool(status),
                "argv": command,
                "cwd": str(source),
                "started_at_utc": started_at.isoformat(),
                "runtime_s": runtime_s,
                "exit_code": completed.returncode,
                "exit_code_semantics": (
                    "rejected natural replay returns 1 when it intentionally differs from the proactive checkpoint "
                    "reference; reproduction is decided by stable-field comparison with the matching H1 lane"
                ),
                "execution_completed": execution_completed,
                "environment": {
                    "python": sys.version,
                    "executable": sys.executable,
                    "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
                },
                "stable_fields": list(STABLE_FIELDS),
                "differences": differences,
                "reproduced": not differences,
            }
            _write(lane_dir / "capture.json", capture)
            rows.append(capture)
    result = {
        "schema": "torch_tm_flowpipe_frozen_lanes_reproduction_v1",
        "source_commit": commit,
        "expected_lane_count": 14,
        "actual_lane_count": len(rows),
        "lane_names": [row["lane"] for row in rows],
        "lanes": rows,
        "passed": len(rows) == 14 and all(row["reproduced"] for row in rows),
    }
    _write(output / "reproduction.json", result)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
