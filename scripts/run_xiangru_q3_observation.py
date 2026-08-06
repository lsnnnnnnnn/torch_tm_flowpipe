#!/usr/bin/env python3
"""Run the private Xiangru observation-only TORA-Q3 exporter.

Only code, hashes, and command metadata belong in the public repository.  The
caller must point ``--output-dir`` at the private verification-evidence root;
the script records the complete dirty instrumentation patch alongside the raw
per-leaf trace so its observational nature can be audited.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time
from typing import Sequence


EXPECTED_BASE_COMMIT = "27d29050a5f214b56f211ca9cb411e734ed80230"
EXPECTED_CHANGED_FILES = {
    "experiments/remainder_ablation/autolirpa_controller_worker.py",
    "experiments/remainder_ablation/run_c2_autolirpa_feasibility.py",
    "experiments/remainder_ablation/run_s0_tora_static_partition_sweep.py",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_capture(argv: Sequence[str], cwd: Path) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        list(argv),
        cwd=cwd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def require_observer_patch(root: Path) -> bytes:
    head = run_capture(["git", "rev-parse", "HEAD"], root)
    if head.returncode or head.stdout.decode().strip() != EXPECTED_BASE_COMMIT:
        raise ValueError("observer worktree is not at the frozen Xiangru commit")
    changed = run_capture(["git", "diff", "--name-only"], root)
    changed_files = set(changed.stdout.decode().splitlines())
    if changed.returncode or changed_files != EXPECTED_CHANGED_FILES:
        raise ValueError(
            f"unexpected observer modifications: {sorted(changed_files)}"
        )
    diff_check = run_capture(["git", "diff", "--check"], root)
    if diff_check.returncode:
        raise ValueError(diff_check.stdout.decode() + diff_check.stderr.decode())
    patch = run_capture(["git", "diff", "--binary"], root)
    if patch.returncode:
        raise ValueError("could not capture observer patch")
    return patch.stdout


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--xiangru-root", type=Path, required=True)
    parser.add_argument("--diffreach-root", type=Path, required=True)
    parser.add_argument("--config-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--diffreach-python", type=Path, required=True)
    parser.add_argument("--cap", type=int, default=200)
    parser.add_argument("--timeout-seconds", type=float, default=7200.0)
    parser.add_argument(
        "--run-id", default="xiangru_complete_q3_b48_t20_observation"
    )
    args = parser.parse_args()

    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    raw = output / "raw"
    raw.mkdir()
    root = args.xiangru_root.resolve()
    diffreach = args.diffreach_root.resolve()
    config_root = args.config_root.resolve()
    patch = require_observer_patch(root)
    (output / "instrumented_private_worktree.patch").write_bytes(patch)

    inner_root = "/" + "home/xiangru4/CROWN-Reach_Development"
    inner_diffreach = "/" + "home/xiangru4/DiffReach"
    command = [
        "bwrap",
        "--ro-bind", "/", "/",
        "--tmpfs", "/home",
        "--tmpfs", "/tmp",
        "--dir", "/home/xiangru4",
        "--ro-bind", str(root), inner_root,
        "--ro-bind", str(diffreach), inner_diffreach,
        "--ro-bind", str(config_root), "/tmp/inputs",
        "--dir", "/tmp/observation",
        "--bind", str(raw), "/tmp/observation",
        "--proc", "/proc",
        "--dev-bind", "/dev", "/dev",
        "--chdir", inner_root,
        str(args.python.absolute()),
        f"{inner_root}/experiments/remainder_ablation/run_s0_tora_static_partition_sweep.py",
        "--device", "cuda",
        "--policies", "b48_static",
        "--methods", "complete_q3",
        "--cap", str(args.cap),
        "--q3-engine", "dynamic",
        "--controller-backend", "autolirpa",
        "--controller-platform", "cuda",
        "--controller-mode", "eager",
        "--controller-composition", "outward",
        "--config", "/tmp/inputs/diffreach_config.json",
        "--diffreach-root", inner_diffreach,
        "--diffreach-python", str(args.diffreach_python.absolute()),
        "--output-json", "/tmp/observation/result.json",
        "--output-markdown", "/tmp/observation/result.md",
        "--controller-trace-json", "/tmp/observation/controller.json",
        "--plant-trace-jsonl", "/tmp/observation/plant.jsonl",
        "--observation-run-id", args.run_id,
    ]
    timed = [
        "/usr/bin/time", "-v", "-o", str(output / "resource_usage.txt"),
        *command,
    ]
    (output / "command.json").write_text(
        json.dumps(
            {
                "argv": timed,
                "command": shlex.join(timed),
                "cwd": str(root),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = "0"
    started_utc = dt.datetime.now(dt.timezone.utc).isoformat()
    started = time.monotonic()
    timeout = False
    try:
        completed = subprocess.run(
            timed,
            cwd=root,
            env=environment,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=args.timeout_seconds,
        )
        exit_code: int | None = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
    except subprocess.TimeoutExpired as error:
        timeout = True
        exit_code = None
        stdout = error.stdout or b""
        stderr = error.stderr or b""
    ended_utc = dt.datetime.now(dt.timezone.utc).isoformat()
    wall_seconds = time.monotonic() - started
    (output / "stdout.log").write_bytes(stdout)
    (output / "stderr.log").write_bytes(stderr)
    artifacts = {}
    for path in sorted(raw.iterdir()):
        if path.is_file():
            artifacts[path.name] = {
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
    metadata = {
        "schema": "xiangru_tora_q3_observation_run_v1",
        "base_commit": EXPECTED_BASE_COMMIT,
        "observer_patch_sha256": sha256(
            output / "instrumented_private_worktree.patch"
        ),
        "start_time_utc": started_utc,
        "end_time_utc": ended_utc,
        "wall_seconds": wall_seconds,
        "timeout_seconds": args.timeout_seconds,
        "timeout_expired": timeout,
        "exit_code": exit_code,
        "stdout_sha256": sha256(output / "stdout.log"),
        "stderr_sha256": sha256(output / "stderr.log"),
        "artifacts": artifacts,
    }
    (output / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 124 if timeout else int(exit_code or 0)


if __name__ == "__main__":
    raise SystemExit(main())
