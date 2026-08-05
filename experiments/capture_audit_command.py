#!/usr/bin/env python3
"""Run one audit command and preserve its raw streams and execution metadata."""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


def _capture_git(cwd: Path) -> dict[str, object] | None:
    probe = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], cwd=cwd, capture_output=True, text=True
    )
    if probe.returncode != 0:
        return None

    def git(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", *arguments], cwd=cwd, capture_output=True, text=True, check=True
        )
        return completed.stdout.rstrip("\n")

    diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD"], cwd=cwd, capture_output=True, check=True
    ).stdout
    import hashlib

    return {
        "root": probe.stdout.strip(),
        "head": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "status_porcelain_v1": git("status", "--porcelain=v1", "--untracked-files=all"),
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
    }


def _capture_runtime_environment() -> dict[str, object]:
    result: dict[str, object] = {
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
    }
    try:
        import torch

        result.update(
            {
                "pytorch_version": torch.__version__,
                "torch_cuda_version": torch.version.cuda,
                "cuda_available": torch.cuda.is_available(),
                "cuda_device_count": torch.cuda.device_count(),
            }
        )
    except ImportError:
        result["pytorch"] = "not importable by capture interpreter"
    return result


def _write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise ValueError("a command is required after --")
    cwd = args.cwd.resolve()
    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=cwd, capture_output=True)
    runtime_s = time.perf_counter() - started
    finished_at = datetime.now(timezone.utc)
    _write(output_dir / "stdout.log", completed.stdout.decode("utf-8", errors="replace"))
    _write(output_dir / "stderr.log", completed.stderr.decode("utf-8", errors="replace"))
    metadata = {
        "schema": "torch_tm_flowpipe_captured_command_v1",
        "argv": command,
        "cwd": str(cwd),
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": finished_at.isoformat(),
        "runtime_s": runtime_s,
        "exit_code": completed.returncode,
        "python": sys.version,
        "platform": platform.platform(),
        "runtime_environment": _capture_runtime_environment(),
        "git": _capture_git(cwd),
        "selected_environment": {
            key: os.environ[key]
            for key in ("CONDA_DEFAULT_ENV", "CONDA_PREFIX", "FLOWSTAR_ROOT")
            if key in os.environ
        },
    }
    _write(output_dir / "command.json", json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    _write(output_dir / "exit_code.txt", f"{completed.returncode}\n")
    return completed.returncode


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cwd", type=Path, default=Path.cwd())
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
