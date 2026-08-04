#!/usr/bin/env python3
"""Collect command-level provenance without interpreting experiment outputs."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Sequence


REPOS = {
    "primary": Path("/srv/local/shengenli/torch_tm_flowpipe_native_reproduction_no_adapters"),
    "primary_user_worktree": Path("/srv/local/shengenli/torch_tm_flowpipe"),
    "flowstar": Path("/srv/local/shengenli/flowstar"),
    "diffreach": Path("/srv/local/shengenli/DiffReach"),
    "xiangru": Path("/srv/local/shengenli/CROWN-Reach_Development"),
}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def slug(text: str) -> str:
    cleaned = "".join(char if char.isalnum() else "_" for char in text).strip("_")
    return cleaned[:96] or "command"


class Recorder:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self.records: list[dict[str, object]] = []

    def run(self, name: str, argv: Sequence[str], cwd: Path | None = None) -> None:
        ordinal = len(self.records) + 1
        stem = f"{ordinal:03d}_{slug(name)}"
        stdout_path = self.output_dir / f"{stem}.stdout.log"
        stderr_path = self.output_dir / f"{stem}.stderr.log"
        started = utc_now()
        try:
            completed = subprocess.run(
                list(argv),
                cwd=cwd,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=os.environ.copy(),
            )
            returncode = completed.returncode
            stdout = completed.stdout
            stderr = completed.stderr
        except FileNotFoundError as error:
            returncode = 127
            stdout = b""
            stderr = f"{error}\n".encode()
        ended = utc_now()
        stdout_path.write_bytes(stdout)
        stderr_path.write_bytes(stderr)
        self.records.append(
            {
                "name": name,
                "cwd": str(cwd.resolve()) if cwd else str(Path.cwd().resolve()),
                "argv": list(argv),
                "start_time_utc": started,
                "end_time_utc": ended,
                "exit_code": returncode,
                "stdout": str(stdout_path.relative_to(self.output_dir.parent)),
                "stderr": str(stderr_path.relative_to(self.output_dir.parent)),
                "stdout_sha256": sha256(stdout_path),
                "stderr_sha256": sha256(stderr_path),
            }
        )

    def finish(self) -> None:
        destination = self.output_dir / "command_records.json"
        destination.write_text(json.dumps(self.records, indent=2) + "\n", encoding="utf-8")


def git_commands(recorder: Recorder, label: str, repo: Path, *, fetch: bool = False) -> None:
    recorder.run(f"{label}_realpath", ["realpath", str(repo)])
    recorder.run(f"{label}_status", ["git", "status", "--short", "--branch"], repo)
    if fetch:
        recorder.run(f"{label}_fetch", ["git", "fetch", "--all", "--prune"], repo)
        recorder.run(f"{label}_status_after_fetch", ["git", "status", "--short", "--branch"], repo)
    recorder.run(f"{label}_head", ["git", "rev-parse", "HEAD"], repo)
    recorder.run(f"{label}_branch", ["git", "branch", "--show-current"], repo)
    recorder.run(f"{label}_remotes", ["git", "remote", "-v"], repo)
    recorder.run(f"{label}_recent_log", ["git", "log", "-12", "--oneline", "--decorate", "--all"], repo)
    recorder.run(f"{label}_branches", ["git", "branch", "-a", "-vv"], repo)
    recorder.run(f"{label}_worktrees", ["git", "worktree", "list"], repo)
    recorder.run(f"{label}_submodules", ["git", "submodule", "status", "--recursive"], repo)
    recorder.run(f"{label}_tracked_diff", ["git", "diff", "--no-ext-diff", "--binary"], repo)
    recorder.run(f"{label}_cached_diff", ["git", "diff", "--cached", "--no-ext-diff", "--binary"], repo)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    recorder = Recorder(output_dir)

    for label, repo in REPOS.items():
        git_commands(recorder, label, repo, fetch=(label == "primary"))

    system_commands: list[tuple[str, list[str]]] = [
        ("hostname", ["hostname"]),
        ("uname", ["uname", "-a"]),
        ("cpu", ["lscpu"]),
        ("memory", ["free", "-h"]),
        ("conda_env_list", ["conda", "env", "list"]),
        ("python_version", ["python", "--version"]),
        ("gcc_version", ["gcc", "--version"]),
        ("gxx_version", ["g++", "--version"]),
        ("nvidia_smi", ["nvidia-smi"]),
        ("nvcc_version", ["nvcc", "--version"]),
    ]
    for name, argv in system_commands:
        recorder.run(name, argv)

    selected_environment = {
        key: os.environ.get(key)
        for key in (
            "CONDA_DEFAULT_ENV",
            "CONDA_PREFIX",
            "CUDA_HOME",
            "CUDA_VISIBLE_DEVICES",
            "CUDNN_PATH",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "PYTHONPATH",
            "CROWN_REACH_GPU",
            "CROWN_REACH_PYTHON",
            "CROWN_REACH_PYTHONPATH",
            "DIFFREACH_ROOT",
            "DIFFREACH_PYTHON",
            "ARCH_COMP_ROOT",
        )
    }
    (output_dir / "selected_environment.json").write_text(
        json.dumps(selected_environment, indent=2) + "\n", encoding="utf-8"
    )
    recorder.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
