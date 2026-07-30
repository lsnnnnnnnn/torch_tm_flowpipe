#!/usr/bin/env python3
"""Capture repository provenance and verify the frozen result artifact."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
WORK_PARENT = REPO_ROOT.parent
BASELINE_WORKTREE = Path(
    os.environ.get(
        "TORCH_BASELINE_ROOT",
        WORK_PARENT / "torch_tm_flowpipe_first_order_bench",
    )
).resolve()
BASELINE_RESULT = (
    BASELINE_WORKTREE
    / "experiments"
    / "first_order_three_way"
    / "results"
    / "20260723T173852Z"
)
REPOSITORIES = {
    "followup": REPO_ROOT,
    "frozen_baseline_worktree": BASELINE_WORKTREE,
    "diffreach": Path(
        os.environ.get("DIFFREACH_ROOT", WORK_PARENT / "DiffReach")
    ).resolve(),
    "flowstar": Path(
        os.environ.get("FLOWSTAR_ROOT", WORK_PARENT / "flowstar")
    ).resolve(),
}


def _run(command: list[str], cwd: Path | None = None) -> dict[str, Any]:
    result = subprocess.run(
        command, cwd=cwd, text=True, capture_output=True, check=False
    )
    return {
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def _manifest(root: Path) -> dict[str, str]:
    manifest: dict[str, str] = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        manifest[str(path.relative_to(root))] = digest.hexdigest()
    return manifest


def _repo(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "head": _run(["git", "rev-parse", "HEAD"], path),
        "branch": _run(["git", "branch", "--show-current"], path),
        "status": _run(["git", "status", "--short", "--branch"], path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stage", choices=("before", "after"), required=True)
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = _manifest(BASELINE_RESULT)
    manifest_path = output / f"baseline_manifest_{args.stage}.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.stage == "before":
        print(f"Captured {len(manifest)} frozen-baseline files", flush=True)
        return
    before_path = output / "baseline_manifest_before.json"
    before = json.loads(before_path.read_text(encoding="utf-8"))
    unchanged = before == manifest
    environment = {
        "timestamp_utc": subprocess.run(
            ["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"],
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip(),
        "platform": platform.platform(),
        "python": sys.version,
        "executable": sys.executable,
        "cpu_count": os.cpu_count(),
        "repositories": {
            name: _repo(path) for name, path in REPOSITORIES.items()
        },
        "frozen_baseline": {
            "path": str(BASELINE_RESULT),
            "file_count": len(manifest),
            "byte_for_byte_unchanged_during_run": unchanged,
            "before_manifest": str(before_path),
            "after_manifest": str(manifest_path),
        },
        "commands": {
            "uname": _run(["uname", "-a"]),
            "lscpu": _run(["lscpu"]),
            "conda_envs": _run(["conda", "env", "list"]),
            "py11_versions": _run(
                [
                    "conda", "run", "-n", "py11", "python", "-c",
                    "import numpy,scipy,torch; "
                    "print('numpy='+numpy.__version__); "
                    "print('scipy='+scipy.__version__); "
                    "print('torch='+torch.__version__)",
                ]
            ),
            "diffreach312_versions": _run(
                [
                    "conda", "run", "-n", "diffreach312", "python", "-c",
                    "import jax,numpy,scipy; "
                    "print('jax='+jax.__version__); "
                    "print('numpy='+numpy.__version__); "
                    "print('scipy='+scipy.__version__)",
                ]
            ),
            "g++": _run(["g++", "--version"]),
        },
    }
    (output / "environment.json").write_text(
        json.dumps(environment, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not unchanged:
        raise SystemExit("frozen baseline result changed during follow-up run")
    print("Frozen baseline byte-for-byte gate passed", flush=True)


if __name__ == "__main__":
    main()
