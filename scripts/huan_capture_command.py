#!/usr/bin/env python3
"""Run one auditable command and capture combined output plus provenance."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


CAPTURED_ENV = (
    "CUDA_VISIBLE_DEVICES",
    "CUDA_HOME",
    "CC",
    "CXX",
    "CUDAHOSTCXX",
    "HYPOTHESIS_PROFILE",
    "HUAN_CONFIG_ROOT",
    "PYTHONPATH",
    "FLOWSTAR_NO_CUDA_KERNEL",
    "PYTHONHASHSEED",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_state(root: Path) -> dict[str, object]:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=root, check=True, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        ).stdout.strip()

    return {
        "root": str(root.resolve()),
        "head": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "dirty": bool(git("status", "--porcelain")),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cwd", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument(
        "--source-root", type=Path, action="append", default=[],
        help="Git source root to record; repeat for multiple repositories",
    )
    parser.add_argument(
        "--artifact", type=Path, action="append", default=[],
        help="Command-produced file whose SHA256 must be captured; repeatable",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a command is required after --")

    started = datetime.now(timezone.utc)
    before = time.monotonic()
    result = subprocess.run(
        command,
        cwd=args.cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    elapsed = time.monotonic() - before
    artifacts = []
    for raw_path in args.artifact:
        path = raw_path if raw_path.is_absolute() else args.cwd / raw_path
        artifacts.append(
            {
                "path": str(path.resolve()),
                "exists": path.is_file(),
                "sha256": _sha256(path) if path.is_file() else None,
                "size": path.stat().st_size if path.is_file() else None,
            }
        )
    header = {
        "schema": "torch_tm_flowpipe.huan_command_capture/2",
        "label": args.label,
        "started_utc": started.isoformat(),
        "cwd": str(args.cwd.resolve()),
        "command": command,
        "environment": {key: os.environ[key] for key in CAPTURED_ENV if key in os.environ},
        "returncode": result.returncode,
        "elapsed_seconds": elapsed,
        "sources": [_source_state(root) for root in args.source_root],
        "artifacts": artifacts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(header, indent=2, sort_keys=True)
        + "\n--- combined stdout/stderr ---\n"
        + result.stdout,
        encoding="utf-8",
    )
    print(json.dumps({"label": args.label, "returncode": result.returncode, "elapsed_seconds": elapsed}, sort_keys=True))
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
