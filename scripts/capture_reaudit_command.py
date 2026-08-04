#!/usr/bin/env python3
"""Run one audit command and retain its exact provenance and output streams."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[1]


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--stdout", type=Path, required=True)
    parser.add_argument("--stderr", type=Path, required=True)
    parser.add_argument("--cwd", type=Path, default=REPO_ROOT)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    arguments = parser.parse_args()
    command = list(arguments.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("a command is required after --")

    cwd = arguments.cwd.resolve()
    started = _utc_now()
    begin = time.perf_counter()
    process = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)
    elapsed = time.perf_counter() - begin
    finished = _utc_now()

    for path in (arguments.record, arguments.stdout, arguments.stderr):
        path.parent.mkdir(parents=True, exist_ok=True)
    arguments.stdout.write_text(process.stdout, encoding="utf-8")
    arguments.stderr.write_text(process.stderr, encoding="utf-8")
    record = {
        "schema_version": "reaudit-command-1.0.0",
        "command": command,
        "cwd": str(cwd),
        "started_utc": started,
        "finished_utc": finished,
        "elapsed_s": elapsed,
        "exit_code": process.returncode,
        "stdout_path": _relative(arguments.stdout),
        "stderr_path": _relative(arguments.stderr),
    }
    arguments.record.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if process.stdout:
        sys.stdout.write(process.stdout)
    if process.stderr:
        sys.stderr.write(process.stderr)
    return process.returncode


if __name__ == "__main__":
    raise SystemExit(main())
