#!/usr/bin/env python3
"""Run one unmodified native command and preserve its raw process evidence."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_env(values: list[str]) -> dict[str, str]:
    additions: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--env must be KEY=VALUE, got {value!r}")
        key, item = value.split("=", 1)
        if not key:
            raise ValueError("--env key must not be empty")
        additions[key] = item
    return additions


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--cwd", required=True, type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=None)
    parser.add_argument("--env", action="append", default=[])
    parser.add_argument("argv", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    argv = args.argv[1:] if args.argv[:1] == ["--"] else args.argv
    if not argv:
        parser.error("a command is required after --")

    output_dir = args.output_dir.resolve()
    cwd = args.cwd.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    stdout_path = output_dir / "stdout.log"
    stderr_path = output_dir / "stderr.log"
    metadata_path = output_dir / "command.json"
    selected_env = parse_env(args.env)
    process_env = os.environ.copy()
    process_env.update(selected_env)

    started_utc = utc_now()
    started_monotonic = time.monotonic()
    timeout_expired = False
    try:
        completed = subprocess.run(
            argv,
            cwd=cwd,
            env=process_env,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=args.timeout_seconds,
        )
        exit_code: int | None = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
    except subprocess.TimeoutExpired as error:
        timeout_expired = True
        exit_code = None
        stdout = error.stdout or b""
        stderr = error.stderr or b""
    except FileNotFoundError as error:
        exit_code = 127
        stdout = b""
        stderr = f"{error}\n".encode()
    ended_utc = utc_now()
    wall_seconds = time.monotonic() - started_monotonic

    stdout_path.write_bytes(stdout)
    stderr_path.write_bytes(stderr)
    metadata = {
        "label": args.label,
        "cwd": str(cwd),
        "argv": argv,
        "environment_overrides": selected_env,
        "start_time_utc": started_utc,
        "end_time_utc": ended_utc,
        "wall_seconds": wall_seconds,
        "timeout_seconds": args.timeout_seconds,
        "timeout_expired": timeout_expired,
        "exit_code": exit_code,
        "stdout": "stdout.log",
        "stderr": "stderr.log",
        "stdout_sha256": sha256(stdout_path),
        "stderr_sha256": sha256(stderr_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return 124 if timeout_expired else int(exit_code or 0)


if __name__ == "__main__":
    raise SystemExit(main())
