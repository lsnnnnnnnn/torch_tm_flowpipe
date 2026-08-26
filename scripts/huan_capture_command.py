#!/usr/bin/env python3
"""Run one auditable command and capture combined output plus provenance."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
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
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cwd", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
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
    header = {
        "schema": "torch_tm_flowpipe.huan_command_capture/1",
        "label": args.label,
        "started_utc": started.isoformat(),
        "cwd": str(args.cwd.resolve()),
        "command": command,
        "environment": {key: os.environ[key] for key in CAPTURED_ENV if key in os.environ},
        "returncode": result.returncode,
        "elapsed_seconds": elapsed,
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
