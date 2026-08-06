#!/usr/bin/env python3
"""Capture tracked and selected untracked private-observer changes as a patch."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess


def command(root: Path, arguments: list[str]) -> bytes:
    completed = subprocess.run(
        arguments, cwd=root, check=True, capture_output=True
    )
    return completed.stdout


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--extra", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.repository.resolve()
    chunks = [command(root, ["git", "diff", "--binary", "--no-ext-diff"])]
    for extra in args.extra:
        path = extra.resolve()
        try:
            relative = path.relative_to(root)
        except ValueError as exception:
            raise ValueError(f"extra file is outside repository: {path}") from exception
        diff = subprocess.run(
            ["git", "diff", "--binary", "--no-index", "--", "/dev/null", str(relative)],
            cwd=root,
            capture_output=True,
        )
        if diff.returncode not in {0, 1}:
            raise RuntimeError(diff.stderr.decode(errors="replace"))
        chunks.append(diff.stdout)
    payload = b"\n".join(chunks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload)
    print(f"wrote {len(payload)} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
