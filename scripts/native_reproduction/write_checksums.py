#!/usr/bin/env python3
"""Write deterministic SHA256SUMS-style provenance for a run directory."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--relative-to",
        type=Path,
        default=Path.cwd(),
        help="base used in checksum path names (default: current directory)",
    )
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    output = args.output.resolve()
    relative_to = args.relative_to.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if not run_root.is_dir():
        raise NotADirectoryError(run_root)
    files = sorted(
        path
        for path in run_root.rglob("*")
        if path.is_file() and path != output
    )
    if not files:
        raise ValueError("run root contains no files")
    lines = []
    for path in files:
        try:
            name = path.relative_to(relative_to)
        except ValueError as exc:
            raise ValueError(f"{path} is outside --relative-to {relative_to}") from exc
        lines.append(f"{sha256(path)}  {name.as_posix()}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(lines)} checksums to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
