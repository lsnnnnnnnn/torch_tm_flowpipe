#!/usr/bin/env python3
"""Build a deterministic SHA256 inventory for one audit evidence tree."""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
from typing import Sequence


def build(root: Path, output_name: str) -> int:
    root = root.resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    output = root / output_name
    rows: list[str] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if not path.is_file() or path == output:
            continue
        relative = path.relative_to(root).as_posix()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append(f"{digest}  {relative}\n")
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    temporary.write_text("".join(rows), encoding="utf-8")
    temporary.replace(output)
    return len(rows)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output-name", default="manifest.sha256")
    return parser.parse_args(argv)


if __name__ == "__main__":
    arguments = parse_args()
    print(f"hashed_files={build(arguments.root, arguments.output_name)}")
