#!/usr/bin/env python3
"""Build a fail-closed manifest for the complete tracked public tree."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import subprocess


SENSITIVE_SUFFIXES = {
    ".onnx",
    ".pt",
    ".pth",
    ".ckpt",
    ".safetensors",
    ".pem",
    ".key",
    ".p12",
    ".pfx",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def tracked_relative_paths(root: Path) -> list[str]:
    completed = subprocess.run(
        ["git", "ls-files", "-z"], cwd=root, check=True, capture_output=True
    )
    relative_paths = [
        value.decode("utf-8", errors="surrogateescape")
        for value in completed.stdout.split(b"\0")
        if value
    ]
    return relative_paths


def index_modes(root: Path) -> dict[str, str]:
    completed = subprocess.run(
        ["git", "ls-files", "-s", "-z"], cwd=root, check=True, capture_output=True
    )
    modes: dict[str, str] = {}
    for record in completed.stdout.split(b"\0"):
        if not record:
            continue
        metadata, raw_path = record.split(b"\t", 1)
        mode = metadata.split(b" ", 1)[0].decode("ascii")
        relative = raw_path.decode("utf-8", errors="surrogateescape")
        modes[relative] = mode
    return modes


def index_bytes(root: Path, relative: str) -> bytes:
    completed = subprocess.run(
        ["git", "show", f":{relative}"], cwd=root, check=True, capture_output=True
    )
    return completed.stdout


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source",
        choices=("working-tree", "index"),
        default="working-tree",
        help="hash current files or the exact staged/index blobs",
    )
    args = parser.parse_args()
    root = args.repository.resolve()
    output = args.output.resolve()

    records: list[tuple[str, bytes]] = []
    modes = index_modes(root)
    for relative in tracked_relative_paths(root):
        path = root / relative
        if path.resolve() == output:
            continue
        # Manifests are generated views over public content. Excluding every
        # manifest prevents circular old/new manifest dependencies while both
        # still cover all implementation, tests, reports, and aggregate data.
        if Path(relative).name == "manifest.sha256":
            continue
        if modes.get(relative) == "120000":
            raise ValueError(f"symlink is not allowed in public manifest: {relative}")
        if Path(relative).suffix.lower() in SENSITIVE_SUFFIXES:
            raise ValueError(f"sensitive binary in tracked public tree: {relative}")
        if args.source == "index":
            data = index_bytes(root, relative)
        else:
            if path.is_symlink():
                raise ValueError(f"symlink is not allowed in public manifest: {relative}")
            if not path.is_file():
                raise FileNotFoundError(path)
            data = path.read_bytes()
        records.append((relative, data))

    lines = [
        f"{sha256_bytes(data)}  {relative}"
        for relative, data in sorted(records)
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"manifested {len(lines)} tracked public files from {args.source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
