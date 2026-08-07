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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tracked_paths(root: Path) -> list[Path]:
    completed = subprocess.run(
        ["git", "ls-files", "-z"], cwd=root, check=True, capture_output=True
    )
    relative_paths = [
        value.decode("utf-8", errors="surrogateescape")
        for value in completed.stdout.split(b"\0")
        if value
    ]
    return [root / relative for relative in relative_paths]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.repository.resolve()
    output = args.output.resolve()

    files: list[Path] = []
    for path in tracked_paths(root):
        if path.resolve() == output:
            continue
        # Manifests are generated views over public content. Excluding every
        # manifest prevents circular old/new manifest dependencies while both
        # still cover all implementation, tests, reports, and aggregate data.
        if path.name == "manifest.sha256":
            continue
        if path.is_symlink():
            raise ValueError(f"symlink is not allowed in public manifest: {path}")
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.suffix.lower() in SENSITIVE_SUFFIXES:
            raise ValueError(f"sensitive binary in tracked public tree: {path}")
        files.append(path)

    lines = [
        f"{sha256(path)}  {path.relative_to(root).as_posix()}"
        for path in sorted(files)
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"manifested {len(lines)} tracked public files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
