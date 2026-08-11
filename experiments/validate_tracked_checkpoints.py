#!/usr/bin/env python3
"""Load every tracked JSON checkpoint and record its content digest."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON token: {value}")


def validate(paths: Iterable[Path], *, root: Path) -> dict[str, Any]:
    rows = []
    for path in sorted(Path(value).resolve() for value in paths):
        relative = path.relative_to(root.resolve()).as_posix()
        value = json.loads(
            path.read_text(encoding="utf-8"), parse_constant=_reject_constant
        )
        rows.append(
            {
                "path": relative,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "top_level_type": type(value).__name__,
                "schema": value.get("schema") if isinstance(value, dict) else None,
            }
        )
    if not rows:
        raise RuntimeError("no tracked JSON checkpoints were discovered")
    return {
        "schema": "tracked_json_checkpoint_load_v1",
        "outcome": "ALL_TRACKED_JSON_CHECKPOINTS_LOADED",
        "checkpoint_count": len(rows),
        "checkpoints": rows,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    tracked = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout.split(b"\0")
    paths = [
        ROOT / value.decode("utf-8")
        for value in tracked
        if value
        and value.decode("utf-8").endswith(".json")
        and "checkpoint" in value.decode("utf-8").lower()
    ]
    report = validate(paths, root=ROOT)
    (output / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    print(json.dumps(run(parse_args(argv)), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
