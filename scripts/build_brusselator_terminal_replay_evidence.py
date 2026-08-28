#!/usr/bin/env python3
"""Build the self-verifying supplemental SR100 terminal replay package."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.verify_brusselator_terminal_replay_evidence import (  # noqa: E402
    DEFAULT_PACKAGE,
    recompute,
    sha256,
)


DEFAULT_RAW = Path("/srv/local/shengenli/brusselator_sr100_terminal_replay_20260828")
RAW_FILES = (
    "command.json",
    "RESULT.json",
    "checkpoint_before/terminal_state.json",
    "checkpoint_before/terminal_state_manifest.json",
    "checkpoint_after/terminal_state.json",
    "checkpoint_after/terminal_state_manifest.json",
)


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def build(raw: Path, output: Path) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    for relative in RAW_FILES:
        source = raw / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = output / "raw" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        manifest.append(
            {
                "path": destination.relative_to(output).as_posix(),
                "source": str(source),
                "size": destination.stat().st_size,
                "sha256": sha256(destination),
            }
        )
    result = recompute(output)
    write_json(output / "CLOSURE_RESULT.json", result)
    write_json(
        output / "MANIFEST.json",
        {
            "schema": "torch_tm_flowpipe.brusselator_terminal_replay_manifest/1",
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "raw_files": manifest,
        },
    )
    (output / "README.md").write_text(
        "# Brusselator SR100 terminal replay closure\n\n"
        "This supplemental package preserves the one terminal attempt and both\n"
        "canonical v5 checkpoints. It recomputes the previously missing rollback\n"
        "gate without rewriting the original three-lane evidence.\n\n"
        "```bash\npython scripts/verify_brusselator_terminal_replay_evidence.py\n```\n\n"
        f"Recomputed status: `{result['status']}`.\n",
        encoding="utf-8",
    )
    files = sorted(
        path for path in output.rglob("*") if path.is_file() and path.name != "SHA256SUMS"
    )
    (output / "SHA256SUMS").write_text(
        "".join(f"{sha256(path)}  {path.relative_to(output).as_posix()}\n" for path in files),
        encoding="ascii",
    )
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--output", type=Path, default=DEFAULT_PACKAGE)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = build(args.raw.resolve(), args.output.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
