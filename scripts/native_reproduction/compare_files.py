#!/usr/bin/env python3
"""Compare explicitly paired raw artifacts byte-for-byte."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pair",
        nargs=2,
        action="append",
        metavar=("REFERENCE", "FRESH"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    rows = []
    for reference_text, fresh_text in args.pair:
        reference = Path(reference_text).resolve()
        fresh = Path(fresh_text).resolve()
        reference_hash = sha256(reference)
        fresh_hash = sha256(fresh)
        rows.append(
            {
                "reference": str(reference),
                "reference_sha256": reference_hash,
                "fresh": str(fresh),
                "fresh_sha256": fresh_hash,
                "comparison": "byte_exact",
                "equal": reference_hash == fresh_hash,
            }
        )
    passed = all(row["equal"] for row in rows)
    payload = {
        "schema_version": 1,
        "status": "PASS_EXACT" if passed else "FAIL",
        "pairs": rows,
        "fallback": False,
        "tolerance": None,
        "tolerance_source": None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
