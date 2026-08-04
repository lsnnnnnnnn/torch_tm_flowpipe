#!/usr/bin/env python3
"""Compare explicitly named fields from two artifacts without fallback semantics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


MISSING = object()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def field_at(value: Any, path: str) -> Any:
    current = value
    for component in path.split("."):
        if not isinstance(current, dict) or component not in current:
            return MISSING
        current = current[component]
    return current


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--fresh", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--field", action="append", required=True)
    args = parser.parse_args()

    reference_path = args.reference.resolve()
    fresh_path = args.fresh.resolve()
    output_path = args.output.resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    fresh = json.loads(fresh_path.read_text(encoding="utf-8"))

    rows = []
    passed = True
    for field in args.field:
        reference_value = field_at(reference, field)
        fresh_value = field_at(fresh, field)
        reference_present = reference_value is not MISSING
        fresh_present = fresh_value is not MISSING
        equal = bool(
            reference_present
            and fresh_present
            and reference_value == fresh_value
        )
        passed &= equal
        rows.append(
            {
                "field": field,
                "reference_present": reference_present,
                "fresh_present": fresh_present,
                "reference": None if reference_value is MISSING else reference_value,
                "fresh": None if fresh_value is MISSING else fresh_value,
                "comparison": "exact",
                "equal": equal,
            }
        )

    result = {
        "schema_version": 1,
        "status": "PASS_EXACT" if passed else "FAIL",
        "reference": str(reference_path),
        "reference_sha256": sha256(reference_path),
        "fresh": str(fresh_path),
        "fresh_sha256": sha256(fresh_path),
        "fields": rows,
        "missing_field_fallback": False,
        "tolerance": None,
        "tolerance_source": None,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
