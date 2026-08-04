#!/usr/bin/env python3
"""Compare explicitly named CSV fields by an explicit row key."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path, key: str) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or key not in reader.fieldnames:
            raise KeyError(f"explicit key {key!r} missing from {path}")
        rows: dict[str, dict[str, str]] = {}
        for row in reader:
            value = row[key]
            if value in rows:
                raise ValueError(f"duplicate key {value!r} in {path}")
            rows[value] = row
        return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--fresh", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--key", required=True)
    parser.add_argument("--field", action="append", required=True)
    parser.add_argument("--excluded-field", action="append", default=[])
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    reference = load(args.reference, args.key)
    fresh = load(args.fresh, args.key)
    rows = []
    passed = set(reference) == set(fresh)
    for key_value in sorted(set(reference) | set(fresh)):
        reference_row = reference.get(key_value)
        fresh_row = fresh.get(key_value)
        fields = []
        for field in args.field:
            reference_present = reference_row is not None and field in reference_row
            fresh_present = fresh_row is not None and field in fresh_row
            equal = bool(
                reference_present
                and fresh_present
                and reference_row[field] == fresh_row[field]
            )
            passed &= equal
            fields.append(
                {
                    "field": field,
                    "reference_present": reference_present,
                    "fresh_present": fresh_present,
                    "reference": reference_row.get(field) if reference_row else None,
                    "fresh": fresh_row.get(field) if fresh_row else None,
                    "equal": equal,
                }
            )
        rows.append({"key": key_value, "fields": fields})
    payload = {
        "schema_version": 1,
        "status": "PASS_EXACT" if passed else "FAIL",
        "reference": str(args.reference.resolve()),
        "reference_sha256": sha256(args.reference),
        "fresh": str(args.fresh.resolve()),
        "fresh_sha256": sha256(args.fresh),
        "key": args.key,
        "fields": args.field,
        "excluded_fields": args.excluded_field,
        "rows": rows,
        "missing_field_fallback": False,
        "tolerance": None,
        "tolerance_source": None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
