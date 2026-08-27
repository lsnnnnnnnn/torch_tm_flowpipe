#!/usr/bin/env python3
"""Verify the package-local Brusselator generic-core evidence from raw data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_brusselator_second_system import analyze, sha256  # noqa: E402


DEFAULT_PACKAGE = ROOT / "artifacts/runs/brusselator_generic_core_validation_20260827"
ALLOWED_STATUSES = {
    "C3_GENERICITY_SOUNDNESS_GATE_FAILED_STOP",
    "C3_GENERIC_POLYNOMIAL_PLANT_CORE_VALIDATED",
    "C3_GENERIC_CORE_VALIDATED__SECOND_SYSTEM_NO_MATERIAL_GAIN",
    "C3_GENERIC_CORE_VALIDATED__SECOND_SYSTEM_PRODUCTION_USEFUL",
}


class SecondSystemPackageError(ValueError):
    """Raised when the evidence package is incomplete or inconsistent."""


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SecondSystemPackageError(f"cannot read JSON {path}: {exc}") from exc


def verify_checksums(package: Path) -> list[str]:
    checksum_path = package / "SHA256SUMS"
    if not checksum_path.is_file():
        return ["SHA256SUMS missing"]
    expected: dict[str, str] = {}
    errors: list[str] = []
    for line in checksum_path.read_text(encoding="ascii").splitlines():
        digest, separator, relative = line.partition("  ")
        if (
            not separator
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or not relative
            or relative in expected
        ):
            errors.append(f"malformed or duplicate checksum line: {line}")
        else:
            expected[relative] = digest
    actual = {
        path.relative_to(package).as_posix()
        for path in package.rglob("*")
        if path.is_file() and path != checksum_path
    }
    for relative in sorted(actual | set(expected)):
        path = package / relative
        if relative not in actual:
            errors.append(f"checksum target missing: {relative}")
        elif relative not in expected:
            errors.append(f"uncovered file: {relative}")
        elif sha256(path) != expected[relative]:
            errors.append(f"checksum mismatch: {relative}")
    return errors


def _manifest_errors(package: Path) -> list[str]:
    manifest = read_json(package / "MANIFEST.json")
    rows = manifest.get("raw_files")
    if not isinstance(rows, list):
        return ["MANIFEST.json raw_files is not a list"]
    errors: list[str] = []
    seen: set[str] = set()
    actual = {
        path.relative_to(package).as_posix()
        for path in (package / "raw").rglob("*")
        if path.is_file()
    }
    for row in rows:
        if not isinstance(row, dict):
            errors.append("manifest row is not an object")
            continue
        relative = str(row.get("path", ""))
        if relative in seen:
            errors.append(f"duplicate manifest path: {relative}")
            continue
        seen.add(relative)
        path = package / relative
        if not relative.startswith("raw/") or not path.is_file():
            errors.append(f"manifest target missing or outside raw/: {relative}")
            continue
        if row.get("sha256") != sha256(path):
            errors.append(f"manifest hash mismatch: {relative}")
        if row.get("size") != path.stat().st_size:
            errors.append(f"manifest size mismatch: {relative}")
    if seen != actual:
        errors.append("manifest raw-file coverage mismatch")
    return errors


def recompute(package: Path) -> dict[str, Any]:
    return analyze(package / "raw", package / "exact_fraction_2d.xml")


def verify(package: Path) -> tuple[dict[str, Any] | None, list[str]]:
    errors = verify_checksums(package)
    try:
        errors.extend(_manifest_errors(package))
        packaged_contract = package / "SECOND_SYSTEM_CONTRACT.md"
        if sha256(packaged_contract) != sha256(ROOT / "SECOND_SYSTEM_CONTRACT.md"):
            errors.append("packaged pre-registration contract differs from repository contract")
        evidence_contract = read_json(package / "EVIDENCE_CONTRACT.json")
        if evidence_contract.get("preregistered_contract_sha256") != sha256(packaged_contract):
            errors.append("EVIDENCE_CONTRACT contract hash mismatch")
        result = recompute(package)
        recorded = read_json(package / "RESULT.json")
        if result != recorded:
            errors.append("RESULT.json does not match raw recomputation")
        if result.get("status") not in ALLOWED_STATUSES:
            errors.append(f"unrecognized terminal status: {result.get('status')!r}")
        canonical = json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False)
        if evidence_contract.get("result_sha256") != hashlib.sha256(canonical.encode()).hexdigest():
            errors.append("EVIDENCE_CONTRACT result hash mismatch")
    except (KeyError, TypeError, ValueError, OSError) as exc:
        return None, errors + [str(exc)]
    return result, errors


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=DEFAULT_PACKAGE)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result, errors = verify(args.package.resolve())
    print(json.dumps({"ok": not errors, "errors": errors, "result": result}, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
