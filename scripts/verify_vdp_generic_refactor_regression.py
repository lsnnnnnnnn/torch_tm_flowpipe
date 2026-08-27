#!/usr/bin/env python3
"""Verify the frozen VDP C2/C3 regression matrix from package-local raw data."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACKAGE = ROOT / "artifacts/runs/vdp_generic_refactor_vdp_zero_regression_20260827"


class RegressionEvidenceError(ValueError):
    """Raised when evidence is incomplete or structurally inconsistent."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RegressionEvidenceError(f"cannot read JSON {path}: {exc}") from exc


def read_csv_gz(path: Path) -> list[dict[str, str]]:
    try:
        with gzip.open(path, "rt", newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    except (OSError, csv.Error) as exc:
        raise RegressionEvidenceError(f"cannot read gzip CSV {path}: {exc}") from exc


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


def _project_summary(summary: Mapping[str, Any], ignored: Sequence[str]) -> dict[str, Any]:
    return {key: value for key, value in summary.items() if key not in set(ignored)}


def _project_rows(rows: Sequence[Mapping[str, str]], ignored: Sequence[str]) -> list[dict[str, str]]:
    ignored_set = set(ignored)
    return [{key: value for key, value in row.items() if key not in ignored_set} for row in rows]


def _numeric_string(value: str) -> float | None:
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def _compare_with_tolerance(
    reference: Any,
    candidate: Any,
    tolerance: float,
    *,
    path: str,
) -> tuple[float, list[str]]:
    """Compare nested JSON/CSV values, accepting finite numeric deltas only."""

    if type(reference) is not type(candidate):
        return 0.0, [f"{path}: type mismatch"]
    if isinstance(reference, Mapping):
        errors: list[str] = []
        if set(reference) != set(candidate):
            errors.append(f"{path}: keys mismatch")
        maximum = 0.0
        for key in sorted(set(reference) & set(candidate)):
            delta, nested = _compare_with_tolerance(
                reference[key], candidate[key], tolerance, path=f"{path}.{key}"
            )
            maximum = max(maximum, delta)
            errors.extend(nested)
        return maximum, errors
    if isinstance(reference, list):
        errors = []
        if len(reference) != len(candidate):
            errors.append(f"{path}: length mismatch {len(reference)} != {len(candidate)}")
        maximum = 0.0
        for index, (left, right) in enumerate(zip(reference, candidate)):
            delta, nested = _compare_with_tolerance(
                left, right, tolerance, path=f"{path}[{index}]"
            )
            maximum = max(maximum, delta)
            errors.extend(nested)
        return maximum, errors
    if isinstance(reference, bool) or reference is None:
        return (0.0, []) if reference == candidate else (0.0, [f"{path}: value mismatch"])
    if isinstance(reference, (int, float)):
        delta = abs(float(reference) - float(candidate))
        if not math.isfinite(delta) or delta > tolerance:
            return delta, [f"{path}: numeric delta {delta!r} exceeds {tolerance!r}"]
        return delta, []
    if isinstance(reference, str):
        if reference == candidate:
            return 0.0, []
        left_number = _numeric_string(reference)
        right_number = _numeric_string(candidate)
        if left_number is not None and right_number is not None:
            delta = abs(left_number - right_number)
            if delta <= tolerance:
                return delta, []
            return delta, [f"{path}: numeric-string delta {delta!r} exceeds {tolerance!r}"]
        try:
            left_json = json.loads(reference)
            right_json = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            return 0.0, [f"{path}: string mismatch"]
        return _compare_with_tolerance(left_json, right_json, tolerance, path=path)
    return (0.0, []) if reference == candidate else (0.0, [f"{path}: value mismatch"])


def _lane_dir(package: Path, side: str, schedule: str, lane: str, label: str) -> Path:
    return package / "raw" / side / schedule / lane / label


def recompute(package: Path) -> dict[str, Any]:
    contract = read_json(package / "EVIDENCE_CONTRACT.json")
    comparison = contract["comparison"]
    tolerance = float(comparison["c3_numeric_tolerance"])
    segment_ignored = comparison["segment_ignored_fields"]
    summary_ignored = comparison["summary_ignored_fields"]
    schedules = [
        ("fixed", label) for label in contract["fixed_horizons"]
    ] + [("native", "T10")]
    matrices: dict[str, Any] = {}
    c2_hash_gate = True
    c3_tolerance_gate = True
    source_gate = True
    outcome_gate = True
    maximum_c3_delta = 0.0
    c3_errors: list[str] = []

    for schedule, label in schedules:
        key = f"{schedule}/{label}"
        matrices[key] = {}
        for lane in ("torch_c2", "torch_c3"):
            reference_dir = _lane_dir(package, "reference", schedule, lane, label)
            candidate_dir = _lane_dir(package, "candidate", schedule, lane, label)
            reference_summary = read_json(reference_dir / "summary.json")
            candidate_summary = read_json(candidate_dir / "summary.json")
            reference_rows = _project_rows(
                read_csv_gz(reference_dir / "segments.csv.gz"), segment_ignored
            )
            candidate_rows = _project_rows(
                read_csv_gz(candidate_dir / "segments.csv.gz"), segment_ignored
            )
            reference_summary_projection = _project_summary(reference_summary, summary_ignored)
            candidate_summary_projection = _project_summary(candidate_summary, summary_ignored)
            lane_result: dict[str, Any] = {
                "reference_segment_rows": len(reference_rows),
                "candidate_segment_rows": len(candidate_rows),
                "reference_segments_sha256": canonical_sha256(reference_rows),
                "candidate_segments_sha256": canonical_sha256(candidate_rows),
                "reference_summary_sha256": canonical_sha256(reference_summary_projection),
                "candidate_summary_sha256": canonical_sha256(candidate_summary_projection),
            }
            if lane == "torch_c2":
                exact = (
                    reference_rows == candidate_rows
                    and reference_summary_projection == candidate_summary_projection
                )
                c2_hash_gate &= exact
                lane_result["exact_scientific_hash_match"] = exact
            else:
                row_delta, row_errors = _compare_with_tolerance(
                    reference_rows,
                    candidate_rows,
                    tolerance,
                    path=f"{key}/{lane}/segments",
                )
                summary_delta, summary_errors = _compare_with_tolerance(
                    reference_summary_projection,
                    candidate_summary_projection,
                    tolerance,
                    path=f"{key}/{lane}/summary",
                )
                maximum_c3_delta = max(maximum_c3_delta, row_delta, summary_delta)
                c3_errors.extend(row_errors + summary_errors)
                lane_result["maximum_numeric_delta"] = max(row_delta, summary_delta)
                lane_result["comparison_errors"] = len(row_errors) + len(summary_errors)

            if schedule == "native":
                reference_attempts = read_csv_gz(reference_dir / "attempts.csv.gz")
                candidate_attempts = read_csv_gz(candidate_dir / "attempts.csv.gz")
                lane_result.update(
                    {
                        "reference_attempt_rows": len(reference_attempts),
                        "candidate_attempt_rows": len(candidate_attempts),
                        "reference_attempts_sha256": canonical_sha256(reference_attempts),
                        "candidate_attempts_sha256": canonical_sha256(candidate_attempts),
                    }
                )
                if lane == "torch_c2":
                    exact_attempts = reference_attempts == candidate_attempts
                    c2_hash_gate &= exact_attempts
                    lane_result["exact_attempt_hash_match"] = exact_attempts
                else:
                    attempt_delta, attempt_errors = _compare_with_tolerance(
                        reference_attempts,
                        candidate_attempts,
                        tolerance,
                        path=f"{key}/{lane}/attempts",
                    )
                    maximum_c3_delta = max(maximum_c3_delta, attempt_delta)
                    c3_errors.extend(attempt_errors)
                    lane_result["maximum_attempt_numeric_delta"] = attempt_delta
                    lane_result["attempt_comparison_errors"] = len(attempt_errors)

            for side, directory in (("reference", reference_dir), ("candidate", candidate_dir)):
                command = read_json(directory / "command.json")
                expected_commit = contract["source_commits"][side][lane]
                source_gate &= (
                    command.get("commit") == expected_commit
                    and command.get("tracked_diff_sha256") == contract["empty_diff_sha256"]
                    and command.get("worktree_status") == ""
                )
            matrices[key][lane] = lane_result

    c3_tolerance_gate &= not c3_errors and maximum_c3_delta <= tolerance
    for lane, expectation in contract["native_expectations"].items():
        summary = read_json(
            _lane_dir(package, "candidate", "native", lane, "T10") / "summary.json"
        )
        outcome_gate &= all(summary.get(field) == value for field, value in expectation.items())
    for label, horizon in contract["fixed_horizons"].items():
        for lane in ("torch_c2", "torch_c3"):
            summary = read_json(
                _lane_dir(package, "candidate", "fixed", lane, label) / "summary.json"
            )
            outcome_gate &= (
                summary.get("status") == "completed"
                and summary.get("completed_requested_horizon") is True
                and abs(float(summary.get("completed_horizon")) - float(horizon)) <= tolerance
            )

    gates = {
        "c2_scientific_hash_unchanged": c2_hash_gate,
        "c3_fixed_and_native_within_tolerance": c3_tolerance_gate,
        "frozen_source_provenance": source_gate,
        "frozen_outcomes": outcome_gate,
    }
    return {
        "schema": "torch_tm_flowpipe.vdp_generic_refactor_regression_result/1",
        "gates": gates,
        "passed": all(gates.values()),
        "maximum_c3_numeric_delta": maximum_c3_delta,
        "c3_comparison_errors": c3_errors[:20],
        "matrix": matrices,
    }


def verify(package: Path) -> tuple[dict[str, Any] | None, list[str]]:
    errors = verify_checksums(package)
    try:
        result = recompute(package)
        recorded = read_json(package / "RESULT.json")
        if result != recorded:
            errors.append("RESULT.json does not match raw recomputation")
        if not result["passed"]:
            errors.append("one or more VDP regression gates failed")
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
