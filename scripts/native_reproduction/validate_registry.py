#!/usr/bin/env python3
"""Validate the native-reproduction registry and its on-disk evidence.

The validator is deliberately fail-closed.  It does not infer missing fields,
repair paths, or translate one tool's result schema into another tool's schema.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


REPRODUCTION_STATUSES = {
    "reproduced_exact",
    "reproduced_with_declared_tolerance",
    "reference_failure_reproduced",
    "native_run_completed_reference_unavailable",
    "native_algorithm_failed",
    "native_unsupported_configuration",
    "environment_failed",
    "build_failed",
    "reference_command_ambiguous",
    "source_identity_unknown",
    "patched_diagnostic_only",
    "not_attempted",
}
REPRODUCED_STATUSES = {
    "reproduced_exact",
    "reproduced_with_declared_tolerance",
    "reference_failure_reproduced",
}
NATIVE_EXECUTION_KINDS = {"author_native", "stock_official", "current_native"}
DIAGNOSTIC_EXECUTION_KINDS = {
    "patched_diagnostic",
    "generated_diagnostic",
    "analysis_only",
}
COMPLETION_STATUSES = {"completed", "partial", "failed", "not_started", "not_applicable"}
CERTIFICATE_STATUSES = {
    "completed",
    "partial",
    "failed",
    "not_exposed",
    "not_applicable",
}
SOUNDNESS_LEVELS = {"formal", "empirical", "unknown"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _artifact_errors(
    artifacts: Any, *, root: Path, field: str, required: bool
) -> list[str]:
    errors: list[str] = []
    if not isinstance(artifacts, list):
        return [f"{field} must be a list"]
    if required and not artifacts:
        errors.append(f"{field} must not be empty")
    for index, artifact in enumerate(artifacts):
        prefix = f"{field}[{index}]"
        if not isinstance(artifact, Mapping):
            errors.append(f"{prefix} must be an object")
            continue
        path_value = artifact.get("path")
        expected = artifact.get("sha256")
        if not _nonempty_string(path_value):
            errors.append(f"{prefix}.path is required")
            continue
        if not _nonempty_string(expected):
            errors.append(f"{prefix}.sha256 is required")
            continue
        path = resolve_path(root, path_value)
        if not path.is_file():
            errors.append(f"{prefix}.path is not a file: {path}")
            continue
        actual = sha256(path)
        if actual != expected:
            errors.append(
                f"{prefix}.sha256 mismatch: expected {expected}, got {actual}"
            )
    return errors


def _command_errors(row: Mapping[str, Any], *, root: Path) -> list[str]:
    value = row.get("command_evidence")
    if not _nonempty_string(value):
        return ["command_evidence is required"]
    path = resolve_path(root, value)
    if not path.is_file():
        return [f"command_evidence is not a file: {path}"]
    try:
        command = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"command_evidence cannot be read: {exc}"]
    errors: list[str] = []
    argv = command.get("argv")
    if not isinstance(argv, list) or not argv or not all(
        _nonempty_string(item) for item in argv
    ):
        errors.append("command_evidence.argv must be a nonempty string list")
    if not _nonempty_string(command.get("cwd")):
        errors.append("command_evidence.cwd is required")
    if not isinstance(command.get("exit_code"), int):
        errors.append("command_evidence.exit_code must be an integer")
    for stream in ("stdout", "stderr"):
        stream_name = command.get(stream)
        digest_name = command.get(f"{stream}_sha256")
        if not _nonempty_string(stream_name):
            errors.append(f"command_evidence.{stream} is required")
            continue
        if not _nonempty_string(digest_name):
            errors.append(f"command_evidence.{stream}_sha256 is required")
            continue
        stream_path = path.parent / stream_name
        if not stream_path.is_file():
            errors.append(f"command_evidence.{stream} is missing: {stream_path}")
            continue
        actual = sha256(stream_path)
        if actual != digest_name:
            errors.append(
                f"command_evidence.{stream}_sha256 mismatch: "
                f"expected {digest_name}, got {actual}"
            )
    return errors


def validate_row(
    row: Mapping[str, Any], *, root: Path, diagnostic: bool = False
) -> list[str]:
    errors: list[str] = []
    status = row.get("reproduction_status")
    completion = row.get("completion_status")
    certificate = row.get("certificate_status")
    soundness = row.get("soundness_level")
    execution_kind = row.get("execution_kind")

    for field in ("id", "repo_path", "native_entrypoint"):
        if not _nonempty_string(row.get(field)):
            errors.append(f"{field} is required")
    if not _nonempty_string(row.get("source_sha")):
        errors.append("source_sha is required")
    if not isinstance(row.get("source_changed"), bool):
        errors.append("source_changed must be boolean")
    if status not in REPRODUCTION_STATUSES:
        errors.append(f"invalid reproduction_status: {status!r}")
    if completion not in COMPLETION_STATUSES:
        errors.append(f"invalid completion_status: {completion!r}")
    if certificate not in CERTIFICATE_STATUSES:
        errors.append(f"invalid certificate_status: {certificate!r}")
    if soundness not in SOUNDNESS_LEVELS:
        errors.append(f"invalid soundness_level: {soundness!r}")
    if not isinstance(row.get("primary_comparison_eligible"), bool):
        errors.append("primary_comparison_eligible must be boolean")

    if diagnostic:
        if execution_kind not in DIAGNOSTIC_EXECUTION_KINDS:
            errors.append(
                "diagnostic execution_kind must be patched_diagnostic, "
                "generated_diagnostic, or analysis_only"
            )
        if status != "patched_diagnostic_only":
            errors.append("diagnostic reproduction_status must be patched_diagnostic_only")
        if row.get("primary_comparison_eligible") is not False:
            errors.append("diagnostic cannot be primary comparison eligible")
    else:
        if execution_kind not in NATIVE_EXECUTION_KINDS:
            errors.append(
                "native execution_kind must be author_native, stock_official, "
                "or current_native"
            )
        if status == "patched_diagnostic_only":
            errors.append("patched_diagnostic_only cannot enter native_reproductions")

    hashes = row.get("input_hashes")
    if not isinstance(hashes, Mapping) or not hashes or not all(
        _nonempty_string(key) and _nonempty_string(value)
        for key, value in hashes.items()
    ):
        errors.append("input_hashes must be a nonempty string mapping")

    errors.extend(_command_errors(row, root=root))
    reproduced = status in REPRODUCED_STATUSES
    errors.extend(
        _artifact_errors(
            row.get("fresh_artifacts"),
            root=root,
            field="fresh_artifacts",
            required=reproduced,
        )
    )
    errors.extend(
        _artifact_errors(
            row.get("reference_artifacts"),
            root=root,
            field="reference_artifacts",
            required=reproduced,
        )
    )
    comparison = row.get("comparison_result")
    if reproduced:
        if not isinstance(comparison, Mapping):
            errors.append("comparison_result object is required for reproduced status")
        else:
            errors.extend(
                _artifact_errors(
                    [comparison],
                    root=root,
                    field="comparison_result",
                    required=True,
                )
            )
    elif comparison is not None and not isinstance(comparison, Mapping):
        errors.append("comparison_result must be an object or null")

    if row.get("source_changed") is True and status == "reproduced_exact":
        errors.append("source_changed=true cannot be reproduced_exact")

    requested = row.get("requested_horizon")
    reached = row.get("reached_horizon")
    if not isinstance(requested, (int, float)) or not math.isfinite(requested):
        errors.append("requested_horizon must be finite")
    if not isinstance(reached, (int, float)) or not math.isfinite(reached):
        errors.append("reached_horizon must be finite")
    if isinstance(requested, (int, float)) and isinstance(reached, (int, float)):
        if reached < requested - 1e-12:
            if completion == "completed":
                errors.append("reached_horizon < requested_horizon cannot be completed")
            if certificate == "completed":
                errors.append(
                    "reached_horizon < requested_horizon cannot have completed certificate"
                )

    if completion in {"partial", "failed", "not_started"} and row.get(
        "primary_comparison_eligible"
    ):
        errors.append("failed/partial/not-started run cannot be primary eligible")
    if certificate in {"partial", "failed"} and row.get(
        "primary_comparison_eligible"
    ):
        errors.append("failed/partial certificate cannot be primary eligible")
    if soundness != "formal" and row.get("certificate_claim") == "formal":
        errors.append("unknown/empirical soundness cannot claim formal certificate")

    tolerance = row.get("tolerance")
    if status == "reproduced_with_declared_tolerance":
        if not isinstance(tolerance, Mapping):
            errors.append("declared-tolerance reproduction requires tolerance object")
        else:
            if not isinstance(tolerance.get("value"), (int, float)) or not math.isfinite(
                tolerance.get("value", math.nan)
            ):
                errors.append("tolerance.value must be finite")
            if not _nonempty_string(tolerance.get("source")):
                errors.append("tolerance.source is required")
    elif tolerance is not None:
        errors.append("tolerance is only allowed for declared-tolerance reproduction")
    return errors


def validate_registry(registry: Mapping[str, Any], *, root: Path) -> list[str]:
    errors: list[str] = []
    if registry.get("schema_version") != 1:
        errors.append("schema_version must equal 1")
    if not _nonempty_string(registry.get("run_id")):
        errors.append("run_id is required")
    native = registry.get("native_reproductions")
    diagnostics = registry.get("diagnostics")
    if not isinstance(native, list):
        errors.append("native_reproductions must be a list")
        native = []
    if not isinstance(diagnostics, list):
        errors.append("diagnostics must be a list")
        diagnostics = []
    seen: set[str] = set()
    for section, rows, diagnostic in (
        ("native_reproductions", native, False),
        ("diagnostics", diagnostics, True),
    ):
        for index, row in enumerate(rows):
            prefix = f"{section}[{index}]"
            if not isinstance(row, Mapping):
                errors.append(f"{prefix} must be an object")
                continue
            row_id = row.get("id")
            if row_id in seen:
                errors.append(f"{prefix}.id is duplicated: {row_id}")
            if isinstance(row_id, str):
                seen.add(row_id)
            errors.extend(
                f"{prefix}: {message}"
                for message in validate_row(row, root=root, diagnostic=diagnostic)
            )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("registry", type=Path)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="root for relative evidence paths (default: current directory)",
    )
    args = parser.parse_args()
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    errors = validate_registry(registry, root=args.root.resolve())
    if errors:
        for error in errors:
            print(error)
        return 1
    print(
        f"validated {len(registry['native_reproductions'])} native rows and "
        f"{len(registry['diagnostics'])} diagnostic rows"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
