"""Fail-closed, read-only checks for the ARCH-COMP26 execution matrix."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "benchmarks/archcomp26/manifest.json"


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def _shape_errors(
    candidate: Any, template: Any, path: str
) -> list[str]:
    if not isinstance(template, dict):
        return []
    if not isinstance(candidate, dict):
        return [f"{path}: expected an object"]
    errors: list[str] = []
    if set(candidate) != set(template):
        missing = sorted(set(template) - set(candidate))
        extra = sorted(set(candidate) - set(template))
        errors.append(f"{path}: incomplete override missing={missing} extra={extra}")
        return errors
    for key, value in candidate.items():
        errors.extend(_shape_errors(value, template[key], f"{path}.{key}"))
    return errors


def resolve_cell(matrix: Mapping[str, Any], instance: str, method: str) -> dict[str, Any]:
    defaults = matrix["cell_defaults"]
    override = matrix["cells"][instance][method]
    return {**defaults, **override}


def validate_matrix(
    manifest: Mapping[str, Any], matrix: Mapping[str, Any]
) -> list[str]:
    errors: list[str] = []
    instance_ids = [row["id"] for row in manifest.get("instances", [])]
    methods = list(manifest.get("methods", []))
    if list(matrix.get("methods", [])) != methods:
        errors.append("matrix methods do not match manifest methods")
    if list(matrix.get("cells", {})) != instance_ids:
        errors.append("matrix instance order does not match manifest")
    defaults = matrix.get("cell_defaults")
    required = matrix.get("required_cell_fields")
    if not isinstance(defaults, dict) or set(defaults) != set(required or []):
        errors.append("cell_defaults do not match required_cell_fields")
        return errors

    enums = matrix.get("enums", {})
    for instance in instance_ids:
        method_cells = matrix.get("cells", {}).get(instance)
        if not isinstance(method_cells, dict) or list(method_cells) != methods:
            errors.append(f"{instance}: methods do not match manifest")
            continue
        for method, override in method_cells.items():
            prefix = f"{instance}/{method}"
            if not isinstance(override, dict):
                errors.append(f"{prefix}: override must be an object")
                continue
            extra = sorted(set(override) - set(defaults))
            if extra:
                errors.append(f"{prefix}: unknown override fields {extra}")
            shape_errors: list[str] = []
            for key, value in override.items():
                if key in defaults:
                    shape_errors.extend(
                        _shape_errors(value, defaults[key], f"{prefix}.{key}")
                    )
            errors.extend(shape_errors)
            if extra or shape_errors:
                continue
            resolved = resolve_cell(matrix, instance, method)
            for dotted, value in (
                ("support.status", resolved["support"]["status"]),
                ("run.status", resolved["run"]["status"]),
                ("run.failure_category", resolved["run"]["failure_category"]),
                ("widths.status", resolved["widths"]["status"]),
            ):
                if value not in enums.get(dotted, []):
                    errors.append(f"{prefix}: invalid {dotted}={value!r}")
            status = resolved["run"]["status"]
            category = resolved["run"]["failure_category"]
            detail = resolved["run"]["failure_detail"]
            if status == "not_started" and (category is not None or detail is not None):
                errors.append(f"{prefix}: not_started cell carries failure data")
            if status in {"failed", "timeout", "interrupted"} and (
                category is None or not isinstance(detail, str) or not detail.strip()
            ):
                errors.append(f"{prefix}: terminal failure lacks category/detail")
            if status == "completed" and category is not None:
                errors.append(f"{prefix}: completed cell carries a failure category")
    return errors


def preflight_reasons(
    manifest: Mapping[str, Any], matrix: Mapping[str, Any], instance: str, method: str
) -> list[str]:
    reasons = validate_matrix(manifest, matrix)
    by_id = {row["id"]: row for row in manifest.get("instances", [])}
    if instance not in by_id:
        return [*reasons, "unknown_instance"]
    if method not in manifest.get("methods", []):
        return [*reasons, "unknown_method"]
    if manifest["execution_policy"]["experiments_paused"] is not False:
        reasons.append("experiments_paused")
    if by_id[instance]["contract"]["status"] != "resolved":
        reasons.append("contract_unresolved")
    cell = resolve_cell(matrix, instance, method)
    if cell["support"]["status"] != "supported":
        reasons.append("support_not_supported")
    command = cell["command"]
    if not isinstance(command["argv"], list) or not command["argv"]:
        reasons.append("command_argv_missing")
    if not isinstance(command["cwd"], str) or not command["cwd"].strip():
        reasons.append("command_cwd_missing")
    source = cell["source_identity"]
    if not all(isinstance(source[key], str) and source[key].strip()
               for key in ("kind", "locator")):
        reasons.append("source_identity_missing")
    if not any(isinstance(source[key], str) and source[key].strip()
               for key in ("revision", "sha256")):
        reasons.append("source_revision_or_sha_missing")
    binary = cell["binary_identity"]
    if not all(isinstance(binary[key], str) and binary[key].strip()
               for key in ("path", "sha256")):
        reasons.append("binary_identity_missing")
    runtime = cell["runtime"]
    timeout = runtime["timeout_s"]
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) \
            or not math.isfinite(timeout) or timeout <= 0:
        reasons.append("runtime_timeout_missing")
    if runtime["hardware"] is None or runtime["resource_limits"] is None:
        reasons.append("runtime_budget_missing")
    if cell["run"]["status"] != "not_started":
        reasons.append("cell_not_not_started")
    return list(dict.fromkeys(reasons))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--preflight", nargs=2, metavar=("INSTANCE", "METHOD"))
    args = parser.parse_args(argv)
    manifest = _load(args.manifest)
    matrix_path = ROOT / manifest["execution_matrix"]["path"]
    matrix = _load(matrix_path)
    errors = validate_matrix(manifest, matrix)
    if args.preflight:
        instance, method = args.preflight
        reasons = preflight_reasons(manifest, matrix, instance, method)
        print(json.dumps({
            "instance": instance,
            "method": method,
            "launchable": not reasons,
            "reasons": reasons,
        }, sort_keys=True))
        return 0 if not reasons else 2
    print(json.dumps({
        "status": "passed" if not errors else "failed",
        "cells": sum(len(row) for row in matrix.get("cells", {}).values()),
        "experiments_paused": manifest["execution_policy"]["experiments_paused"],
        "errors": errors,
    }, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
