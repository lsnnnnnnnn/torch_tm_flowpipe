"""Fail-closed, read-only checks for the ARCH-COMP26 execution matrix."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "benchmarks/archcomp26/manifest.json"
MATRIX_SCHEMA = "archcomp26-execution-matrix-v3"
RESULT_SCHEMA = "archcomp26-cell-result-v2"
INSTANCE_SCHEMA = "archcomp26-instance-contract-v1"
OFFICIAL_ASSETS_SCHEMA = "archcomp26-official-assets-v1"
OFFICIAL_ASSETS_AUDIT_SCHEMA = "archcomp26-official-assets-audit-v1"
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
TERMINAL_STATUSES = {"completed", "failed", "timeout", "interrupted", "skipped"}
CANONICAL_OUTCOMES = {
    "completed",
    "validation_rejected",
    "nonfinite",
    "timeout",
    "process_error",
    "compile_error",
    "missing_dependency",
    "trajectory_sanity_failed",
    "analytic_containment_failed",
    "schema_invalid",
    "incomplete_unknown",
}
SOUNDNESS_CLASSES = {
    "formally outward by construction",
    "safeguarded outward under declared IEEE/backend assumptions",
    "independently outward replayed for exact benchmark workload",
    "empirically sampled only",
    "unsound/ineligible on a demonstrated counterexample",
    "unknown",
}
SOUNDNESS_SCOPES = {
    "primitive", "one step", "fixed workload", "multi-step lane", "native build"
}
PLAN_FIELDS = (
    "support",
    "command",
    "source_identity",
    "binary_identity",
    "arithmetic",
    "numerics",
    "controller_execution",
    "property_checker",
    "runtime",
    "measurement_plan",
)
METHOD_PROFILE_FIELDS = {
    "integration.step_size",
    "integration.solution_order",
    "integration.point_order",
    "integration.validation_order",
    "controller_update.nn_calls",
    "remainder.cutoff",
    "remainder.cap",
    "remainder.symbolic_queue",
    "arithmetic.mode",
    "arithmetic.controller_domain",
    "arithmetic.relaxation",
    "arithmetic.dtype",
    "arithmetic.transport",
    "runtime.hardware",
    "runtime.cpu_threads",
    "runtime.resource_limits",
    "runtime.timeout",
    "commands",
    "source_identity",
    "binary_identity",
    "property.checker",
}
CELL_DEFAULT_SHAPE = {
    "support": {"status": None, "blockers": None},
    "command": {"argv": None, "cwd": None},
    "source_identity": {
        "kind": None, "locator": None, "revision": None, "sha256": None,
    },
    "binary_identity": {"path": None, "sha256": None},
    "arithmetic": {
        "mode": None, "controller_domain": None, "relaxation": None,
        "dtype": None, "transport": None,
    },
    "numerics": {
        "integration": {
            "step_size": {"mode": None, "value": None},
            "solution_order": {"mode": None, "value": None},
            "point_order": {"mode": None, "value": None},
            "validation_order": {"mode": None, "value": None},
            "semantics": None,
        },
        "remainder": {
            "cutoff": {"mode": None, "value": None},
            "cap": {"mode": None, "value": None},
            "symbolic_queue": {"mode": None, "value": None},
            "semantics": None,
        },
    },
    "controller_execution": {
        "scheduled_updates": None,
        "nn_calls": {"mode": None, "value": None},
        "nn_call_semantics": None,
    },
    "property_checker": {
        "mode": None, "identity": None, "semantics": None,
        "certificate_semantics": None, "early_stop_policy": None,
    },
    "runtime": {
        "hardware": None, "cpu_threads": None, "gpu": None,
        "resource_limits": None, "timeout_s": None,
    },
    "measurement_plan": {
        "cold_runs": None, "target_steady_runs": None, "steady_runs": None,
        "shortfall_reason": None,
        "fresh_process_per_run": None, "timing_boundary_version": None,
    },
    "run": {
        "status": None, "failure_category": None, "failure_detail": None,
    },
    "result_record": {"schema_version": None, "path": None, "sha256": None},
}


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value!r}")


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(
        path.read_text(encoding="utf-8"), parse_constant=_reject_nonfinite
    )
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None


def _is_finite_number(value: Any, *, positive: bool = False) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        and (value > 0 if positive else value >= 0)
    )


def _bound_file(
    root: Path, raw_path: Any, expected_sha: Any, label: str
) -> tuple[Path | None, list[str]]:
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None, [f"{label}: repository-relative path is missing"]
    relative = Path(raw_path)
    if relative.is_absolute():
        return None, [f"{label}: path must be repository-relative"]
    resolved_root = root.resolve()
    candidate = (resolved_root / relative).resolve()
    try:
        candidate.relative_to(resolved_root)
    except ValueError:
        return None, [f"{label}: path escapes the repository"]
    if not candidate.is_file():
        return None, [f"{label}: file is missing"]
    if not _is_sha256(expected_sha):
        return None, [f"{label}: SHA-256 must be 64 lowercase hex characters"]
    if _sha256(candidate) != expected_sha:
        return None, [f"{label}: SHA-256 does not match current file"]
    return candidate, []


def _exact_keys(value: Any, expected: set[str], label: str) -> list[str]:
    if not isinstance(value, dict):
        return [f"{label}: expected an object"]
    if set(value) == expected:
        return []
    return [
        f"{label}: fields differ missing={sorted(expected - set(value))} "
        f"extra={sorted(set(value) - expected)}"
    ]


def _nonempty(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, dict, tuple)):
        return bool(value)
    return True


def _dotted(value: Mapping[str, Any], path: str) -> Any:
    current: Any = value
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _has_dotted(value: Mapping[str, Any], path: str) -> bool:
    current: Any = value
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return False
        current = current[part]
    return True


def _is_non_negative_int(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value >= 0


def _mode_is(value: Any, *allowed: str) -> bool:
    return isinstance(value, str) and value in allowed


def _tagged_value_errors(
    value: Any,
    label: str,
    *,
    configured_mode: str,
    integer: bool = False,
    positive: bool = False,
    allow_adaptive: bool = False,
) -> list[str]:
    errors = _exact_keys(value, {"mode", "value"}, label)
    if errors:
        return errors
    allowed = {"unresolved", configured_mode, "not_applicable"}
    if allow_adaptive:
        allowed.add("adaptive")
    mode = value["mode"]
    raw = value["value"]
    if not _mode_is(mode, *allowed):
        return [f"{label}.mode: invalid applicability tag {mode!r}"]
    if mode != configured_mode:
        if raw is not None:
            errors.append(f"{label}.value: must be null when mode is {mode!r}")
        return errors
    if integer:
        valid = _is_non_negative_int(raw) and (not positive or raw > 0)
        expected = "positive integer" if positive else "non-negative integer"
    else:
        valid = _is_finite_number(raw, positive=positive)
        expected = (
            "finite positive number" if positive else "finite non-negative number"
        )
    if not valid:
        errors.append(f"{label}.value: expected a {expected}")
    return errors


def _contract_type_errors(
    fields: Mapping[str, Any], profile: str, instance: str
) -> list[str]:
    errors: list[str] = []
    list_fields = ("variable_order", "controller.input_output_order")
    for dotted in list_fields:
        value = _dotted(fields, dotted)
        if not isinstance(value, list) or not value or not all(
            isinstance(item, str) and item for item in value
        ) or len(set(value)) != len(value):
            errors.append(f"{instance}: resolved contract field {dotted} has invalid order")
    partitions = _dotted(fields, "initial_set.partitions")
    if not isinstance(partitions, list) or not partitions:
        errors.append(
            f"{instance}: resolved contract field initial_set.partitions must be a non-empty array"
        )
    positive_numbers = {
        "full_execution_contract_v1": (
            "integration.horizon", "controller_update.period",
        ),
        "discrete_execution_contract_v1": ("discrete.sample_period",),
    }.get(profile, ())
    for dotted in positive_numbers:
        if not _is_finite_number(_dotted(fields, dotted), positive=True):
            errors.append(f"{instance}: resolved contract field {dotted} must be positive")
    positive_integers = {
        "full_execution_contract_v1": (),
        "discrete_execution_contract_v1": (
            "discrete.transition_count", "controller_update.period_steps",
        ),
    }.get(profile, ())
    for dotted in positive_integers:
        value = _dotted(fields, dotted)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            errors.append(
                f"{instance}: resolved contract field {dotted} must be a positive integer"
            )
    updates_path = (
        "controller_update.scheduled_updates"
        if isinstance(profile, str) and profile in {
            "full_execution_contract_v1", "discrete_execution_contract_v1"
        }
        else None
    )
    if updates_path is not None:
        value = _dotted(fields, updates_path)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            errors.append(
                f"{instance}: resolved contract field {updates_path} "
                "must be a non-negative integer"
            )
    text_fields = [
        "controller_update.schedule_semantics",
        "property.formula",
        "property.pass_condition",
        (
            "property.time_semantics"
            if profile == "full_execution_contract_v1"
            else "property.step_semantics"
        ),
    ]
    for dotted in text_fields:
        value = _dotted(fields, dotted)
        if not isinstance(value, str) or not value.strip():
            errors.append(
                f"{instance}: resolved contract field {dotted} "
                "must be non-empty text"
            )
    if profile == "discrete_execution_contract_v1":
        index_set = _dotted(fields, "discrete.index_set")
        transitions = _dotted(fields, "discrete.transition_count")
        if not isinstance(index_set, list) or not index_set or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in index_set
        ) or index_set != list(range(len(index_set))) \
                or not isinstance(transitions, int) \
                or len(index_set) != transitions + 1:
            errors.append(
                f"{instance}: discrete.index_set must be contiguous 0..transition_count"
            )
    return errors


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


def _validate_contracts(
    manifest: Mapping[str, Any], root: Path
) -> list[str]:
    errors: list[str] = []
    profiles = manifest.get("unresolved_field_profiles")
    if not isinstance(profiles, dict):
        return ["manifest unresolved_field_profiles must be an object"]
    schema_link = manifest.get("instance_contract_record")
    schema_errors = _exact_keys(
        schema_link, {"path", "schema_version", "sha256"},
        "instance_contract_record",
    )
    errors.extend(schema_errors)
    if not schema_errors:
        if schema_link["schema_version"] != INSTANCE_SCHEMA:
            errors.append("instance_contract_record has the wrong schema_version")
        _, bound_errors = _bound_file(
            root, schema_link["path"], schema_link["sha256"],
            "instance_contract_record",
        )
        errors.extend(bound_errors)

    inventory_by_id: dict[str, Any] = {}
    inventory_link = _dotted(manifest, "official_sources.asset_inventory")
    inventory_errors = _exact_keys(
        inventory_link,
        {"path", "schema_version", "sha256", "audit_receipt", "status"},
        "official_sources.asset_inventory",
    )
    errors.extend(inventory_errors)
    inventory_path: Path | None = None
    if not inventory_errors:
        if inventory_link["schema_version"] != OFFICIAL_ASSETS_SCHEMA:
            errors.append("official asset inventory has the wrong schema_version")
        inventory_path, bound_errors = _bound_file(
            root,
            inventory_link["path"],
            inventory_link["sha256"],
            "official_sources.asset_inventory",
        )
        errors.extend(bound_errors)
        audit_link = inventory_link["audit_receipt"]
        audit_errors = _exact_keys(
            audit_link, {"path", "schema_version", "sha256"},
            "official_sources.asset_inventory.audit_receipt",
        )
        errors.extend(audit_errors)
        audit_path: Path | None = None
        if not audit_errors:
            if audit_link["schema_version"] != OFFICIAL_ASSETS_AUDIT_SCHEMA:
                errors.append("official asset audit receipt has the wrong schema_version")
            audit_path, audit_bound_errors = _bound_file(
                root, audit_link["path"], audit_link["sha256"],
                "official_sources.asset_inventory.audit_receipt",
            )
            errors.extend(audit_bound_errors)
        if audit_path is not None:
            try:
                audit = _load(audit_path)
            except (OSError, ValueError, json.JSONDecodeError) as error:
                errors.append(f"official asset audit receipt cannot be loaded: {error}")
            else:
                pinned_repository = _dotted(
                    manifest, "official_sources.benchmark_repository"
                )
                expected_inventory = {
                    "path": inventory_link["path"],
                    "schema_version": inventory_link["schema_version"],
                    "sha256": inventory_link["sha256"],
                }
                if audit.get("schema_version") != audit_link["schema_version"]:
                    errors.append("official asset audit receipt schema mismatch")
                if audit.get("inventory") != expected_inventory:
                    errors.append("official asset audit receipt inventory binding mismatch")
                if not isinstance(pinned_repository, Mapping) or (
                    audit.get("source_repository") != pinned_repository.get("url")
                    or audit.get("remote_ref_observed_commit")
                    != pinned_repository.get("commit")
                    or audit.get("detached_checkout_commit")
                    != pinned_repository.get("commit")
                ):
                    errors.append("official asset audit receipt repository identity mismatch")
                verification = audit.get("verification")
                if not isinstance(verification, Mapping) or (
                    verification.get("result") != "passed"
                    or verification.get("missing_count") != 0
                    or verification.get("mismatch_count") != 0
                ):
                    errors.append("official asset audit receipt did not pass")
    if inventory_path is not None:
        try:
            inventory = _load(inventory_path)
            pinned_repository = _dotted(manifest, "official_sources.benchmark_repository")
            if (
                inventory.get("schema_version") != inventory_link["schema_version"]
                or not isinstance(pinned_repository, Mapping)
                or inventory.get("source_repository") != pinned_repository.get("url")
                or inventory.get("source_commit") != pinned_repository.get("commit")
            ):
                errors.append("official asset inventory repository identity mismatch")
            inventory_by_id = {
                row["id"]: row for row in inventory.get("instances", [])
                if isinstance(row, dict) and "id" in row
            }
        except (OSError, ValueError, json.JSONDecodeError) as error:
            errors.append(f"official asset inventory cannot be loaded: {error}")

    def source_is_bound(
        instance: str, stem: str, source: Any, sha256: Any
    ) -> list[str]:
        label = f"{instance}.contract.{stem}"
        if not isinstance(source, str) or not _is_sha256(sha256):
            return [f"{label}: source/SHA-256 identity is invalid"]
        relative = Path(source)
        if not relative.is_absolute() and (root / relative).is_file():
            _, local_errors = _bound_file(root, source, sha256, label)
            return local_errors
        official = inventory_by_id.get(instance, {})
        candidates: list[Mapping[str, Any]] = []
        if stem in {"dynamics", "transition"}:
            item = official.get(stem)
            if isinstance(item, Mapping):
                candidates.append(item)
        elif stem == "controller":
            candidates.extend(
                item for item in official.get("controller_candidates", [])
                if isinstance(item, Mapping)
            )
        elif stem == "initial_set":
            item = official.get("specification")
            if isinstance(item, Mapping):
                candidates.append(item)
        if not any(
            item.get("path") == source and item.get("sha256") == sha256
            for item in candidates
        ):
            return [
                f"{label}: external identity is absent from the pinned official inventory"
            ]
        return []

    for row in manifest.get("instances", []):
        instance = row.get("id", "<missing-instance>")
        contract = row.get("contract")
        if not isinstance(contract, dict):
            errors.append(f"{instance}: contract must be an object")
            continue
        status = contract.get("status")
        if status == "unresolved":
            profile = contract.get("unresolved_field_profile")
            if not isinstance(profile, str) or profile not in profiles:
                errors.append(f"{instance}: unknown unresolved contract profile {profile!r}")
            continue
        if status != "resolved":
            errors.append(f"{instance}: invalid contract status {status!r}")
            continue
        profile = contract.get("unresolved_field_profile")
        required = profiles.get(profile) if isinstance(profile, str) else None
        if not isinstance(required, list) or not required:
            errors.append(f"{instance}: resolved contract has no known field profile")
            continue
        link = contract.get("record")
        link_errors = _exact_keys(
            link, {"path", "sha256", "schema_version"}, f"{instance}.contract.record"
        )
        errors.extend(link_errors)
        if link_errors:
            continue
        if link["schema_version"] != INSTANCE_SCHEMA:
            errors.append(f"{instance}.contract.record: wrong schema_version")
        record_path, bound_errors = _bound_file(
            root, link["path"], link["sha256"], f"{instance}.contract.record"
        )
        errors.extend(bound_errors)
        if record_path is None:
            continue
        try:
            record = _load(record_path)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            errors.append(f"{instance}.contract.record: cannot load: {error}")
            continue
        record_errors = _exact_keys(
            record, {"schema_version", "instance_id", "profile", "fields", "evidence"},
            f"{instance}.contract.record",
        )
        errors.extend(record_errors)
        if record_errors:
            continue
        if record["schema_version"] != INSTANCE_SCHEMA \
                or record["instance_id"] != instance or record["profile"] != profile:
            errors.append(f"{instance}.contract.record: identity/profile mismatch")
        fields = record["fields"]
        if not isinstance(fields, Mapping):
            errors.append(f"{instance}.contract.record.fields: expected an object")
            continue
        for dotted in sorted(METHOD_PROFILE_FIELDS):
            if _has_dotted(fields, dotted):
                errors.append(
                    f"{instance}: method-specific field {dotted} must live in "
                    "the execution cell"
                )
        shared_required = [
            dotted for dotted in required if dotted not in METHOD_PROFILE_FIELDS
        ]
        for dotted in shared_required:
            value = _dotted(fields, dotted)
            if not _nonempty(value):
                errors.append(f"{instance}: resolved contract field {dotted} is missing")
            elif dotted.endswith("sha256") and not _is_sha256(value):
                errors.append(f"{instance}: resolved contract field {dotted} is not SHA-256")
        errors.extend(_contract_type_errors(fields, profile, instance))
        for stem in ("dynamics", "transition", "controller", "initial_set"):
            source_key = f"{stem}.source"
            sha_key = f"{stem}.sha256"
            if source_key not in shared_required:
                continue
            errors.extend(source_is_bound(
                instance, stem, _dotted(fields, source_key), _dotted(fields, sha_key)
            ))
        partitions = _dotted(fields, "initial_set.partitions")
        boxes_sha = _dotted(fields, "initial_set.boxes_sha256")
        if _nonempty(partitions) and boxes_sha != _canonical_sha256(partitions):
            errors.append(
                f"{instance}: initial_set.boxes_sha256 does not bind partitions"
            )
        evidence = record["evidence"]
        supported: set[str] = set()
        if not isinstance(evidence, list) or not evidence:
            errors.append(f"{instance}.contract.record.evidence: expected non-empty array")
            evidence = []
        for index, item in enumerate(evidence):
            label = f"{instance}.contract.record.evidence[{index}]"
            item_errors = _exact_keys(item, {"path", "sha256", "supports"}, label)
            errors.extend(item_errors)
            if item_errors:
                continue
            supports = item["supports"]
            if not isinstance(supports, list) or not supports \
                    or not all(isinstance(name, str) and name for name in supports):
                errors.append(f"{label}.supports: expected non-empty string array")
            else:
                if len(supports) != len(set(supports)):
                    errors.append(f"{label}.supports: duplicate fields are not allowed")
                unknown = sorted(set(supports) - set(shared_required))
                if unknown:
                    errors.append(f"{label}.supports: unknown shared fields {unknown}")
                supported.update(supports)
            _, item_bound_errors = _bound_file(
                root, item["path"], item["sha256"], label
            )
            errors.extend(item_bound_errors)
        missing_support = sorted(set(shared_required) - supported)
        if missing_support:
            errors.append(
                f"{instance}.contract.record.evidence: unsupported fields {missing_support}"
            )
    return errors


def _validate_extent(value: Any, label: str) -> list[str]:
    errors = _exact_keys(value, {"kind", "value"}, label)
    if errors:
        return errors
    kind = value["kind"]
    extent = value["value"]
    if not isinstance(kind, str) or kind not in {"time_s", "steps"}:
        errors.append(f"{label}.kind: expected time_s or steps")
    if not _is_finite_number(extent):
        errors.append(f"{label}.value: expected a finite non-negative number")
    elif kind == "steps" and (
        isinstance(extent, bool) or not isinstance(extent, int)
    ):
        errors.append(f"{label}.value: step extent must be an integer")
    elif kind == "steps" and not isinstance(extent, int):
        errors.append(f"{label}.value: step extent must be an integer")
    return errors


def _validate_failure(value: Any, label: str) -> list[str]:
    expected = {
        "stage", "reason_code", "step", "time_s", "detail", "exit_code", "signal"
    }
    errors = _exact_keys(value, expected, label)
    if errors:
        return errors
    for name in ("stage", "reason_code", "detail"):
        if not isinstance(value[name], str) or not value[name].strip():
            errors.append(f"{label}.{name}: expected a non-empty string")
    step = value["step"]
    if step is not None and (
        isinstance(step, bool) or not isinstance(step, int) or step < 0
    ):
        errors.append(f"{label}.step: expected null or a non-negative integer")
    time_s = value["time_s"]
    if time_s is not None and not _is_finite_number(time_s):
        errors.append(f"{label}.time_s: expected null or finite non-negative time")
    exit_code = value["exit_code"]
    if exit_code is not None and (
        isinstance(exit_code, bool) or not isinstance(exit_code, int)
    ):
        errors.append(f"{label}.exit_code: expected null or an integer")
    signal = value["signal"]
    if signal is not None and (
        isinstance(signal, bool) or not isinstance(signal, int) or signal <= 0
    ):
        errors.append(f"{label}.signal: expected null or a positive integer")
    return errors


def _validate_artifact(
    value: Any, root: Path, label: str, *, role_required: bool = False
) -> list[str]:
    if not isinstance(value, dict):
        return [f"{label}: expected an object"]
    required = {"path", "sha256"} | ({"role"} if role_required else set())
    allowed = {"path", "sha256", "role"}
    missing = required - set(value)
    extra = set(value) - allowed
    if missing or extra:
        return [
            f"{label}: fields differ missing={sorted(missing)} extra={sorted(extra)}"
        ]
    errors: list[str] = []
    if "role" in value and (
        not isinstance(value["role"], str) or not value["role"].strip()
    ):
        errors.append(f"{label}.role: expected a non-empty string")
    _, bound_errors = _bound_file(root, value["path"], value["sha256"], label)
    errors.extend(bound_errors)
    return errors


def _validate_sample(value: Any, root: Path, label: str) -> list[str]:
    expected = {
        "role", "index", "outcome", "timing_s", "peak_memory_bytes",
        "validated_extent", "accepted_steps", "rejected_steps", "nn_calls",
        "failure", "artifact",
    }
    errors = _exact_keys(value, expected, label)
    if errors:
        return errors
    role = value["role"]
    if not isinstance(role, str) or role not in {"cold", "steady", "diagnostic"}:
        errors.append(f"{label}.role: invalid sample role {role!r}")
    if isinstance(value["index"], bool) or not isinstance(value["index"], int) \
            or value["index"] < 0:
        errors.append(f"{label}.index: expected a non-negative integer")
    outcome = value["outcome"]
    if not isinstance(outcome, str) or outcome not in CANONICAL_OUTCOMES:
        errors.append(f"{label}.outcome: invalid canonical outcome {outcome!r}")
    timing_keys = {
        "process_total", "driver_total", "solver_core", "compile",
        "validation", "observer_output", "plot_report",
    }
    timing = value["timing_s"]
    timing_errors = _exact_keys(timing, timing_keys, f"{label}.timing_s")
    errors.extend(timing_errors)
    if not timing_errors:
        for name, seconds in timing.items():
            if seconds is not None and not _is_finite_number(
                seconds, positive=name == "process_total"
            ):
                errors.append(f"{label}.timing_s.{name}: invalid duration")
        if outcome == "completed" and not _is_finite_number(
            timing["process_total"], positive=True
        ):
            errors.append(f"{label}: completed sample lacks positive process_total")
    memory = value["peak_memory_bytes"]
    memory_errors = _exact_keys(memory, {"host", "device"}, f"{label}.peak_memory_bytes")
    errors.extend(memory_errors)
    if not memory_errors:
        for name, count in memory.items():
            if count is not None and (
                isinstance(count, bool) or not isinstance(count, int) or count < 0
            ):
                errors.append(f"{label}.peak_memory_bytes.{name}: invalid byte count")
    errors.extend(_validate_extent(value["validated_extent"], f"{label}.validated_extent"))
    for name in ("accepted_steps", "rejected_steps", "nn_calls"):
        count = value[name]
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            errors.append(f"{label}.{name}: expected a non-negative integer")
    failure = value["failure"]
    if outcome == "completed" and failure is not None:
        errors.append(f"{label}: completed sample carries a failure")
    if outcome != "completed" and not isinstance(failure, dict):
        errors.append(f"{label}: non-completed sample lacks first-failure data")
    elif outcome != "completed":
        errors.extend(_validate_failure(failure, f"{label}.failure"))
    errors.extend(_validate_artifact(value["artifact"], root, f"{label}.artifact"))
    return errors


def _validate_width_measurement(
    value: Any, root: Path, label: str, coordinates: list[str]
) -> list[str]:
    errors = _exact_keys(value, {"status", "domain", "per_coordinate", "artifact"}, label)
    if errors:
        return errors
    status = value["status"]
    if not isinstance(status, str) or status not in {"complete", "unavailable"}:
        errors.append(f"{label}.status: invalid width status {status!r}")
        return errors
    domain = value["domain"]
    domain_errors = _exact_keys(domain, {"kind", "start", "end"}, f"{label}.domain")
    errors.extend(domain_errors)
    if not domain_errors:
        if not isinstance(domain["kind"], str) \
                or domain["kind"] not in {"time_s", "steps"}:
            errors.append(f"{label}.domain.kind: invalid domain kind")
        if not _is_finite_number(domain["start"]) or not _is_finite_number(domain["end"]):
            errors.append(f"{label}.domain: bounds must be finite and non-negative")
        elif domain["end"] < domain["start"]:
            errors.append(f"{label}.domain: end precedes start")
    if status == "unavailable":
        if value["per_coordinate"] != [] or value["artifact"] is not None:
            errors.append(f"{label}: unavailable width must have no values or artifact")
        return errors
    rows = value["per_coordinate"]
    if not isinstance(rows, list) or [
        row.get("coordinate") if isinstance(row, dict) else None for row in rows
    ] != coordinates:
        errors.append(f"{label}.per_coordinate: coordinate order does not match")
    else:
        for index, row in enumerate(rows):
            row_label = f"{label}.per_coordinate[{index}]"
            row_errors = _exact_keys(
                row, {"coordinate", "union", "per_partition_width"}, row_label
            )
            errors.extend(row_errors)
            if row_errors:
                continue
            union = row["union"]
            union_errors = _exact_keys(union, {"lo", "hi", "width"}, f"{row_label}.union")
            errors.extend(union_errors)
            union_width = union.get("width") if isinstance(union, Mapping) else None
            if not union_errors:
                lo, hi, width = union["lo"], union["hi"], union["width"]
                if not all(_is_finite_number(number) for number in (width,)) \
                        or not all(
                            not isinstance(number, bool)
                            and isinstance(number, (int, float))
                            and math.isfinite(number)
                            for number in (lo, hi)
                        ):
                    errors.append(f"{row_label}.union: invalid finite interval")
                elif hi < lo or not math.isclose(width, hi - lo, rel_tol=1e-12, abs_tol=1e-12):
                    errors.append(f"{row_label}.union: width does not equal hi-lo")
            partition = row["per_partition_width"]
            partition_errors = _exact_keys(
                partition, {"mean", "max"}, f"{row_label}.per_partition_width"
            )
            errors.extend(partition_errors)
            if not partition_errors:
                mean, maximum = partition["mean"], partition["max"]
                if not _is_finite_number(mean) or not _is_finite_number(maximum) \
                        or mean > maximum:
                    errors.append(f"{row_label}.per_partition_width: invalid mean/max")
                elif _is_finite_number(union_width) \
                        and maximum > union_width and not math.isclose(
                    maximum, union_width, rel_tol=1e-12, abs_tol=1e-12
                ):
                    errors.append(
                        f"{row_label}.per_partition_width: max exceeds union width"
                    )
    if value["artifact"] is None:
        errors.append(f"{label}: complete width lacks artifact")
    else:
        errors.extend(_validate_artifact(value["artifact"], root, f"{label}.artifact"))
    return errors


def _validate_widths(value: Any, root: Path, label: str) -> list[str]:
    expected = {
        "status", "common_prefix", "coordinate_order", "coordinate_units",
        "aggregation_semantics", "endpoint", "last_segment_tube",
        "full_horizon_tube", "trajectory_artifact",
    }
    errors = _exact_keys(value, expected, label)
    if errors:
        return errors
    status = value["status"]
    if not isinstance(status, str) \
            or status not in {"complete", "partial", "unavailable"}:
        errors.append(f"{label}.status: invalid width status {status!r}")
    errors.extend(_validate_extent(value["common_prefix"], f"{label}.common_prefix"))
    coordinates = value["coordinate_order"]
    units = value["coordinate_units"]
    if not isinstance(coordinates, list) or not coordinates \
            or not all(isinstance(item, str) and item for item in coordinates) \
            or len(set(coordinates)) != len(coordinates):
        errors.append(f"{label}.coordinate_order: expected unique non-empty names")
        coordinates = []
    if not isinstance(units, list) or len(units) != len(coordinates) \
            or not all(isinstance(item, str) and item for item in units):
        errors.append(f"{label}.coordinate_units: must align with coordinate_order")
    if value["aggregation_semantics"] != "union_and_per_partition":
        errors.append(f"{label}.aggregation_semantics: unsupported semantics")
    for name in ("endpoint", "last_segment_tube", "full_horizon_tube"):
        errors.extend(_validate_width_measurement(
            value[name], root, f"{label}.{name}", coordinates
        ))
    if status == "complete":
        if any(value[name].get("status") != "complete" for name in (
            "endpoint", "last_segment_tube", "full_horizon_tube"
        ) if isinstance(value[name], dict)):
            errors.append(f"{label}: complete widths require all three complete views")
        if value["trajectory_artifact"] is None:
            errors.append(f"{label}: complete widths lack trajectory artifact")
    if value["trajectory_artifact"] is not None:
        errors.extend(_validate_artifact(
            value["trajectory_artifact"], root, f"{label}.trajectory_artifact"
        ))
    return errors


def _resolved_contract_record(
    instance_row: Mapping[str, Any], root: Path
) -> Mapping[str, Any] | None:
    link = _dotted(instance_row, "contract.record")
    if not isinstance(link, Mapping):
        return None
    path, _ = _bound_file(
        root, link.get("path"), link.get("sha256"), "resolved instance contract"
    )
    if path is None:
        return None
    try:
        return _load(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def _contract_extent(contract: Mapping[str, Any]) -> dict[str, Any] | None:
    profile = contract.get("profile")
    fields = contract.get("fields")
    if not isinstance(fields, Mapping):
        return None
    if profile == "full_execution_contract_v1":
        return {"kind": "time_s", "value": _dotted(fields, "integration.horizon")}
    if profile == "discrete_execution_contract_v1":
        return {"kind": "steps", "value": _dotted(fields, "discrete.transition_count")}
    return None


def _contract_work_counts(
    contract: Mapping[str, Any], cell: Mapping[str, Any]
) -> tuple[int | None, int | None]:
    fields = contract.get("fields")
    if not isinstance(fields, Mapping):
        return None, None
    nn_mode = _dotted(cell, "controller_execution.nn_calls.mode")
    nn_calls = _dotted(cell, "controller_execution.nn_calls.value")
    if nn_mode == "not_applicable":
        nn_calls = 0
    elif nn_mode != "exact" or not _is_non_negative_int(nn_calls):
        nn_calls = None
    profile = contract.get("profile")
    if profile == "discrete_execution_contract_v1":
        accepted = _dotted(fields, "discrete.transition_count")
        if isinstance(accepted, bool) or not isinstance(accepted, int) \
                or accepted < 0:
            accepted = None
        return accepted, nn_calls
    if profile != "full_execution_contract_v1":
        return None, nn_calls
    horizon = _dotted(fields, "integration.horizon")
    step_mode = _dotted(cell, "numerics.integration.step_size.mode")
    step = _dotted(cell, "numerics.integration.step_size.value")
    if not _is_finite_number(horizon, positive=True) \
            or step_mode != "fixed" or not _is_finite_number(step, positive=True):
        return None, nn_calls
    ratio = horizon / step
    nearest = round(ratio)
    accepted = nearest if math.isclose(
        ratio, nearest, rel_tol=1e-12, abs_tol=1e-12
    ) else math.ceil(ratio)
    return accepted, nn_calls


def _accepted_steps_for_extent(
    contract: Mapping[str, Any], cell: Mapping[str, Any], extent: Any
) -> int | None:
    if not isinstance(extent, Mapping):
        return None
    value = extent.get("value")
    profile = contract.get("profile")
    if profile == "discrete_execution_contract_v1":
        if extent.get("kind") != "steps" or isinstance(value, bool) \
                or not isinstance(value, int) or value < 0:
            return None
        return value
    if profile != "full_execution_contract_v1" \
            or extent.get("kind") != "time_s" or not _is_finite_number(value):
        return None
    if value == 0:
        return 0
    step_mode = _dotted(cell, "numerics.integration.step_size.mode")
    step = _dotted(cell, "numerics.integration.step_size.value")
    if step_mode != "fixed" or not _is_finite_number(step, positive=True):
        return None
    ratio = value / step
    nearest = round(ratio)
    return nearest if math.isclose(
        ratio, nearest, rel_tol=1e-12, abs_tol=1e-12
    ) else math.ceil(ratio)


def _extent_is_prefix(value: Any, requested: Any) -> bool:
    if not isinstance(value, Mapping) or not isinstance(requested, Mapping):
        return False
    if value.get("kind") != requested.get("kind"):
        return False
    prefix = value.get("value")
    end = requested.get("value")
    return _is_finite_number(prefix) and _is_finite_number(end) and prefix <= end


def _last_segment_start(
    contract: Mapping[str, Any], cell: Mapping[str, Any], end: Any
) -> float | int | None:
    if not _is_finite_number(end):
        return None
    if end == 0:
        return 0
    if contract.get("profile") == "discrete_execution_contract_v1":
        return max(0, end - 1)
    step_mode = _dotted(cell, "numerics.integration.step_size.mode")
    step = _dotted(cell, "numerics.integration.step_size.value")
    if step_mode != "fixed" or not _is_finite_number(step, positive=True):
        return None
    ratio = end / step
    nearest = round(ratio)
    count = nearest if math.isclose(
        ratio, nearest, rel_tol=1e-12, abs_tol=1e-12
    ) else math.ceil(ratio)
    return max(0.0, (count - 1) * step)


def _same_domain(actual: Any, expected: Mapping[str, Any]) -> bool:
    if not isinstance(actual, Mapping) or set(actual) != {"kind", "start", "end"}:
        return False
    if actual.get("kind") != expected.get("kind"):
        return False
    return all(
        _is_finite_number(actual.get(name))
        and _is_finite_number(expected.get(name))
        and math.isclose(
            float(actual[name]), float(expected[name]),
            rel_tol=1e-12, abs_tol=1e-12,
        )
        for name in ("start", "end")
    )


def _validate_result_record(
    record: Mapping[str, Any],
    root: Path,
    prefix: str,
    instance_row: Mapping[str, Any],
    method: str,
    cell: Mapping[str, Any],
) -> list[str]:
    expected = {
        "schema_version", "instance_id", "method", "contract_identity",
        "measurement_plan", "run", "property", "eligibility", "samples",
        "widths", "artifacts",
    }
    errors = _exact_keys(record, expected, f"{prefix}.result_record")
    if errors:
        return errors
    if record["schema_version"] != RESULT_SCHEMA:
        errors.append(f"{prefix}.result_record: wrong schema_version")
    if record["instance_id"] != instance_row["id"] or record["method"] != method:
        errors.append(f"{prefix}.result_record: instance/method identity mismatch")
    identity = record["contract_identity"]
    identity_errors = _exact_keys(
        identity, {"instance_contract_sha256", "cell_plan_sha256"},
        f"{prefix}.result_record.contract_identity",
    )
    errors.extend(identity_errors)
    if not identity_errors:
        expected_instance = _dotted(instance_row, "contract.record.sha256")
        expected_plan = _canonical_sha256({name: cell[name] for name in PLAN_FIELDS})
        if not _is_sha256(expected_instance) \
                or identity["instance_contract_sha256"] != expected_instance:
            errors.append(f"{prefix}.result_record: instance contract identity mismatch")
        if identity["cell_plan_sha256"] != expected_plan:
            errors.append(f"{prefix}.result_record: cell plan identity mismatch")
    if record["measurement_plan"] != cell["measurement_plan"]:
        errors.append(f"{prefix}.result_record: measurement plan mismatch")

    contract = _resolved_contract_record(instance_row, root)
    expected_extent = _contract_extent(contract) if contract is not None else None
    contract_variables = (
        _dotted(contract, "fields.variable_order") if contract is not None else None
    )
    expected_accepted, expected_nn_calls = (
        _contract_work_counts(contract, cell)
        if contract is not None else (None, None)
    )

    run_value = record["run"]
    run = run_value if isinstance(run_value, dict) else {}
    run_keys = {
        "status", "outcome", "requested_extent", "validated_extent",
        "requested_horizon_completed", "accepted_steps", "rejected_steps",
        "nn_calls", "first_failure",
    }
    run_errors = _exact_keys(run, run_keys, f"{prefix}.result_record.run")
    errors.extend(run_errors)
    if not run_errors:
        if run["status"] != cell["run"]["status"]:
            errors.append(f"{prefix}.result_record: run status mismatch")
        if not isinstance(run["status"], str) or run["status"] not in TERMINAL_STATUSES:
            errors.append(f"{prefix}.result_record: run status is not terminal")
        if not isinstance(run["outcome"], str) \
                or run["outcome"] not in CANONICAL_OUTCOMES:
            errors.append(f"{prefix}.result_record: invalid canonical outcome")
        errors.extend(_validate_extent(
            run["requested_extent"], f"{prefix}.result_record.run.requested_extent"
        ))
        errors.extend(_validate_extent(
            run["validated_extent"], f"{prefix}.result_record.run.validated_extent"
        ))
        for name in ("accepted_steps", "rejected_steps", "nn_calls"):
            count = run[name]
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                errors.append(f"{prefix}.result_record.run.{name}: invalid count")
        if not isinstance(run["requested_horizon_completed"], bool):
            errors.append(f"{prefix}.result_record.run: completion flag must be Boolean")
        if expected_extent is None:
            errors.append(
                f"{prefix}.result_record: cannot bind resolved contract extent"
            )
        elif run["requested_extent"] != expected_extent:
            errors.append(
                f"{prefix}.result_record: requested extent does not match "
                "resolved instance contract"
            )
        if not _extent_is_prefix(run["validated_extent"], run["requested_extent"]):
            errors.append(
                f"{prefix}.result_record: validated extent is outside requested extent"
            )
        expected_prefix_steps = (
            _accepted_steps_for_extent(contract, cell, run["validated_extent"])
            if contract is not None else None
        )
        adaptive_steps = (
            contract is not None
            and contract.get("profile") == "full_execution_contract_v1"
            and _dotted(cell, "numerics.integration.step_size.mode") == "adaptive"
        )
        if expected_prefix_steps is None and not adaptive_steps:
            errors.append(
                f"{prefix}.result_record: cannot derive validated prefix work"
            )
        elif expected_prefix_steps is not None \
                and run["accepted_steps"] != expected_prefix_steps:
            errors.append(
                f"{prefix}.result_record: accepted steps disagree with validated extent"
            )
        if run["status"] == "completed":
            if run["outcome"] != "completed" or not run["requested_horizon_completed"]:
                errors.append(f"{prefix}.result_record: completed run lacks completed outcome/horizon")
            if run["validated_extent"] != run["requested_extent"]:
                errors.append(f"{prefix}.result_record: completed run has partial validated extent")
            if run["first_failure"] is not None:
                errors.append(f"{prefix}.result_record: completed run carries first failure")
            if expected_accepted is not None \
                    and run["accepted_steps"] != expected_accepted:
                errors.append(
                    f"{prefix}.result_record: accepted steps do not match contract"
                )
            if expected_nn_calls is not None \
                    and run["nn_calls"] != expected_nn_calls:
                errors.append(
                    f"{prefix}.result_record: NN calls do not match contract"
                )
        elif run["outcome"] != cell["run"]["failure_category"]:
            errors.append(f"{prefix}.result_record: outcome/failure category mismatch")
        if run["status"] != "completed" and run["requested_horizon_completed"] is not False:
            errors.append(
                f"{prefix}.result_record: non-completed run claims completed horizon"
            )
        if run["status"] == "skipped" and any(
            run[name] != 0 for name in ("accepted_steps", "rejected_steps", "nn_calls")
        ):
            errors.append(f"{prefix}.result_record: skipped run must have zero work counts")
        if isinstance(run["status"], str) \
                and run["status"] not in {"completed", "skipped"}:
            if expected_accepted is not None \
                    and isinstance(run["accepted_steps"], int) \
                    and not isinstance(run["accepted_steps"], bool) \
                    and run["accepted_steps"] > expected_accepted:
                errors.append(
                    f"{prefix}.result_record: accepted steps exceed contract"
                )
            if expected_nn_calls is not None \
                    and isinstance(run["nn_calls"], int) \
                    and not isinstance(run["nn_calls"], bool) \
                    and run["nn_calls"] > expected_nn_calls:
                errors.append(f"{prefix}.result_record: NN calls exceed contract")
        if run["status"] != "completed" and not isinstance(run["first_failure"], dict):
            errors.append(f"{prefix}.result_record: terminal non-completion lacks first failure")
        elif run["status"] != "completed":
            errors.extend(_validate_failure(
                run["first_failure"], f"{prefix}.result_record.run.first_failure"
            ))
            if run["first_failure"].get("reason_code") != run["outcome"]:
                errors.append(
                    f"{prefix}.result_record: first failure reason disagrees with outcome"
                )
            if run["first_failure"].get("detail") != cell["run"]["failure_detail"]:
                errors.append(
                    f"{prefix}.result_record: first failure detail disagrees with cell"
                )

    property_raw = record["property"]
    property_value = property_raw if isinstance(property_raw, dict) else {}
    property_keys = {
        "status", "checker", "certificate_status", "certificate_semantics", "artifact"
    }
    property_errors = _exact_keys(
        property_value, property_keys, f"{prefix}.result_record.property"
    )
    errors.extend(property_errors)
    if not property_errors:
        if not isinstance(property_value["status"], str) or property_value["status"] not in {
            "passed", "failed", "not_applicable", "not_checked"
        } or not isinstance(property_value["certificate_status"], str) \
                or property_value["certificate_status"] not in {
            "passed", "failed", "not_applicable", "not_checked"
        }:
            errors.append(f"{prefix}.result_record.property: invalid status")
        checker_mode = _dotted(cell, "property_checker.mode")
        if run.get("status") == "completed" and checker_mode == "configured" and (
            not _mode_is(property_value["status"], "passed", "failed")
            or property_value["certificate_status"] == "not_checked"
            or not _nonempty(property_value["checker"])
            or not _nonempty(property_value["certificate_semantics"])
            or property_value["artifact"] is None
        ):
            errors.append(
                f"{prefix}.result_record: completed run lacks explicit "
                "property/certificate"
            )
        if run.get("status") == "completed" \
                and checker_mode == "not_applicable" and (
            property_value["status"] != "not_applicable"
            or property_value["checker"] is not None
            or property_value["certificate_status"] != "not_applicable"
            or property_value["certificate_semantics"] is not None
            or property_value["artifact"] is not None
        ):
            errors.append(
                f"{prefix}.result_record: not-applicable property checker "
                "has property evidence"
            )
        if run.get("status") == "skipped" and (
            property_value["status"] != "not_checked"
            or property_value["certificate_status"] != "not_checked"
        ):
            errors.append(
                f"{prefix}.result_record: skipped run cannot claim property evidence"
            )
        if property_value["artifact"] is not None:
            errors.extend(_validate_artifact(
                property_value["artifact"], root,
                f"{prefix}.result_record.property.artifact",
            ))
        planned_checker = _dotted(cell, "property_checker.identity")
        if checker_mode == "configured" \
                and property_value["checker"] is not None \
                and property_value["checker"] != planned_checker:
            errors.append(
                f"{prefix}.result_record: property checker disagrees with cell plan"
            )
        if checker_mode == "configured" \
                and property_value["certificate_semantics"] != _dotted(
                    cell, "property_checker.certificate_semantics"
                ):
            errors.append(
                f"{prefix}.result_record: certificate semantics disagree with "
                "cell plan"
            )

    eligibility_raw = record["eligibility"]
    eligibility = eligibility_raw if isinstance(eligibility_raw, dict) else {}
    eligibility_keys = {
        "mathematical_contract_known", "requested_horizon_completed",
        "certificate_semantics_passed", "finite_outputs",
        "numerical_soundness_class", "soundness_scope", "formal_claim_eligible",
        "performance_measurement_eligible", "cross_tool_ranking_eligible",
    }
    eligibility_errors = _exact_keys(
        eligibility, eligibility_keys, f"{prefix}.result_record.eligibility"
    )
    errors.extend(eligibility_errors)
    if not eligibility_errors:
        for name in eligibility_keys - {"numerical_soundness_class", "soundness_scope"}:
            if not isinstance(eligibility[name], bool):
                errors.append(f"{prefix}.result_record.eligibility.{name}: expected Boolean")
        if not isinstance(eligibility["numerical_soundness_class"], str) \
                or eligibility["numerical_soundness_class"] not in SOUNDNESS_CLASSES:
            errors.append(f"{prefix}.result_record.eligibility: invalid soundness class")
        if not isinstance(eligibility["soundness_scope"], str) \
                or eligibility["soundness_scope"] not in SOUNDNESS_SCOPES:
            errors.append(f"{prefix}.result_record.eligibility: invalid soundness scope")
        if eligibility["requested_horizon_completed"] != run.get(
            "requested_horizon_completed"
        ):
            errors.append(f"{prefix}.result_record: horizon eligibility disagrees with run")
        certificate_passed = property_value.get("certificate_status") == "passed"
        if eligibility["certificate_semantics_passed"] is not certificate_passed:
            errors.append(
                f"{prefix}.result_record: certificate eligibility disagrees with property"
            )
        if eligibility["mathematical_contract_known"] is not True:
            errors.append(
                f"{prefix}.result_record: bound resolved contract must be marked known"
            )
        if run.get("status") == "completed" \
                and eligibility["finite_outputs"] is not True:
            errors.append(
                f"{prefix}.result_record: completed run must have finite outputs"
            )
        if run.get("outcome") == "nonfinite" \
                and eligibility["finite_outputs"] is not False:
            errors.append(
                f"{prefix}.result_record: nonfinite outcome cannot claim finite outputs"
            )

    samples = record["samples"]
    if not isinstance(samples, list):
        errors.append(f"{prefix}.result_record.samples: expected an array")
        samples = []
    sample_keys: set[tuple[Any, Any]] = set()
    for index, sample in enumerate(samples):
        errors.extend(_validate_sample(
            sample, root, f"{prefix}.result_record.samples[{index}]"
        ))
        if isinstance(sample, dict):
            key = (sample.get("role"), sample.get("index"))
            if isinstance(key[0], str) and isinstance(key[1], int) \
                    and not isinstance(key[1], bool):
                if key in sample_keys:
                    errors.append(
                        f"{prefix}.result_record.samples: duplicate role/index {key!r}"
                    )
                sample_keys.add(key)
    plan = cell["measurement_plan"]
    cold_runs = plan.get("cold_runs") if isinstance(plan, Mapping) else None
    steady_runs = plan.get("steady_runs") if isinstance(plan, Mapping) else None
    if any(
        isinstance(count, bool) or not isinstance(count, int) or count < 0
        for count in (cold_runs, steady_runs)
    ):
        errors.append(f"{prefix}.result_record: invalid planned sample counts")
        expected_keys: set[tuple[str, int]] = set()
    else:
        expected_keys = {
            *(("cold", index) for index in range(cold_runs)),
            *(("steady", index) for index in range(steady_runs)),
        }
    unexpected_formal = sorted(
        key for key in sample_keys
        if key[0] in {"cold", "steady"} and key not in expected_keys
    )
    if unexpected_formal:
        errors.append(
            f"{prefix}.result_record.samples: formal samples outside plan "
            f"{unexpected_formal}"
        )
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            continue
        label = f"{prefix}.result_record.samples[{index}]"
        sample_extent = sample.get("validated_extent")
        if not _extent_is_prefix(sample_extent, run.get("requested_extent")):
            errors.append(f"{label}: validated extent is outside requested extent")
        sample_steps = (
            _accepted_steps_for_extent(contract, cell, sample_extent)
            if contract is not None else None
        )
        if sample_steps is not None and sample.get("accepted_steps") != sample_steps:
            errors.append(f"{label}.accepted_steps: disagrees with validated extent")
        if expected_nn_calls is not None and isinstance(sample.get("nn_calls"), int) \
                and not isinstance(sample.get("nn_calls"), bool) \
                and sample["nn_calls"] > expected_nn_calls:
            errors.append(f"{label}.nn_calls: exceeds cell plan")
        if sample.get("outcome") == "completed":
            if sample_extent != run.get("requested_extent"):
                errors.append(f"{label}: completed sample has partial extent")
            if expected_accepted is not None \
                    and sample.get("accepted_steps") != expected_accepted:
                errors.append(f"{label}.accepted_steps: disagrees with cell plan")
            if expected_nn_calls is not None \
                    and sample.get("nn_calls") != expected_nn_calls:
                errors.append(f"{label}.nn_calls: disagrees with cell plan")
        elif isinstance(sample.get("failure"), Mapping) \
                and sample["failure"].get("reason_code") != sample.get("outcome"):
            errors.append(f"{label}: failure reason disagrees with outcome")
    if run.get("status") == "completed":
        if sample_keys != expected_keys or len(samples) != len(expected_keys):
            errors.append(f"{prefix}.result_record: completed run lacks every planned sample")
        if any(
            not isinstance(sample, dict) or sample.get("outcome") != "completed"
            for sample in samples
        ):
            errors.append(f"{prefix}.result_record: completed run has non-completed sample")
        for index, sample in enumerate(samples):
            if not isinstance(sample, dict):
                continue
            key = (sample.get("role"), sample.get("index"))
            if not isinstance(key[0], str) or not isinstance(key[1], int) \
                    or isinstance(key[1], bool) or key not in expected_keys:
                continue
            label = f"{prefix}.result_record.samples[{index}]"
            if sample.get("validated_extent") != run.get("requested_extent"):
                errors.append(f"{label}: extent disagrees with completed run")
            for name in ("accepted_steps", "rejected_steps", "nn_calls"):
                if sample.get(name) != run.get(name):
                    errors.append(f"{label}.{name}: disagrees with run summary")
    elif run.get("status") == "skipped":
        if samples:
            errors.append(f"{prefix}.result_record: skipped run must have no samples")
    elif samples:
        matching_failure = any(
            isinstance(sample, Mapping)
            and sample.get("outcome") == run.get("outcome")
            and sample.get("validated_extent") == run.get("validated_extent")
            and sample.get("accepted_steps") == run.get("accepted_steps")
            and sample.get("rejected_steps") == run.get("rejected_steps")
            and sample.get("nn_calls") == run.get("nn_calls")
            and sample.get("failure") == run.get("first_failure")
            for sample in samples
        )
        if not matching_failure:
            errors.append(
                f"{prefix}.result_record: terminal failure has no matching sample"
            )

    widths = record["widths"]
    errors.extend(_validate_widths(
        widths, root, f"{prefix}.result_record.widths"
    ))
    if isinstance(widths, dict):
        completed = run.get("status") == "completed"
        if completed and widths.get("status") != "complete":
            errors.append(
                f"{prefix}.result_record: completed run lacks complete widths"
            )
        if run.get("status") == "skipped" and (
            widths.get("status") != "unavailable"
            or any(
                not isinstance(widths.get(name), dict)
                or widths[name].get("status") != "unavailable"
                for name in ("endpoint", "last_segment_tube", "full_horizon_tube")
            )
        ):
            errors.append(
                f"{prefix}.result_record: skipped run must have unavailable widths"
            )
        if widths.get("coordinate_order") != contract_variables:
            errors.append(
                f"{prefix}.result_record.widths: coordinate order does not match contract"
            )
        common_prefix = widths.get("common_prefix")
        if common_prefix != run.get("validated_extent"):
            errors.append(
                f"{prefix}.result_record.widths: common prefix disagrees with validated extent"
            )
        if completed and (not isinstance(common_prefix, dict) or not _is_finite_number(
            common_prefix.get("value"), positive=True
        )):
            errors.append(
                f"{prefix}.result_record.widths: completed common prefix must be positive"
            )
        validated = run.get("validated_extent")
        if isinstance(validated, dict) and set(validated) == {"kind", "value"} \
                and contract is not None:
            kind, end = validated["kind"], validated["value"]
            expected_domains = {
                "endpoint": {"kind": kind, "start": end, "end": end},
                "last_segment_tube": {
                    "kind": kind,
                    "start": _last_segment_start(contract, cell, end),
                    "end": end,
                },
                "full_horizon_tube": {"kind": kind, "start": 0, "end": end},
            }
            for name, expected_domain in expected_domains.items():
                view = widths.get(name)
                domain = view.get("domain") if isinstance(view, dict) else None
                adaptive_last_segment = (
                    name == "last_segment_tube"
                    and _dotted(cell, "numerics.integration.step_size.mode")
                    == "adaptive"
                )
                if adaptive_last_segment:
                    domain_matches = (
                        isinstance(domain, Mapping)
                        and set(domain) == {"kind", "start", "end"}
                        and domain.get("kind") == kind
                        and _is_finite_number(domain.get("start"))
                        and _is_finite_number(domain.get("end"))
                        and 0 <= domain["start"] <= domain["end"]
                        and math.isclose(
                            float(domain["end"]), float(end),
                            rel_tol=1e-12, abs_tol=1e-12,
                        )
                    )
                else:
                    domain_matches = _same_domain(domain, expected_domain)
                if not domain_matches:
                    scope = "completed run" if completed else "validated prefix"
                    errors.append(
                        f"{prefix}.result_record.widths.{name}: domain disagrees "
                        f"with {scope}"
                    )

    artifacts = record["artifacts"]
    if not isinstance(artifacts, list):
        errors.append(f"{prefix}.result_record.artifacts: expected an array")
        artifacts = []
    roles: set[str] = set()
    for index, artifact in enumerate(artifacts):
        errors.extend(_validate_artifact(
            artifact, root, f"{prefix}.result_record.artifacts[{index}]",
            role_required=True,
        ))
        if isinstance(artifact, dict) and isinstance(artifact.get("role"), str):
            roles.add(artifact["role"])
    if not {"command", "run_log", "result"} <= roles:
        errors.append(f"{prefix}.result_record: command/run_log/result artifacts are required")

    if isinstance(eligibility, dict) \
            and eligibility.get("performance_measurement_eligible") is True:
        performance_prerequisites = (
            run.get("status") == "completed",
            run.get("requested_horizon_completed") is True,
            eligibility.get("finite_outputs") is True,
            sample_keys == expected_keys if run.get("status") == "completed" else False,
            all(
                isinstance(sample, dict) and sample.get("outcome") == "completed"
                for sample in samples
            ),
        )
        if not all(performance_prerequisites):
            errors.append(
                f"{prefix}.result_record: performance eligibility lacks prerequisites"
            )
    if isinstance(eligibility, dict) \
            and eligibility.get("cross_tool_ranking_eligible") is True:
        soundness = eligibility.get("numerical_soundness_class")
        scope = eligibility.get("soundness_scope")
        prerequisites = (
            eligibility.get("mathematical_contract_known") is True,
            eligibility.get("requested_horizon_completed") is True,
            eligibility.get("certificate_semantics_passed") is True,
            eligibility.get("finite_outputs") is True,
            eligibility.get("performance_measurement_eligible") is True,
            _dotted(cell, "measurement_plan.steady_runs")
            == _dotted(cell, "measurement_plan.target_steady_runs"),
            run.get("status") == "completed",
            property_value.get("certificate_status") == "passed",
            isinstance(soundness, str) and soundness not in {
                "empirically sampled only", "unknown",
                "unsound/ineligible on a demonstrated counterexample",
            },
            isinstance(scope, str)
            and scope in {"fixed workload", "multi-step lane", "native build"},
            isinstance(record["widths"], dict)
            and record["widths"].get("status") == "complete",
        )
        if not all(prerequisites):
            errors.append(f"{prefix}.result_record: ranking eligibility lacks prerequisites")
    if isinstance(eligibility, dict) \
            and eligibility.get("formal_claim_eligible") is True:
        soundness = eligibility.get("numerical_soundness_class")
        scope = eligibility.get("soundness_scope")
        formal_prerequisites = (
            eligibility.get("mathematical_contract_known") is True,
            eligibility.get("requested_horizon_completed") is True,
            eligibility.get("certificate_semantics_passed") is True,
            eligibility.get("finite_outputs") is True,
            isinstance(soundness, str) and soundness not in {
                "empirically sampled only", "unknown",
                "unsound/ineligible on a demonstrated counterexample",
            },
            isinstance(scope, str)
            and scope in {"fixed workload", "multi-step lane", "native build"},
            run.get("status") == "completed",
            property_value.get("certificate_status") == "passed",
        )
        if not all(formal_prerequisites):
            errors.append(f"{prefix}.result_record: formal eligibility lacks prerequisites")
    return errors


def _execution_plan_reasons(
    cell: Mapping[str, Any],
    profile: str | None,
    contract: Mapping[str, Any] | None = None,
) -> list[str]:
    reasons: list[str] = []
    if _dotted(cell, "support.status") != "supported":
        reasons.append("support_not_supported")
    blockers = _dotted(cell, "support.blockers")
    if blockers != []:
        reasons.append("support_blocked")
    argv = _dotted(cell, "command.argv")
    if not isinstance(argv, list) or not argv \
            or not all(isinstance(item, str) and item for item in argv):
        reasons.append("command_argv_missing")
    cwd = _dotted(cell, "command.cwd")
    if not isinstance(cwd, str) or not cwd.strip():
        reasons.append("command_cwd_missing")
    source = cell.get("source_identity")
    if not isinstance(source, Mapping) or not all(
        isinstance(source.get(key), str) and source[key].strip()
        for key in ("kind", "locator")
    ):
        reasons.append("source_identity_missing")
    if not isinstance(source, Mapping) or not any((
        isinstance(source.get("revision"), str) and source["revision"].strip(),
        _is_sha256(source.get("sha256")),
    )):
        reasons.append("source_revision_or_sha_missing")
    binary = cell.get("binary_identity")
    if not isinstance(binary, Mapping) or not (
        isinstance(binary.get("path"), str) and binary["path"].strip()
        and _is_sha256(binary.get("sha256"))
    ):
        reasons.append("binary_identity_missing")
    runtime = cell.get("runtime")
    timeout = runtime.get("timeout_s") if isinstance(runtime, Mapping) else None
    if not _is_finite_number(timeout, positive=True):
        reasons.append("runtime_timeout_missing")
    if not isinstance(runtime, Mapping) or not _nonempty(runtime.get("hardware")) \
            or not _nonempty(runtime.get("resource_limits")):
        reasons.append("runtime_budget_missing")
    if not isinstance(runtime, Mapping) or not _nonempty(runtime.get("gpu")):
        reasons.append("runtime_gpu_missing")
    threads = runtime.get("cpu_threads") if isinstance(runtime, Mapping) else None
    if isinstance(threads, bool) or not isinstance(threads, int) or threads <= 0:
        reasons.append("runtime_cpu_threads_missing")
    arithmetic = cell.get("arithmetic")
    if not isinstance(arithmetic, Mapping):
        reasons.append("arithmetic_missing")
    else:
        for name, value in arithmetic.items():
            if not _nonempty(value):
                reasons.append(f"arithmetic_{name}_missing")
    integration = _dotted(cell, "numerics.integration")
    remainder = _dotted(cell, "numerics.remainder")
    if profile == "full_execution_contract_v1":
        if not _mode_is(
            _dotted(integration, "step_size.mode"), "fixed", "adaptive"
        ):
            reasons.append("numerics_step_size_unresolved")
        elif _dotted(integration, "step_size.mode") == "adaptive":
            reasons.append("adaptive_step_policy_not_executable")
        for name in ("solution_order", "point_order", "validation_order"):
            if not _mode_is(
                _dotted(integration, f"{name}.mode"),
                "configured", "not_applicable",
            ):
                reasons.append(f"numerics_{name}_unresolved")
        for name in ("cutoff", "cap", "symbolic_queue"):
            if not _mode_is(
                _dotted(remainder, f"{name}.mode"),
                "configured", "not_applicable",
            ):
                reasons.append(f"remainder_{name}_unresolved")
    elif profile == "discrete_execution_contract_v1":
        for name in (
            "step_size", "solution_order", "point_order", "validation_order",
        ):
            if _dotted(integration, f"{name}.mode") != "not_applicable":
                reasons.append(f"numerics_{name}_not_marked_not_applicable")
        for name in ("cutoff", "cap", "symbolic_queue"):
            if _dotted(remainder, f"{name}.mode") != "not_applicable":
                reasons.append(f"remainder_{name}_not_marked_not_applicable")
    else:
        reasons.append("numerics_profile_unknown")
    if not _nonempty(_dotted(integration, "semantics")):
        reasons.append("integration_semantics_missing")
    if not _nonempty(_dotted(remainder, "semantics")):
        reasons.append("remainder_semantics_missing")
    controller_execution = cell.get("controller_execution")
    scheduled_updates = _dotted(controller_execution, "scheduled_updates")
    nn_calls_mode = _dotted(controller_execution, "nn_calls.mode")
    if not _is_non_negative_int(scheduled_updates):
        reasons.append("controller_scheduled_updates_missing")
    if not _mode_is(nn_calls_mode, "exact", "adaptive", "not_applicable"):
        reasons.append("controller_nn_calls_missing")
    elif nn_calls_mode == "adaptive":
        reasons.append("adaptive_nn_call_policy_not_executable")
    elif nn_calls_mode == "not_applicable":
        reasons.append("controller_nn_calls_not_applicable")
    if not _nonempty(_dotted(controller_execution, "nn_call_semantics")):
        reasons.append("controller_nn_call_semantics_missing")
    expected_updates = (
        _dotted(contract, "fields.controller_update.scheduled_updates")
        if isinstance(contract, Mapping) else None
    )
    if _is_non_negative_int(expected_updates) \
            and scheduled_updates != expected_updates:
        reasons.append("controller_scheduled_updates_contract_mismatch")
    property_checker = cell.get("property_checker")
    checker_mode = _dotted(property_checker, "mode")
    if not _mode_is(checker_mode, "configured", "not_applicable"):
        reasons.append("property_checker_mode_unresolved")
    elif checker_mode == "not_applicable":
        reasons.append("property_checker_not_applicable")
    else:
        for name in (
            "identity", "semantics", "certificate_semantics",
            "early_stop_policy",
        ):
            if not _nonempty(_dotted(property_checker, name)):
                reasons.append(f"property_checker_{name}_missing")
        if _dotted(property_checker, "early_stop_policy") != "never":
            reasons.append("property_early_stop_not_executable")
    return reasons


def _validate_cell_types(
    cell: Mapping[str, Any], prefix: str
) -> list[str]:
    errors: list[str] = []
    blockers = cell["support"]["blockers"]
    if not isinstance(blockers, list) or not all(
        isinstance(item, str) and item.strip() for item in blockers
    ):
        errors.append(f"{prefix}.support.blockers: expected non-empty strings")
    argv = cell["command"]["argv"]
    if argv is not None and (
        not isinstance(argv, list) or not argv
        or not all(isinstance(item, str) and item for item in argv)
    ):
        errors.append(f"{prefix}.command.argv: expected null or non-empty string array")
    cwd = cell["command"]["cwd"]
    if cwd is not None and (not isinstance(cwd, str) or not cwd.strip()):
        errors.append(f"{prefix}.command.cwd: expected null or non-empty string")
    for group in ("source_identity", "binary_identity", "arithmetic"):
        for name, value in cell[group].items():
            if value is not None and not _nonempty(value):
                errors.append(f"{prefix}.{group}.{name}: empty value")
    for group in ("source_identity", "binary_identity"):
        sha = cell[group].get("sha256")
        if sha is not None and not _is_sha256(sha):
            errors.append(f"{prefix}.{group}.sha256: invalid SHA-256")
    integration = cell["numerics"]["integration"]
    errors.extend(_tagged_value_errors(
        integration["step_size"], f"{prefix}.numerics.integration.step_size",
        configured_mode="fixed", positive=True, allow_adaptive=True,
    ))
    for name in ("solution_order", "point_order", "validation_order"):
        errors.extend(_tagged_value_errors(
            integration[name], f"{prefix}.numerics.integration.{name}",
            configured_mode="configured", integer=True, positive=True,
        ))
    integration_semantics = integration["semantics"]
    if integration_semantics is not None and not (
        isinstance(integration_semantics, str) and integration_semantics.strip()
    ):
        errors.append(f"{prefix}.numerics.integration.semantics: empty value")
    remainder = cell["numerics"]["remainder"]
    for name in ("cutoff", "cap"):
        errors.extend(_tagged_value_errors(
            remainder[name], f"{prefix}.numerics.remainder.{name}",
            configured_mode="configured",
        ))
    errors.extend(_tagged_value_errors(
        remainder["symbolic_queue"],
        f"{prefix}.numerics.remainder.symbolic_queue",
        configured_mode="configured", integer=True,
    ))
    if remainder["semantics"] is not None and not (
        isinstance(remainder["semantics"], str)
        and remainder["semantics"].strip()
    ):
        errors.append(f"{prefix}.numerics.remainder.semantics: empty value")
    controller_execution = cell["controller_execution"]
    scheduled_updates = controller_execution["scheduled_updates"]
    if scheduled_updates is not None and not _is_non_negative_int(
        scheduled_updates
    ):
        errors.append(
            f"{prefix}.controller_execution.scheduled_updates: expected null or a "
            "non-negative integer"
        )
    errors.extend(_tagged_value_errors(
        controller_execution["nn_calls"],
        f"{prefix}.controller_execution.nn_calls",
        configured_mode="exact", integer=True, allow_adaptive=True,
    ))
    semantics = controller_execution["nn_call_semantics"]
    if semantics is not None and not (
        isinstance(semantics, str) and semantics.strip()
    ):
        errors.append(f"{prefix}.controller_execution.nn_call_semantics: empty value")
    property_checker = cell["property_checker"]
    checker_mode = property_checker["mode"]
    if not _mode_is(
        checker_mode, "unresolved", "configured", "not_applicable"
    ):
        errors.append(
            f"{prefix}.property_checker.mode: invalid applicability tag "
            f"{checker_mode!r}"
        )
    for name in (
        "identity", "semantics", "certificate_semantics",
        "early_stop_policy",
    ):
        value = property_checker[name]
        if checker_mode == "configured":
            if not (isinstance(value, str) and value.strip()):
                errors.append(f"{prefix}.property_checker.{name}: expected text")
        elif value is not None:
            errors.append(
                f"{prefix}.property_checker.{name}: must be null when mode is "
                f"{checker_mode!r}"
            )
    policy = property_checker["early_stop_policy"]
    if checker_mode == "configured" and not _mode_is(
        policy, "never", "on_pass", "on_fail", "on_decisive"
    ):
        errors.append(
            f"{prefix}.property_checker.early_stop_policy: invalid policy"
        )
    threads = cell["runtime"]["cpu_threads"]
    if threads is not None and (
        isinstance(threads, bool) or not isinstance(threads, int) or threads <= 0
    ):
        errors.append(f"{prefix}.runtime.cpu_threads: expected a positive integer")
    timeout = cell["runtime"]["timeout_s"]
    if timeout is not None and not _is_finite_number(timeout, positive=True):
        errors.append(f"{prefix}.runtime.timeout_s: expected a finite positive number")
    plan = cell["measurement_plan"]
    fixed_plan = {
        "cold_runs": 1,
        "target_steady_runs": 5,
        "fresh_process_per_run": True,
        "timing_boundary_version": "total_configuration_v1",
    }
    if any(plan.get(name) != value for name, value in fixed_plan.items()):
        errors.append(
            f"{prefix}.measurement_plan: expected one cold run, a five-steady "
            "target, fresh processes, and the frozen timing boundary"
        )
    steady_runs = plan.get("steady_runs")
    if isinstance(steady_runs, bool) or not isinstance(steady_runs, int) \
            or not 1 <= steady_runs <= 5:
        errors.append(
            f"{prefix}.measurement_plan.steady_runs: expected an integer from 1 to 5"
        )
    shortfall = plan.get("shortfall_reason")
    if isinstance(steady_runs, int) and not isinstance(steady_runs, bool):
        if steady_runs < 5 and not (
            isinstance(shortfall, str) and shortfall.strip()
        ):
            errors.append(
                f"{prefix}.measurement_plan.shortfall_reason: required below target"
            )
        if steady_runs == 5 and shortfall is not None:
            errors.append(
                f"{prefix}.measurement_plan.shortfall_reason: must be null at target"
            )
    link = cell["result_record"]
    values = (link["schema_version"], link["path"], link["sha256"])
    if any(value is None for value in values) and any(value is not None for value in values):
        errors.append(f"{prefix}.result_record: partial pointer")
    if all(value is not None for value in values):
        if link["schema_version"] != RESULT_SCHEMA:
            errors.append(f"{prefix}.result_record: wrong schema_version")
        if not _is_sha256(link["sha256"]):
            errors.append(f"{prefix}.result_record: invalid SHA-256")
    return errors


def validate_matrix(
    manifest: Mapping[str, Any], matrix: Mapping[str, Any], *, root: Path = ROOT
) -> list[str]:
    errors: list[str] = []
    manifest_matrix = manifest.get("execution_matrix", {})
    if matrix.get("schema_version") != MATRIX_SCHEMA \
            or manifest_matrix.get("schema_version") != MATRIX_SCHEMA:
        errors.append("matrix schema_version does not match the v3 contract")
    if matrix.get("source_manifest") != "benchmarks/archcomp26/manifest.json":
        errors.append("matrix source_manifest is not the canonical manifest")
    result_contract = matrix.get("result_record_contract")
    contract_errors = _exact_keys(
        result_contract, {"path", "schema_version", "sha256"},
        "result_record_contract",
    )
    errors.extend(contract_errors)
    if not contract_errors:
        if result_contract["schema_version"] != RESULT_SCHEMA:
            errors.append("result_record_contract has the wrong schema_version")
        _, bound_errors = _bound_file(
            root,
            result_contract["path"],
            result_contract["sha256"],
            "result_record_contract",
        )
        errors.extend(bound_errors)
    errors.extend(_validate_contracts(manifest, root))
    instance_ids = [row["id"] for row in manifest.get("instances", [])]
    instances_by_id = {
        row["id"]: row for row in manifest.get("instances", [])
        if isinstance(row, dict) and "id" in row
    }
    methods = list(manifest.get("methods", []))
    if list(matrix.get("methods", [])) != methods:
        errors.append("matrix methods do not match manifest methods")
    if list(matrix.get("cells", {})) != instance_ids:
        errors.append("matrix instance order does not match manifest")
    defaults = matrix.get("cell_defaults")
    required = matrix.get("required_cell_fields")
    expected_required = list(CELL_DEFAULT_SHAPE)
    if not isinstance(required, list) or required != expected_required:
        errors.append("required_cell_fields do not match the v3 cell contract")
        return errors
    default_shape_errors = _shape_errors(
        defaults, CELL_DEFAULT_SHAPE, "cell_defaults"
    )
    if default_shape_errors:
        errors.extend(default_shape_errors)
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
            ):
                is_string_enum = dotted != "run.failure_category"
                if (is_string_enum and not isinstance(value, str)) or (
                    not is_string_enum and value is not None
                    and not isinstance(value, str)
                ) or value not in enums.get(dotted, []):
                    errors.append(f"{prefix}: invalid {dotted}={value!r}")
            errors.extend(_validate_cell_types(resolved, prefix))
            status = resolved["run"]["status"]
            category = resolved["run"]["failure_category"]
            detail = resolved["run"]["failure_detail"]
            if status == "not_started" and (category is not None or detail is not None):
                errors.append(f"{prefix}: not_started cell carries failure data")
            if isinstance(status, str) \
                    and status in {"failed", "timeout", "interrupted", "skipped"} and (
                category is None or not isinstance(detail, str) or not detail.strip()
            ):
                errors.append(f"{prefix}: terminal failure lacks category/detail")
            if status == "completed" and category is not None:
                errors.append(f"{prefix}: completed cell carries a failure category")
            if status == "completed" and detail is not None:
                errors.append(f"{prefix}: completed cell carries failure detail")
            if status == "timeout" and category != "timeout":
                errors.append(f"{prefix}: timeout status must use timeout category")
            if category == "timeout" and status != "timeout":
                errors.append(f"{prefix}: timeout category must use timeout status")
            if status == "interrupted" and category != "incomplete_unknown":
                errors.append(
                    f"{prefix}: interrupted status must use incomplete_unknown category"
                )
            if resolved["support"]["status"] == "unsupported" \
                    and status == "completed":
                errors.append(f"{prefix}: unsupported cell cannot be completed")
            if status == "skipped":
                if resolved["support"]["status"] != "unsupported" \
                        or not resolved["support"]["blockers"]:
                    errors.append(
                        f"{prefix}: skipped cell requires unsupported status and blockers"
                    )
            elif isinstance(status, str) and status in TERMINAL_STATUSES:
                instance_row = instances_by_id[instance]
                profile = _dotted(
                    instance_row, "contract.unresolved_field_profile"
                )
                contract_record = (
                    _resolved_contract_record(instance_row, root)
                    if instance_row["contract"].get("status") == "resolved"
                    else None
                )
                plan_reasons = _execution_plan_reasons(
                    resolved, profile, contract_record
                )
                if plan_reasons:
                    errors.append(
                        f"{prefix}: terminal cell lacks executable plan "
                        f"({', '.join(plan_reasons)})"
                    )
            link = resolved["result_record"]
            has_link = all(link[name] is not None for name in (
                "schema_version", "path", "sha256"
            ))
            if status == "not_started" and has_link:
                errors.append(f"{prefix}: not_started cell carries a result record")
            if isinstance(status, str) and status in TERMINAL_STATUSES and not has_link:
                errors.append(f"{prefix}: terminal cell lacks a result record")
            if isinstance(status, str) and status in TERMINAL_STATUSES \
                    and instances_by_id[instance]["contract"].get("status") != "resolved":
                errors.append(f"{prefix}: terminal cell has unresolved instance contract")
            if has_link:
                record_path, bound_errors = _bound_file(
                    root, link["path"], link["sha256"], f"{prefix}.result_record"
                )
                errors.extend(bound_errors)
                if record_path is not None:
                    try:
                        record = _load(record_path)
                    except (OSError, ValueError, json.JSONDecodeError) as error:
                        errors.append(f"{prefix}.result_record: cannot load: {error}")
                    else:
                        errors.extend(_validate_result_record(
                            record, root, prefix, instances_by_id[instance], method,
                            resolved,
                        ))
    statuses = [
        _dotted(resolve_cell(matrix, instance, method), "run.status")
        for instance in instance_ids
        for method in methods
        if instance in matrix.get("cells", {})
        and method in matrix["cells"][instance]
    ]
    if statuses:
        if all(status == "not_started" for status in statuses):
            expected_status = "not_started"
        elif any(status == "running" for status in statuses):
            expected_status = "running"
        elif all(
            isinstance(status, str) and status in TERMINAL_STATUSES
            for status in statuses
        ):
            expected_status = "terminal"
        else:
            expected_status = "in_progress"
        if matrix.get("status") != expected_status:
            errors.append(
                f"matrix status {matrix.get('status')!r} does not match {expected_status!r}"
            )
    return errors


def preflight_reasons(
    manifest: Mapping[str, Any],
    matrix: Mapping[str, Any],
    instance: str,
    method: str,
    *,
    root: Path = ROOT,
) -> list[str]:
    reasons = validate_matrix(manifest, matrix, root=root)
    by_id = {row["id"]: row for row in manifest.get("instances", [])}
    if instance not in by_id:
        return [*reasons, "unknown_instance"]
    if method not in manifest.get("methods", []):
        return [*reasons, "unknown_method"]
    if manifest["execution_policy"]["experiments_paused"] is not False:
        reasons.append("experiments_paused")
    contract = by_id[instance]["contract"]
    if contract["status"] != "resolved":
        reasons.append("contract_unresolved")
    elif any(
        error.startswith(f"{instance}:")
        or error.startswith(f"{instance}.contract")
        for error in _validate_contracts(manifest, root)
    ):
        reasons.append("contract_invalid")
    cell = resolve_cell(matrix, instance, method)
    contract_record = (
        _resolved_contract_record(by_id[instance], root)
        if contract["status"] == "resolved" else None
    )
    reasons.extend(_execution_plan_reasons(
        cell, contract.get("unresolved_field_profile"), contract_record
    ))
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
