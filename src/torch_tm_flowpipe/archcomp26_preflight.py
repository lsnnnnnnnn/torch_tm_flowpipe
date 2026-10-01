"""Fail-closed, read-only checks for the ARCH-COMP26 execution matrix."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import struct
from typing import Any, Mapping
import xml.etree.ElementTree as ET
import zlib


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "benchmarks/archcomp26/manifest.json"
MATRIX_SCHEMA = "archcomp26-execution-matrix-v6"
RESULT_SCHEMA = "archcomp26-cell-result-v5"
INSTANCE_SCHEMA = "archcomp26-instance-contract-v1"
OFFICIAL_ASSETS_SCHEMA = "archcomp26-official-assets-v1"
OFFICIAL_ASSETS_AUDIT_SCHEMA = "archcomp26-official-assets-audit-v1"
PARTITION_LEDGER_SCHEMA = "archcomp26-partition-ledger-v1"
PRELAUNCH_AUDIT_SCHEMA = "archcomp26-prelaunch-audit-v1"
NATIVE_LAUNCH_SCHEMA = "archcomp26-native-job-launch-v1"
NATIVE_TERMINAL_SCHEMA = "archcomp26-native-job-terminal-v1"
PROCESS_SCAN_SCHEMA = "archcomp26-process-scan-v1"
ACTIVE_RUN_SCHEMA = "archcomp26-active-run-v2"
CAMPAIGN_LOCK_SCHEMA = "archcomp26-campaign-lock-v2"
ATTEMPT_LEDGER_SCHEMA = "archcomp26-attempt-ledger-v2"
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
EXPECTED_INSTANCE_IDS = (
    "acc-safe-distance",
    "airplane-continuous",
    "airplane-discrete",
    "attitude-control-avoid",
    "balancing-reach",
    "docking-constraint",
    "double-pendulum-less-robust",
    "double-pendulum-more-robust",
    "nav-standard",
    "nav-robust",
    "quad-reach",
    "single-pendulum-reach",
    "tora-remain",
    "tora-reach-sigmoid",
    "tora-reach-tanh",
    "unicycle-reach",
)
EXPECTED_PROFILE_BY_INSTANCE = {
    instance_id: (
        "discrete_execution_contract_v1"
        if instance_id == "airplane-discrete"
        else "full_execution_contract_v1"
    )
    for instance_id in EXPECTED_INSTANCE_IDS
}
EXPECTED_METHODS = ("pytorch_gpu", "huan", "xiangru", "flowstar_native")
VIEWABLE_PLOT_ROLES = {
    "plot_png": ".png",
    "plot_pdf": ".pdf",
}
PLOT_ARTIFACT_ROLES = {
    **VIEWABLE_PLOT_ROLES,
    "plot_svg": ".svg",
}
SERVER_RESEARCH_ROOT = "/srv/local/shengenli/flowstar_acceleration_20260921T153643Z"
ORIGINAL_NATIVE_QUAD_IDENTITY = {
    "job_id": "native-quad-matched-20260929-full1000-v1",
    "run_directory": (
        SERVER_RESEARCH_ROOT
        + "/runs/native_quad_matched_20260929/initial_affine_cover_variant/"
        "full1000_v1"
    ),
    "watch_directory": (
        SERVER_RESEARCH_ROOT
        + "/runs/native_quad_matched_20260929/initial_affine_cover_variant/"
        "full1000_v1_watch"
    ),
    "command_sha256": (
        "db3f45544bf0cd4da63f90cf3973a7b76347f585809121783b87d709d818c62e"
    ),
    "terminal_status": "timeout",
    "result_sha256": (
        "036943cd0b2a0e040017e30c0ac82d0b41f6eb645d924d2294dfe70e2bde97f2"
    ),
}
# The handoff records that the prepared 24-hour replacement was never launched.
# Any newly discovered or deliberately launched replacement must first be added
# here as an independently reviewed identity instead of being self-declared by
# a prelaunch receipt.
EXPECTED_NATIVE_REPLACEMENTS: tuple[Mapping[str, str], ...] = ()
RESOURCE_LIMIT_FIELDS = {
    "exclusive_host", "exclusive_gpu_device",
    "max_host_memory_bytes", "max_device_memory_bytes",
}
TERMINAL_STATUSES = {
    "completed", "failed", "timeout", "interrupted", "early_stopped", "skipped"
}
CANONICAL_OUTCOMES = {
    "completed",
    "validation_rejected",
    "nonfinite",
    "timeout",
    "process_error",
    "compile_error",
    "missing_dependency",
    "environment_error",
    "resource_guard",
    "out_of_memory",
    "property_early_stop_pass",
    "property_early_stop_fail",
    "trajectory_sanity_failed",
    "analytic_containment_failed",
    "schema_invalid",
    "incomplete_unknown",
}
TIMING_FIELDS = {
    "process_total", "driver_total", "compile", "controller_nn",
    "solver_core", "validation", "observer", "output", "plot_report",
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
SHARED_PROFILE_FIELDS = {
    "full_execution_contract_v1": (
        "dynamics.source", "dynamics.sha256", "dynamics.equations",
        "variable_order", "controller.source", "controller.sha256",
        "controller.input_output_order", "initial_set.source",
        "initial_set.sha256", "initial_set.partitions",
        "initial_set.boxes_sha256", "disturbance", "integration.horizon",
        "controller_update.period", "controller_update.scheduled_updates",
        "controller_update.schedule_points",
        "controller_update.schedule_semantics", "property.formula",
        "property.time_semantics", "property.pass_condition",
        "width_comparison.coordinate_units",
        "width_comparison.sample_points",
        "width_comparison.aggregation_semantics",
    ),
    "discrete_execution_contract_v1": (
        "transition.source", "transition.sha256", "transition.state_update",
        "transition.control_application_order", "variable_order",
        "controller.source", "controller.sha256",
        "controller.input_output_order", "initial_set.source",
        "initial_set.sha256", "initial_set.partitions",
        "initial_set.boxes_sha256", "disturbance", "discrete.index_set",
        "discrete.transition_count", "discrete.sample_period",
        "controller_update.period_steps", "controller_update.scheduled_updates",
        "controller_update.schedule_points",
        "controller_update.schedule_semantics", "property.formula",
        "property.step_semantics", "property.pass_condition",
        "width_comparison.coordinate_units",
        "width_comparison.sample_points",
        "width_comparison.aggregation_semantics",
    ),
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
CAMPAIGN_SHAPE = {
    "schema_version": None,
    "campaign_id": None,
    "host_identity": None,
    "hardware_identity": None,
    "cpu_thread_budget": None,
    "gpu_device_budget": None,
    "timeout_s": None,
    "resource_limits": None,
    "prelaunch_audit": {"path": None, "sha256": None},
    "active_run_receipt": {"path": None, "sha256": None},
    "launch_guard": {
        "schema_version": None, "protocol": None, "lock_path": None,
        "hold_scope": None, "wrapper_module": None, "wrapper_path": None,
        "wrapper_sha256": None,
        "lock_identity": {
            "device": None, "inode": None, "owner_uid": None,
            "mode": None, "nlink": None,
        },
    },
    "timing_boundary": {
        "version": None, "start_event": None, "stop_event": None,
        "phase_fields": None,
    },
    "rotation": {
        "policy": None, "steady_rounds": None,
        "schedule_artifact": {"path": None, "sha256": None},
    },
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


CAMPAIGN_CONFIGURATION_FIELDS = (
    "schema_version", "campaign_id", "host_identity", "hardware_identity",
    "cpu_thread_budget", "gpu_device_budget", "timeout_s", "resource_limits",
    "launch_guard", "timing_boundary", "rotation",
)


def campaign_configuration_sha256(campaign: Mapping[str, Any]) -> str:
    """Hash only frozen campaign configuration, never mutable run pointers."""
    return _canonical_sha256({
        name: campaign.get(name) for name in CAMPAIGN_CONFIGURATION_FIELDS
    })


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None


def _parse_utc(value: Any) -> datetime | None:
    if not isinstance(value, str) or re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z", value
    ) is None:
        return None
    try:
        return datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        return None


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


def _outcome_matches_category(outcome: Any, category: Any) -> bool:
    if category == "property_early_stop":
        return outcome in {
            "property_early_stop_pass", "property_early_stop_fail"
        }
    return outcome == category


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
    variables = _dotted(fields, "variable_order")
    if isinstance(partitions, list) and isinstance(variables, list):
        for index, box in enumerate(partitions):
            label = f"{instance}: initial_set.partitions[{index}]"
            if not isinstance(box, dict) or list(box) != variables:
                errors.append(f"{label} must contain variable_order exactly")
                continue
            for variable, bounds in box.items():
                if not isinstance(bounds, list) or len(bounds) != 2 or not all(
                    not isinstance(bound, bool)
                    and isinstance(bound, (int, float))
                    and math.isfinite(bound)
                    for bound in bounds
                ) or bounds[0] > bounds[1]:
                    errors.append(
                        f"{label}.{variable} must be a finite [lo, hi] interval"
                    )
    disturbance = _dotted(fields, "disturbance")
    disturbance_errors = _exact_keys(
        disturbance, {"mode", "bounds"}, f"{instance}: disturbance"
    )
    errors.extend(disturbance_errors)
    if not disturbance_errors:
        mode = disturbance["mode"]
        bounds = disturbance["bounds"]
        if not _mode_is(mode, "none", "box"):
            errors.append(f"{instance}: disturbance.mode must be none or box")
        if not isinstance(bounds, dict):
            errors.append(f"{instance}: disturbance.bounds must be an object")
        elif mode == "none" and bounds:
            errors.append(f"{instance}: disturbance none mode must have empty bounds")
        elif mode == "box":
            if not isinstance(variables, list) or list(bounds) != variables:
                errors.append(
                    f"{instance}: disturbance box must contain variable_order exactly"
                )
            else:
                for variable, interval in bounds.items():
                    if not isinstance(interval, list) or len(interval) != 2 \
                            or not all(
                                not isinstance(bound, bool)
                                and isinstance(bound, (int, float))
                                and math.isfinite(bound)
                                for bound in interval
                            ) or interval[0] > interval[1]:
                        errors.append(
                            f"{instance}: disturbance.bounds.{variable} is invalid"
                        )
    if profile == "full_execution_contract_v1":
        equations = _dotted(fields, "dynamics.equations")
        if not isinstance(equations, list) or equations == [] \
                or not isinstance(variables, list) \
                or len(equations) != len(variables) or not all(
                    isinstance(equation, str) and equation.strip()
                    for equation in equations
                ):
            errors.append(
                f"{instance}: dynamics.equations must align with variable_order"
            )
    if profile == "discrete_execution_contract_v1":
        state_update = _dotted(fields, "transition.state_update")
        if not isinstance(state_update, list) or state_update == [] \
                or not isinstance(variables, list) \
                or len(state_update) != len(variables) or not all(
                    isinstance(update, str) and update.strip()
                    for update in state_update
                ):
            errors.append(
                f"{instance}: transition.state_update must align with variable_order"
            )
        control_order = _dotted(fields, "transition.control_application_order")
        if not isinstance(control_order, str) or not control_order.strip():
            errors.append(
                f"{instance}: transition.control_application_order must be text"
            )
    units = _dotted(fields, "width_comparison.coordinate_units")
    if not isinstance(units, list) or not isinstance(variables, list) \
            or len(units) != len(variables) or not all(
        isinstance(item, str) and item.strip() for item in units
    ):
        errors.append(
            f"{instance}: resolved contract field "
            "width_comparison.coordinate_units must align with variable_order"
        )
    if _dotted(fields, "width_comparison.aggregation_semantics") \
            != "union_and_per_partition":
        errors.append(
            f"{instance}: resolved contract field "
            "width_comparison.aggregation_semantics is unsupported"
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
    sample_points = _dotted(fields, "width_comparison.sample_points")
    expected_end = (
        _dotted(fields, "integration.horizon")
        if profile == "full_execution_contract_v1"
        else _dotted(fields, "discrete.transition_count")
    )
    points_are_numbers = isinstance(sample_points, list) and len(sample_points) >= 2 \
        and all(_is_finite_number(point) for point in sample_points)
    if profile == "discrete_execution_contract_v1" and points_are_numbers:
        points_are_numbers = all(
            isinstance(point, int) and not isinstance(point, bool)
            for point in sample_points
        )
    if not points_are_numbers or sample_points[0] != 0 \
            or sample_points[-1] != expected_end \
            or any(left >= right for left, right in zip(
                sample_points, sample_points[1:]
            )):
        errors.append(
            f"{instance}: resolved contract field width_comparison.sample_points "
            "must be a strictly increasing 0-to-horizon grid"
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
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            errors.append(
                f"{instance}: resolved contract field {updates_path} "
                "must be a positive integer"
            )
        schedule_points = _dotted(fields, "controller_update.schedule_points")
        extent_end = (
            _dotted(fields, "integration.horizon")
            if profile == "full_execution_contract_v1"
            else _dotted(fields, "discrete.transition_count")
        )
        points_valid = isinstance(schedule_points, list) \
            and len(schedule_points) == value \
            and _is_finite_number(extent_end, positive=True) \
            and all(_is_finite_number(point) for point in schedule_points)
        if profile == "discrete_execution_contract_v1" and points_valid:
            points_valid = all(
                isinstance(point, int) and not isinstance(point, bool)
                for point in schedule_points
            )
        if not points_valid or schedule_points[0] != 0 \
                or any(left >= right for left, right in zip(
                    schedule_points, schedule_points[1:]
                )) or not all(point < extent_end for point in schedule_points):
            errors.append(
                f"{instance}: controller_update.schedule_points must bind every "
                "scheduled update in order"
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


def _validate_prelaunch_audit(
    campaign: Mapping[str, Any],
    root: Path,
    *,
    require_fresh: bool = False,
    now_utc: datetime | None = None,
) -> list[str]:
    link = campaign.get("prelaunch_audit")
    if not isinstance(link, Mapping):
        return ["comparison_campaign.prelaunch_audit: expected an object"]
    values = (link.get("path"), link.get("sha256"))
    if all(value is None for value in values):
        return []
    if any(value is None for value in values):
        return ["comparison_campaign.prelaunch_audit: partial pointer"]
    path, errors = _bound_file(
        root, link.get("path"), link.get("sha256"),
        "comparison_campaign.prelaunch_audit",
    )
    if path is None:
        return errors
    try:
        receipt = _load(path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return [*errors, f"comparison_campaign.prelaunch_audit: cannot load: {error}"]
    expected = {
        "schema_version", "campaign_id", "checked_at_utc", "valid_until_utc",
        "server_research_root", "process_scan_artifact", "original_native_quad",
        "replacement_jobs", "duplicate_launch_absent",
    }
    errors.extend(_exact_keys(
        receipt, expected, "comparison_campaign.prelaunch_audit.receipt"
    ))
    if errors:
        return errors
    if receipt["schema_version"] != PRELAUNCH_AUDIT_SCHEMA:
        errors.append("comparison_campaign.prelaunch_audit: wrong schema_version")
    if receipt["campaign_id"] != campaign.get("campaign_id"):
        errors.append("comparison_campaign.prelaunch_audit: campaign identity mismatch")
    checked_at = _parse_utc(receipt["checked_at_utc"])
    valid_until = _parse_utc(receipt["valid_until_utc"])
    if checked_at is None:
        errors.append("comparison_campaign.prelaunch_audit: invalid checked_at_utc")
    if valid_until is None:
        errors.append("comparison_campaign.prelaunch_audit: invalid valid_until_utc")
    if checked_at is not None and valid_until is not None and not (
        checked_at < valid_until <= checked_at + timedelta(hours=24)
    ):
        errors.append(
            "comparison_campaign.prelaunch_audit: validity window must be positive "
            "and no longer than 24 hours"
        )
    if require_fresh and checked_at is not None and valid_until is not None:
        current = now_utc or datetime.now(timezone.utc)
        if current.tzinfo is None:
            current = current.replace(tzinfo=timezone.utc)
        if not checked_at < current < valid_until:
            errors.append(
                "comparison_campaign.prelaunch_audit: receipt is not fresh at launch"
            )
    server_root = receipt["server_research_root"]
    expected_root = SERVER_RESEARCH_ROOT
    if server_root != expected_root:
        errors.append("comparison_campaign.prelaunch_audit: wrong server research root")
    job_fields = {
        "job_id", "run_directory", "watch_directory",
        "launch_identity_artifact", "terminal_status",
        "terminal_evidence_artifact",
    }
    terminal_statuses = {"completed", "failed", "timeout", "interrupted"}
    original = receipt["original_native_quad"]
    original_errors = _exact_keys(
        original, job_fields,
        "comparison_campaign.prelaunch_audit.original_native_quad",
    )
    errors.extend(original_errors)
    jobs = [] if original_errors else [original]
    replacements = receipt["replacement_jobs"]
    if not isinstance(replacements, list):
        errors.append("comparison_campaign.prelaunch_audit.replacement_jobs: expected array")
    else:
        for index, replacement in enumerate(replacements):
            item_errors = _exact_keys(
                replacement, job_fields,
                f"comparison_campaign.prelaunch_audit.replacement_jobs[{index}]",
            )
            errors.extend(item_errors)
            if not item_errors:
                jobs.append(replacement)
    if isinstance(replacements, list) and len(replacements) \
            != len(EXPECTED_NATIVE_REPLACEMENTS):
        errors.append(
            "comparison_campaign.prelaunch_audit.replacement_jobs: does not match "
            "the independently frozen replacement inventory"
        )
    observed_directories: list[str] = []
    observed_job_ids: list[str] = []
    for index, job in enumerate(jobs):
        label = (
            "original_native_quad" if index == 0 else f"replacement_jobs[{index - 1}]"
        )
        directory = job["run_directory"]
        watch_directory = job["watch_directory"]
        job_id = job["job_id"]
        if not isinstance(job_id, str) or not job_id.strip():
            errors.append(
                f"comparison_campaign.prelaunch_audit.{label}: job_id is invalid"
            )
        else:
            observed_job_ids.append(job_id)
        for directory_name, candidate in (
            ("run directory", directory), ("watch directory", watch_directory)
        ):
            directory_path = (
                PurePosixPath(candidate) if isinstance(candidate, str) else None
            )
            if not isinstance(candidate, str) or directory_path is None \
                    or not candidate.startswith(expected_root + "/") \
                    or ".." in directory_path.parts \
                    or "." in directory_path.parts \
                    or str(directory_path) != candidate:
                errors.append(
                    f"comparison_campaign.prelaunch_audit.{label}: "
                    f"{directory_name} is outside root"
                )
        if isinstance(directory, str):
            observed_directories.append(directory)
        expected_job = (
            ORIGINAL_NATIVE_QUAD_IDENTITY
            if index == 0 else EXPECTED_NATIVE_REPLACEMENTS[index - 1]
        ) if index == 0 or index - 1 < len(EXPECTED_NATIVE_REPLACEMENTS) else None
        if expected_job is not None and any(
            job.get(name) != expected_job.get(name)
            for name in (
                "job_id", "run_directory", "watch_directory", "terminal_status",
            )
        ):
            errors.append(
                f"comparison_campaign.prelaunch_audit.{label}: identity does not "
                "match the independently frozen job"
            )
        if job["terminal_status"] not in terminal_statuses:
            errors.append(
                f"comparison_campaign.prelaunch_audit.{label}: job is not terminal"
            )
        launch_label = (
            f"comparison_campaign.prelaunch_audit.{label}.launch_identity_artifact"
        )
        launch_link = job["launch_identity_artifact"]
        launch_artifact_errors = _validate_artifact(
            launch_link, root, launch_label
        )
        errors.extend(launch_artifact_errors)
        launch_path, launch_errors = _bound_file(
            root,
            launch_link.get("path") if isinstance(launch_link, Mapping) else None,
            launch_link.get("sha256") if isinstance(launch_link, Mapping) else None,
            launch_label,
        )
        errors.extend(error for error in launch_errors if error not in errors)
        if launch_path is not None and not launch_artifact_errors:
            try:
                launch = _load(launch_path)
            except (OSError, ValueError, json.JSONDecodeError) as error:
                errors.append(f"{launch_label}: cannot load: {error}")
            else:
                launch_fields = {
                    "schema_version", "server_research_root", "run_directory",
                    "watch_directory", "job_id", "command_sha256",
                    "created_at_utc",
                }
                launch_shape = _exact_keys(launch, launch_fields, launch_label)
                errors.extend(launch_shape)
                if not launch_shape:
                    if launch["schema_version"] != NATIVE_LAUNCH_SCHEMA:
                        errors.append(f"{launch_label}: wrong schema_version")
                    if (
                        launch["server_research_root"] != expected_root
                        or launch["run_directory"] != directory
                        or launch["watch_directory"] != watch_directory
                        or launch["job_id"] != job["job_id"]
                    ):
                        errors.append(f"{launch_label}: job identity mismatch")
                    expected_command = (
                        expected_job.get("command_sha256")
                        if expected_job is not None else None
                    )
                    if not _is_sha256(launch["command_sha256"]) \
                            or launch["command_sha256"] != expected_command:
                        errors.append(f"{launch_label}: command_sha256 is invalid")
                    created = _parse_utc(launch["created_at_utc"])
                    if created is None or (
                        checked_at is not None and created > checked_at
                    ):
                        errors.append(f"{launch_label}: invalid creation time")

        terminal_label = (
            f"comparison_campaign.prelaunch_audit.{label}.terminal_evidence_artifact"
        )
        terminal_link = job["terminal_evidence_artifact"]
        terminal_artifact_errors = _validate_artifact(
            terminal_link, root, terminal_label
        )
        errors.extend(terminal_artifact_errors)
        terminal_path, terminal_errors = _bound_file(
            root,
            terminal_link.get("path") if isinstance(terminal_link, Mapping) else None,
            terminal_link.get("sha256") if isinstance(terminal_link, Mapping) else None,
            terminal_label,
        )
        errors.extend(error for error in terminal_errors if error not in errors)
        if terminal_path is not None and not terminal_artifact_errors:
            try:
                terminal = _load(terminal_path)
            except (OSError, ValueError, json.JSONDecodeError) as error:
                errors.append(f"{terminal_label}: cannot load: {error}")
            else:
                terminal_fields = {
                    "schema_version", "server_research_root", "run_directory",
                    "job_id", "terminal_status", "checked_at_utc",
                    "result_artifact",
                }
                terminal_shape = _exact_keys(
                    terminal, terminal_fields, terminal_label
                )
                errors.extend(terminal_shape)
                if not terminal_shape:
                    if terminal["schema_version"] != NATIVE_TERMINAL_SCHEMA:
                        errors.append(f"{terminal_label}: wrong schema_version")
                    if (
                        terminal["server_research_root"] != expected_root
                        or terminal["run_directory"] != directory
                        or terminal["job_id"] != job["job_id"]
                        or terminal["terminal_status"] != job["terminal_status"]
                    ):
                        errors.append(f"{terminal_label}: terminal identity mismatch")
                    terminal_checked = _parse_utc(terminal["checked_at_utc"])
                    if terminal_checked is None or (
                        checked_at is not None and terminal_checked > checked_at
                    ):
                        errors.append(f"{terminal_label}: invalid terminal check time")
                    errors.extend(_validate_artifact(
                        terminal["result_artifact"], root,
                        f"{terminal_label}.result_artifact",
                    ))
                    expected_result_sha = (
                        expected_job.get("result_sha256")
                        if expected_job is not None else None
                    )
                    if terminal["result_artifact"].get("sha256") \
                            != expected_result_sha:
                        errors.append(
                            f"{terminal_label}: result identity does not match "
                            "the independently frozen job"
                        )

    if len(observed_job_ids) != len(set(observed_job_ids)) \
            or len(observed_directories) != len(set(observed_directories)):
        errors.append(
            "comparison_campaign.prelaunch_audit: duplicate job identity or run directory"
        )

    scan_label = "comparison_campaign.prelaunch_audit.process_scan_artifact"
    scan_link = receipt["process_scan_artifact"]
    scan_artifact_errors = _validate_artifact(scan_link, root, scan_label)
    errors.extend(scan_artifact_errors)
    scan_path, scan_errors = _bound_file(
        root,
        scan_link.get("path") if isinstance(scan_link, Mapping) else None,
        scan_link.get("sha256") if isinstance(scan_link, Mapping) else None,
        scan_label,
    )
    errors.extend(error for error in scan_errors if error not in errors)
    if scan_path is not None and not scan_artifact_errors:
        try:
            scan = _load(scan_path)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            errors.append(f"{scan_label}: cannot load: {error}")
        else:
            scan_fields = {
                "schema_version", "campaign_id", "server_research_root",
                "checked_at_utc", "active_owned_processes",
                "observed_run_directories",
            }
            scan_shape = _exact_keys(scan, scan_fields, scan_label)
            errors.extend(scan_shape)
            if not scan_shape:
                if scan["schema_version"] != PROCESS_SCAN_SCHEMA:
                    errors.append(f"{scan_label}: wrong schema_version")
                if (
                    scan["campaign_id"] != campaign.get("campaign_id")
                    or scan["server_research_root"] != expected_root
                    or scan["checked_at_utc"] != receipt["checked_at_utc"]
                ):
                    errors.append(f"{scan_label}: audit identity mismatch")
                if scan["active_owned_processes"] != []:
                    errors.append(f"{scan_label}: owned benchmark process is still active")
                directories = scan["observed_run_directories"]
                if not isinstance(directories, list) or len(directories) != len(
                    set(directories)
                ) or set(directories) != set(observed_directories):
                    errors.append(f"{scan_label}: observed run directories mismatch")
    if receipt["duplicate_launch_absent"] is not True:
        errors.append(
            "comparison_campaign.prelaunch_audit: duplicate launch absence is not confirmed"
        )
    return errors


def _active_run_receipt(
    campaign: Mapping[str, Any], root: Path
) -> tuple[Mapping[str, Any] | None, list[str]]:
    label = "comparison_campaign.active_run_receipt"
    link = campaign.get("active_run_receipt")
    if not isinstance(link, Mapping):
        return None, [f"{label}: expected an object"]
    values = (link.get("path"), link.get("sha256"))
    if all(value is None for value in values):
        return None, []
    if any(value is None for value in values):
        return None, [f"{label}: partial pointer"]
    errors = _validate_artifact(link, root, label)
    path, bound_errors = _bound_file(
        root, link.get("path"), link.get("sha256"), label
    )
    errors.extend(error for error in bound_errors if error not in errors)
    if path is None:
        return None, errors
    try:
        receipt = _load(path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return None, [*errors, f"{label}: cannot load: {error}"]
    fields = {
        "schema_version", "campaign_id", "instance_id", "method",
        "role", "index", "attempt_index", "invocation_id", "pid",
        "started_at_utc", "prelaunch_audit", "lock_artifact",
    }
    shape = _exact_keys(receipt, fields, label)
    errors.extend(shape)
    if shape:
        return receipt, errors
    if receipt["schema_version"] != ACTIVE_RUN_SCHEMA:
        errors.append(f"{label}: wrong schema_version")
    if receipt["campaign_id"] != campaign.get("campaign_id"):
        errors.append(f"{label}: campaign identity mismatch")
    if not isinstance(receipt["instance_id"], str) or not receipt["instance_id"]:
        errors.append(f"{label}: instance_id is invalid")
    if receipt["method"] not in EXPECTED_METHODS:
        errors.append(f"{label}: method is invalid")
    if receipt["role"] not in {"cold", "steady", "diagnostic"}:
        errors.append(f"{label}: role is invalid")
    for name in ("index", "attempt_index"):
        if not _is_non_negative_int(receipt[name]):
            errors.append(f"{label}.{name}: invalid index")
    if not isinstance(receipt["invocation_id"], str) \
            or not receipt["invocation_id"].strip():
        errors.append(f"{label}: invocation_id is invalid")
    if not isinstance(receipt["pid"], int) or isinstance(receipt["pid"], bool) \
            or receipt["pid"] <= 0:
        errors.append(f"{label}: pid is invalid")
    if _parse_utc(receipt["started_at_utc"]) is None:
        errors.append(f"{label}: started_at_utc is invalid")
    audit_link = receipt["prelaunch_audit"]
    audit_label = f"{label}.prelaunch_audit"
    audit_errors = _validate_artifact(audit_link, root, audit_label)
    errors.extend(audit_errors)
    if audit_link != campaign.get("prelaunch_audit"):
        errors.append(f"{label}: prelaunch audit identity mismatch")
    audit_checked: datetime | None = None
    audit_path, _ = _bound_file(
        root,
        audit_link.get("path") if isinstance(audit_link, Mapping) else None,
        audit_link.get("sha256") if isinstance(audit_link, Mapping) else None,
        audit_label,
    )
    if audit_path is not None and not audit_errors:
        try:
            audit_receipt = _load(audit_path)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            errors.append(f"{audit_label}: cannot load: {error}")
        else:
            audit_checked = _parse_utc(audit_receipt.get("checked_at_utc"))
    lock_label = f"{label}.lock_artifact"
    lock_link = receipt["lock_artifact"]
    lock_errors = _validate_artifact(lock_link, root, lock_label)
    errors.extend(lock_errors)
    lock_path, _ = _bound_file(
        root,
        lock_link.get("path") if isinstance(lock_link, Mapping) else None,
        lock_link.get("sha256") if isinstance(lock_link, Mapping) else None,
        lock_label,
    )
    if lock_path is not None and not lock_errors:
        try:
            lock = _load(lock_path)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            errors.append(f"{lock_label}: cannot load: {error}")
        else:
            lock_fields = {
                "schema_version", "campaign_id", "host_identity", "instance_id",
                "method", "invocation_id", "acquired_at_utc", "lock_path",
                "protocol", "wrapper_pid", "lock_identity",
            }
            lock_shape = _exact_keys(lock, lock_fields, lock_label)
            errors.extend(lock_shape)
            if not lock_shape:
                if lock["schema_version"] != CAMPAIGN_LOCK_SCHEMA:
                    errors.append(f"{lock_label}: wrong schema_version")
                expected_identity = {
                    "campaign_id": campaign.get("campaign_id"),
                    "host_identity": campaign.get("host_identity"),
                    "instance_id": receipt["instance_id"],
                    "method": receipt["method"],
                    "invocation_id": receipt["invocation_id"],
                    "lock_path": _dotted(campaign, "launch_guard.lock_path"),
                    "protocol": _dotted(campaign, "launch_guard.protocol"),
                    "lock_identity": _dotted(
                        campaign, "launch_guard.lock_identity"
                    ),
                }
                if any(lock.get(name) != value for name, value in expected_identity.items()):
                    errors.append(f"{lock_label}: lock identity mismatch")
                acquired = _parse_utc(lock.get("acquired_at_utc"))
                started = _parse_utc(receipt["started_at_utc"])
                if acquired is None or started is None or audit_checked is None \
                        or not acquired < audit_checked < started:
                    errors.append(
                        f"{lock_label}: lock/audit/spawn order is invalid"
                    )
                if not _is_non_negative_int(lock.get("wrapper_pid")) \
                        or lock["wrapper_pid"] <= 0:
                    errors.append(f"{lock_label}: wrapper_pid is invalid")
    return receipt, errors


def _validate_campaign_structure(
    campaign: Any, root: Path
) -> list[str]:
    errors = _shape_errors(campaign, CAMPAIGN_SHAPE, "comparison_campaign")
    if errors:
        return errors
    if campaign["schema_version"] != "archcomp26-comparison-campaign-v1":
        errors.append("comparison_campaign: wrong schema_version")
    for name in ("campaign_id", "host_identity", "hardware_identity"):
        value = campaign[name]
        if value is not None and not (
            isinstance(value, str) and value.strip()
        ):
            errors.append(f"comparison_campaign.{name}: expected null or text")
    threads = campaign["cpu_thread_budget"]
    if threads is not None and (
        isinstance(threads, bool) or not isinstance(threads, int) or threads <= 0
    ):
        errors.append(
            "comparison_campaign.cpu_thread_budget: expected null or a positive integer"
        )
    timeout = campaign["timeout_s"]
    if timeout is not None and not _is_finite_number(timeout, positive=True):
        errors.append(
            "comparison_campaign.timeout_s: expected null or a finite positive number"
        )
    gpu_budget = campaign["gpu_device_budget"]
    if gpu_budget is not None and not _nonempty(gpu_budget):
        errors.append("comparison_campaign.gpu_device_budget: empty value")
    limits = campaign["resource_limits"]
    if limits is not None:
        limit_errors = _exact_keys(
            limits, RESOURCE_LIMIT_FIELDS, "comparison_campaign.resource_limits"
        )
        errors.extend(limit_errors)
        if not limit_errors:
            for name in ("exclusive_host", "exclusive_gpu_device"):
                if limits[name] is not True:
                    errors.append(
                        f"comparison_campaign.resource_limits.{name}: must be true"
                    )
            host_memory = limits["max_host_memory_bytes"]
            device_memory = limits["max_device_memory_bytes"]
            if not _is_non_negative_int(host_memory) or host_memory <= 0:
                errors.append(
                    "comparison_campaign.resource_limits.max_host_memory_bytes: "
                    "expected a positive integer"
                )
            if device_memory is not None and not _is_non_negative_int(device_memory):
                errors.append(
                    "comparison_campaign.resource_limits.max_device_memory_bytes: "
                    "expected null or a non-negative integer"
                )
    errors.extend(_validate_prelaunch_audit(campaign, root))
    _, active_errors = _active_run_receipt(campaign, root)
    errors.extend(active_errors)
    guard = campaign["launch_guard"]
    expected_guard = {
        "schema_version": "archcomp26-atomic-launch-guard-v2",
        "protocol": "posix-flock-exclusive-nonblocking-v1",
        "lock_path": SERVER_RESEARCH_ROOT + "/.archcomp26/launch.lock",
        "hold_scope": "fresh_audit_through_terminal_fsync_v1",
        "wrapper_module": "torch_tm_flowpipe.archcomp26_launch",
        "wrapper_path": "src/torch_tm_flowpipe/archcomp26_launch.py",
    }
    if any(guard.get(name) != value for name, value in expected_guard.items()):
        errors.append(
            "comparison_campaign.launch_guard: does not match the fixed atomic "
            "launch protocol"
        )
    lock_identity = guard["lock_identity"]
    lock_identity_errors = _exact_keys(
        lock_identity, {"device", "inode", "owner_uid", "mode", "nlink"},
        "comparison_campaign.launch_guard.lock_identity",
    )
    errors.extend(lock_identity_errors)
    if not lock_identity_errors:
        frozen = [
            lock_identity[name] for name in ("device", "inode", "owner_uid")
        ]
        if any(value is None for value in frozen) and any(
            value is not None for value in frozen
        ):
            errors.append(
                "comparison_campaign.launch_guard.lock_identity: partial identity"
            )
        for name, value in zip(("device", "inode", "owner_uid"), frozen):
            if value is not None and (
                not _is_non_negative_int(value) or name == "inode" and value == 0
            ):
                errors.append(
                    "comparison_campaign.launch_guard.lock_identity."
                    f"{name}: expected null or a non-negative integer"
                )
        if lock_identity["mode"] != 0o600 or lock_identity["nlink"] != 1:
            errors.append(
                "comparison_campaign.launch_guard.lock_identity: "
                "mode/nlink must be 0600/1"
            )
    wrapper_sha = guard.get("wrapper_sha256")
    if wrapper_sha is not None:
        _, guard_errors = _bound_file(
            root, guard.get("wrapper_path"), wrapper_sha,
            "comparison_campaign.launch_guard.wrapper",
        )
        errors.extend(guard_errors)
    boundary = campaign["timing_boundary"]
    expected_boundary = {
        "version": "total_configuration_v2",
        "start_event": "immediately_before_fresh_process_spawn",
        "stop_event": "after_result_and_width_artifacts_are_durable",
        "phase_fields": [
            "driver_total", "compile", "controller_nn", "solver_core",
            "validation", "observer", "output", "plot_report",
        ],
    }
    if boundary != expected_boundary:
        errors.append(
            "comparison_campaign.timing_boundary: does not match the frozen v2 boundary"
        )
    rotation = campaign["rotation"]
    if rotation["policy"] \
            != "balanced_round_robin_by_instance_and_steady_round_v1":
        errors.append("comparison_campaign.rotation.policy: unsupported policy")
    if rotation["steady_rounds"] != 5:
        errors.append("comparison_campaign.rotation.steady_rounds: expected 5")
    schedule = rotation["schedule_artifact"]
    schedule_values = (schedule["path"], schedule["sha256"])
    if any(value is None for value in schedule_values) \
            and any(value is not None for value in schedule_values):
        errors.append("comparison_campaign.rotation.schedule_artifact: partial pointer")
    if all(value is not None for value in schedule_values):
        _, bound_errors = _bound_file(
            root, schedule["path"], schedule["sha256"],
            "comparison_campaign.rotation.schedule_artifact",
        )
        errors.extend(bound_errors)
    return errors


def _campaign_reasons(
    campaign: Any,
    root: Path,
    cell: Mapping[str, Any] | None = None,
    *,
    require_fresh_audit: bool = False,
    now_utc: datetime | None = None,
) -> list[str]:
    if _validate_campaign_structure(campaign, root):
        return ["comparison_campaign_invalid"]
    reasons: list[str] = []
    for name in (
        "campaign_id", "host_identity", "hardware_identity",
        "cpu_thread_budget", "gpu_device_budget", "timeout_s", "resource_limits",
    ):
        if not _nonempty(campaign[name]):
            reasons.append(f"comparison_campaign_{name}_missing")
    if not _is_sha256(_dotted(campaign, "launch_guard.wrapper_sha256")):
        reasons.append("comparison_campaign_atomic_launcher_unavailable")
    if any(
        not _is_non_negative_int(
            _dotted(campaign, f"launch_guard.lock_identity.{name}")
        )
        or name == "inode"
        and _dotted(campaign, f"launch_guard.lock_identity.{name}") == 0
        for name in ("device", "inode", "owner_uid")
    ):
        reasons.append("comparison_campaign_lock_identity_missing")
    audit = campaign["prelaunch_audit"]
    if audit["path"] is None or audit["sha256"] is None:
        reasons.append("comparison_campaign_prelaunch_audit_missing")
    elif require_fresh_audit and _validate_prelaunch_audit(
        campaign, root, require_fresh=True, now_utc=now_utc
    ):
        reasons.append("comparison_campaign_prelaunch_audit_stale_or_invalid")
    active = campaign["active_run_receipt"]
    if require_fresh_audit and (
        active["path"] is not None or active["sha256"] is not None
    ):
        reasons.append("comparison_campaign_active_run_exists")
    schedule = campaign["rotation"]["schedule_artifact"]
    if schedule["path"] is None or schedule["sha256"] is None:
        reasons.append("comparison_campaign_schedule_missing")
    if cell is not None:
        runtime = cell.get("runtime")
        if not isinstance(runtime, Mapping):
            reasons.append("comparison_campaign_runtime_missing")
        else:
            matches = {
                "hardware": "hardware_identity",
                "cpu_threads": "cpu_thread_budget",
                "gpu": "gpu_device_budget",
                "timeout_s": "timeout_s",
                "resource_limits": "resource_limits",
            }
            for runtime_name, campaign_name in matches.items():
                if runtime.get(runtime_name) != campaign.get(campaign_name):
                    reasons.append(
                        f"comparison_campaign_{runtime_name}_mismatch"
                    )
        if _dotted(cell, "measurement_plan.timing_boundary_version") \
                != _dotted(campaign, "timing_boundary.version"):
            reasons.append("comparison_campaign_timing_boundary_mismatch")
    return reasons


def _campaign_schedule_entries(
    campaign: Mapping[str, Any], root: Path
) -> tuple[dict[tuple[str, str, str, int], Mapping[str, Any]], list[str]]:
    link = _dotted(campaign, "rotation.schedule_artifact")
    if not isinstance(link, Mapping):
        return {}, ["comparison_campaign.schedule: missing artifact pointer"]
    path, errors = _bound_file(
        root, link.get("path"), link.get("sha256"),
        "comparison_campaign.rotation.schedule_artifact",
    )
    if path is None:
        return {}, errors
    try:
        payload = _load(path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return {}, [f"comparison_campaign.schedule: cannot load: {error}"]
    errors.extend(_exact_keys(
        payload, {"schema_version", "campaign_id", "entries"},
        "comparison_campaign.schedule",
    ))
    if errors:
        return {}, errors
    if payload["schema_version"] != "archcomp26-campaign-schedule-v1":
        errors.append("comparison_campaign.schedule: wrong schema_version")
    if payload["campaign_id"] != campaign.get("campaign_id"):
        errors.append("comparison_campaign.schedule: campaign identity mismatch")
    raw_entries = payload["entries"]
    if not isinstance(raw_entries, list):
        return {}, [*errors, "comparison_campaign.schedule.entries: expected an array"]
    entries: dict[tuple[str, str, str, int], Mapping[str, Any]] = {}
    fields = {
        "instance_id", "method", "role", "index", "round_index",
        "sequence_position",
    }
    for index, entry in enumerate(raw_entries):
        label = f"comparison_campaign.schedule.entries[{index}]"
        entry_errors = _exact_keys(entry, fields, label)
        errors.extend(entry_errors)
        if entry_errors:
            continue
        if not all(isinstance(entry[name], str) and entry[name] for name in (
            "instance_id", "method", "role",
        )) or not all(_is_non_negative_int(entry[name]) for name in (
            "index", "round_index", "sequence_position",
        )) or not _mode_is(entry["role"], "cold", "steady"):
            errors.append(f"{label}: invalid schedule entry")
            continue
        key = (
            entry["instance_id"], entry["method"], entry["role"], entry["index"]
        )
        if key in entries:
            errors.append(f"{label}: duplicate formal sample key {key!r}")
        entries[key] = entry
    return entries, errors


def _validate_campaign_schedule(
    campaign: Mapping[str, Any],
    root: Path,
    instance_ids: list[str],
    methods: list[str],
) -> list[str]:
    link = _dotted(campaign, "rotation.schedule_artifact")
    if not isinstance(link, Mapping) or link.get("path") is None:
        return []
    entries, errors = _campaign_schedule_entries(campaign, root)
    expected_keys = {
        (instance, method, role, index)
        for instance in instance_ids
        for method in methods
        for role, count in (("cold", 1), ("steady", 5))
        for index in range(count)
    }
    if set(entries) != expected_keys:
        errors.append(
            "comparison_campaign.schedule: formal sample coverage does not match "
            "the full 16-by-4 campaign"
        )
    method_count = len(methods)
    for key, entry in entries.items():
        instance, method, role, index = key
        if instance not in instance_ids or method not in methods or method_count == 0:
            continue
        round_index = index if role == "steady" else 0
        rotation_offset = (
            instance_ids.index(instance) + round_index
        ) % method_count
        expected_position = (
            methods.index(method) - rotation_offset
        ) % method_count
        if entry["round_index"] != round_index \
                or entry["sequence_position"] != expected_position:
            errors.append(
                f"comparison_campaign.schedule: {key!r} violates balanced rotation"
            )
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
    if set(profiles) != set(SHARED_PROFILE_FIELDS):
        errors.append(
            "manifest unresolved_field_profiles do not match the fixed shared profiles"
        )
    for profile, fixed_fields in SHARED_PROFILE_FIELDS.items():
        if profiles.get(profile) != list(fixed_fields):
            errors.append(
                f"manifest unresolved_field_profiles.{profile} does not match "
                "the code-fixed shared contract"
            )
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
        profile = contract.get("unresolved_field_profile")
        expected_profile = EXPECTED_PROFILE_BY_INSTANCE.get(instance)
        if expected_profile is not None and profile != expected_profile:
            errors.append(
                f"{instance}: contract profile {profile!r} does not match "
                f"the fixed instance profile {expected_profile!r}"
            )
        if status == "unresolved":
            if not isinstance(profile, str) or profile not in profiles:
                errors.append(f"{instance}: unknown unresolved contract profile {profile!r}")
            continue
        if status != "resolved":
            errors.append(f"{instance}: invalid contract status {status!r}")
            continue
        required = SHARED_PROFILE_FIELDS.get(profile) \
            if isinstance(profile, str) else None
        if not isinstance(required, tuple) or not required:
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
        shared_required = list(required)
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


def _validate_typed_plot_artifact(
    value: Mapping[str, Any], root: Path, label: str
) -> list[str]:
    role = value.get("role")
    if role not in PLOT_ARTIFACT_ROLES and role != "matlab_script":
        return []
    path, errors = _bound_file(
        root, value.get("path"), value.get("sha256"), label
    )
    if path is None:
        return errors
    expected_suffix = (
        PLOT_ARTIFACT_ROLES[role] if role in PLOT_ARTIFACT_ROLES else ".m"
    )
    if path.suffix.lower() != expected_suffix:
        errors.append(f"{label}: {role} must use a {expected_suffix} file")
        return errors
    payload = path.read_bytes()
    if role == "plot_png":
        png_error: str | None = None
        ihdr: tuple[int, int, int, int, int, int, int] | None = None
        idat_parts: list[bytes] = []
        palette_seen = False
        idat_started = False
        idat_ended = False
        if not payload.startswith(b"\x89PNG\r\n\x1a\n"):
            png_error = "lacks a PNG signature"
        else:
            offset = 8
            chunk_names: list[bytes] = []
            while offset < len(payload):
                if offset + 12 > len(payload):
                    png_error = "has a truncated PNG chunk"
                    break
                length = struct.unpack(">I", payload[offset:offset + 4])[0]
                chunk_end = offset + 12 + length
                if chunk_end > len(payload):
                    png_error = "has a truncated PNG chunk payload"
                    break
                name = payload[offset + 4:offset + 8]
                data = payload[offset + 8:offset + 8 + length]
                expected_crc = struct.unpack(">I", payload[offset + 8 + length:chunk_end])[0]
                if zlib.crc32(name + data) & 0xFFFFFFFF != expected_crc:
                    png_error = "has a PNG CRC mismatch"
                    break
                chunk_names.append(name)
                if len(chunk_names) == 1:
                    if name != b"IHDR" or length != 13:
                        png_error = "has an invalid PNG IHDR"
                        break
                    ihdr = struct.unpack(">IIBBBBB", data)
                    width, height, bit_depth, colour_type, compression, filtering, \
                        interlace = ihdr
                    valid_depths = {
                        0: {1, 2, 4, 8, 16}, 2: {8, 16}, 3: {1, 2, 4, 8},
                        4: {8, 16}, 6: {8, 16},
                    }
                    if width <= 0 or height <= 0 \
                            or width * height > 100_000_000 \
                            or bit_depth not in valid_depths.get(colour_type, set()) \
                            or compression != 0 or filtering != 0 \
                            or interlace not in {0, 1}:
                        png_error = "has an invalid PNG IHDR"
                        break
                if name == b"PLTE":
                    palette_seen = True
                if name == b"IDAT":
                    if idat_ended:
                        png_error = "has non-contiguous PNG IDAT chunks"
                        break
                    idat_started = True
                    idat_parts.append(data)
                elif idat_started:
                    idat_ended = True
                if name == b"IEND" and length != 0:
                    png_error = "has an invalid PNG IEND"
                    break
                offset = chunk_end
                if name == b"IEND":
                    break
            if png_error is None and (
                not chunk_names
                or chunk_names[0] != b"IHDR"
                or b"IDAT" not in chunk_names
                or chunk_names[-1] != b"IEND"
                or offset != len(payload)
            ):
                png_error = "is not a complete PNG image"
        if png_error is None and ihdr is not None:
            width, height, bit_depth, colour_type, _, _, interlace = ihdr
            if colour_type == 3 and not palette_seen:
                png_error = "has an indexed image without a palette"
            else:
                decoder = zlib.decompressobj()
                try:
                    decoded = decoder.decompress(b"".join(idat_parts)) + decoder.flush()
                except zlib.error:
                    decoded = b""
                    png_error = "has an undecodable PNG IDAT stream"
                if png_error is None and (
                    not decoder.eof or decoder.unused_data or decoder.unconsumed_tail
                ):
                    png_error = "has an incomplete PNG IDAT stream"
                if png_error is None:
                    channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[colour_type]
                    bits_per_pixel = channels * bit_depth
                    passes = (
                        [(0, 0, 1, 1)] if interlace == 0 else [
                            (0, 0, 8, 8), (4, 0, 8, 8), (0, 4, 4, 8),
                            (2, 0, 4, 4), (0, 2, 2, 4), (1, 0, 2, 2),
                            (0, 1, 1, 2),
                        ]
                    )
                    cursor = 0
                    for x0, y0, x_step, y_step in passes:
                        pass_width = max(0, (width - x0 + x_step - 1) // x_step)
                        pass_height = max(0, (height - y0 + y_step - 1) // y_step)
                        if pass_width == 0 or pass_height == 0:
                            continue
                        row_size = (pass_width * bits_per_pixel + 7) // 8
                        for _ in range(pass_height):
                            if cursor + 1 + row_size > len(decoded) \
                                    or decoded[cursor] > 4:
                                png_error = "has invalid decoded PNG scanlines"
                                break
                            cursor += 1 + row_size
                        if png_error is not None:
                            break
                    if png_error is None and cursor != len(decoded):
                        png_error = "has an invalid decoded PNG byte count"
        if png_error is not None:
            errors.append(f"{label}: plot_png {png_error}")
    elif role == "plot_pdf":
        try:
            from pypdf import PdfReader
            reader = PdfReader(path, strict=True)
            pages = list(reader.pages)
            valid_pages = bool(pages) and all(
                float(page.mediabox.width) > 0 and float(page.mediabox.height) > 0
                for page in pages
            )
        except (ImportError, OSError, ValueError, TypeError, KeyError):
            valid_pages = False
        except Exception:  # pypdf exposes parser-specific exception subclasses.
            valid_pages = False
        if not valid_pages:
            errors.append(f"{label}: plot_pdf is not a complete page-bearing PDF")
    elif role == "plot_svg":
        try:
            root_element = ET.fromstring(payload)
        except ET.ParseError:
            root_element = None
        root_name = (
            root_element.tag.rsplit("}", 1)[-1].lower()
            if root_element is not None and isinstance(root_element.tag, str)
            else None
        )
        graphical = {
            "circle", "ellipse", "image", "line", "path", "polygon",
            "polyline", "rect", "text", "use",
        }
        has_graphics = root_element is not None and any(
            isinstance(element.tag, str)
            and element.tag.rsplit("}", 1)[-1].lower() in graphical
            for element in root_element.iter()
        )
        if root_name != "svg" or not has_graphics:
            errors.append(f"{label}: plot_svg is not a parseable SVG with graphics")
    elif role == "matlab_script" and not payload.strip():
        errors.append(f"{label}: matlab_script is empty")
    return errors


def _validate_partition_coverage(
    value: Any,
    root: Path,
    label: str,
    contract: Mapping[str, Any] | None,
    *,
    outcome: str,
    ledger_scope: str,
    invocation_id: str | None,
) -> list[str]:
    expected = {
        "requested_partitions", "completed_partitions", "failed_partitions",
        "unattempted_partitions", "boxes_sha256", "ledger_artifact",
    }
    errors = _exact_keys(value, expected, label)
    if errors:
        return errors
    counts = [
        value[name] for name in (
            "requested_partitions", "completed_partitions",
            "failed_partitions", "unattempted_partitions",
        )
    ]
    if any(not _is_non_negative_int(count) for count in counts) \
            or value["requested_partitions"] <= 0:
        errors.append(f"{label}: partition counts must be non-negative integers")
    elif sum(counts[1:]) != counts[0]:
        errors.append(f"{label}: completed/failed/unattempted do not cover requested")
    partitions = (
        _dotted(contract, "fields.initial_set.partitions")
        if isinstance(contract, Mapping) else None
    )
    boxes_sha = (
        _dotted(contract, "fields.initial_set.boxes_sha256")
        if isinstance(contract, Mapping) else None
    )
    if not isinstance(partitions, list) or not partitions:
        errors.append(f"{label}: cannot bind initial-set partition count")
    elif value["requested_partitions"] != len(partitions):
        errors.append(f"{label}: requested partition count disagrees with contract")
    if value["boxes_sha256"] != boxes_sha:
        errors.append(f"{label}: boxes SHA-256 disagrees with contract")
    ledger = value["ledger_artifact"]
    if outcome == "skipped":
        if ledger is not None or counts[1:] != [0, 0, counts[0]]:
            errors.append(f"{label}: skipped coverage must be wholly unattempted")
    else:
        if ledger is None:
            errors.append(f"{label}: executed coverage lacks a partition ledger")
        else:
            artifact_errors = _validate_artifact(
                ledger, root, f"{label}.ledger_artifact"
            )
            errors.extend(artifact_errors)
            ledger_path, ledger_errors = _bound_file(
                root, ledger.get("path") if isinstance(ledger, Mapping) else None,
                ledger.get("sha256") if isinstance(ledger, Mapping) else None,
                f"{label}.ledger_artifact",
            )
            if not artifact_errors:
                errors.extend(ledger_errors)
            if ledger_path is not None and not artifact_errors:
                try:
                    payload = _load(ledger_path)
                except (OSError, ValueError, json.JSONDecodeError) as error:
                    errors.append(f"{label}.ledger_artifact: cannot load: {error}")
                else:
                    ledger_fields = {
                        "schema_version", "instance_id", "scope", "invocation_id",
                        "boxes_sha256", "entries",
                    }
                    payload_errors = _exact_keys(
                        payload, ledger_fields, f"{label}.ledger_artifact.payload"
                    )
                    errors.extend(payload_errors)
                    if not payload_errors:
                        if payload["schema_version"] != PARTITION_LEDGER_SCHEMA:
                            errors.append(
                                f"{label}.ledger_artifact: wrong schema_version"
                            )
                        expected_instance = (
                            contract.get("instance_id")
                            if isinstance(contract, Mapping) else None
                        )
                        if payload["instance_id"] != expected_instance:
                            errors.append(
                                f"{label}.ledger_artifact: instance identity mismatch"
                            )
                        if payload["scope"] != ledger_scope \
                                or payload["invocation_id"] != invocation_id:
                            errors.append(
                                f"{label}.ledger_artifact: scope/invocation mismatch"
                            )
                        if payload["boxes_sha256"] != boxes_sha:
                            errors.append(
                                f"{label}.ledger_artifact: boxes SHA-256 mismatch"
                            )
                        entries = payload["entries"]
                        expected_entry_fields = {"index", "box_sha256", "outcome"}
                        entry_outcomes: list[str] = []
                        if not isinstance(entries, list) or not isinstance(
                            partitions, list
                        ) or len(entries) != len(partitions):
                            errors.append(
                                f"{label}.ledger_artifact: entries do not cover every partition"
                            )
                        else:
                            for index, (entry, partition) in enumerate(zip(
                                entries, partitions
                            )):
                                entry_label = f"{label}.ledger_artifact.entries[{index}]"
                                entry_errors = _exact_keys(
                                    entry, expected_entry_fields, entry_label
                                )
                                errors.extend(entry_errors)
                                if entry_errors:
                                    continue
                                if entry["index"] != index:
                                    errors.append(
                                        f"{entry_label}: index does not match contract order"
                                    )
                                if entry["box_sha256"] != _canonical_sha256(partition):
                                    errors.append(
                                        f"{entry_label}: box hash does not match contract"
                                    )
                                entry_outcome = entry["outcome"]
                                if entry_outcome not in {
                                    "completed", "failed", "unattempted"
                                }:
                                    errors.append(f"{entry_label}: invalid outcome")
                                else:
                                    entry_outcomes.append(entry_outcome)
                            derived = [
                                len(entries),
                                entry_outcomes.count("completed"),
                                entry_outcomes.count("failed"),
                                entry_outcomes.count("unattempted"),
                            ]
                            if derived != counts:
                                errors.append(
                                    f"{label}: counts disagree with partition ledger"
                                )
    if outcome == "completed" and counts[1:] != [counts[0], 0, 0]:
        errors.append(f"{label}: completed coverage does not include every partition")
    return errors


def _validate_controller_update_trace(
    value: Any,
    root: Path,
    label: str,
    contract: Mapping[str, Any] | None,
    *,
    observed_count: Any,
    validated_extent: Any,
    outcome: str,
) -> list[str]:
    errors = _exact_keys(
        value, {"kind", "points", "boundary_update_status", "artifact"}, label
    )
    if errors:
        return errors
    expected_points = (
        _dotted(contract, "fields.controller_update.schedule_points")
        if isinstance(contract, Mapping) else None
    )
    expected_kind = (
        "steps"
        if isinstance(contract, Mapping)
        and contract.get("profile") == "discrete_execution_contract_v1"
        else "time_s"
    )
    points = value["points"]
    if value["kind"] != expected_kind:
        errors.append(f"{label}.kind: disagrees with contract profile")
    if not isinstance(points, list) or not all(
        _is_finite_number(point) for point in points
    ) or any(left >= right for left, right in zip(points, points[1:])):
        errors.append(f"{label}.points: expected strictly increasing finite points")
        points = []
    elif expected_kind == "steps" and any(
        isinstance(point, bool) or not isinstance(point, int) for point in points
    ):
        errors.append(f"{label}.points: discrete update points must be integers")
    if not _is_non_negative_int(observed_count) or len(points) != observed_count:
        errors.append(f"{label}.points: count disagrees with observed updates")
    if isinstance(expected_points, list) and points != expected_points[:len(points)]:
        errors.append(f"{label}.points: not a prefix of the contract schedule")
    extent_value = (
        validated_extent.get("value")
        if isinstance(validated_extent, Mapping) else None
    )
    if _is_finite_number(extent_value) and any(
        point > extent_value for point in points
    ):
        errors.append(f"{label}.points: update lies beyond validated extent")
    boundary_scheduled = (
        isinstance(expected_points, list) and extent_value in expected_points
    )
    boundary_status = value["boundary_update_status"]
    if boundary_scheduled:
        if boundary_status not in {"executed", "not_executed"}:
            errors.append(
                f"{label}.boundary_update_status: scheduled boundary requires an "
                "executed/not_executed decision"
            )
    elif boundary_status != "not_scheduled":
        errors.append(
            f"{label}.boundary_update_status: no update is scheduled at the boundary"
        )
    if outcome != "skipped" and isinstance(expected_points, list) \
            and _is_finite_number(extent_value):
        required_prefix = [
            point for point in expected_points if point < extent_value
        ]
        if boundary_scheduled and boundary_status == "executed":
            required_prefix.append(extent_value)
        if points != required_prefix:
            errors.append(
                f"{label}.points: does not match scheduled updates before/at "
                "the validated extent"
            )
    artifact = value["artifact"]
    if outcome == "skipped":
        expected_skipped_boundary = (
            "not_executed" if boundary_scheduled else "not_scheduled"
        )
        if points or artifact is not None \
                or boundary_status != expected_skipped_boundary:
            errors.append(f"{label}: skipped run must have an empty update trace")
    elif artifact is None:
        errors.append(f"{label}: executed run lacks update-trace artifact")
    else:
        errors.extend(_validate_artifact(artifact, root, f"{label}.artifact"))
    if outcome == "completed" and isinstance(expected_points, list) \
            and points != expected_points:
        errors.append(f"{label}: completed run did not observe the full schedule")
    return errors


def _validate_sample(
    value: Any,
    root: Path,
    label: str,
    contract: Mapping[str, Any] | None,
    campaign: Mapping[str, Any],
) -> list[str]:
    expected = {
        "role", "index", "attempt_index", "included_in_timing", "campaign",
        "process_identity", "outcome",
        "timing_s", "peak_memory_bytes",
        "validated_extent", "accepted_steps", "rejected_steps",
        "controller_updates", "controller_update_trace", "nn_calls",
        "partition_coverage", "failure", "artifact",
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
    if isinstance(value["attempt_index"], bool) \
            or not isinstance(value["attempt_index"], int) \
            or value["attempt_index"] < 0:
        errors.append(f"{label}.attempt_index: expected a non-negative integer")
    if not isinstance(value["included_in_timing"], bool):
        errors.append(f"{label}.included_in_timing: expected Boolean")
    if role == "diagnostic" and value["included_in_timing"] is not False:
        errors.append(f"{label}: diagnostic attempt cannot enter timing statistics")
    campaign_value = value["campaign"]
    campaign_errors = _exact_keys(
        campaign_value, {"campaign_id", "round_index", "sequence_position"},
        f"{label}.campaign",
    )
    errors.extend(campaign_errors)
    if not campaign_errors:
        if campaign_value["campaign_id"] != campaign.get("campaign_id"):
            errors.append(f"{label}.campaign_id: disagrees with comparison campaign")
        for name in ("round_index", "sequence_position"):
            item = campaign_value[name]
            if item is not None and not _is_non_negative_int(item):
                errors.append(f"{label}.campaign.{name}: invalid index")
        if _mode_is(role, "cold", "steady") and (
            campaign_value["round_index"] is None
            or campaign_value["sequence_position"] is None
        ):
            errors.append(f"{label}.campaign: formal sample lacks rotation position")
    process = value["process_identity"]
    process_errors = _exact_keys(
        process, {"invocation_id", "pid", "started_at_utc", "finished_at_utc"},
        f"{label}.process_identity",
    )
    errors.extend(process_errors)
    if not process_errors:
        if not isinstance(process["invocation_id"], str) \
                or not process["invocation_id"].strip():
            errors.append(f"{label}.process_identity.invocation_id: expected text")
        if not isinstance(process["pid"], int) or isinstance(process["pid"], bool) \
                or process["pid"] <= 0:
            errors.append(f"{label}.process_identity.pid: expected a positive integer")
        started = _parse_utc(process["started_at_utc"])
        finished = _parse_utc(process["finished_at_utc"])
        if started is None:
            errors.append(
                f"{label}.process_identity.started_at_utc: expected UTC timestamp"
            )
        if finished is None:
            errors.append(
                f"{label}.process_identity.finished_at_utc: expected UTC timestamp"
            )
        if started is not None and finished is not None and finished <= started:
            errors.append(
                f"{label}.process_identity: finish must follow process start"
            )
    outcome = value["outcome"]
    if not isinstance(outcome, str) or outcome not in CANONICAL_OUTCOMES:
        errors.append(f"{label}.outcome: invalid canonical outcome {outcome!r}")
    timing_keys = TIMING_FIELDS
    timing = value["timing_s"]
    timing_errors = _exact_keys(timing, timing_keys, f"{label}.timing_s")
    errors.extend(timing_errors)
    if not timing_errors:
        for name, seconds in timing.items():
            if seconds is not None and not _is_finite_number(
                seconds, positive=name == "process_total"
            ):
                errors.append(f"{label}.timing_s.{name}: invalid duration")
        if not _is_finite_number(timing["process_total"], positive=True):
            errors.append(f"{label}: executed sample lacks positive process_total")
        elif not process_errors:
            started = _parse_utc(process["started_at_utc"])
            finished = _parse_utc(process["finished_at_utc"])
            if started is not None and finished is not None:
                receipt_wall = (finished - started).total_seconds()
                tolerance = max(1.0, 0.02 * float(timing["process_total"]))
                if not math.isclose(
                    receipt_wall, float(timing["process_total"]),
                    rel_tol=0.0, abs_tol=tolerance,
                ):
                    errors.append(
                        f"{label}: process receipt wall disagrees with process_total"
                    )
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
    for name in (
        "accepted_steps", "rejected_steps", "controller_updates", "nn_calls"
    ):
        count = value[name]
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            errors.append(f"{label}.{name}: expected a non-negative integer")
    errors.extend(_validate_partition_coverage(
        value["partition_coverage"], root, f"{label}.partition_coverage",
        contract, outcome=outcome, ledger_scope="attempt",
        invocation_id=(
            process.get("invocation_id") if isinstance(process, Mapping) else None
        ),
    ))
    errors.extend(_validate_controller_update_trace(
        value["controller_update_trace"], root,
        f"{label}.controller_update_trace", contract,
        observed_count=value["controller_updates"],
        validated_extent=value["validated_extent"], outcome=outcome,
    ))
    failure = value["failure"]
    if outcome == "completed" and failure is not None:
        errors.append(f"{label}: completed sample carries a failure")
    if outcome != "completed" and not isinstance(failure, dict):
        errors.append(f"{label}: non-completed sample lacks first-failure data")
    elif outcome != "completed":
        errors.extend(_validate_failure(failure, f"{label}.failure"))
    errors.extend(_validate_artifact(value["artifact"], root, f"{label}.artifact"))
    return errors


def _validate_coordinate_widths(
    rows: Any, label: str, coordinates: list[str]
) -> list[str]:
    errors: list[str] = []
    if not isinstance(rows, list) or [
        row.get("coordinate") if isinstance(row, dict) else None for row in rows
    ] != coordinates:
        errors.append(f"{label}: coordinate order does not match")
    else:
        for index, row in enumerate(rows):
            row_label = f"{label}[{index}]"
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
    errors.extend(_validate_coordinate_widths(
        value["per_coordinate"], f"{label}.per_coordinate", coordinates
    ))
    if value["artifact"] is None:
        errors.append(f"{label}: complete width lacks artifact")
    else:
        errors.extend(_validate_artifact(value["artifact"], root, f"{label}.artifact"))
    return errors


def _validate_widths(value: Any, root: Path, label: str) -> list[str]:
    expected = {
        "status", "validated_prefix", "coordinate_order", "coordinate_units",
        "aggregation_semantics", "endpoint", "last_segment_tube",
        "full_horizon_tube", "series", "trajectory_artifact",
    }
    errors = _exact_keys(value, expected, label)
    if errors:
        return errors
    status = value["status"]
    if not isinstance(status, str) \
            or status not in {"complete", "partial", "unavailable"}:
        errors.append(f"{label}.status: invalid width status {status!r}")
    errors.extend(_validate_extent(
        value["validated_prefix"], f"{label}.validated_prefix"
    ))
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
    series = value["series"]
    if not isinstance(series, list):
        errors.append(f"{label}.series: expected an array")
    else:
        previous: tuple[str, float] | None = None
        for index, observation in enumerate(series):
            observation_label = f"{label}.series[{index}]"
            observation_errors = _exact_keys(
                observation, {"extent", "per_coordinate"}, observation_label
            )
            errors.extend(observation_errors)
            if observation_errors:
                continue
            extent = observation["extent"]
            errors.extend(_validate_extent(
                extent, f"{observation_label}.extent"
            ))
            errors.extend(_validate_coordinate_widths(
                observation["per_coordinate"],
                f"{observation_label}.per_coordinate", coordinates,
            ))
            if isinstance(extent, Mapping) and _is_finite_number(
                extent.get("value")
            ):
                current = (str(extent.get("kind")), float(extent["value"]))
                if previous is not None and (
                    current[0] != previous[0] or current[1] <= previous[1]
                ):
                    errors.append(
                        f"{observation_label}.extent: series must be strictly ordered"
                    )
                previous = current
    if status == "complete":
        if any(value[name].get("status") != "complete" for name in (
            "endpoint", "last_segment_tube", "full_horizon_tube"
        ) if isinstance(value[name], dict)):
            errors.append(f"{label}: complete widths require all three complete views")
        if value["trajectory_artifact"] is None:
            errors.append(f"{label}: complete widths lack trajectory artifact")
        if not series:
            errors.append(f"{label}: complete widths lack time-series observations")
    if status == "partial" and not series and not any(
        isinstance(value[name], Mapping)
        and value[name].get("status") == "complete"
        for name in ("endpoint", "last_segment_tube", "full_horizon_tube")
    ):
        errors.append(f"{label}: partial widths contain no measured evidence")
    if status == "unavailable":
        if series != []:
            errors.append(f"{label}: unavailable widths must have an empty series")
        if any(
            isinstance(value[name], Mapping)
            and value[name].get("status") != "unavailable"
            for name in ("endpoint", "last_segment_tube", "full_horizon_tube")
        ) or value["trajectory_artifact"] is not None:
            errors.append(
                f"{label}: unavailable widths cannot carry measured evidence"
            )
    if isinstance(series, list) and series and value["trajectory_artifact"] is None:
        errors.append(f"{label}: width series lacks a trajectory artifact")
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


def _validate_attempt_ledger(
    link: Any,
    samples: list[Any],
    root: Path,
    label: str,
    *,
    campaign: Mapping[str, Any],
    instance_id: str,
    method: str,
) -> list[str]:
    errors = _validate_artifact(link, root, label)
    path, _ = _bound_file(
        root,
        link.get("path") if isinstance(link, Mapping) else None,
        link.get("sha256") if isinstance(link, Mapping) else None,
        label,
    )
    if path is None or errors:
        return errors
    try:
        ledger = _load(path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return [*errors, f"{label}: cannot load: {error}"]
    fields = {
        "schema_version", "campaign_id", "instance_id", "method", "entries",
        "head_sha256",
    }
    shape = _exact_keys(ledger, fields, label)
    errors.extend(shape)
    if shape:
        return errors
    if ledger["schema_version"] != ATTEMPT_LEDGER_SCHEMA:
        errors.append(f"{label}: wrong schema_version")
    if (
        ledger["campaign_id"] != campaign.get("campaign_id")
        or ledger["instance_id"] != instance_id
        or ledger["method"] != method
    ):
        errors.append(f"{label}: campaign/cell identity mismatch")
    entries = ledger["entries"]
    if not isinstance(entries, list):
        errors.append(f"{label}.entries: expected an array")
        return errors
    entry_fields = {
        "sequence", "previous_entry_sha256", "role", "index", "attempt_index",
        "included_in_timing", "invocation_id", "outcome", "artifact",
        "prelaunch_audit",
    }
    ledger_by_invocation: dict[str, Mapping[str, Any]] = {}
    previous_hash: str | None = None
    for index, entry in enumerate(entries):
        entry_label = f"{label}.entries[{index}]"
        entry_shape = _exact_keys(entry, entry_fields, entry_label)
        errors.extend(entry_shape)
        if entry_shape:
            continue
        if entry["sequence"] != index:
            errors.append(f"{entry_label}: non-contiguous sequence")
        if entry["previous_entry_sha256"] != previous_hash:
            errors.append(f"{entry_label}: hash-chain predecessor mismatch")
        if entry["role"] not in {"cold", "steady", "diagnostic"}:
            errors.append(f"{entry_label}: invalid role")
        for name in ("index", "attempt_index"):
            if not _is_non_negative_int(entry[name]):
                errors.append(f"{entry_label}.{name}: invalid index")
        if not isinstance(entry["included_in_timing"], bool):
            errors.append(f"{entry_label}.included_in_timing: expected Boolean")
        if entry["role"] == "diagnostic" and entry["included_in_timing"] is not False:
            errors.append(f"{entry_label}: diagnostic cannot enter timing statistics")
        invocation = entry["invocation_id"]
        if not isinstance(invocation, str) or not invocation.strip():
            errors.append(f"{entry_label}: invalid invocation_id")
        elif invocation in ledger_by_invocation:
            errors.append(f"{entry_label}: duplicate invocation_id")
        else:
            ledger_by_invocation[invocation] = entry
        if entry["outcome"] not in CANONICAL_OUTCOMES:
            errors.append(f"{entry_label}: invalid outcome")
        errors.extend(_validate_artifact(
            entry["artifact"], root, f"{entry_label}.artifact"
        ))
        errors.extend(_validate_artifact(
            entry["prelaunch_audit"], root, f"{entry_label}.prelaunch_audit"
        ))
        previous_hash = _canonical_sha256(entry)
    if ledger["head_sha256"] != previous_hash:
        errors.append(f"{label}: head_sha256 does not bind the final entry")

    samples_by_invocation: dict[str, Mapping[str, Any]] = {}
    for sample in samples:
        if not isinstance(sample, Mapping):
            continue
        invocation = _dotted(sample, "process_identity.invocation_id")
        if isinstance(invocation, str):
            samples_by_invocation[invocation] = sample
    if set(samples_by_invocation) != set(ledger_by_invocation):
        errors.append(f"{label}: entries do not exactly match result samples")
    for invocation in set(samples_by_invocation) & set(ledger_by_invocation):
        sample = samples_by_invocation[invocation]
        entry = ledger_by_invocation[invocation]
        expected = {
            "role": sample.get("role"),
            "index": sample.get("index"),
            "attempt_index": sample.get("attempt_index"),
            "included_in_timing": sample.get("included_in_timing"),
            "outcome": sample.get("outcome"),
            "artifact": sample.get("artifact"),
        }
        if any(entry.get(name) != value for name, value in expected.items()):
            errors.append(
                f"{label}: entry for invocation {invocation!r} disagrees with sample"
            )
        started = _parse_utc(_dotted(sample, "process_identity.started_at_utc"))
        if started is not None:
            audit_errors = _validate_prelaunch_audit(
                {
                    "campaign_id": campaign.get("campaign_id"),
                    "prelaunch_audit": entry.get("prelaunch_audit"),
                },
                root,
                require_fresh=True,
                now_utc=started,
            )
            errors.extend(
                f"{label}: prelaunch audit for invocation {invocation!r}: {error}"
                for error in audit_errors
            )
    attempts_by_slot: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        role = entry.get("role")
        slot_index = entry.get("index")
        attempt_index = entry.get("attempt_index")
        if isinstance(role, str) and _is_non_negative_int(slot_index) \
                and _is_non_negative_int(attempt_index):
            attempts_by_slot.setdefault((role, slot_index), []).append(entry)
    for slot, attempts in attempts_by_slot.items():
        ordered = sorted(attempts, key=lambda item: item["attempt_index"])
        actual_indices = [item["attempt_index"] for item in ordered]
        if actual_indices != list(range(len(ordered))):
            errors.append(
                f"{label}: slot {slot!r} attempt indices are not contiguous from zero"
            )
        if slot[0] not in {"cold", "steady"}:
            continue
        completed = [item for item in ordered if item.get("outcome") == "completed"]
        included = [
            item for item in ordered if item.get("included_in_timing") is True
        ]
        if len(completed) > 1:
            errors.append(
                f"{label}: slot {slot!r} has more than one completed formal attempt"
            )
        if completed:
            winner = completed[0]
            if winner is not ordered[-1]:
                errors.append(
                    f"{label}: slot {slot!r} has attempts after its first completion"
                )
            if included != [winner]:
                errors.append(
                    f"{label}: slot {slot!r} must include its sole first completed "
                    "attempt and no other attempt"
                )
        elif included:
            errors.append(
                f"{label}: slot {slot!r} includes timing without a completed attempt"
            )
    previous_finished: datetime | None = None
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            continue
        sample = samples_by_invocation.get(entry.get("invocation_id"))
        started = _parse_utc(_dotted(sample, "process_identity.started_at_utc")) \
            if isinstance(sample, Mapping) else None
        finished = _parse_utc(_dotted(sample, "process_identity.finished_at_utc")) \
            if isinstance(sample, Mapping) else None
        if started is not None and previous_finished is not None \
                and started < previous_finished:
            errors.append(
                f"{label}.entries[{index}]: append order disagrees with process time"
            )
        if finished is not None:
            previous_finished = finished
    return errors


def _validate_result_record(
    record: Mapping[str, Any],
    root: Path,
    prefix: str,
    instance_row: Mapping[str, Any],
    method: str,
    cell: Mapping[str, Any],
    campaign: Mapping[str, Any],
) -> list[str]:
    expected = {
        "schema_version", "instance_id", "method", "contract_identity",
        "measurement_plan", "run", "property", "eligibility", "samples",
        "widths", "artifacts", "attempt_ledger",
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
        identity, {
            "instance_contract_sha256", "cell_plan_sha256",
            "campaign_configuration_sha256",
        },
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
        if identity["campaign_configuration_sha256"] \
                != campaign_configuration_sha256(campaign):
            errors.append(
                f"{prefix}.result_record: campaign configuration identity mismatch"
            )
    if record["measurement_plan"] != cell["measurement_plan"]:
        errors.append(f"{prefix}.result_record: measurement plan mismatch")

    contract = _resolved_contract_record(instance_row, root)
    expected_extent = _contract_extent(contract) if contract is not None else None
    contract_variables = (
        _dotted(contract, "fields.variable_order") if contract is not None else None
    )
    contract_units = (
        _dotted(contract, "fields.width_comparison.coordinate_units")
        if contract is not None else None
    )
    contract_width_points = (
        _dotted(contract, "fields.width_comparison.sample_points")
        if contract is not None else None
    )
    contract_aggregation = (
        _dotted(contract, "fields.width_comparison.aggregation_semantics")
        if contract is not None else None
    )
    expected_accepted, expected_nn_calls = (
        _contract_work_counts(contract, cell)
        if contract is not None else (None, None)
    )
    expected_controller_updates = (
        _dotted(contract, "fields.controller_update.scheduled_updates")
        if contract is not None else None
    )

    run_value = record["run"]
    run = run_value if isinstance(run_value, dict) else {}
    run_keys = {
        "status", "outcome", "requested_extent", "validated_extent",
        "requested_horizon_completed", "accepted_steps", "rejected_steps",
        "controller_updates", "controller_update_trace", "nn_calls",
        "partition_coverage", "first_failure",
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
        for name in (
            "accepted_steps", "rejected_steps", "controller_updates", "nn_calls"
        ):
            count = run[name]
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                errors.append(f"{prefix}.result_record.run.{name}: invalid count")
        errors.extend(_validate_partition_coverage(
            run["partition_coverage"], root,
            f"{prefix}.result_record.run.partition_coverage", contract,
            outcome=run["status"], ledger_scope="run_summary", invocation_id=None,
        ))
        errors.extend(_validate_controller_update_trace(
            run["controller_update_trace"], root,
            f"{prefix}.result_record.run.controller_update_trace", contract,
            observed_count=run["controller_updates"],
            validated_extent=run["validated_extent"], outcome=run["status"],
        ))
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
            if not _is_non_negative_int(expected_controller_updates) \
                    or run["controller_updates"] != expected_controller_updates:
                errors.append(
                    f"{prefix}.result_record: controller updates do not match contract"
                )
        elif not _outcome_matches_category(
            run["outcome"], cell["run"]["failure_category"]
        ):
            errors.append(f"{prefix}.result_record: outcome/failure category mismatch")
        if run["status"] != "completed" and run["requested_horizon_completed"] is not False:
            errors.append(
                f"{prefix}.result_record: non-completed run claims completed horizon"
            )
        if run["status"] == "skipped" and any(
            run[name] != 0 for name in (
                "accepted_steps", "rejected_steps", "controller_updates", "nn_calls"
            )
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
            if _is_non_negative_int(expected_controller_updates) \
                    and isinstance(run["controller_updates"], int) \
                    and not isinstance(run["controller_updates"], bool) \
                    and run["controller_updates"] > expected_controller_updates:
                errors.append(
                    f"{prefix}.result_record: controller updates exceed contract"
                )
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
        if run.get("status") == "early_stopped":
            outcome = run.get("outcome")
            expected_property = (
                "passed" if outcome == "property_early_stop_pass"
                else "failed" if outcome == "property_early_stop_fail"
                else None
            )
            policy = _dotted(cell, "property_checker.early_stop_policy")
            allowed_policies = {
                "property_early_stop_pass": {"on_pass", "on_decisive"},
                "property_early_stop_fail": {"on_fail", "on_decisive"},
            }.get(outcome, set())
            requested = run.get("requested_extent")
            validated = run.get("validated_extent")
            strict_prefix = (
                isinstance(requested, Mapping)
                and isinstance(validated, Mapping)
                and requested.get("kind") == validated.get("kind")
                and _is_finite_number(requested.get("value"), positive=True)
                and _is_finite_number(validated.get("value"))
                and validated["value"] < requested["value"]
            )
            if policy not in allowed_policies:
                errors.append(
                    f"{prefix}.result_record: early-stop outcome is forbidden by "
                    "the configured policy"
                )
            if not strict_prefix:
                errors.append(
                    f"{prefix}.result_record: early stop must end before requested extent"
                )
            if (
                checker_mode != "configured"
                or property_value["status"] != expected_property
                or property_value["certificate_status"] != "passed"
                or property_value["checker"] != _dotted(
                    cell, "property_checker.identity"
                )
                or not _nonempty(property_value["certificate_semantics"])
                or property_value["artifact"] is None
            ):
                errors.append(
                    f"{prefix}.result_record: early stop lacks matching property "
                    "and certificate evidence"
                )

    eligibility_raw = record["eligibility"]
    eligibility = eligibility_raw if isinstance(eligibility_raw, dict) else {}
    eligibility_keys = {
        "mathematical_contract_known", "requested_horizon_completed",
        "certificate_semantics_passed", "finite_outputs",
        "numerical_soundness_class", "soundness_scope", "formal_claim_eligible",
        "performance_measurement_eligible",
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
    sample_keys: set[tuple[Any, Any, Any]] = set()
    invocation_ids: set[str] = set()
    formal_artifact_paths: set[str] = set()
    formal_artifact_shas: set[str] = set()
    schedule_entries, _ = _campaign_schedule_entries(campaign, root)
    for index, sample in enumerate(samples):
        errors.extend(_validate_sample(
            sample, root, f"{prefix}.result_record.samples[{index}]",
            contract, campaign,
        ))
        if isinstance(sample, dict):
            key = (
                sample.get("role"), sample.get("index"),
                sample.get("attempt_index"),
            )
            if isinstance(key[0], str) and _is_non_negative_int(key[1]) \
                    and _is_non_negative_int(key[2]):
                if key in sample_keys:
                    errors.append(
                        f"{prefix}.result_record.samples: duplicate "
                        f"role/index/attempt_index {key!r}"
                    )
                sample_keys.add(key)
            process = sample.get("process_identity")
            invocation_id = (
                process.get("invocation_id")
                if isinstance(process, Mapping) else None
            )
            if isinstance(invocation_id, str):
                if invocation_id in invocation_ids:
                    errors.append(
                        f"{prefix}.result_record.samples: duplicate invocation_id "
                        f"{invocation_id!r}"
                    )
                invocation_ids.add(invocation_id)
            if _mode_is(key[0], "cold", "steady") \
                    and _is_non_negative_int(key[1]):
                scheduled = schedule_entries.get((
                    instance_row["id"], method, key[0], key[1]
                ))
                campaign_value = sample.get("campaign")
                if not isinstance(scheduled, Mapping) or not isinstance(
                    campaign_value, Mapping
                ) or campaign_value.get("round_index") != scheduled.get(
                    "round_index"
                ) or campaign_value.get("sequence_position") != scheduled.get(
                    "sequence_position"
                ):
                    errors.append(
                        f"{prefix}.result_record.samples[{index}]: campaign "
                        "position disagrees with the frozen rotation schedule"
                    )
                artifact = sample.get("artifact")
                path = artifact.get("path") if isinstance(artifact, Mapping) else None
                sha = artifact.get("sha256") if isinstance(artifact, Mapping) else None
                if isinstance(path, str):
                    if path in formal_artifact_paths:
                        errors.append(
                            f"{prefix}.result_record.samples: formal samples reuse "
                            "an artifact path"
                        )
                    formal_artifact_paths.add(path)
                if isinstance(sha, str):
                    if sha in formal_artifact_shas:
                        errors.append(
                            f"{prefix}.result_record.samples: formal samples reuse "
                            "artifact bytes"
                        )
                    formal_artifact_shas.add(sha)
    errors.extend(_validate_attempt_ledger(
        record["attempt_ledger"], samples, root,
        f"{prefix}.result_record.attempt_ledger",
        campaign=campaign,
        instance_id=instance_row["id"], method=method,
    ))
    plan = cell["measurement_plan"]
    formal_slots = {
        (key[0], key[1]) for key in sample_keys
        if key[0] in {"cold", "steady"}
    }
    included_samples = [
        sample for sample in samples
        if isinstance(sample, Mapping)
        and sample.get("included_in_timing") is True
    ]
    included_slots = {
        (sample.get("role"), sample.get("index"))
        for sample in included_samples
        if _mode_is(sample.get("role"), "cold", "steady")
    }
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
        key for key in formal_slots
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
        if _is_non_negative_int(expected_controller_updates) \
                and isinstance(sample.get("controller_updates"), int) \
                and not isinstance(sample.get("controller_updates"), bool) \
                and sample["controller_updates"] > expected_controller_updates:
            errors.append(f"{label}.controller_updates: exceeds contract")
        if sample.get("outcome") == "completed":
            if sample_extent != run.get("requested_extent"):
                errors.append(f"{label}: completed sample has partial extent")
            if expected_accepted is not None \
                    and sample.get("accepted_steps") != expected_accepted:
                errors.append(f"{label}.accepted_steps: disagrees with cell plan")
            if expected_nn_calls is not None \
                    and sample.get("nn_calls") != expected_nn_calls:
                errors.append(f"{label}.nn_calls: disagrees with cell plan")
            if _is_non_negative_int(expected_controller_updates) \
                    and sample.get("controller_updates") \
                    != expected_controller_updates:
                errors.append(
                    f"{label}.controller_updates: disagrees with contract"
                )
        elif isinstance(sample.get("failure"), Mapping) \
                and sample["failure"].get("reason_code") != sample.get("outcome"):
            errors.append(f"{label}: failure reason disagrees with outcome")
    if run.get("status") == "completed":
        if included_slots != expected_keys or len(included_samples) != len(expected_keys):
            errors.append(
                f"{prefix}.result_record: completed run lacks exactly one included "
                "sample for every planned slot"
            )
        if any(
            not isinstance(sample, dict) or sample.get("outcome") != "completed"
            for sample in included_samples
        ):
            errors.append(
                f"{prefix}.result_record: included timing sample is not completed"
            )
        for index, sample in enumerate(samples):
            if not isinstance(sample, dict):
                continue
            key = (sample.get("role"), sample.get("index"))
            if not isinstance(key[0], str) or not isinstance(key[1], int) \
                    or isinstance(key[1], bool) or key not in expected_keys \
                    or sample.get("included_in_timing") is not True:
                continue
            label = f"{prefix}.result_record.samples[{index}]"
            if sample.get("validated_extent") != run.get("requested_extent"):
                errors.append(f"{label}: extent disagrees with completed run")
            for name in (
                "accepted_steps", "rejected_steps", "controller_updates", "nn_calls"
            ):
                if sample.get(name) != run.get(name):
                    errors.append(f"{label}.{name}: disagrees with run summary")
            sample_coverage = sample.get("partition_coverage")
            run_coverage = run.get("partition_coverage")
            coverage_fields = (
                "requested_partitions", "completed_partitions",
                "failed_partitions", "unattempted_partitions", "boxes_sha256",
            )
            if not isinstance(sample_coverage, Mapping) or not isinstance(
                run_coverage, Mapping
            ) or any(
                sample_coverage.get(name) != run_coverage.get(name)
                for name in coverage_fields
            ):
                errors.append(
                    f"{label}.partition_coverage: disagrees with run summary"
                )
            sample_trace = sample.get("controller_update_trace")
            run_trace = run.get("controller_update_trace")
            if not isinstance(sample_trace, Mapping) or not isinstance(
                run_trace, Mapping
            ) or (
                sample_trace.get("kind"), sample_trace.get("points"),
                sample_trace.get("boundary_update_status"),
            ) != (
                run_trace.get("kind"), run_trace.get("points"),
                run_trace.get("boundary_update_status"),
            ):
                errors.append(
                    f"{label}.controller_update_trace: disagrees with run summary"
                )
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
            and sample.get("controller_updates") == run.get("controller_updates")
            and isinstance(sample.get("controller_update_trace"), Mapping)
            and isinstance(run.get("controller_update_trace"), Mapping)
            and sample["controller_update_trace"].get("kind")
            == run["controller_update_trace"].get("kind")
            and sample["controller_update_trace"].get("points")
            == run["controller_update_trace"].get("points")
            and sample["controller_update_trace"].get("boundary_update_status")
            == run["controller_update_trace"].get("boundary_update_status")
            and sample.get("nn_calls") == run.get("nn_calls")
            and isinstance(sample.get("partition_coverage"), Mapping)
            and isinstance(run.get("partition_coverage"), Mapping)
            and all(
                sample["partition_coverage"].get(name)
                == run["partition_coverage"].get(name)
                for name in (
                    "requested_partitions", "completed_partitions",
                    "failed_partitions", "unattempted_partitions", "boxes_sha256",
                )
            )
            and sample.get("failure") == run.get("first_failure")
            for sample in samples
        )
        if not matching_failure:
            errors.append(
                f"{prefix}.result_record: terminal failure has no matching sample"
            )
    else:
        errors.append(
            f"{prefix}.result_record: terminal non-completion lacks a raw sample"
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
        if widths.get("coordinate_units") != contract_units:
            errors.append(
                f"{prefix}.result_record.widths: coordinate units do not match contract"
            )
        if widths.get("aggregation_semantics") != contract_aggregation:
            errors.append(
                f"{prefix}.result_record.widths: aggregation does not match contract"
            )
        validated_prefix = widths.get("validated_prefix")
        if validated_prefix != run.get("validated_extent"):
            errors.append(
                f"{prefix}.result_record.widths: validated prefix disagrees with run"
            )
        if completed and (not isinstance(validated_prefix, dict) or not _is_finite_number(
            validated_prefix.get("value"), positive=True
        )):
            errors.append(
                f"{prefix}.result_record.widths: completed prefix must be positive"
            )
        series = widths.get("series")
        if widths.get("status") != "unavailable" \
                and isinstance(series, list) and isinstance(contract_width_points, list) \
                and isinstance(validated_prefix, Mapping):
            kind = validated_prefix.get("kind")
            end = validated_prefix.get("value")
            expected_series_extents = [
                {"kind": kind, "value": point}
                for point in contract_width_points
                if _is_finite_number(end) and point <= end
            ]
            actual_series_extents = [
                observation.get("extent")
                if isinstance(observation, Mapping) else None
                for observation in series
            ]
            if actual_series_extents != expected_series_extents:
                errors.append(
                    f"{prefix}.result_record.widths.series: does not cover the "
                    "contract grid through the validated prefix"
                )
        if not completed and isinstance(widths.get("endpoint"), Mapping) \
                and widths["endpoint"].get("status") != "unavailable":
            errors.append(
                f"{prefix}.result_record.widths.endpoint: full endpoint is only "
                "available after completed horizon"
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
        artifact_label = f"{prefix}.result_record.artifacts[{index}]"
        errors.extend(_validate_artifact(
            artifact, root, artifact_label,
            role_required=True,
        ))
        if isinstance(artifact, dict) and isinstance(artifact.get("role"), str):
            roles.add(artifact["role"])
            errors.extend(_validate_typed_plot_artifact(
                artifact, root, artifact_label
            ))
    if not {"command", "run_log", "result"} <= roles:
        errors.append(f"{prefix}.result_record: command/run_log/result artifacts are required")

    if isinstance(eligibility, dict) \
            and eligibility.get("performance_measurement_eligible") is True:
        performance_prerequisites = (
            run.get("status") == "completed",
            run.get("requested_horizon_completed") is True,
            eligibility.get("finite_outputs") is True,
            included_slots == expected_keys
            and len(included_samples) == len(expected_keys)
            if run.get("status") == "completed" else False,
            all(
                isinstance(sample, dict) and sample.get("outcome") == "completed"
                for sample in included_samples
            ),
        )
        if not all(performance_prerequisites):
            errors.append(
                f"{prefix}.result_record: performance eligibility lacks prerequisites"
            )
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
    elif scheduled_updates and not _is_finite_number(
        _dotted(controller_execution, "nn_calls.value"), positive=True
    ):
        reasons.append("controller_nn_calls_zero_for_scheduled_updates")
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
    for group in ("source_identity", "binary_identity"):
        for name, value in cell[group].items():
            if value is not None and not (
                isinstance(value, str) and value.strip()
            ):
                errors.append(f"{prefix}.{group}.{name}: expected null or text")
    for name, value in cell["arithmetic"].items():
        if value is not None and not (
            isinstance(value, str) and value.strip()
        ):
            errors.append(f"{prefix}.arithmetic.{name}: expected null or text")
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
    if _is_non_negative_int(scheduled_updates) and scheduled_updates > 0 \
            and _dotted(controller_execution, "nn_calls.mode") == "exact" \
            and not (
                _is_non_negative_int(_dotted(controller_execution, "nn_calls.value"))
                and _dotted(controller_execution, "nn_calls.value") > 0
            ):
        errors.append(
            f"{prefix}.controller_execution.nn_calls.value: must be positive "
            "when controller updates are scheduled"
        )
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
    for name in ("hardware", "gpu"):
        value = cell["runtime"][name]
        if value is not None and not (
            isinstance(value, str) and value.strip()
        ):
            errors.append(f"{prefix}.runtime.{name}: expected null or text")
    limits = cell["runtime"]["resource_limits"]
    if limits is not None and not isinstance(limits, dict):
        errors.append(f"{prefix}.runtime.resource_limits: expected null or object")
    elif isinstance(limits, dict):
        limit_errors = _exact_keys(
            limits, RESOURCE_LIMIT_FIELDS, f"{prefix}.runtime.resource_limits"
        )
        errors.extend(limit_errors)
        if not limit_errors:
            for name in ("exclusive_host", "exclusive_gpu_device"):
                if limits[name] is not True:
                    errors.append(
                        f"{prefix}.runtime.resource_limits.{name}: must be true"
                    )
            if not _is_non_negative_int(limits["max_host_memory_bytes"]) \
                    or limits["max_host_memory_bytes"] <= 0:
                errors.append(
                    f"{prefix}.runtime.resource_limits.max_host_memory_bytes: "
                    "expected a positive integer"
                )
            device_memory = limits["max_device_memory_bytes"]
            if device_memory is not None and not _is_non_negative_int(device_memory):
                errors.append(
                    f"{prefix}.runtime.resource_limits.max_device_memory_bytes: "
                    "expected null or a non-negative integer"
                )
    timeout = cell["runtime"]["timeout_s"]
    if timeout is not None and not _is_finite_number(timeout, positive=True):
        errors.append(f"{prefix}.runtime.timeout_s: expected a finite positive number")
    plan = cell["measurement_plan"]
    fixed_plan = {
        "cold_runs": 1,
        "target_steady_runs": 5,
        "fresh_process_per_run": True,
        "timing_boundary_version": "total_configuration_v2",
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
    manifest: Mapping[str, Any],
    matrix: Mapping[str, Any],
    *,
    root: Path = ROOT,
    now_utc: datetime | None = None,
) -> list[str]:
    errors: list[str] = []
    manifest_matrix = manifest.get("execution_matrix", {})
    if matrix.get("schema_version") != MATRIX_SCHEMA \
            or manifest_matrix.get("schema_version") != MATRIX_SCHEMA:
        errors.append("matrix schema_version does not match the v6 contract")
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
    errors.extend(_validate_campaign_structure(
        matrix.get("comparison_campaign"), root
    ))
    errors.extend(_validate_contracts(manifest, root))
    instance_ids = [row["id"] for row in manifest.get("instances", [])]
    instances_by_id = {
        row["id"]: row for row in manifest.get("instances", [])
        if isinstance(row, dict) and "id" in row
    }
    methods = list(manifest.get("methods", []))
    scope_errors: list[str] = []
    if instance_ids != list(EXPECTED_INSTANCE_IDS):
        scope_errors.append(
            "manifest instances do not match the frozen 16-instance ARCH-COMP scope"
        )
    if methods != list(EXPECTED_METHODS):
        scope_errors.append(
            "manifest methods do not match the frozen four-method comparison scope"
        )
    if scope_errors:
        return [*errors, *scope_errors]
    errors.extend(_validate_campaign_schedule(
        matrix.get("comparison_campaign", {}), root,
        list(EXPECTED_INSTANCE_IDS), list(EXPECTED_METHODS),
    ))
    if list(matrix.get("methods", [])) != methods:
        errors.append("matrix methods do not match manifest methods")
    if list(matrix.get("cells", {})) != instance_ids:
        errors.append("matrix instance order does not match manifest")
    defaults = matrix.get("cell_defaults")
    required = matrix.get("required_cell_fields")
    expected_required = list(CELL_DEFAULT_SHAPE)
    if not isinstance(required, list) or required != expected_required:
        errors.append("required_cell_fields do not match the v6 cell contract")
        return errors
    default_shape_errors = _shape_errors(
        defaults, CELL_DEFAULT_SHAPE, "cell_defaults"
    )
    if default_shape_errors:
        errors.extend(default_shape_errors)
        return errors

    enums = matrix.get("enums", {})
    campaign_invocations: set[str] = set()
    campaign_formal_paths: set[str] = set()
    campaign_formal_shas: set[str] = set()
    campaign_process_receipts: list[tuple[datetime, datetime, str]] = []
    campaign_rotation_receipts: dict[
        tuple[str, str, int], list[tuple[int, datetime, datetime, str]]
    ] = {}
    running_cells: list[tuple[str, str, Mapping[str, Any]]] = []
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
            if status == "running":
                running_cells.append((instance, method, resolved))
            if status == "not_started" and (category is not None or detail is not None):
                errors.append(f"{prefix}: not_started cell carries failure data")
            if status == "running" and (category is not None or detail is not None):
                errors.append(f"{prefix}: running cell carries terminal failure data")
            if status == "running" \
                    and manifest["execution_policy"]["experiments_paused"] is not False:
                errors.append(f"{prefix}: running cell is forbidden while experiments are paused")
            if isinstance(status, str) \
                    and status in {
                        "failed", "timeout", "interrupted", "early_stopped", "skipped"
                    } and (
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
            if status == "early_stopped" and category != "property_early_stop":
                errors.append(
                    f"{prefix}: early_stopped status must use property_early_stop category"
                )
            if category == "property_early_stop" and status != "early_stopped":
                errors.append(
                    f"{prefix}: property_early_stop category must use early_stopped status"
                )
            if resolved["support"]["status"] == "unsupported" \
                    and _mode_is(status, "running", "completed"):
                errors.append(f"{prefix}: unsupported cell cannot run or complete")
            if status == "skipped":
                if resolved["support"]["status"] != "unsupported" \
                        or not resolved["support"]["blockers"]:
                    errors.append(
                        f"{prefix}: skipped cell requires unsupported status and blockers"
                    )
            elif isinstance(status, str) and (
                status == "running" or status in TERMINAL_STATUSES
            ):
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
                plan_reasons.extend(_campaign_reasons(
                    matrix.get("comparison_campaign"), root, resolved
                ))
                if plan_reasons:
                    errors.append(
                        f"{prefix}: active/terminal cell lacks executable plan "
                        f"({', '.join(plan_reasons)})"
                    )
            link = resolved["result_record"]
            has_link = all(link[name] is not None for name in (
                "schema_version", "path", "sha256"
            ))
            if _mode_is(status, "not_started", "running") and has_link:
                errors.append(f"{prefix}: nonterminal cell carries a result record")
            if isinstance(status, str) and status in TERMINAL_STATUSES and not has_link:
                errors.append(f"{prefix}: terminal cell lacks a result record")
            if isinstance(status, str) and status in TERMINAL_STATUSES \
                    and instances_by_id[instance]["contract"].get("status") != "resolved":
                errors.append(f"{prefix}: terminal cell has unresolved instance contract")
            if status == "running" \
                    and instances_by_id[instance]["contract"].get("status") != "resolved":
                errors.append(f"{prefix}: running cell has unresolved instance contract")
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
                            resolved, matrix["comparison_campaign"],
                        ))
                        for sample in record.get("samples", []):
                            if not isinstance(sample, Mapping):
                                continue
                            process = sample.get("process_identity")
                            invocation = (
                                process.get("invocation_id")
                                if isinstance(process, Mapping) else None
                            )
                            if isinstance(invocation, str):
                                if invocation in campaign_invocations:
                                    errors.append(
                                        f"{prefix}.result_record: campaign reuses "
                                        f"invocation_id {invocation!r}"
                                    )
                                campaign_invocations.add(invocation)
                            if isinstance(process, Mapping):
                                started = _parse_utc(process.get("started_at_utc"))
                                finished = _parse_utc(process.get("finished_at_utc"))
                                if started is not None and finished is not None:
                                    campaign_process_receipts.append((
                                        started, finished,
                                        f"{prefix}:{sample.get('role')}/{sample.get('index')}",
                                    ))
                            if not _mode_is(
                                sample.get("role"), "cold", "steady"
                            ) or sample.get("included_in_timing") is not True:
                                continue
                            campaign_value = sample.get("campaign")
                            process_value = sample.get("process_identity")
                            if isinstance(campaign_value, Mapping) and isinstance(
                                process_value, Mapping
                            ):
                                round_index = campaign_value.get("round_index")
                                position = campaign_value.get("sequence_position")
                                started = _parse_utc(
                                    process_value.get("started_at_utc")
                                )
                                finished = _parse_utc(
                                    process_value.get("finished_at_utc")
                                )
                                if _is_non_negative_int(round_index) \
                                        and _is_non_negative_int(position) \
                                        and started is not None \
                                        and finished is not None:
                                    group = (
                                        instance, str(sample.get("role")), round_index,
                                    )
                                    campaign_rotation_receipts.setdefault(
                                        group, []
                                    ).append((
                                        position, started, finished, method,
                                    ))
                            artifact = sample.get("artifact")
                            if not isinstance(artifact, Mapping):
                                continue
                            for name, seen in (
                                ("path", campaign_formal_paths),
                                ("sha256", campaign_formal_shas),
                            ):
                                value = artifact.get(name)
                                if isinstance(value, str):
                                    if value in seen:
                                        errors.append(
                                            f"{prefix}.result_record: campaign formal "
                                            f"samples reuse artifact {name}"
                                        )
                                    seen.add(value)
    active_receipt, _ = _active_run_receipt(
        matrix.get("comparison_campaign", {}), root
    )
    if len(running_cells) > 1:
        errors.append(
            "comparison_campaign.resource_limits: exclusive_host permits at most "
            "one running cell"
        )
    if not running_cells and active_receipt is not None:
        errors.append(
            "comparison_campaign.active_run_receipt: present without a running cell"
        )
    if running_cells and active_receipt is None:
        errors.append(
            "comparison_campaign.active_run_receipt: running cell lacks an active "
            "lock receipt"
        )
    if len(running_cells) == 1 and active_receipt is not None:
        instance, method, running_cell = running_cells[0]
        if (
            active_receipt.get("instance_id") != instance
            or active_receipt.get("method") != method
        ):
            errors.append(
                "comparison_campaign.active_run_receipt: running cell identity mismatch"
            )
        started = _parse_utc(active_receipt.get("started_at_utc"))
        freshness_errors = _validate_prelaunch_audit(
            matrix.get("comparison_campaign", {}), root,
            require_fresh=True, now_utc=started or now_utc,
        )
        if freshness_errors:
            errors.append(
                "comparison_campaign.active_run_receipt: prelaunch audit was not "
                "fresh when the active process started"
            )
        current = now_utc or datetime.now(timezone.utc)
        if current.tzinfo is None:
            current = current.replace(tzinfo=timezone.utc)
        timeout = _dotted(running_cell, "runtime.timeout_s")
        if started is not None and (
            started > current
            or not _is_finite_number(timeout, positive=True)
            or current > started + timedelta(seconds=float(timeout) + 60.0)
        ):
            errors.append(
                "comparison_campaign.active_run_receipt: active receipt is outside "
                "the runtime timeout window"
            )
        if started is not None:
            overlapping = [
                identity
                for _, finished, identity in campaign_process_receipts
                if finished > started
            ]
            if overlapping:
                errors.append(
                    "comparison_campaign.active_run_receipt: active process "
                    "overlaps prior process receipts despite exclusive_host: "
                    f"{sorted(overlapping)!r}"
                )
        audit_link = _dotted(
            matrix.get("comparison_campaign", {}), "prelaunch_audit"
        )
        audit_path, _ = _bound_file(
            root,
            audit_link.get("path") if isinstance(audit_link, Mapping) else None,
            audit_link.get("sha256") if isinstance(audit_link, Mapping) else None,
            "comparison_campaign.prelaunch_audit",
        )
        if started is not None and audit_path is not None:
            try:
                audit_receipt = _load(audit_path)
            except (OSError, ValueError, json.JSONDecodeError):
                audit_receipt = {}
            checked_at = _parse_utc(audit_receipt.get("checked_at_utc"))
            valid_until = _parse_utc(audit_receipt.get("valid_until_utc"))
            if checked_at is None or valid_until is None \
                    or not checked_at < started < valid_until:
                errors.append(
                    "comparison_campaign.active_run_receipt: prelaunch audit did "
                    "not precede the active process"
                )
    for group, receipts in campaign_rotation_receipts.items():
        positions = [item[0] for item in receipts]
        if len(set(positions)) != len(positions):
            errors.append(
                f"comparison_campaign.rotation: duplicate actual position in {group!r}"
            )
            continue
        ordered = sorted(receipts)
        for previous, current in zip(ordered, ordered[1:]):
            if previous[2] > current[1]:
                errors.append(
                    "comparison_campaign.rotation: actual process receipts overlap "
                    f"or violate sequence in {group!r} between "
                    f"{previous[3]!r} and {current[3]!r}"
                )
    for instance in EXPECTED_INSTANCE_IDS:
        round_windows: list[tuple[int, datetime, datetime, str]] = []
        for (group_instance, role, round_index), receipts in (
            campaign_rotation_receipts.items()
        ):
            if group_instance != instance or not receipts:
                continue
            campaign_round = 0 if role == "cold" else round_index + 1
            round_windows.append((
                campaign_round,
                min(item[1] for item in receipts),
                max(item[2] for item in receipts),
                role,
            ))
        round_windows.sort()
        for previous, current in zip(round_windows, round_windows[1:]):
            if previous[2] > current[1]:
                errors.append(
                    "comparison_campaign.rotation: cold/steady rounds overlap "
                    f"or run out of order for {instance!r} between "
                    f"round {previous[0]} and {current[0]}"
                )
    limits = matrix.get("comparison_campaign", {}).get("resource_limits")
    if isinstance(limits, Mapping) and limits.get("exclusive_host") is True:
        ordered_receipts = sorted(campaign_process_receipts)
        for previous, current in zip(ordered_receipts, ordered_receipts[1:]):
            if previous[1] > current[0]:
                errors.append(
                    "comparison_campaign.resource_limits: process receipts overlap "
                    f"despite exclusive_host between {previous[2]!r} and "
                    f"{current[2]!r}"
                )
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
    now_utc: datetime | None = None,
) -> list[str]:
    reasons = validate_matrix(manifest, matrix, root=root, now_utc=now_utc)
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
    reasons.extend(_campaign_reasons(
        matrix.get("comparison_campaign"), root, cell,
        require_fresh_audit=True, now_utc=now_utc,
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
