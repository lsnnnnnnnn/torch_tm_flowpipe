"""Export honest box/affine-interval projections from saved observers.

The QUAD observer contract is ``bounds[B, state, 4]`` with columns
``tube_lo, tube_hi, endpoint_lo, endpoint_hi``.  This module deliberately
calls the resulting geometry a box projection: it does not claim the
coordinate correlation of Flow*'s octagon projection.  Plot-spec v3 can form
one declared affine coordinate from those boxes; it discloses that source
coordinate correlations remain unavailable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

SCHEMA = "torch-tm-flowpipe-projection-v2"
PLOT_SPEC_SCHEMA = "torch-tm-flowpipe-plot-spec-v1"
PLOT_SPEC_SCHEMA_V2 = "torch-tm-flowpipe-plot-spec-v2"
PLOT_SPEC_SCHEMA_V3 = "torch-tm-flowpipe-plot-spec-v3"
OBSERVER_RE = re.compile(r"observer_(\d+)\.pt$")
SHA256_RE = re.compile(r"[0-9a-f]{64}")
DERIVED_COORDINATE_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
VIEW_COLUMNS = {"tube": (0, 1), "endpoint": (2, 3)}
MAX_STATE_STATE_BOXES = 100_000
RESULT_SUMMARY_KEYS = (
    "status", "completed_step", "overall_horizon_completed",
    "all_steps_all_lanes_accepted", "first_failure", "accepted_lane_steps",
    "original_target_steps", "budget_steps", "common_completed_step",
    "requested_periods_completed", "full_T5_full1024_completed",
    "full_T5_root1_B2_completed", "original_full1024_completed",
    "fullbatch_qualification", "end_to_end_strict_certificate",
    "returncode", "solver_returncode", "process_s", "native_process_s",
    "supervisor_process_s",
)
RESULT_TIMING_KEYS = {"process_s", "native_process_s", "supervisor_process_s"}
REGION_ROLES = {"safe", "target", "unsafe", "informational"}
PROPERTY_QUANTIFIERS = {"all_times", "endpoint", "eventually", "conjunction"}
RESULT_IDENTITY_BOUND = {
    "verified_equal_to_observer_sidecars_direct",
    "verified_to_observer_sidecars_via_hashed_INPUT",
}
INPUT_IDENTITY_BOUND = {
    "verified_equal_to_observer_sidecars_direct",
    "verified_to_observer_sidecars_via_identity_bound_RESULT.input_sha256",
}


def _native_range_record(state_count: int) -> struct.Struct:
    return struct.Struct(f"<QQd{4 * state_count}d")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _step(path: Path) -> int:
    match = OBSERVER_RE.fullmatch(path.name)
    if not match:
        raise ValueError(f"not an observer_<step>.pt file: {path}")
    value = int(match.group(1))
    if value < 1:
        raise ValueError(f"observer step must be positive: {path}")
    return value


def discover_observers(root: Path) -> list[Path]:
    paths = sorted(root.glob("observer_*.pt"), key=_step)
    if not paths:
        raise ValueError(f"no observer_<step>.pt files in {root}")
    steps = [_step(path) for path in paths]
    if len(steps) != len(set(steps)):
        raise ValueError(f"duplicate observer steps in {root}")
    return paths


def _missing_ranges(observed: Iterable[int], expected_steps: int) -> list[list[int]]:
    present = set(observed)
    return _compact_ranges(
        step for step in range(1, expected_steps + 1) if step not in present
    )


def _compact_ranges(values: Iterable[int]) -> list[list[int]]:
    ordered = sorted(set(values))
    ranges: list[list[int]] = []
    for step in ordered:
        if not ranges or step != ranges[-1][1] + 1:
            ranges.append([step, step])
        else:
            ranges[-1][1] = step
    return ranges


def _expand_ranges(value: Any, label: str) -> list[int]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list of [start,end] ranges")
    steps: list[int] = []
    for interval in value:
        if (
            not isinstance(interval, list)
            or len(interval) != 2
            or any(not isinstance(step, int) or isinstance(step, bool) for step in interval)
            or interval[0] < 1
            or interval[1] < interval[0]
        ):
            raise ValueError(f"{label} contains an invalid range")
        steps.extend(range(interval[0], interval[1] + 1))
    if value != _compact_ranges(steps):
        raise ValueError(f"{label} must be sorted, disjoint, and compact")
    return steps


def _parse_display_steps(value: str | None, expected_steps: int) -> set[int] | None:
    if value is None or value == "all":
        return None
    selected: set[int] = set()
    for token in value.split(","):
        token = token.strip()
        if not token:
            raise ValueError("display step selection contains an empty item")
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start, end = int(start_text), int(end_text)
        else:
            start = end = int(token)
        if start < 1 or end < start or end > expected_steps:
            raise ValueError(f"invalid display step range {token!r}")
        selected.update(range(start, end + 1))
    if not selected:
        raise ValueError("display step selection is empty")
    return selected


def _canonical_affine_transform(
    name: str,
    transform: dict[str, Any],
    coordinate_names: list[str],
) -> dict[str, Any]:
    return {
        "kind": "affine",
        "name": name,
        "offset": float(transform["offset"]),
        "terms": [
            {
                "coordinate": coordinate,
                "index": coordinate_names.index(coordinate),
                "coefficient": float(transform["coefficients"][coordinate]),
            }
            for coordinate in coordinate_names
            if coordinate in transform["coefficients"]
        ],
    }


def _parse_projection(
    value: str,
    coordinate_names: list[str],
    derived_coordinates: dict[str, Any] | None = None,
) -> dict[str, Any]:
    derived_coordinates = derived_coordinates or {}
    axes = [part.strip() for part in value.split(",")]
    if len(axes) != 2 or any(not axis for axis in axes):
        raise ValueError("projection must contain exactly two coordinates, e.g. t,x3")
    time_axes = [index for index, axis in enumerate(axes) if axis in {"t", "time"}]
    if len(time_axes) > 1:
        raise ValueError("projection cannot use time on both axes")
    if time_axes:
        if time_axes != [0]:
            raise ValueError("time-state projection must use time as the first coordinate")
        if axes[1] not in coordinate_names and axes[1] not in derived_coordinates:
            raise ValueError(f"unknown state coordinate {axes[1]!r}")
        projection = {
            "kind": "time-state",
            "x": "t",
            "y": axes[1],
        }
        if axes[1] in derived_coordinates:
            projection["y_transform"] = _canonical_affine_transform(
                axes[1], derived_coordinates[axes[1]], coordinate_names
            )
        else:
            projection["y_index"] = coordinate_names.index(axes[1])
        return projection
    unknown = [
        axis for axis in axes
        if axis not in coordinate_names and axis not in derived_coordinates
    ]
    if unknown:
        raise ValueError(f"unknown state coordinates: {unknown}")
    if any(axis in derived_coordinates for axis in axes):
        raise ValueError(
            "affine derived coordinates are supported only as the y axis of "
            "a time-state projection"
        )
    if axes[0] == axes[1]:
        raise ValueError("state-state projection requires two distinct coordinates")
    return {
        "kind": "state-state",
        "x": axes[0],
        "y": axes[1],
        "x_index": coordinate_names.index(axes[0]),
        "y_index": coordinate_names.index(axes[1]),
    }


def _affine_interval(
    bounds: Any,
    lo_column: int,
    hi_column: int,
    transform: dict[str, Any],
) -> tuple[float, float]:
    """Return a binary64-outward interval image of one saved state box."""
    lower = float(transform["offset"])
    upper = float(transform["offset"])
    for term in transform["terms"]:
        coefficient = float(term["coefficient"])
        index = int(term["index"])
        raw_lo = float(bounds[index][lo_column])
        raw_hi = float(bounds[index][hi_column])
        selected_lo, selected_hi = (
            (raw_lo, raw_hi) if coefficient > 0.0 else (raw_hi, raw_lo)
        )
        product_lo = math.nextafter(coefficient * selected_lo, -math.inf)
        product_hi = math.nextafter(coefficient * selected_hi, math.inf)
        lower = math.nextafter(lower + product_lo, -math.inf)
        upper = math.nextafter(upper + product_hi, math.inf)
    if not math.isfinite(lower) or not math.isfinite(upper):
        raise ValueError("affine derived-coordinate interval is nonfinite")
    return lower, upper


def _load_observer(path: Path, state_count: int) -> tuple[Any, Any, Any]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - dependency error is environment-specific
        raise RuntimeError("exporting observer .pt files requires PyTorch") from exc
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as exc:  # pragma: no cover - only reached on unsupported old PyTorch
        raise RuntimeError("PyTorch with fail-closed weights_only loading is required") from exc
    if not isinstance(payload, dict) or not {"bounds", "accepted", "status"} <= payload.keys():
        raise ValueError(f"{path}: expected bounds, accepted, and status tensors")
    bounds, accepted, status = payload["bounds"], payload["accepted"], payload["status"]
    if bounds.dtype != torch.float64:
        raise ValueError(f"{path}: bounds must use torch.float64, got {bounds.dtype}")
    if bounds.ndim != 3 or tuple(bounds.shape[1:]) != (state_count, 4):
        raise ValueError(
            f"{path}: bounds shape must be [lanes,{state_count},4], got {tuple(bounds.shape)}"
        )
    lanes = bounds.shape[0]
    if lanes < 1:
        raise ValueError(f"{path}: observer must contain at least one lane")
    if tuple(accepted.shape) != (lanes,) or accepted.dtype != torch.bool:
        raise ValueError(f"{path}: accepted must be bool[{lanes}]")
    if tuple(status.shape) != (lanes,):
        raise ValueError(f"{path}: status must have shape [{lanes}]")
    if status.dtype != torch.int8:
        raise ValueError(f"{path}: status must use torch.int8, got {status.dtype}")
    chosen = bounds[accepted]
    if chosen.numel() and (
        not bool(torch.isfinite(chosen).all())
        or not bool((chosen[..., 0] <= chosen[..., 1]).all())
        or not bool((chosen[..., 2] <= chosen[..., 3]).all())
    ):
        raise ValueError(f"{path}: accepted bounds are nonfinite or unordered")
    return bounds, accepted, status


def _source_identity(path: Path, step: int) -> dict[str, Any]:
    digest = _sha256(path)
    sidecar_path = path.with_suffix(".json")
    identity: dict[str, Any] = {
        "source_file": path.name,
        "source_sha256": digest,
        "sidecar": None,
    }
    if not sidecar_path.is_file():
        return identity
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if sidecar.get("step") != step:
        raise ValueError(f"{sidecar_path}: sidecar step does not match filename")
    if sidecar.get("pt_sha256") != digest:
        raise ValueError(f"{sidecar_path}: pt_sha256 does not match {path.name}")
    identity["sidecar"] = {
        "source_file": sidecar_path.name,
        "source_sha256": _sha256(sidecar_path),
        "pt_sha256_verified": True,
        "source_identity_sha256": hashlib.sha256(
            json.dumps(
                sidecar.get("source_identity"), sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest(),
    }
    return identity


def _observer_series_identity(paths: list[Path]) -> dict[str, Any]:
    identities = []
    files = []
    for path in paths:
        sidecar_path = path.with_suffix(".json")
        digest = _sha256(path)
        entry: dict[str, Any] = {
            "step": _step(path),
            "source_file": path.name,
            "source_sha256": digest,
            "sidecar": None,
        }
        if sidecar_path.is_file():
            sidecar = _json_object(sidecar_path)
            if sidecar.get("step") != _step(path):
                raise ValueError(f"{sidecar_path}: sidecar step does not match filename")
            if sidecar.get("pt_sha256") != digest:
                raise ValueError(f"{sidecar_path}: pt_sha256 does not match {path.name}")
            entry["sidecar"] = {
                "source_file": sidecar_path.name,
                "source_sha256": _sha256(sidecar_path),
                "pt_sha256_verified": True,
                "source_identity_sha256": _identity_digest(
                    sidecar.get("source_identity")
                ),
            }
            if "source_identity" in sidecar:
                identity = _optional_source_identity(
                    sidecar, "source_identity", f"{sidecar_path}: source_identity"
                )
                identities.append(identity)
        files.append(entry)
    if not identities:
        return {
            "status": "unavailable",
            "source_identity": None,
            "sidecars_with_identity": 0,
            "observer_files": len(paths),
            "files": files,
        }
    canonical = json.dumps(identities[0], sort_keys=True, separators=(",", ":"))
    if any(
        json.dumps(identity, sort_keys=True, separators=(",", ":")) != canonical
        for identity in identities[1:]
    ):
        raise ValueError("observer sidecars do not share one source_identity")
    complete = len(identities) == len(paths)
    return {
        "status": (
            "verified_equal_across_all_observer_sidecars"
            if complete
            else "partial_sidecar_identity_coverage_unbound"
        ),
        "sidecars_with_identity": len(identities),
        "observer_files": len(paths),
        "source_identity_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "source_identity": identities[0] if complete else None,
        "files": files,
    }


def _json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _identity_digest(value: Any) -> str | None:
    if value is None:
        return None
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _optional_object(
    container: dict[str, Any], key: str, label: str
) -> dict[str, Any] | None:
    value = container.get(key)
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _optional_source_identity(
    container: dict[str, Any], key: str, label: str
) -> dict[str, Any] | None:
    if key not in container:
        return None
    value = container[key]
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    if not value:
        raise ValueError(f"{label} must not be empty")
    return value


def _one_declared_number(
    candidates: list[tuple[str, Any]], label: str, *, integer: bool = False
) -> int | float | None:
    present: list[tuple[str, int | float]] = []
    for source, value in candidates:
        if value is None:
            continue
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value <= 0
            or (integer and int(value) != value)
        ):
            raise ValueError(f"{source}: invalid declared {label} {value!r}")
        present.append((source, int(value) if integer else float(value)))
    if not present:
        return None
    first = present[0][1]
    if any(value != first for _, value in present[1:]):
        details = ", ".join(f"{source}={value!r}" for source, value in present)
        raise ValueError(f"adjacent run evidence disagrees on {label}: {details}")
    return first


def _result_summary(value: dict[str, Any], path: Path) -> dict[str, Any]:
    output = {key: value[key] for key in RESULT_SUMMARY_KEYS if key in value}
    for key in RESULT_TIMING_KEYS & output.keys():
        timing = output[key]
        if (
            not isinstance(timing, (int, float))
            or isinstance(timing, bool)
            or not math.isfinite(timing)
            or timing < 0
        ):
            raise ValueError(f"{path}: {key} must be a finite nonnegative number")
    return output


def _observer_declared_contract(
    input_value: dict[str, Any] | None,
    result_value: dict[str, Any] | None,
    binding: dict[str, Any],
) -> dict[str, Any]:
    input_value = input_value or {}
    result_value = result_value or {}
    input_identity = _optional_source_identity(
        input_value, "source_identity", "INPUT.json source_identity"
    ) or {}
    result_identity = _optional_source_identity(
        result_value, "source_identity", "RESULT.json source_identity"
    ) or {}
    input_settings = _optional_object(
        input_identity, "settings", "INPUT.json source_identity.settings"
    ) or {}
    input_config = _optional_object(
        input_identity, "resolved_config", "INPUT.json source_identity.resolved_config"
    ) or {}
    result_settings = _optional_object(
        result_identity, "settings", "RESULT.json source_identity.settings"
    ) or {}
    result_config = _optional_object(
        result_identity, "resolved_config", "RESULT.json source_identity.resolved_config"
    ) or {}

    def field(
        candidates: list[tuple[str, str, Any]], label: str, *, integer: bool = False
    ) -> tuple[int | float | None, dict[str, Any]]:
        value = _one_declared_number(
            [(source, declared) for source, _, declared in candidates],
            label,
            integer=integer,
        )
        present = [(source, artifact) for source, artifact, declared in candidates
                   if declared is not None]
        bound = [source for source, artifact in present if (
            artifact == "INPUT.json"
            and binding.get("input_series") in INPUT_IDENTITY_BOUND
        ) or (
            artifact == "RESULT.json"
            and binding.get("result_series") in RESULT_IDENTITY_BOUND
        )]
        return value, {
            "declaration_sources": [source for source, _ in present],
            "identity_bound_sources": bound,
            "series_binding": (
                "identity_bound_to_observer_series"
                if bound
                else (
                    "adjacent_declaration_unbound_to_observer_series"
                    if present
                    else "declaration_unavailable"
                )
            ),
        }

    expected_candidates = [
        ("INPUT.json budget_steps", "INPUT.json", input_value.get("budget_steps")),
        ("INPUT.json execution_budget_steps", "INPUT.json",
         input_value.get("execution_budget_steps")),
        ("INPUT.json source_identity.execution_budget_steps", "INPUT.json",
         input_identity.get("execution_budget_steps")),
        ("RESULT.json budget_steps", "RESULT.json", result_value.get("budget_steps")),
        ("RESULT.json execution_budget_steps", "RESULT.json",
         result_value.get("execution_budget_steps")),
        ("RESULT.json source_identity.execution_budget_steps", "RESULT.json",
         result_identity.get("execution_budget_steps")),
    ]
    expected_steps, expected_steps_provenance = field(
        expected_candidates, "expected steps", integer=True
    )
    if expected_steps is None:
        expected_steps, expected_steps_provenance = field([
            ("INPUT.json original_target_steps", "INPUT.json",
             input_value.get("original_target_steps")),
            ("RESULT.json original_target_steps", "RESULT.json",
             result_value.get("original_target_steps")),
        ], "expected steps", integer=True)
    step_size, step_size_provenance = field([
        ("INPUT.json source_identity.settings.step", "INPUT.json",
         input_settings.get("step")),
        ("INPUT.json source_identity.resolved_config.ode_step_size", "INPUT.json",
         input_config.get("ode_step_size")),
        ("RESULT.json source_identity.settings.step", "RESULT.json",
         result_settings.get("step")),
        ("RESULT.json source_identity.resolved_config.ode_step_size", "RESULT.json",
         result_config.get("ode_step_size")),
    ], "step size")
    expected_lanes, expected_lanes_provenance = field([
        ("INPUT.json batch_size", "INPUT.json", input_value.get("batch_size")),
        ("INPUT.json source_identity.batch_size", "INPUT.json",
         input_identity.get("batch_size")),
        ("RESULT.json batch_size", "RESULT.json", result_value.get("batch_size")),
        ("RESULT.json source_identity.batch_size", "RESULT.json",
         result_identity.get("batch_size")),
    ], "observer lane count", integer=True)
    return {
        "expected_steps": expected_steps,
        "step_size": step_size,
        "expected_lanes": expected_lanes,
        "provenance": {
            "expected_steps": expected_steps_provenance,
            "step_size": step_size_provenance,
            "expected_lanes": expected_lanes_provenance,
        },
    }


def _evidence_binding(
    input_path: Path,
    result_path: Path,
    input_value: dict[str, Any] | None,
    result_value: dict[str, Any] | None,
    observer_identity: dict[str, Any] | None,
) -> dict[str, Any]:
    input_result = "unavailable"
    if input_value is not None and result_value is not None:
        expected_hash = _sha256(input_path)
        declared_hash = result_value.get("input_sha256")
        if declared_hash is None:
            input_result = "RESULT.input_sha256_unavailable"
        elif declared_hash != expected_hash:
            raise ValueError(f"{result_path}: input_sha256 does not match INPUT.json")
        else:
            input_result = "verified_by_RESULT.input_sha256"

    input_identity = (
        None if input_value is None else _optional_source_identity(
            input_value, "source_identity", "INPUT.json source_identity"
        )
    )
    result_identity = (
        None if result_value is None else _optional_source_identity(
            result_value, "source_identity", "RESULT.json source_identity"
        )
    )
    identities = [identity for identity in (input_identity, result_identity)
                  if identity is not None]
    if identities:
        first_digest = _identity_digest(identities[0])
        if any(_identity_digest(identity) != first_digest for identity in identities[1:]):
            raise ValueError("adjacent INPUT.json and RESULT.json source_identity disagree")
    else:
        first_digest = None

    observer_value = None if observer_identity is None else observer_identity.get("source_identity")
    observer_digest = _identity_digest(observer_value)
    if observer_value is None:
        input_series = "adjacent_INPUT_unbound_to_series"
        result_series = "adjacent_RESULT_unbound_to_series"
    else:
        if input_identity is not None and _identity_digest(input_identity) != observer_digest:
            raise ValueError("adjacent INPUT.json source_identity does not match observer sidecars")
        if result_identity is not None and _identity_digest(result_identity) != observer_digest:
            raise ValueError("adjacent RESULT.json source_identity does not match observer sidecars")
        input_direct = input_identity is not None
        result_direct = result_identity is not None
        if input_direct:
            input_series = "verified_equal_to_observer_sidecars_direct"
        elif input_result == "verified_by_RESULT.input_sha256" and result_direct:
            input_series = (
                "verified_to_observer_sidecars_via_identity_bound_RESULT.input_sha256"
            )
        else:
            input_series = "INPUT_lacks_identity_binding"
        if result_direct:
            result_series = "verified_equal_to_observer_sidecars_direct"
        elif (
            input_result == "verified_by_RESULT.input_sha256"
            and input_direct
        ):
            result_series = "verified_to_observer_sidecars_via_hashed_INPUT"
        else:
            result_series = "RESULT_lacks_identity_or_hashed_INPUT_binding"
    series = result_series if result_value is not None else input_series
    return {
        "input_result": input_result,
        "series": series,
        "input_series": input_series,
        "result_series": result_series,
        "run_source_identity_sha256": first_digest,
    }


def _run_evidence(
    root: Path, observer_identity: dict[str, Any] | None = None
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    input_path = root / "INPUT.json"
    result_path = root / "RESULT.json"
    input_value = _json_object(input_path) if input_path.is_file() else None
    result_value = _json_object(result_path) if result_path.is_file() else None
    for name in ("INPUT.json", "RESULT.json", "steps.jsonl"):
        path = root / name
        if path.is_file():
            output[name] = {"sha256": _sha256(path)}
    if result_value is not None:
        output["RESULT.json"].update(_result_summary(result_value, result_path))
    output["binding"] = _evidence_binding(
        input_path, result_path, input_value, result_value, observer_identity
    )
    output["declared_contract"] = _observer_declared_contract(
        input_value, result_value, output["binding"]
    )
    steps_path = root / "steps.jsonl"
    if steps_path.is_file():
        count = 0
        first = last = None
        advance_seconds = 0.0
        advance_rows = 0
        for line in steps_path.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{steps_path}: every row must be a JSON object")
            step = row.get("step")
            if not isinstance(step, int) or isinstance(step, bool) or step < 1:
                raise ValueError(f"{steps_path}: step must be a positive integer")
            if last is not None and step <= last:
                raise ValueError(f"{steps_path}: steps must be unique and strictly increasing")
            first = step if first is None else first
            last = step
            count += 1
            if "advance_s" in row:
                value = row["advance_s"]
                if (
                    not isinstance(value, (int, float))
                    or isinstance(value, bool)
                    or not math.isfinite(value)
                    or value < 0
                ):
                    raise ValueError(f"{steps_path}: invalid advance_s at step {step}")
                advance_seconds += float(value)
                advance_rows += 1
        actual_steps_hash = output["steps.jsonl"]["sha256"]
        declared_hashes: list[tuple[str, Any]] = []
        if result_value is not None:
            if "steps_sha256" in result_value:
                declared_hashes.append(
                    ("RESULT.json steps_sha256", result_value.get("steps_sha256"))
                )
            source_hashes = _optional_object(
                result_value,
                "source_files_sha256",
                "RESULT.json source_files_sha256",
            )
            if source_hashes is not None and "steps.jsonl" in source_hashes:
                declared_hashes.append((
                    'RESULT.json source_files_sha256["steps.jsonl"]',
                    source_hashes.get("steps.jsonl"),
                ))
        for source, declared_hash in declared_hashes:
            if declared_hash != actual_steps_hash:
                raise ValueError(f"{source} does not match steps.jsonl")
        if declared_hashes:
            steps_binding = (
                "verified_to_series_via_identity_bound_RESULT_hash"
                if output["binding"].get("result_series") in RESULT_IDENTITY_BOUND
                else "verified_to_adjacent_RESULT_only; RESULT_unbound_to_series"
            )
        else:
            steps_binding = "adjacent_file_unbound_to_RESULT_or_series"
        output["steps.jsonl"].update(
            {"row_count": count, "first_step": first, "last_step": last,
             "advance_s_sum": advance_seconds if advance_rows else None,
             "advance_s_rows": advance_rows, "binding": steps_binding}
        )
    return output


def _frame_geometry(
    path: Path,
    projection: dict[str, Any],
    view: str,
    step_size: float,
    state_count: int,
    partial_policy: str,
    payload: tuple[Any, Any, Any] | None = None,
) -> dict[str, Any]:
    bounds, accepted, status = payload or _load_observer(path, state_count)
    step = _step(path)
    accepted_count = int(accepted.sum().item())
    lanes = int(accepted.numel())
    complete = accepted_count == lanes
    if not complete and partial_policy == "reject":
        raise ValueError(
            f"{path}: only {accepted_count}/{lanes} lanes accepted; "
            "use --partial-policy mark to export accepted-only geometry"
        )
    lo_column, hi_column = VIEW_COLUMNS[view]
    chosen = bounds[accepted]
    accepted_lane_ids = accepted.nonzero(as_tuple=False).flatten().tolist()
    if projection["kind"] == "time-state":
        if accepted_count:
            if "y_transform" in projection:
                intervals = [
                    _affine_interval(row, lo_column, hi_column, projection["y_transform"])
                    for row in chosen
                ]
                lo = min(interval[0] for interval in intervals)
                hi = max(interval[1] for interval in intervals)
            else:
                state = projection["y_index"]
                lo = float(chosen[:, state, lo_column].amin().item())
                hi = float(chosen[:, state, hi_column].amax().item())
            if view == "tube":
                xlo, xhi = (step - 1) * step_size, step * step_size
            else:
                xlo = xhi = step * step_size
            boxes = [[xlo, xhi, lo, hi]]
        else:
            boxes = []
        aggregation = "all-accepted-lanes axis-aligned union hull"
    else:
        x_index, y_index = projection["x_index"], projection["y_index"]
        boxes = [
            [float(row[x_index, lo_column]), float(row[x_index, hi_column]),
             float(row[y_index, lo_column]), float(row[y_index, hi_column])]
            for row in chosen
        ]
        aggregation = "one axis-aligned box per accepted lane"
    counts = Counter(int(value) for value in status.tolist())
    identity = _source_identity(path, step)
    return {
        "step": step,
        "time_start": (step - 1) * step_size,
        "time_end": step * step_size,
        **identity,
        "accepted_lanes": accepted_count,
        "total_lanes": lanes,
        "complete": complete,
        "status_counts": {str(key): counts[key] for key in sorted(counts)},
        "acceptance_semantics": "accepted/status tensors stored in observer",
        "aggregation": aggregation,
        "lane_ids": accepted_lane_ids if projection["kind"] == "state-state" else None,
        "boxes": boxes,
    }


def _native_frame_geometry(
    step: int,
    rows: list[tuple[int, list[float], int]],
    *,
    projection: dict[str, Any],
    step_size: float,
    expected_lanes: int,
) -> dict[str, Any]:
    lanes = [lane for lane, _, _ in rows]
    complete = len(rows) == expected_lanes
    if projection["kind"] == "time-state":
        boxes = [[
            rows[0][1][0],
            rows[0][1][1],
            min(box[2] for _, box, _ in rows),
            max(box[3] for _, box, _ in rows),
        ]]
        aggregation = "all-recorded-lanes axis-aligned union hull"
        lane_ids = None
    else:
        boxes = [box for _, box, _ in rows]
        aggregation = "one axis-aligned box per recorded lane"
        lane_ids = lanes
    return {
        "step": step,
        "time_start": (step - 1) * step_size,
        "time_end": step * step_size,
        "source_record_index_min": min(index for _, _, index in rows),
        "source_record_index_max": max(index for _, _, index in rows),
        "source_record_count": len(rows),
        "source_record_order": "not assumed contiguous",
        "accepted_lanes": None,
        "recorded_lanes": len(rows),
        "total_lanes": expected_lanes,
        "complete": complete,
        "status_counts": {"unavailable_in_ranges_bin": len(rows)},
        "acceptance_semantics": "record presence only; acceptance is not encoded in ranges.bin",
        "aggregation": aggregation,
        "lane_ids": lane_ids,
        "boxes": boxes,
    }


def _native_companions(path: Path) -> dict[str, Any]:
    output: dict[str, Any] = {}
    input_path = path.parent / "INPUT.json"
    result_path = path.parent / "RESULT.json"
    input_value = _json_object(input_path) if input_path.is_file() else None
    result_value = _json_object(result_path) if result_path.is_file() else None
    for name in ("INPUT.json", "RESULT.json"):
        companion = path.parent / name
        if companion.is_file():
            output[name] = {"sha256": _sha256(companion)}
    if result_value is not None:
        output["RESULT.json"].update(_result_summary(result_value, result_path))
    output["binding"] = _evidence_binding(
        input_path, result_path, input_value, result_value, None
    )
    scientific = {} if input_value is None else (input_value.get("scientific") or {})
    step_size = _one_declared_number(
        [("INPUT.json scientific.h", scientific.get("h"))], "step size"
    )
    requested_periods = _one_declared_number(
        [("INPUT.json requested_periods", None if input_value is None else input_value.get(
            "requested_periods"
        ))],
        "requested periods",
        integer=True,
    )
    period = _one_declared_number(
        [("INPUT.json scientific.period", scientific.get("period"))], "control period"
    )
    expected_steps = None
    if requested_periods is not None and period is not None and step_size is not None:
        ratio = period / step_size
        rounded = round(ratio)
        if not math.isclose(ratio, rounded, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("INPUT.json control period is not an integer number of steps")
        expected_steps = int(requested_periods) * rounded
    expected_lanes = _one_declared_number(
        [("INPUT.json scientific.leaf_count", scientific.get("leaf_count"))],
        "native lane count",
        integer=True,
    )
    output["declared_contract"] = {
        "expected_steps": expected_steps,
        "step_size": step_size,
        "expected_lanes": expected_lanes,
        "provenance": {
            "expected_steps": {
                "declaration_sources": (
                    ["INPUT.json requested_periods + scientific.period/h"]
                    if expected_steps is not None else []
                ),
                "identity_bound_sources": [],
                "series_binding": (
                    "adjacent_declaration_unbound_to_ranges_bin"
                    if expected_steps is not None else "declaration_unavailable"
                ),
            },
            "step_size": {
                "declaration_sources": (
                    ["INPUT.json scientific.h"] if step_size is not None else []
                ),
                "identity_bound_sources": [],
                "series_binding": (
                    "adjacent_declaration_unbound_to_ranges_bin"
                    if step_size is not None else "declaration_unavailable"
                ),
            },
            "expected_lanes": {
                "declaration_sources": (
                    ["INPUT.json scientific.leaf_count"]
                    if expected_lanes is not None else []
                ),
                "identity_bound_sources": [],
                "series_binding": (
                    "adjacent_declaration_unbound_to_ranges_bin"
                    if expected_lanes is not None else "declaration_unavailable"
                ),
            },
        },
    }
    return output


def _validate_adjacent_contract(
    evidence: dict[str, Any], *, step_size: float, expected_steps: int,
    observed_lanes: int | None,
) -> None:
    declared = evidence.get("declared_contract", {})
    provenance = declared.get("provenance", {})
    checks: dict[str, Any] = {}
    declaration_binding: dict[str, str] = {}
    declared_step = declared.get("step_size")
    if declared_step is not None:
        if not math.isclose(float(declared_step), step_size, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError(
                f"--step-size {step_size!r} disagrees with adjacent run declaration "
                f"{declared_step!r}"
            )
        checks["step_size"] = "matched"
    else:
        checks["step_size"] = "explicit_cli_unverified"
    declaration_binding["step_size"] = provenance.get("step_size", {}).get(
        "series_binding", "declaration_unavailable"
    )
    declared_steps = declared.get("expected_steps")
    if declared_steps is not None:
        if int(declared_steps) != expected_steps:
            raise ValueError(
                f"--expected-steps {expected_steps} disagrees with adjacent run declaration "
                f"{declared_steps}"
            )
        checks["expected_steps"] = "matched"
    else:
        checks["expected_steps"] = "explicit_cli_unverified"
    declaration_binding["expected_steps"] = provenance.get("expected_steps", {}).get(
        "series_binding", "declaration_unavailable"
    )
    declared_lanes = declared.get("expected_lanes")
    if declared_lanes is not None and observed_lanes is not None:
        if int(declared_lanes) != observed_lanes:
            raise ValueError(
                f"observed lane count {observed_lanes} disagrees with adjacent run declaration "
                f"{declared_lanes}"
            )
        checks["lane_count"] = "matched"
    else:
        checks["lane_count"] = "saved_series_constant_unbound_to_initial_set"
    declaration_binding["lane_count"] = provenance.get("expected_lanes", {}).get(
        "series_binding", "declaration_unavailable"
    )
    steps = evidence.get("steps.jsonl", {})
    if steps.get("last_step") is not None and steps["last_step"] > expected_steps:
        raise ValueError("steps.jsonl contains a step beyond --expected-steps")
    binding = evidence.get("binding", {})
    run_identity_bound = (
        binding.get("input_series") in INPUT_IDENTITY_BOUND
        or binding.get("result_series") in RESULT_IDENTITY_BOUND
    )
    fully_identity_bound = all(
        checks[field] == "matched"
        and declaration_binding[field] == "identity_bound_to_observer_series"
        for field in ("step_size", "expected_steps", "lane_count")
    )
    evidence["plot_contract_check"] = {
        "status": (
            "matched_identity_bound_declarations"
            if fully_identity_bound
            else (
                "identity_bound_run_with_unverified_or_unbound_plot_fields"
                if run_identity_bound
                else "matched_or_unavailable_adjacent_unbound_declarations"
            )
        ),
        "declaration_binding": declaration_binding,
        **checks,
    }


def _native_frames(
    path: Path,
    *,
    projection: dict[str, Any],
    view: str,
    step_size: float,
    state_count: int,
    expected_lanes: int,
    expected_steps: int,
    display_steps: set[int] | None,
    partial_policy: str,
) -> tuple[list[dict[str, Any]], list[int], list[int]]:
    record = _native_range_record(state_count)
    size = path.stat().st_size
    if size == 0 or size % record.size:
        raise ValueError(
            f"{path}: byte size must be a positive multiple of {record.size}"
        )
    seen = [bytearray(expected_lanes) for _ in range(expected_steps + 1)]
    counts = [0] * (expected_steps + 1)
    selected_rows: dict[int, list[tuple[int, list[float], int]]] = {}
    selected_box_count = 0
    lo_column, hi_column = VIEW_COLUMNS[view]

    with path.open("rb") as handle:
        for record_index in range(size // record.size):
            lane, step, recorded_h, *flat = record.unpack(
                handle.read(record.size)
            )
            if step < 1:
                raise ValueError(f"{path}: record {record_index} has a nonpositive step")
            if step > expected_steps:
                raise ValueError(
                    f"{path}: record {record_index} step {step} exceeds expected horizon"
                )
            if lane >= expected_lanes:
                raise ValueError(
                    f"{path}: record {record_index} lane {lane} exceeds expected lanes"
                )
            if seen[step][lane]:
                raise ValueError(f"{path}: duplicate lane {lane} at step {step}")
            seen[step][lane] = 1
            counts[step] += 1
            if recorded_h != step_size:
                raise ValueError(
                    f"{path}: record {record_index} h={recorded_h!r} != {step_size!r}"
                )
            bounds = [flat[index:index + 4] for index in range(0, len(flat), 4)]
            if any(
                not all(math.isfinite(value) for value in row)
                or row[0] > row[1]
                or row[2] > row[3]
                for row in bounds
            ):
                raise ValueError(f"{path}: record {record_index} has invalid bounds")
            if display_steps is None or step in display_steps:
                selected_box_count += 1
                if (
                    projection["kind"] == "state-state"
                    and selected_box_count > MAX_STATE_STATE_BOXES
                ):
                    raise ValueError(
                        f"state-state display would exceed {MAX_STATE_STATE_BOXES} boxes; "
                        "select explicit --display-steps without changing the numerical run"
                    )
                if projection["kind"] == "time-state":
                    if view == "tube":
                        xlo, xhi = (step - 1) * step_size, step * step_size
                    else:
                        xlo = xhi = step * step_size
                    if "y_transform" in projection:
                        ylo, yhi = _affine_interval(
                            bounds, lo_column, hi_column, projection["y_transform"]
                        )
                    else:
                        state = projection["y_index"]
                        ylo, yhi = bounds[state][lo_column], bounds[state][hi_column]
                    box = [
                        xlo,
                        xhi,
                        ylo,
                        yhi,
                    ]
                else:
                    x_index, y_index = projection["x_index"], projection["y_index"]
                    box = [
                        bounds[x_index][lo_column],
                        bounds[x_index][hi_column],
                        bounds[y_index][lo_column],
                        bounds[y_index][hi_column],
                    ]
                selected_rows.setdefault(int(step), []).append((int(lane), box, record_index))
    observed_steps = [step for step in range(1, expected_steps + 1) if counts[step]]
    partial_steps = [step for step in observed_steps if counts[step] != expected_lanes]
    if partial_steps and partial_policy == "reject":
        first = partial_steps[0]
        raise ValueError(
            f"{path}: step {first} has {counts[first]}/{expected_lanes} recorded lanes"
        )
    requested = observed_steps if display_steps is None else sorted(display_steps)
    unavailable = sorted(set(requested) - set(observed_steps))
    if unavailable:
        raise ValueError(f"{path}: requested display steps lack range records: {unavailable[:8]}")
    frames = [
        _native_frame_geometry(
            step,
            selected_rows[step],
            projection=projection,
            step_size=step_size,
            expected_lanes=expected_lanes,
        )
        for step in requested
    ]
    return frames, observed_steps, partial_steps


def _plain_text(value: Any, label: str, *, required: bool = True) -> str | None:
    if value is None and not required:
        return None
    if (
        not isinstance(value, str)
        or not value.strip()
        or any(ord(character) < 32 for character in value)
    ):
        raise ValueError(f"{label} must be a nonempty string without control characters")
    return value


def _validate_plot_spec(
    spec: dict[str, Any] | None,
    benchmark: str,
    coordinate_names: list[str],
    *,
    instance_id: str | None = None,
    numerical_horizon: float | None = None,
) -> dict[str, Any]:
    if spec is None or spec == {}:
        return {"status": "no_plot_spec", "identity_binding": "none"}
    if not isinstance(spec, dict):
        raise ValueError("plot spec must be an object")
    schema = spec.get("schema")
    if schema not in {PLOT_SPEC_SCHEMA, PLOT_SPEC_SCHEMA_V2, PLOT_SPEC_SCHEMA_V3}:
        raise ValueError(
            "plot spec must use schema "
            f"{PLOT_SPEC_SCHEMA}, {PLOT_SPEC_SCHEMA_V2}, or {PLOT_SPEC_SCHEMA_V3}"
        )
    if spec.get("benchmark") != benchmark:
        raise ValueError("plot spec benchmark does not match --benchmark")
    if spec.get("coordinate_names") not in (None, coordinate_names):
        raise ValueError("spec coordinate_names do not match the exporter coordinates")
    _plain_text(spec.get("contract_status"), "plot spec contract_status")
    official = schema in {PLOT_SPEC_SCHEMA_V2, PLOT_SPEC_SCHEMA_V3}
    version = "v2" if schema == PLOT_SPEC_SCHEMA_V2 else "v3"
    warning = _plain_text(
        spec.get("warning"),
        "plot spec warning",
        required=official,
    )
    binding = spec.get("identity_binding")
    if schema == PLOT_SPEC_SCHEMA:
        if binding != "informational_legacy_unbound":
            raise ValueError("v1 plot spec must be informational_legacy_unbound")
        binding_result = {
            "status": "accepted_with_explicit_unbound_legacy_scope",
            "identity_binding": binding,
            "contract_status": spec["contract_status"],
        }
    else:
        if not instance_id:
            raise ValueError(f"{version} plot spec requires an explicit --instance-id")
        _plain_text(instance_id, "instance id")
        if spec.get("instance_id") != instance_id:
            raise ValueError("plot spec instance_id does not match --instance-id")
        if spec.get("coordinate_names") != coordinate_names:
            raise ValueError(
                f"{version} plot spec must declare the exact exporter coordinates"
            )
        if spec.get("model_domain") != "continuous_time":
            raise ValueError(
                f"{version} plot spec renderer currently supports only continuous_time"
            )
        if binding != "official_contract_sources_hash_declared":
            raise ValueError(
                f"{version} plot spec must declare "
                "official_contract_sources_hash_declared"
            )
        if spec.get("run_binding") != "series_source_identity_plot_contract_required":
            raise ValueError(
                f"{version} plot spec must declare "
                "run_binding=series_source_identity_plot_contract_required"
            )
        source_refs = spec.get("source_refs")
        if not isinstance(source_refs, list) or not source_refs:
            raise ValueError(f"{version} plot spec source_refs must be a nonempty list")
        seen_source_paths: set[str] = set()
        for index, source_ref in enumerate(source_refs, 1):
            if not isinstance(source_ref, dict) or set(source_ref) != {"path", "sha256"}:
                raise ValueError(
                    f"{version} plot spec source_ref {index} must contain only path and sha256"
                )
            path = _plain_text(
                source_ref.get("path"), f"{version} plot spec source_ref {index} path"
            )
            assert path is not None
            path_parts = Path(path).parts
            if (
                not path_parts
                or path == "."
                or Path(path).is_absolute()
                or "\\" in path
                or ".." in path_parts
                or "/".join(path_parts) != path
            ):
                raise ValueError(
                    f"{version} plot spec source_ref {index} path must be a safe relative path"
                )
            digest = source_ref.get("sha256")
            if not isinstance(digest, str) or SHA256_RE.fullmatch(digest) is None:
                raise ValueError(
                    f"{version} plot spec source_ref {index} sha256 must be lowercase hexadecimal"
                )
            if path in seen_source_paths:
                raise ValueError(f"{version} plot spec source_ref paths must be unique")
            seen_source_paths.add(path)
        horizon = spec.get("horizon")
        if not isinstance(horizon, dict) or set(horizon) != {"kind", "start", "end"}:
            raise ValueError(
                f"{version} plot spec horizon must contain kind, start, and end"
            )
        if horizon.get("kind") != "continuous_time":
            raise ValueError(
                f"{version} plot spec horizon kind must be continuous_time"
            )
        start, end = horizon.get("start"), horizon.get("end")
        if (
            not isinstance(start, (int, float))
            or isinstance(start, bool)
            or not isinstance(end, (int, float))
            or isinstance(end, bool)
            or not math.isfinite(start)
            or not math.isfinite(end)
            or float(start) != 0.0
            or end <= start
        ):
            raise ValueError(
                f"{version} plot spec horizon must be a finite positive interval from 0"
            )
        if numerical_horizon is not None and not math.isclose(
            float(end), numerical_horizon, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(
                f"{version} plot spec horizon does not match the numerical horizon"
            )
        quantifier = spec.get("property_quantifier")
        if quantifier not in PROPERTY_QUANTIFIERS:
            raise ValueError(
                f"{version} plot spec property_quantifier must be one of "
                f"{sorted(PROPERTY_QUANTIFIERS)}"
            )
        binding_result = {
            "status": "official_content_requires_series_instance_binding",
            "identity_binding": binding,
            "run_binding": spec["run_binding"],
            "instance_id": instance_id,
            "contract_status": spec["contract_status"],
            "source_ref_count": len(source_refs),
            "warning": warning,
        }

    derived_coordinates = spec.get("derived_coordinates", {})
    if schema != PLOT_SPEC_SCHEMA_V3 and derived_coordinates:
        raise ValueError("affine derived coordinates require plot spec v3")
    if not isinstance(derived_coordinates, dict):
        raise ValueError("plot spec derived_coordinates must be an object")
    if schema == PLOT_SPEC_SCHEMA_V3 and len(derived_coordinates) != 1:
        raise ValueError("v3 plot spec requires exactly one derived coordinate")
    for name, transform in derived_coordinates.items():
        if (
            not isinstance(name, str)
            or DERIVED_COORDINATE_RE.fullmatch(name) is None
            or name in {"t", "time", *coordinate_names}
        ):
            raise ValueError(f"invalid or conflicting derived coordinate name {name!r}")
        if (
            not isinstance(transform, dict)
            or set(transform) != {"kind", "offset", "coefficients"}
            or transform.get("kind") != "affine"
        ):
            raise ValueError(
                f"derived coordinate {name!r} must be one affine transform"
            )
        offset = transform.get("offset")
        coefficients = transform.get("coefficients")
        if (
            not isinstance(offset, (int, float))
            or isinstance(offset, bool)
            or not math.isfinite(offset)
        ):
            raise ValueError(f"derived coordinate {name!r} offset must be finite")
        if not isinstance(coefficients, dict) or not coefficients:
            raise ValueError(
                f"derived coordinate {name!r} coefficients must be nonempty"
            )
        for coordinate, coefficient in coefficients.items():
            if coordinate not in coordinate_names:
                raise ValueError(
                    f"derived coordinate {name!r} uses unknown coordinate {coordinate!r}"
                )
            if (
                not isinstance(coefficient, (int, float))
                or isinstance(coefficient, bool)
                or not math.isfinite(coefficient)
                or coefficient == 0
            ):
                raise ValueError(
                    f"derived coordinate {name!r} coefficients must be finite and nonzero"
                )

    initial_set = spec.get("initial_set", {})
    regions = spec.get("regions", [])
    if not isinstance(initial_set, dict):
        raise ValueError("plot spec initial_set must be an object")
    if not isinstance(regions, list) or any(not isinstance(region, dict) for region in regions):
        raise ValueError("plot spec regions must be a list of objects")
    if official and (not initial_set or not regions):
        raise ValueError(f"{version} plot spec requires nonempty initial_set and regions")
    units = spec.get("units", {})
    if (
        not isinstance(units, dict)
        or any(key not in {"t", *coordinate_names, *derived_coordinates} for key in units)
        or any(not isinstance(unit, str) or any(ord(char) < 32 for char in unit)
               for unit in units.values())
    ):
        raise ValueError("plot spec units must map known coordinates to strings")
    if initial_set:
        _plain_text(initial_set.get("label", "Initial set"), "plot spec initial_set label")
    bounded_owners = [("initial_set", initial_set.get("bounds", {}))]
    bounded_owners.extend(
        (f"region {index}", region["bounds"])
        for index, region in enumerate(regions, 1)
        if "bounds" in region
    )
    for owner, bounds in bounded_owners:
        if not isinstance(bounds, dict):
            raise ValueError(f"plot spec {owner} bounds must be an object")
        for coordinate, interval in bounds.items():
            if coordinate not in coordinate_names:
                raise ValueError(f"plot spec {owner} uses unknown coordinate {coordinate!r}")
            if (
                not isinstance(interval, list)
                or len(interval) != 2
                or not all(
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(value)
                    for value in interval
                )
                or interval[0] > interval[1]
            ):
                raise ValueError(f"plot spec {owner} has invalid bounds for {coordinate}")
    for index, region in enumerate(regions, 1):
        _plain_text(region.get("label", f"Region {index}"), f"plot spec region {index} label")
        role = region.get("role", "informational")
        if role not in REGION_ROLES:
            raise ValueError(
                f"plot spec region {index} role must be one of {sorted(REGION_ROLES)}"
            )
        has_bounds = "bounds" in region
        has_constraint = "constraint" in region
        if schema == PLOT_SPEC_SCHEMA_V3:
            if has_bounds == has_constraint:
                raise ValueError(
                    f"v3 plot spec region {index} must contain exactly one of "
                    "bounds or constraint"
                )
            if has_constraint:
                constraint = region["constraint"]
                if (
                    not isinstance(constraint, dict)
                    or set(constraint) != {"kind", "coordinate", "operator", "value"}
                    or constraint.get("kind") != "threshold"
                    or constraint.get("coordinate") not in derived_coordinates
                    or constraint.get("operator") not in {">=", "<="}
                ):
                    raise ValueError(
                        f"v3 plot spec region {index} has an invalid affine threshold"
                    )
                threshold = constraint.get("value")
                if (
                    not isinstance(threshold, (int, float))
                    or isinstance(threshold, bool)
                    or not math.isfinite(threshold)
                ):
                    raise ValueError(
                        f"v3 plot spec region {index} threshold must be finite"
                    )
        elif has_constraint:
            raise ValueError("affine threshold regions require plot spec v3")
        timing = region.get("time", {"kind": "all"})
        if not isinstance(timing, dict):
            raise ValueError(f"plot spec region {index} time must be an object")
        kind = timing.get("kind", "all")
        if kind == "endpoint":
            values = [timing.get("at")]
        elif kind == "interval":
            values = [timing.get("lo"), timing.get("hi")]
        elif kind == "all":
            values = []
        else:
            raise ValueError(f"plot spec region {index} has unknown time kind {kind!r}")
        if any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            for value in values
        ):
            raise ValueError(f"plot spec region {index} has invalid time bounds")
        if kind == "interval" and values[0] > values[1]:
            raise ValueError(f"plot spec region {index} has reversed time bounds")
    if official:
        if set(initial_set.get("bounds", {})) != set(coordinate_names):
            raise ValueError(
                f"{version} plot spec initial_set must bound every coordinate"
            )
        property_regions = [
            region for region in regions if region.get("role", "informational") != "informational"
        ]
        if not property_regions:
            raise ValueError(f"{version} plot spec requires property regions")
        if schema == PLOT_SPEC_SCHEMA_V2 and any(
            not region.get("bounds") for region in property_regions
        ):
            raise ValueError("v2 plot spec requires nonempty property-region bounds")
        if schema == PLOT_SPEC_SCHEMA_V3 and (
            len(property_regions) != 1 or "constraint" not in property_regions[0]
        ):
            raise ValueError(
                "v3 plot spec requires exactly one affine threshold property region"
            )
        horizon = spec["horizon"]
        quantifier = spec["property_quantifier"]
        for region in property_regions:
            if "time" not in region:
                raise ValueError(
                    f"{version} property regions must declare time explicitly"
                )
            timing = region.get("time", {"kind": "all"})
            kind = timing.get("kind", "all")
            if quantifier == "endpoint" and (
                kind != "endpoint"
                or not math.isclose(
                    float(timing["at"]), float(horizon["end"]),
                    rel_tol=0.0, abs_tol=1e-12,
                )
            ):
                raise ValueError(
                    f"{version} endpoint property regions must be at the horizon endpoint"
                )
            if quantifier == "all_times" and kind not in {"all", "interval"}:
                raise ValueError(
                    f"{version} all_times property regions must use all or an interval"
                )
            if quantifier == "eventually" and kind != "interval":
                raise ValueError(
                    f"{version} eventually property regions must use an explicit interval"
                )
        if quantifier == "conjunction" and len(property_regions) < 2:
            raise ValueError(
                f"{version} conjunction requires at least two property regions"
            )
    return binding_result


def _validate_plot_series_binding(
    spec_binding: dict[str, Any],
    benchmark: str,
    instance_id: str | None,
    coordinate_names: list[str],
    step_size: float,
    expected_steps: int,
    series: list[dict[str, Any]],
) -> dict[str, Any]:
    if spec_binding.get("run_binding") != "series_source_identity_plot_contract_required":
        return spec_binding
    for item in series:
        identity_record = item.get("source_identity")
        if (
            item.get("source_kind") != "torch-observer-pt-v1"
            or not isinstance(identity_record, dict)
            or identity_record.get("status")
            != "verified_equal_across_all_observer_sidecars"
            or not isinstance(identity_record.get("source_identity"), dict)
        ):
            raise ValueError(
                "official plot spec requires every series to have one source_identity "
                "verified across all observer sidecars"
            )
        identity = identity_record["source_identity"]
        if identity.get("benchmark") != benchmark:
            raise ValueError("series source_identity benchmark does not match plot spec")
        if identity.get("instance_id") != instance_id:
            raise ValueError("series source_identity instance_id does not match plot spec")
        if identity.get("coordinate_names") != coordinate_names:
            raise ValueError("series source_identity coordinate_names do not match plot spec")
        identity_step_size = identity.get("step_size")
        if (
            not isinstance(identity_step_size, (int, float))
            or isinstance(identity_step_size, bool)
            or not math.isclose(
                float(identity_step_size), step_size, rel_tol=0.0, abs_tol=1e-15
            )
        ):
            raise ValueError("series source_identity step_size does not match geometry")
        identity_expected_steps = identity.get("expected_steps")
        if (
            not isinstance(identity_expected_steps, int)
            or isinstance(identity_expected_steps, bool)
            or identity_expected_steps != expected_steps
        ):
            raise ValueError("series source_identity expected_steps does not match geometry")
    return {
        **spec_binding,
        "status": "official_content_series_plot_contract_binding_verified",
    }


def export_geometry(
    series: list[tuple[str, Path]],
    *,
    benchmark: str,
    instance_id: str | None = None,
    coordinate_names: list[str],
    projection_text: str,
    view: str,
    step_size: float,
    expected_steps: int | None = None,
    expected_lanes: int | None = None,
    display_steps: set[int] | None = None,
    partial_policy: str = "reject",
    spec: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not series:
        raise ValueError("at least one series is required")
    _plain_text(benchmark, "benchmark")
    if instance_id is not None:
        _plain_text(instance_id, "instance id")
    if not math.isfinite(step_size) or step_size <= 0:
        raise ValueError("step size must be a finite positive number")
    if view not in VIEW_COLUMNS:
        raise ValueError(f"unsupported view {view!r}")
    if partial_policy not in {"reject", "mark"}:
        raise ValueError("partial policy must be reject or mark")
    if not coordinate_names or len(coordinate_names) != len(set(coordinate_names)):
        raise ValueError("coordinate names must be nonempty and unique")
    for name in coordinate_names:
        _plain_text(name, "coordinate name")
    if (
        not isinstance(expected_steps, int)
        or isinstance(expected_steps, bool)
        or expected_steps < 1
    ):
        raise ValueError("expected_steps is required and must be positive")
    if display_steps is not None and not display_steps:
        raise ValueError("display step selection is empty")
    if display_steps is not None and any(
        step < 1 or step > expected_steps for step in display_steps
    ):
        raise ValueError("display steps must lie inside the expected horizon")
    spec_binding = _validate_plot_spec(
        spec,
        benchmark,
        coordinate_names,
        instance_id=instance_id,
        numerical_horizon=expected_steps * step_size,
    )
    derived_coordinates = (
        spec.get("derived_coordinates", {}) if isinstance(spec, dict) else {}
    )
    projection = _parse_projection(
        projection_text, coordinate_names, derived_coordinates
    )
    if expected_lanes is not None and expected_lanes < 1:
        raise ValueError("expected_lanes must be positive when provided")
    exported = []
    labels: set[str] = set()
    for label, root in series:
        _plain_text(label, "series label")
        if label in labels:
            raise ValueError(f"series labels must be nonempty and unique: {label!r}")
        labels.add(label)
        if root.is_file():
            if root.name != "ranges.bin":
                raise ValueError(f"only the native ranges.bin file format is supported: {root}")
            if expected_lanes is None or expected_lanes < 1:
                raise ValueError("native ranges.bin export requires --expected-lanes")
            frames, observed_steps, observed_partial_steps = _native_frames(
                root,
                projection=projection,
                view=view,
                step_size=step_size,
                state_count=len(coordinate_names),
                expected_lanes=expected_lanes,
                expected_steps=expected_steps,
                display_steps=display_steps,
                partial_policy=partial_policy,
            )
            evidence = _native_companions(root)
            _validate_adjacent_contract(
                evidence,
                step_size=step_size,
                expected_steps=expected_steps,
                observed_lanes=expected_lanes,
            )
            source = {
                "source_kind": "native-ranges-bin-v1",
                "source_path": str(root.resolve()),
                "source_sha256": _sha256(root),
                "record_size": _native_range_record(len(coordinate_names)).size,
                "record_schema": (
                    f"little-endian <lane:uint64,step:uint64,h:float64,"
                    f"{4 * len(coordinate_names)}xfloat64>; four bounds per physical state"
                ),
                "physical_state_count_from_explicit_coordinates": len(coordinate_names),
                "lane_universe_binding": "explicit_cli_expected_lanes",
                "solver_run_evidence": evidence,
            }
        else:
            paths = discover_observers(root)
            observed_steps = [_step(path) for path in paths]
            if max(observed_steps) > expected_steps:
                raise ValueError("an observer step exceeds the expected horizon")
            requested = observed_steps if display_steps is None else sorted(display_steps)
            unavailable = sorted(set(requested) - set(observed_steps))
            if unavailable:
                raise ValueError(
                    f"{root}: requested display steps lack observer files: {unavailable[:8]}"
                )
            requested_set = set(requested)
            frames = []
            observed_partial_steps = []
            displayed_box_count = 0
            lane_counts: set[int] = set()
            for path in paths:
                payload = _load_observer(path, len(coordinate_names))
                accepted = payload[1]
                step = _step(path)
                lanes = int(accepted.numel())
                lane_counts.add(lanes)
                if len(lane_counts) != 1:
                    raise ValueError(
                        f"{root}: observer lane count changes across saved steps; "
                        "refusing to treat a pruned survivor set as the full initial set"
                    )
                accepted_count = int(accepted.sum().item())
                if accepted_count != lanes:
                    observed_partial_steps.append(step)
                    if partial_policy == "reject":
                        raise ValueError(
                            f"{path}: only {accepted_count}/{lanes} lanes accepted; "
                            "use --partial-policy mark to export accepted-only geometry"
                        )
                if step in requested_set:
                    if projection["kind"] == "state-state":
                        displayed_box_count += accepted_count
                        if displayed_box_count > MAX_STATE_STATE_BOXES:
                            raise ValueError(
                                f"state-state display would exceed {MAX_STATE_STATE_BOXES} boxes; "
                                "select explicit --display-steps without changing the numerical run"
                            )
                    frames.append(_frame_geometry(
                        path,
                        projection,
                        view,
                        step_size,
                        len(coordinate_names),
                        partial_policy,
                        payload,
                    ))
            observer_identity = _observer_series_identity(paths)
            evidence = _run_evidence(root, observer_identity)
            observer_lanes = next(iter(lane_counts))
            _validate_adjacent_contract(
                evidence,
                step_size=step_size,
                expected_steps=expected_steps,
                observed_lanes=observer_lanes,
            )
            declared_lanes = evidence["declared_contract"].get("expected_lanes")
            if declared_lanes is None:
                lane_binding = "constant_saved_lane_count_only; original_initial_set_unverified"
            elif evidence["declared_contract"]["provenance"]["expected_lanes"][
                "series_binding"
            ] == "identity_bound_to_observer_series":
                lane_binding = "identity_bound_run_declaration"
            else:
                lane_binding = "matched_adjacent_declaration_unbound_to_observers"
            source = {
                "source_kind": "torch-observer-pt-v1",
                "source_path": str(root.resolve()),
                "source_identity": observer_identity,
                "observer_lane_count": observer_lanes,
                "lane_universe_binding": lane_binding,
                "solver_run_evidence": evidence,
            }
        displayed = [frame["step"] for frame in frames]
        unobserved_ranges = _missing_ranges(observed_steps, expected_steps)
        omitted = sorted(set(observed_steps) - set(displayed))
        exported.append(
            {
                "label": label,
                **source,
                "observed_steps": observed_steps,
                "displayed_steps": displayed,
                "expected_steps": expected_steps,
                "display_selection": {
                    "policy": "all-observed" if display_steps is None else "explicit",
                    "requested_steps": None if display_steps is None else sorted(display_steps),
                },
                "projection_unobserved_step_count": expected_steps - len(observed_steps),
                "projection_unobserved_step_ranges": unobserved_ranges,
                "projection_unobserved_reason": (
                    "no projection record is present; this field does not infer solver failure, "
                    "timeout, rejection, or intentional save sampling"
                ),
                "display_omitted_observed_step_count": len(omitted),
                "display_omitted_observed_step_ranges": _compact_ranges(omitted),
                "projection_coverage_semantics": (
                    "complete saved projection sequence"
                    if not unobserved_ranges
                    else "saved projection files are a subset or prefix; solver status is separate"
                ),
                "projection_partial_step_ranges": (
                    None
                    if observed_partial_steps is None
                    else _compact_ranges(observed_partial_steps)
                ),
                "projection_partial_coverage_semantics": (
                    "known for every observed native record step"
                    if root.is_file()
                    else (
                        "known because every observed observer file is inspected"
                        if observed_partial_steps is not None
                        else "unknown for display-omitted observer files"
                    )
                ),
                "displayed_partial_step_ranges": _compact_ranges(
                    frame["step"] for frame in frames if not frame["complete"]
                ),
                "frames": frames,
            }
        )
    spec_binding = _validate_plot_series_binding(
        spec_binding, benchmark, instance_id, coordinate_names,
        step_size, expected_steps, exported,
    )
    geometry = {
        "schema": SCHEMA,
        "benchmark": benchmark,
        "coordinate_names": coordinate_names,
        "projection": projection,
        "view": view,
        "observer_columns": ["tube_lo", "tube_hi", "endpoint_lo", "endpoint_hi"],
        "step_size": step_size,
        "step_size_hex": step_size.hex(),
        "expected_steps": expected_steps,
        "partial_policy": partial_policy,
        "interpolation": "none",
        "geometry_class": (
            "axis-aligned interval image of a declared affine coordinate; "
            "source-coordinate correlations unavailable; not a Flow* octagon"
            if "y_transform" in projection
            else "axis-aligned box projection; not a Flow* octagon"
        ),
        "semantics": (
            "tube bounds cover one local integration step"
            if view == "tube"
            else "endpoint bounds are evaluated at the end of one integration step"
        ),
        "series": exported,
        "spec": spec or {},
        "spec_binding": spec_binding,
    }
    if instance_id is not None:
        geometry["instance_id"] = instance_id
    return validate_geometry(geometry)


def validate_geometry(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("schema") != SCHEMA:
        raise ValueError(f"geometry must use schema {SCHEMA}")
    benchmark = value.get("benchmark")
    instance_id = value.get("instance_id")
    coordinate_names = value.get("coordinate_names")
    _plain_text(benchmark, "geometry benchmark")
    if instance_id is not None:
        _plain_text(instance_id, "geometry instance_id")
    if (
        not isinstance(coordinate_names, list)
        or not coordinate_names
        or any(not isinstance(name, str) or not name for name in coordinate_names)
        or len(coordinate_names) != len(set(coordinate_names))
    ):
        raise ValueError("geometry coordinate_names must be nonempty unique strings")
    for name in coordinate_names:
        _plain_text(name, "geometry coordinate name")
    if value.get("view") not in VIEW_COLUMNS:
        raise ValueError("geometry has an unsupported view")
    if value.get("interpolation") != "none":
        raise ValueError("geometry must explicitly disable interpolation")
    step_size = value.get("step_size")
    if (
        not isinstance(step_size, (int, float))
        or isinstance(step_size, bool)
        or not math.isfinite(step_size)
        or step_size <= 0
    ):
        raise ValueError("geometry step_size must be finite and positive")
    if value.get("step_size_hex") != float(step_size).hex():
        raise ValueError("geometry step_size_hex does not match step_size")
    expected_steps = value.get("expected_steps")
    if not isinstance(expected_steps, int) or isinstance(expected_steps, bool) or expected_steps < 1:
        raise ValueError("geometry expected_steps must be a positive integer")
    spec = value.get("spec")
    if not isinstance(spec, dict):
        raise ValueError("geometry spec must be a JSON object")
    derived_coordinates = (
        spec.get("derived_coordinates", {})
        if spec.get("schema") == PLOT_SPEC_SCHEMA_V3
        else {}
    )
    projection = value.get("projection")
    if not isinstance(projection, dict) or projection.get("kind") not in {
        "time-state", "state-state"
    }:
        raise ValueError("geometry has an invalid projection")
    try:
        canonical_projection = _parse_projection(
            f"{projection['x']},{projection['y']}",
            coordinate_names,
            derived_coordinates,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("geometry projection coordinates are invalid") from exc
    if projection != canonical_projection:
        raise ValueError("geometry projection metadata is inconsistent")
    if value.get("partial_policy") not in {"reject", "mark"}:
        raise ValueError("geometry has an invalid partial policy")
    if not isinstance(value.get("series"), list) or not value["series"]:
        raise ValueError("geometry must contain at least one series")
    labels: set[str] = set()
    total_boxes = 0
    for item in value["series"]:
        if not isinstance(item, dict):
            raise ValueError("every geometry series must be an object")
        label = item.get("label")
        if not isinstance(label, str) or label in labels:
            raise ValueError("geometry series labels must be nonempty and unique")
        _plain_text(label, "geometry series label")
        labels.add(label)
        if not isinstance(item.get("frames"), list) or not item["frames"]:
            raise ValueError("every series must contain at least one frame")
        if any(not isinstance(frame, dict) for frame in item["frames"]):
            raise ValueError("every frame must be an object")
        if any(not isinstance(frame.get("complete"), bool) for frame in item["frames"]):
            raise ValueError("frame complete must be boolean")
        if item.get("expected_steps") != expected_steps:
            raise ValueError("series expected_steps must match geometry expected_steps")
        source_kind = item.get("source_kind")
        if source_kind not in {
            "torch-observer-pt-v1", "native-ranges-bin-v1"
        }:
            raise ValueError("geometry series has an unsupported source_kind")
        if not isinstance(item.get("lane_universe_binding"), str):
            raise ValueError("geometry series must declare lane_universe_binding")
        evidence = item.get("solver_run_evidence")
        if not isinstance(evidence, dict) or not isinstance(evidence.get("binding"), dict):
            raise ValueError("geometry series must retain solver evidence binding metadata")
        observed = item.get("observed_steps")
        displayed = item.get("displayed_steps")
        if (
            not isinstance(observed, list)
            or any(
                not isinstance(step, int) or isinstance(step, bool)
                or step < 1 or step > expected_steps
                for step in observed
            )
            or observed != sorted(observed)
            or len(observed) != len(set(observed))
        ):
            raise ValueError("series observed_steps must be sorted unique horizon steps")
        if (
            not isinstance(displayed, list)
            or any(not isinstance(step, int) or isinstance(step, bool) for step in displayed)
            or displayed != sorted(displayed)
            or len(displayed) != len(set(displayed))
            or not set(displayed) <= set(observed)
        ):
            raise ValueError("series displayed_steps must be sorted unique observed steps")
        frame_steps = [frame.get("step") for frame in item["frames"]]
        if frame_steps != displayed:
            raise ValueError("frame steps must exactly match displayed_steps")
        selection = item.get("display_selection")
        if not isinstance(selection, dict) or selection.get("policy") not in {
            "all-observed", "explicit"
        }:
            raise ValueError("series display_selection is invalid")
        requested_steps = selection.get("requested_steps")
        if selection["policy"] == "all-observed":
            if requested_steps is not None or displayed != observed:
                raise ValueError("all-observed display selection is inconsistent")
        elif requested_steps != displayed:
            raise ValueError("explicit display selection must match displayed_steps")
        missing = [step for step in range(1, expected_steps + 1) if step not in set(observed)]
        omitted = sorted(set(observed) - set(displayed))
        if (
            item.get("projection_unobserved_step_count") != len(missing)
            or item.get("projection_unobserved_step_ranges") != _compact_ranges(missing)
        ):
            raise ValueError("projection-unobserved coverage fields are inconsistent")
        if (
            item.get("display_omitted_observed_step_count") != len(omitted)
            or item.get("display_omitted_observed_step_ranges") != _compact_ranges(omitted)
        ):
            raise ValueError("display-omitted coverage fields are inconsistent")
        displayed_partial = [
            frame["step"] for frame in item["frames"] if not frame.get("complete")
        ]
        if item.get("displayed_partial_step_ranges") != _compact_ranges(displayed_partial):
            raise ValueError("displayed partial-step fields are inconsistent")
        projection_partial_ranges = item.get("projection_partial_step_ranges")
        if projection_partial_ranges is not None:
            projection_partial = _expand_ranges(
                projection_partial_ranges, "projection_partial_step_ranges"
            )
            if not set(projection_partial) <= set(observed):
                raise ValueError("projection partial steps must be observed steps")
            if set(projection_partial) & set(displayed) != set(displayed_partial):
                raise ValueError("projection and displayed partial-step fields disagree")
            if value["partial_policy"] == "reject" and projection_partial:
                raise ValueError("reject partial policy cannot hide incomplete observed frames")
        elif displayed_partial:
            raise ValueError("displayed partial steps require known projection partial coverage")
        if value["partial_policy"] == "reject" and displayed_partial:
            raise ValueError("reject partial policy cannot contain incomplete displayed frames")
        frame_lane_counts: set[int] = set()
        for frame in item["frames"]:
            boxes = frame.get("boxes")
            if not isinstance(boxes, list):
                raise ValueError("frame boxes must be a list")
            step = frame.get("step")
            expected_start = (step - 1) * step_size
            expected_end = step * step_size
            time_start = frame.get("time_start")
            time_end = frame.get("time_end")
            if (
                not isinstance(time_start, (int, float))
                or isinstance(time_start, bool)
                or not isinstance(time_end, (int, float))
                or isinstance(time_end, bool)
                or not math.isclose(float(time_start), expected_start, rel_tol=0.0,
                                    abs_tol=1e-15)
                or not math.isclose(float(time_end), expected_end, rel_tol=0.0,
                                    abs_tol=1e-15)
            ):
                raise ValueError("frame time interval does not match step and step_size")
            total_lanes = frame.get("total_lanes")
            if (
                not isinstance(total_lanes, int)
                or isinstance(total_lanes, bool)
                or total_lanes < 1
            ):
                raise ValueError("frame total_lanes must be a positive integer")
            frame_lane_counts.add(total_lanes)
            if source_kind == "torch-observer-pt-v1":
                accepted_lanes = frame.get("accepted_lanes")
                if (
                    not isinstance(accepted_lanes, int)
                    or isinstance(accepted_lanes, bool)
                    or accepted_lanes < 0
                    or accepted_lanes > total_lanes
                    or frame["complete"] != (accepted_lanes == total_lanes)
                ):
                    raise ValueError("observer frame acceptance counts are inconsistent")
                if frame.get("acceptance_semantics") != (
                    "accepted/status tensors stored in observer"
                ):
                    raise ValueError("observer frame acceptance semantics are inconsistent")
                represented_lanes = accepted_lanes
            else:
                recorded_lanes = frame.get("recorded_lanes")
                if (
                    not isinstance(recorded_lanes, int)
                    or isinstance(recorded_lanes, bool)
                    or recorded_lanes < 1
                    or recorded_lanes > total_lanes
                    or frame.get("accepted_lanes") is not None
                    or frame["complete"] != (recorded_lanes == total_lanes)
                ):
                    raise ValueError("native frame record counts are inconsistent")
                if frame.get("acceptance_semantics") != (
                    "record presence only; acceptance is not encoded in ranges.bin"
                ):
                    raise ValueError("native frame acceptance semantics are inconsistent")
                represented_lanes = recorded_lanes
            lane_ids = frame.get("lane_ids")
            if projection["kind"] == "state-state":
                if (
                    not isinstance(lane_ids, list)
                    or any(not isinstance(lane, int) or isinstance(lane, bool)
                           for lane in lane_ids)
                    or len(lane_ids) != len(boxes)
                    or len(boxes) != represented_lanes
                ):
                    raise ValueError("state-state frames must preserve one lane id per box")
                if len(lane_ids) != len(set(lane_ids)):
                    raise ValueError("state-state frame lane ids must be unique")
            elif lane_ids is not None:
                raise ValueError("time-state frames must use lane_ids=null")
            elif len(boxes) != (1 if represented_lanes else 0):
                raise ValueError("time-state frames must contain one union-hull box")
            total_boxes += len(boxes)
            for box in boxes:
                if (
                    not isinstance(box, list)
                    or
                    len(box) != 4
                    or not all(
                        isinstance(number, (int, float))
                        and not isinstance(number, bool)
                        and math.isfinite(number)
                        for number in box
                    )
                ):
                    raise ValueError("every projected box must contain four finite numbers")
                if float(box[0]) > float(box[1]) or float(box[2]) > float(box[3]):
                    raise ValueError("projected box bounds must be ordered")
                if projection["kind"] == "time-state":
                    expected_x = (
                        [expected_start, expected_end]
                        if value["view"] == "tube"
                        else [expected_end, expected_end]
                    )
                    if not all(
                        math.isclose(float(box[index]), expected_x[index], rel_tol=0.0,
                                     abs_tol=1e-15)
                        for index in (0, 1)
                    ):
                        raise ValueError(
                            "time-state box x bounds do not match frame/view semantics"
                        )
        if len(frame_lane_counts) != 1:
            raise ValueError("displayed frame lane counts must remain constant within a series")
        if source_kind == "torch-observer-pt-v1" and item.get(
            "observer_lane_count"
        ) != next(iter(frame_lane_counts)):
            raise ValueError("observer_lane_count does not match displayed frames")
    if projection["kind"] == "state-state" and total_boxes > MAX_STATE_STATE_BOXES:
        raise ValueError(f"geometry exceeds the {MAX_STATE_STATE_BOXES}-box render guard")
    binding = _validate_plot_spec(
        spec,
        benchmark,
        coordinate_names,
        instance_id=instance_id,
        numerical_horizon=expected_steps * step_size,
    )
    binding = _validate_plot_series_binding(
        binding, benchmark, instance_id, coordinate_names,
        step_size, expected_steps, value["series"],
    )
    if value.get("spec_binding") != binding:
        raise ValueError("geometry spec_binding is inconsistent with its plot spec")
    horizon = expected_steps * step_size
    for region in spec.get("regions", []):
        timing = region.get("time", {"kind": "all"})
        if timing.get("kind", "all") == "endpoint":
            lo = hi = float(timing["at"])
        elif timing.get("kind", "all") == "interval":
            lo, hi = float(timing["lo"]), float(timing["hi"])
        else:
            lo, hi = 0.0, horizon
        if lo < 0.0 or hi > horizon:
            raise ValueError("plot spec region lies outside the numerical horizon")
    _region_boxes(value)
    _region_thresholds(value)
    return value


def _region_boxes(geometry: dict[str, Any]) -> list[dict[str, Any]]:
    projection = geometry["projection"]
    spec = geometry.get("spec", {})
    regions = []
    for region in spec.get("regions", []):
        bounds = region.get("bounds", {})
        if projection["kind"] == "state-state":
            if projection["x"] not in bounds or projection["y"] not in bounds:
                continue
            timing = region.get("time", {"kind": "all"})
            xlo, xhi = bounds[projection["x"]]
            ylo, yhi = bounds[projection["y"]]
        else:
            if projection["y"] not in bounds:
                continue
            ylo, yhi = bounds[projection["y"]]
            timing = region.get("time", {"kind": "all"})
            kind = timing.get("kind", "all")
            if kind == "endpoint":
                xlo = xhi = float(timing["at"])
            elif kind == "interval":
                xlo, xhi = float(timing["lo"]), float(timing["hi"])
            elif kind == "all":
                xlo = 0.0
                xhi = geometry["expected_steps"] * geometry["step_size"]
            else:
                raise ValueError(f"unknown region time kind {kind!r}")
            horizon = geometry["expected_steps"] * geometry["step_size"]
            if xlo < 0.0 or xhi < xlo or xhi > horizon:
                raise ValueError("time-state region lies outside the numerical horizon")
        timing = region.get("time", {"kind": "all"})
        kind = timing.get("kind", "all")
        label = str(region.get("label", "Region"))
        if kind == "endpoint":
            display_label = f"{label} [endpoint t={float(timing['at']):.17g}]"
        elif kind == "interval":
            display_label = (
                f"{label} [t in [{float(timing['lo']):.17g},"
                f" {float(timing['hi']):.17g}]]"
            )
        else:
            display_label = label
        regions.append({
            **region,
            "display_label": display_label,
            "box": [float(xlo), float(xhi), float(ylo), float(yhi)],
        })
    return regions


def _region_thresholds(geometry: dict[str, Any]) -> list[dict[str, Any]]:
    projection = geometry["projection"]
    if projection["kind"] != "time-state":
        return []
    horizon = geometry["expected_steps"] * geometry["step_size"]
    thresholds = []
    for region in geometry.get("spec", {}).get("regions", []):
        constraint = region.get("constraint")
        if not isinstance(constraint, dict) or constraint.get("coordinate") != projection["y"]:
            continue
        timing = region.get("time", {"kind": "all"})
        kind = timing.get("kind", "all")
        if kind == "endpoint":
            xlo = xhi = float(timing["at"])
            time_label = f"endpoint t={xlo:.17g}"
        elif kind == "interval":
            xlo, xhi = float(timing["lo"]), float(timing["hi"])
            time_label = f"t in [{xlo:.17g}, {xhi:.17g}]"
        else:
            xlo, xhi = 0.0, horizon
            time_label = f"all t in [0, {horizon:.17g}]"
        value = float(constraint["value"])
        inequality = (
            f"{constraint['coordinate']} {constraint['operator']} {value:.17g}"
        )
        thresholds.append({
            **region,
            "display_label": (
                f"{region.get('label', 'Region')} [{inequality}; {time_label}]"
            ),
            "line": [xlo, xhi, value, value],
        })
    return thresholds


def _unprojected_region_labels(geometry: dict[str, Any]) -> list[str]:
    projection = geometry["projection"]
    labels = []
    for index, region in enumerate(geometry.get("spec", {}).get("regions", []), 1):
        constraint = region.get("constraint")
        if isinstance(constraint, dict):
            represented = (
                projection["kind"] == "time-state"
                and projection["y"] == constraint.get("coordinate")
            )
            if not represented:
                labels.append(
                    f"{region.get('label', f'Region {index}')} "
                    f"(derived coordinate {constraint.get('coordinate')} is not plotted)"
                )
            continue
        bounds = region.get("bounds", {})
        represented = (
            projection["x"] in bounds and projection["y"] in bounds
            if projection["kind"] == "state-state"
            else projection["y"] in bounds
        )
        if not represented:
            label = str(region.get("label", f"Region {index}"))
            constrained = sorted(set(bounds) & {projection["x"], projection["y"]})
            reason = (
                f"only {','.join(constrained)} is constrained"
                if constrained
                else "neither plotted coordinate is constrained"
            )
            labels.append(f"{label} ({reason})")
    return labels


def _initial_box(geometry: dict[str, Any]) -> list[float] | None:
    projection = geometry["projection"]
    bounds = geometry.get("spec", {}).get("initial_set", {}).get("bounds", {})
    if projection["kind"] == "time-state":
        if "y_transform" in projection:
            coordinate_names = geometry["coordinate_names"]
            if set(bounds) != set(coordinate_names):
                return None
            ylo, yhi = _affine_interval(
                [bounds[name] for name in coordinate_names],
                0,
                1,
                projection["y_transform"],
            )
            return [0.0, 0.0, ylo, yhi]
        if projection["y"] not in bounds:
            return None
        ylo, yhi = bounds[projection["y"]]
        return [0.0, 0.0, float(ylo), float(yhi)]
    if projection["x"] not in bounds or projection["y"] not in bounds:
        return None
    xlo, xhi = bounds[projection["x"]]
    ylo, yhi = bounds[projection["y"]]
    return [float(xlo), float(xhi), float(ylo), float(yhi)]


def _range_count(ranges: list[list[int]]) -> int:
    return sum(end - start + 1 for start, end in ranges)


def _short_value(value: Any, limit: int = 120) -> str:
    rendered = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return rendered if len(rendered) <= limit else rendered[:limit - 3] + "..."


def _run_evidence_note(item: dict[str, Any]) -> str:
    evidence = item["solver_run_evidence"]
    result = evidence.get("RESULT.json")
    binding = evidence.get("binding", {}).get("series", "unavailable")
    if not isinstance(result, dict):
        return "run RESULT evidence unavailable"
    prefix = (
        "identity-bound run evidence"
        if binding in RESULT_IDENTITY_BOUND
        else "adjacent run evidence UNBOUND to plotted series"
    )
    facts = []
    for key in ("status", "completed_step", "common_completed_step", "first_failure"):
        if key in result:
            facts.append(f"{key}={_short_value(result[key])}")
    for key in (
        "overall_horizon_completed", "all_steps_all_lanes_accepted",
        "requested_periods_completed", "full_T5_full1024_completed",
        "full_T5_root1_B2_completed", "original_full1024_completed",
        "fullbatch_qualification", "end_to_end_strict_certificate",
    ):
        if key in result:
            facts.append(f"{key}={str(result[key]).lower()}")
    return prefix + (": " + ", ".join(facts) if facts else ": status fields unavailable")


def _coverage_lines(geometry: dict[str, Any]) -> list[str]:
    lines = ["No interpolation; missing projection steps are not inferred as solver failures."]
    if "y_transform" in geometry["projection"]:
        lines.append(
            "Derived affine intervals are outward images of saved coordinate boxes; "
            "source-coordinate correlations are unavailable."
        )
    for item in geometry["series"]:
        displayed_partial_count = _range_count(item["displayed_partial_step_ranges"])
        projection_ranges = item["projection_partial_step_ranges"]
        partial_text = (
            f"projection partial unknown; displayed partial {displayed_partial_count}"
            if projection_ranges is None
            else (
                f"projection partial {_range_count(projection_ranges)}; "
                f"displayed partial {displayed_partial_count}"
            )
        )
        scope = (
            "record coverage only; acceptance unknown"
            if item["source_kind"] == "native-ranges-bin-v1"
            else f"accepted/status coverage; lane scope={item['lane_universe_binding']}"
        )
        lines.append(
            f"{item['label']}: projection-unobserved "
            f"{item['projection_unobserved_step_count']}/{item['expected_steps']}, "
            f"display-omitted {item['display_omitted_observed_step_count']}, "
            f"{partial_text}; {scope}."
        )
        lines.append(f"{item['label']}: {_run_evidence_note(item)}.")
        contract = item["solver_run_evidence"].get("plot_contract_check", {})
        declaration_binding = contract.get("declaration_binding", {})
        lines.append(
            f"{item['label']}: plot contract={contract.get('status', 'unavailable')}; "
            + ", ".join(
                f"{field}={contract.get(field, 'unavailable')}/"
                f"{declaration_binding.get(field, 'unavailable')}"
                for field in ("step_size", "expected_steps", "lane_count")
            )
            + "."
        )
    unprojected = _unprojected_region_labels(geometry)
    if unprojected:
        lines.append(
            "Plot-spec regions not drawn in this projection: " + "; ".join(unprojected) + "."
        )
    binding = geometry["spec_binding"]["identity_binding"]
    if binding == "informational_legacy_unbound":
        lines.append(
            geometry.get("spec", {}).get(
                "warning", "Legacy informational plot spec; not identity-bound."
            )
        )
    elif binding == "official_contract_sources_hash_declared":
        lines.append(
            f"Plot contract content: instance={geometry['instance_id']}; "
            "official source hashes declared; series benchmark, instance, coordinates, "
            "step size, and horizon identity verified from all observer sidecars."
        )
        lines.append(geometry["spec_binding"]["warning"])
    return lines


def _coverage_note(geometry: dict[str, Any]) -> str:
    return "\n".join(_coverage_lines(geometry))


def _series_legend_label(item: dict[str, Any], *, partial: bool) -> str:
    if item["source_kind"] == "native-ranges-bin-v1":
        suffix = (
            "partial record subset; acceptance unknown"
            if partial
            else "record-complete; acceptance unknown"
        )
    else:
        suffix = "partial accepted subset" if partial else "accepted lanes"
    return f"{item['label']} ({suffix})"


def _region_style(role: str) -> tuple[str, str, str]:
    return {
        "safe": ("0.1804 0.5451 0.3412", "0.0000 0.3922 0.0000", "-"),
        "target": ("0.6980 0.0941 0.5333", "0.4784 0.0275 0.3686", "-"),
        "unsafe": ("0.8392 0.1529 0.1569", "0.6471 0.0000 0.1490", "--"),
        "informational": ("0.4980 0.4980 0.4980", "0.2500 0.2500 0.2500", ":"),
    }[role]


def _region_mpl_style(role: str) -> tuple[str, str, str]:
    return {
        "safe": ("#2e8b57", "#006400", "-"),
        "target": ("#b2188b", "#7a075e", "-"),
        "unsafe": ("#d62728", "#a50026", "--"),
        "informational": ("#7f7f7f", "#404040", ":"),
    }[role]


def _matlab_quote(value: str) -> str:
    checked = _plain_text(value, "MATLAB text")
    assert checked is not None
    return checked.replace("'", "''")


def write_matlab(geometry: dict[str, Any], path: Path) -> None:
    """Write a self-contained MATLAB script; MATLAB execution is a separate check."""
    geometry = validate_geometry(geometry)
    derived_projection = "y_transform" in geometry["projection"]
    lines = [
        "% Generated by torch_tm_flowpipe.flowpipe_plot",
        (
            "% Axis-aligned interval image of a declared affine coordinate; "
            "source-coordinate correlations are unavailable."
            if derived_projection
            else "% Axis-aligned box projection; this is not a Flow* octagon."
        ),
        "% Missing observer steps are not interpolated.",
        "figure('Color','w'); hold on; box on; grid on;",
    ]
    colors = ["0.1216 0.4667 0.7059", "1.0000 0.4980 0.0549", "0.1725 0.6275 0.1725"]
    legend_handles = []
    legend_labels = []
    for index, item in enumerate(geometry["series"]):
        color = colors[index % len(colors)]
        handle = f"h_{index + 1}"
        lines.append(
            f"{handle} = plot(nan, nan, '-', 'Color', [{color}], 'LineWidth', 1.0);"
        )
        for partial, suffix in ((False, "complete"), (True, "partial")):
            boxes = [
                box
                for frame in item["frames"]
                if (not frame["complete"]) == partial
                for box in frame["boxes"]
            ]
            variable = f"boxes_{index + 1}_{suffix}"
            lines.append(f"{variable} = [")
            lines.extend(
                "  " + " ".join(format(float(value), ".17g") for value in box) + ";"
                for box in boxes
            )
            lines.append("];" )
            style = "--" if partial else "-"
            face_alpha = "0.02" if partial else "0.10"
            lines.append(f"for i = 1:size({variable},1)")
            lines.append(f"  b = {variable}(i,:);")
            lines.append("  if b(1) == b(2) && b(3) == b(4)")
            if partial:
                lines.append(
                    f"    plot(b(1), b(3), 'o', 'Color', [{color}], 'MarkerSize', 4);"
                )
            else:
                lines.append(
                    f"    plot(b(1), b(3), '.', 'Color', [{color}], 'MarkerSize', 8);"
                )
            lines.append("  elseif b(1) == b(2) || b(3) == b(4)")
            lines.append(
                f"    plot([b(1) b(2)], [b(3) b(4)], '{style}', "
                f"'Color', [{color}], 'LineWidth', 0.8);"
            )
            lines.append("  else")
            lines.append(
                f"    patch([b(1) b(2) b(2) b(1)], [b(3) b(3) b(4) b(4)], "
                f"[{color}], 'FaceAlpha', {face_alpha}, 'EdgeColor', [{color}], "
                f"'LineStyle', '{style}', 'LineWidth', 0.35);"
            )
            lines.append("  end")
            lines.append("end")
        if any(frame["complete"] and frame["boxes"] for frame in item["frames"]):
            legend_handles.append(handle)
            legend_labels.append(_series_legend_label(item, partial=False))
        if any(not frame["complete"] and frame["boxes"] for frame in item["frames"]):
            partial_handle = f"h_{index + 1}_partial"
            lines.append(
                f"{partial_handle} = plot(nan, nan, '--', 'Color', [{color}], "
                "'LineWidth', 1.0);"
            )
            legend_handles.append(partial_handle)
            legend_labels.append(_series_legend_label(item, partial=True))
    initial = _initial_box(geometry)
    if initial is not None:
        lines.append("initial_box = [" + " ".join(format(value, ".17g") for value in initial) + "];")
        lines.append("if initial_box(1) == initial_box(2) && initial_box(3) == initial_box(4)")
        lines.append("  h_initial = plot(initial_box(1), initial_box(3), 'ko', 'MarkerSize', 4);")
        lines.append("elseif initial_box(1) == initial_box(2) || initial_box(3) == initial_box(4)")
        lines.append("  h_initial = plot([initial_box(1) initial_box(2)], "
                     "[initial_box(3) initial_box(4)], 'k-', 'LineWidth', 1.2);")
        lines.append("else")
        lines.append("  h_initial = patch([initial_box(1) initial_box(2) initial_box(2) initial_box(1)], "
                     "[initial_box(3) initial_box(3) initial_box(4) initial_box(4)], 'w', "
                     "'FaceColor', 'none', 'EdgeColor', 'k', 'LineWidth', 1.2);")
        lines.append("end")
        legend_handles.append("h_initial")
        legend_labels.append(geometry.get("spec", {}).get("initial_set", {}).get("label", "Initial set"))
    for index, region in enumerate(_region_boxes(geometry), 1):
        box = region["box"]
        face_color, edge_color, line_style = _region_style(
            region.get("role", "informational")
        )
        lines.append(f"region_{index} = [" + " ".join(format(value, ".17g") for value in box) + "];")
        lines.append(
            f"if region_{index}(1) == region_{index}(2) && "
            f"region_{index}(3) == region_{index}(4)"
        )
        lines.append(
            f"  h_region_{index} = plot(region_{index}(1), region_{index}(3), "
            f"'o', 'Color', [{edge_color}], 'MarkerSize', 5);"
        )
        lines.append(
            f"elseif region_{index}(1) == region_{index}(2) || "
            f"region_{index}(3) == region_{index}(4)"
        )
        lines.append(
            f"  h_region_{index} = plot([region_{index}(1) region_{index}(2)], "
            f"[region_{index}(3) region_{index}(4)], '{line_style}', "
            f"'Color', [{edge_color}], 'LineWidth', 2);"
        )
        lines.append("else")
        lines.append(
            f"  h_region_{index} = patch([region_{index}(1) region_{index}(2) "
            f"region_{index}(2) region_{index}(1)], [region_{index}(3) "
            f"region_{index}(3) region_{index}(4) region_{index}(4)], "
            f"[{face_color}], 'FaceAlpha', 0.12, 'EdgeColor', [{edge_color}], "
            f"'LineStyle', '{line_style}', 'LineWidth', 1.0);"
        )
        lines.append("end")
        legend_handles.append(f"h_region_{index}")
        legend_labels.append(region.get("display_label", region.get("label", f"Region {index}")))
    for index, region in enumerate(_region_thresholds(geometry), 1):
        xlo, xhi, ylo, yhi = region["line"]
        _, edge_color, line_style = _region_style(
            region.get("role", "informational")
        )
        if xlo == xhi:
            lines.append(
                f"h_threshold_{index} = plot({xlo:.17g}, {ylo:.17g}, 'o', "
                f"'Color', [{edge_color}], 'MarkerSize', 5);"
            )
        else:
            lines.append(
                f"h_threshold_{index} = plot([{xlo:.17g} {xhi:.17g}], "
                f"[{ylo:.17g} {yhi:.17g}], '{line_style}', "
                f"'Color', [{edge_color}], 'LineWidth', 2);"
            )
        legend_handles.append(f"h_threshold_{index}")
        legend_labels.append(region["display_label"])
    projection = geometry["projection"]
    units = geometry.get("spec", {}).get("units", {})
    x_label = projection["x"] + (f" [{units[projection['x']]}]" if projection["x"] in units else "")
    y_label = projection["y"] + (f" [{units[projection['y']]}]" if projection["y"] in units else "")
    lines.extend([
        f"xlabel('{_matlab_quote(x_label)}');",
        f"ylabel('{_matlab_quote(y_label)}');",
        f"title('{_matlab_quote(geometry['benchmark'])}: "
        f"{_matlab_quote(geometry['view'])} "
        f"{_matlab_quote('affine interval projection' if derived_projection else 'box projection')}');",
    ])
    if legend_handles:
        lines.append(
            "legend([" + " ".join(legend_handles) + "], {" + ", ".join(
                "'" + _matlab_quote(label) + "'" for label in legend_labels
            ) + "}, 'Location', 'best');"
        )
    coverage_lines = _coverage_lines(geometry)
    footer_height = min(0.30, 0.025 * len(coverage_lines) + 0.015)
    matlab_lines = "; ".join(
        "'" + _matlab_quote(line) + "'" for line in coverage_lines
    )
    lines.append(
        f"annotation('textbox', [0.01 0.005 0.98 {footer_height:.3f}], "
        f"'String', {{{matlab_lines}}}, 'EdgeColor', 'none', 'FontSize', 7, "
        "'Interpreter', 'none', 'FitBoxToText', 'on');"
    )
    lines.append(
        "% Use exportgraphics(gcf, 'plot.pdf', 'ContentType', 'vector') if desired."
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_matplotlib(geometry: dict[str, Any], prefix: Path) -> tuple[Path, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Patch, Rectangle

    geometry = validate_geometry(geometry)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(9.2, 5.8))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    handles = []
    for index, item in enumerate(geometry["series"]):
        color = colors[index % len(colors)]
        complete_patches = []
        partial_patches = []
        complete_lines = []
        partial_lines = []
        complete_points = []
        partial_points = []
        for frame in item["frames"]:
            for xlo, xhi, ylo, yhi in frame["boxes"]:
                if xlo == xhi and ylo == yhi:
                    target = partial_points if not frame["complete"] else complete_points
                    target.append((xlo, ylo))
                elif xlo == xhi or ylo == yhi:
                    target = partial_lines if not frame["complete"] else complete_lines
                    target.append(([xlo, xhi], [ylo, yhi]))
                else:
                    target = partial_patches if not frame["complete"] else complete_patches
                    target.append(Rectangle((xlo, ylo), xhi - xlo, yhi - ylo))
        if complete_patches:
            axis.add_collection(PatchCollection(
                complete_patches, facecolor=color, edgecolor=color, linewidth=.25, alpha=.14
            ))
        if partial_patches:
            axis.add_collection(PatchCollection(
                partial_patches, facecolor="none", edgecolor=color, linewidth=.8,
                linestyle="--", alpha=.9
            ))
        for x, y in complete_lines:
            axis.plot(x, y, color=color, linewidth=.75, alpha=.8)
        for x, y in partial_lines:
            axis.plot(x, y, color=color, linewidth=.9, linestyle="--", alpha=.9)
        if complete_points:
            axis.scatter(*zip(*complete_points), color=color, s=8, alpha=.8)
        if partial_points:
            axis.scatter(*zip(*partial_points), facecolors="none", edgecolors=color, s=16)
        has_complete = any(frame["complete"] and frame["boxes"] for frame in item["frames"])
        has_partial = any(not frame["complete"] and frame["boxes"] for frame in item["frames"])
        if has_complete:
            handles.append(Patch(
                facecolor=color, edgecolor=color, alpha=.25,
                label=_series_legend_label(item, partial=False)
            ))
        if has_partial:
            handles.append(Patch(
                fill=False, edgecolor=color, linestyle="--",
                label=_series_legend_label(item, partial=True)
            ))
    initial = _initial_box(geometry)
    if initial is not None:
        xlo, xhi, ylo, yhi = initial
        if xlo == xhi and ylo == yhi:
            axis.plot(xlo, ylo, "ko", markersize=4)
        elif xlo == xhi or ylo == yhi:
            axis.plot([xlo, xhi], [ylo, yhi], "k-", linewidth=1.3)
        else:
            axis.add_patch(Rectangle((xlo, ylo), xhi - xlo, yhi - ylo,
                                     fill=False, edgecolor="black", linewidth=1.3))
        handles.append(Patch(fill=False, edgecolor="black", label=geometry.get("spec", {}).get(
            "initial_set", {}).get("label", "Initial set")))
    for region in _region_boxes(geometry):
        xlo, xhi, ylo, yhi = region["box"]
        face_color, edge_color, line_style = _region_mpl_style(
            region.get("role", "informational")
        )
        if xlo == xhi and ylo == yhi:
            axis.plot(xlo, ylo, "o", color=edge_color, markersize=5)
        elif xlo == xhi or ylo == yhi:
            axis.plot([xlo, xhi], [ylo, yhi], color=edge_color,
                      linestyle=line_style, linewidth=2)
        else:
            axis.add_patch(Rectangle((xlo, ylo), xhi - xlo, yhi - ylo,
                                     facecolor=face_color, edgecolor=edge_color, alpha=.12,
                                     linestyle=line_style, linewidth=1.0))
        handles.append(Patch(
            facecolor=face_color,
            edgecolor=edge_color,
            alpha=.2,
            linestyle=line_style,
            label=region.get("display_label", region.get("label", "Region")),
        ))
    for region in _region_thresholds(geometry):
        xlo, xhi, ylo, yhi = region["line"]
        _, edge_color, line_style = _region_mpl_style(
            region.get("role", "informational")
        )
        if xlo == xhi:
            axis.plot(xlo, ylo, "o", color=edge_color, markersize=5)
        else:
            axis.plot([xlo, xhi], [ylo, yhi], color=edge_color,
                      linestyle=line_style, linewidth=2)
        handles.append(Patch(
            fill=False,
            edgecolor=edge_color,
            linestyle=line_style,
            label=region["display_label"],
        ))
    projection = geometry["projection"]
    units = geometry.get("spec", {}).get("units", {})
    x_unit = units.get(projection["x"], "")
    y_unit = units.get(projection["y"], "")
    axis.set_xlabel(projection["x"] + (f" [{x_unit}]" if x_unit else ""))
    axis.set_ylabel(projection["y"] + (f" [{y_unit}]" if y_unit else ""))
    projection_label = (
        "affine interval projection"
        if "y_transform" in projection
        else "box projection"
    )
    axis.set_title(f"{geometry['benchmark']}: {geometry['view']} {projection_label}")
    axis.grid(alpha=.2)
    axis.autoscale()
    if handles:
        axis.legend(handles=handles, loc="best", fontsize=8)
    coverage_lines = [
        (
            "Axis-aligned interval image of a declared affine coordinate; "
            "source-coordinate correlations unavailable."
            if "y_transform" in projection
            else "Axis-aligned box projection (not octagon)."
        ),
        *_coverage_lines(geometry),
    ]
    fig.text(
        .01,
        .01,
        "\n".join(coverage_lines),
        fontsize=7,
        color="#333333",
        wrap=True,
    )
    footer_margin = min(.34, .025 + .024 * len(coverage_lines))
    fig.tight_layout(rect=(0, footer_margin, 1, 1))
    png = prefix.with_suffix(".png")
    pdf = prefix.with_suffix(".pdf")
    fig.savefig(png, dpi=240, metadata={"Software": "torch_tm_flowpipe.flowpipe_plot"})
    fig.savefig(pdf, metadata={"Creator": "torch_tm_flowpipe.flowpipe_plot"})
    plt.close(fig)
    return png, pdf


def _parse_series(values: list[str]) -> list[tuple[str, Path]]:
    output = []
    for value in values:
        if "=" not in value:
            raise ValueError("--series must use LABEL=OBSERVER_DIRECTORY_OR_RANGES_BIN")
        label, root = value.split("=", 1)
        output.append((label.strip(), Path(root).expanduser()))
    return output


def _artifact_receipt(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }


def _run_timing_receipts(geometry: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    for item in geometry["series"]:
        evidence = item["solver_run_evidence"]
        result = evidence.get("RESULT.json", {})
        steps = evidence.get("steps.jsonl", {})
        output.append({
            "label": item["label"],
            "result_evidence_binding": evidence["binding"].get(
                "result_series", evidence["binding"].get("series", "unavailable")
            ),
            "process_s": result.get("process_s"),
            "native_process_s": result.get("native_process_s"),
            "supervisor_process_s": result.get("supervisor_process_s"),
            "advance_s_sum": steps.get("advance_s_sum"),
            "advance_s_binding": steps.get(
                "binding", "steps.jsonl unavailable"
            ),
            "advance_s_rows": steps.get("advance_s_rows"),
            "steps_row_count": steps.get("row_count"),
            "steps_first_step": steps.get("first_step"),
            "steps_last_step": steps.get("last_step"),
            "advance_s_coverage": (
                "all_steps_rows"
                if steps.get("row_count") is not None
                and steps.get("row_count") > 0
                and steps.get("advance_s_rows") == steps.get("row_count")
                else (
                    "partial_steps_rows"
                    if steps.get("advance_s_rows")
                    else "unavailable"
                )
            ),
            "observer_generation_s": None,
            "observer_generation_note": (
                "not recoverable from these saved artifacts unless separately recorded by the run"
            ),
        })
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export and render box projections from saved flowpipe observers"
    )
    parser.add_argument("--series", action="append", default=[], metavar="LABEL=PATH")
    parser.add_argument("--geometry", type=Path, help="redraw an existing geometry JSON")
    parser.add_argument("--benchmark", default="unknown")
    parser.add_argument(
        "--instance-id",
        help="exact benchmark instance id (required by v2/v3 official plot specs)",
    )
    parser.add_argument("--projection", default="t,x1")
    parser.add_argument("--view", choices=sorted(VIEW_COLUMNS), default="tube")
    parser.add_argument("--step-size", type=float)
    parser.add_argument("--coordinate-names", default="")
    parser.add_argument(
        "--state-count", type=int,
        help="physical state count; defaults to 12 only when coordinate names are omitted",
    )
    parser.add_argument(
        "--expected-steps", type=int,
        help="full numerical horizon in steps (required for source export)",
    )
    parser.add_argument(
        "--expected-lanes", type=int,
        help="expected lane count for native ranges.bin inputs",
    )
    parser.add_argument(
        "--display-steps",
        help="comma-separated saved steps/ranges to draw, e.g. 1,20,100-120 (default: all observed)",
    )
    parser.add_argument("--partial-policy", choices=("reject", "mark"), default="reject")
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="output path prefix")
    parser.add_argument("--no-render", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    total_started = time.perf_counter()
    args = _parser().parse_args(argv)
    source_started = time.perf_counter()
    geometry_artifact: Path
    if args.geometry:
        if args.series:
            raise ValueError("--geometry and --series are mutually exclusive")
        geometry = validate_geometry(json.loads(args.geometry.read_text(encoding="utf-8")))
        if args.instance_id is not None and geometry.get("instance_id") != args.instance_id:
            raise ValueError("--instance-id does not match the saved geometry")
        geometry_artifact = args.geometry
        mode = "geometry-redraw"
    else:
        if not args.series:
            raise ValueError("at least one --series is required when --geometry is absent")
        if args.state_count is not None and args.state_count < 1:
            raise ValueError("state count must be positive")
        if args.coordinate_names:
            names = [part.strip() for part in args.coordinate_names.split(",")]
            if args.state_count is not None and len(names) != args.state_count:
                raise ValueError("--state-count must match --coordinate-names")
        else:
            state_count = args.state_count or 12
            names = [f"x{index}" for index in range(1, state_count + 1)]
        spec = json.loads(args.spec.read_text(encoding="utf-8")) if args.spec else {}
        if args.spec and not isinstance(spec, dict):
            raise ValueError("explicit --spec JSON must be an object")
        if args.expected_steps is None:
            raise ValueError("--expected-steps is required when exporting source data")
        if args.step_size is None:
            raise ValueError("--step-size is required when exporting source data")
        geometry = export_geometry(
            _parse_series(args.series),
            benchmark=args.benchmark,
            instance_id=args.instance_id,
            coordinate_names=names,
            projection_text=args.projection,
            view=args.view,
            step_size=args.step_size,
            expected_steps=args.expected_steps,
            expected_lanes=args.expected_lanes,
            display_steps=_parse_display_steps(args.display_steps, args.expected_steps),
            partial_policy=args.partial_policy,
            spec=spec,
        )
        geometry_artifact = args.output.with_suffix(".geometry.json")
        _json_dump(geometry_artifact, geometry)
        mode = "source-export"
    source_seconds = time.perf_counter() - source_started
    matlab_path = args.output.with_suffix(".m")
    matlab_started = time.perf_counter()
    write_matlab(geometry, matlab_path)
    matlab_seconds = time.perf_counter() - matlab_started
    rendered: tuple[Path, Path] | None = None
    render_seconds = None
    if not args.no_render:
        render_started = time.perf_counter()
        rendered = render_matplotlib(geometry, args.output)
        render_seconds = time.perf_counter() - render_started
    artifacts = {
        "geometry": _artifact_receipt(geometry_artifact),
        "matlab": _artifact_receipt(matlab_path),
        "png": None,
        "pdf": None,
    }
    if rendered is not None:
        artifacts["png"] = _artifact_receipt(rendered[0])
        artifacts["pdf"] = _artifact_receipt(rendered[1])
    receipt_path = args.output.with_suffix(".render.json")
    receipt = {
        "schema": "torch-tm-flowpipe-render-receipt-v1",
        "mode": mode,
        "timings_seconds": {
            "source_read_and_projection_export_or_geometry_validation": source_seconds,
            "matlab_script_generation": matlab_seconds,
            "matplotlib_png_pdf_render": render_seconds,
            "total_before_receipt_write": time.perf_counter() - total_started,
        },
        "run_timing_evidence": _run_timing_receipts(geometry),
        "artifacts": artifacts,
        "matlab_runtime_check": {
            "status": "not_performed_by_generator",
            "note": "script generation is not a MATLAB/Octave execution claim",
        },
    }
    if geometry.get("instance_id") is not None:
        receipt["instance_id"] = geometry["instance_id"]
    _json_dump(receipt_path, receipt)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        raise SystemExit(f"flowpipe-plot: {error}") from error
