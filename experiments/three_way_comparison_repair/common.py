"""Shared definitions for the corrected three-way comparison.

The repair deliberately keeps the experiment schema independent from the
historical common-contract runner.  In particular, ``endpoint_raw`` and
``endpoint_tightened`` can never alias through an interval-kind rename.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
SPEC_PATH = HERE / "benchmark_spec.yaml"
FROZEN_RESULT = (
    REPO_ROOT
    / "experiments"
    / "three_way_common_contract"
    / "results"
    / "20260724T132534Z"
)

PROTOCOL_TUBE = "one_step_tube"
PROTOCOL_RAW = "one_step_raw_endpoint"
PROTOCOL_BOX = "common_box_raw_endpoint_carry"
PROTOCOL_NATIVE = "native_representation"
PROTOCOL_STRESS = "deliberate_low_order_stress"
PROTOCOL_SANITY = "known_working_tool_sanity"
PROTOCOLS = (
    PROTOCOL_TUBE,
    PROTOCOL_RAW,
    PROTOCOL_BOX,
    PROTOCOL_NATIVE,
    PROTOCOL_STRESS,
    PROTOCOL_SANITY,
)

FAILURE_CATEGORIES = {
    "order_configuration_rejected",
    "first_picard_inclusion_failed",
    "candidate_remainder_too_small",
    "fixed_order_exhausted",
    "adaptive_order_max_reached",
    "fixed_step_validation_failed",
    "adaptive_step_min_reached",
    "refinement_non_subset",
    "nonfinite_polynomial",
    "nonfinite_remainder",
    "composition_failure",
    "extraction_failure",
    "wrapper_failure",
    "unknown_internal_failure",
}

RAW_FIELDS = [
    "run_id",
    "timestamp",
    "tool",
    "tool_variant",
    "protocol",
    "system",
    "h",
    "requested_horizon",
    "step_index",
    "absolute_time",
    "state_index",
    "interval_kind",
    "lower",
    "upper",
    "width",
    "exact_lower",
    "exact_upper",
    "lower_error",
    "upper_error",
    "exact_width",
    "inflation_ratio",
    "native_validation_status",
    "analytic_reference_status",
    "sampled_trajectory_status",
    "failure_category",
    "failure_message",
    "local_order",
    "local_basis",
    "carried_representation",
    "step_policy",
    "cutoff",
    "configured_candidate_remainder",
    "native_returned_remainder",
    "postprocessed_remainder",
    "remainder_overwrite_applied",
    "endpoint_tightening_applied",
    "endpoint_semantics",
    "polynomial_width",
    "remainder_width",
    "build_time_s",
    "warmup_time_s",
    "steady_runtime_s",
    "dtype",
    "device",
    "repository_sha",
    "environment",
]


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def load_spec(path: str | Path = SPEC_PATH) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        spec = yaml.safe_load(handle)
    if not isinstance(spec, dict) or int(spec.get("schema_version", 0)) != 2:
        raise ValueError(f"invalid repair benchmark specification: {path}")
    return spec


def git_sha(path: str | Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=path, text=True
    ).strip()


def exact_steps(h: float, horizon: float) -> int:
    count = int(round(float(horizon) / float(h)))
    if not math.isclose(count * h, horizon, abs_tol=1e-12):
        raise ValueError(f"{horizon=} is not an integer multiple of {h=}")
    return count


def exact_endpoint(
    system: str,
    t: float,
    initial_box: Sequence[Sequence[float]],
) -> list[tuple[float, float]] | None:
    if system == "riccati":
        lo, hi = map(float, initial_box[0])
        return [(lo / (1.0 - t * lo), hi / (1.0 - t * hi))]
    if system == "harmonic":
        c, s = math.cos(t), math.sin(t)
        boxes = [tuple(map(float, bounds)) for bounds in initial_box]

        def hull(a: float, x: tuple[float, float], b: float, y: tuple[float, float]):
            values = [a * x[i] + b * y[j] for i in (0, 1) for j in (0, 1)]
            return min(values), max(values)

        return [hull(c, boxes[0], s, boxes[1]), hull(-s, boxes[0], c, boxes[1])]
    return None


def exact_tube(
    system: str,
    t0: float,
    t1: float,
    initial_box: Sequence[Sequence[float]],
) -> list[tuple[float, float]] | None:
    if system == "riccati":
        start = exact_endpoint(system, t0, initial_box)
        end = exact_endpoint(system, t1, initial_box)
        assert start is not None and end is not None
        return [(min(start[0][0], end[0][0]), max(start[0][1], end[0][1]))]
    if system != "harmonic":
        return None
    candidates = [t0, t1]
    lo_k = math.floor(t0 / (math.pi / 4)) - 2
    hi_k = math.ceil(t1 / (math.pi / 4)) + 2
    candidates.extend(
        k * math.pi / 4
        for k in range(lo_k, hi_k + 1)
        if t0 <= k * math.pi / 4 <= t1
    )
    boxes = [exact_endpoint(system, value, initial_box) for value in candidates]
    return [
        (
            min(box[state][0] for box in boxes if box is not None),
            max(box[state][1] for box in boxes if box is not None),
        )
        for state in range(2)
    ]


def reference_for_row(
    system: str,
    interval_kind: str,
    absolute_time: float,
    h: float,
    initial_box: Sequence[Sequence[float]],
) -> list[tuple[float, float]] | None:
    if interval_kind in {"endpoint_raw", "endpoint_tightened"}:
        return exact_endpoint(system, absolute_time, initial_box)
    if interval_kind == "tube":
        return exact_tube(
            system, max(0.0, absolute_time - h), absolute_time, initial_box
        )
    return None


def make_run_id(
    tool: str,
    variant: str,
    protocol: str,
    system: str,
    h: float | str,
    horizon: float | str,
) -> str:
    h_text = f"{h:.17g}" if isinstance(h, (int, float)) else str(h)
    horizon_text = (
        f"{horizon:.17g}"
        if isinstance(horizon, (int, float))
        else str(horizon)
    )
    payload = f"{tool}|{variant}|{protocol}|{system}|{h_text}|{horizon_text}"
    return f"{tool}_{hashlib.sha256(payload.encode()).hexdigest()[:16]}"


def make_row(
    *,
    tool: str,
    variant: str,
    protocol: str,
    system: str,
    h: float,
    horizon: float,
    step_index: int,
    absolute_time: float,
    state_index: int,
    interval_kind: str,
    lower: float | str,
    upper: float | str,
    exact: tuple[float, float] | None,
    native_validation_status: str,
    analytic_reference_status: str,
    sampled_trajectory_status: str = "not_checked",
    failure_category: str = "",
    failure_message: str = "",
    local_order: int | str = "",
    local_basis: str = "",
    carried_representation: str = "",
    step_policy: str = "",
    cutoff: float | str = "",
    configured_candidate_remainder: float | str = "",
    native_returned_remainder: float | str = "",
    postprocessed_remainder: float | str = "",
    remainder_overwrite_applied: bool = False,
    endpoint_tightening_applied: bool = False,
    endpoint_semantics: str = "",
    polynomial_width: float | str = "",
    remainder_width: float | str = "",
    build_time_s: float | str = "",
    warmup_time_s: float | str = "",
    steady_runtime_s: float | str = "",
    dtype: str = "",
    device: str = "cpu",
    repository_sha: str = "",
    environment: str = "",
) -> dict[str, Any]:
    finite_bounds = lower != "" and upper != ""
    width: float | str = float(upper) - float(lower) if finite_bounds else ""
    exact_lo: float | str = "" if exact is None else exact[0]
    exact_hi: float | str = "" if exact is None else exact[1]
    exact_width: float | str = "" if exact is None else exact[1] - exact[0]
    lower_error: float | str = "" if exact is None or not finite_bounds else float(lower) - exact[0]
    upper_error: float | str = "" if exact is None or not finite_bounds else exact[1] - float(upper)
    inflation: float | str = ""
    if isinstance(exact_width, float) and exact_width > 0 and isinstance(width, float):
        inflation = width / exact_width
    return {
        "run_id": make_run_id(tool, variant, protocol, system, h, horizon),
        "timestamp": timestamp(),
        "tool": tool,
        "tool_variant": variant,
        "protocol": protocol,
        "system": system,
        "h": h,
        "requested_horizon": horizon,
        "step_index": step_index,
        "absolute_time": absolute_time,
        "state_index": state_index,
        "interval_kind": interval_kind,
        "lower": lower,
        "upper": upper,
        "width": width,
        "exact_lower": exact_lo,
        "exact_upper": exact_hi,
        "lower_error": lower_error,
        "upper_error": upper_error,
        "exact_width": exact_width,
        "inflation_ratio": inflation,
        "native_validation_status": native_validation_status,
        "analytic_reference_status": analytic_reference_status,
        "sampled_trajectory_status": sampled_trajectory_status,
        "failure_category": failure_category,
        "failure_message": failure_message,
        "local_order": local_order,
        "local_basis": local_basis,
        "carried_representation": carried_representation,
        "step_policy": step_policy,
        "cutoff": cutoff,
        "configured_candidate_remainder": configured_candidate_remainder,
        "native_returned_remainder": native_returned_remainder,
        "postprocessed_remainder": postprocessed_remainder,
        "remainder_overwrite_applied": remainder_overwrite_applied,
        "endpoint_tightening_applied": endpoint_tightening_applied,
        "endpoint_semantics": endpoint_semantics,
        "polynomial_width": polynomial_width,
        "remainder_width": remainder_width,
        "build_time_s": build_time_s,
        "warmup_time_s": warmup_time_s,
        "steady_runtime_s": steady_runtime_s,
        "dtype": dtype,
        "device": device,
        "repository_sha": repository_sha,
        "environment": environment,
    }


def write_csv(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RAW_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_manifest(root: str | Path) -> list[dict[str, Any]]:
    root = Path(root)
    manifest: list[dict[str, Any]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        manifest.append(
            {
                "path": str(path.relative_to(root)),
                "size": path.stat().st_size,
                "sha256": digest.hexdigest(),
            }
        )
    return manifest


def manifest_digest(manifest: Sequence[Mapping[str, Any]]) -> str:
    payload = json.dumps(list(manifest), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()
