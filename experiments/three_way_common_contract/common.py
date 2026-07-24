"""Shared contracts, exact references, and schemas for the three-tool study."""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
SPEC_PATH = HERE / "benchmark_spec.yaml"

PROTOCOL_A = "one_step_common_input"
PROTOCOL_B = "multi_step_common_box_carry"
PROTOCOL_C = "native_low_order"
PROTOCOLS = (PROTOCOL_A, PROTOCOL_B, PROTOCOL_C)

RAW_FIELDS = [
    "run_id",
    "timestamp",
    "tool",
    "tool_variant",
    "protocol",
    "system",
    "state_index",
    "state_name",
    "h",
    "horizon",
    "step_index",
    "time",
    "interval_kind",
    "lower",
    "upper",
    "width",
    "exact_lower",
    "exact_upper",
    "exact_width",
    "exact_inflation_ratio",
    "row_status",
    "native_validation_status",
    "first_failure_time",
    "successful_horizon",
    "local_order",
    "local_retained_basis",
    "carried_representation",
    "reset_policy",
    "validator",
    "interval_remainder_width",
    "polynomial_width",
    "dtype",
    "device",
    "build_time_s",
    "jit_compile_time_s",
    "first_execution_time_s",
    "steady_runtime_per_step_s",
    "orchestration_time_s",
    "validation_attempts",
    "tool_git_sha",
    "adapter_git_sha",
    "extraction_workaround",
    "message",
]

RUN_FIELDS = [
    "run_id",
    "timestamp",
    "tool",
    "tool_variant",
    "protocol",
    "system",
    "h",
    "horizon",
    "requested_steps",
    "completed_steps",
    "run_status",
    "native_validation_status",
    "first_failure_time",
    "successful_horizon",
    "local_order",
    "local_retained_basis",
    "measured_polynomial_support",
    "carried_representation",
    "reset_policy",
    "validator",
    "dtype",
    "device",
    "build_time_s",
    "jit_compile_time_s",
    "first_execution_time_s",
    "steady_runtime_per_step_s",
    "orchestration_time_s",
    "validation_attempts",
    "tool_git_sha",
    "adapter_git_sha",
    "extraction_workaround",
    "message",
]


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def load_spec(path: str | Path = SPEC_PATH) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict) or tuple(value.get("protocols", ())) != PROTOCOLS:
        raise ValueError(f"invalid common-contract specification: {path}")
    return value


def exact_steps(h: float, horizon: float) -> int:
    ratio = float(horizon) / float(h)
    steps = int(round(ratio))
    if not math.isclose(steps * float(h), float(horizon), abs_tol=1e-12):
        raise ValueError(f"horizon {horizon} is not an integer multiple of h={h}")
    return steps


def iter_configurations(
    spec: Mapping[str, Any],
    *,
    smoke: bool = False,
    protocols: Sequence[str] | None = None,
    systems: Sequence[str] | None = None,
) -> Iterable[dict[str, Any]]:
    selected_protocols = tuple(protocols or PROTOCOLS)
    selected_systems = set(systems or spec["systems"].keys())
    for protocol in selected_protocols:
        if protocol not in PROTOCOLS:
            raise ValueError(f"unknown protocol: {protocol}")
        for system_name, protocol_spec in spec["protocols"][protocol].items():
            if system_name not in selected_systems:
                continue
            if protocol == PROTOCOL_A:
                step_sizes = (
                    [float(protocol_spec["smoke_h"])]
                    if smoke
                    else [float(value) for value in protocol_spec["step_sizes"]]
                )
                pairs = [(h, h) for h in step_sizes]
            elif smoke:
                pairs = [
                    (
                        float(protocol_spec["smoke"]["h"]),
                        float(protocol_spec["smoke"]["horizon"]),
                    )
                ]
            else:
                pairs = [
                    (float(item["h"]), float(item["horizon"]))
                    for item in protocol_spec["configurations"]
                ]
            for h, horizon in pairs:
                yield {
                    "protocol": protocol,
                    "system": system_name,
                    "h": h,
                    "horizon": horizon,
                    "steps": exact_steps(h, horizon),
                }


def configuration_key(
    tool: str,
    tool_variant: str,
    protocol: str,
    system: str,
    h: float,
    horizon: float,
) -> tuple[str, str, str, str, float, float]:
    return (
        str(tool),
        str(tool_variant),
        str(protocol),
        str(system),
        round(float(h), 12),
        round(float(horizon), 12),
    )


def make_run_id(
    tool: str,
    tool_variant: str,
    protocol: str,
    system: str,
    h: float,
    horizon: float,
) -> str:
    payload = "|".join(map(str, configuration_key(
        tool, tool_variant, protocol, system, h, horizon
    )))
    return f"{tool}_{tool_variant}_{protocol}_{system}_{hashlib.sha256(payload.encode()).hexdigest()[:10]}"


def power(value: Any, exponent: int) -> Any:
    if exponent == 0:
        return 1.0
    result = value
    for _ in range(1, exponent):
        result = result * value
    return result


def evaluate_rhs(state: Sequence[Any], system_spec: Mapping[str, Any]) -> list[Any]:
    outputs: list[Any] = []
    for polynomial in system_spec["rhs"]:
        result: Any = 0.0
        for term in polynomial["terms"]:
            value: Any = float(term["coefficient"])
            for coordinate, exponent in zip(state, term["powers"]):
                exponent = int(exponent)
                if exponent:
                    value = value * power(coordinate, exponent)
            result = result + value
        outputs.append(result)
    return outputs


def flowstar_expression(
    polynomial: Mapping[str, Any], state_names: Sequence[str]
) -> str:
    parts: list[str] = []
    for term in polynomial["terms"]:
        coefficient = float(term["coefficient"])
        factors: list[str] = []
        for name, exponent in zip(state_names, term["powers"]):
            exponent = int(exponent)
            if exponent == 1:
                factors.append(name)
            elif exponent > 1:
                factors.append(f"{name}^{exponent}")
        magnitude = abs(coefficient)
        if not factors or not math.isclose(magnitude, 1.0):
            factors.insert(0, f"{magnitude:.17g}")
        body = "*".join(factors) if factors else "0"
        if not parts:
            parts.append(body if coefficient >= 0 else f"-{body}")
        else:
            parts.append((" + " if coefficient >= 0 else " - ") + body)
    return "".join(parts) or "0"


def exact_endpoint(
    system: str,
    t: float,
    initial_box: Sequence[Sequence[float]],
) -> list[tuple[float, float]] | None:
    if system == "riccati":
        lo0, hi0 = map(float, initial_box[0])
        return [(lo0 / (1.0 - lo0 * t), hi0 / (1.0 - hi0 * t))]
    if system == "harmonic":
        c, s = math.cos(t), math.sin(t)
        (xlo, xhi), (ylo, yhi) = [
            tuple(map(float, bounds)) for bounds in initial_box
        ]

        def affine_hull(
            a: float,
            bounds_a: tuple[float, float],
            b: float,
            bounds_b: tuple[float, float],
        ) -> tuple[float, float]:
            values = [
                a * bounds_a[i] + b * bounds_b[j]
                for i in (0, 1)
                for j in (0, 1)
            ]
            return min(values), max(values)

        return [
            affine_hull(c, (xlo, xhi), s, (ylo, yhi)),
            affine_hull(-s, (xlo, xhi), c, (ylo, yhi)),
        ]
    return None


def exact_tube(
    system: str,
    t_lo: float,
    t_hi: float,
    initial_box: Sequence[Sequence[float]],
) -> list[tuple[float, float]] | None:
    if system == "riccati":
        start = exact_endpoint(system, t_lo, initial_box)
        end = exact_endpoint(system, t_hi, initial_box)
        assert start is not None and end is not None
        return [
            (min(start[0][0], end[0][0]), max(start[0][1], end[0][1]))
        ]
    if system != "harmonic":
        return None
    candidates = [float(t_lo), float(t_hi)]
    # Every extremum of sin/cos linear combinations lies on the pi/4 grid.
    k_min = math.floor((t_lo - math.pi) / (math.pi / 4))
    k_max = math.ceil((t_hi + math.pi) / (math.pi / 4))
    candidates.extend(
        k * math.pi / 4
        for k in range(k_min, k_max + 1)
        if t_lo <= k * math.pi / 4 <= t_hi
    )
    boxes = [
        exact_endpoint(system, value, initial_box)
        for value in candidates
    ]
    return [
        (
            min(box[state][0] for box in boxes if box is not None),
            max(box[state][1] for box in boxes if box is not None),
        )
        for state in range(2)
    ]


def exact_interval_for_row(
    system: str,
    interval_kind: str,
    time_value: float,
    h: float,
    initial_box: Sequence[Sequence[float]],
) -> list[tuple[float, float]] | None:
    if interval_kind == "endpoint":
        return exact_endpoint(system, time_value, initial_box)
    if interval_kind == "tube":
        return exact_tube(system, max(0.0, time_value - h), time_value, initial_box)
    return None


def deterministic_initial_points(
    initial_box: Sequence[Sequence[float]],
) -> list[tuple[float, ...]]:
    bounds = [tuple(map(float, pair)) for pair in initial_box]
    points = set(itertools.product(*[(lo, hi) for lo, hi in bounds]))
    midpoint = tuple((lo + hi) / 2.0 for lo, hi in bounds)
    points.add(midpoint)
    for coordinate in range(len(bounds)):
        for endpoint in bounds[coordinate]:
            point = list(midpoint)
            point[coordinate] = endpoint
            points.add(tuple(point))
    return sorted(points)


def interval_metrics(
    lower: float,
    upper: float,
    exact: tuple[float, float] | None,
) -> dict[str, Any]:
    width = float(upper) - float(lower)
    if exact is None:
        return {
            "width": width,
            "exact_lower": "",
            "exact_upper": "",
            "exact_width": "",
            "exact_inflation_ratio": "",
        }
    exact_width = float(exact[1]) - float(exact[0])
    return {
        "width": width,
        "exact_lower": exact[0],
        "exact_upper": exact[1],
        "exact_width": exact_width,
        "exact_inflation_ratio": width / exact_width if exact_width > 0 else "",
    }


def make_row(
    run: Mapping[str, Any],
    *,
    state_index: int,
    state_name: str,
    step_index: int,
    time_value: float,
    interval_kind: str,
    lower: float | str,
    upper: float | str,
    exact: tuple[float, float] | None = None,
    polynomial_width: float | str = "",
    interval_remainder_width: float | str = "",
    row_status: str | None = None,
    native_validation_status: str | None = None,
    message: str | None = None,
) -> dict[str, Any]:
    row = {field: run.get(field, "") for field in RAW_FIELDS}
    row.update(
        state_index=int(state_index),
        state_name=state_name,
        step_index=int(step_index),
        time=float(time_value),
        interval_kind=interval_kind,
        lower=lower,
        upper=upper,
        polynomial_width=polynomial_width,
        interval_remainder_width=interval_remainder_width,
    )
    if isinstance(lower, (int, float)) and isinstance(upper, (int, float)):
        row.update(interval_metrics(float(lower), float(upper), exact))
    if row_status is not None:
        row["row_status"] = row_status
    if native_validation_status is not None:
        row["native_validation_status"] = native_validation_status
    if message is not None:
        row["message"] = message
    return row


def base_run(
    *,
    tool: str,
    tool_variant: str,
    config: Mapping[str, Any],
    local_order: str | int,
    local_retained_basis: str,
    carried_representation: str,
    reset_policy: str,
    validator: str,
    dtype: str,
    device: str,
    tool_git_sha: str,
    adapter_git_sha: str,
    extraction_workaround: str = "",
) -> dict[str, Any]:
    return {
        "run_id": make_run_id(
            tool,
            tool_variant,
            str(config["protocol"]),
            str(config["system"]),
            float(config["h"]),
            float(config["horizon"]),
        ),
        "timestamp": utc_timestamp(),
        "tool": tool,
        "tool_variant": tool_variant,
        "protocol": config["protocol"],
        "system": config["system"],
        "h": config["h"],
        "horizon": config["horizon"],
        "requested_steps": config["steps"],
        "completed_steps": 0,
        "run_status": "pending",
        "row_status": "pending",
        "native_validation_status": "not_run",
        "first_failure_time": "",
        "successful_horizon": 0.0,
        "local_order": local_order,
        "local_retained_basis": local_retained_basis,
        "measured_polynomial_support": "",
        "carried_representation": carried_representation,
        "reset_policy": reset_policy,
        "validator": validator,
        "dtype": dtype,
        "device": device,
        "build_time_s": 0.0,
        "jit_compile_time_s": 0.0,
        "first_execution_time_s": "",
        "steady_runtime_per_step_s": "",
        "orchestration_time_s": "",
        "validation_attempts": "",
        "tool_git_sha": tool_git_sha,
        "adapter_git_sha": adapter_git_sha,
        "extraction_workaround": extraction_workaround,
        "message": "",
    }


def median(values: Sequence[float]) -> float:
    return statistics.median(float(value) for value in values) if values else math.nan


def git_sha(path: str | Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(
    path: str | Path,
    rows: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(fields),
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def copy_runtime_fields(run: Mapping[str, Any], rows: list[dict[str, Any]]) -> None:
    for row in rows:
        for field in (
            "first_failure_time",
            "successful_horizon",
            "build_time_s",
            "jit_compile_time_s",
            "first_execution_time_s",
            "steady_runtime_per_step_s",
            "orchestration_time_s",
            "validation_attempts",
        ):
            row[field] = run.get(field, "")
