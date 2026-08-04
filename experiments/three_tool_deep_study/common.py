"""Shared schema, interval helpers, and benchmark utilities for the deep study."""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
CANONICAL_SPEC = REPO_ROOT / "benchmarks" / "canonical.yaml"
SCHEMA_VERSION = 2


def unavailable(reason: str) -> dict[str, str]:
    if not reason.strip():
        raise ValueError("unavailable values require a reason")
    return {"availability": "unavailable", "reason": reason}


def is_unavailable(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and value.get("availability") == "unavailable"
        and bool(value.get("reason"))
    )


def load_spec(path: str | Path = CANONICAL_SPEC) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if value.get("schema_version") != 1:
        raise ValueError("unsupported benchmark schema")
    work_parent = REPO_ROOT.parent
    repository_defaults = {
        "torch": REPO_ROOT,
        "torch_repaired_base": REPO_ROOT,
        "diffreach": work_parent / "DiffReach",
        "flowstar_original": work_parent / "flowstar",
    }
    repository_environment = {
        "torch": "TORCH_REPO_ROOT",
        "torch_repaired_base": "TORCH_REPAIRED_ROOT",
        "diffreach": "DIFFREACH_ROOT",
        "flowstar_original": "FLOWSTAR_ROOT",
    }
    configured = value.setdefault("repositories", {})
    # ``flowstar_audit`` is a historical diagnostic route.  Preserve it only
    # for an explicitly historical specification; never synthesize it for the
    # canonical supported runner.
    if "flowstar_audit" in configured:
        repository_defaults["flowstar_audit"] = work_parent / "flowstar-audit"
        repository_environment["flowstar_audit"] = "FLOWSTAR_AUDIT_ROOT"
    for name, default in repository_defaults.items():
        override = os.environ.get(repository_environment[name])
        configured_path = Path(str(configured.get(name, "")))
        if override:
            resolved = Path(override).expanduser().resolve()
        elif configured_path.is_absolute() and configured_path.exists():
            resolved = configured_path.resolve()
        else:
            resolved = default.resolve()
        configured[name] = str(resolved)
    return value


def git_sha(path: str | Path) -> str:
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha256_manifest(paths: Iterable[str | Path], root: str | Path) -> list[dict[str, Any]]:
    root_path = Path(root).resolve()
    rows: list[dict[str, Any]] = []
    for candidate in sorted({Path(path).resolve() for path in paths}):
        if not candidate.is_file():
            continue
        digest = hashlib.sha256()
        with candidate.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        try:
            name = str(candidate.relative_to(root_path))
        except ValueError:
            name = str(candidate)
        rows.append(
            {"path": name, "size": candidate.stat().st_size, "sha256": digest.hexdigest()}
        )
    return rows


def _down(value: float) -> float:
    return math.nextafter(float(value), -math.inf)


def _up(value: float) -> float:
    return math.nextafter(float(value), math.inf)


def interval_add(left: Sequence[float], right: Sequence[float]) -> list[float]:
    return [_down(left[0] + right[0]), _up(left[1] + right[1])]


def interval_mul(left: Sequence[float], right: Sequence[float]) -> list[float]:
    products = [
        left[0] * right[0],
        left[0] * right[1],
        left[1] * right[0],
        left[1] * right[1],
    ]
    return [_down(min(products)), _up(max(products))]


def interval_pow(value: Sequence[float], exponent: int) -> list[float]:
    if exponent == 0:
        return [1.0, 1.0]
    if exponent % 2 == 0 and value[0] <= 0.0 <= value[1]:
        return [0.0, _up(max(abs(value[0]), abs(value[1])) ** exponent)]
    candidates = [value[0] ** exponent, value[1] ** exponent]
    return [_down(min(candidates)), _up(max(candidates))]


def term_interval(
    coefficient: float,
    exponents: Sequence[int],
    domains: Sequence[Sequence[float]],
) -> list[float]:
    result = [float(coefficient), float(coefficient)]
    for exponent, domain in zip(exponents, domains):
        result = interval_mul(result, interval_pow(domain, int(exponent)))
    return result


def evaluate_polynomial_point(
    terms: Sequence[Mapping[str, Any]], point: Sequence[float]
) -> float:
    total = 0.0
    for term in terms:
        value = float(term["coefficient"])
        for coordinate, exponent in zip(point, term["exponents"]):
            value *= float(coordinate) ** int(exponent)
        total += value
    return total


def evaluate_polynomial_interval(
    terms: Sequence[Mapping[str, Any]], domains: Sequence[Sequence[float]]
) -> list[float]:
    result = [0.0, 0.0]
    for term in terms:
        result = interval_add(
            result,
            term_interval(term["coefficient"], term["exponents"], domains),
        )
    return result


def classify_exponent(exponents: Sequence[int], time_index: int | None) -> str:
    exp = tuple(int(value) for value in exponents)
    degree = sum(exp)
    time_degree = 0 if time_index is None else exp[time_index]
    state_degree = degree - time_degree
    if degree == 0:
        return "constant"
    if time_degree == degree:
        return "time_only"
    if time_degree and state_degree:
        return "time_state" if degree == 2 else "time_state_higher"
    if state_degree == 1:
        return "affine_state"
    if state_degree == 2:
        return "state_state"
    return "higher_order"


def summarize_terms(
    terms: Sequence[Mapping[str, Any]], time_index: int | None
) -> dict[str, Any]:
    families: dict[str, int] = {}
    max_degree = 0
    for term in terms:
        family = classify_exponent(term["exponents"], time_index)
        families[family] = families.get(family, 0) + 1
        max_degree = max(max_degree, sum(map(int, term["exponents"])))
    return {
        "term_count": len(terms),
        "max_degree": max_degree,
        "monomial_families": families,
    }


def decorate_state(state: dict[str, Any], time_index: int | None) -> dict[str, Any]:
    terms = state["polynomial_terms"]
    constant = 0.0
    affine: dict[str, float] = {}
    time_only: list[dict[str, Any]] = []
    time_state: list[dict[str, Any]] = []
    state_state: list[dict[str, Any]] = []
    higher: list[dict[str, Any]] = []
    for term in terms:
        family = classify_exponent(term["exponents"], time_index)
        if family == "constant":
            constant += float(term["coefficient"])
        elif family == "affine_state":
            affine[",".join(map(str, term["exponents"]))] = float(term["coefficient"])
        elif family == "time_only":
            time_only.append(term)
        elif family == "time_state":
            time_state.append(term)
        elif family == "state_state":
            state_state.append(term)
        elif family != "constant":
            higher.append(term)
    return {
        **state,
        "constant_term": constant,
        "affine_state_generator_coefficients": affine,
        "time_only_coefficients": time_only,
        "time_state_coefficients": time_state,
        "state_state_coefficients": state_state,
        "higher_order_coefficients": higher,
        **summarize_terms(terms, time_index),
    }


def canonical_record(
    *,
    tool: str,
    variant: str,
    system: str,
    h: float,
    variable_names: Sequence[str],
    variable_roles: Sequence[str],
    domains: Sequence[Sequence[float]],
    states: Sequence[Mapping[str, Any]],
    raw_endpoint: Sequence[Mapping[str, Any]],
    raw_endpoint_box: Sequence[Sequence[float]],
    tube_box: Sequence[Sequence[float]],
    validation_trace: Sequence[Mapping[str, Any]],
    reset_metadata: Mapping[str, Any],
    native_metadata: Mapping[str, Any],
    system_definition: Mapping[str, Any] | None = None,
    requested_horizon: float | None = None,
    segment_start: float = 0.0,
    accepted_step: float | None = None,
    tightened_endpoint: Sequence[Mapping[str, Any]] | None = None,
    tightened_endpoint_box: Sequence[Sequence[float]] | None = None,
    outcome: Mapping[str, Any] | None = None,
    execution_metadata: Mapping[str, Any] | None = None,
    basis_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    time_index = (
        list(variable_roles).index("local_time")
        if "local_time" in variable_roles
        else None
    )
    requested_horizon_value = (
        float(h) if requested_horizon is None else float(requested_horizon)
    )
    accepted_step_value: float | dict[str, str] = (
        unavailable("backend rejected the requested step")
        if accepted_step is None
        else float(accepted_step)
    )
    tightened_record: Any
    tightened_box_record: Any
    if tightened_endpoint is None:
        tightened_record = unavailable(
            "backend does not expose a distinct tightened endpoint"
        )
        tightened_box_record = unavailable(
            "backend does not expose a distinct tightened endpoint box"
        )
    else:
        if tightened_endpoint_box is None:
            raise ValueError(
                "tightened endpoint states require a tightened endpoint box"
            )
        tightened_record = [
            decorate_state(dict(state), None) for state in tightened_endpoint
        ]
        tightened_box_record = [
            list(map(float, box)) for box in tightened_endpoint_box
        ]
    system_record: Any = (
        dict(system_definition)
        if system_definition is not None
        else unavailable("system definition was not supplied by this adapter")
    )
    outcome_record = {
        "status": "success",
        "category": "",
        "reason": "",
        "requested_horizon_reached": True,
        **dict(outcome or {}),
    }
    execution_record = {
        "backend": unavailable("backend identifier was not supplied"),
        "dtype": unavailable("dtype was not exposed"),
        "device": unavailable("device was not exposed"),
        "repository_commit": unavailable("repository commit was not supplied"),
        "runtime": {
            "setup_s": unavailable("setup timing was not measured"),
            "propagation_s": unavailable("propagation timing was not measured"),
            "export_s": unavailable("export timing was not measured"),
        },
        **dict(execution_metadata or {}),
    }
    basis_record = {
        "name": unavailable("basis name was not supplied"),
        "requested_order": unavailable("requested order was not supplied"),
        "native_order": unavailable("native order was not exposed"),
        "coefficient_representation": "sparse_monomial_terms",
        **dict(basis_metadata or {}),
    }
    variable_semantics = [
        {"name": name, "role": role, "domain": list(map(float, domain))}
        for name, role, domain in zip(variable_names, variable_roles, domains)
    ]
    decorated_states = [
        decorate_state(dict(state), time_index) for state in states
    ]
    decorated_raw_endpoint = [
        decorate_state(dict(state), None) for state in raw_endpoint
    ]
    raw_endpoint_box_record = [
        list(map(float, box)) for box in raw_endpoint_box
    ]
    tube_box_record = [list(map(float, box)) for box in tube_box]
    record = {
        "schema_version": SCHEMA_VERSION,
        "tool": tool,
        "variant": variant,
        "system": system,
        "state_dimension": len(states),
        "h": float(h),
        "local_time_domain": [0.0, float(h)],
        "variable_names": list(variable_names),
        "variable_roles": list(variable_roles),
        "local_time_index": time_index,
        "generator_domains": [
            list(map(float, domain))
            for role, domain in zip(variable_roles, domains)
            if role != "local_time"
        ],
        "domains": [list(map(float, domain)) for domain in domains],
        "states": decorated_states,
        "raw_endpoint": decorated_raw_endpoint,
        "raw_endpoint_box": raw_endpoint_box_record,
        "whole_tube_box": tube_box_record,
        "validation_trace": list(validation_trace),
        "reset_preconditioning_metadata": dict(reset_metadata),
        "native_metadata": dict(native_metadata),
        "system_definition": system_record,
        "time_semantics": {
            "initial_time": 0.0,
            "requested_horizon": requested_horizon_value,
            "segment_start": float(segment_start),
            "segment_end": float(segment_start) + float(h),
        },
        "step_semantics": {
            "requested_step": float(h),
            "accepted_step": accepted_step_value,
        },
        "variable_semantics": variable_semantics,
        "dependency_semantics": {
            "generator_variables": [
                item["name"]
                for item in variable_semantics
                if item["role"]
                in {"state_generator", "dependency_generator", "noise_generator"}
            ],
            "noise_variables": [
                item["name"]
                for item in variable_semantics
                if item["role"] == "noise_generator"
            ],
            "independent_remainder_semantics": (
                "per-state interval independent of polynomial generators"
            ),
        },
        "polynomial_representation": basis_record,
        "enclosures": {
            "tube": {"states": decorated_states, "box": tube_box_record},
            "endpoint_raw": {
                "states": decorated_raw_endpoint,
                "box": raw_endpoint_box_record,
            },
            "endpoint_tightened": {
                "states": tightened_record,
                "box": tightened_box_record,
            },
        },
        "reset_carry_policy": dict(reset_metadata),
        "outcome": outcome_record,
        "execution": execution_record,
    }
    return record


def deterministic_points(domains: Sequence[Sequence[float]], limit: int = 32) -> list[list[float]]:
    axes = [[float(domain[0]), 0.5 * (domain[0] + domain[1]), float(domain[1])] for domain in domains]
    values = [list(point) for point in itertools.product(*axes)]
    if len(values) <= limit:
        return values
    stride = max(1, len(values) // limit)
    return values[::stride][:limit]


def validate_record(record: Mapping[str, Any], tolerance: float = 1e-10) -> dict[str, Any]:
    required = {
        "state_dimension",
        "local_time_domain",
        "generator_domains",
        "states",
        "raw_endpoint",
        "raw_endpoint_box",
        "whole_tube_box",
        "validation_trace",
        "reset_preconditioning_metadata",
        "system_definition",
        "time_semantics",
        "step_semantics",
        "variable_semantics",
        "dependency_semantics",
        "polynomial_representation",
        "enclosures",
        "reset_carry_policy",
        "outcome",
        "execution",
    }
    missing = sorted(required - set(record))
    if missing:
        raise ValueError(f"record missing fields: {missing}")
    if record.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported CIR schema version: {record.get('schema_version')}"
        )
    if is_unavailable(record["system_definition"]):
        raise ValueError("system definition is required for a CIR export")
    if len(record["variable_semantics"]) != len(record["domains"]):
        raise ValueError("variable semantics/domain length mismatch")
    for item, name, role, domain in zip(
        record["variable_semantics"],
        record["variable_names"],
        record["variable_roles"],
        record["domains"],
    ):
        if (
            item.get("name") != name
            or item.get("role") != role
            or list(item.get("domain", [])) != list(domain)
        ):
            raise ValueError("variable semantics do not match legacy fields")
    requested_step = record["step_semantics"].get("requested_step")
    accepted_step_value = record["step_semantics"].get("accepted_step")
    if not isinstance(requested_step, (int, float)):
        raise ValueError("requested step must be numeric")
    if not (
        isinstance(accepted_step_value, (int, float))
        or is_unavailable(accepted_step_value)
    ):
        raise ValueError("accepted step must be numeric or explicitly unavailable")
    tightened = record["enclosures"]["endpoint_tightened"]
    if not (
        (
            isinstance(tightened.get("states"), list)
            and isinstance(tightened.get("box"), list)
        )
        or (
            is_unavailable(tightened.get("states"))
            and is_unavailable(tightened.get("box"))
        )
    ):
        raise ValueError(
            "tightened endpoint must be populated or explicitly unavailable"
        )
    execution = record["execution"]
    for key in ("backend", "dtype", "device", "repository_commit", "runtime"):
        if key not in execution or execution[key] is None:
            raise ValueError(f"execution field {key} is missing")
    for state in list(record["states"]) + list(record["raw_endpoint"]):
        if state.get("native_structured_symbolic_remainder") is None:
            raise ValueError(
                "unavailable native structured remainders must be explicit"
            )
    domains = record["domains"]
    point_checks = 0
    tube_violations = 0
    for point in deterministic_points(domains):
        for index, state in enumerate(record["states"]):
            value = evaluate_polynomial_point(state["polynomial_terms"], point)
            rem = state["independent_interval_remainder"]
            possible = [value + rem[0], value + rem[1]]
            tube = record["whole_tube_box"][index]
            point_checks += 1
            if possible[0] < tube[0] - tolerance or possible[1] > tube[1] + tolerance:
                tube_violations += 1
    endpoint_violations = 0
    generator_domains = record["generator_domains"]
    for point in deterministic_points(generator_domains):
        for index, state in enumerate(record["raw_endpoint"]):
            value = evaluate_polynomial_point(state["polynomial_terms"], point)
            rem = state["independent_interval_remainder"]
            box = record["raw_endpoint_box"][index]
            if value + rem[0] < box[0] - tolerance or value + rem[1] > box[1] + tolerance:
                endpoint_violations += 1
    endpoint_tube_violations = sum(
        endpoint[0] < tube[0] - tolerance or endpoint[1] > tube[1] + tolerance
        for endpoint, tube in zip(record["raw_endpoint_box"], record["whole_tube_box"])
    )
    native_point_checks = 0
    native_point_violations = 0
    for sample in record.get("native_metadata", {}).get(
        "native_point_samples", []
    ):
        point = sample.get("point", [])
        if "polynomial_values" in sample:
            for state_index, expected in enumerate(
                sample["polynomial_values"]
            ):
                if state_index >= len(record["states"]):
                    native_point_violations += 1
                    continue
                actual = evaluate_polynomial_point(
                    record["states"][state_index]["polynomial_terms"],
                    point,
                )
                native_point_checks += 1
                if not math.isclose(
                    actual,
                    float(expected),
                    rel_tol=0.0,
                    abs_tol=tolerance,
                ):
                    native_point_violations += 1
        elif "total_interval" in sample:
            state_index = int(sample.get("state", 0))
            kind = str(sample.get("kind", "tube"))
            states = (
                record["raw_endpoint"]
                if kind == "endpoint"
                else record["states"]
            )
            if state_index >= len(states):
                native_point_violations += 1
                continue
            state = states[state_index]
            actual = evaluate_polynomial_point(
                state["polynomial_terms"], point
            )
            remainder = state["independent_interval_remainder"]
            actual_interval = [
                actual + float(remainder[0]),
                actual + float(remainder[1]),
            ]
            expected = list(map(float, sample["total_interval"]))
            native_point_checks += 1
            if any(
                not math.isclose(
                    got,
                    want,
                    rel_tol=0.0,
                    abs_tol=tolerance,
                )
                for got, want in zip(actual_interval, expected)
            ):
                native_point_violations += 1
    return {
        "schema_version": SCHEMA_VERSION,
        "schema_fields_passed": True,
        "point_evaluation_checks": point_checks,
        "tube_point_violations": tube_violations,
        "endpoint_evaluation_violations": endpoint_violations,
        "endpoint_vs_tube_violations": endpoint_tube_violations,
        "native_point_evaluation_checks": native_point_checks,
        "native_point_evaluation_violations": native_point_violations,
        "passed": not (
            tube_violations
            or endpoint_violations
            or endpoint_tube_violations
            or native_point_violations
        ),
    }


def affine_project_state(
    state: Mapping[str, Any], domains: Sequence[Sequence[float]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    kept: list[dict[str, Any]] = []
    discarded: list[dict[str, Any]] = []
    remainder = list(map(float, state["independent_interval_remainder"]))
    for term in state["polynomial_terms"]:
        degree = sum(map(int, term["exponents"]))
        if degree <= 1:
            kept.append(dict(term))
            continue
        contribution = term_interval(
            float(term["coefficient"]), term["exponents"], domains
        )
        remainder = interval_add(remainder, contribution)
        discarded.append({**dict(term), "range_contribution": contribution})
    return {
        "polynomial_terms": kept,
        "independent_interval_remainder": remainder,
        "native_structured_symbolic_remainder": unavailable(
            "affine projection has no native structured symbolic remainder"
        ),
    }, discarded


def analytic_endpoint(
    system: str, initial_box: Sequence[Sequence[float]], time_value: float
) -> list[list[float]] | None:
    t = float(time_value)
    if system == "riccati":
        lo, hi = map(float, initial_box[0])
        return [[lo / (1.0 - lo * t), hi / (1.0 - hi * t)]]
    if system == "harmonic":
        c, s = math.cos(t), math.sin(t)
        x, y = initial_box

        def affine(a: float, left: Sequence[float], b: float, right: Sequence[float]) -> list[float]:
            values = [
                a * lx + b * ry
                for lx in left
                for ry in right
            ]
            return [min(values), max(values)]

        return [
            affine(c, x, s, y),
            affine(-s, x, c, y),
        ]
    return None


def analytic_contained(
    system: str,
    initial_box: Sequence[Sequence[float]],
    time_value: float,
    candidate: Sequence[Sequence[float]],
    tolerance: float = 1e-12,
) -> bool | None:
    exact = analytic_endpoint(system, initial_box, time_value)
    if exact is None:
        return None
    return all(
        got[0] <= expected[0] + tolerance
        and got[1] >= expected[1] - tolerance
        for got, expected in zip(candidate, exact)
    )
