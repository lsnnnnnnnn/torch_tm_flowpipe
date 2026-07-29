"""Shared schema, interval helpers, and benchmark utilities for the deep study."""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
SCHEMA_VERSION = 1


def load_spec(path: str | Path = HERE / "benchmark_spec.yaml") -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if value.get("schema_version") != 1:
        raise ValueError("unsupported benchmark schema")
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
) -> dict[str, Any]:
    time_index = (
        list(variable_roles).index("local_time")
        if "local_time" in variable_roles
        else None
    )
    return {
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
        "states": [decorate_state(dict(state), time_index) for state in states],
        "raw_endpoint": [
            decorate_state(dict(state), None) for state in raw_endpoint
        ],
        "raw_endpoint_box": [list(map(float, box)) for box in raw_endpoint_box],
        "whole_tube_box": [list(map(float, box)) for box in tube_box],
        "validation_trace": list(validation_trace),
        "reset_preconditioning_metadata": dict(reset_metadata),
        "native_metadata": dict(native_metadata),
    }


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
    }
    missing = sorted(required - set(record))
    if missing:
        raise ValueError(f"record missing fields: {missing}")
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
    return {
        "point_evaluation_checks": point_checks,
        "tube_point_violations": tube_violations,
        "endpoint_evaluation_violations": endpoint_violations,
        "endpoint_vs_tube_violations": endpoint_tube_violations,
        "passed": not (tube_violations or endpoint_violations or endpoint_tube_violations),
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
        "native_structured_symbolic_remainder": None,
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
