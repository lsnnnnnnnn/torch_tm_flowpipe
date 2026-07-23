"""Shared canonical definitions and result schema for the three-way benchmark."""
from __future__ import annotations

import csv
import contextlib
import hashlib
import json
import math
import signal
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
SPEC_PATH = HERE / "benchmark_spec.yaml"

STATUS_VALUES = {
    "certified_ok",
    "produced_enclosure_unverified",
    "validation_failed",
    "contraction_failed",
    "unsupported_order",
    "timeout",
    "numerical_error",
    "sample_violation",
    "skipped",
}


@contextlib.contextmanager
def configuration_timeout(seconds: float):
    """Raise TimeoutError after a wall-clock configuration budget on POSIX."""
    seconds = float(seconds)
    if seconds <= 0 or not hasattr(signal, "setitimer"):
        yield
        return
    previous_handler = signal.getsignal(signal.SIGALRM)

    def _raise_timeout(signum: int, frame: Any) -> None:
        del signum, frame
        raise TimeoutError(f"configuration exceeded {seconds:g} seconds")

    signal.signal(signal.SIGALRM, _raise_timeout)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous_timer)
        signal.signal(signal.SIGALRM, previous_handler)

RAW_FIELDS = [
    "run_id", "timestamp", "tool", "protocol", "system", "state_index", "h",
    "horizon", "step_index", "time", "interval_kind", "lower", "upper",
    "width", "status", "validation_status", "first_failure_time",
    "requested_order_label", "retained_basis", "effective_max_degree",
    "truncate_to_affine", "nonzero_Lt", "dependency_mode",
    "symbolic_remainder_size", "cutoff", "partitions", "dtype", "device",
    "batch_size", "build_time_s", "warmup_time_s",
    "steady_runtime_median_s", "steady_runtime_iqr_s", "git_commit",
    "environment", "validation_attempts", "successful_horizon", "message",
]


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def load_spec(path: str | Path = SPEC_PATH) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict) or "systems" not in data:
        raise ValueError(f"invalid benchmark specification: {path}")
    return data


def resolved_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    out = json.loads(json.dumps(spec))
    out["configurations"] = list(iter_configurations(spec, smoke=False))
    return out


def exact_steps(h: float, horizon: float) -> int:
    ratio = float(horizon) / float(h)
    steps = int(round(ratio))
    if not math.isclose(steps * float(h), float(horizon), rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"horizon is not an integer number of steps: T={horizon}, h={h}")
    return steps


def iter_configurations(
    spec: Mapping[str, Any],
    *,
    smoke: bool,
    systems: Sequence[str] | None = None,
) -> Iterable[dict[str, Any]]:
    selected = set(systems or spec["systems"].keys())
    for name, system in spec["systems"].items():
        if name not in selected:
            continue
        if smoke:
            pairs = [(float(system["smoke"]["h"]), float(system["smoke"]["horizon"]))]
        else:
            pairs = [
                (float(h), float(horizon))
                for h in system["step_sizes"]
                for horizon in system["horizons"]
            ]
        for h, horizon in pairs:
            yield {
                "system": name,
                "h": h,
                "horizon": horizon,
                "steps": exact_steps(h, horizon),
            }


def power(value: Any, exponent: int) -> Any:
    if exponent == 0:
        return 1.0
    out = value
    for _ in range(1, exponent):
        out = out * value
    return out


def evaluate_rhs(state: Sequence[Any], system_spec: Mapping[str, Any]) -> list[Any]:
    """Evaluate the machine-readable sparse polynomial RHS on arbitrary scalars/TMs."""
    outputs: list[Any] = []
    for polynomial in system_spec["rhs"]:
        acc: Any = 0.0
        for term in polynomial["terms"]:
            value: Any = float(term["coefficient"])
            for coordinate, exponent in zip(state, term["powers"]):
                if int(exponent):
                    value = value * power(coordinate, int(exponent))
            acc = acc + value
        outputs.append(acc)
    return outputs


def flowstar_expression(polynomial: Mapping[str, Any], state_names: Sequence[str]) -> str:
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


def exact_endpoint(system: str, t: float, initial_box: Sequence[Sequence[float]]) -> list[tuple[float, float]] | None:
    if system == "riccati":
        lo0, hi0 = map(float, initial_box[0])
        return [(lo0 / (1.0 - lo0 * t), hi0 / (1.0 - hi0 * t))]
    if system == "harmonic":
        c, s = math.cos(t), math.sin(t)
        (xlo, xhi), (ylo, yhi) = [tuple(map(float, bounds)) for bounds in initial_box]

        def affine_hull(a: float, bounds_a: tuple[float, float], b: float, bounds_b: tuple[float, float]) -> tuple[float, float]:
            vals = [
                a * bounds_a[i] + b * bounds_b[j]
                for i in (0, 1)
                for j in (0, 1)
            ]
            return min(vals), max(vals)

        return [
            affine_hull(c, (xlo, xhi), s, (ylo, yhi)),
            affine_hull(-s, (xlo, xhi), c, (ylo, yhi)),
        ]
    return None


def git_sha(path: str | Path) -> str:
    proc = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    return proc.stdout.strip() if proc.returncode == 0 else "unknown"


def make_run_id(tool: str, protocol: str, system: str, h: float, horizon: float) -> str:
    label = f"{tool}|{protocol}|{system}|{h:.17g}|{horizon:.17g}"
    digest = hashlib.sha256(label.encode()).hexdigest()[:10]
    return f"{tool}_{protocol}_{system}_{digest}"


def median_iqr(values: Sequence[float]) -> tuple[float, float]:
    vals = sorted(float(v) for v in values)
    if not vals:
        return math.nan, math.nan
    if len(vals) == 1:
        return vals[0], 0.0
    quartiles = statistics.quantiles(vals, n=4, method="inclusive")
    return statistics.median(vals), quartiles[2] - quartiles[0]


def finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def interval_row(
    *,
    run: Mapping[str, Any],
    state_index: int,
    step_index: int,
    time_value: float,
    interval_kind: str,
    lower: float | str,
    upper: float | str,
) -> dict[str, Any]:
    row = {field: run.get(field, "") for field in RAW_FIELDS}
    row.update(
        state_index=int(state_index),
        step_index=int(step_index),
        time=float(time_value),
        interval_kind=interval_kind,
        lower=lower,
        upper=upper,
        width=(float(upper) - float(lower)) if finite_number(lower) and finite_number(upper) else "",
    )
    return row


def write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str] | None = None) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(fields or RAW_FIELDS)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def output_dir_from_args(value: str | None) -> Path:
    if value:
        path = Path(value).resolve()
    else:
        path = HERE / "results" / utc_timestamp()
    path.mkdir(parents=True, exist_ok=True)
    return path


def raw_run_template(
    *,
    tool: str,
    protocol: str,
    system: str,
    h: float,
    horizon: float,
    requested_order_label: str,
    retained_basis: str,
    effective_max_degree: int | str,
    truncate_to_affine: bool | str,
    nonzero_lt: bool | str,
    dependency_mode: str,
    symbolic_remainder_size: int,
    cutoff: float | None,
    dtype: str,
    device: str,
    git_commit: str,
    environment: str,
) -> dict[str, Any]:
    return {
        "run_id": make_run_id(tool, protocol, system, h, horizon),
        "timestamp": utc_timestamp(),
        "tool": tool,
        "protocol": protocol,
        "system": system,
        "h": h,
        "horizon": horizon,
        "status": "produced_enclosure_unverified",
        "validation_status": "unverified",
        "first_failure_time": "",
        "requested_order_label": requested_order_label,
        "retained_basis": retained_basis,
        "effective_max_degree": effective_max_degree,
        "truncate_to_affine": truncate_to_affine,
        "nonzero_Lt": nonzero_lt,
        "dependency_mode": dependency_mode,
        "symbolic_remainder_size": symbolic_remainder_size,
        "cutoff": "" if cutoff is None else cutoff,
        "partitions": 1,
        "dtype": dtype,
        "device": device,
        "batch_size": 1,
        "build_time_s": 0.0,
        "warmup_time_s": "",
        "steady_runtime_median_s": "",
        "steady_runtime_iqr_s": "",
        "git_commit": git_commit,
        "environment": environment,
        "validation_attempts": "",
        "successful_horizon": 0.0,
        "message": "",
    }
