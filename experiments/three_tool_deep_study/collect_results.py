#!/usr/bin/env python3
"""Collect provenance, normalized result tables, and correctness gates."""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import platform
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from common import (
    analytic_endpoint,
    git_sha,
    load_spec,
    sha256_manifest,
    validate_record,
    write_csv,
    write_json,
)

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
CONDA = Path("/srv/local/shengenli/miniforge3/condabin/conda")


def _run(command: list[str], cwd: Path | None = None) -> dict[str, Any]:
    try:
        process = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
            timeout=180,
        )
        return {
            "command": command,
            "returncode": process.returncode,
            "stdout": process.stdout,
            "stderr": process.stderr,
        }
    except Exception as exc:
        return {
            "command": command,
            "returncode": -1,
            "stdout": "",
            "stderr": f"{type(exc).__name__}: {exc}",
        }


def _json_probe(environment: str, source: str) -> dict[str, Any]:
    result = _run(
        [str(CONDA), "run", "-n", environment, "python", "-c", source]
    )
    if result["returncode"] == 0:
        try:
            return json.loads(result["stdout"])
        except json.JSONDecodeError:
            pass
    return result


def _repo_audit(path: Path) -> dict[str, Any]:
    commands = {
        "status_short": ["git", "status", "--short"],
        "branches": ["git", "branch", "-avv"],
        "head": ["git", "rev-parse", "HEAD"],
        "log": [
            "git",
            "log",
            "--oneline",
            "--decorate",
            "--graph",
            "--all",
            "-n",
            "100",
        ],
        "remotes": ["git", "remote", "-v"],
        "worktrees": ["git", "worktree", "list"],
        "untracked": ["git", "ls-files", "--others", "--exclude-standard"],
    }
    return {
        "path": str(path),
        **{
            name: _run(command, cwd=path)
            for name, command in commands.items()
        },
    }


def _frozen_files(spec: Mapping[str, Any]) -> list[Path]:
    paths: list[Path] = []
    for relative in spec["frozen_inputs"]:
        root = REPO_ROOT / relative
        if not root.exists():
            continue
        paths.extend(path for path in root.rglob("*") if path.is_file())
    return paths


def collect_environment(
    spec: Mapping[str, Any], output: Path, *, initialize: bool
) -> dict[str, Any]:
    repositories = {
        name: Path(path)
        for name, path in spec["repositories"].items()
    }
    torch_probe = _json_probe(
        str(spec["environments"]["torch"]),
        (
            "import json,platform,torch;"
            "print(json.dumps({'python':platform.python_version(),"
            "'torch':torch.__version__,'cuda_version':torch.version.cuda,"
            "'cuda_available':torch.cuda.is_available(),"
            "'cuda_devices':torch.cuda.device_count(),"
            "'default_dtype':str(torch.get_default_dtype())}))"
        ),
    )
    jax_probe = _json_probe(
        str(spec["environments"]["diffreach"]),
        (
            "import json,platform,jax,jaxlib;"
            "jax.config.update('jax_enable_x64',True);"
            "print(json.dumps({'python':platform.python_version(),"
            "'jax':jax.__version__,'jaxlib':jaxlib.__version__,"
            "'x64':bool(jax.config.jax_enable_x64),"
            "'devices':[str(x) for x in jax.devices()]}))"
        ),
    )
    cpu_model = ""
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        cpu_model = next(
            (
                line.split(":", 1)[1].strip()
                for line in cpuinfo.read_text(
                    encoding="utf-8", errors="replace"
                ).splitlines()
                if line.startswith("model name")
            ),
            "",
        )
    environment = {
        "date_utc": _run(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"])[
            "stdout"
        ].strip(),
        "host_python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_model": cpu_model,
        "cpu_count": os.cpu_count(),
        "torch_environment": spec["environments"]["torch"],
        "diffreach_environment": spec["environments"]["diffreach"],
        "flowstar_environment": spec["environments"]["flowstar"],
        "torch_probe": torch_probe,
        "jax_probe": jax_probe,
        "gcc": _run(["gcc", "--version"]),
        "gxx": _run(["g++", "--version"]),
        "mpfr": _run(["pkg-config", "--modversion", "mpfr"]),
        "gmp": _run(["pkg-config", "--modversion", "gmp"]),
        "tmux": _run(["tmux", "-V"]),
        "repositories": {
            name: {
                "path": str(path),
                "sha": git_sha(path),
            }
            for name, path in repositories.items()
        },
    }
    write_json(output / "environment.json", environment)
    audit = {
        name: _repo_audit(path)
        for name, path in repositories.items()
    }
    write_json(output / "repository_audit.json", audit)
    manifest = sha256_manifest(_frozen_files(spec), REPO_ROOT)
    if initialize:
        write_json(
            output / "frozen_input_manifest_initial.json", manifest
        )
    else:
        initial_path = output / "frozen_input_manifest_initial.json"
        initial = (
            json.loads(initial_path.read_text(encoding="utf-8"))
            if initial_path.exists()
            else []
        )
        integrity = {
            "initial_file_count": len(initial),
            "final_file_count": len(manifest),
            "unchanged": initial == manifest,
            "added_or_changed": [
                row for row in manifest if row not in initial
            ],
            "removed_or_changed": [
                row for row in initial if row not in manifest
            ],
        }
        write_json(output / "frozen_input_manifest_final.json", manifest)
        write_json(output / "frozen_input_integrity.json", integrity)
        if initial and not integrity["unchanged"]:
            raise RuntimeError("a frozen input artifact changed")
    return environment


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _truth(value: Any) -> bool | None:
    text = str(value).strip().lower()
    if text in {"true", "passed", "1"}:
        return True
    if text in {"false", "failed", "0"}:
        return False
    return None


def _number(value: Any, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _tool_family(tool: Any) -> str:
    text = str(tool)
    if text.startswith("diffreach"):
        return "diffreach"
    if text.startswith("torch"):
        return "torch_tm_flowpipe"
    return text


def _annotate_required_metrics(
    spec: Mapping[str, Any], rows: list[dict[str, Any]]
) -> None:
    """Add common metric semantics without fabricating missing quantities."""
    affine_widths: dict[tuple[str, ...], float] = {}
    for row in rows:
        if (
            row.get("protocol") == "common_affine_carry"
            and row.get("interval_kind") == "endpoint_raw"
        ):
            key = (
                _tool_family(row.get("tool")),
                str(row.get("system", "")),
                str(row.get("h", "")),
                str(row.get("time", "")),
                str(row.get("state_index", "")),
            )
            affine_widths[key] = _number(row.get("width"), math.nan)

    for row in rows:
        system_name = str(row.get("system", ""))
        state_index = int(_number(row.get("state_index"), 0.0))
        lower = _number(row.get("lower"), math.nan)
        upper = _number(row.get("upper"), math.nan)
        if math.isfinite(lower) and math.isfinite(upper):
            center = 0.5 * (lower + upper)
            row["interval_center"] = center
            initial = spec["systems"].get(system_name, {}).get(
                "initial_box", []
            )
            if state_index < len(initial):
                initial_center = 0.5 * (
                    float(initial[state_index][0])
                    + float(initial[state_index][1])
                )
                row["center_shift_from_initial"] = (
                    center - initial_center
                )
            else:
                row["center_shift_from_initial"] = "unavailable"
                row["center_shift_unavailable_reason"] = (
                    "state index is outside the system definition"
                )
        else:
            row["interval_center"] = "unavailable"
            row["center_shift_from_initial"] = "unavailable"
            row["center_shift_unavailable_reason"] = (
                "row has no finite interval"
            )

        h = _number(row.get("h"), math.nan)
        row["requested_step"] = h if math.isfinite(h) else "unavailable"
        validation = _truth(row.get("native_validation_passed"))
        is_failure = (
            str(row.get("interval_kind", "")) == "failure"
            or bool(row.get("failure_category"))
        )
        row["accepted_step"] = (
            h
            if math.isfinite(h) and validation is not False and not is_failure
            else "unavailable"
        )
        if row["accepted_step"] == "unavailable":
            row["accepted_step_unavailable_reason"] = (
                "native validation rejected the requested step or the "
                "backend did not expose an accepted step"
            )
        step_index = _number(row.get("step_index"), math.nan)
        row["accepted_steps"] = row.get(
            "completed_steps",
            (
                max(0, int(step_index) - 1)
                if is_failure and math.isfinite(step_index)
                else row.get("step_index", "unavailable")
            ),
        )
        if row.get("successful_horizon") not in ("", None):
            row["successful_horizon"] = row["successful_horizon"]
        elif is_failure and math.isfinite(step_index) and math.isfinite(h):
            row["successful_horizon"] = max(0.0, (step_index - 1) * h)
        else:
            row["successful_horizon"] = row.get("time", "unavailable")

        if system_name in {"riccati", "harmonic"}:
            row["reference_oracle"] = "analytic_interval_solution"
        else:
            row["reference_oracle"] = (
                "deterministic_DOP853_trajectory_sanity_non_proof"
            )

        if (
            row.get("protocol") == "common_box_carry"
            and row.get("interval_kind") == "endpoint_raw"
        ):
            key = (
                _tool_family(row.get("tool")),
                system_name,
                str(row.get("h", "")),
                str(row.get("time", "")),
                str(row.get("state_index", "")),
            )
            affine = affine_widths.get(key, math.nan)
            width = _number(row.get("width"), math.nan)
            if math.isfinite(affine) and affine > 0 and math.isfinite(width):
                row["dependency_loss_metric"] = width / affine
                row["dependency_loss_semantics"] = (
                    "box_width_over_same_time_affine_carry_width"
                )
            else:
                row["dependency_loss_metric"] = "unavailable"
                row["dependency_loss_semantics"] = (
                    "no same-time affine reference row"
                )
        elif row.get("protocol") == "common_affine_carry":
            row["dependency_loss_metric"] = "reference"
            row["dependency_loss_semantics"] = (
                "affine carry reference for box-reset dependency loss"
            )
        else:
            row["dependency_loss_metric"] = "unavailable"
            row["dependency_loss_semantics"] = (
                "dependency loss requires a matched carry/reset control"
            )

        setup = next(
            (
                row.get(key)
                for key in (
                    "setup_time_s",
                    "build_time_s",
                    "compile_time_s",
                    "jit_time_s",
                )
                if row.get(key) not in ("", None)
            ),
            "unavailable",
        )
        propagation = next(
            (
                row.get(key)
                for key in (
                    "propagation_time_s",
                    "runtime_s",
                    "execution_time_s",
                    "step_runtime_s",
                )
                if row.get(key) not in ("", None)
            ),
            "unavailable",
        )
        export = next(
            (
                row.get(key)
                for key in ("export_time_s", "serialization_time_s")
                if row.get(key) not in ("", None)
            ),
            "unavailable",
        )
        row["runtime_setup_s"] = setup
        row["runtime_propagation_s"] = propagation
        row["runtime_export_s"] = export
        row["runtime_unavailable_semantics"] = (
            "unavailable means the backend did not isolate this component; "
            "it is not zero"
        )
        if not any(
            row.get(key) not in ("", None)
            for key in (
                "memory_kib",
                "peak_process_rss_kib",
                "peak_device_memory_bytes",
            )
        ):
            row["memory_measurement"] = "unavailable"
            row["memory_unavailable_reason"] = (
                "this protocol did not isolate process/device peak memory"
            )
        else:
            row["memory_measurement"] = next(
                row.get(key)
                for key in (
                    "memory_kib",
                    "peak_process_rss_kib",
                    "peak_device_memory_bytes",
                )
                if row.get(key) not in ("", None)
            )


def _last_rows(
    rows: Iterable[Mapping[str, Any]], protocol: str
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("protocol") != protocol:
            continue
        key = tuple(
            str(row.get(field, ""))
            for field in (
                "tool",
                "variant",
                "system",
                "h",
                "state_index",
                "interval_kind",
            )
        )
        grouped[key].append(row)
    result: list[dict[str, Any]] = []
    for values in grouped.values():
        maximum = max(_number(row.get("step_index")) for row in values)
        result.extend(
            dict(row)
            for row in values
            if _number(row.get("step_index")) == maximum
        )
    return result


def _one_step_summary(
    spec: Mapping[str, Any], rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        if row.get("protocol") != "one_step_common_input":
            continue
        item = dict(row)
        exact = analytic_endpoint(
            str(row.get("system")),
            spec["systems"][str(row.get("system"))]["initial_box"],
            _number(row.get("time")),
        )
        state = int(_number(row.get("state_index")))
        if exact is not None:
            exact_width = exact[state][1] - exact[state][0]
            item["analytic_exact_width"] = exact_width
            item["exact_inflation_ratio"] = (
                _number(row.get("width")) / exact_width
                if exact_width
                else ""
            )
        else:
            item["analytic_exact_width"] = ""
            item["exact_inflation_ratio"] = ""
        result.append(item)
    return result


def _rhs_numpy(
    system: Mapping[str, Any], state: Any
) -> Any:
    import numpy as np

    values = []
    for component in system["rhs"]:
        total = 0.0
        for term in component["terms"]:
            value = float(term["coefficient"])
            for coordinate, power in zip(state, term["powers"]):
                value *= float(coordinate) ** int(power)
            total += value
        values.append(total)
    return np.asarray(values, dtype=np.float64)


def _trajectory_sample_points(
    initial_box: Iterable[Iterable[float]],
) -> list[tuple[float, ...]]:
    axes = []
    for lower, upper in initial_box:
        lo, hi = float(lower), float(upper)
        axes.append((lo, 0.5 * (lo + hi), hi))
    return list(itertools.product(*axes))


def _annotate_trajectory_sanity(
    spec: Mapping[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, int]:
    """Apply non-proof deterministic trajectory checks to nonlinear rows."""
    import numpy as np
    from scipy.integrate import solve_ivp

    nonlinear = {"coupled_quadratic", "van_der_pol"}
    tolerance = float(spec["trajectory_tolerance"])
    horizons: dict[str, float] = defaultdict(float)
    for row in rows:
        system = str(row.get("system", ""))
        time_value = _number(row.get("time"), math.nan)
        if system in nonlinear and math.isfinite(time_value):
            horizons[system] = max(horizons[system], time_value)

    solutions: dict[str, list[Any]] = {}
    for system_name, horizon in horizons.items():
        system = spec["systems"][system_name]
        system_solutions = []
        for point in _trajectory_sample_points(system["initial_box"]):
            solution = solve_ivp(
                lambda _time, state, _system=system: _rhs_numpy(
                    _system, state
                ),
                (0.0, max(horizon, 1e-15)),
                np.asarray(point, dtype=np.float64),
                method="DOP853",
                rtol=1e-12,
                atol=1e-14,
                dense_output=True,
            )
            if not solution.success or solution.sol is None:
                raise RuntimeError(
                    f"trajectory sanity integration failed for {system_name}: "
                    f"{solution.message}"
                )
            system_solutions.append(solution.sol)
        solutions[system_name] = system_solutions

    checked = passed = failed = not_applicable = 0
    for row in rows:
        system_name = str(row.get("system", ""))
        kind = str(row.get("interval_kind", ""))
        if system_name not in nonlinear:
            if kind in {
                "tube",
                "endpoint_raw",
                "endpoint_tightened_supplemental",
            }:
                row["trajectory_sanity_passed"] = (
                    "not_applicable_analytic_reference"
                )
                not_applicable += 1
            continue
        if kind not in {
            "tube",
            "endpoint_raw",
            "endpoint_tightened_supplemental",
        }:
            continue
        lower = _number(row.get("lower"), math.nan)
        upper = _number(row.get("upper"), math.nan)
        time_value = _number(row.get("time"), math.nan)
        state_index = int(_number(row.get("state_index")))
        if not all(math.isfinite(value) for value in (lower, upper, time_value)):
            row["trajectory_sanity_passed"] = "not_checked_nonfinite"
            not_applicable += 1
            continue
        if kind == "tube":
            h = _number(row.get("h"), math.nan)
            if not math.isfinite(h):
                row["trajectory_sanity_passed"] = (
                    "not_checked_adaptive_local_step"
                )
                not_applicable += 1
                continue
            times = np.linspace(max(0.0, time_value - h), time_value, 5)
        else:
            times = np.asarray([time_value])
        values = np.concatenate(
            [
                np.asarray(solution(times))[state_index].reshape(-1)
                for solution in solutions[system_name]
            ]
        )
        contained = bool(
            np.min(values) >= lower - tolerance
            and np.max(values) <= upper + tolerance
        )
        row["trajectory_sanity_passed"] = contained
        checked += 1
        if contained:
            passed += 1
        else:
            failed += 1
    return {
        "checked": checked,
        "passed": passed,
        "failed": failed,
        "not_applicable": not_applicable,
        "proof": False,
    }


def collect_tables(
    spec: Mapping[str, Any], output: Path
) -> dict[str, Any]:
    controlled = _read_csv(output / "controlled_raw.csv")
    native = _read_csv(output / "native_raw.csv")
    _annotate_required_metrics(spec, controlled + native)
    trajectory_checks = _annotate_trajectory_sanity(
        spec, controlled + native
    )
    repetitions: list[dict[str, Any]] = []
    for tool in ("torch", "diffreach", "flowstar"):
        repetitions.extend(
            _read_csv(output / f"pareto_repetitions_{tool}.csv")
        )
    correctness_rows = _read_csv(output / "flowstar_correctness.csv")
    flowstar_ablation = _read_csv(
        output / "flowstar_component_ablation.csv"
    )
    controlled_summaries: list[dict[str, Any]] = []
    native_summaries: list[dict[str, Any]] = []
    for tool in ("torch", "diffreach", "flowstar"):
        controlled_path = output / f"controlled_{tool}_summary.json"
        native_path = output / f"native_{tool}_summary.json"
        if controlled_path.exists():
            controlled_summaries.extend(
                json.loads(controlled_path.read_text(encoding="utf-8"))
            )
        if native_path.exists():
            native_summaries.extend(
                json.loads(native_path.read_text(encoding="utf-8"))
            )
    acceleration: list[dict[str, Any]] = []
    for tool in ("torch", "diffreach", "flowstar"):
        acceleration.extend(
            _read_csv(output / f"acceleration_{tool}.csv")
        )
    raw: list[dict[str, Any]] = []
    for source, values in (
        ("controlled", controlled),
        ("native", native),
        ("controlled_summary", controlled_summaries),
        ("native_summary", native_summaries),
        ("runtime_repetition", repetitions),
        ("acceleration", acceleration),
        ("flowstar_correctness", correctness_rows),
        ("flowstar_ablation", flowstar_ablation),
    ):
        raw.extend({"result_source": source, **row} for row in values)
    write_csv(output / "raw_results.csv", raw)

    one_step = _one_step_summary(spec, controlled)
    affine = _last_rows(controlled, "common_affine_carry")
    box = _last_rows(controlled, "common_box_carry")
    native_low = _last_rows(native, "native_low_order")
    write_csv(output / "one_step_summary.csv", one_step)
    write_csv(output / "affine_carry_summary.csv", affine)
    write_csv(output / "box_carry_summary.csv", box)
    write_csv(output / "native_low_order_summary.csv", native_low)

    failures: list[dict[str, Any]] = []
    for row in raw:
        validation = _truth(
            row.get(
                "native_validation_passed",
                row.get("native_validation_status", ""),
            )
        )
        category = row.get(
            "failure_category", row.get("failure_message", "")
        )
        analytic = _truth(row.get("analytic_reference_contained"))
        trajectory = _truth(row.get("trajectory_sanity_passed"))
        if (
            validation is False
            or analytic is False
            or trajectory is False
            or category
        ):
            if not category:
                if validation is False:
                    category = "native_validation_failure"
                elif analytic is False:
                    category = "analytic_reference_violation"
                else:
                    category = "trajectory_sanity_failure"
            failures.append(
                {
                    "result_source": row.get("result_source", ""),
                    "tool": row.get("tool", ""),
                    "variant": row.get("variant", ""),
                    "protocol": row.get("protocol", ""),
                    "system": row.get("system", ""),
                    "h": row.get("h", ""),
                    "step_index": row.get("step_index", ""),
                    "failure_category": category,
                    "message": row.get(
                        "message", row.get("failure_message", "")
                    ),
                }
            )
    write_csv(output / "failure_summary.csv", failures)

    representation_records: dict[tuple[Any, ...], tuple[Path, dict[str, Any]]] = {}
    for path in sorted((output / "common_segments").glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        key = (
            record["tool"],
            record["variant"],
            record["system"],
            float(record["h"]),
            record.get("native_metadata", {}).get("requested_order"),
        )
        representation_records[key] = (path, record)
    representation_checks = []
    for path, record in representation_records.values():
        representation_checks.append(
            {
                "path": str(path),
                "tool": record["tool"],
                "system": record["system"],
                **validate_record(record),
            }
        )
    analytic_rows = [
        row
        for row in controlled + native
        if str(row.get("analytic_reference_contained", "")).strip()
    ]
    native_rows = [
        row
        for row in controlled + native
        if str(row.get("native_validation_passed", "")).strip()
    ]
    controlled_trajectory_failures = sum(
        _truth(row.get("trajectory_sanity_passed")) is False
        for row in controlled
    )
    native_trajectory_failures = sum(
        _truth(row.get("trajectory_sanity_passed")) is False
        for row in native
    )

    failed_native_configurations = {
        tuple(
            str(row.get(field, ""))
            for field in ("tool", "variant", "system", "h")
        )
        for row in native
        if _truth(row.get("trajectory_sanity_passed")) is False
    }
    pareto_path = output / "native_pareto_summary.csv"
    if pareto_path.exists():
        pareto_rows = _read_csv(pareto_path)
        for row in pareto_rows:
            key = tuple(
                str(row.get(field, ""))
                for field in ("tool", "variant", "system", "h")
            )
            trajectory_passed = key not in failed_native_configurations
            native_passed = (
                _truth(row.get("native_validation_passed")) is not False
            )
            row["trajectory_sanity_passed"] = trajectory_passed
            row["primary_numerical_eligible"] = (
                trajectory_passed and native_passed
            )
            if not row["primary_numerical_eligible"]:
                row["width_runtime_pareto"] = False
        write_csv(pareto_path, pareto_rows)
    flowstar_summary_path = output / "flowstar_correctness_summary.json"
    flowstar_summary = (
        json.loads(flowstar_summary_path.read_text(encoding="utf-8"))
        if flowstar_summary_path.exists()
        else {}
    )
    frozen_path = output / "frozen_input_integrity.json"
    frozen = (
        json.loads(frozen_path.read_text(encoding="utf-8"))
        if frozen_path.exists()
        else {}
    )
    correctness = {
        "controlled_and_native_rows_checked": len(controlled) + len(native),
        "native_validation_checks": len(native_rows),
        "native_validation_failures": sum(
            _truth(row.get("native_validation_passed")) is False
            for row in native_rows
        ),
        "analytic_checks": len(analytic_rows),
        "analytic_violations": sum(
            _truth(row.get("analytic_reference_contained")) is False
            for row in analytic_rows
        ),
        "trajectory_sanity": {
            **trajectory_checks,
            "controlled_failures": controlled_trajectory_failures,
            "native_candidate_failures": native_trajectory_failures,
        },
        "common_segment_records": len(representation_checks),
        "common_segment_point_checks": sum(
            int(row["point_evaluation_checks"])
            for row in representation_checks
        ),
        "common_segment_native_round_trip_checks": sum(
            int(row["native_point_evaluation_checks"])
            for row in representation_checks
        ),
        "common_segment_native_round_trip_violations": sum(
            int(row["native_point_evaluation_violations"])
            for row in representation_checks
        ),
        "common_segment_representation_failures": sum(
            not row["passed"] for row in representation_checks
        ),
        "common_segment_endpoint_tube_violations": sum(
            int(row["endpoint_vs_tube_violations"])
            for row in representation_checks
        ),
        "flowstar": flowstar_summary,
        "frozen_inputs": frozen,
        "failure_rows": len(failures),
    }
    correctness["primary_gates_passed"] = bool(
        correctness["native_validation_failures"] == 0
        and correctness["analytic_violations"] == 0
        and controlled_trajectory_failures == 0
        and correctness["common_segment_representation_failures"] == 0
        and correctness[
            "common_segment_native_round_trip_violations"
        ]
        == 0
        and correctness["common_segment_endpoint_tube_violations"] == 0
        and flowstar_summary.get("passed", False)
        and frozen.get("unchanged", False)
    )
    write_json(output / "correctness_checks.json", correctness)
    return {
        "raw_rows": len(raw),
        "one_step_rows": len(one_step),
        "affine_rows": len(affine),
        "box_rows": len(box),
        "native_low_order_rows": len(native_low),
        "failure_rows": len(failures),
        "primary_gates_passed": correctness["primary_gates_passed"],
    }


def verify_completed_output(
    output: Path, *, require_ten_repetitions: bool
) -> dict[str, Any]:
    checks_path = output / "correctness_checks.json"
    if not checks_path.exists():
        raise SystemExit("missing correctness_checks.json")
    correctness = json.loads(checks_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    if not correctness.get("primary_gates_passed", False):
        failures.append("primary correctness gates did not pass")

    plot_names = (
        "one_step_tube_width_vs_h",
        "one_step_endpoint_raw_width_vs_h",
        "exact_inflation_ratios",
        "common_affine_carry_width_vs_time",
        "common_box_carry_width_vs_time",
        "affine_vs_box_carry",
        "native_low_order_width_curves",
        "native_practical_width_runtime_pareto",
        "successful_horizon_vs_runtime",
        "polynomial_remainder_decomposition",
        "monomial_family_support",
        "torch_reset_order_ablation",
        "diffreach_affine_quasi_symbolic_ablation",
        "flowstar_order_step_qr_symbolic_refinement_ablation",
        "matched_basis_results",
        "common_defect_vs_native_remainder",
        "runtime_decomposition",
        "failure_categories",
    )
    mandatory_plots = [
        f"{index:02d}_{name}.png"
        for index, name in enumerate(plot_names, start=1)
    ]
    missing_plots = [
        name
        for name in mandatory_plots
        if not (output / "plots" / name).is_file()
    ]
    if missing_plots:
        failures.append(f"missing mandatory plots: {missing_plots}")
    for name in (
        "three_tool_deep_study_report.md",
        "executive_summary.md",
        "raw_results.csv",
        "correctness_checks.json",
        "failure_summary.csv",
    ):
        if not (output / name).is_file():
            failures.append(f"missing final artifact: {name}")

    pareto = (
        json.loads(
            (output / "pareto_checks.json").read_text(encoding="utf-8")
        )
        if (output / "pareto_checks.json").exists()
        else {}
    )
    if require_ten_repetitions and not pareto.get(
        "all_selected_have_ten_repetitions", False
    ):
        failures.append(
            "selected practical configurations lack ten repetitions"
        )

    acceleration = _read_csv(output / "acceleration_summary.csv")
    if require_ten_repetitions:
        status = {
            str(row.get("backend")): str(
                row.get("backend_status")
            ).lower()
            for row in acceleration
        }
        for backend in ("torch_cpu", "jax_cpu", "flowstar_cpu"):
            if status.get(backend) != "available":
                failures.append(
                    f"missing available acceleration row for {backend}"
                )
        environment = json.loads(
            (output / "environment.json").read_text(encoding="utf-8")
        )
        torch_cuda = bool(
            environment.get("torch_probe", {}).get(
                "cuda_available", False
            )
        )
        if torch_cuda and status.get("torch_cuda") != "available":
            failures.append(
                "Torch CUDA is visible but no available CUDA row exists"
            )
        jax_devices = [
            str(value).lower()
            for value in environment.get("jax_probe", {}).get(
                "devices", []
            )
        ]
        jax_gpu = any(
            "gpu" in value or "cuda" in value for value in jax_devices
        )
        expected_jax_status = "available" if jax_gpu else "unavailable"
        if status.get("jax_cuda") != expected_jax_status:
            failures.append(
                "JAX CUDA capability row does not match enumerated devices"
            )

    result = {
        "passed": not failures,
        "failures": failures,
        "require_ten_repetitions": require_ten_repetitions,
        "primary_gates_passed": correctness.get(
            "primary_gates_passed", False
        ),
        "mandatory_plot_count": len(mandatory_plots) - len(missing_plots),
        "pareto_checks": pareto,
    }
    write_json(output / "final_acceptance.json", result)
    if failures:
        raise SystemExit("; ".join(failures))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", default=str(HERE / "benchmark_spec.yaml"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--mode",
        choices=["initialize", "finalize", "tables", "verify", "all"],
        required=True,
    )
    parser.add_argument(
        "--require-ten-repetitions", action="store_true"
    )
    args = parser.parse_args()
    spec = load_spec(args.spec)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {}
    if args.mode in {"initialize", "all"}:
        result["environment_initial"] = collect_environment(
            spec, output, initialize=True
        )
    if args.mode in {"finalize", "all"}:
        result["environment_final"] = collect_environment(
            spec, output, initialize=False
        )
    if args.mode in {"tables", "all"}:
        result["tables"] = collect_tables(spec, output)
    if args.mode == "verify":
        result["verify"] = verify_completed_output(
            output,
            require_ten_repetitions=args.require_ten_repetitions,
        )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
