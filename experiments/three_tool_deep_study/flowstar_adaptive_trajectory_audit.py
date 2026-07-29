#!/usr/bin/env python3
"""Audit adaptive Flow* endpoints against native fixed-time evaluation."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from scipy.integrate import solve_ivp

from common import git_sha, load_spec, write_csv, write_json

HERE = Path(__file__).resolve().parent


def _parse_tokens(line: str, prefix: str) -> dict[str, str] | None:
    if not line.startswith(prefix):
        return None
    return {
        key: value
        for token in line[len(prefix) :].split()
        if "=" in token
        for key, value in [token.split("=", 1)]
    }


def _parse_endpoint_paths(stdout: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        fields = _parse_tokens(line, "PARITY_ENDPOINT_PATH ")
        if fields is None:
            continue
        rows.append(
            {
                key: (
                    fields[key]
                    if key == "label"
                    else (
                        int(fields[key])
                        if key in {"step", "state"}
                        else float(fields[key])
                    )
                )
                for key in fields
            }
        )
    return rows


def _parse_parity_rows(stdout: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        fields = _parse_tokens(line, "PARITY_ROW ")
        if fields is None:
            continue
        rows.append(
            {
                key: (
                    fields[key]
                    if key == "label"
                    else (
                        int(fields[key])
                        if key in {"step", "state"}
                        else float(fields[key])
                    )
                )
                for key in fields
            }
        )
    return rows


def _rhs(system: Mapping[str, Any], state: np.ndarray) -> np.ndarray:
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


def _reference_solutions(
    system: Mapping[str, Any], horizon: float
) -> list[tuple[tuple[float, ...], Any]]:
    axes = [
        (float(lower), 0.5 * (float(lower) + float(upper)), float(upper))
        for lower, upper in system["initial_box"]
    ]
    result = []
    for point in itertools.product(*axes):
        solution = solve_ivp(
            lambda _time, state: _rhs(system, state),
            (0.0, horizon),
            np.asarray(point, dtype=np.float64),
            method="DOP853",
            rtol=1e-12,
            atol=1e-14,
            dense_output=True,
        )
        if not solution.success or solution.sol is None:
            raise RuntimeError(
                f"DOP853 reference integration failed: {solution.message}"
            )
        result.append((tuple(map(float, point)), solution.sol))
    return result


def _trajectory_failures(
    rows: Iterable[Mapping[str, Any]],
    references: Iterable[tuple[tuple[float, ...], Any]],
    *,
    lower_field: str,
    upper_field: str,
    tolerance: float,
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    reference_list = list(references)
    for row in rows:
        time_value = float(row["time"])
        state = int(row["state"])
        values = [
            (
                point,
                float(solution(time_value)[state]),
            )
            for point, solution in reference_list
        ]
        minimum_point, minimum = min(values, key=lambda item: item[1])
        maximum_point, maximum = max(values, key=lambda item: item[1])
        lower, upper = float(row[lower_field]), float(row[upper_field])
        if minimum < lower - tolerance or maximum > upper + tolerance:
            failures.append(
                {
                    "segment_index": int(row["step"]),
                    "absolute_time": time_value,
                    "state_index": state,
                    "flowstar_lower": lower,
                    "flowstar_upper": upper,
                    "reference_lower": minimum,
                    "reference_upper": maximum,
                    "reference_lower_initial_point": list(minimum_point),
                    "reference_upper_initial_point": list(maximum_point),
                    "lower_under_enclosure_gap": max(0.0, lower - minimum),
                    "upper_under_enclosure_gap": max(0.0, maximum - upper),
                    "tolerance": tolerance,
                }
            )
    return failures


def _compile(
    source: Path,
    executable: Path,
    flowstar_root: Path,
    timeout: float,
) -> dict[str, Any]:
    command = [
        "g++",
        "-O3",
        "-w",
        "-fpermissive",
        "-std=c++11",
        "-I",
        str(flowstar_root / "flowstar-toolbox"),
        str(source),
        "-L",
        str(flowstar_root / "flowstar-toolbox"),
        "-o",
        str(executable),
        "-lflowstar",
        "-lmpfr",
        "-lgmp",
        "-lgsl",
        "-lgslcblas",
        "-lm",
        "-lglpk",
    ]
    started = time.perf_counter()
    process = subprocess.run(
        command, capture_output=True, text=True, check=False, timeout=timeout
    )
    return {
        "command": command,
        "returncode": process.returncode,
        "stdout": process.stdout,
        "stderr": process.stderr,
        "seconds": time.perf_counter() - started,
    }


def _execute(
    executable: Path,
    *,
    cwd: Path,
    timeout: float,
    leaf_patch: bool,
    full_picard: bool,
) -> dict[str, Any]:
    environment = os.environ.copy()
    environment["FLOWSTAR_AUDIT_CACHE_LEAF_TRUNCATION"] = (
        "1" if leaf_patch else "0"
    )
    environment["FLOWSTAR_AUDIT_REVALIDATE_REFINEMENT"] = (
        "1" if full_picard else "0"
    )
    started = time.perf_counter()
    process = subprocess.run(
        [str(executable)],
        cwd=cwd,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )
    return {
        "returncode": process.returncode,
        "stdout": process.stdout,
        "stderr": process.stderr,
        "seconds": time.perf_counter() - started,
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_adaptive_audit(
    spec: Mapping[str, Any],
    output: Path,
    *,
    parity_summary: Mapping[str, Any],
    parity_output: Path,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    timeout = float(spec["timeout_s"])
    audit_root = Path(spec["repositories"]["flowstar_audit"])
    stock_root = Path(spec["repositories"]["flowstar_original"])
    generated_dir = (
        parity_output
        / "logs"
        / "flowstar_original_parity"
        / "generated_identical"
    )
    generated_executable = generated_dir / "parity"
    generated_source = generated_dir / "parity.cpp"
    original_stdout = (
        parity_output
        / "logs"
        / "flowstar_original_parity"
        / "original"
        / "stdout.txt"
    )
    if not generated_executable.exists() or not generated_source.exists():
        raise RuntimeError("original parity gate did not leave its generated harness")

    variants: dict[str, dict[str, Any]] = {}
    for name, leaf_patch, full_picard in (
        ("generated_identical_stock", False, False),
        ("variable_leaf_patch", True, False),
        ("full_picard_revalidation", False, True),
        ("variable_leaf_plus_full_picard", True, True),
    ):
        result = _execute(
            generated_executable,
            cwd=generated_dir,
            timeout=timeout,
            leaf_patch=leaf_patch,
            full_picard=full_picard,
        )
        (output / f"{name}.stdout.txt").write_text(
            result.pop("stdout"), encoding="utf-8"
        )
        (output / f"{name}.stderr.txt").write_text(
            result.pop("stderr"), encoding="utf-8"
        )
        variants[name] = {
            **result,
            "stdout_path": str(output / f"{name}.stdout.txt"),
            "leaf_patch": leaf_patch,
            "full_picard_revalidation": full_picard,
            "flowstar_root": str(audit_root),
            "flowstar_sha": git_sha(audit_root),
        }

    stock_executable = output / "stock_upstream_parity"
    compile_result = _compile(
        generated_source, stock_executable, stock_root, timeout
    )
    (output / "stock_upstream.compile.stdout.txt").write_text(
        compile_result.pop("stdout"), encoding="utf-8"
    )
    (output / "stock_upstream.compile.stderr.txt").write_text(
        compile_result.pop("stderr"), encoding="utf-8"
    )
    if compile_result["returncode"]:
        raise RuntimeError("stock upstream Flow* audit harness did not compile")
    stock_result = _execute(
        stock_executable,
        cwd=output,
        timeout=timeout,
        leaf_patch=False,
        full_picard=False,
    )
    (output / "stock_upstream.stdout.txt").write_text(
        stock_result.pop("stdout"), encoding="utf-8"
    )
    (output / "stock_upstream.stderr.txt").write_text(
        stock_result.pop("stderr"), encoding="utf-8"
    )
    variants["stock_upstream"] = {
        **stock_result,
        "stdout_path": str(output / "stock_upstream.stdout.txt"),
        "leaf_patch": False,
        "full_picard_revalidation": False,
        "flowstar_root": str(stock_root),
        "flowstar_sha": git_sha(stock_root),
        "compile": compile_result,
    }

    system = spec["systems"]["van_der_pol"]
    references = _reference_solutions(system, 10.0)
    tolerance = float(spec["trajectory_tolerance"])
    variant_summaries: dict[str, Any] = {}
    authoritative_rows: list[dict[str, Any]] = []
    for name, metadata in variants.items():
        stdout = Path(metadata["stdout_path"]).read_text(encoding="utf-8")
        paths = _parse_endpoint_paths(stdout)
        parity_rows = _parse_parity_rows(stdout)
        if not paths or not parity_rows:
            raise RuntimeError(f"{name} emitted no adaptive endpoint audit rows")
        checks = {}
        for path_name, lower, upper in (
            ("collapsed_evaluate_time", "export_lower", "export_upper"),
            ("native_fixed_domain", "direct_lower", "direct_upper"),
            ("native_end_first", "native_lower", "native_upper"),
            ("repaired_raw_endpoint", "repaired_lower", "repaired_upper"),
        ):
            failures = _trajectory_failures(
                paths,
                references,
                lower_field=lower,
                upper_field=upper,
                tolerance=tolerance,
            )
            checks[path_name] = {
                "trajectory_failures": len(failures),
                "first_failure": failures[0] if failures else None,
                "failures": failures,
            }
        variant_summaries[name] = {
            **metadata,
            "segments": max(int(row["step"]) for row in paths),
            "endpoint_path_rows": len(paths),
            "checks": checks,
            "first_endpoint_path": paths[0],
            "first_path_divergence": next(
                (
                    row
                    for row in paths
                    if row["export_lower"] != row["direct_lower"]
                    or row["export_upper"] != row["direct_upper"]
                ),
                None,
            ),
        }
        if name == "variable_leaf_patch":
            authoritative_rows = parity_rows

    first_failure = variant_summaries["variable_leaf_patch"]["checks"][
        "collapsed_evaluate_time"
    ]["first_failure"]
    if first_failure is None:
        raise RuntimeError("the known pre-repair adaptive failure was not reproduced")
    step = int(first_failure["segment_index"])
    leaf_paths = _parse_endpoint_paths(
        Path(variants["variable_leaf_patch"]["stdout_path"]).read_text(
            encoding="utf-8"
        )
    )
    state_path = next(
        row
        for row in leaf_paths
        if int(row["step"]) == step
        and int(row["state"]) == int(first_failure["state_index"])
    )
    previous_time = max(
        (
            float(row["time"])
            for row in leaf_paths
            if int(row["step"]) == step - 1
        ),
        default=0.0,
    )
    accepted_step = float(first_failure["absolute_time"]) - previous_time
    first_failure.update(
        {
            "segment_start": previous_time,
            "requested_step": accepted_step,
            "accepted_step": accepted_step,
            "local_tau_domain": [0.0, accepted_step],
            "initial_box": system["initial_box"],
            "native_fixed_domain_lower": state_path["direct_lower"],
            "native_fixed_domain_upper": state_path["direct_upper"],
            "native_end_first_lower": state_path["native_lower"],
            "native_end_first_upper": state_path["native_upper"],
            "repaired_lower": state_path["repaired_lower"],
            "repaired_upper": state_path["repaired_upper"],
        }
    )

    write_csv(
        output / "adaptive_flowstar_repaired_rows.csv",
        [
            {
                "step": row["step"],
                "time": row["time"],
                "state": row["state"],
                "tlo": row["tube_lower"],
                "thi": row["tube_upper"],
                "elo": row["endpoint_lower"],
                "ehi": row["endpoint_upper"],
                "poly": row["poly_width"],
                "rem": row["remainder_width"],
            }
            for row in authoritative_rows
        ],
    )
    source_locations = {
        "generated_exporter": (
            "experiments/three_way_comparison_repair/"
            "run_flowstar_audit.py::render_parity_cpp"
        ),
        "flowpipe_compose": "flowstar-toolbox/Continuous.cpp::Flowpipe::compose",
        "collapsed_endpoint": (
            "flowstar-toolbox/TaylorModel.h::"
            "TaylorModelVec::evaluate_time"
        ),
        "coefficient_substitution": (
            "flowstar-toolbox/Polynomial.h::Polynomial::evaluate_time"
        ),
        "native_fixed_domain": (
            "flowstar-toolbox/TaylorModel.h::TaylorModelVec::intEval"
        ),
        "adaptive_symbolic_advance": (
            "flowstar-toolbox/Continuous.cpp::"
            "Flowpipe::advance_adaptive_stepsize(..., Symbolic_Remainder&)"
        ),
    }
    benchmark_source = (
        stock_root / "benchmarks" / "continuous" / "vanderpol" / "vanderpol.cpp"
    )
    exporter_source = (
        HERE.parent / "three_way_comparison_repair" / "run_flowstar_audit.py"
    )
    passed = bool(
        parity_summary.get("passed", False)
        and variant_summaries["variable_leaf_patch"]["checks"][
            "native_fixed_domain"
        ]["trajectory_failures"]
        == 0
        and variant_summaries["variable_leaf_patch"]["checks"][
            "repaired_raw_endpoint"
        ]["trajectory_failures"]
        == 0
    )
    trace = {
        "classification": (
            "collapsed evaluate_time endpoint under-enclosure; native "
            "fixed-domain evaluation contains all deterministic samples"
        ),
        "sampling_scope": "deterministic_DOP853_numerical_sanity_check_non_proof",
        "equation": ["x' = y", "y' = y - x - x^2*y"],
        "mu": 1.0,
        "state_order": ["position", "velocity"],
        "first_failure": first_failure,
        "variants": variant_summaries,
        "original_benchmark": {
            "stdout": str(original_stdout),
            "segments": parity_summary.get("original_segments"),
            "reached_horizon_10": parity_summary.get(
                "original_reached_horizon_10"
            ),
            "generated_schedule_agreement": parity_summary.get(
                "schedule_agreement"
            ),
        },
        "provenance": {
            "torch_sha": git_sha(HERE.parents[1]),
            "flowstar_stock_sha": git_sha(stock_root),
            "flowstar_audit_sha": git_sha(audit_root),
            "benchmark_source": str(benchmark_source),
            "benchmark_sha256": _sha256(benchmark_source),
            "exporter_source": str(exporter_source),
            "exporter_sha256": _sha256(exporter_source),
        },
        "source_locations": source_locations,
        "repair": {
            "decision": "use_native_fixed_domain_hull",
            "raw_endpoint_semantics": (
                "hull(collapsed evaluate_time endpoint, composed native "
                "flowpipe evaluated with local time fixed to accepted h)"
            ),
            "collapsed_padding_is_explicit_independent_remainder": True,
            "excluded_from_authoritative": False,
        },
        "passed": passed,
    }
    write_json(output / "flowstar_adaptive_failure_trace.json", trace)
    write_json(
        output / "flowstar_adaptive_trajectory_summary.json",
        {
            "passed": passed,
            "classification": trace["classification"],
            "first_failure": first_failure,
            "variant_failure_counts": {
                name: {
                    path: value["trajectory_failures"]
                    for path, value in summary["checks"].items()
                }
                for name, summary in variant_summaries.items()
            },
            "original_schedule_parity": parity_summary.get("passed", False),
            "authoritative_variant": "variable_leaf_patch",
            "authoritative_repaired_trajectory_failures": variant_summaries[
                "variable_leaf_patch"
            ]["checks"]["repaired_raw_endpoint"]["trajectory_failures"],
            "excluded_from_authoritative": False,
        },
    )
    if not passed:
        raise SystemExit("adaptive Flow* trajectory audit failed")
    return trace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--parity-output", required=True)
    parser.add_argument("--parity-summary", required=True)
    parser.add_argument("--spec", default=str(HERE / "benchmark_spec.yaml"))
    args = parser.parse_args()
    summary = json.loads(Path(args.parity_summary).read_text(encoding="utf-8"))
    run_adaptive_audit(
        load_spec(args.spec),
        Path(args.output_dir).resolve(),
        parity_summary=summary,
        parity_output=Path(args.parity_output).resolve(),
    )


if __name__ == "__main__":
    main()
