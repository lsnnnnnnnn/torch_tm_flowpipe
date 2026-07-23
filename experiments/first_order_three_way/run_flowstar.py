#!/usr/bin/env python3
"""Generate, compile, and run fixed-setting Flow* toolbox benchmarks."""
from __future__ import annotations

import argparse
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from common import (
    flowstar_expression,
    git_sha,
    interval_row,
    iter_configurations,
    load_spec,
    median_iqr,
    output_dir_from_args,
    raw_run_template,
    utc_timestamp,
    write_csv,
    write_json,
)

FLOWSTAR_ROOT_DEFAULT = Path("/srv/local/shengenli/flowstar")
ROW_RE = re.compile(
    r"^FLOWSTAR_ROW\s+(?P<step>\d+)\s+(?P<time>[-+0-9.eE]+)\s+"
    r"(?P<kind>\w+)\s+(?P<state>\d+)\s+(?P<lo>[-+0-9.eE]+)\s+(?P<hi>[-+0-9.eE]+)$"
)
RUNTIME_RE = re.compile(r"^FLOWSTAR_RUNTIME_S\s+([-+0-9.eE]+)$", re.MULTILINE)
SEGMENTS_RE = re.compile(r"^FLOWSTAR_SEGMENTS\s+(\d+)$", re.MULTILINE)
DEGREE_RE = re.compile(r"^FLOWSTAR_EFFECTIVE_MAX_DEGREE\s+(\d+)$", re.MULTILINE)


def _cpp_number(value: float) -> str:
    if not math.isfinite(float(value)):
        raise ValueError(value)
    return f"{float(value):.17g}"


def render_cpp(
    system: Mapping[str, Any],
    *,
    h: float,
    horizon: float,
    order: int,
    remainder_estimation: float,
    cutoff: float,
) -> str:
    state_names = list(system["state_names"])
    expressions = [flowstar_expression(poly, state_names) for poly in system["rhs"]]
    lines = [
        '#include "Continuous.h"',
        "#include <algorithm>",
        "#include <ctime>",
        "#include <cstdio>",
        "#include <vector>",
        "using namespace flowstar;",
        "using namespace std;",
        "int main() {",
        "  Variables vars;",
    ]
    for index, name in enumerate(state_names):
        lines.append(f'  int state_{index}_id = vars.declareVar("{name}");')
    quoted = ", ".join(f'"{expr}"' for expr in expressions)
    lines.extend(
        [
            f"  ODE<Real> ode({{{quoted}}}, vars);",
            "  Computational_Setting setting(vars);",
            f"  bool fixed_ok = setting.setFixedStepsize({_cpp_number(h)}, {int(order)});",
            "  if(!fixed_ok) {",
            f'    printf("FLOWSTAR_UNSUPPORTED_ORDER {int(order)}\\n");',
            "    return 3;",
            "  }",
            f"  setting.setCutoffThreshold({_cpp_number(cutoff)});",
            "  vector<Interval> remainder_estimation(vars.size());",
            "  for(unsigned int i = 0; i < vars.size(); ++i) {",
            f"    remainder_estimation[i] = Interval(-{_cpp_number(abs(remainder_estimation))}, {_cpp_number(abs(remainder_estimation))});",
            "  }",
            "  setting.setRemainderEstimation(remainder_estimation);",
            "  setting.printOff();",
            "  vector<Interval> box(vars.size());",
        ]
    )
    for index, (lo, hi) in enumerate(system["initial_box"]):
        lines.append(f"  box[state_{index}_id] = Interval({_cpp_number(lo)}, {_cpp_number(hi)});")
    lines.extend(
        [
            "  for(unsigned int d = 0; d < box.size(); ++d) {",
            '    printf("FLOWSTAR_ROW 0 0 endpoint %u %.17g %.17g\\n", d, box[d].inf(), box[d].sup());',
            "  }",
            "  Flowpipe initial_set(box);",
            "  vector<Constraint> safe_set;",
            "  Result_of_Reachability result;",
            "  clock_t begin = clock();",
            f"  ode.reach(result, initial_set, {_cpp_number(horizon)}, setting, safe_set);",
            "  clock_t end = clock();",
            '  printf("FLOWSTAR_RUNTIME_S %.17g\\n", (double)(end - begin) / CLOCKS_PER_SEC);',
            '  printf("FLOWSTAR_COMPLETED %d\\n", result.isCompleted() ? 1 : 0);',
            "  result.transformToTaylorModels(setting);",
            '  printf("FLOWSTAR_SEGMENTS %u\\n", result.tmv_flowpipes.size());',
            "  unsigned int step_index = 0;",
            "  unsigned int effective_degree = 0;",
            "  double absolute_time = 0.0;",
            "  for(list<TaylorModelFlowpipe>::const_iterator it = result.tmv_flowpipes.tmv_flowpipes.begin();",
            "      it != result.tmv_flowpipes.tmv_flowpipes.end(); ++it) {",
            "    ++step_index;",
            "    double local_h = it->domain[0].sup();",
            "    absolute_time += local_h;",
            "    vector<Interval> tube;",
            "    it->tmv_flowpipe.intEval(tube, it->domain);",
            "    for(unsigned int d = 0; d < tube.size(); ++d) {",
            '      printf("FLOWSTAR_ROW %u %.17g tube %u %.17g %.17g\\n", step_index, absolute_time, d, tube[d].inf(), tube[d].sup());',
            "      effective_degree = std::max(effective_degree, it->tmv_flowpipe.tms[d].degree());",
            "    }",
            "    Real t;",
            "    it->domain[0].sup(t);",
            "    vector<Real> powers;",
            "    powers.push_back(1);",
            "    powers.push_back(t);",
            "    Real p = t;",
            "    for(unsigned int k = 2; k <= setting.tm_setting.step_end_exp_table.size(); ++k) {",
            "      p *= t;",
            "      powers.push_back(p);",
            "    }",
            "    TaylorModelVec<Real> endpoint_tm;",
            "    it->tmv_flowpipe.evaluate_time(endpoint_tm, powers);",
            "    vector<Interval> endpoint_domain = it->domain;",
            "    endpoint_domain[0] = Interval(0.0);",
            "    vector<Interval> endpoint;",
            "    endpoint_tm.intEval(endpoint, endpoint_domain);",
            "    for(unsigned int d = 0; d < endpoint.size(); ++d) {",
            '      printf("FLOWSTAR_ROW %u %.17g endpoint %u %.17g %.17g\\n", step_index, absolute_time, d, endpoint[d].inf(), endpoint[d].sup());',
            "    }",
            "  }",
            '  printf("FLOWSTAR_EFFECTIVE_MAX_DEGREE %u\\n", effective_degree);',
            "  return result.isCompleted() ? 0 : 4;",
            "}",
            "",
        ]
    )
    return "\n".join(lines)


def _compile(
    source: Path,
    executable: Path,
    *,
    flowstar_root: Path,
    timeout_s: float,
    log_dir: Path,
) -> tuple[str, float, str]:
    command = [
        # GCC 15 diagnoses a const-incorrect, uninstantiated template body in
        # this 2020-era toolbox. -fpermissive matches the successful historical
        # build without modifying the read-only external repository.
        "g++", "-O3", "-w", "-fpermissive", "-std=c++11",
        "-I", str(flowstar_root / "flowstar-toolbox"),
        str(source),
        "-L", str(flowstar_root / "flowstar-toolbox"),
        "-o", str(executable),
        "-lflowstar", "-lmpfr", "-lgmp", "-lgsl", "-lgslcblas", "-lm", "-lglpk",
    ]
    started = time.perf_counter()
    try:
        proc = subprocess.run(command, text=True, capture_output=True, timeout=timeout_s, check=False)
        status = "ok" if proc.returncode == 0 else "compile_failed"
        message = "" if proc.returncode == 0 else f"compiler returned {proc.returncode}"
    except subprocess.TimeoutExpired as exc:
        proc = None
        status, message = "timeout", "compilation timed out"
        (log_dir / "compile.stdout.txt").write_text(exc.stdout or "", encoding="utf-8")
        (log_dir / "compile.stderr.txt").write_text(exc.stderr or "", encoding="utf-8")
    elapsed = time.perf_counter() - started
    if proc is not None:
        (log_dir / "compile.stdout.txt").write_text(proc.stdout or "", encoding="utf-8")
        (log_dir / "compile.stderr.txt").write_text(proc.stderr or "", encoding="utf-8")
    return status, elapsed, message


def _execute(executable: Path, *, timeout_s: float, log_prefix: Path) -> tuple[str, float, str, str]:
    started = time.perf_counter()
    try:
        proc = subprocess.run(
            [str(executable)],
            text=True,
            capture_output=True,
            timeout=timeout_s,
            check=False,
            env={**os.environ},
        )
        elapsed = time.perf_counter() - started
        stdout, stderr = proc.stdout or "", proc.stderr or ""
        if "FLOWSTAR_UNSUPPORTED_ORDER" in stdout:
            status = "unsupported_order"
        elif proc.returncode == 0 and "FLOWSTAR_COMPLETED 1" in stdout:
            status = "certified_ok"
        elif "FLOWSTAR_COMPLETED 0" in stdout or proc.returncode == 4:
            status = "validation_failed"
        else:
            status = "numerical_error"
    except subprocess.TimeoutExpired as exc:
        elapsed = time.perf_counter() - started
        stdout, stderr = exc.stdout or "", exc.stderr or ""
        status = "timeout"
    log_prefix.parent.mkdir(parents=True, exist_ok=True)
    log_prefix.with_suffix(".stdout.txt").write_text(stdout, encoding="utf-8")
    log_prefix.with_suffix(".stderr.txt").write_text(stderr, encoding="utf-8")
    return status, elapsed, stdout, stderr


def _parse_rows(stdout: str) -> list[dict[str, Any]]:
    parsed: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        match = ROW_RE.match(line.strip())
        if match:
            parsed.append(
                {
                    "step_index": int(match.group("step")),
                    "time": float(match.group("time")),
                    "interval_kind": match.group("kind"),
                    "state_index": int(match.group("state")),
                    "lower": float(match.group("lo")),
                    "upper": float(match.group("hi")),
                }
            )
    return parsed


def _make_output(
    *,
    spec: Mapping[str, Any],
    config: Mapping[str, Any],
    order: int,
    protocol: str,
    compile_status: str,
    build_s: float,
    compile_message: str,
    first_status: str,
    first_wall_s: float,
    first_stdout: str,
    repeated_walls: list[float],
    repeated_internal: list[float],
    source_generation_s: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    system_name = str(config["system"])
    h, horizon, steps = float(config["h"]), float(config["horizon"]), int(config["steps"])
    parsed = _parse_rows(first_stdout)
    segments_match = SEGMENTS_RE.search(first_stdout)
    segments = int(segments_match.group(1)) if segments_match else 0
    degree_match = DEGREE_RE.search(first_stdout)
    measured_degree: int | str = int(degree_match.group(1)) if degree_match else ""
    if order == 1:
        retained = "unsupported: Flow* toolbox API rejects fixed order <2"
    else:
        retained = "complete_total_degree_2(local_time,normalized_generators)"
    run = raw_run_template(
        tool="flowstar",
        protocol=protocol,
        system=system_name,
        h=h,
        horizon=horizon,
        requested_order_label=f"fixed_order={order}",
        retained_basis=retained,
        effective_max_degree=measured_degree,
        truncate_to_affine=False,
        nonzero_lt="not_applicable",
        dependency_mode="toolbox_normalized_preconditioning",
        symbolic_remainder_size=0,
        cutoff=float(spec["flowstar"]["cutoff"]),
        dtype="MPFR_interval",
        device="cpu",
        git_commit=git_sha(Path("/srv/local/shengenli/flowstar")),
        environment="system_g++_flowstar_toolbox",
    )
    status = first_status
    if compile_status != "ok":
        status = "timeout" if compile_status == "timeout" else "numerical_error"
    complete = status == "certified_ok" and segments == steps
    if status == "certified_ok" and segments != steps:
        status = "validation_failed"
    if complete:
        failure_time: float | str = ""
    elif status == "validation_failed":
        failure_time = min(horizon, (segments + 1) * h)
    else:
        failure_time = 0.0
    median_s, iqr_s = median_iqr(repeated_walls)
    run.update(
        status=status,
        validation_status="validated" if complete else status,
        first_failure_time=failure_time,
        successful_horizon=horizon if complete else segments * h,
        build_time_s=build_s,
        warmup_time_s=first_wall_s,
        steady_runtime_median_s=median_s,
        steady_runtime_iqr_s=iqr_s,
        message=compile_message if compile_status != "ok" else (
            "Flow* Computational_Setting::setFixedStepsize rejects order 1" if status == "unsupported_order" else ""
        ),
    )
    rows = [
        interval_row(
            run=run,
            state_index=item["state_index"],
            step_index=item["step_index"],
            time_value=item["time"],
            interval_kind=item["interval_kind"],
            lower=item["lower"],
            upper=item["upper"],
        )
        for item in parsed
    ]
    if not rows:
        for state_index in range(len(spec["systems"][system_name]["state_names"])):
            rows.append(
                interval_row(
                    run=run,
                    state_index=state_index,
                    step_index=0,
                    time_value=0.0,
                    interval_kind="failure_marker",
                    lower="",
                    upper="",
                )
            )
    metadata = {
        **run,
        "source_generation_time_s": source_generation_s,
        "compile_status": compile_status,
        "requested_steps": steps,
        "completed_segments": segments,
        "fixed_step": True,
        "fixed_order": True,
        "adaptive_step": False,
        "adaptive_order": False,
        "remainder_estimation": float(spec["flowstar"]["remainder_estimation"]),
        "preconditioning": spec["flowstar"]["preconditioning"],
        "normalization": "Flowpipe normalized coordinates; time domain [0,h], generators [-1,1]",
        "timing_repetitions_wall_s": repeated_walls,
        "timing_repetitions_internal_reach_s": repeated_internal,
        "internal_reach_median_s": median_iqr(repeated_internal)[0] if repeated_internal else "",
        "interval_semantics": {
            "endpoint": "TaylorModelVec.evaluate_time at local h, then interval evaluation",
            "tube": "TaylorModelVec.intEval over the entire stored segment domain",
        },
    }
    return rows, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", default=str(HERE / "benchmark_spec.yaml"))
    parser.add_argument("--output-dir")
    parser.add_argument("--flowstar-root", default=os.environ.get("FLOWSTAR_ROOT", str(FLOWSTAR_ROOT_DEFAULT)))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--systems", nargs="*")
    args = parser.parse_args()
    spec = load_spec(args.spec)
    output_dir = output_dir_from_args(args.output_dir)
    flowstar_root = Path(args.flowstar_root).resolve()
    timeout_s = float(spec["timeout_s"])
    all_rows: list[dict[str, Any]] = []
    for config in iter_configurations(spec, smoke=args.smoke, systems=args.systems):
        for order, protocols in (
            (1, ("native_first_order_setting", "strict_common_affine")),
            (2, ("supplementary_native_representations",)),
        ):
            configuration_deadline = time.monotonic() + timeout_s

            def remaining_budget() -> float:
                return max(0.001, configuration_deadline - time.monotonic())

            stem = (
                f"flowstar_{config['system']}_h{config['h']:.17g}_"
                f"T{config['horizon']:.17g}_o{order}".replace(".", "p")
            )
            run_dir = output_dir / "flowstar" / stem
            run_dir.mkdir(parents=True, exist_ok=True)
            source = run_dir / f"{stem}.cpp"
            executable = run_dir / stem
            started = time.perf_counter()
            source.write_text(
                render_cpp(
                    spec["systems"][config["system"]],
                    h=float(config["h"]),
                    horizon=float(config["horizon"]),
                    order=order,
                    remainder_estimation=float(spec["flowstar"]["remainder_estimation"]),
                    cutoff=float(spec["flowstar"]["cutoff"]),
                ),
                encoding="utf-8",
            )
            source_generation_s = time.perf_counter() - started
            compile_status, build_s, compile_message = _compile(
                source,
                executable,
                flowstar_root=flowstar_root,
                timeout_s=remaining_budget(),
                log_dir=run_dir,
            )
            if compile_status == "ok" and time.monotonic() < configuration_deadline:
                first_status, first_wall_s, first_stdout, _ = _execute(
                    executable,
                    timeout_s=remaining_budget(),
                    log_prefix=run_dir / "warmup",
                )
            elif compile_status == "ok":
                compile_status = "timeout"
                compile_message = f"configuration exceeded {timeout_s:g} seconds"
                first_status, first_wall_s, first_stdout = "timeout", 0.0, ""
            else:
                first_status, first_wall_s, first_stdout = "numerical_error", 0.0, ""
            repeated_walls: list[float] = []
            repeated_internal: list[float] = []
            if compile_status == "ok":
                for repetition in range(int(spec["steady_repetitions"])):
                    if time.monotonic() >= configuration_deadline:
                        compile_status = "timeout"
                        compile_message = f"configuration exceeded {timeout_s:g} seconds"
                        break
                    repeat_status, wall, stdout, _ = _execute(
                        executable,
                        timeout_s=remaining_budget(),
                        log_prefix=run_dir / f"repeat_{repetition + 1}",
                    )
                    repeated_walls.append(wall)
                    if repeat_status == "timeout":
                        compile_status = "timeout"
                        compile_message = f"configuration exceeded {timeout_s:g} seconds"
                        break
                    runtime_match = RUNTIME_RE.search(stdout)
                    if runtime_match:
                        repeated_internal.append(float(runtime_match.group(1)))
            for protocol in protocols:
                rows, metadata = _make_output(
                    spec=spec,
                    config=config,
                    order=order,
                    protocol=protocol,
                    compile_status=compile_status,
                    build_s=build_s,
                    compile_message=compile_message,
                    first_status=first_status,
                    first_wall_s=first_wall_s,
                    first_stdout=first_stdout,
                    repeated_walls=repeated_walls,
                    repeated_internal=repeated_internal,
                    source_generation_s=source_generation_s,
                )
                all_rows.extend(rows)
                write_json(output_dir / "per_run" / f"{metadata['run_id']}.json", metadata)
            print(
                f"flowstar o{order} {config['system']} h={config['h']} T={config['horizon']} "
                f"compile={compile_status} run={first_status}",
                flush=True,
            )
    suffix = "smoke" if args.smoke else "full"
    write_csv(output_dir / f"flowstar_raw_{suffix}.csv", all_rows)
    write_json(
        output_dir / f"flowstar_manifest_{suffix}.json",
        {"timestamp": utc_timestamp(), "flowstar_root": str(flowstar_root), "rows": len(all_rows)},
    )
    print(output_dir)


if __name__ == "__main__":
    main()
