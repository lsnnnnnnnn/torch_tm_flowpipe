#!/usr/bin/env python3
"""Compile and run Flow* fixed-order-2 segments for all three contracts."""
from __future__ import annotations

import argparse
import math
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from common import (
    PROTOCOL_A,
    PROTOCOL_B,
    PROTOCOL_C,
    RAW_FIELDS,
    RUN_FIELDS,
    base_run,
    copy_runtime_fields,
    exact_interval_for_row,
    flowstar_expression,
    git_sha,
    iter_configurations,
    load_spec,
    make_row,
    median,
    write_csv,
    write_json,
)

FLOWSTAR_ROOT = Path(
    os.environ.get("FLOWSTAR_ROOT", "/srv/local/shengenli/flowstar")
).resolve()

ROW_RE = re.compile(
    r"^COMMON_ROW (?P<step>\d+) (?P<time>[-+0-9.eE]+) "
    r"(?P<kind>\w+) (?P<state>\d+) (?P<lo>[-+0-9.eE]+) "
    r"(?P<hi>[-+0-9.eE]+) (?P<poly>[-+0-9.eE]+) "
    r"(?P<rem>[-+0-9.eE]+)$"
)
STEP_RE = re.compile(
    r"^COMMON_STEP (?P<step>\d+) (?P<advanced>-?\d+) "
    r"(?P<seconds>[-+0-9.eE]+)$"
)
SUPPORT_RE = re.compile(
    r"^COMMON_SUPPORT (?P<step>\d+) (?P<state>\d+) "
    r"(?P<degree>\d+) (?P<terms>\d+)$"
)
FAIL_RE = re.compile(r"^COMMON_FAILURE (?P<step>\d+) (?P<code>-?\d+)$")
POINT_RE = re.compile(
    r"^COMMON_POINT (?P<point>\d+) (?P<state>\d+) "
    r"(?P<lo>[-+0-9.eE]+) (?P<hi>[-+0-9.eE]+)$"
)
ORDER_GUARD_RE = re.compile(
    r"^COMMON_ORDER_GUARD order1_supported=(?P<order1>[01]) "
    r"order2_supported=(?P<order2>[01])$"
)


def _number(value: float) -> str:
    if not math.isfinite(value):
        raise ValueError(value)
    return f"{value:.17g}"


def render_cpp(
    system: Mapping[str, Any],
    *,
    protocol: str,
    h: float,
    horizon: float,
    remainder_estimation: float,
    cutoff: float,
) -> str:
    state_names = list(system["state_names"])
    expressions = [
        flowstar_expression(polynomial, state_names)
        for polynomial in system["rhs"]
    ]
    declarations = "\n".join(
        f'  int state_{index}_id = vars.declareVar("{name}");'
        for index, name in enumerate(state_names)
    )
    box_assignments = "\n".join(
        f"  initial_box[state_{index}_id] = "
        f"Interval({_number(float(lower))}, {_number(float(upper))});"
        for index, (lower, upper) in enumerate(system["initial_box"])
    )
    point_blocks: list[str] = []
    for point_index, point in enumerate(system["point_checks"]):
        assignments = "\n".join(
            f"    point[state_{state}_id] = Interval({_number(float(value))});"
            for state, value in enumerate(point)
        )
        point_blocks.append(
            f"""
  {{
    vector<Interval> point(vars.size(), Interval(0.0));
{assignments}
    for(unsigned int state = 0; state < ode.expressions.size(); ++state) {{
      Interval value;
      ode.expressions[state].evaluate(value, point);
      printf("COMMON_POINT {point_index} %u %.17g %.17g\\n",
             state, value.inf(), value.sup());
    }}
  }}"""
        )
    quoted = ", ".join(f'"{expression}"' for expression in expressions)
    if protocol == PROTOCOL_B:
        carry = """
    // The controlled protocol carries only the extracted componentwise box.
    current = Flowpipe(endpoint_box);
"""
    elif protocol == PROTOCOL_C:
        carry = """
    // Native Flow* carry: retain the full accepted Taylor-model flowpipe.
    current = next;
"""
    else:
        carry = """
    // A one-step run has no carried representation.
"""
    return f"""
#include "Continuous.h"
#include <cmath>
#include <cstdio>
#include <ctime>
#include <vector>
using namespace flowstar;
using namespace std;

static vector<Real> endpoint_powers(const Interval &time_domain) {{
  Real h;
  time_domain.sup(h);
  vector<Real> powers;
  powers.push_back(1);
  powers.push_back(h);
  powers.push_back(h * h);
  return powers;
}}

static void print_tmv(
    unsigned int step,
    double absolute_time,
    const char *kind,
    const TaylorModelVec<Real> &tmv,
    const vector<Interval> &domain) {{
  vector<Interval> box;
  tmv.intEval(box, domain);
  for(unsigned int state = 0; state < tmv.tms.size(); ++state) {{
    Interval polynomial_range;
    tmv.tms[state].polyRange(polynomial_range, domain);
    const Interval &remainder = tmv.tms[state].remainder;
    printf("COMMON_ROW %u %.17g %s %u %.17g %.17g %.17g %.17g\\n",
           step, absolute_time, kind, state,
           box[state].inf(), box[state].sup(),
           polynomial_range.sup() - polynomial_range.inf(),
           remainder.sup() - remainder.inf());
  }}
}}

int main() {{
  Variables vars;
{declarations}
  ODE<Real> ode({{{quoted}}}, vars);
{''.join(point_blocks)}
  Computational_Setting order1_probe(vars);
  bool order1_supported = order1_probe.setFixedStepsize({_number(h)}, 1);
  Computational_Setting setting(vars);
  const unsigned int local_order = 2;
  bool order2_supported = setting.setFixedStepsize({_number(h)}, local_order);
  printf("COMMON_ORDER_GUARD order1_supported=%d order2_supported=%d\\n",
         order1_supported ? 1 : 0, order2_supported ? 1 : 0);
  if(order1_supported || !order2_supported) return 3;
  setting.setCutoffThreshold({_number(cutoff)});
  vector<Interval> estimates(
      vars.size(),
      Interval(-{_number(abs(remainder_estimation))},
               {_number(abs(remainder_estimation))}));
  setting.setRemainderEstimation(estimates);
  setting.printOff();
  vector<Constraint> invariant;
  vector<Interval> initial_box(vars.size());
{box_assignments}
  Flowpipe current(initial_box);
  const unsigned int steps =
      (unsigned int)floor({_number(horizon)} / {_number(h)} + 0.5);
  double absolute_time = 0.0;
  for(unsigned int step = 1; step <= steps; ++step) {{
    clock_t begin = clock();
    Flowpipe next;
    int advanced = current.advance(
        next, ode.expressions, setting.tm_setting, invariant, setting.g_setting);
    clock_t end = clock();
    printf("COMMON_STEP %u %d %.17g\\n", step, advanced,
           (double)(end - begin) / CLOCKS_PER_SEC);
    if(advanced != 1) {{
      printf("COMMON_FAILURE %u %d\\n", step, advanced);
      break;
    }}

    // Audited extraction workaround: advance proves the configured candidate
    // is a self-map, but stock refinement can return a later unvalidated image.
    // Restore the proved candidate before extraction and before native carry.
    for(unsigned int state = 0; state < next.tmvPre.tms.size(); ++state)
      next.tmvPre.tms[state].remainder =
          setting.tm_setting.remainder_estimation[state];

    absolute_time += next.domain[0].sup();
    TaylorModelVec<Real> composed;
    next.compose(composed, local_order, setting.tm_setting.cutoff_threshold);
    print_tmv(step, absolute_time, "tube", composed, next.domain);

    TaylorModelVec<Real> endpoint;
    composed.evaluate_time(endpoint, endpoint_powers(next.domain[0]));
    vector<Interval> endpoint_domain = next.domain;
    endpoint_domain[0] = Interval(0.0);
    print_tmv(step, absolute_time, "endpoint", endpoint, endpoint_domain);
    vector<Interval> endpoint_box;
    endpoint.intEval(endpoint_box, endpoint_domain);
    for(unsigned int state = 0; state < composed.tms.size(); ++state)
      printf("COMMON_SUPPORT %u %u %u %u\\n", step, state,
             composed.tms[state].degree(),
             (unsigned int)composed.tms[state].expansion.terms.size());
{carry}
  }}
  return 0;
}}
""".lstrip()


def _compile(
    source: Path,
    executable: Path,
    timeout_s: float,
) -> tuple[subprocess.CompletedProcess[str], float]:
    command = [
        "g++",
        "-O3",
        "-w",
        "-fpermissive",
        "-std=c++11",
        "-I",
        str(FLOWSTAR_ROOT / "flowstar-toolbox"),
        str(source),
        "-L",
        str(FLOWSTAR_ROOT / "flowstar-toolbox"),
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
    result = subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout_s,
    )
    return result, time.perf_counter() - started


def _execute(
    executable: Path,
    timeout_s: float,
) -> tuple[subprocess.CompletedProcess[str], float]:
    started = time.perf_counter()
    result = subprocess.run(
        [str(executable)],
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout_s,
    )
    return result, time.perf_counter() - started


def _parse(stdout: str) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    step_times: list[float] = []
    supports: list[dict[str, int]] = []
    points: list[dict[str, Any]] = []
    failure_step: int | None = None
    order_guard: dict[str, bool] | None = None
    for line in stdout.splitlines():
        match = ROW_RE.match(line)
        if match:
            rows.append(
                {
                    "step": int(match["step"]),
                    "time": float(match["time"]),
                    "kind": match["kind"],
                    "state": int(match["state"]),
                    "lower": float(match["lo"]),
                    "upper": float(match["hi"]),
                    "polynomial_width": float(match["poly"]),
                    "remainder_width": float(match["rem"]),
                }
            )
            continue
        match = STEP_RE.match(line)
        if match:
            step_times.append(float(match["seconds"]))
            continue
        match = SUPPORT_RE.match(line)
        if match:
            supports.append(
                {
                    "step": int(match["step"]),
                    "state": int(match["state"]),
                    "degree": int(match["degree"]),
                    "terms": int(match["terms"]),
                }
            )
            continue
        match = FAIL_RE.match(line)
        if match:
            failure_step = int(match["step"])
            continue
        match = POINT_RE.match(line)
        if match:
            points.append(
                {
                    "point_index": int(match["point"]),
                    "state": int(match["state"]),
                    "lower": float(match["lo"]),
                    "upper": float(match["hi"]),
                }
            )
            continue
        match = ORDER_GUARD_RE.match(line)
        if match:
            order_guard = {
                "order1_supported": bool(int(match["order1"])),
                "order2_supported": bool(int(match["order2"])),
            }
    return {
        "rows": rows,
        "step_times": step_times,
        "supports": supports,
        "points": points,
        "failure_step": failure_step,
        "order_guard": order_guard,
    }


def _append_initial_rows(
    rows: list[dict[str, Any]],
    run: Mapping[str, Any],
    system: Mapping[str, Any],
) -> None:
    for state_index, (state_name, bounds) in enumerate(
        zip(system["state_names"], system["initial_box"])
    ):
        lower, upper = map(float, bounds)
        rows.append(
            make_row(
                run,
                state_index=state_index,
                state_name=state_name,
                step_index=0,
                time_value=0.0,
                interval_kind="endpoint",
                lower=lower,
                upper=upper,
                exact=(lower, upper),
                polynomial_width=upper - lower,
                interval_remainder_width=0.0,
                row_status="validated",
                native_validation_status="initial_set",
            )
        )


def run_configuration(
    spec: Mapping[str, Any],
    config: Mapping[str, Any],
    output: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    system_name = str(config["system"])
    system = spec["systems"][system_name]
    protocol = str(config["protocol"])
    h = float(config["h"])
    steps = int(config["steps"])
    if protocol == PROTOCOL_A:
        carried_representation = "none_one_segment"
        reset_policy = "not_applicable"
    elif protocol == PROTOCOL_B:
        carried_representation = "componentwise_axis_aligned_box"
        reset_policy = "endpoint_box_exact_no_inflation"
    else:
        carried_representation = "native_Flowstar_Taylor_model_flowpipe"
        reset_policy = "native_QR_normalized_Taylor_model_carry"
    workaround = str(spec["flowstar"]["extraction_workaround"])
    run = base_run(
        tool="flowstar",
        tool_variant="minimum_supported_fixed_order_2",
        config=config,
        local_order=2,
        local_retained_basis="complete_total_degree_2(local_time,normalized_generators)",
        carried_representation=carried_representation,
        reset_policy=reset_policy,
        validator="Flowstar_public_advance_initial_Picard_candidate_inclusion",
        dtype="MPFR_interval",
        device="cpu",
        tool_git_sha=git_sha(FLOWSTAR_ROOT),
        adapter_git_sha=git_sha(REPO_ROOT),
        extraction_workaround=workaround,
    )
    rows: list[dict[str, Any]] = []
    _append_initial_rows(rows, run, system)
    run_dir = (
        output
        / "logs"
        / "flowstar"
        / protocol
        / f"{system_name}_h{h:g}_T{float(config['horizon']):g}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    source = run_dir / "common_contract.cpp"
    executable = run_dir / "common_contract"
    remainder = float(
        spec["flowstar"]["remainder_estimation"][system_name]
    )
    source.write_text(
        render_cpp(
            system,
            protocol=protocol,
            h=h,
            horizon=float(config["horizon"]),
            remainder_estimation=remainder,
            cutoff=float(spec["flowstar"]["cutoff"]),
        ),
        encoding="utf-8",
    )
    orchestration_started = time.perf_counter()
    compiled, build_time = _compile(
        source, executable, float(spec["timeout_s"])
    )
    run["build_time_s"] = build_time
    (run_dir / "compile.stdout.txt").write_text(
        compiled.stdout or "", encoding="utf-8"
    )
    (run_dir / "compile.stderr.txt").write_text(
        compiled.stderr or "", encoding="utf-8"
    )
    if compiled.returncode != 0:
        run.update(
            run_status="build_failed",
            row_status="build_failed",
            native_validation_status="not_run",
            first_failure_time=0.0,
            successful_horizon=0.0,
            message=f"Flow* compilation returned {compiled.returncode}",
        )
        for state_index, state_name in enumerate(system["state_names"]):
            rows.append(
                make_row(
                    run,
                    state_index=state_index,
                    state_name=state_name,
                    step_index=1,
                    time_value=h,
                    interval_kind="failure_marker",
                    lower="",
                    upper="",
                    row_status="build_failed",
                    native_validation_status="not_run",
                    message=run["message"],
                )
            )
        run["orchestration_time_s"] = time.perf_counter() - orchestration_started
        copy_runtime_fields(run, rows)
        return rows, run, []
    process, executable_runtime = _execute(
        executable, float(spec["timeout_s"])
    )
    (run_dir / "run.stdout.txt").write_text(
        process.stdout or "", encoding="utf-8"
    )
    (run_dir / "run.stderr.txt").write_text(
        process.stderr or "", encoding="utf-8"
    )
    parsed = _parse(process.stdout or "")
    run["executable_runtime_s"] = executable_runtime
    run["first_execution_time_s"] = (
        parsed["step_times"][0] if parsed["step_times"] else math.nan
    )
    run["steady_runtime_per_step_s"] = median(
        parsed["step_times"][1:] or parsed["step_times"]
    )
    if protocol == PROTOCOL_A and process.returncode == 0:
        repetitions: list[float] = []
        for _ in range(int(spec["steady_repetitions"])):
            repeated, _ = _execute(executable, float(spec["timeout_s"]))
            repeated_parsed = _parse(repeated.stdout or "")
            if repeated.returncode or not repeated_parsed["step_times"]:
                raise RuntimeError("Flow* repeated one-step timing run failed")
            repetitions.append(repeated_parsed["step_times"][0])
        run["timing_repetitions_s"] = repetitions
        run["steady_runtime_per_step_s"] = median(repetitions)
    analytic_violation_step: int | None = None
    for item in parsed["rows"]:
        exact_boxes = exact_interval_for_row(
            system_name,
            item["kind"],
            item["time"],
            h,
            system["initial_box"],
        )
        exact = None if exact_boxes is None else exact_boxes[item["state"]]
        contained = (
            True
            if exact is None
            else item["lower"] <= exact[0] + 1e-12
            and item["upper"] >= exact[1] - 1e-12
        )
        if not contained and analytic_violation_step is None:
            analytic_violation_step = item["step"]
        rows.append(
            make_row(
                run,
                state_index=item["state"],
                state_name=system["state_names"][item["state"]],
                step_index=item["step"],
                time_value=item["time"],
                interval_kind=item["kind"],
                lower=item["lower"],
                upper=item["upper"],
                exact=exact,
                polynomial_width=item["polynomial_width"],
                interval_remainder_width=item["remainder_width"],
                row_status=(
                    "validated" if contained else "analytic_reference_violation"
                ),
                native_validation_status="validated",
                message=(
                    "" if contained else "analytic exact interval is not contained"
                ),
            )
        )
    completed = max(
        (item["step"] for item in parsed["rows"] if item["kind"] == "endpoint"),
        default=0,
    )
    order_guard = parsed["order_guard"]
    guard_ok = bool(
        order_guard
        and not order_guard["order1_supported"]
        and order_guard["order2_supported"]
    )
    if analytic_violation_step is not None:
        run.update(
            run_status="analytic_reference_violation",
            native_validation_status="validated_but_reference_failed",
            first_failure_time=analytic_violation_step * h,
            successful_horizon=(analytic_violation_step - 1) * h,
            completed_steps=analytic_violation_step - 1,
            message="analytic exact interval is not contained",
        )
    elif not guard_ok:
        run.update(
            run_status="order_guard_failed",
            native_validation_status="not_run",
            first_failure_time=0.0,
            successful_horizon=0.0,
            completed_steps=0,
            message="Flow* minimum fixed-order guard did not report order 1 unsupported/order 2 supported",
        )
    elif process.returncode != 0:
        run.update(
            run_status="execution_failed",
            native_validation_status="failed",
            first_failure_time=(completed + 1) * h,
            successful_horizon=completed * h,
            completed_steps=completed,
            message=f"Flow* executable returned {process.returncode}",
        )
    elif completed < steps:
        failure_step = parsed["failure_step"] or completed + 1
        run.update(
            run_status="validation_failed",
            native_validation_status="failed",
            first_failure_time=failure_step * h,
            successful_horizon=completed * h,
            completed_steps=completed,
            message="Flow* public advance returned validation failure",
        )
        for state_index, state_name in enumerate(system["state_names"]):
            rows.append(
                make_row(
                    run,
                    state_index=state_index,
                    state_name=state_name,
                    step_index=failure_step,
                    time_value=failure_step * h,
                    interval_kind="failure_marker",
                    lower="",
                    upper="",
                    row_status="validation_failed",
                    native_validation_status="failed",
                    message=run["message"],
                )
            )
    else:
        run.update(
            run_status="success",
            row_status="validated",
            native_validation_status="validated",
            completed_steps=steps,
            successful_horizon=float(config["horizon"]),
        )
    run["validation_attempts"] = len(parsed["step_times"])
    run["measured_polynomial_support"] = str(
        {
            "maximum_degree": max(
                (item["degree"] for item in parsed["supports"]), default=0
            ),
            "maximum_terms_per_state": max(
                (item["terms"] for item in parsed["supports"]), default=0
            ),
        }
    )
    run["minimum_order_guard"] = order_guard
    run["source"] = str(source)
    run["stdout_log"] = str(run_dir / "run.stdout.txt")
    run["orchestration_time_s"] = max(
        time.perf_counter() - orchestration_started - build_time - executable_runtime,
        0.0,
    )
    copy_runtime_fields(run, rows)
    point_values = []
    by_point: dict[int, list[dict[str, Any]]] = {}
    for item in parsed["points"]:
        by_point.setdefault(item["point_index"], []).append(item)
    for point_index, point in enumerate(system["point_checks"]):
        components = sorted(
            by_point.get(point_index, []), key=lambda value: value["state"]
        )
        if components:
            point_values.append(
                {
                    "system": system_name,
                    "point_index": point_index,
                    "point": list(map(float, point)),
                    "value": [
                        0.5 * (item["lower"] + item["upper"])
                        for item in components
                    ],
                    "intervals": [
                        [item["lower"], item["upper"]] for item in components
                    ],
                }
            )
    return rows, run, point_values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", default=str(HERE / "benchmark_spec.yaml"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--protocols", nargs="*")
    parser.add_argument("--systems", nargs="*")
    args = parser.parse_args()
    spec = load_spec(args.spec)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    runs: list[dict[str, Any]] = []
    point_map: dict[tuple[str, int], dict[str, Any]] = {}
    for config in iter_configurations(
        spec,
        smoke=args.smoke,
        protocols=args.protocols,
        systems=args.systems,
    ):
        config_rows, run, points = run_configuration(spec, config, output)
        rows.extend(config_rows)
        runs.append(run)
        for point in points:
            point_map[(point["system"], point["point_index"])] = point
        print(
            f"Flow* {config['protocol']} {config['system']} "
            f"h={config['h']:g} T={config['horizon']:g}: "
            f"{run['completed_steps']}/{run['requested_steps']} {run['run_status']}",
            flush=True,
        )
    write_csv(output / "flowstar_raw_results.csv", rows, RAW_FIELDS)
    write_csv(output / "flowstar_runs.csv", runs, RUN_FIELDS)
    write_json(output / "flowstar_runs.json", runs)
    write_json(
        output / "flowstar_point_evaluations.json",
        {
            "tool": "flowstar",
            "dtype": "MPFR_interval",
            "device": "cpu",
            "values": [
                point_map[key] for key in sorted(point_map)
            ],
        },
    )


if __name__ == "__main__":
    main()
