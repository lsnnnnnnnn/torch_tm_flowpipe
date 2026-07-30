#!/usr/bin/env python3
"""Run Flow* order-2 local construction with safe affine carry."""
from __future__ import annotations

import argparse
import csv
import json
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
BASELINE_EXPERIMENT = HERE.parent / "first_order_three_way"
if str(BASELINE_EXPERIMENT) not in sys.path:
    sys.path.insert(0, str(BASELINE_EXPERIMENT))

from common import exact_endpoint, flowstar_expression, git_sha, load_spec

FLOWSTAR_ROOT = Path(
    os.environ.get("FLOWSTAR_ROOT", HERE.parents[1].parent / "flowstar")
).resolve()
ROW_RE = re.compile(
    r"^MATCHED_ROW (?P<step>\d+) (?P<time>[-+0-9.eE]+) "
    r"(?P<kind>\w+) (?P<state>\d+) (?P<lo>[-+0-9.eE]+) "
    r"(?P<hi>[-+0-9.eE]+)$"
)
STEP_TIME_RE = re.compile(
    r"^MATCHED_STEP_TIME (?P<step>\d+) (?P<seconds>[-+0-9.eE]+)$"
)
COUNT_RE = re.compile(
    r"^MATCHED_COUNTS (?P<step>\d+) (?P<retained>\d+) (?P<discarded>\d+)$"
)

FIELDS = [
    "tool", "protocol", "system", "mode", "basis", "h", "horizon",
    "state_index", "step_index", "time",
    "interval_kind", "lower", "upper", "width", "local_construction_basis",
    "local_construction_order", "carried_basis", "carried_max_degree",
    "projection_method", "reset_method", "validator", "numerical_backend",
    "native_validation_passed", "exact_reference_contained",
    "sampled_trajectory_contained", "directed_rounding_or_mpfr",
    "floating_point_enclosure_candidate", "validation_failed",
    "python_orchestration_time_s", "compile_time_s", "first_call_time_s",
    "steady_step_time_s", "number_of_steps", "number_of_retained_coefficients",
    "number_of_discarded_candidates", "successful_horizon", "message",
]


def _number(value: float) -> str:
    if not math.isfinite(value):
        raise ValueError(value)
    return f"{value:.17g}"


def render_cpp(
    system: Mapping[str, Any],
    *,
    h: float,
    horizon: float,
    remainder_estimation: float,
    cutoff: float,
    affine_carry: bool,
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
    quoted = ", ".join(f'"{expression}"' for expression in expressions)
    carry_block = """
    unsigned int terms_before = 0;
    for(unsigned int state = 0; state < endpoint.tms.size(); ++state)
      terms_before += endpoint.tms[state].expansion.terms.size();
    endpoint.ctrunc(endpoint_domain, 1);
    unsigned int terms_after = 0;
    for(unsigned int state = 0; state < endpoint.tms.size(); ++state)
      terms_after += endpoint.tms[state].expansion.terms.size();
    printf("MATCHED_COUNTS %u %u %u\\n",
           step, terms_after, terms_before - terms_after);
    vector<Interval> affine_box;
    endpoint.intEval(affine_box, endpoint_domain);
    print_box(step, absolute_time, "affine_carry", affine_box);
    for(unsigned int state = 0; state < endpoint.tms.size(); ++state) {
      printf("MATCHED_CARRY_DEGREE %u %u %u\\n",
             step, state, endpoint.tms[state].degree());
    }
    current = Flowpipe(
        endpoint, endpoint_domain, setting.tm_setting.cutoff_threshold);
""" if affine_carry else """
    unsigned int terms_after = 0;
    for(unsigned int state = 0; state < endpoint.tms.size(); ++state)
      terms_after += endpoint.tms[state].expansion.terms.size();
    printf("MATCHED_COUNTS %u %u 0\\n", step, terms_after);
    for(unsigned int state = 0; state < endpoint.tms.size(); ++state) {
      printf("MATCHED_CARRY_DEGREE %u %u %u\\n",
             step, state, endpoint.tms[state].degree());
    }
    current = next;
"""
    return f"""
#include "Continuous.h"
#include <cmath>
#include <cstdio>
#include <ctime>
#include <vector>
using namespace flowstar;
using namespace std;

static void print_box(unsigned int step, double absolute_time, const char *kind,
                      const vector<Interval> &box) {{
  for(unsigned int state = 0; state < box.size(); ++state) {{
    printf("MATCHED_ROW %u %.17g %s %u %.17g %.17g\\n",
           step, absolute_time, kind, state,
           box[state].inf(), box[state].sup());
  }}
}}

static vector<Real> endpoint_powers(const Interval &time_domain) {{
  Real h;
  time_domain.sup(h);
  vector<Real> powers;
  powers.push_back(1);
  powers.push_back(h);
  Real h2 = h * h;
  powers.push_back(h2);
  return powers;
}}

int main() {{
  Variables vars;
{declarations}
  ODE<Real> ode({{{quoted}}}, vars);
  Computational_Setting setting(vars);
  const unsigned int local_order = 2;
  if(!setting.setFixedStepsize({_number(h)}, local_order)) return 3;
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
  print_box(0, 0.0, "endpoint", initial_box);
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
    printf("MATCHED_STEP_TIME %u %.17g\\n", step,
           (double)(end - begin) / CLOCKS_PER_SEC);
    printf("MATCHED_ADVANCE %u %d\\n", step, advanced);
    if(advanced != 1) return 4;

    // Keep the configured remainder candidate that the first Picard inclusion
    // check proved self-mapping.  Do not use the toolbox's un-revalidated
    // refinement image.
    for(unsigned int state = 0; state < next.tmvPre.tms.size(); ++state) {{
      next.tmvPre.tms[state].remainder =
          setting.tm_setting.remainder_estimation[state];
    }}
    absolute_time += next.domain[0].sup();
    TaylorModelVec<Real> composed;
    next.compose(composed, local_order, setting.tm_setting.cutoff_threshold);
    vector<Interval> tube;
    composed.intEval(tube, next.domain);
    print_box(step, absolute_time, "tube", tube);

    TaylorModelVec<Real> endpoint;
    composed.evaluate_time(endpoint, endpoint_powers(next.domain[0]));
    vector<Interval> endpoint_domain = next.domain;
    endpoint_domain[0] = Interval(0.0);
    vector<Interval> endpoint_box;
    endpoint.intEval(endpoint_box, endpoint_domain);
    print_box(step, absolute_time, "endpoint", endpoint_box);

{carry_block}
  }}
  printf("MATCHED_COMPLETED_STEPS %u\\n", steps);
  return 0;
}}
""".lstrip()


def _compile_and_run(
    *,
    source: Path,
    executable: Path,
    timeout_s: float,
) -> tuple[subprocess.CompletedProcess[str], subprocess.CompletedProcess[str], float, float]:
    system_include = os.environ.get("FLOWSTAR_SYSTEM_INCLUDE", "/opt/homebrew/include")
    system_library = os.environ.get("FLOWSTAR_SYSTEM_LIB", "/opt/homebrew/lib")
    command = [
        "g++", "-O3", "-w", "-fpermissive", "-std=c++11",
        "-I", str(FLOWSTAR_ROOT / "flowstar-toolbox"),
        "-I", system_include,
        str(source),
        "-L", str(FLOWSTAR_ROOT / "flowstar-toolbox"),
        "-L", system_library,
        "-o", str(executable),
        "-lflowstar", "-lmpfr", "-lgmp", "-lgsl", "-lgslcblas", "-lm", "-lglpk",
    ]
    started = time.perf_counter()
    compiled = subprocess.run(
        command, text=True, capture_output=True, check=False, timeout=timeout_s
    )
    compile_s = time.perf_counter() - started
    if compiled.returncode:
        return compiled, subprocess.CompletedProcess([], 99, "", ""), compile_s, 0.0
    started = time.perf_counter()
    run = subprocess.run(
        [str(executable)],
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout_s,
    )
    runtime_s = time.perf_counter() - started
    return compiled, run, compile_s, runtime_s


def run_configuration(
    *,
    spec: Mapping[str, Any],
    system_name: str,
    h: float,
    horizon: float,
    output: Path,
    protocol: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    system = spec["systems"][system_name]
    steps = round(horizon / h)
    affine_carry = protocol == "matched_affine_carry"
    mode_name = (
        "order2_local_affine_carry"
        if affine_carry
        else "order2_complete_carry"
    )
    run_dir = output / "flowstar" / protocol / f"{system_name}_h{h:g}_T{horizon:g}"
    run_dir.mkdir(parents=True, exist_ok=True)
    source = run_dir / f"{mode_name}.cpp"
    executable = run_dir / mode_name
    # The frozen 1e-4 candidate is rejected on the first Van der Pol step.
    # A 1e-3 candidate passes the same native inclusion test and is retained
    # without unsafe refinement.
    remainder_estimation = max(
        float(spec["flowstar"]["remainder_estimation"]),
        1e-3 if system_name == "van_der_pol" else 0.0,
    )
    source.write_text(
        render_cpp(
            system,
            h=h,
            horizon=horizon,
            remainder_estimation=remainder_estimation,
            cutoff=float(spec["flowstar"]["cutoff"]),
            affine_carry=affine_carry,
        ),
        encoding="utf-8",
    )
    orchestration_started = time.perf_counter()
    compiled, run, compile_s, runtime_s = _compile_and_run(
        source=source,
        executable=executable,
        timeout_s=float(spec["timeout_s"]),
    )
    orchestration_s = time.perf_counter() - orchestration_started
    (run_dir / "compile.stdout.txt").write_text(compiled.stdout or "", encoding="utf-8")
    (run_dir / "compile.stderr.txt").write_text(compiled.stderr or "", encoding="utf-8")
    (run_dir / "run.stdout.txt").write_text(run.stdout or "", encoding="utf-8")
    (run_dir / "run.stderr.txt").write_text(run.stderr or "", encoding="utf-8")
    parsed = []
    step_times = []
    counts = []
    for line in (run.stdout or "").splitlines():
        match = ROW_RE.match(line)
        if match:
            parsed.append(
                {
                    "step": int(match["step"]),
                    "time": float(match["time"]),
                    "kind": match["kind"],
                    "state": int(match["state"]),
                    "lower": float(match["lo"]),
                    "upper": float(match["hi"]),
                }
            )
        match = STEP_TIME_RE.match(line)
        if match:
            step_times.append(float(match["seconds"]))
        match = COUNT_RE.match(line)
        if match:
            counts.append(
                (
                    int(match["step"]),
                    int(match["retained"]),
                    int(match["discarded"]),
                )
            )
    completed = max(
        (row["step"] for row in parsed if row["kind"] == "endpoint"),
        default=0,
    )
    metadata = {
        "tool": "flowstar",
        "protocol": protocol,
        "system": system_name,
        "mode": mode_name,
        "basis": "B1_carry" if affine_carry else "B2",
        "h": h,
        "horizon": horizon,
        "local_construction_basis": "complete_total_degree_2",
        "local_construction_order": 2,
        "carried_basis": (
            "constant+affine_state_generators+independent_interval"
            if affine_carry else "complete_total_degree_2"
        ),
        "carried_max_degree": 1 if affine_carry else 2,
        "projection_method": (
            "Flowstar_ctrunc_degree_1_to_interval_remainder"
            if affine_carry else "none"
        ),
        "reset_method": (
            "stepwise_normalized_affine_carry"
            if affine_carry else "Flowstar_native_normalized_carry"
        ),
        "validator": "Flowstar_initial_Picard_candidate_inclusion_retained",
        "numerical_backend": "Flowstar_MPFR_interval_cpu",
        "directed_rounding_or_mpfr": True,
        "floating_point_enclosure_candidate": False,
        "python_orchestration_time_s": max(
            orchestration_s - compile_s - runtime_s, 0.0
        ),
        "compile_time_s": compile_s,
        "first_call_time_s": step_times[0] if step_times else math.nan,
        "steady_step_time_s": (
            statistics.median(step_times[1:] or step_times)
            if step_times else math.nan
        ),
        "number_of_steps": steps,
        "number_of_retained_coefficients": (
            counts[-1][1] if counts else ""
        ),
        "number_of_discarded_candidates": sum(item[2] for item in counts),
        "successful_horizon": completed * h,
        "remainder_estimation": remainder_estimation,
        "message": "" if run.returncode == 0 else f"Flow* returned {run.returncode}",
    }
    rows = []
    exact_checks = 0
    exact_violations = 0
    endpoints = {
        (row["step"], row["state"]): row
        for row in parsed if row["kind"] == "endpoint"
    }
    for item in parsed:
        if item["kind"] not in {"endpoint", "tube", "affine_carry"}:
            continue
        exact_ok: bool | str = ""
        if item["kind"] == "endpoint":
            exact = exact_endpoint(system_name, item["time"], system["initial_box"])
            if exact is not None:
                expected = exact[item["state"]]
                exact_checks += 1
                exact_ok = (
                    item["lower"] <= expected[0] + 1e-12
                    and item["upper"] >= expected[1] - 1e-12
                )
                if not exact_ok:
                    exact_violations += 1
        rows.append(
            {
                **metadata,
                "state_index": item["state"],
                "step_index": item["step"],
                "time": item["time"],
                "interval_kind": item["kind"],
                "lower": item["lower"],
                "upper": item["upper"],
                "width": item["upper"] - item["lower"],
                # Rows are printed only after this step's public advance call
                # returned success.  A later step can fail and make the process
                # return nonzero without invalidating these accepted prefixes.
                "native_validation_passed": True,
                "exact_reference_contained": exact_ok,
                "sampled_trajectory_contained": "",
                "validation_failed": exact_ok is False,
            }
        )
    maximum_allowed_degree = 1 if affine_carry else 2
    carried_degrees = [
        int(value)
        for value in re.findall(
            r"^MATCHED_CARRY_DEGREE \d+ \d+ (\d+)$",
            run.stdout or "",
            re.MULTILINE,
        )
    ]
    degree_ok = bool(carried_degrees) and max(carried_degrees) <= maximum_allowed_degree
    summary = {
        **metadata,
        "h": h,
        "horizon": horizon,
        "requested_steps": steps,
        "completed_steps": completed,
        "returncode": run.returncode,
        "native_validation_passed": run.returncode == 0 and completed == steps,
        "exact_reference_checks": exact_checks,
        "exact_reference_violations": exact_violations,
        "exact_reference_contained": exact_violations == 0 if exact_checks else None,
        "carry_degree_gate_passed": degree_ok,
        "source": str(source),
        "log": str(run_dir / "run.stdout.txt"),
    }
    return rows, summary


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=FIELDS, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    spec = load_spec(HERE / "benchmark_spec.yaml")
    configs = {
        "riccati": (0.01, 0.1 if args.smoke else 1.0),
        "harmonic": (0.01, 0.1 if args.smoke else 10.0),
        "van_der_pol": (0.005, 0.02 if args.smoke else 2.0),
    }
    all_rows = []
    summaries = []
    for protocol in ("matched_affine_carry", "complete_degree_two_reference"):
        for system_name, (h, horizon) in configs.items():
            rows, summary = run_configuration(
                spec=spec,
                system_name=system_name,
                h=h,
                horizon=horizon,
                output=output,
                protocol=protocol,
            )
            all_rows.extend(rows)
            summaries.append(summary)
            print(
                f"Flow* {protocol} {system_name}: "
                f"{summary['completed_steps']}/{summary['requested_steps']} steps",
                flush=True,
            )
    _write_csv(output / "flowstar_raw_results.csv", all_rows)
    (output / "flowstar_summary.json").write_text(
        json.dumps(
            {
                "flowstar_git_commit": git_sha(FLOWSTAR_ROOT),
                "summaries": summaries,
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
