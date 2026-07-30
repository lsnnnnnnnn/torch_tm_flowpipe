#!/usr/bin/env python3
"""Generate and run the focused Flow* representation-layer audit."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
FLOWSTAR_ROOT = Path(
    os.environ.get("FLOWSTAR_ROOT", REPO_ROOT.parent / "flowstar")
).resolve()
ROW_RE = re.compile(
    r"^AUDIT_ROW (?P<step>\d+) (?P<time>[-+0-9.eE]+) (?P<path>[A-Za-z0-9_]+) "
    r"(?P<state>\d+) (?P<lo>[-+0-9.eE]+) (?P<hi>[-+0-9.eE]+)$"
)
DOMAIN_RE = re.compile(
    r"^AUDIT_DOMAIN (?P<layer>[A-Za-z0-9_]+) (?P<step>\d+) (?P<var>\d+) "
    r"(?P<lo>[-+0-9.eE]+) (?P<hi>[-+0-9.eE]+)$"
)


def _number(value: float) -> str:
    if not math.isfinite(value):
        raise ValueError(value)
    return f"{value:.17g}"


def render_cpp(
    *,
    h: float = 0.02,
    horizon: float = 0.1,
    order: int = 2,
    remainder_estimation: float = 1e-4,
    expression: str = "x^2",
) -> str:
    """Render the smallest completed failing baseline configuration."""
    return f"""
#include "Continuous.h"
#include <cstdio>
#include <list>
#include <vector>
using namespace flowstar;
using namespace std;

static void print_box(unsigned int step, double absolute_time, const char *path,
                      const vector<Interval> &box) {{
  for(unsigned int d = 0; d < box.size(); ++d) {{
    printf("AUDIT_ROW %u %.17g %s %u %.17g %.17g\\n",
           step, absolute_time, path, d, box[d].inf(), box[d].sup());
  }}
}}

static void print_domain(const char *layer, unsigned int step,
                         const vector<Interval> &domain) {{
  for(unsigned int i = 0; i < domain.size(); ++i) {{
    printf("AUDIT_DOMAIN %s %u %u %.17g %.17g\\n",
           layer, step, i, domain[i].inf(), domain[i].sup());
  }}
}}

static void print_tm_decomposition(unsigned int step, double absolute_time,
                                   const char *poly_name, const char *rem_name,
                                   const TaylorModelVec<Real> &tmv,
                                   const vector<Interval> &domain) {{
  for(unsigned int d = 0; d < tmv.tms.size(); ++d) {{
    Interval polynomial;
    tmv.tms[d].polyRange(polynomial, domain);
    vector<Interval> one(1, polynomial);
    print_box(step, absolute_time, poly_name, one);
    one[0] = tmv.tms[d].remainder;
    print_box(step, absolute_time, rem_name, one);
  }}
}}

static vector<Real> endpoint_powers(const Interval &time_domain, unsigned int order) {{
  Real h;
  time_domain.sup(h);
  vector<Real> powers;
  powers.push_back(1);
  powers.push_back(h);
  Real p = h;
  for(unsigned int k = 2; k <= order; ++k) {{
    p *= h;
    powers.push_back(p);
  }}
  return powers;
}}

static void eval_tm_paths(unsigned int step, double absolute_time,
                          const char *direct_name, const char *sub_name,
                          const TaylorModelVec<Real> &tmv,
                          const vector<Interval> &domain, unsigned int order) {{
  vector<Interval> endpoint_domain = domain;
  endpoint_domain[0] = domain[0].sup();
  vector<Interval> direct;
  tmv.intEval(direct, endpoint_domain);
  print_box(step, absolute_time, direct_name, direct);

  TaylorModelVec<Real> endpoint_tm;
  tmv.evaluate_time(endpoint_tm, endpoint_powers(domain[0], order));
  vector<Interval> substituted_domain = domain;
  substituted_domain[0] = Interval(0.0);
  vector<Interval> substituted;
  endpoint_tm.intEval(substituted, substituted_domain);
  print_box(step, absolute_time, sub_name, substituted);
}}

int main() {{
  Variables vars;
  vars.declareVar("x");
  ODE<Real> ode({{"{expression}"}}, vars);
  Computational_Setting setting(vars);
  const unsigned int order = {int(order)};
  if(!setting.setFixedStepsize({_number(h)}, order)) return 3;
  setting.setCutoffThreshold(1e-15);
  vector<Interval> estimates(
      1, Interval(-{_number(abs(remainder_estimation))},
                  {_number(abs(remainder_estimation))}));
  setting.setRemainderEstimation(estimates);
  setting.printOff();

  vector<Interval> initial_box(1, Interval(0.0, 0.1));
  Flowpipe initial_set(initial_box);
  Result_of_Reachability raw;
  vector<Constraint> safe;
  ode.reach(raw, initial_set, {_number(horizon)}, setting, safe);
  printf("AUDIT_COMPLETED %d\\n", raw.isCompleted() ? 1 : 0);
  printf("AUDIT_RAW_SEGMENTS %u\\n", (unsigned int)raw.flowpipes.size());

  Result_of_Reachability transformed = raw;
  unsigned int step = 0;
  double absolute_time = 0.0;
  for(list<Flowpipe>::const_iterator it = raw.flowpipes.begin();
      it != raw.flowpipes.end(); ++it) {{
    ++step;
    absolute_time += it->domain[0].sup();
    print_domain("raw", step, it->domain);

    vector<Interval> tmv_pre_range;
    it->tmvPre.intEval(tmv_pre_range, it->domain);
    print_box(step, absolute_time, "raw_tmvPre_domain_eval", tmv_pre_range);
    vector<Interval> tmv_range;
    it->tmv.intEval(tmv_range, it->domain);
    print_box(step, absolute_time, "raw_tmv_domain_eval", tmv_range);

    vector<Interval> official_int_eval;
    it->intEval(official_int_eval, order, setting.tm_setting.cutoff_threshold);
    print_box(step, absolute_time, "raw_Flowpipe_intEval_tube", official_int_eval);

    TaylorModelVec<Real> composed;
    it->compose(composed, order, setting.tm_setting.cutoff_threshold);
    vector<Interval> composed_tube;
    composed.intEval(composed_tube, it->domain);
    print_box(step, absolute_time, "raw_compose_tube", composed_tube);
    print_tm_decomposition(step, absolute_time, "raw_compose_polynomial_tube",
                           "raw_compose_remainder_tube", composed, it->domain);
    eval_tm_paths(step, absolute_time, "raw_compose_endpoint_direct",
                  "raw_compose_endpoint_substitute", composed, it->domain, order);
    vector<Interval> raw_endpoint_domain = it->domain;
    raw_endpoint_domain[0] = it->domain[0].sup();
    print_tm_decomposition(step, absolute_time, "raw_compose_polynomial_endpoint",
                           "raw_compose_remainder_endpoint", composed,
                           raw_endpoint_domain);

    TaylorModelVec<Real> composed_normal;
    it->compose_normal(composed_normal, setting.tm_setting.step_exp_table,
                       order, setting.tm_setting.cutoff_threshold);
    vector<Interval> normal_tube;
    composed_normal.intEval(normal_tube, it->domain);
    print_box(step, absolute_time, "raw_compose_normal_tube", normal_tube);
    eval_tm_paths(step, absolute_time, "raw_compose_normal_endpoint_direct",
                  "raw_compose_normal_endpoint_substitute",
                  composed_normal, it->domain, order);
  }}

  transformed.transformToTaylorModels(setting);
  printf("AUDIT_TRANSFORMED_SEGMENTS %u\\n",
         (unsigned int)transformed.tmv_flowpipes.size());
  step = 0;
  absolute_time = 0.0;
  for(list<TaylorModelFlowpipe>::const_iterator it =
          transformed.tmv_flowpipes.tmv_flowpipes.begin();
      it != transformed.tmv_flowpipes.tmv_flowpipes.end(); ++it) {{
    ++step;
    absolute_time += it->domain[0].sup();
    print_domain("transformed", step, it->domain);
    vector<Interval> tube;
    it->tmv_flowpipe.intEval(tube, it->domain);
    print_box(step, absolute_time, "transformed_tube", tube);
    print_tm_decomposition(step, absolute_time, "transformed_polynomial_tube",
                           "transformed_remainder_tube",
                           it->tmv_flowpipe, it->domain);
    eval_tm_paths(step, absolute_time, "transformed_endpoint_direct",
                  "transformed_endpoint_substitute",
                  it->tmv_flowpipe, it->domain, order);
    vector<Interval> transformed_endpoint_domain = it->domain;
    transformed_endpoint_domain[0] = it->domain[0].sup();
    print_tm_decomposition(step, absolute_time,
                           "transformed_polynomial_endpoint",
                           "transformed_remainder_endpoint",
                           it->tmv_flowpipe, transformed_endpoint_domain);
  }}

  // Flow* first proves that the configured candidate remainder maps into
  // itself, then replaces it with a refined image.  The toolbox accepts a
  // later refinement even when that image is no longer self-mapping.  This
  // focused path retains the already-proved candidate remainder instead.
  Flowpipe safe_current(initial_box);
  absolute_time = 0.0;
  const unsigned int requested_steps =
      (unsigned int)floor({_number(horizon)} / {_number(h)} + 0.5);
  for(step = 1; step <= requested_steps; ++step) {{
    Flowpipe safe_next;
    int advanced = safe_current.advance(
        safe_next, ode.expressions, setting.tm_setting, safe, setting.g_setting);
    printf("AUDIT_SAFE_ADVANCE %u %d\\n", step, advanced);
    if(advanced != 1) return 5;
    for(unsigned int d = 0; d < safe_next.tmvPre.tms.size(); ++d) {{
      safe_next.tmvPre.tms[d].remainder =
          setting.tm_setting.remainder_estimation[d];
    }}
    absolute_time += safe_next.domain[0].sup();
    TaylorModelVec<Real> safe_composed;
    safe_next.compose(
        safe_composed, order, setting.tm_setting.cutoff_threshold);
    vector<Interval> safe_tube;
    safe_composed.intEval(safe_tube, safe_next.domain);
    print_box(step, absolute_time, "safe_candidate_tube", safe_tube);
    eval_tm_paths(step, absolute_time, "safe_candidate_endpoint_direct",
                  "safe_candidate_endpoint_substitute",
                  safe_composed, safe_next.domain, order);
    safe_current = safe_next;
  }}
  return raw.isCompleted() ? 0 : 4;
}}
""".lstrip()


def parse_output(stdout: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    domains: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        match = ROW_RE.match(line)
        if match:
            rows.append(
                {
                    "step": int(match["step"]),
                    "time": float(match["time"]),
                    "path": match["path"],
                    "state": int(match["state"]),
                    "lower": float(match["lo"]),
                    "upper": float(match["hi"]),
                }
            )
            continue
        match = DOMAIN_RE.match(line)
        if match:
            domains.append(
                {
                    "layer": match["layer"],
                    "step": int(match["step"]),
                    "variable": int(match["var"]),
                    "lower": float(match["lo"]),
                    "upper": float(match["hi"]),
                }
            )
    return rows, domains


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]) if rows else [],
            lineterminator="\n",
        )
        if rows:
            writer.writeheader()
            writer.writerows(rows)


def analyze(
    rows: list[dict[str, Any]],
    *,
    h: float,
    tolerance: float = 1e-12,
) -> dict[str, Any]:
    by_key = {
        (row["path"], row["step"], row["state"]): row
        for row in rows
    }
    raw_violations = []
    safe_violations = []
    for step in sorted(
        row["step"] for row in rows
        if row["path"] == "safe_candidate_endpoint_direct"
    ):
        exact = 0.1 / (1.0 - 0.1 * (step * h))
        for path, target in (
            ("transformed_endpoint_direct", raw_violations),
            ("safe_candidate_endpoint_direct", safe_violations),
        ):
            row = by_key[(path, step, 0)]
            if row["lower"] > tolerance or row["upper"] < exact - tolerance:
                target.append(
                    {
                        "step": step,
                        "time": step * h,
                        "path": path,
                        "expected": [0.0, exact],
                        "exported": [row["lower"], row["upper"]],
                    }
                )

    pairs = (
        ("raw_compose_tube", "transformed_tube"),
        ("raw_compose_endpoint_direct", "transformed_endpoint_direct"),
        ("raw_compose_endpoint_direct", "raw_compose_endpoint_substitute"),
        ("transformed_endpoint_direct", "transformed_endpoint_substitute"),
    )
    agreement = []
    for left, right in pairs:
        deltas = []
        for step in range(1, 1 + max(row["step"] for row in rows)):
            left_row = by_key[(left, step, 0)]
            right_row = by_key[(right, step, 0)]
            deltas.extend(
                [
                    abs(left_row["lower"] - right_row["lower"]),
                    abs(left_row["upper"] - right_row["upper"]),
                ]
            )
        agreement.append(
            {
                "left": left,
                "right": right,
                "max_abs_delta": max(deltas),
            }
        )

    sample_violations = []
    samples_checked = 0
    steps = max(row["step"] for row in rows)
    for step in range(1, steps + 1):
        tube = by_key[("safe_candidate_tube", step, 0)]
        for sample in range(25):
            x0 = 0.1 * sample / 24.0
            for substep in range(4):
                absolute_time = ((step - 1) + substep / 3.0) * h
                exact = x0 / (1.0 - x0 * absolute_time)
                samples_checked += 1
                if exact < tube["lower"] - tolerance or exact > tube["upper"] + tolerance:
                    sample_violations.append(
                        {
                            "step": step,
                            "time": absolute_time,
                            "initial": x0,
                            "exact": exact,
                            "tube": [tube["lower"], tube["upper"]],
                        }
                    )
    return {
        "minimal_baseline_failure": {
            "h": h,
            "horizon": steps * h,
            "first_violating_step": raw_violations[0]["step"] if raw_violations else None,
            "first_violation": raw_violations[0] if raw_violations else None,
        },
        "unfixed_exact_endpoint_checks": steps,
        "unfixed_exact_endpoint_violations": len(raw_violations),
        "safe_exact_endpoint_checks": steps,
        "safe_exact_endpoint_violations": len(safe_violations),
        "safe_sample_checks": samples_checked,
        "safe_sample_violations": len(sample_violations),
        "sample_checks_are_formal_proof": False,
        "raw_transformed_agreement": agreement,
        "max_raw_transformed_delta": max(
            item["max_abs_delta"] for item in agreement[:2]
        ),
        "safe_candidate_gate_passed": (
            not safe_violations
            and not sample_violations
            and max(item["max_abs_delta"] for item in agreement[:2]) <= tolerance
        ),
        "unfixed_violations": raw_violations,
        "safe_violations": safe_violations,
        "sample_violations": sample_violations,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--h", type=float, default=0.02)
    parser.add_argument("--horizon", type=float, default=0.1)
    parser.add_argument("--order", type=int, default=2)
    parser.add_argument("--remainder-estimation", type=float, default=1e-4)
    parser.add_argument("--expression", default="x^2")
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = output / "flowstar_extraction_audit.cpp"
    executable = output / "flowstar_extraction_audit"
    source.write_text(
        render_cpp(
            h=args.h,
            horizon=args.horizon,
            order=args.order,
            remainder_estimation=args.remainder_estimation,
            expression=args.expression,
        ),
        encoding="utf-8",
    )
    compile_command = [
        "g++", "-O2", "-w", "-fpermissive", "-std=c++11",
        "-I", str(FLOWSTAR_ROOT / "flowstar-toolbox"),
        str(source),
        "-L", str(FLOWSTAR_ROOT / "flowstar-toolbox"),
        "-o", str(executable),
        "-lflowstar", "-lmpfr", "-lgmp", "-lgsl", "-lgslcblas", "-lm", "-lglpk",
    ]
    started = time.perf_counter()
    compiled = subprocess.run(compile_command, text=True, capture_output=True, check=False)
    compile_s = time.perf_counter() - started
    (output / "compile.stdout.txt").write_text(compiled.stdout, encoding="utf-8")
    (output / "compile.stderr.txt").write_text(compiled.stderr, encoding="utf-8")
    if compiled.returncode:
        raise SystemExit(f"Flow* audit compilation failed ({compiled.returncode})")
    started = time.perf_counter()
    run = subprocess.run([str(executable)], text=True, capture_output=True, check=False)
    runtime_s = time.perf_counter() - started
    (output / "run.stdout.txt").write_text(run.stdout, encoding="utf-8")
    (output / "run.stderr.txt").write_text(run.stderr, encoding="utf-8")
    rows, domains = parse_output(run.stdout)
    write_csv(output / "flowstar_extraction_rows.csv", rows)
    write_csv(output / "flowstar_domains.csv", domains)
    correctness = analyze(rows, h=args.h)
    (output / "flowstar_correctness.json").write_text(
        json.dumps(correctness, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = {
        "h": args.h,
        "horizon": args.horizon,
        "order": args.order,
        "remainder_estimation": args.remainder_estimation,
        "expression": args.expression,
        "compile_command": compile_command,
        "compile_time_s": compile_s,
        "runtime_s": runtime_s,
        "returncode": run.returncode,
        "rows": len(rows),
        "domains": len(domains),
        "correctness": correctness,
        "flowstar_root": str(FLOWSTAR_ROOT),
    }
    (output / "flowstar_extraction_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if run.returncode:
        raise SystemExit(f"Flow* audit failed ({run.returncode})")


if __name__ == "__main__":
    main()
