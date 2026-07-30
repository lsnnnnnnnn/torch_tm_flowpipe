#!/usr/bin/env python3
"""Run stock, reinjection, refinement, failure, and original-parity Flow* audits."""
from __future__ import annotations

import argparse
import csv
import math
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

from common import (
    PROTOCOL_BOX,
    PROTOCOL_NATIVE,
    PROTOCOL_RAW,
    PROTOCOL_SANITY,
    PROTOCOL_STRESS,
    PROTOCOL_TUBE,
    exact_steps,
    git_sha,
    load_spec,
    make_row,
    reference_for_row,
    write_csv,
    write_json,
)

HERE = Path(__file__).resolve().parent

ROW_RE = re.compile(
    r"^REPAIR_ROW step=(?P<step>\d+) time=(?P<time>[-+0-9.eE]+) "
    r"kind=(?P<kind>\w+) state=(?P<state>\d+) "
    r"lower=(?P<lower>[-+0-9.eE]+) upper=(?P<upper>[-+0-9.eE]+) "
    r"poly_width=(?P<poly>[-+0-9.eE]+) remainder_width=(?P<remainder>[-+0-9.eE]+) "
    r"native_remainder_width=(?P<native>[-+0-9.eE]+) "
    r"postprocessed_remainder_width=(?P<post>[-+0-9.eE]+)$"
)
STEP_RE = re.compile(
    r"^REPAIR_STEP step=(?P<step>\d+) code=(?P<code>-?\d+) "
    r"seconds=(?P<seconds>[-+0-9.eE]+)$"
)
ORDER_RE = re.compile(
    r"^REPAIR_ORDER requested=(?P<order>\d+) accepted=(?P<accepted>[01])$"
)
PARITY_STEP_RE = re.compile(
    r"^time =\s*(?P<time>[-+0-9.eE]+),\s*step =\s*(?P<step>[-+0-9.eE]+),"
    r"\s*order =\s*(?P<order>\d+)"
)
PARITY_ROW_RE = re.compile(
    r"^PARITY_ROW label=(?P<label>\w+) step=(?P<step>\d+) "
    r"time=(?P<time>[-+0-9.eE]+) state=(?P<state>\d+) "
    r"tube_lower=(?P<tlo>[-+0-9.eE]+) tube_upper=(?P<thi>[-+0-9.eE]+) "
    r"endpoint_lower=(?P<elo>[-+0-9.eE]+) endpoint_upper=(?P<ehi>[-+0-9.eE]+) "
    r"poly_width=(?P<poly>[-+0-9.eE]+) remainder_width=(?P<rem>[-+0-9.eE]+)$"
)


def _number(value: float) -> str:
    if not math.isfinite(value):
        raise ValueError(value)
    return f"{value:.17g}"


def _flowstar_expression(
    polynomial: Mapping[str, Any], state_names: list[str]
) -> str:
    pieces: list[str] = []
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
            factors.insert(0, _number(magnitude))
        body = "*".join(factors) if factors else "0"
        if not pieces:
            pieces.append(body if coefficient >= 0 else f"-{body}")
        else:
            pieces.append((" + " if coefficient >= 0 else " - ") + body)
    return "".join(pieces)


def render_fixed_cpp(
    system: Mapping[str, Any],
    *,
    protocol: str,
    h: float,
    horizon: float,
    order: int,
    candidate: float,
    cutoff: float,
    variant: str,
    precision_bits: int = 53,
) -> str:
    names = list(system["state_names"])
    expressions = [
        _flowstar_expression(polynomial, names) for polynomial in system["rhs"]
    ]
    declarations = "\n".join(
        f'  int state_{index}_id = vars.declareVar("{name}");'
        for index, name in enumerate(names)
    )
    assignments = "\n".join(
        f"  initial_box[state_{index}_id] = "
        f"Interval({_number(float(bounds[0]))}, {_number(float(bounds[1]))});"
        for index, bounds in enumerate(system["initial_box"])
    )
    quoted = ", ".join(f'"{value}"' for value in expressions)
    mutation = ""
    if variant == "flowstar_candidate_reinjection_diagnostic":
        mutation = """
    for(unsigned int state = 0; state < next.tmvPre.tms.size(); ++state)
      next.tmvPre.tms[state].remainder =
          setting.tm_setting.remainder_estimation[state];
"""
    if protocol == PROTOCOL_BOX:
        carry = "    current = Flowpipe(endpoint_box);"
    elif protocol in {PROTOCOL_NATIVE, PROTOCOL_STRESS}:
        carry = "    current = next;"
    else:
        carry = ""
    return f"""
#include "Continuous.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
using namespace flowstar;
using namespace std;

static vector<Real> endpoint_powers(const Interval &time_domain, unsigned int order) {{
  Real h;
  time_domain.sup(h);
  vector<Real> powers(order + 1, 1);
  for(unsigned int i = 1; i <= order; ++i) powers[i] = powers[i-1] * h;
  return powers;
}}

static void print_tmv(
    unsigned int step,
    double absolute_time,
    const char *kind,
    const TaylorModelVec<Real> &tmv,
    const vector<Interval> &domain,
    const vector<double> &native_widths,
    const vector<double> &post_widths) {{
  vector<Interval> box;
  tmv.intEval(box, domain);
  for(unsigned int state = 0; state < tmv.tms.size(); ++state) {{
    Interval polynomial_range;
    tmv.tms[state].polyRange(polynomial_range, domain);
    const Interval &remainder = tmv.tms[state].remainder;
    printf(
        "REPAIR_ROW step=%u time=%.17g kind=%s state=%u "
        "lower=%.17g upper=%.17g poly_width=%.17g remainder_width=%.17g "
        "native_remainder_width=%.17g postprocessed_remainder_width=%.17g\\n",
        step, absolute_time, kind, state, box[state].inf(), box[state].sup(),
        polynomial_range.sup() - polynomial_range.inf(),
        remainder.sup() - remainder.inf(), native_widths[state],
        post_widths[state]);
  }}
}}

int main() {{
  intervalNumPrecision = {precision_bits};
  Variables vars;
{declarations}
  ODE<Real> ode({{{quoted}}}, vars);
  Computational_Setting setting(vars);
  bool accepted = setting.setFixedStepsize({_number(h)}, {order});
  printf("REPAIR_ORDER requested={order} accepted=%d\\n", accepted ? 1 : 0);
  if(!accepted) return 3;
  setting.setCutoffThreshold({_number(cutoff)});
  vector<Interval> estimates(
      vars.size(), Interval(-{_number(abs(candidate))}, {_number(abs(candidate))}));
  setting.setRemainderEstimation(estimates);
  setting.printOff();
  vector<Constraint> invariant;
  vector<Interval> initial_box(vars.size());
{assignments}
  Flowpipe current(initial_box);
  unsigned int steps = (unsigned int)floor(
      {_number(horizon)} / {_number(h)} + 0.5);
  double absolute_time = 0.0;
  for(unsigned int step = 1; step <= steps; ++step) {{
    char step_buffer[32];
    snprintf(step_buffer, sizeof(step_buffer), "%u", step);
    setenv("FLOWSTAR_AUDIT_STEP", step_buffer, 1);
    clock_t begin = clock();
    Flowpipe next;
    int advanced = current.advance(
        next, ode.expressions, setting.tm_setting, invariant, setting.g_setting);
    clock_t end = clock();
    printf("REPAIR_STEP step=%u code=%d seconds=%.17g\\n", step, advanced,
           (double)(end - begin) / CLOCKS_PER_SEC);
    if(advanced != 1) break;
    vector<double> native_widths(next.tmvPre.tms.size(), 0.0);
    for(unsigned int state = 0; state < next.tmvPre.tms.size(); ++state)
      native_widths[state] =
          next.tmvPre.tms[state].remainder.sup() -
          next.tmvPre.tms[state].remainder.inf();
{mutation}
    vector<double> post_widths(next.tmvPre.tms.size(), 0.0);
    for(unsigned int state = 0; state < next.tmvPre.tms.size(); ++state)
      post_widths[state] =
          next.tmvPre.tms[state].remainder.sup() -
          next.tmvPre.tms[state].remainder.inf();
    absolute_time += next.domain[0].sup();
    TaylorModelVec<Real> composed;
    next.compose(composed, {order}, setting.tm_setting.cutoff_threshold);
    print_tmv(step, absolute_time, "tube", composed, next.domain,
              native_widths, post_widths);
    TaylorModelVec<Real> endpoint;
    composed.evaluate_time(endpoint, endpoint_powers(next.domain[0], {order}));
    vector<Interval> endpoint_domain = next.domain;
    endpoint_domain[0] = Interval(0.0);
    print_tmv(step, absolute_time, "endpoint_raw", endpoint, endpoint_domain,
              native_widths, post_widths);
    vector<Interval> endpoint_box;
    endpoint.intEval(endpoint_box, endpoint_domain);
{carry}
  }}
  return 0;
}}
""".lstrip()


def render_parity_cpp(label: str) -> str:
    return f"""
#include "Continuous.h"
#include <cstdio>
#include <list>
#include <vector>
using namespace flowstar;
using namespace std;

static vector<Real> endpoint_powers(const Interval &time_domain, unsigned int order) {{
  Real h;
  time_domain.sup(h);
  vector<Real> powers(order + 1, 1);
  for(unsigned int i = 1; i <= order; ++i) powers[i] = powers[i-1] * h;
  return powers;
}}

int main() {{
  Variables vars;
  int x_id = vars.declareVar("x");
  int y_id = vars.declareVar("y");
  int t_id = vars.declareVar("t");
  ODE<Real> ode({{"y", "(1 - x^2) * y - x", "1"}}, vars);
  Computational_Setting setting(vars);
  vector<Interval> box(vars.size());
  box[x_id] = Interval(1.1, 1.4);
  box[y_id] = Interval(2.35, 2.45);
  Flowpipe initial_set(box);
  vector<Constraint> safe_set = {{Constraint("y - 2.75", vars)}};
  Result_of_Reachability result;
  Symbolic_Remainder symbolic(initial_set, 100);
  ode.reach(result, initial_set, 10.0, setting, safe_set, symbolic);
  double absolute_time = 0.0;
  unsigned int step = 0;
  for(list<Flowpipe>::const_iterator it = result.flowpipes.begin();
      it != result.flowpipes.end(); ++it) {{
    ++step;
    absolute_time += it->domain[0].sup();
    TaylorModelVec<Real> composed;
    it->compose(composed, setting.tm_setting.order,
                setting.tm_setting.cutoff_threshold);
    vector<Interval> tube;
    composed.intEval(tube, it->domain);
    TaylorModelVec<Real> endpoint;
    composed.evaluate_time(
        endpoint, endpoint_powers(it->domain[0], setting.tm_setting.order));
    vector<Interval> endpoint_domain = it->domain;
    endpoint_domain[0] = Interval(0.0);
    vector<Interval> endpoint_box;
    endpoint.intEval(endpoint_box, endpoint_domain);
    vector<Interval> direct_domain = it->domain;
    Real accepted_step;
    it->domain[0].sup(accepted_step);
    direct_domain[0] = Interval(accepted_step);
    vector<Interval> direct_box;
    composed.intEval(direct_box, direct_domain);
    TaylorModelVec<Real> endpoint_pre;
    it->tmvPre.evaluate_time(
        endpoint_pre, endpoint_powers(it->domain[0], setting.tm_setting.order));
    vector<Interval> endpoint_tmv_poly_range;
    it->tmv.polyRange(endpoint_tmv_poly_range, endpoint_domain);
    TaylorModelVec<Real> native_endpoint;
    endpoint_pre.insert_ctrunc(
        native_endpoint, it->tmv, endpoint_tmv_poly_range, endpoint_domain,
        setting.tm_setting.order, setting.tm_setting.cutoff_threshold);
    vector<Interval> native_endpoint_box;
    native_endpoint.intEval(native_endpoint_box, endpoint_domain);
    for(unsigned int state = 0; state < 2; ++state) {{
      Interval polynomial_range;
      composed.tms[state].polyRange(polynomial_range, it->domain);
      double repaired_lower = endpoint_box[state].inf();
      double repaired_upper = endpoint_box[state].sup();
      if(direct_box[state].inf() < repaired_lower)
        repaired_lower = direct_box[state].inf();
      if(direct_box[state].sup() > repaired_upper)
        repaired_upper = direct_box[state].sup();
      printf(
          "PARITY_ROW label={label} step=%u time=%.17g state=%u "
          "tube_lower=%.17g tube_upper=%.17g "
          "endpoint_lower=%.17g endpoint_upper=%.17g "
          "poly_width=%.17g remainder_width=%.17g\\n",
          step, absolute_time, state, tube[state].inf(), tube[state].sup(),
          repaired_lower, repaired_upper,
          polynomial_range.sup() - polynomial_range.inf(),
          composed.tms[state].remainder.sup() -
              composed.tms[state].remainder.inf());
      printf(
          "PARITY_ENDPOINT_PATH label={label} step=%u time=%.17g state=%u "
          "export_lower=%.17g export_upper=%.17g "
          "direct_lower=%.17g direct_upper=%.17g "
          "native_lower=%.17g native_upper=%.17g "
          "repaired_lower=%.17g repaired_upper=%.17g "
          "padding_lower=%.17g padding_upper=%.17g\\n",
          step, absolute_time, state,
          endpoint_box[state].inf(), endpoint_box[state].sup(),
          direct_box[state].inf(), direct_box[state].sup(),
          native_endpoint_box[state].inf(), native_endpoint_box[state].sup(),
          repaired_lower, repaired_upper,
          repaired_lower - endpoint_box[state].inf(),
          repaired_upper - endpoint_box[state].sup());
    }}
  }}
  printf("PARITY_STATUS label={label} completed=%d segments=%u horizon=%.17g\\n",
         result.isCompleted() ? 1 : 0, step, absolute_time);
  return result.isCompleted() ? 0 : 2;
}}
""".lstrip()


def _compile(
    source: Path, executable: Path, flowstar_root: Path, timeout: float
) -> tuple[subprocess.CompletedProcess[str], float]:
    system_include = os.environ.get("FLOWSTAR_SYSTEM_INCLUDE", "/opt/homebrew/include")
    system_library = os.environ.get("FLOWSTAR_SYSTEM_LIB", "/opt/homebrew/lib")
    command = [
        "g++",
        "-O3",
        "-w",
        "-fpermissive",
        "-std=c++11",
        "-I",
        str(flowstar_root / "flowstar-toolbox"),
        "-I",
        system_include,
        str(source),
        "-L",
        str(flowstar_root / "flowstar-toolbox"),
        "-L",
        system_library,
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
        command,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )
    return process, time.perf_counter() - started


def _execute(
    executable: Path,
    *,
    cwd: Path,
    timeout: float,
    trace: bool = False,
    no_refinement: bool = False,
    revalidate_refinement: bool = False,
) -> tuple[subprocess.CompletedProcess[str], float]:
    environment = os.environ.copy()
    environment["FLOWSTAR_AUDIT_TRACE"] = "1" if trace else "0"
    environment["FLOWSTAR_AUDIT_DISABLE_REFINEMENT"] = (
        "1" if no_refinement else "0"
    )
    environment["FLOWSTAR_AUDIT_REVALIDATE_REFINEMENT"] = (
        "1" if revalidate_refinement else "0"
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
    return process, time.perf_counter() - started


def _parse_fixed(stdout: str) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    traces: list[dict[str, str]] = []
    order: dict[str, Any] | None = None
    for line in stdout.splitlines():
        match = ROW_RE.match(line)
        if match:
            rows.append(
                {
                    key: (
                        int(match[key])
                        if key in {"step", "state"}
                        else (
                            match[key]
                            if key == "kind"
                            else float(match[key])
                        )
                    )
                    for key in match.groupdict()
                }
            )
            continue
        match = STEP_RE.match(line)
        if match:
            steps.append(
                {
                    "step": int(match["step"]),
                    "code": int(match["code"]),
                    "seconds": float(match["seconds"]),
                }
            )
            continue
        match = ORDER_RE.match(line)
        if match:
            order = {
                "requested": int(match["order"]),
                "accepted": bool(int(match["accepted"])),
            }
            continue
        if line.startswith("FLOWSTAR_AUDIT "):
            fields: dict[str, str] = {}
            for token in line[len("FLOWSTAR_AUDIT ") :].split():
                if "=" in token:
                    key, value = token.split("=", 1)
                    fields[key] = value
            traces.append(fields)
    return {"rows": rows, "steps": steps, "traces": traces, "order": order}


def _failure_category(parsed: Mapping[str, Any]) -> tuple[str, str]:
    order = parsed.get("order")
    if order and not order["accepted"]:
        return "order_configuration_rejected", "setFixedStepsize rejected the order"
    for trace in reversed(parsed["traces"]):
        if trace.get("phase") == "advance_return":
            reason = trace.get("reason", "unknown_internal_failure")
            if reason == "first_picard_inclusion_failed":
                return reason, (
                    "Flowpipe::advance fixed-step/fixed-order first Picard image "
                    f"was not a subset; state={trace.get('failing_state', '?')} "
                    f"source_line={trace.get('source_line', '?')}"
                )
    return "unknown_internal_failure", "no structured Flow* return reason was emitted"


def run_fixed_case(
    spec: Mapping[str, Any],
    output: Path,
    *,
    system_name: str,
    protocol: str,
    h: float,
    horizon: float,
    order: int,
    candidate: float,
    cutoff: float,
    variant: str,
    no_refinement: bool = False,
    revalidate_refinement: bool = False,
    sensitivity_label: str = "",
    precision_bits: int = 53,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, str]]]:
    flowstar_root = Path(spec["repositories"]["flowstar_audit"])
    timeout = float(spec["timeout_s"])
    system = spec["systems"][system_name]
    tag = (
        f"{variant}_{protocol}_{system_name}_h{h:g}_T{horizon:g}_"
        f"o{order}_r{candidate:g}_c{cutoff:g}_p{precision_bits}"
    ).replace("+", "")
    run_dir = output / "logs" / "flowstar" / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    source = run_dir / "repair.cpp"
    executable = run_dir / "repair"
    source.write_text(
        render_fixed_cpp(
            system,
            protocol=protocol,
            h=h,
            horizon=horizon,
            order=order,
            candidate=candidate,
            cutoff=cutoff,
            variant=variant,
            precision_bits=precision_bits,
        ),
        encoding="utf-8",
    )
    compiled, build_time = _compile(source, executable, flowstar_root, timeout)
    (run_dir / "compile.stdout.txt").write_text(compiled.stdout, encoding="utf-8")
    (run_dir / "compile.stderr.txt").write_text(compiled.stderr, encoding="utf-8")
    if compiled.returncode:
        parsed = {"rows": [], "steps": [], "traces": [], "order": None}
        process = subprocess.CompletedProcess([], compiled.returncode, "", "")
        execution_time = 0.0
    else:
        process, execution_time = _execute(
            executable,
            cwd=run_dir,
            timeout=timeout,
            trace=True,
            no_refinement=no_refinement,
            revalidate_refinement=revalidate_refinement,
        )
        parsed = _parse_fixed(process.stdout)
    (run_dir / "run.stdout.txt").write_text(process.stdout or "", encoding="utf-8")
    (run_dir / "run.stderr.txt").write_text(process.stderr or "", encoding="utf-8")
    if protocol == PROTOCOL_TUBE:
        selected_kind = "tube"
    else:
        selected_kind = "endpoint_raw"
    if protocol == PROTOCOL_BOX:
        carried = "componentwise_box_from_raw_endpoint"
    elif protocol in {PROTOCOL_NATIVE, PROTOCOL_STRESS}:
        carried = "stock_returned_flowpipe"
    else:
        carried = "none_one_step"
    rows: list[dict[str, Any]] = []
    overwrite = variant == "flowstar_candidate_reinjection_diagnostic"
    for item in parsed["rows"]:
        if item["kind"] != selected_kind:
            continue
        absolute_time = float(item["time"])
        exact_boxes = reference_for_row(
            system_name,
            item["kind"],
            absolute_time,
            h,
            system["initial_box"],
        )
        exact = None if exact_boxes is None else exact_boxes[int(item["state"])]
        lower, upper = float(item["lower"]), float(item["upper"])
        contains = (
            True
            if exact is None
            else lower <= exact[0] and upper >= exact[1]
        )
        rows.append(
            make_row(
                tool="flowstar",
                variant=variant,
                protocol=protocol,
                system=system_name,
                h=h,
                horizon=horizon,
                step_index=int(item["step"]),
                absolute_time=absolute_time,
                state_index=int(item["state"]),
                interval_kind=item["kind"],
                lower=lower,
                upper=upper,
                exact=exact,
                native_validation_status="advance_returned_1",
                analytic_reference_status=(
                    "not_available"
                    if exact is None
                    else ("passed" if contains else "failed")
                ),
                local_order=order,
                local_basis=f"complete_total_degree_{order}",
                carried_representation=carried,
                step_policy=f"fixed_{h:.17g}",
                cutoff=cutoff,
                configured_candidate_remainder=candidate,
                native_returned_remainder=item["native"],
                postprocessed_remainder=item["post"],
                remainder_overwrite_applied=overwrite,
                endpoint_tightening_applied=False,
                endpoint_semantics=(
                    "whole_segment_tau_in_[0,h]"
                    if item["kind"] == "tube"
                    else "raw_composed_flowpipe_substitution_tau_equals_h"
                ),
                polynomial_width=item["poly"],
                remainder_width=item["remainder"],
                build_time_s=build_time,
                warmup_time_s=(
                    parsed["steps"][0]["seconds"] if parsed["steps"] else ""
                ),
                steady_runtime_s=(
                    parsed["steps"][int(item["step"]) - 1]["seconds"]
                    if len(parsed["steps"]) >= int(item["step"])
                    else ""
                ),
                dtype=f"MPFR_interval_{precision_bits}_bit",
                device="cpu",
                repository_sha=git_sha(flowstar_root),
                environment="system_g++_flowstar_audit",
            )
        )
    requested_steps = exact_steps(h, horizon)
    completed_steps = max(
        (int(item["step"]) for item in parsed["rows"] if item["kind"] == "endpoint_raw"),
        default=0,
    )
    failed = (
        compiled.returncode != 0
        or process.returncode != 0
        or completed_steps < requested_steps
    )
    category, failure_message = ("", "")
    if failed:
        if compiled.returncode:
            category, failure_message = (
                "wrapper_failure",
                f"generated C++ compilation returned {compiled.returncode}",
            )
        else:
            category, failure_message = _failure_category(parsed)
        failure_step = completed_steps + 1
        for state_index in range(len(system["state_names"])):
            rows.append(
                make_row(
                    tool="flowstar",
                    variant=variant,
                    protocol=protocol,
                    system=system_name,
                    h=h,
                    horizon=horizon,
                    step_index=failure_step,
                    absolute_time=failure_step * h,
                    state_index=state_index,
                    interval_kind="failure",
                    lower="",
                    upper="",
                    exact=None,
                    native_validation_status="failed",
                    analytic_reference_status="not_checked",
                    failure_category=category,
                    failure_message=failure_message,
                    local_order=order,
                    local_basis=f"complete_total_degree_{order}",
                    carried_representation=carried,
                    step_policy=f"fixed_{h:.17g}",
                    cutoff=cutoff,
                    configured_candidate_remainder=candidate,
                    remainder_overwrite_applied=overwrite,
                    endpoint_tightening_applied=False,
                    endpoint_semantics="not_available",
                    build_time_s=build_time,
                    dtype=f"MPFR_interval_{precision_bits}_bit",
                    device="cpu",
                    repository_sha=git_sha(flowstar_root),
                    environment="system_g++_flowstar_audit",
                )
            )
    run = {
        "tool": "flowstar",
        "variant": variant,
        "protocol": protocol,
        "system": system_name,
        "h": h,
        "requested_horizon": horizon,
        "order": order,
        "candidate_remainder": candidate,
        "cutoff": cutoff,
        "no_refinement": no_refinement,
        "revalidate_refinement": revalidate_refinement,
        "sensitivity_label": sensitivity_label,
        "interval_precision_bits": precision_bits,
        "requested_steps": requested_steps,
        "completed_steps": completed_steps,
        "status": "failed" if failed else "success",
        "failure_category": category,
        "failure_message": failure_message,
        "build_time_s": build_time,
        "execution_time_s": execution_time,
        "source": str(source),
        "stdout_log": str(run_dir / "run.stdout.txt"),
    }
    return rows, run, parsed["traces"]


def _parse_schedule(stdout: str) -> list[dict[str, Any]]:
    rows = []
    for line in stdout.splitlines():
        match = PARITY_STEP_RE.match(line)
        if match:
            rows.append(
                {
                    "time": float(match["time"]),
                    "step_size": float(match["step"]),
                    "order": int(match["order"]),
                }
            )
    return rows


def _parse_parity_rows(stdout: str) -> list[dict[str, Any]]:
    rows = []
    for line in stdout.splitlines():
        match = PARITY_ROW_RE.match(line)
        if match:
            rows.append(
                {
                    key: (
                        int(match[key])
                        if key in {"step", "state"}
                        else (
                            match[key]
                            if key == "label"
                            else float(match[key])
                        )
                    )
                    for key in match.groupdict()
                }
            )
    return rows


def run_original_parity(
    spec: Mapping[str, Any], output: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    flowstar_root = Path(spec["repositories"]["flowstar_audit"])
    timeout = float(spec["timeout_s"])
    parity_dir = output / "logs" / "flowstar_original_parity"
    parity_dir.mkdir(parents=True, exist_ok=True)
    original_executable = (
        flowstar_root / "benchmarks" / "continuous" / "vanderpol" / "vanderpol"
    )
    if not original_executable.exists():
        subprocess.run(
            ["make", "-j4"],
            cwd=original_executable.parent,
            check=True,
            timeout=timeout,
        )
    original_dir = parity_dir / "original"
    original_dir.mkdir(exist_ok=True)
    original, original_runtime = _execute(
        original_executable, cwd=original_dir, timeout=timeout
    )
    (original_dir / "stdout.txt").write_text(original.stdout, encoding="utf-8")
    (original_dir / "stderr.txt").write_text(original.stderr, encoding="utf-8")
    schedules: dict[str, list[dict[str, Any]]] = {
        "original": _parse_schedule(original.stdout)
    }
    extracted: dict[str, list[dict[str, Any]]] = {}
    build_times: dict[str, float] = {}
    runtimes: dict[str, float] = {"original": original_runtime}
    for label in ("generated_identical", "repaired_generic_identical"):
        run_dir = parity_dir / label
        run_dir.mkdir(exist_ok=True)
        source = run_dir / "parity.cpp"
        executable = run_dir / "parity"
        source.write_text(render_parity_cpp(label), encoding="utf-8")
        compiled, build_time = _compile(source, executable, flowstar_root, timeout)
        build_times[label] = build_time
        (run_dir / "compile.stdout.txt").write_text(compiled.stdout, encoding="utf-8")
        (run_dir / "compile.stderr.txt").write_text(compiled.stderr, encoding="utf-8")
        if compiled.returncode:
            process = subprocess.CompletedProcess([], compiled.returncode, "", compiled.stderr)
            runtime = 0.0
        else:
            process, runtime = _execute(executable, cwd=run_dir, timeout=timeout)
        runtimes[label] = runtime
        (run_dir / "stdout.txt").write_text(process.stdout or "", encoding="utf-8")
        (run_dir / "stderr.txt").write_text(process.stderr or "", encoding="utf-8")
        schedules[label] = _parse_schedule(process.stdout or "")
        extracted[label] = _parse_parity_rows(process.stdout or "")
    original_schedule = schedules["original"]
    schedule_agreement = all(
        len(schedules[label]) == len(original_schedule)
        and all(
            abs(left["time"] - right["time"]) <= 5e-7
            and abs(left["step_size"] - right["step_size"]) <= 5e-7
            and left["order"] == right["order"]
            for left, right in zip(original_schedule, schedules[label])
        )
        for label in ("generated_identical", "repaired_generic_identical")
    )
    generated = extracted.get("generated_identical", [])
    generic = extracted.get("repaired_generic_identical", [])
    bound_agreement = (
        len(generated) == len(generic)
        and all(
            left["step"] == right["step"]
            and left["state"] == right["state"]
            and max(
                abs(left[key] - right[key])
                for key in ("time", "tlo", "thi", "elo", "ehi", "poly", "rem")
            )
            <= 1e-14
            for left, right in zip(generated, generic)
        )
    )
    original_reached = bool(
        original.returncode == 0
        and original_schedule
        and abs(original_schedule[-1]["time"] - 10.0) <= 5e-7
    )
    rows: list[dict[str, Any]] = []
    for label, schedule in schedules.items():
        for index, item in enumerate(schedule, start=1):
            rows.append(
                {
                    "implementation": label,
                    "step_index": index,
                    "absolute_time": item["time"],
                    "step_size": item["step_size"],
                    "order": item["order"],
                    "completed": original_reached if label == "original" else True,
                }
            )
    summary = {
        "original_sha": git_sha(Path(spec["repositories"]["flowstar_original"])),
        "audit_sha": git_sha(flowstar_root),
        "original_reached_horizon_10": original_reached,
        "original_segments": len(original_schedule),
        "generated_segments": len(schedules["generated_identical"]),
        "generic_segments": len(schedules["repaired_generic_identical"]),
        "schedule_agreement": schedule_agreement,
        "generated_vs_generic_bound_agreement": bound_agreement,
        "passed": original_reached and schedule_agreement and bound_agreement,
        "runtimes_s": runtimes,
        "build_times_s": build_times,
    }
    return rows, summary


def _primary_cases(spec: Mapping[str, Any], smoke: bool):
    flow = spec["flowstar"]
    for system_name, benchmark in spec["benchmarks"].items():
        candidate = float(flow["candidate_remainder"][system_name])
        one_steps = [float(benchmark["smoke"]["h"])] if smoke else [
            float(value) for value in benchmark["one_step_h"]
        ]
        for h in one_steps:
            for protocol in (PROTOCOL_TUBE, PROTOCOL_RAW):
                yield system_name, protocol, h, h, "flowstar_stock", False
                yield (
                    system_name,
                    protocol,
                    h,
                    h,
                    "flowstar_candidate_reinjection_diagnostic",
                    False,
                )
        multi = [benchmark["smoke"]] if smoke else benchmark["multi_step"]
        for config in multi:
            h, horizon = float(config["h"]), float(config["horizon"])
            yield system_name, PROTOCOL_BOX, h, horizon, "flowstar_stock", False
            yield (
                system_name,
                PROTOCOL_BOX,
                h,
                horizon,
                "flowstar_candidate_reinjection_diagnostic",
                False,
            )
            yield system_name, PROTOCOL_NATIVE, h, horizon, "flowstar_stock", False
            yield system_name, PROTOCOL_STRESS, h, horizon, "flowstar_stock", False
        # Clearly labeled no-refinement diagnostic: one representative local step.
        yield (
            system_name,
            PROTOCOL_RAW,
            float(benchmark["smoke"]["h"]),
            float(benchmark["smoke"]["h"]),
            "flowstar_no_refinement_diagnostic",
            True,
        )
        if system_name == "riccati":
            yield (
                system_name,
                PROTOCOL_RAW,
                float(benchmark["smoke"]["h"]),
                float(benchmark["smoke"]["h"]),
                "flowstar_refinement_revalidated_diagnostic",
                False,
            )
    if not smoke:
        yield "riccati", PROTOCOL_BOX, 0.01, 0.1, "flowstar_stock", False
        yield (
            "riccati",
            PROTOCOL_BOX,
            0.01,
            0.1,
            "flowstar_candidate_reinjection_diagnostic",
            False,
        )
        yield "harmonic", PROTOCOL_BOX, 0.01, 1.0, "flowstar_stock", False


def _sensitivity_cases(spec: Mapping[str, Any]):
    flow = spec["flowstar"]
    base_order = int(flow["stock_order"])
    base_h = 0.01
    base_candidate = float(flow["candidate_remainder"]["riccati"])
    base_cutoff = float(flow["cutoff"])
    for order in flow["sensitivity"]["orders"]:
        yield f"order={order}", int(order), base_h, base_candidate, base_cutoff, False, 53
    for h in flow["sensitivity"]["step_sizes"]:
        yield f"h={h}", base_order, float(h), base_candidate, base_cutoff, False, 53
    for candidate in flow["sensitivity"]["candidate_remainders"]:
        yield (
            f"candidate={candidate}",
            base_order,
            base_h,
            float(candidate),
            base_cutoff,
            False,
            53,
        )
    for cutoff in flow["sensitivity"]["cutoffs"]:
        yield (
            f"cutoff={cutoff}",
            base_order,
            base_h,
            base_candidate,
            float(cutoff),
            False,
            53,
        )
    yield "refinement=disabled", base_order, base_h, base_candidate, base_cutoff, True, 53
    yield "refinement=native", base_order, base_h, base_candidate, base_cutoff, False, 53
    yield "precision=256", base_order, base_h, base_candidate, base_cutoff, False, 256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", default=str(HERE / "benchmark_spec.yaml"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    spec = load_spec(args.spec)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    flow = spec["flowstar"]
    rows: list[dict[str, Any]] = []
    runs: list[dict[str, Any]] = []
    traces: list[dict[str, str]] = []
    for system, protocol, h, horizon, variant, no_refinement in _primary_cases(
        spec, args.smoke
    ):
        case_rows, run, case_traces = run_fixed_case(
            spec,
            output,
            system_name=system,
            protocol=protocol,
            h=h,
            horizon=horizon,
            order=int(flow["stock_order"]),
            candidate=float(flow["candidate_remainder"][system]),
            cutoff=float(flow["cutoff"]),
            variant=variant,
            no_refinement=no_refinement,
            revalidate_refinement=(
                variant == "flowstar_refinement_revalidated_diagnostic"
            ),
        )
        rows.extend(case_rows)
        runs.append(run)
        for trace in case_traces:
            traces.append(
                {
                    "run_variant": variant,
                    "protocol": protocol,
                    "system": system,
                    "h": h,
                    "requested_horizon": horizon,
                    **trace,
                }
            )
        print(
            f"Flow* {variant} {protocol} {system} h={h:g} T={horizon:g}: "
            f"{run['status']} ({run['completed_steps']}/{run['requested_steps']})",
            flush=True,
        )
    sensitivity_runs: list[dict[str, Any]] = []
    if not args.smoke:
        for (
            label,
            order,
            h,
            candidate,
            cutoff,
            no_refinement,
            precision_bits,
        ) in _sensitivity_cases(
            spec
        ):
            _, run, case_traces = run_fixed_case(
                spec,
                output,
                system_name="riccati",
                protocol=PROTOCOL_NATIVE,
                h=h,
                horizon=1.0,
                order=order,
                candidate=candidate,
                cutoff=cutoff,
                variant=(
                    "flowstar_no_refinement_diagnostic"
                    if no_refinement
                    else "flowstar_stock"
                ),
                no_refinement=no_refinement,
                sensitivity_label=label,
                precision_bits=precision_bits,
            )
            sensitivity_runs.append(run)
            for trace in case_traces:
                traces.append(
                    {
                        "run_variant": run["variant"],
                        "protocol": run["protocol"],
                        "system": run["system"],
                        "h": run["h"],
                        "requested_horizon": run["requested_horizon"],
                        "sensitivity_label": label,
                        **trace,
                    }
                )
    parity_rows, parity_summary = run_original_parity(spec, output)
    # Sanity marker in the unified schema. Bounds are retained in the dedicated
    # parity trace because its adaptive segments do not share fixed h.
    rows.append(
        make_row(
            tool="flowstar",
            variant="flowstar_original_benchmark_configuration",
            protocol=PROTOCOL_SANITY,
            system="van_der_pol",
            h="",
            horizon=10.0,
            step_index=len(
                [
                    row
                    for row in parity_rows
                    if row["implementation"] == "original"
                ]
            ),
            absolute_time=10.0,
            state_index=0,
            interval_kind="sanity_status",
            lower="",
            upper="",
            exact=None,
            native_validation_status=(
                "passed" if parity_summary["passed"] else "failed"
            ),
            analytic_reference_status="not_available",
            failure_category=(
                "" if parity_summary["passed"] else "unknown_internal_failure"
            ),
            failure_message=(
                "" if parity_summary["passed"] else "original benchmark parity failed"
            ),
            local_order=4,
            local_basis="complete_total_degree_4",
            carried_representation="stock_symbolic_remainder_flowpipe",
            step_policy="adaptive_0.002_to_0.1",
            cutoff=1e-10,
            configured_candidate_remainder=1e-4,
            remainder_overwrite_applied=False,
            endpoint_tightening_applied=False,
            endpoint_semantics="stock_original_benchmark",
            dtype="MPFR_interval",
            device="cpu",
            repository_sha=git_sha(Path(spec["repositories"]["flowstar_audit"])),
            environment="system_g++_flowstar_audit",
        )
    )
    write_csv(output / "flowstar_audit.csv", rows)
    write_json(output / "flowstar_runs.json", runs)
    write_json(output / "flowstar_original_parity_summary.json", parity_summary)
    with (output / "flowstar_refinement_trace.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fields = sorted({key for row in traces for key in row})
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(traces)
    with (output / "flowstar_parameter_sensitivity.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fields = sorted({key for row in sensitivity_runs for key in row}) or [
            "sensitivity_label"
        ]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sensitivity_runs)
    with (output / "flowstar_original_parity.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fields = list(parity_rows[0]) if parity_rows else ["implementation"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(parity_rows)


if __name__ == "__main__":
    main()
