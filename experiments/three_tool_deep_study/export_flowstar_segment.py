#!/usr/bin/env python3
"""Export an officially composed Flow* segment to the common representation."""
from __future__ import annotations

import argparse
import math
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

from common import (
    canonical_record,
    git_sha,
    load_spec,
    unavailable,
    write_json,
)

HERE = Path(__file__).resolve().parent
TERM_RE = re.compile(
    r"^FS_TERM kind=(?P<kind>tube|endpoint) state=(?P<state>\d+) "
    r"coefficient=(?P<coefficient>[-+0-9.eE]+) exponents=(?P<exponents>[0-9,]*)$"
)
BOX_RE = re.compile(
    r"^FS_BOX kind=(?P<kind>tube|endpoint) state=(?P<state>\d+) "
    r"lower=(?P<lower>[-+0-9.eE]+) upper=(?P<upper>[-+0-9.eE]+)$"
)
REMAINDER_RE = re.compile(
    r"^FS_REMAINDER kind=(?P<kind>tube|endpoint) state=(?P<state>\d+) "
    r"lower=(?P<lower>[-+0-9.eE]+) upper=(?P<upper>[-+0-9.eE]+)$"
)
DOMAIN_RE = re.compile(
    r"^FS_DOMAIN index=(?P<index>\d+) lower=(?P<lower>[-+0-9.eE]+) "
    r"upper=(?P<upper>[-+0-9.eE]+)$"
)
SAMPLE_RE = re.compile(
    r"^FS_SAMPLE kind=(?P<kind>tube|endpoint) sample=(?P<sample>\d+) "
    r"state=(?P<state>\d+) lower=(?P<lower>[-+0-9.eE]+) "
    r"upper=(?P<upper>[-+0-9.eE]+) point=(?P<point>[-+0-9.eE,]+)$"
)
ENDPOINT_PATH_RE = re.compile(
    r"^FS_ENDPOINT_PATH state=(?P<state>\d+) "
    r"collapsed_lower=(?P<collapsed_lower>[-+0-9.eE]+) "
    r"collapsed_upper=(?P<collapsed_upper>[-+0-9.eE]+) "
    r"native_lower=(?P<native_lower>[-+0-9.eE]+) "
    r"native_upper=(?P<native_upper>[-+0-9.eE]+) "
    r"repaired_lower=(?P<repaired_lower>[-+0-9.eE]+) "
    r"repaired_upper=(?P<repaired_upper>[-+0-9.eE]+) "
    r"padding_lower=(?P<padding_lower>[-+0-9.eE]+) "
    r"padding_upper=(?P<padding_upper>[-+0-9.eE]+)$"
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


def render_cpp(
    system: Mapping[str, Any],
    *,
    h: float,
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
    environment = {
        "flowstar_stock": "",
        "flowstar_full_picard_revalidated": (
            '  setenv("FLOWSTAR_AUDIT_REVALIDATE_REFINEMENT", "1", 1);'
        ),
        "flowstar_root_cause_patch": (
            '  setenv("FLOWSTAR_AUDIT_CACHE_LEAF_TRUNCATION", "1", 1);'
        ),
    }[variant]
    return f"""
#include "Continuous.h"
#include <cstdio>
#include <cstdlib>
#include <list>
#include <string>
#include <vector>
using namespace flowstar;
using namespace std;

static vector<Real> endpoint_powers(
    const Interval &time_domain, unsigned int order) {{
  Real h;
  time_domain.sup(h);
  vector<Real> powers(order + 1, 1);
  for(unsigned int i = 1; i <= order; ++i)
    powers[i] = powers[i-1] * h;
  return powers;
}}

static void print_terms(
    const char *kind, const TaylorModelVec<Real> &tmv) {{
  for(unsigned int state = 0; state < tmv.tms.size(); ++state) {{
    for(list<Term<Real> >::const_iterator term =
            tmv.tms[state].expansion.terms.begin();
        term != tmv.tms[state].expansion.terms.end(); ++term) {{
      Real coefficient;
      vector<unsigned int> degrees;
      term->getCoefficient(coefficient);
      term->getDegrees(degrees);
      printf("FS_TERM kind=%s state=%u coefficient=%.17g exponents=",
             kind, state, coefficient.toDouble());
      for(unsigned int i = 0; i < degrees.size(); ++i)
        printf("%s%u", i ? "," : "", degrees[i]);
      printf("\\n");
    }}
    printf(
        "FS_REMAINDER kind=%s state=%u lower=%.17g upper=%.17g\\n",
        kind, state, tmv.tms[state].remainder.inf(),
        tmv.tms[state].remainder.sup());
  }}
}}

static void print_box(
    const char *kind, const TaylorModelVec<Real> &tmv,
    const vector<Interval> &domain) {{
  vector<Interval> box;
  tmv.intEval(box, domain);
  for(unsigned int state = 0; state < box.size(); ++state)
    printf("FS_BOX kind=%s state=%u lower=%.17g upper=%.17g\\n",
           kind, state, box[state].inf(), box[state].sup());
}}

static void print_sample(
    const char *kind, unsigned int sample,
    const TaylorModelVec<Real> &tmv, const vector<Interval> &domain,
    bool endpoint) {{
  vector<Interval> point = domain;
  for(unsigned int i = 0; i < point.size(); ++i) {{
    double value;
    if(endpoint && i == 0)
      value = 0.0;
    else if(sample == 0)
      value = point[i].inf();
    else if(sample == 1)
      value = 0.5 * (point[i].inf() + point[i].sup());
    else
      value = point[i].sup();
    point[i] = Interval(value);
  }}
  vector<Interval> box;
  tmv.intEval(box, point);
  for(unsigned int state = 0; state < box.size(); ++state) {{
    printf("FS_SAMPLE kind=%s sample=%u state=%u lower=%.17g upper=%.17g point=",
           kind, sample, state, box[state].inf(), box[state].sup());
    for(unsigned int i = endpoint ? 1 : 0; i < point.size(); ++i)
      printf("%s%.17g", i > (endpoint ? 1u : 0u) ? "," : "", point[i].inf());
    printf("\\n");
  }}
}}

int main() {{
  intervalNumPrecision = {precision_bits};
  setenv("FLOWSTAR_AUDIT_TRACE", "1", 1);
  unsetenv("FLOWSTAR_AUDIT_DISABLE_REFINEMENT");
  unsetenv("FLOWSTAR_AUDIT_REVALIDATE_REFINEMENT");
  unsetenv("FLOWSTAR_AUDIT_CACHE_LEAF_TRUNCATION");
{environment}
  setenv("FLOWSTAR_AUDIT_STEP", "1", 1);
  Variables vars;
{declarations}
  ODE<Real> ode({{{quoted}}}, vars);
  Computational_Setting setting(vars);
  if(!setting.setFixedStepsize({_number(h)}, {order})) return 3;
  setting.setCutoffThreshold({_number(cutoff)});
  setting.setRemainderEstimation(
      vector<Interval>(vars.size(),
        Interval(-{_number(abs(candidate))}, {_number(abs(candidate))})));
  setting.printOff();
  vector<Interval> initial_box(vars.size());
{assignments}
  Flowpipe current(initial_box);
  Flowpipe next;
  vector<Constraint> invariant;
  clock_t begin = clock();
  int advanced = current.advance(
      next, ode.expressions, setting.tm_setting, invariant, setting.g_setting);
  clock_t end = clock();
  printf("FS_STATUS advanced=%d seconds=%.17g variant={variant}\\n",
         advanced, (double)(end - begin) / CLOCKS_PER_SEC);
  if(advanced != 1) return 4;

  TaylorModelVec<Real> composed;
  next.compose(composed, {order}, setting.tm_setting.cutoff_threshold);
  for(unsigned int i = 0; i < next.domain.size(); ++i)
    printf("FS_DOMAIN index=%u lower=%.17g upper=%.17g\\n",
           i, next.domain[i].inf(), next.domain[i].sup());
  print_terms("tube", composed);
  print_box("tube", composed, next.domain);

  TaylorModelVec<Real> endpoint;
  composed.evaluate_time(
      endpoint, endpoint_powers(next.domain[0], {order}));
  vector<Interval> endpoint_domain = next.domain;
  endpoint_domain[0] = Interval(0.0);
  vector<Interval> collapsed_endpoint_box;
  endpoint.intEval(collapsed_endpoint_box, endpoint_domain);
  vector<Interval> native_endpoint_domain = next.domain;
  Real accepted_step;
  next.domain[0].sup(accepted_step);
  native_endpoint_domain[0] = Interval(accepted_step);
  vector<Interval> native_endpoint_box;
  composed.intEval(native_endpoint_box, native_endpoint_domain);
  for(unsigned int state = 0; state < endpoint.tms.size(); ++state) {{
    double repaired_lower = collapsed_endpoint_box[state].inf();
    double repaired_upper = collapsed_endpoint_box[state].sup();
    if(native_endpoint_box[state].inf() < repaired_lower)
      repaired_lower = native_endpoint_box[state].inf();
    if(native_endpoint_box[state].sup() > repaired_upper)
      repaired_upper = native_endpoint_box[state].sup();
    double padding_lower =
        repaired_lower - collapsed_endpoint_box[state].inf();
    double padding_upper =
        repaired_upper - collapsed_endpoint_box[state].sup();
    endpoint.tms[state].remainder +=
        Interval(padding_lower, padding_upper);
    printf(
        "FS_ENDPOINT_PATH state=%u "
        "collapsed_lower=%.17g collapsed_upper=%.17g "
        "native_lower=%.17g native_upper=%.17g "
        "repaired_lower=%.17g repaired_upper=%.17g "
        "padding_lower=%.17g padding_upper=%.17g\\n",
        state, collapsed_endpoint_box[state].inf(),
        collapsed_endpoint_box[state].sup(),
        native_endpoint_box[state].inf(), native_endpoint_box[state].sup(),
        repaired_lower, repaired_upper, padding_lower, padding_upper);
  }}
  print_terms("endpoint", endpoint);
  print_box("endpoint", endpoint, endpoint_domain);
  for(unsigned int sample = 0; sample < 3; ++sample) {{
    print_sample("tube", sample, composed, next.domain, false);
    print_sample("endpoint", sample, endpoint, endpoint_domain, true);
  }}
  return 0;
}}
""".lstrip()


def _ensure_library(flowstar_root: Path, timeout: float) -> None:
    toolbox = flowstar_root / "flowstar-toolbox"
    library = toolbox / "libflowstar.a"
    headers = [toolbox / "Term.h", toolbox / "expression.h"]
    stale = not library.exists() or any(
        header.stat().st_mtime > library.stat().st_mtime for header in headers
    )
    if stale:
        subprocess.run(["make", "clean"], cwd=toolbox, check=True, timeout=timeout)
    subprocess.run(["make", "-j4"], cwd=toolbox, check=True, timeout=timeout)


def _compile(
    source: Path, executable: Path, flowstar_root: Path, timeout: float
) -> float:
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
    (source.parent / "compile.stdout.txt").write_text(
        process.stdout, encoding="utf-8"
    )
    (source.parent / "compile.stderr.txt").write_text(
        process.stderr, encoding="utf-8"
    )
    if process.returncode:
        raise RuntimeError(
            f"Flow* exporter compilation failed: {process.returncode}: "
            f"{process.stderr[-2000:]}"
        )
    return time.perf_counter() - started


def _parse(
    stdout: str,
    *,
    variant: str,
    system: str,
    system_definition: Mapping[str, Any],
    h: float,
    requested_order: int,
) -> dict[str, Any]:
    terms: dict[str, dict[int, list[dict[str, Any]]]] = {
        "tube": {},
        "endpoint": {},
    }
    remainders: dict[str, dict[int, list[float]]] = {"tube": {}, "endpoint": {}}
    boxes: dict[str, dict[int, list[float]]] = {"tube": {}, "endpoint": {}}
    domains: dict[int, list[float]] = {}
    samples: list[dict[str, Any]] = []
    endpoint_paths: dict[int, dict[str, float]] = {}
    traces: list[dict[str, str]] = []
    status: dict[str, str] = {}
    for line in stdout.splitlines():
        match = TERM_RE.match(line)
        if match:
            kind, state = match["kind"], int(match["state"])
            exponent_text = match["exponents"]
            exponents = [] if not exponent_text else list(map(int, exponent_text.split(",")))
            terms[kind].setdefault(state, []).append(
                {
                    "exponents": exponents,
                    "coefficient": float(match["coefficient"]),
                }
            )
            continue
        match = REMAINDER_RE.match(line)
        if match:
            remainders[match["kind"]][int(match["state"])] = [
                float(match["lower"]),
                float(match["upper"]),
            ]
            continue
        match = BOX_RE.match(line)
        if match:
            boxes[match["kind"]][int(match["state"])] = [
                float(match["lower"]),
                float(match["upper"]),
            ]
            continue
        match = DOMAIN_RE.match(line)
        if match:
            domains[int(match["index"])] = [
                float(match["lower"]),
                float(match["upper"]),
            ]
            continue
        match = SAMPLE_RE.match(line)
        if match:
            samples.append(
                {
                    "kind": match["kind"],
                    "sample": int(match["sample"]),
                    "state": int(match["state"]),
                    "point": list(map(float, match["point"].split(","))),
                    "total_interval": [
                        float(match["lower"]),
                        float(match["upper"]),
                    ],
                }
            )
            continue
        match = ENDPOINT_PATH_RE.match(line)
        if match:
            endpoint_paths[int(match["state"])] = {
                key: float(match[key])
                for key in (
                    "collapsed_lower",
                    "collapsed_upper",
                    "native_lower",
                    "native_upper",
                    "repaired_lower",
                    "repaired_upper",
                    "padding_lower",
                    "padding_upper",
                )
            }
            continue
        if line.startswith("FLOWSTAR_AUDIT "):
            fields: dict[str, str] = {}
            for token in line[len("FLOWSTAR_AUDIT ") :].split():
                if "=" in token:
                    key, value = token.split("=", 1)
                    fields[key] = value
            traces.append(fields)
            continue
        if line.startswith("FS_STATUS "):
            for token in line[len("FS_STATUS ") :].split():
                if "=" in token:
                    key, value = token.split("=", 1)
                    status[key] = value
    dimension = len(boxes["tube"])
    if status.get("advanced") != "1" or not dimension:
        raise RuntimeError(f"Flow* export failed: {status}")
    ordered_domains = [domains[index] for index in sorted(domains)]
    tube_states = []
    endpoint_states = []
    for state in range(dimension):
        tube_states.append(
            {
                "polynomial_terms": terms["tube"][state],
                "independent_interval_remainder": remainders["tube"][state],
                "native_structured_symbolic_remainder": unavailable(
                    "Flow* structured symbolic remainder is not losslessly "
                    "available as one per-state interval"
                ),
            }
        )
        # evaluate_time leaves a zero time exponent in the term dimension.
        endpoint_states.append(
            {
                "polynomial_terms": [
                    {
                        **term,
                        "exponents": term["exponents"][1:],
                    }
                    for term in terms["endpoint"][state]
                ],
                "independent_interval_remainder": remainders["endpoint"][state],
                "native_structured_symbolic_remainder": unavailable(
                    "Flow* structured symbolic remainder is not losslessly "
                    "available as one per-state interval"
                ),
            }
        )
    record = canonical_record(
        tool="flowstar",
        variant=variant,
        system=system,
        h=h,
        variable_names=["tau", *[f"xi_{index}" for index in range(dimension)]],
        variable_roles=["local_time", *["state_generator"] * dimension],
        domains=ordered_domains,
        states=tube_states,
        raw_endpoint=endpoint_states,
        raw_endpoint_box=[boxes["endpoint"][state] for state in range(dimension)],
        tube_box=[boxes["tube"][state] for state in range(dimension)],
        validation_trace=traces,
        reset_metadata={
            "reset": "Flowpipe_normalized_composition",
            "preconditioning": "native_diagonal_scaling_one_step",
            "composition": "Flowpipe::compose_official",
            "post_advance_remainder_mutation": False,
            "candidate_reinjection": False,
        },
        native_metadata={
            "status": "validated",
            "order": max(
                (
                    sum(term["exponents"])
                    for state_terms in terms["tube"].values()
                    for term in state_terms
                ),
                default=0,
            ),
            "advance_seconds": float(status.get("seconds", "nan")),
            "directed_rounding_or_mpfr": True,
            "floating_point_enclosure_candidate": False,
            "native_point_samples": samples,
            "endpoint_path_audit": [
                {"state": state, **endpoint_paths[state]}
                for state in sorted(endpoint_paths)
            ],
            "endpoint_path_semantics": (
                "raw endpoint is the hull of composed.evaluate_time and "
                "the composed native flowpipe evaluated on tau=[h,h]; "
                "the hull delta is explicit in the independent remainder"
            ),
        },
        system_definition={
            "name": system,
            "state_names": list(system_definition["state_names"]),
            "equations": system_definition["rhs"],
            "initial_domain": system_definition["initial_box"],
        },
        accepted_step=h,
        outcome={
            "status": "success",
            "category": "",
            "reason": "",
            "requested_horizon_reached": True,
        },
        execution_metadata={
            "backend": "flowstar_cpp_mpfr",
            "dtype": unavailable(
                "MPFR precision is attached after process execution"
            ),
            "device": "cpu",
            "repository_commit": unavailable(
                "Flow* repository SHA is attached after process execution"
            ),
            "runtime": {
                "setup_s": unavailable(
                    "build timing is attached after process execution"
                ),
                "propagation_s": float(status.get("seconds", "nan")),
                "export_s": unavailable(
                    "native print/export time is included in process execution"
                ),
            },
        },
        basis_metadata={
            "name": f"complete_total_degree_{requested_order}",
            "requested_order": requested_order,
            "native_order": requested_order,
            "coefficient_representation": (
                "Flow* sparse Polynomial<Real> terms"
            ),
        },
    )
    record["native_validation_passed"] = True
    if len(endpoint_paths) != dimension:
        raise RuntimeError(
            "Flow* exporter did not emit one endpoint-path audit per state"
        )
    for state, path in endpoint_paths.items():
        repaired = boxes["endpoint"][state]
        native = [path["native_lower"], path["native_upper"]]
        if (
            repaired[0] > native[0] + 1e-12
            or repaired[1] < native[1] - 1e-12
        ):
            raise RuntimeError(
                "repaired Flow* endpoint does not contain native fixed-domain "
                f"evaluation for state {state}: {repaired} vs {native}"
            )
    return record


def export_segment(
    spec: Mapping[str, Any],
    *,
    system_name: str,
    h: float,
    order: int,
    variant: str,
    work_dir: str | Path,
    precision_bits: int = 53,
) -> dict[str, Any]:
    if variant not in spec["flowstar"]["audit_variants"]:
        raise ValueError(f"unknown Flow* variant {variant}")
    flowstar_root = Path(spec["repositories"]["flowstar_audit"])
    timeout = float(spec["timeout_s"])
    _ensure_library(flowstar_root, timeout)
    run_dir = Path(work_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    source = run_dir / "export_segment.cpp"
    executable = run_dir / "export_segment"
    source.write_text(
        render_cpp(
            spec["systems"][system_name],
            h=h,
            order=order,
            candidate=float(spec["flowstar"]["candidate_remainder"][system_name]),
            cutoff=float(spec["flowstar"]["cutoff"]),
            variant=variant,
            precision_bits=precision_bits,
        ),
        encoding="utf-8",
    )
    compile_time = _compile(source, executable, flowstar_root, timeout)
    environment = os.environ.copy()
    started = time.perf_counter()
    process = subprocess.run(
        [str(executable)],
        cwd=run_dir,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )
    execution_time = time.perf_counter() - started
    (run_dir / "run.stdout.txt").write_text(process.stdout, encoding="utf-8")
    (run_dir / "run.stderr.txt").write_text(process.stderr, encoding="utf-8")
    if process.returncode:
        raise RuntimeError(
            f"Flow* exporter returned {process.returncode}; "
            f"stdout={process.stdout[-4000:]}; stderr={process.stderr[-2000:]}"
        )
    record = _parse(
        process.stdout,
        variant=variant,
        system=system_name,
        system_definition=spec["systems"][system_name],
        h=h,
        requested_order=order,
    )
    record["native_metadata"]["compile_time_s"] = compile_time
    record["native_metadata"]["execution_time_s"] = execution_time
    record["native_metadata"]["requested_order"] = order
    record["native_metadata"]["interval_precision_bits"] = precision_bits
    record["execution"]["dtype"] = f"MPFR_interval_{precision_bits}_bit"
    record["execution"]["repository_commit"] = git_sha(flowstar_root)
    record["execution"]["runtime"]["setup_s"] = compile_time
    record["execution"]["runtime"]["propagation_s"] = execution_time
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", default=str(HERE / "benchmark_spec.yaml"))
    parser.add_argument("--system", default="coupled_quadratic")
    parser.add_argument("--h", type=float, default=0.01)
    parser.add_argument("--order", type=int, default=2)
    parser.add_argument(
        "--variant",
        default="flowstar_root_cause_patch",
        choices=[
            "flowstar_stock",
            "flowstar_full_picard_revalidated",
            "flowstar_root_cause_patch",
        ],
    )
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--precision-bits", type=int, default=53)
    args = parser.parse_args()
    record = export_segment(
        load_spec(args.spec),
        system_name=args.system,
        h=args.h,
        order=args.order,
        variant=args.variant,
        work_dir=args.work_dir,
        precision_bits=args.precision_bits,
    )
    write_json(args.output, record)
    print(args.output)


if __name__ == "__main__":
    main()
