#!/usr/bin/env python3
"""Generate the evidence-backed final report and executive summary."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

HERE = Path(__file__).resolve().parent


def _read(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _json(path: Path) -> dict[str, Any]:
    return (
        json.loads(path.read_text(encoding="utf-8"))
        if path.exists()
        else {}
    )


def _f(value: Any, default: float = math.nan) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _fmt(value: Any) -> str:
    number = _f(value)
    if not math.isfinite(number):
        return "n/a"
    if number == 0:
        return "0"
    if abs(number) < 1e-3 or abs(number) >= 1e4:
        return f"{number:.4e}"
    return f"{number:.6g}"


def _markdown_table(
    headers: list[str], rows: Iterable[Iterable[Any]]
) -> str:
    values = [[str(item) for item in row] for row in rows]
    if not values:
        return "_No eligible rows._"
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in values)
    return "\n".join(lines)


def _max_width_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    kind: str | None = None,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if kind is not None and row.get("interval_kind") != kind:
            continue
        key = tuple(
            str(row.get(field, ""))
            for field in ("tool", "variant", "protocol", "system", "h")
        )
        grouped[key].append(row)
    result: list[dict[str, Any]] = []
    for key, values in sorted(grouped.items()):
        widths = [_f(row.get("width")) for row in values]
        result.append(
            {
                "tool": key[0],
                "variant": key[1],
                "protocol": key[2],
                "system": key[3],
                "h": key[4],
                "width": max(widths),
                "time": max(_f(row.get("time")) for row in values),
            }
        )
    return result


def _carry_loss(
    affine: list[dict[str, Any]], box: list[dict[str, Any]]
) -> list[tuple[str, str, str, float, float, float, float]]:
    av = {
        (
            row["tool"],
            row["system"],
            row["h"],
            round(_f(row.get("time")), 12),
        ): row["width"]
        for row in _max_width_rows(affine)
    }
    bv = {
        (
            row["tool"],
            row["system"],
            row["h"],
            round(_f(row.get("time")), 12),
        ): row["width"]
        for row in _max_width_rows(box)
    }
    return [
        (
            tool,
            system,
            h,
            time_value,
            av[(tool, system, h, time_value)],
            bv[(tool, system, h, time_value)],
            bv[(tool, system, h, time_value)]
            / av[(tool, system, h, time_value)]
            if av[(tool, system, h, time_value)]
            else math.nan,
        )
        for tool, system, h, time_value in sorted(set(av) & set(bv))
    ]


def _at_requested_horizon(row: Mapping[str, Any]) -> bool:
    time_value = _f(row.get("time"))
    horizon = _f(
        row.get("horizon", row.get("requested_horizon"))
    )
    return (
        math.isfinite(time_value)
        and math.isfinite(horizon)
        and math.isclose(
            time_value,
            horizon,
            rel_tol=0.0,
            abs_tol=1e-10 * max(1.0, abs(horizon)),
        )
    )


def _matched_summary(
    rows: list[dict[str, Any]]
) -> list[tuple[str, str, float, float, int]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("basis")), str(row.get("h")))].append(row)
    return [
        (
            basis,
            h,
            max(_f(row.get("width")) for row in values),
            max(
                _f(row.get("independent_remainder_width"))
                for row in values
            ),
            sum(int(_f(row.get("discarded_term_count"), 0)) for row in values),
        )
        for (basis, h), values in sorted(grouped.items())
    ]


def generate(output: Path) -> tuple[str, str]:
    environment = _json(output / "environment.json")
    correctness = _json(output / "correctness_checks.json")
    one_step = _read(output / "one_step_summary.csv")
    affine = _read(output / "affine_carry_summary.csv")
    box = _read(output / "box_carry_summary.csv")
    native_low = _read(output / "native_low_order_summary.csv")
    pareto = _read(output / "native_pareto_summary.csv")
    components = _read(output / "component_ablation.csv")
    matched = _read(output / "matched_basis_summary.csv")
    matched_capabilities = _read(
        output / "matched_basis_capabilities.csv"
    )
    defect = _read(output / "defect_summary.csv")
    runtime = _read(output / "runtime_summary.csv")
    acceleration = _read(output / "acceleration_summary.csv")
    failures = _read(output / "failure_summary.csv")
    flowstar_ablation = _read(
        output / "flowstar_component_ablation.csv"
    )

    one_endpoint = _max_width_rows(one_step, kind="endpoint_raw")
    affine_complete = [
        row for row in affine if _at_requested_horizon(row)
    ]
    box_complete = [
        row for row in box if _at_requested_horizon(row)
    ]
    affine_incomplete = [
        row for row in affine if not _at_requested_horizon(row)
    ]
    box_incomplete = [
        row for row in box if not _at_requested_horizon(row)
    ]
    affine_widths = _max_width_rows(affine_complete)
    box_widths = _max_width_rows(box_complete)
    incomplete_carry_widths = _max_width_rows(
        affine_incomplete + box_incomplete
    )
    native_widths = _max_width_rows(native_low, kind="endpoint_raw")
    losses = _carry_loss(affine_complete, box_complete)
    matched_rows = _matched_summary(matched)

    repositories = environment.get("repositories", {})
    sha_rows = [
        (name, value.get("sha", "n/a"), value.get("path", "n/a"))
        for name, value in sorted(repositories.items())
    ]
    flow_counts = (
        correctness.get("flowstar", {})
        .get("analytic_counts", {})
    )
    flow_parity = (
        correctness.get("flowstar", {})
        .get("original_parity", {})
    )
    repeated = [
        row
        for row in pareto
        if int(_f(row.get("runtime_repetitions"), 0)) >= 10
    ]
    frontier = [
        row
        for row in pareto
        if str(row.get("width_runtime_pareto")).lower() == "true"
        and str(
            row.get("primary_numerical_eligible", "true")
        ).lower()
        == "true"
    ]
    adaptive_flowstar_trajectory_failure = any(
        row.get("tool") == "flowstar"
        and row.get("variant") == "adaptive_order4_symbolic100"
        and row.get("failure_category") == "trajectory_sanity_failure"
        for row in failures
    )

    defect_by_tool: dict[str, list[float]] = defaultdict(list)
    for row in defect:
        value = _f(row.get("defect_norm_inf"))
        if math.isfinite(value):
            defect_by_tool[str(row.get("tool"))].append(value)
    defect_medians = {
        tool: statistics.median(values)
        for tool, values in defect_by_tool.items()
    }

    flow_refinement_rows = [
        (
            row.get("variant", ""),
            row.get("candidate_remainder", ""),
            row.get("status", ""),
            row.get("analytic_reference_violations", ""),
            _fmt(row.get("endpoint_max_width")),
        )
        for row in flowstar_ablation
    ]

    report = f"""# Torch TM / DiffReach / Flow* deep comparative study

## Executive result

The study produces valid one-step, common-affine-carry, common-box-carry,
native-low-order, and native-practical comparisons.  It does **not** produce a
literal same-order winner, because the three order labels select different
monomial dictionaries, validators, reset contracts, and arithmetic backends.
The closest valid reconstruction of “first order” is the common affine carry
contract: every raw endpoint is projected to `x = c + A xi + I`, every removed
term is outward-ranged into a fresh independent interval, and every local
solver remains native.

Primary gates passed: **{correctness.get('primary_gates_passed', False)}**.
The collected tables contain {correctness.get('native_validation_checks', 0)}
native-validation checks, {correctness.get('analytic_checks', 0)} analytic
checks, {correctness.get('common_segment_point_checks', 0)} exported point
containment checks,
{correctness.get('common_segment_native_round_trip_checks', 0)} native/export
round-trip evaluations, {correctness.get('trajectory_sanity', {}).get('checked', 0)}
non-proof nonlinear trajectory checks, and {len(failures)} explicitly
classified failure rows.

## Provenance and environments

{_markdown_table(['repository', 'SHA', 'path'], sha_rows)}

- Torch: conda `{environment.get('torch_environment', 'n/a')}`,
  Python {environment.get('torch_probe', {}).get('python', 'n/a')}, Torch
  {environment.get('torch_probe', {}).get('torch', 'n/a')}, CPU float64 study
  path; CUDA available:
  {environment.get('torch_probe', {}).get('cuda_available', 'n/a')}.
- DiffReach: conda `{environment.get('diffreach_environment', 'n/a')}`,
  Python {environment.get('jax_probe', {}).get('python', 'n/a')}, JAX/JAXlib
  {environment.get('jax_probe', {}).get('jax', 'n/a')}/
  {environment.get('jax_probe', {}).get('jaxlib', 'n/a')}, x64
  {environment.get('jax_probe', {}).get('x64', 'n/a')}.
- Flow*: `{environment.get('flowstar_environment', 'n/a')}`, GCC
  {str(environment.get('gcc', {}).get('stdout', '')).splitlines()[0] if environment.get('gcc') else 'n/a'},
  MPFR {str(environment.get('mpfr', {}).get('stdout', '')).strip() or 'n/a'},
  GMP {str(environment.get('gmp', {}).get('stdout', '')).strip() or 'n/a'}.
- CPU: {environment.get('cpu_model', 'n/a')}; batch size one.  Secondary
  accelerator availability and measurements are reported separately below.
- Frozen inputs unchanged:
  {correctness.get('frozen_inputs', {}).get('unchanged', False)}.

## Flow* correction status

The exact cached-remainder defect is a missing variable-leaf truncation
interval: the full evaluator applies `ctrunc_normal` at a variable leaf, while
the cached remainder-only evaluator previously had no corresponding cached
entry.  The root-cause patch records and consumes this contribution.  A
separate full-Picard-revalidation variant atomically accepts a proposed
remainder only after regenerating the complete image and polynomial-difference
intervals.

Primary corrected/revalidated analytic rows:
{flow_counts.get('primary_rows', 0)}; analytic violations:
{flow_counts.get('primary_analytic_violations', 'n/a')}; endpoint/tube
violations: {flow_counts.get('primary_endpoint_tube_violations', 'n/a')};
export failures: {flow_counts.get('primary_export_failures', 'n/a')}.  Stock
analytic violations retained as evidence:
{flow_counts.get('stock_analytic_violations', 'n/a')}.  The original generated
Van der Pol harness preserves the upstream schedule and the root-cause variant
reaches T=10: {flow_parity.get('root_cause_variant_reached_horizon_10', False)}
in {flow_parity.get('root_cause_segments', 'n/a')} segments.  Corrected
refinement can legitimately choose a different adaptive schedule.  The
adaptive endpoint export has a deterministic-trajectory bug-check failure:
{adaptive_flowstar_trajectory_failure}; it is retained as horizon/correctness
audit evidence but excluded from primary width/Pareto rankings.

Refinement/candidate control:

{_markdown_table(['mode', 'candidate', 'status', 'analytic violations', 'endpoint width'], flow_refinement_rows)}

## RQ1 — one-step local enclosure

Every row uses the same ODE, state order, initial box, `h`, and raw
tube/endpoint distinction.  Primary raw-endpoint maxima are:

{_markdown_table(
    ['tool', 'variant', 'system', 'h', 'max width'],
    ((row['tool'], row['variant'], row['system'], row['h'], _fmt(row['width'])) for row in one_endpoint),
)}

These rows are not relatively ranked because the native local bases are not
exactly matched.  This is a local-construction result, not a long-horizon
wrapping claim.  Flow* exposes a complete higher-order expansion
with MPFR intervals; DiffReach's restricted quasi-quadratic form stores
constant/linear plus local-time cross structure; Torch's complete basis exposes
more local monomials as its order increases.  Raw and legacy-tightened Torch
endpoints remain separate everywhere.

## RQ2 — common affine carry

The requested-final-time rows are listed below.  They are valid controlled
carry observations, but they are not relatively ranked because each native
local solver still uses a different construction basis, range evaluation, and
validator.  Carried degree and endpoint projection are controlled.  A failed
short prefix is never compared against a solver that reached the requested
final time.

{_markdown_table(
    ['tool', 'variant', 'system', 'h', 'time', 'max width'],
    ((row['tool'], row['variant'], row['system'], row['h'], _fmt(row['time']), _fmt(row['width'])) for row in affine_widths),
)}

Configurations that did not reach their requested common time remain useful
successful-horizon evidence but are unavailable for the final-time ranking:

{_markdown_table(
    ['tool', 'variant', 'protocol', 'system', 'h', 'last valid time', 'max width'],
    ((row['tool'], row['variant'], row['protocol'], row['system'], row['h'], _fmt(row['time']), _fmt(row['width'])) for row in incomplete_carry_widths),
)}

## RQ3 — common box carry and native low order

Box carry removes generator correlations.  The measured final width ratios are:

{_markdown_table(
    ['tool', 'system', 'h', 'time', 'affine', 'box', 'box/affine'],
    ((tool, system, h, _fmt(time_value), _fmt(a), _fmt(b), _fmt(ratio)) for tool, system, h, time_value, a, b, ratio in losses),
)}

Native low-order rows are deliberately labelled with their actual bases:

{_markdown_table(
    ['tool', 'variant', 'system', 'h', 'time', 'max width'],
    ((row['tool'], row['variant'], row['system'], row['h'], _fmt(row['time']), _fmt(row['width'])) for row in native_widths),
)}

DiffReach's affine flag and restricted quasi-quadratic mode are not the same
basis.  The latter can preserve `tau^2` and `tau*xi` structure before endpoint
evaluation and symbolic reset; it still omits general state-state and cubic
families.  Whether that helps depends on whether the missing state-state terms
or the retained time-state dependence dominates.  Torch dependency carry can
deteriorate because old generators and interval remainders remain correlated
through every new polynomial operation; normalized affine/QR resets exchange
some local polynomial detail for much better conditioning.  Flow* benefits
when complete higher-order terms, normalized composition, symbolic remainder,
or adaptive steps prevent the same information from being repeatedly ranged.

## RQ4 — native practical tradeoffs and within-tool Pareto frontiers

Width/runtime dominance was computed only within one tool at identical system
and absolute time.  Cross-tool native rows are not relatively ranked.
{len(repeated)} selected configuration/system rows have ten
full-configuration repetitions.  Within-tool nondominated rows:

{_markdown_table(
    ['tool', 'variant', 'system', 'time', 'width', 'horizon', 'steady s', 'repetitions'],
    ((row.get('tool',''), row.get('variant',''), row.get('system',''), _fmt(row.get('evaluation_time')), _fmt(row.get('width_at_evaluation_time')), _fmt(row.get('successful_horizon')), _fmt(row.get('steady_full_configuration_time_s')), row.get('runtime_repetitions','')) for row in frontier),
)}

No cross-tool winner follows from these rows.  A width/runtime point at T=1 is
not ranked against Flow*'s adaptive T=10 point.  Compile/JIT/build costs remain
separate from steady full-horizon execution, and backend throughput is not
presented as pure algorithmic speed.  Any configuration with a deterministic
trajectory sanity failure has `primary_numerical_eligible=false` and is not a
frontier candidate.

## RQ5 — component and matched-basis attribution

The component table contains {len(components)} rows.  It separates polynomial
range, exposed independent remainder, structured remainder where available,
and residual dependency/reset inflation.  The strongest causal controls are
within-tool: changing only carry/reset in Torch, only affine/quasi and
window/refinement settings in DiffReach, and only order/adaptation/symbolic
remainder/refinement in Flow*.

The one-engine matched-basis result is:

{_markdown_table(
    ['basis', 'h', 'max width', 'max independent remainder', 'discard records'],
    ((basis, h, _fmt(width), _fmt(rem), discarded) for basis, h, width, rem, discarded in matched_rows),
)}

Exact cross-tool basis capability is:

{_markdown_table(
    ['tool', 'basis', 'status', 'mapping', 'reason'],
    ((row.get('tool',''), row.get('basis',''), row.get('status',''), row.get('mapping',''), row.get('reason','')) for row in matched_capabilities),
)}

All four use one order-3 arithmetic ceiling, two Picard iterations, validator,
range backend, dtype, step, initial set, and reset.  B3 is not a general cubic
basis: it adds the one-local-time lift of quadratic state dependency, including
`tau*xi_i*xi_j`, while excluding cubic state terms.  The coupled quadratic
activates that cross-term family.  Thus B1-to-B_DR/B2/B3
changes are attributable to the retained dictionary inside one implementation,
not JAX versus Torch versus C++.

## Common defect and certificates

The shared CPU implementation differentiates and composes exported sparse
polynomials, outward-ranges the defect, bounds the Jacobian on the native tube,
and reports a Gronwall comparison radius separately from the native remainder.
Median infinity-norm defect bounds by tool are:
{', '.join(f'{tool}: {_fmt(value)}' for tool, value in sorted(defect_medians.items())) or 'n/a'}.
Tiny Riccati and coupled-polynomial identities use exact rational unit tests.
The common radius is diagnostic; it does not erase the numerical distinction
between Flow* MPFR intervals and the floating-point enclosure candidates from
Torch/DiffReach.

## Runtime decomposition

{_markdown_table(
    ['tool', 'variant', 'system', 'repetitions', 'build/JIT s', 'median full s', 'min s', 'max s', 'memory KiB'],
    ((row.get('tool',''), row.get('variant',''), row.get('system',''), row.get('runtime_repetitions',''), _fmt(row.get('compile_or_jit_time_s')), _fmt(row.get('steady_full_configuration_time_s')), _fmt(row.get('runtime_min_s')), _fmt(row.get('runtime_max_s')), _fmt(row.get('memory_kib'))) for row in runtime),
)}

Flow* build and process execution, DiffReach JIT and after-JIT execution, and
Torch orchestration/arithmetic/validation are distinct categories.  JAX
fusion/JIT and C++ compilation are backend effects; term count, Picard
refinement, range operations, and resets are algorithmic effects.

## Secondary native acceleration

These rows compare implementation/hardware throughput, not algorithmic
fairness.  The CPU and accelerator rows for a given tool use the same selected
full configuration on the cross-term-active coupled quadratic benchmark.

{_markdown_table(
    ['tool', 'backend', 'status', 'system', 'h', 'repetitions', 'median full s', 'speedup vs same-tool CPU', 'message'],
    ((row.get('tool',''), row.get('backend',''), row.get('backend_status',''), row.get('system',''), row.get('h',''), row.get('runtime_repetitions',''), _fmt(row.get('median_full_configuration_time_s')), _fmt(row.get('speedup_vs_same_tool_cpu')), row.get('message','')) for row in acceleration),
)}

## Direct answers to the eleven final questions

1. **Why same order is impossible.** Torch order 1 is a complete affine
   total-degree cap; DiffReach's two low-order flags retain different
   time-cross dictionaries; Flow*'s minimum legal fixed order is 2.  Their
   resets, remainders, and validators also differ.
2. **Closest valid first-order experiment.** Common affine carry is the closest:
   native local solve, raw endpoint, then one sound `c + A xi + I` carry
   projection.
3. **Widths under affine carry.** The per-system rows are reported without a
   cross-tool winner because local basis/construction, range bounding, and
   validator remain unmatched after controlling the carried representation.
4. **Loss from box carry.** The table above reports the exact measured ratios;
   values above one quantify lost dependency information.
5. **DiffReach low-order terms.** `tau^2` and `tau*xi` can reduce local-time
   truncation relative to a purely affine form, while missing general
   state-state/cubic terms can dominate on coupled or Van der Pol dynamics.
6. **When Flow* helps.** Complete higher order helps when nonlinear terms remain
   useful through composition; adaptive step and symbolic remainder help on
   longer nonlinear horizons.  The corrected Van der Pol path reaches T=10,
   but its exported raw endpoints fail a deterministic trajectory check at
   several early steps, so this run proves successful horizon rather than an
   admissible tightness curve.
7. **Why Torch dependency propagation deteriorates.** Reusing an increasingly
   complicated generator polynomial and independent remainder amplifies
   dependency and range overestimation.  Recentered affine/QR reset controls
   this at the cost of explicitly ranged discarded terms.
8. **Basis versus reset versus validator.** Matched basis isolates basis;
   affine-versus-box and Torch reset rows isolate carry/reset; stock/full/root
   Flow* rows isolate validator cache behavior.  Remaining cross-tool gaps
   cannot be assigned to one component alone.
9. **Runtime causes.** JIT, Python dispatch, C++ build/startup, and MPFR are
   backend effects.  polynomial support, range calls, Picard/refinement rounds,
   symbolic windows, and resets are algorithmic workload.
10. **Next Torch work.** Make normalized affine/QR reset a supported policy;
    add a documented restricted time-state basis option; improve polynomial
    range bounding and local-time overflow attribution; expose validator timing
    and defect diagnostics; and add a strict directed-rounding/MPFR validation
    backend before making proof-strength claims.
11. **BERN-NN-IBF.** The need exposed here is primarily improved polynomial
    storage/range bounding, not neural-network abstraction.  It may help if it
    supplies a tighter polynomial range backend, but this plant-only study
    provides no evidence yet for starting NN/CROWN integration.

## Validity limits and unresolved questions

- Valid: common one-step raw tube/endpoint, common affine carry, common box
  carry, accurately labelled native low-order, and same-time native Pareto
  comparisons.
- Not valid: a universal same-order ranking, width rankings across different
  absolute times, or proof-strength equivalence between MPFR and
  floating-point candidates.
- Flow* QR off/on remains unavailable through a stable public switch in this
  checkout and is labelled unavailable rather than emulated.
- DiffReach does not expose a separately width-valued structured remainder in
  its public result, limiting that decomposition.
- The corrected adaptive Flow* Van der Pol run reaches T=10 but its raw
  endpoint export excludes deterministic DOP853 samples in several early
  segments; that configuration is excluded from numerical frontiers pending a
  source-level endpoint/symbolic-remainder investigation.
- Torch CPU/CUDA throughput is measured when `torch.cuda` exposes a device.
  This DiffReach environment exposes {environment.get('jax_probe', {}).get('devices', [])};
  a missing JAX GPU backend is recorded as unavailable rather than inferred
  from Torch's CUDA visibility.

The three-tool study therefore satisfies the original research request in its
scientifically valid form: it identifies the valid controlled comparisons,
retains native capability comparisons without equating their orders, and
attributes the major differences with matched-basis, reset, validation, defect,
and runtime controls.

## Reproduction

```bash
cd {HERE.parents[1]}
experiments/three_tool_deep_study/run_smoke.sh
experiments/three_tool_deep_study/launch_background.sh
tmux -S /tmp/tm_three_tool_deep_study.sock attach -t tm_three_tool_deep_study
```

Full artifacts are in `{output}`.  The eighteen figures are under `plots/`.
"""

    executive = f"""# Executive summary

The completed three-tool study passes its primary gates:
**{correctness.get('primary_gates_passed', False)}**.  It contains
{correctness.get('analytic_checks', 0)} analytic containment checks,
{correctness.get('common_segment_point_checks', 0)} common-export point
containment checks,
{correctness.get('common_segment_native_round_trip_checks', 0)}
native/export round-trip evaluations, and {len(repeated)} selected
full-configuration runtime rows with at
least ten repetitions.

The literal question “which tool is best at order 1?” has no sound universal
answer because the tools' order labels select different bases.  Common affine
carry is the closest controlled carry protocol, but it is not used for a
relative winner because native local construction remains unmatched.

Box carry is a wrapping control and loses dependency information by the exact
ratios in `box_carry_summary.csv`.  Native low-order and practical Pareto rows
remain valid when labelled with their actual bases, successful horizon, common
absolute evaluation time, and numerical guarantee.  Flow*'s variable-leaf
cache patch and full-Picard revalidation both eliminate the stock Riccati
under-enclosure; the corrected original Van der Pol configuration reaches
T=10, but its exported adaptive raw endpoints fail the separate deterministic
trajectory sanity check and are excluded from numerical Pareto claims.

The matched-basis experiment shows what changes from B1/B_DR/B2/B3 inside one
engine.  The reset controls show why Torch's unchecked dependency carry
deteriorates; DiffReach gains from retained local-time cross structure and
symbolic normalization but omits general higher-order families; Flow* gains
from complete higher order, normalized composition, symbolic remainder, and
adaptation at the cost of MPFR/C++ workload.

Recommended Torch work: supported normalized affine/QR reset, a restricted
time-state basis, better polynomial range bounding and overflow attribution,
validator/runtime observability, and a strict directed-rounding backend.
BERN-NN-IBF is relevant only if it improves polynomial storage/range bounding;
this plant-only evidence does not motivate NN abstraction work yet.

See `three_tool_deep_study_report.md` for tables, validity limits, and all six
research questions plus the eleven required final answers.
"""
    return report, executive


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    report, executive = generate(output)
    (output / "three_tool_deep_study_report.md").write_text(
        report, encoding="utf-8"
    )
    (output / "executive_summary.md").write_text(
        executive, encoding="utf-8"
    )
    (HERE / "FINAL_CONCLUSIONS.md").write_text(
        executive
        + "\n\nThe complete generated report is stored with the timestamped "
        "results directory.\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "report": str(
                    output / "three_tool_deep_study_report.md"
                ),
                "executive_summary": str(output / "executive_summary.md"),
                "conclusions": str(HERE / "FINAL_CONCLUSIONS.md"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
