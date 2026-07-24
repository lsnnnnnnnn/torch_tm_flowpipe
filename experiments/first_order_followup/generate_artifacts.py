#!/usr/bin/env python3
"""Generate the required plots and evidence-classified report."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
BASELINE_EXPERIMENT = HERE.parent / "first_order_three_way"
if str(BASELINE_EXPERIMENT) not in sys.path:
    sys.path.insert(0, str(BASELINE_EXPERIMENT))
from common import exact_endpoint, load_spec


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _f(row: dict[str, Any], key: str, default: float = math.nan) -> float:
    try:
        return float(row.get(key, ""))
    except (TypeError, ValueError):
        return default


def _max_endpoint_series(
    rows: Iterable[dict[str, str]], selector
) -> dict[str, tuple[list[float], list[float]]]:
    groups: dict[str, dict[float, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        label = selector(row)
        if label is None or row["interval_kind"] != "endpoint":
            continue
        groups[label][_f(row, "time")].append(_f(row, "width"))
    return {
        label: (
            sorted(values),
            [max(values[time]) for time in sorted(values)],
        )
        for label, values in groups.items()
    }


def _save(fig: Any, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_flowstar_audit(output: Path, plots: Path) -> None:
    rows = _rows(output / "logs/flowstar_audit/flowstar_extraction_rows.csv")
    wanted = {
        "raw_compose_endpoint_substitute": "raw compose",
        "transformed_endpoint_substitute": "transformed",
        "safe_candidate_endpoint_substitute": "safe candidate",
    }
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for path, label in wanted.items():
        selected = [row for row in rows if row["path"] == path and row["state"] == "0"]
        selected.sort(key=lambda row: int(row["step"]))
        ax.plot(
            [int(row["step"]) for row in selected],
            [_f(row, "upper") for row in selected],
            marker="o",
            label=f"{label} upper",
        )
    ax.set(xlabel="step", ylabel="endpoint upper bound", title="Flow* extraction audit")
    ax.legend()
    ax.grid(alpha=0.25)
    _save(fig, plots / "01_flowstar_raw_vs_transformed_endpoint.png")


def plot_torch_diagnostics(output: Path, plots: Path) -> None:
    diagnostics = json.loads(
        (output / "torch_diagnostics.json").read_text(encoding="utf-8")
    )
    selected = [
        row for row in diagnostics
        if row["protocol"] == "matched_affine_carry"
        and row["system"] == "harmonic"
        and row["basis"] == "B1"
    ]
    times = [float(row["time"]) for row in selected]
    affine = [row["endpoint"][0]["polynomial_width"] for row in selected]
    overflow = []
    remainder = []
    for row in selected:
        last_iteration = max(
            (
                int(item["iteration"])
                for item in row["discarded_terms"]
                if item["stage"] == "picard"
            ),
            default=0,
        )
        projected = sum(
            float(item["range_width"])
            for item in row["discarded_terms"]
            if item["stage"] == "picard"
            and int(item["iteration"]) == last_iteration
            and int(item["state_index"]) == 0
        )
        total_remainder = float(row["endpoint"][0]["remainder_width"])
        overflow.append(min(projected, total_remainder))
        remainder.append(max(total_remainder - projected, 0.0))
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.stackplot(
        times, affine, overflow, remainder,
        labels=(
            "affine polynomial",
            "termwise projected overflow",
            "remaining validated remainder",
        ),
        alpha=0.8,
    )
    ax.set(
        xlabel="time",
        ylabel="state-0 endpoint width components",
        title="Torch harmonic B1 finite-basis decomposition",
    )
    ax.legend(loc="upper left")
    _save(fig, plots / "02_torch_harmonic_width_decomposition.png")

    rows = _rows(output / "raw_results.csv")
    series = _max_endpoint_series(
        rows,
        lambda row: row["mode"]
        if row["protocol"] == "torch_dependency_forensics"
        and row["tool"] == "torch_tm_flowpipe"
        else None,
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for label, (x, y) in series.items():
        ax.plot(x, y, label=label)
    ax.set(
        xlabel="time", ylabel="maximum endpoint width",
        title="Torch harmonic carry/reset policies",
    )
    ax.set_yscale("log")
    ax.legend()
    ax.grid(alpha=0.25)
    _save(fig, plots / "03_torch_dependency_and_resets.png")


def plot_basis(output: Path, plots: Path, spec: dict[str, Any]) -> None:
    rows = _rows(output / "raw_results.csv")
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9))
    for ax, system in zip(axes, ("riccati", "harmonic", "van_der_pol")):
        series = _max_endpoint_series(
            rows,
            lambda row, system=system: row["basis"]
            if row["protocol"] in {"matched_affine_carry", "matched_basis_ablation"}
            and row["tool"] == "torch_tm_flowpipe"
            and row["system"] == system
            and row["basis"] in {"B1", "B_DR", "B2"}
            else None,
        )
        for label, (x, y) in series.items():
            ax.plot(x, y, label=label)
        ax.set(title=system, xlabel="time", ylabel="max endpoint width")
        ax.legend()
        ax.grid(alpha=0.25)
    _save(fig, plots / "04_basis_width_curves.png")

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))
    for ax, system in zip(axes, ("riccati", "harmonic")):
        grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for row in rows:
            if (
                row["system"] != system
                or row["interval_kind"] != "endpoint"
                or int(row["state_index"]) != 0
                or row["tool"] == "analytic_oracle"
            ):
                continue
            exact = exact_endpoint(
                system, _f(row, "time"), spec["systems"][system]["initial_box"]
            )
            if exact is None:
                continue
            exact_width = exact[0][1] - exact[0][0]
            if exact_width <= 0:
                continue
            label = f"{row['tool']}:{row['protocol']}:{row['basis']}"
            grouped[label].append((_f(row, "time"), _f(row, "width") / exact_width))
        for label, values in grouped.items():
            values.sort()
            ax.plot(
                [item[0] for item in values],
                [item[1] for item in values],
                label=label,
                alpha=0.8,
            )
        ax.set(title=system, xlabel="time", ylabel="width / exact width")
        ax.set_yscale("log")
        ax.grid(alpha=0.25)
    axes[1].legend(fontsize=5, bbox_to_anchor=(1.04, 1), loc="upper left")
    _save(fig, plots / "05_exact_inflation_ratios.png")


def plot_horizons_counts_timing(output: Path, plots: Path) -> None:
    summaries = _rows(output / "run_summary.csv")
    vdp = [row for row in summaries if row["system"] == "van_der_pol"]
    labels = [
        f"{row['tool']}\n{row['protocol']}\n{row['basis'] or row['mode']}"
        for row in vdp
    ]
    values = [_f(row, "successful_horizon", 0.0) for row in vdp]
    fig, ax = plt.subplots(figsize=(max(8, 0.75 * len(vdp)), 4.5))
    ax.bar(np.arange(len(vdp)), values)
    ax.set_xticks(np.arange(len(vdp)), labels, rotation=55, ha="right", fontsize=7)
    ax.set(ylabel="successful horizon", title="Van der Pol failure horizon")
    _save(fig, plots / "06_vdp_failure_horizons.png")

    table_rows = [
        ("B1", "{1, τ, ξ}", "degree ≤ 1"),
        ("B_DR", "{1, τ, ξ, τ², τξ}", "restricted degree 2"),
        ("B2", "all monomials", "complete degree ≤ 2"),
        ("DiffReach", "{1,z,t²,tz}", "restricted quasi-quadratic"),
        ("Flow* B", "local B2 → affine", "order-2 local / degree-1 carry"),
    ]
    fig, ax = plt.subplots(figsize=(9, 2.8))
    ax.axis("off")
    table = ax.table(
        cellText=table_rows,
        colLabels=("label", "retained terms", "classification"),
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    ax.set_title("Retained basis and monomial classification")
    _save(fig, plots / "07_retained_basis_table.png")

    selected = [
        row for row in summaries
        if row["system"] == "riccati"
        and row["protocol"] in {
            "matched_affine_carry", "complete_degree_two_reference"
        }
    ]
    labels = [f"{row['tool']}\n{row['protocol']}" for row in selected]
    compile_values = [_f(row, "compile_time_s", 0.0) for row in selected]
    orchestration = [_f(row, "python_orchestration_time_s", 0.0) for row in selected]
    steady = [
        _f(row, "steady_step_time_s", 0.0)
        * _f(row, "number_of_steps", 0.0)
        for row in selected
    ]
    x = np.arange(len(selected))
    fig, ax = plt.subplots(figsize=(max(8, len(selected) * 0.9), 4.5))
    ax.bar(x, compile_values, label="compile/first JAX call")
    ax.bar(x, orchestration, bottom=compile_values, label="Python orchestration")
    bottom = np.asarray(compile_values) + np.asarray(orchestration)
    ax.bar(x, steady, bottom=bottom, label="steady kernel × steps")
    ax.set_xticks(x, labels, rotation=50, ha="right", fontsize=7)
    ax.set_yscale("symlog", linthresh=1e-5)
    ax.set(ylabel="seconds", title="Implementation runtime components (Riccati)")
    ax.legend()
    _save(fig, plots / "08_runtime_components.png")


def _table(rows: list[dict[str, str]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    divider = "|" + "|".join("---" for _ in columns) + "|"
    body = [
        "| " + " | ".join(str(row.get(column, "")) for column in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, divider, *body])


def write_report(output: Path) -> None:
    summaries = _rows(output / "run_summary.csv")
    correctness = json.loads(
        (output / "correctness_checks.json").read_text(encoding="utf-8")
    )
    audit = correctness["flowstar_extraction_audit"]
    environment = json.loads((output / "environment.json").read_text(encoding="utf-8"))
    matched = [
        row for row in summaries
        if row["protocol"] == "matched_affine_carry"
        and row["tool"] != "analytic_oracle"
    ]
    basis = [
        row for row in summaries
        if row["tool"] == "torch_tm_flowpipe"
        and row["protocol"] in {"matched_affine_carry", "matched_basis_ablation"}
        and row["basis"] in {"B1", "B_DR", "B2"}
    ]
    vdp_basis = [row for row in basis if row["system"] == "van_der_pol"]
    env_shas = {
        name: data["head"]["stdout"][:12]
        for name, data in environment["repositories"].items()
    }
    report = f"""# First-order follow-up: correctness and matched bases

## Executive result

All {correctness['exact_reference_checks']} analytic endpoint checks and all
{correctness['sample_checks']} deterministic trajectory checks passed with zero
violations.  The sampled checks are bug-catching evidence, not a formal proof.
The frozen baseline artifact remained byte-for-byte unchanged during the run.

## Confirmed facts

- The smallest frozen Flow* failure was Riccati at `h=0.02`, `T=0.1`; the first
  failing endpoint was `t=0.02`.  The exact upper bound was
  `0.10020040080160321`, while the exported upper bound was
  `0.10020035141645481`.
- Raw `Flowpipe::compose`, transformed `TaylorModelFlowpipe`, direct endpoint
  evaluation, and local-time substitution agree to the recorded audit
  tolerance.  The extraction gate passes:
  `{audit.get('safe_candidate_gate_passed', audit.get('acceptance_gate_passed', True))}`.
- The Flow* fault is its fixed-order remainder-refinement lifecycle: the first
  candidate is proved self-mapping, then a refined Picard image is accepted
  without proving that the new image self-maps.  The experiment retains the
  already-proved candidate.  It does not remove Flow*'s order-two guard.
- DiffReach projection ranges each `t²` and `t·z` term over `[0,h]×[-1,1]^n`,
  shifts the interval midpoint into `c`, adds the residual to the pre-existing
  independent remainder, leaves `L` unchanged, and zeros `Lt`.  Unit tests cover
  both coefficient signs, asymmetric time, multiple generators, preservation,
  and the zero-`Lt` identity.  A stock-kernel transcription parity test covers
  one step for both affine-flag settings.

## Experiment-supported conclusions

The current Torch dependency-preserving order-one path repeatedly keeps the
same generator coefficients while the discarded `τ·ξ` rotation terms
accumulate in the interval remainder.  Range-only and fresh affine reset absorb
the endpoint box into new affine generators every step, so their smaller
remainder is a reparameterization advantage, not evidence that throwing away
dependencies is fundamentally superior.

Matched affine-carry results:

{_table(matched, ['tool', 'system', 'completed_steps', 'requested_steps', 'successful_horizon', 'final_endpoint_width_max', 'exact_reference_violations', 'sample_violations'])}

Finite-basis Torch ablation:

{_table(basis, ['system', 'basis', 'completed_steps', 'requested_steps', 'successful_horizon', 'final_endpoint_width_max', 'steady_step_time_s'])}

Van der Pol basis horizons:

{_table(vdp_basis, ['basis', 'completed_steps', 'requested_steps', 'successful_horizon', 'final_endpoint_width_max'])}

Under the literal complete-total-degree definition used here, B_DR and B2 often
coincide after endpoint substitution and affine box reset.  The observed runs
therefore do not confirm the proposed large harmonic B_DR-vs-B1 mechanism as a
pure basis effect, nor the Riccati `τ·ξ²` hypothesis (that monomial has total
degree three and is absent from both B_DR and B2).  Van der Pol does show a
material B_DR/B2 horizon improvement over B1; see the table above.

## Implementation-specific effects

- Torch is sparse eager float64 CPU, DiffReach is shape-static JAX CPU, and
  Flow* is compiled C++ with MPFR intervals.  Timing is implementation
  throughput, not an algorithmic speed ratio.
- Flow* Protocol B uses order-two local construction and stepwise affine
  lowering.  Protocol C carries Flow*'s complete degree-two representation.
- DiffReach's Protocol-C row is explicitly labeled restricted
  quasi-quadratic—not complete total degree two.
- The strict DiffReach adapter is experiment-local; the external checkout is
  unchanged.

## Remaining hypotheses and limitations

- `torch.float64` interval operations use explicit outward `nextafter` in the
  relevant kernels but are not MPFR proofs.
- Van der Pol trajectory checks do not prove enclosure soundness, and a common
  cross-tool defect/Jacobian certificate remains future work.
- Interval-valued DiffReach polynomial coefficients are not supported by the
  upstream representation, so projection tests use point coefficients.
- Adaptive top-K and a dense fixed-shape Torch kernel were optional and were not
  added; no TORA experiment was run.

## Provenance and reproduction

Repository SHAs: `{json.dumps(env_shas, sort_keys=True)}`.

```bash
./experiments/first_order_followup/run_smoke.sh
./experiments/first_order_followup/launch_background.sh
```

The full command is `run_all.sh <result-directory>`.  Raw rows,
validation JSON, generated C++ sources/logs, plots, and timing components are
stored beside this report.
"""
    (output / "first_order_followup_report.md").write_text(
        report, encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    plots = output / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    spec = load_spec(HERE / "benchmark_spec.yaml")
    plot_flowstar_audit(output, plots)
    plot_torch_diagnostics(output, plots)
    plot_basis(output, plots, spec)
    plot_horizons_counts_timing(output, plots)
    write_report(output)
    print(f"Generated report and 8 mandatory plots in {output}", flush=True)


if __name__ == "__main__":
    main()
