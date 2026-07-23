#!/usr/bin/env python3
"""Generate an evidence-backed Markdown report from collected benchmark data."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from common import load_spec, output_dir_from_args

PRIMARY = "native_first_order_setting"
STRICT = "strict_common_affine"
SUPPLEMENTAL = "supplementary_native_representations"
GOOD = {"certified_ok"}

TOOL_LABEL = {
    "torch_tm_flowpipe": "Torch TM",
    "flowstar": "Flow*",
    "diffreach": "DiffReach",
}
PROTOCOL_LABEL = {
    PRIMARY: "native first-order setting",
    STRICT: "strict common affine",
    SUPPLEMENTAL: "supplemental native representation",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def number(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def fmt(value: Any, digits: int = 5) -> str:
    result = number(value)
    if not math.isfinite(result):
        return "—"
    if result == 0:
        return "0"
    if abs(result) >= 1.0e4 or abs(result) < 1.0e-3:
        return f"{result:.{digits}g}"
    return f"{result:.{digits}f}".rstrip("0").rstrip(".")


def esc(value: Any) -> str:
    return str(value).replace("|", r"\|").replace("\n", " ")


def table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    rendered = [[esc(value) for value in row] for row in rows]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rendered)
    return "\n".join(lines)


def unique_runs(summary: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    seen: set[str] = set()
    out: list[dict[str, str]] = []
    for row in summary:
        if row["run_id"] not in seen:
            seen.add(row["run_id"])
            out.append(dict(row))
    return out


def repo_sha(environment: Mapping[str, Any], name: str) -> str:
    try:
        return environment["repositories"][name]["head"]["stdout"].strip()
    except (KeyError, TypeError, AttributeError):
        return "unknown"


def status_rows(runs: Sequence[Mapping[str, str]], protocol: str) -> list[list[str]]:
    counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for run in runs:
        if run["protocol"] == protocol:
            counts[(run["tool"], run["system"])][run["status"]] += 1
    out: list[list[str]] = []
    for (tool, system), statuses in sorted(counts.items()):
        out.append(
            [
                TOOL_LABEL.get(tool, tool),
                system,
                str(sum(statuses.values())),
                ", ".join(f"{key}: {value}" for key, value in sorted(statuses.items())),
            ]
        )
    return out


def native_inflation(summary: Sequence[Mapping[str, str]]) -> list[list[str]]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in summary:
        ratio = number(row.get("exact_inflation_ratio"))
        if (
            row["protocol"] == PRIMARY
            and row["status"] in GOOD
            and math.isfinite(ratio)
        ):
            grouped[(row["tool"], row["system"])].append(ratio)
    out: list[list[str]] = []
    for (tool, system), values in sorted(grouped.items()):
        out.append(
            [
                TOOL_LABEL.get(tool, tool),
                system,
                str(len(values)),
                fmt(min(values)),
                fmt(statistics.median(values)),
                fmt(max(values)),
            ]
        )
    return out


def width_change(summary: Sequence[Mapping[str, str]]) -> tuple[list[list[str]], list[float]]:
    """Compare DiffReach quasi-quadratic and affine final box widths per run."""
    by_key: dict[tuple[str, str, str, str], dict[str, Mapping[str, str]]] = defaultdict(dict)
    for row in summary:
        if (
            row["tool"] == "diffreach"
            and row["state_index"] == "0"
            and row["status"] in GOOD
            and row["protocol"] in {PRIMARY, SUPPLEMENTAL}
        ):
            key = (row["system"], row["h"], row["horizon"], row["state_index"])
            by_key[key][row["protocol"]] = row
    output: list[list[str]] = []
    changes: list[float] = []
    by_system: dict[str, list[float]] = defaultdict(list)
    for key, protocols in by_key.items():
        if PRIMARY not in protocols or SUPPLEMENTAL not in protocols:
            continue
        affine = number(protocols[PRIMARY].get("sum_final_widths"))
        quasi = number(protocols[SUPPLEMENTAL].get("sum_final_widths"))
        if affine > 0 and math.isfinite(quasi):
            change = 100.0 * (affine - quasi) / affine
            changes.append(change)
            by_system[key[0]].append(change)
    for system, values in sorted(by_system.items()):
        output.append(
            [
                system,
                str(len(values)),
                fmt(min(values), 4) + "%",
                fmt(statistics.median(values), 4) + "%",
                fmt(max(values), 4) + "%",
            ]
        )
    return output, changes


def vdp_horizons(runs: Sequence[Mapping[str, str]]) -> list[list[str]]:
    grouped: dict[tuple[str, str, float], list[Mapping[str, str]]] = defaultdict(list)
    for run in runs:
        if run["system"] == "van_der_pol":
            grouped[(run["tool"], run["protocol"], number(run["h"]))].append(run)
    out: list[list[str]] = []
    for (tool, protocol, h), values in sorted(grouped.items()):
        certified = [
            number(run["horizon"])
            for run in values
            if run["status"] in GOOD and math.isfinite(number(run["horizon"]))
        ]
        failure_times = [
            number(run["first_failure_time"])
            for run in values
            if run["status"] not in GOOD and math.isfinite(number(run["first_failure_time"]))
        ]
        statuses = Counter(run["status"] for run in values)
        out.append(
            [
                TOOL_LABEL.get(tool, tool),
                PROTOCOL_LABEL.get(protocol, protocol),
                fmt(h),
                fmt(max(certified) if certified else math.nan),
                fmt(min(failure_times) if failure_times else math.nan),
                ", ".join(f"{key}:{value}" for key, value in sorted(statuses.items())),
            ]
        )
    return out


def runtime_rows(
    runs: Sequence[Mapping[str, str]],
    protocol: str,
) -> list[list[str]]:
    grouped: dict[tuple[str, str], list[Mapping[str, str]]] = defaultdict(list)
    for run in runs:
        if run["protocol"] == protocol and run["status"] in GOOD:
            grouped[(run["tool"], run["system"])].append(run)
    out: list[list[str]] = []
    for (tool, system), values in sorted(grouped.items()):
        build = [number(run["build_time_s"]) for run in values]
        warmup = [number(run["warmup_time_s"]) for run in values]
        steady = [number(run["steady_runtime_median_s"]) for run in values]
        build = [value for value in build if math.isfinite(value)]
        warmup = [value for value in warmup if math.isfinite(value)]
        steady = [value for value in steady if math.isfinite(value)]
        out.append(
            [
                TOOL_LABEL.get(tool, tool),
                system,
                fmt(statistics.median(build) if build else math.nan),
                fmt(statistics.median(warmup) if warmup else math.nan),
                fmt(statistics.median(steady) if steady else math.nan),
                values[0]["device"],
            ]
        )
    return out


def runtime_quality_tradeoff(runs: Sequence[Mapping[str, str]]) -> str:
    """Summarize paired primary Torch/DiffReach runtime and width tradeoffs."""
    indexed: dict[tuple[str, str, str], dict[str, Mapping[str, str]]] = defaultdict(dict)
    for run in runs:
        if run["protocol"] == PRIMARY and run["status"] in GOOD:
            indexed[(run["system"], run["h"], run["horizon"])][run["tool"]] = run
    pairs: list[tuple[str, float, float]] = []
    for key, tools in indexed.items():
        if "torch_tm_flowpipe" not in tools or "diffreach" not in tools:
            continue
        torch_run, diff_run = tools["torch_tm_flowpipe"], tools["diffreach"]
        torch_time = number(torch_run["steady_runtime_median_s"])
        diff_time = number(diff_run["steady_runtime_median_s"])
        torch_width = number(torch_run["sum_final_widths"])
        diff_width = number(diff_run["sum_final_widths"])
        if min(torch_time, diff_time, torch_width, diff_width) <= 0:
            continue
        faster = "DiffReach" if diff_time < torch_time else "Torch TM"
        speedup = max(torch_time, diff_time) / min(torch_time, diff_time)
        faster_width = diff_width if faster == "DiffReach" else torch_width
        other_width = torch_width if faster == "DiffReach" else diff_width
        pairs.append((faster, speedup, faster_width / other_width))
    if not pairs:
        return "No certified primary Torch/DiffReach pairs were available."
    faster_counts = Counter(pair[0] for pair in pairs)
    wider_materially = sum(pair[2] > 1.10 for pair in pairs)
    narrower_materially = sum(pair[2] < 1.0 / 1.10 for pair in pairs)
    median_speedup = statistics.median(pair[1] for pair in pairs)
    median_width_ratio = statistics.median(pair[2] for pair in pairs)
    common_faster, common_count = faster_counts.most_common(1)[0]
    vdp_failures: dict[tuple[str, str], float] = {}
    for run in runs:
        failure = number(run.get("first_failure_time"))
        if (
            run["protocol"] == PRIMARY
            and run["system"] == "van_der_pol"
            and run["status"] not in GOOD
            and run["tool"] in {"torch_tm_flowpipe", "diffreach"}
            and failure > 0
        ):
            key = (run["tool"], run["h"])
            vdp_failures[key] = min(failure, vdp_failures.get(key, failure))
    failure_clause = ""
    if vdp_failures:
        failure_clause = " First Van der Pol failures were " + "; ".join(
            f"{TOOL_LABEL.get(tool, tool)} h={fmt(h)}: t={fmt(time_value)}"
            for (tool, h), time_value in sorted(vdp_failures.items())
        ) + "."
    return (
        f"Across {len(pairs)} paired certified primary configurations, {common_faster} "
        f"was faster in {common_count}; the median faster/slower speed ratio was "
        f"{fmt(median_speedup)}× and the faster method's median final summed-width "
        f"ratio was {fmt(median_width_ratio)}×. The faster result was >10% wider in "
        f"{wider_materially} pairs and >10% narrower in {narrower_materially} pairs."
        f"{failure_clause}"
    )


def exact_check_counts(checks: Mapping[str, Any]) -> list[list[str]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for check in checks.get("checks", []):
        grouped[str(check["check"])].append(check)
    rows: list[list[str]] = []
    for name, values in sorted(grouped.items()):
        rows.append(
            [
                name,
                str(len(values)),
                str(sum(int(value["checked"]) for value in values)),
                str(sum(int(value["violations"]) for value in values)),
                "analytic exact-hull containment" if name == "exact_endpoint_containment" else "sanity check only",
            ]
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", default=str(HERE / "benchmark_spec.yaml"))
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    spec = load_spec(args.spec)
    output_dir = output_dir_from_args(args.output_dir)
    summary = read_csv(output_dir / "run_summary.csv")
    runs = unique_runs(summary)
    environment = read_json(output_dir / "environment.json")
    checks = read_json(output_dir / "correctness_checks.json")

    sha_rows = [
        ["torch_tm_flowpipe benchmark worktree", repo_sha(environment, "torch_benchmark_worktree")],
        ["torch_tm_flowpipe original checkout", repo_sha(environment, "torch_original_checkout")],
        ["DiffReach", repo_sha(environment, "diffreach")],
        ["Flow* toolbox", repo_sha(environment, "flowstar")],
    ]
    execution_shas: dict[str, set[str]] = defaultdict(set)
    for run in runs:
        if run.get("git_commit"):
            execution_shas[run["tool"]].add(run["git_commit"])
    execution_sha_rows = [
        [
            TOOL_LABEL.get(tool, tool),
            ", ".join(sorted(shas)),
        ]
        for tool, shas in sorted(execution_shas.items())
    ]
    benchmarks = []
    for name, system in spec["systems"].items():
        rhs_terms = {
            "riccati": r"$\dot{x}=x^2$",
            "harmonic": r"$\dot{x}_1=x_2,\ \dot{x}_2=-x_1$",
            "van_der_pol": r"$\dot{x}_1=x_2,\ \dot{x}_2=(1-x_1^2)x_2-x_1$",
        }[name]
        benchmarks.append(
            [
                name,
                rhs_terms,
                str(system["initial_box"]),
                ", ".join(str(value) for value in system["step_sizes"]),
                ", ".join(str(value) for value in system["horizons"]),
            ]
        )

    inflation_rows = native_inflation(summary)
    quasi_rows, quasi_changes = width_change(summary)
    semantics_rows = [
        [
            "Torch TM",
            "order=1",
            "all monomials of total degree ≤1 in local time and initial generators",
            "1",
            "higher-degree products are interval-bounded and added to the TM remainder",
        ],
        [
            "Flow*",
            "fixed order=1",
            "unsupported by this toolbox API; setFixedStepsize requires order ≥2",
            "—",
            "no order-2 result is relabeled first order",
        ],
        [
            "DiffReach",
            "TRUNCATE_TO_AFFINE=True",
            "affine final polynomial, but t² and t·z are created during integration",
            "2 transient",
            "Lt interval radius is embedded into reused L generators, not an independent remainder",
        ],
        [
            "DiffReach",
            "TRUNCATE_TO_AFFINE=False",
            "{1, z, t², t·z} as implemented by c/L/Lt (restricted quasi-quadratic)",
            "2 restricted",
            "supplemental, not a complete total-degree-2 basis",
        ],
    ]

    primary_counts = Counter(run["status"] for run in runs if run["protocol"] == PRIMARY)
    strict_counts = Counter(run["status"] for run in runs if run["protocol"] == STRICT)
    supplemental_counts = Counter(run["status"] for run in runs if run["protocol"] == SUPPLEMENTAL)
    successful = sum(status == "certified_ok" for status in (run["status"] for run in runs))
    failures = len(runs) - successful
    environment_decisions = environment.get("environment_decisions", {})
    torch_probe = (
        environment.get("software_hardware", {})
        .get("py11_torch", {})
        .get("stdout", "")
    )
    cuda_visible = "cuda_available True" in str(torch_probe)
    hardware_sentence = (
        "CUDA devices were visible, but the frozen benchmark specification selected "
        "float64 CPU batch 1 for Torch and CPU JAX for DiffReach."
        if cuda_visible
        else "CUDA devices were not visible to the audited run; all measurements were CPU-only."
    )
    tradeoff_sentence = runtime_quality_tradeoff(runs)

    harmonic_native = [
        row
        for row in summary
        if row["system"] == "harmonic"
        and row["protocol"] == PRIMARY
        and row["status"] in GOOD
        and math.isfinite(number(row["exact_inflation_ratio"]))
    ]
    harmonic_by_tool: dict[str, list[float]] = defaultdict(list)
    for row in harmonic_native:
        harmonic_by_tool[row["tool"]].append(number(row["exact_inflation_ratio"]))
    harmonic_ranking = sorted(
        (
            statistics.median(values),
            TOOL_LABEL.get(tool, tool),
        )
        for tool, values in harmonic_by_tool.items()
        if values
    )
    harmonic_supplemental: dict[str, list[float]] = defaultdict(list)
    for row in summary:
        ratio = number(row.get("exact_inflation_ratio"))
        if (
            row["system"] == "harmonic"
            and row["protocol"] == SUPPLEMENTAL
            and row["status"] in GOOD
            and math.isfinite(ratio)
        ):
            harmonic_supplemental[row["tool"]].append(ratio)
    if harmonic_ranking:
        native_details = ", ".join(
            f"{label} {fmt(value)}" for value, label in harmonic_ranking
        )
        wrapping_answer = (
            f"{harmonic_ranking[0][1]} controlled wrapping best among supported native "
            f"first-order settings (median exact inflation: {native_details})."
        )
        if harmonic_supplemental:
            supplemental_details = ", ".join(
                f"{TOOL_LABEL.get(tool, tool)} {fmt(statistics.median(values))}"
                for tool, values in sorted(harmonic_supplemental.items())
            )
            wrapping_answer += (
                f" Supplemental medians were {supplemental_details}; Flow* there is "
                "fixed order 2 and is not a first-order comparison."
            )
    else:
        wrapping_answer = "No native implementation completed a validated harmonic run."

    quasi_sentence = (
        f"Across {len(quasi_changes)} paired certified configurations, disabling affine "
        f"projection changed DiffReach's final summed width by a median reduction of "
        f"{fmt(statistics.median(quasi_changes), 4)}% "
        f"(range {fmt(min(quasi_changes), 4)}% to {fmt(max(quasi_changes), 4)}%)."
        if quasi_changes
        else "No paired certified DiffReach configurations were available for this comparison."
    )

    lines = [
        "# First-order three-way reachability benchmark",
        "",
        "## 1. Executive summary",
        "",
        (
            f"The canonical sweep produced {len(runs)} tool/protocol/configuration runs: "
            f"{successful} certified and {failures} unsupported or failed. The central result "
            "is semantic rather than a winner: the three projects do not expose the same "
            "object under their apparent first-order controls. Flow* fixed order 1 is explicitly "
            "rejected by the installed toolbox, and DiffReach's affine flag has transient "
            "degree-two time terms plus a nonstandard final projection. Torch TM is therefore "
            "the only primary path here that directly executes complete-total-degree-one "
            "Taylor models."
        ),
        "",
        "This report does not claim that one tool is globally better based on three plant systems.",
        "",
        "## 2. Repository SHAs and environments",
        "",
        table(["repository", "audited HEAD"], sha_rows),
        "",
        "Commits recorded in the actual adapter rows:",
        "",
        table(["adapter", "execution commit"], execution_sha_rows),
        "",
        f"- Torch/plotting: {environment_decisions.get('torch_and_plotting', 'unknown')}.",
        f"- DiffReach: {environment_decisions.get('diffreach', 'unknown')}.",
        f"- Flow*: {environment_decisions.get('flowstar', 'unknown')}.",
        f"- Host: Intel Xeon Gold 6138 CPU. {hardware_sentence}",
        "- Full command output, branch listings, remotes, 100-commit graphs, untracked files, compiler, OS, CPU, memory, and device probes are preserved in `environment.json` and `environment.txt`.",
        "",
        "The declared DiffReach editable install was attempted but pip found an internal dependency conflict: "
        "`jax2onnx` required Equinox ≥0.13.1 while available `immrax[cuda]` releases required Equinox ~=0.12.2. "
        "The plant-only analytic path used a minimal read-only source import in `diffreach312`.",
        "",
        "## 3. Benchmark definitions",
        "",
        table(["system", "dynamics", "initial box", "step sizes", "horizons"], benchmarks),
        "",
        (
            f"All configurations use a fixed grid, batch size 1, one partition, seed "
            f"{spec['random_seed']}, {spec['sample_trajectories']} deterministic initial samples, "
            f"and {spec['steady_repetitions']} steady timing repetitions. Main protocols do not "
            "rescue failures by raising order, adapting the step, or partitioning."
        ),
        "",
        "## 4. What order 1 means in each implementation",
        "",
        table(
            ["implementation", "setting", "retained/effective basis", "degree", "discarded-term handling"],
            semantics_rows,
        ),
        "",
        (
            "This distinction is measured by automated support diagnostics, not inferred from option "
            "names. In particular, DiffReach's final `Lt==0` under the affine flag does not establish "
            "common affine semantics because nonzero `Lt` is observed immediately after time integration."
        ),
        "",
        "## 5. Primary native-first-order results",
        "",
        f"Aggregate statuses: {dict(sorted(primary_counts.items()))}.",
        "",
        table(["tool", "system", "configurations", "status counts"], status_rows(runs, PRIMARY)),
        "",
        "Exact-width inflation for certified Riccati and harmonic endpoint states:",
        "",
        table(["tool", "system", "states", "minimum", "median", "maximum"], inflation_rows),
        "",
        "Flow* entries are absent from quantitative order-one tables because its public fixed-step "
        "configuration API returned false for order 1. Unsupported is a result, not a missing run.",
        "",
        "## 6. Strict-common-affine results",
        "",
        f"Aggregate statuses: {dict(sorted(strict_counts.items()))}.",
        "",
        table(["tool", "system", "configurations", "status counts"], status_rows(runs, STRICT)),
        "",
        (
            "Torch dependency-preserving order 1 meets the declared common affine polynomial basis. "
            "Flow* cannot instantiate fixed order 1. DiffReach is marked unsupported because its "
            "tested affine projection converts dropped Lt radius into existing generator coefficients; "
            "the benchmark found no demonstrated independent-remainder projection with matching semantics."
        ),
        "",
        "## 7. Supplemental native representations",
        "",
        f"Aggregate statuses: {dict(sorted(supplemental_counts.items()))}.",
        "",
        table(["tool", "system", "configurations", "status counts"], status_rows(runs, SUPPLEMENTAL)),
        "",
        (
            "The supplements are Torch range-only restarts, Flow* fixed total-degree 2 diagnostics, "
            "and DiffReach with `TRUNCATE_TO_AFFINE=False`. They illuminate wrapping and basis effects "
            "but are not relabeled as primary first-order results."
        ),
        "",
        "## 8. Tightness analysis",
        "",
        wrapping_answer,
        "",
        "DiffReach quasi-quadratic change relative to its affine-dynamics path "
        "(positive means the quasi-quadratic result is narrower):",
        "",
        table(["system", "paired runs", "minimum", "median", "maximum"], quasi_rows),
        "",
        quasi_sentence,
        "",
        (
            "For Riccati, nonlinear products are the direct precision-loss point: Torch moves all "
            "degree-two-and-higher products to an interval remainder at order 1, while DiffReach "
            "temporarily keeps its restricted Lt terms and then either projects or retains them. "
            "Range-only Torch additionally discards cross-step generator dependence. Flow* supplies "
            "no scalar order-one trajectory because the installed API refuses that order."
        ),
        "",
        "## 9. Validation and failure analysis",
        "",
        "Van der Pol validated horizons and first reported failure times:",
        "",
        table(
            ["tool", "protocol", "h", "max certified T", "earliest failure time", "status counts"],
            vdp_horizons(runs),
        ),
        "",
        (
            "Each adapter stops exporting enclosures after its first failed contraction, failed "
            "validation, non-finite interval, or timeout. The collector can further downgrade a "
            "nominally certified run to `sample_violation`; it never heals a failure."
        ),
        "",
        "## 10. Runtime analysis",
        "",
        "Primary certified configurations:",
        "",
        table(
            ["tool", "system", "median build", "median warmup", "median steady", "device"],
            runtime_rows(runs, PRIMARY),
        ),
        "",
        "Supplemental certified configurations:",
        "",
        table(
            ["tool", "system", "median build", "median warmup", "median steady", "device"],
            runtime_rows(runs, SUPPLEMENTAL),
        ),
        "",
        tradeoff_sentence,
        "",
        (
            "Build/source generation, first-call or JIT time, and steady runtime are deliberately "
            "separate. These are CPU measurements on one shared host, not a hardware-fair CPU/GPU "
            "claim. A fast unsupported run is not treated as useful speed, and timing is interpreted "
            "alongside width and validation horizon. The all-CPU sweep includes the requested "
            "batch-1 subsets Riccati h=0.01/T=1, harmonic h=0.01/T=5, and Van der Pol "
            "h=0.01/T=0.5."
        ),
        "",
        "## 11. Exact references and sampled-trajectory sanity checks",
        "",
        table(["check", "runs", "values checked", "violations", "interpretation"], exact_check_counts(checks)),
        "",
        (
            "Riccati and harmonic endpoint hulls are analytic. Van der Pol and whole-segment tubes "
            "use high-accuracy SciPy DOP853 trajectories only as bug-catching samples. Samples do "
            "not prove soundness; passing them is strictly weaker than a formal enclosure proof."
        ),
        "",
        "## 12. Limitations",
        "",
        "- The retained polynomial bases differ materially, including DiffReach's restricted Lt basis.",
        "- Numerical soundness backends differ: Torch floating-point intervals, Flow* MPFR intervals, and JAX floating-point interval arithmetic are not interchangeable guarantees.",
        "- All reported primary timings are CPU-only by specification; visible GPUs were deliberately excluded, and GPU results could alter performance but not representation semantics.",
        "- Endpoint enclosures and whole-segment tubes are distinct and were extracted/evaluated separately.",
        "- The benchmark exercises plant dynamics only; it imports no controller or CROWN component.",
        "- Sampled trajectories are sanity checks and never establish formal soundness.",
        "- Flow* total-degree-two results and range-only/quasi-quadratic ablations are supplemental, not substitutes for missing common-affine runs.",
        "",
        "## 13. Recommended next experiment",
        "",
        (
            "Implement and unit-test in DiffReach an explicit projection that sends every Lt term "
            "to a fresh independent interval remainder, then expose or safely enable Flow* fixed "
            "order 1 in a separate toolbox branch. Re-run the same frozen plant spec with those two "
            "semantics changes before expanding to partitions, controllers, or GPU batching."
        ),
        "",
        "## Conclusion",
        "",
        (
            "**Scalar nonlinear precision:** Torch order 1 loses nonlinear dependence when degree-two "
            "products enter its interval remainder; Torch range-only loses additional cross-step "
            "dependence. DiffReach loses or reshapes precision at its Lt projection, while its "
            "quasi-quadratic mode retains restricted time interactions. Flow* order 1 is unsupported."
        ),
        "",
        f"**Linear oscillator wrapping:** {wrapping_answer}",
        "",
        (
            "**Van der Pol failure horizon:** The per-step-size values are reported in the validation "
            "table above; unsupported configurations have no validated horizon and are not interpreted "
            "as numerical failures."
        ),
        "",
        (
            "**Same DiffReach basis?** No. Its native affine path creates nonzero degree-two Lt terms "
            "during integration and its default non-affine path retains a restricted quasi-quadratic "
            "basis, unlike complete-total-degree-one Torch semantics. Flow* order 1 did not run."
        ),
        "",
        f"**Quasi-quadratic improvement:** {quasi_sentence}",
        "",
        (
            f"**Runtime versus quality:** {tradeoff_sentence} Unsupported order-one Flow* timings "
            "are not gains, and validation failures are not treated as speedups."
        ),
        "",
        "## Reproduction",
        "",
        "```bash",
        "cd /srv/local/shengenli/torch_tm_flowpipe_first_order_bench",
        "experiments/first_order_three_way/run_smoke.sh",
        "experiments/first_order_three_way/run_all.sh",
        "# or: experiments/first_order_three_way/launch_background.sh",
        "```",
        "",
    ]
    report = output_dir / "first_order_three_way_report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
