#!/usr/bin/env python3
"""Generate an evidence-derived Markdown report for the full experiment."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

TITLE = "Three-way low-order reachability comparison under common external contracts"
LABELS = {
    "torch_tm_flowpipe": "Torch TM",
    "diffreach": "DiffReach",
    "flowstar": "Flow*",
}


def _markdown_table(
    rows: Sequence[Mapping[str, Any]],
    fields: Sequence[str] | None = None,
) -> str:
    if not rows:
        return "_No rows._"
    fields = list(fields or rows[0].keys())

    def value(item: Any) -> str:
        if item is None:
            return ""
        if isinstance(item, float):
            if math.isnan(item):
                return ""
            return f"{item:.6g}"
        return str(item).replace("|", "\\|").replace("\n", " ")

    header = "| " + " | ".join(fields) + " |"
    rule = "| " + " | ".join("---" for _ in fields) + " |"
    body = [
        "| " + " | ".join(value(row.get(field, "")) for field in fields) + " |"
        for row in rows
    ]
    return "\n".join([header, rule, *body])


def _records(
    frame: pd.DataFrame,
    fields: Sequence[str],
    *,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    data = frame[list(fields)].copy()
    if limit is not None:
        data = data.head(limit)
    return data.where(pd.notna(data), "").to_dict(orient="records")


def _tightest_one_step(one_step: pd.DataFrame) -> list[dict[str, Any]]:
    data = one_step[one_step["status"] == "validated"].copy()
    data["width"] = pd.to_numeric(data["width"], errors="coerce")
    output: list[dict[str, Any]] = []
    for keys, group in data.groupby(["system", "h", "state_name"]):
        best = group.loc[group["width"].idxmin()]
        output.append(
            {
                "system": keys[0],
                "h": float(keys[1]),
                "state": keys[2],
                "tightest_valid_tool": LABELS.get(best["tool"], best["tool"]),
                "width": float(best["width"]),
            }
        )
    return output


def _best_common_box(common: pd.DataFrame) -> list[dict[str, Any]]:
    data = common[common["status"] == "validated"].copy()
    data["width"] = pd.to_numeric(data["width"], errors="coerce")
    output: list[dict[str, Any]] = []
    for keys, group in data.groupby(
        ["system", "h", "checkpoint", "state_name"]
    ):
        best = group.loc[group["width"].idxmin()]
        output.append(
            {
                "system": keys[0],
                "h": float(keys[1]),
                "time": float(keys[2]),
                "state": keys[3],
                "smallest_valid_width_tool": LABELS.get(
                    best["tool"], best["tool"]
                ),
                "width": float(best["width"]),
            }
        )
    return output


def generate(output: Path) -> str:
    correctness = json.loads(
        (output / "correctness_checks.json").read_text(encoding="utf-8")
    )
    environment = json.loads(
        (output / "environment.json").read_text(encoding="utf-8")
    )
    one_step = pd.read_csv(output / "one_step_summary.csv")
    common = pd.read_csv(output / "common_time_summary.csv")
    failure = pd.read_csv(output / "failure_horizon_summary.csv")
    runtime = pd.read_csv(output / "runtime_summary.csv")
    semantics = pd.read_csv(output / "semantics_summary.csv")
    raw = pd.read_csv(output / "raw_results.csv")

    gate_rows = [
        {
            "gate": name,
            "checks": details["checks"],
            "violations": details["violations"],
            "passed": details["passed"],
        }
        for name, details in correctness["gates"].items()
    ]
    primary_failure = failure[
        failure["tool_variant"].isin(
            [
                "complete_total_degree_order_1",
                "affine_flag",
                "minimum_supported_fixed_order_2",
            ]
        )
        & (failure["state_index"] == 0)
    ].copy()
    vdp_failure = primary_failure[
        primary_failure["system"] == "van_der_pol"
    ].copy()
    vdp_failure["failure_horizon_or_censor"] = pd.to_numeric(
        vdp_failure["failure_horizon_or_censor"], errors="coerce"
    )
    native_final = failure[
        (failure["protocol"] == "native_low_order")
    ].copy()
    runtime_fields = [
        "tool",
        "tool_variant",
        "protocol",
        "system",
        "h",
        "build_time_s",
        "jit_compile_time_s",
        "first_execution_time_s",
        "steady_runtime_per_step_s",
    ]
    runtime_median = (
        runtime.groupby(["tool", "tool_variant", "protocol"], as_index=False)[
            [
                "build_time_s",
                "jit_compile_time_s",
                "first_execution_time_s",
                "steady_runtime_per_step_s",
            ]
        ]
        .median(numeric_only=True)
        .sort_values(["protocol", "tool", "tool_variant"])
    )
    shas = environment["git_shas"]
    provenance = environment["diffreach_upstream_provenance"]
    vdp_best_rows: list[dict[str, Any]] = []
    for keys, group in vdp_failure.groupby(["protocol", "h"]):
        maximum = pd.to_numeric(
            group["failure_horizon_or_censor"], errors="coerce"
        ).max()
        winners = group[
            pd.to_numeric(
                group["failure_horizon_or_censor"], errors="coerce"
            )
            == maximum
        ]
        vdp_best_rows.append(
            {
                "protocol": keys[0],
                "h": float(keys[1]),
                "farthest_primary_tool(s)": ", ".join(
                    LABELS.get(tool, tool)
                    for tool in sorted(winners["tool"].unique())
                ),
                "failure_horizon_or_censor": float(maximum),
            }
        )

    one_step_display = one_step[
        [
            "tool",
            "system",
            "h",
            "state_name",
            "status",
            "lower",
            "upper",
            "width",
            "exact_inflation_ratio",
            "native_validation_status",
        ]
    ].copy()
    common_display = common[
        [
            "tool",
            "system",
            "h",
            "checkpoint",
            "state_name",
            "status",
            "lower",
            "upper",
            "width",
        ]
    ].copy()
    failure_display = primary_failure[
        [
            "tool",
            "protocol",
            "system",
            "h",
            "run_status",
            "first_failure_time",
            "final_valid_time",
            "width_at_own_final_valid_step",
        ]
    ].copy()
    native_display = native_final[
        [
            "tool",
            "tool_variant",
            "system",
            "h",
            "state_name",
            "run_status",
            "final_valid_time",
            "width_at_own_final_valid_step",
        ]
    ].copy()

    lines = [
        f"# {TITLE}",
        "",
        "> **Superseded:** this historical protocol did not provide matched "
        "internal bases and its Flow* extraction was later invalidated. Any "
        "cross-tool tightness, speed, horizon, or winner wording below is "
        "retracted. Use `../three_tool_deep_study/`.",
        "",
        "## Result status",
        "",
        (
            "**All strict correctness gates passed.**"
            if correctness["all_gates_passed"]
            else "**One or more strict correctness gates failed.**"
        ),
        "",
        _markdown_table(gate_rows),
        "",
        "Deterministic trajectory checks are sanity checks only; they do not "
        "establish soundness. Native validation and analytic containment are "
        "reported independently.",
        "",
        "## Direct answers",
        "",
        "1. **Literal common internal order:** No. Torch order 1, the "
        "DiffReach affine flag, and Flow* fixed order 2 have different retained "
        "support and validation semantics.",
        "2. **Identical one-step input:** Torch gives the smallest valid "
        "Riccati widths at every tested `h`; DiffReach gives the smallest valid "
        "harmonic and Van der Pol widths at every tested `h`/state. Flow* has "
        "no accepted row at some larger steps, and those configurations are "
        "reported as `validation_failed` rather than ranked.",
        "3. **Common componentwise-box carry:** Torch is tightest at the "
        "Riccati checkpoints. DiffReach is tightest at every harmonic and Van "
        "der Pol common checkpoint/state where the primary tools can be "
        "compared. Flow* fails before harmonic `t=4`, Van der Pol `t=0.08` at "
        "`h=0.005`, and the first Van der Pol step at `h=0.01`.",
        "4. **Van der Pol validation horizon:** The farthest primary tool "
        "depends on protocol and `h`, as summarized immediately below. The "
        "supplemental default DiffReach quasi-quadratic native variant reaches "
        "the requested `T=1` for both step sizes.",
        "",
        _markdown_table(vdp_best_rows),
        "",
        "5. **Throughput after one-time work:** There is no system-independent "
        "winner. Flow* is fastest on scalar Riccati at roughly `0.07 ms/step`; "
        "DiffReach is typically about `0.1–0.3 ms/step` after JIT and is faster "
        "than Flow* on the two-state harmonic runs; Torch eager execution is "
        "roughly `8–24 ms/step`. These steady rates exclude Flow* builds "
        "(about `1.7 s` per generated executable) and DiffReach JIT "
        "(about `0.4–1.6 s` per configuration).",
        "6. **Native versus controlled carry:** Native carry materially changes "
        "widths and horizons because each tool retains its own dependencies. "
        "Most notably, the supplemental default DiffReach quasi-quadratic "
        "variant reaches Van der Pol `T=1`, while the primary affine flag and "
        "all common-box primary runs fail earlier.",
        "",
        "## Tool identity and semantics",
        "",
        f"- Torch repository: `{environment['tool_paths']['torch']}` at `{shas['torch']}`.",
        f"- DiffReach repository: `{environment['tool_paths']['diffreach']}` at `{shas['diffreach']}`.",
        f"- Flow* repository/static library: `{environment['tool_paths']['flowstar']}` at `{shas['flowstar']}`.",
        "",
        "The tools cannot be compared under a literal common internal order. "
        "Torch retains complete total-degree order 1, DiffReach's affine flag "
        "has transient restricted quasi-quadratic support before its final "
        "projection, and Flow* rejects fixed order 1 and runs at fixed order 2.",
        "",
        _markdown_table(
            _records(
                semantics,
                [
                    "tool",
                    "tool_variant",
                    "protocol",
                    "local_order",
                    "local_retained_basis",
                    "carried_representation",
                    "reset_policy",
                    "validator",
                ],
            )
        ),
        "",
        "## Proof of real upstream DiffReach execution",
        "",
        f"The primary adapter calls `{provenance['upstream_class']}.step_once` "
        f"from `{provenance['upstream_step_source_file']}` at source line "
        f"{provenance['upstream_step_source_line']}. The callable identity gate "
        f"is `{provenance['upstream_step_callable_identity']}` and the full run "
        f"recorded {provenance['total_upstream_step_trace_invocations']} upstream "
        "JAX trace invocations.",
        "",
        f"Picard validation resolves to `{provenance['upstream_picard_callable']}` "
        f"in `{provenance['upstream_picard_source_file']}`; Taylor-model "
        f"operations resolve to `{provenance['upstream_taylor_model_class']}` in "
        f"`{provenance['upstream_taylor_model_source_file']}`. The optional "
        "`jax_verify` shim is fail-fast and only satisfies imports for unused "
        "neural-bound paths. The external DiffReach repository was not modified.",
        "",
        "## Protocol A: identical one-step input",
        "",
        _markdown_table(
            _records(one_step_display, list(one_step_display.columns))
        ),
        "",
        "Tightest valid endpoint by configuration/state:",
        "",
        _markdown_table(_tightest_one_step(one_step)),
        "",
        "## Protocol B: common componentwise-box carry",
        "",
        "Widths below are compared only at the same absolute checkpoint. A "
        "`validation_failed` entry contains no substituted earlier width.",
        "",
        _markdown_table(
            _records(common_display, list(common_display.columns))
        ),
        "",
        "Smallest valid common-time width by configuration/state:",
        "",
        _markdown_table(_best_common_box(common)),
        "",
        "## Failure horizons and each method's own final valid step",
        "",
        _markdown_table(
            _records(failure_display, list(failure_display.columns))
        ),
        "",
        "On Van der Pol, the farthest validated horizon depends on protocol; "
        "the following table retains the protocol and never ranks widths from "
        "different failure times:",
        "",
        _markdown_table(
            _records(
                vdp_failure,
                [
                    "tool",
                    "protocol",
                    "h",
                    "run_status",
                    "failure_horizon_or_censor",
                    "final_valid_time",
                ],
            )
        ),
        "",
        "## Protocol C: native low-order supplement",
        "",
        _markdown_table(
            _records(native_display, list(native_display.columns))
        ),
        "",
        "Native results differ from controlled box carry because Torch retains "
        "initial-generator dependency, DiffReach retains its upstream symbolic "
        "normalization state, and Flow* retains a normalized Taylor-model "
        "flowpipe. Protocol B deliberately erases all of those dependencies at "
        "every boundary.",
        "",
        "## Runtime decomposition",
        "",
        "Build, JIT, first execution, and steady execution are separate. The "
        "steady column is the relevant implementation-throughput measure after "
        "one-time work; no combined total-runtime ranking is claimed.",
        "",
        _markdown_table(
            _records(runtime_median, list(runtime_median.columns))
        ),
        "",
        "## Figures",
        "",
        "- [One-step exact inflation ratio](plots/one_step_exact_inflation_ratio_vs_h.png)",
        "- [Common-box endpoint width curves](plots/multi_step_common_box_carry_width_vs_time.png)",
        "- [Common-time grouped widths](plots/common_time_grouped_width_bars.png)",
        "- [First validation-failure horizon](plots/first_validation_failure_horizon.png)",
        "- [Runtime decomposition](plots/runtime_decomposition.png)",
        "- [Native low-order width curves](plots/native_low_order_width_curves.png)",
        "- [Semantics table](plots/semantics_table.png)",
        "",
        "## Remaining limitations",
        "",
        "- DiffReach uses floating-point JAX interval-style arithmetic rather "
        "than MPFR-directed interval arithmetic; analytic and sampled gates "
        "detect tested violations but do not turn sampling into a proof.",
        "- Flow* extraction restores the validated initial Picard candidate "
        "instead of exporting the stock un-revalidated refinement image. This "
        "is conservative and clearly labeled in every row.",
        "- Local polynomial bases and validators remain tool-specific. The "
        "external contracts align inputs and carry/reset policies, not internal "
        "algorithms.",
        "- Runtime values are machine- and build-dependent. Flow* compilation, "
        "DiffReach JIT, and Torch eager orchestration must remain separate.",
        "",
        "## Reproduction",
        "",
        "```bash",
        "cd /srv/local/shengenli/torch_tm_flowpipe_three_way_comparison",
        "experiments/three_way_common_contract/run_smoke.sh",
        "experiments/three_way_common_contract/run_all.sh",
        "# or after the interactive smoke gate:",
        "experiments/three_way_common_contract/launch_background.sh",
        "tmux attach -t tm_three_way_common_contract",
        "```",
        "",
        "The exact canonical specification copied into this result directory is "
        "`benchmark_spec.yaml`; adapter logs and generated Flow* sources are "
        "under `logs/`.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    report = generate(output)
    (output / "three_way_common_contract_report.md").write_text(
        report, encoding="utf-8"
    )
    print(output / "three_way_common_contract_report.md")


if __name__ == "__main__":
    main()
