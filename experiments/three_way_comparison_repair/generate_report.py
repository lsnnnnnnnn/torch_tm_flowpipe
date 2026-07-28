#!/usr/bin/env python3
"""Generate the detailed technical report from validated repair artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

HERE = Path(__file__).resolve().parent


def _table(frame: pd.DataFrame, columns: list[str] | None = None) -> str:
    if frame.empty:
        return "_No rows._"
    selected = frame if columns is None else frame[columns]
    headers = [str(value) for value in selected.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in selected.itertuples(index=False, name=None):
        lines.append(
            "| "
            + " | ".join(
                str(value).replace("|", "\\|").replace("\n", " ") for value in row
            )
            + " |"
        )
    return "\n".join(lines)


def _value(
    raw: pd.DataFrame,
    *,
    tool: str,
    variant: str,
    kind: str,
    system: str = "riccati",
    h: float = 0.01,
    field: str = "width",
) -> float | None:
    data = raw[
        (raw.tool == tool)
        & (raw.tool_variant == variant)
        & (raw.interval_kind == kind)
        & (raw.system == system)
        & (raw.h == h)
        & (raw.step_index == 1)
        & (raw.state_index == 0)
    ]
    return None if data.empty else float(data.iloc[0][field])


def _fmt(value: float | None) -> str:
    return "not available" if value is None else f"{value:.12g}"


def generate(output: Path) -> tuple[str, str]:
    raw = pd.read_csv(output / "raw_results.csv", low_memory=False)
    for column in (
        "h",
        "width",
        "lower",
        "upper",
        "lower_error",
        "upper_error",
        "remainder_width",
    ):
        raw[column] = pd.to_numeric(raw[column], errors="coerce")
    checks = json.loads(
        (output / "correctness_checks.json").read_text(encoding="utf-8")
    )
    parity = json.loads(
        (output / "flowstar_original_parity_summary.json").read_text(
            encoding="utf-8"
        )
    )
    historical = json.loads(
        (output / "historical_reproduction.json").read_text(encoding="utf-8")
    )
    repository = json.loads(
        (output / "repository_summary.json").read_text(encoding="utf-8")
    )
    torch_raw = _value(
        raw,
        tool="torch_tm_flowpipe",
        variant="torch_order1",
        kind="endpoint_raw",
    )
    torch_tight = _value(
        raw,
        tool="torch_tm_flowpipe",
        variant="torch_order1",
        kind="endpoint_tightened",
    )
    flow_stock = _value(
        raw, tool="flowstar", variant="flowstar_stock", kind="endpoint_raw"
    )
    flow_candidate = _value(
        raw,
        tool="flowstar",
        variant="flowstar_candidate_reinjection_diagnostic",
        kind="endpoint_raw",
    )
    stock_upper_error = _value(
        raw,
        tool="flowstar",
        variant="flowstar_stock",
        kind="endpoint_raw",
        field="upper_error",
    )
    revalidated = _value(
        raw,
        tool="flowstar",
        variant="flowstar_refinement_revalidated_diagnostic",
        kind="endpoint_raw",
    )
    failure_frame = pd.read_csv(output / "corrected_failure_horizon_summary.csv")
    flow_failures = failure_frame[
        (failure_frame.tool == "flowstar")
        & failure_frame.failure_category.fillna("").ne("")
    ]
    claim_frame = pd.read_csv(output / "claim_audit.csv")
    one_step = pd.read_csv(output / "corrected_one_step_summary.csv")
    valid_two_way = one_step[
        one_step.tool.isin(["torch_tm_flowpipe", "diffreach"])
        & (one_step.interval_kind == "endpoint_raw")
        & (one_step.analytic_reference_status != "failed")
    ]
    outcome = checks["outcome"]
    executive = "\n".join(
        [
            "# Executive summary",
            "",
            f"The repair selects **Outcome {outcome}**.",
            "",
            "The historical three-way ranking is invalid. The generated Flow* "
            "adapter overwrote the native refined remainder after every successful "
            "`advance`, and Torch's displayed endpoint used an additional fixed-time "
            "residual tightening that Flow* and DiffReach did not use.",
            "",
            f"Stock Flow* reproduces the Riccati under-enclosure: at h=0.01 its "
            f"upper endpoint misses the analytic upper bound by {_fmt(stock_upper_error)}. "
            "A regenerated full-Picard inclusion test rejects the remainder-only "
            "refinement image. The original Flow* Van der Pol benchmark nonetheless "
            f"reaches T=10 with {parity['original_segments']} segments, and both "
            "identical-settings generated harnesses reproduce its schedule.",
            "",
            "Accordingly, the report publishes the semantics-corrected Torch versus "
            "DiffReach rows, keeps Flow* stock and diagnostics visible, and issues no "
            "three-way width ranking.",
            "",
        ]
    )
    sections = [
        "# Three-way comparison correctness repair",
        "",
        "## 1. Executive summary",
        "",
        executive.replace("# Executive summary\n\n", ""),
        "## 2. Why the previous experiment was invalid",
        "",
        "The old figures mixed endpoint semantics and changed one solver's output "
        "after validation. Candidate reinjection was presented as an extraction "
        "workaround, Torch `final_tm` was a fixed-time residual recomputation, the "
        "Flow* wrapper reduced all failures to one integer, and fixed low order was "
        "mistaken for general tool capability. Common-box resets also removed the "
        "dependencies that native carry is designed to preserve.",
        "",
        "## 3. Exact code-level confounders",
        "",
        "- **Flow* remainder overwrite (confirmed code fact):** the historical "
        "generated C++ assigned `setting.tm_setting.remainder_estimation[state]` "
        "to `next.tmvPre.tms[state].remainder` after `advance`.",
        "- **Torch endpoint tightening (confirmed code fact):** `flowpipe_step_from_tm` "
        "re-evaluated the Picard residual at `tau=h` and stored that result in "
        "`final_tm`. The repaired API exposes raw and tightened endpoint TMs.",
        "- **Generic Flow* failure reporting (confirmed code fact):** return code 0 "
        "lost the failing inclusion state and source site. The audit build emits "
        "structured return reasons and refinement rounds.",
        "- **Low-order configuration (confirmed code fact):** the old Flow* row used "
        "fixed order 2 even though the upstream benchmark uses adaptive steps and "
        "order 4.",
        "- **Reset semantics (inference from code paths):** common-box carry "
        "restarts every native representation from an axis-aligned box.",
        "",
        "## 4. Historical result reproduction",
        "",
        f"Top-level historical artifact regeneration status: "
        f"`{historical['status']}`. Report match: "
        f"`{historical['report_sha_match']}`; plot matches: "
        f"{historical['plot_sha_matches']}/{historical['plot_count']}. The frozen "
        "directory itself was never used as an output directory.",
        "",
        "## 5. Flow* original benchmark parity",
        "",
        f"The actual local upstream benchmark reached T=10: "
        f"`{parity['original_reached_horizon_10']}`. Original/generated/generic "
        f"segment counts are {parity['original_segments']}/"
        f"{parity['generated_segments']}/{parity['generic_segments']}. "
        f"Schedule agreement is `{parity['schedule_agreement']}` and generated "
        f"versus generic bound agreement is "
        f"`{parity['generated_vs_generic_bound_agreement']}`.",
        "",
        "## 6. Flow* stock refinement investigation",
        "",
        f"At Riccati h=0.01 the stock raw endpoint width is {_fmt(flow_stock)}; "
        f"candidate reinjection produces {_fmt(flow_candidate)}. The stock upper "
        f"miss is {_fmt(stock_upper_error)}. The diagnostic that fully revalidates "
        f"the refined remainder returns width {_fmt(revalidated)} and contains the "
        "analytic endpoint.",
        "",
        "The first source-level failing operation is the remainder-only refinement "
        "acceptance. Its final scalar remainder is not a self-map when "
        "`Picard_ctrunc_normal` and the polynomial-difference interval are regenerated. "
        "The audit trace records `subset=0` and restores the already accepted initial "
        "remainder. This is a conservative diagnostic fallback, not a merged upstream fix.",
        "",
        "Repeating the base run with `intervalNumPrecision=256` produces the "
        "same first-step upper bound to the exported precision and therefore "
        "does not remove the violation. This is evidence against default "
        "53-bit numeric rounding as the first cause.",
        "",
        "## 7. Exact Flow* failure classification",
        "",
        _table(
            flow_failures.head(30),
            [
                "tool_variant",
                "protocol",
                "system",
                "h",
                "successful_horizon",
                "failure_category",
                "failure_message",
            ],
        ),
        "",
        "Every observed failure has a structured category. `unknown_internal_failure` "
        "is retained only when no instrumented return record exists.",
        "",
        "## 8. Torch endpoint semantics",
        "",
        "For each validated segment, `endpoint_raw_tm` is the direct substitution "
        "of `tau=h` in the validated segment. `endpoint_tightened_tm` uses the "
        "fixed-time residual formula described in `TORCH_ENDPOINT_AUDIT.md`. "
        "`final_tm` remains the tightened endpoint for backward compatibility.",
        "",
        f"Riccati h=0.01 raw width: **{_fmt(torch_raw)}**. Tightened width: "
        f"**{_fmt(torch_tight)}**. The primary protocols use only the former.",
        "",
        "## 9. DiffReach endpoint semantics",
        "",
        "The adapter invokes the saved upstream `CT_Dyn_Reach.step_once`, including "
        "upstream Picard construction, remainder refinement, and symbolic carry. It "
        "composes the returned local TM with the upstream parameterization, evaluates "
        "the full time box for the tube, and fixes time to h for the raw endpoint. "
        "There is no endpoint-specific residual recomputation. The sole numeric "
        "override changes the upstream float32 constructor default to explicit x64.",
        "",
        "## 10. Corrected comparison protocols",
        "",
        "- **A — one_step_tube:** identical ODE, initial box, and h; full segment.",
        "- **B — one_step_raw_endpoint:** direct h-substitution in each validated segment.",
        "- **C — common_box_raw_endpoint_carry:** only raw endpoint boxes are carried.",
        "- **D — native_representation:** stock native carry; Torch raw and legacy "
        "tightened carry are distinct variants.",
        "- **E — deliberate_low_order_stress:** Torch order 1, DiffReach affine, "
        "Flow* minimum legal order 2; diagnostic only.",
        "- **F — known_working_tool_sanity:** original Flow* Van der Pol configuration.",
        "",
        "## 11. Corrected one-step tube results",
        "",
        "See `corrected_one_step_summary.csv`. Only rows passing their applicable "
        "correctness gates are interpretable; stock Flow* Riccati rows are retained "
        "as failed audit evidence.",
        "",
        "## 12. Corrected raw-endpoint results",
        "",
        _table(
            valid_two_way.head(30),
            [
                "tool",
                "tool_variant",
                "system",
                "h",
                "state_index",
                "lower",
                "upper",
                "width",
                "inflation_ratio",
            ],
        ),
        "",
        "## 13. Corrected common-box-carry results",
        "",
        "See `corrected_common_time_summary.csv`. Widths are compared only at equal "
        "absolute time. Failed segments contribute no later points.",
        "",
        "## 14. Native-representation results",
        "",
        "Native rows are configuration-specific and do not constitute a common-basis "
        "ranking. Torch raw-endpoint carry and legacy tightened-endpoint carry are "
        "separate variants.",
        "",
        "## 15. Deliberate low-order stress results",
        "",
        "These rows intentionally use different legal minima and are never described "
        "as same-order performance.",
        "",
        "## 16. Runtime results with implementation caveats",
        "",
        "Flow* compile time, DiffReach JIT time, first execution, and steady per-step "
        "time are separate schema fields. No combined winner is reported.",
        "",
        "## 17. Numerical soundness differences",
        "",
        "Flow* uses directed MPFR interval arithmetic, Torch uses float64 tensor "
        "interval operations, and DiffReach uses JAX float64 interval-style arithmetic. "
        "Analytic references are proofs for the two closed-form systems; deterministic "
        "Van der Pol trajectories are bug-catching checks, not proofs.",
        "",
        "## 18. Claim-by-claim correction",
        "",
        _table(
            claim_frame,
            ["old_claim", "status", "confounder", "corrected_wording"],
        ),
        "",
        "## 19. Confirmed facts",
        "",
        "- The historical Flow* adapter overwrote native remainders.",
        "- Torch fixed-time tightening materially changes Riccati width.",
        "- The original Flow* benchmark reaches T=10 in the audited build.",
        "- Stock Flow* refined Riccati under-enclosure is reproducible.",
        "- Full-Picard revalidation rejects the remainder-only refined scalar image.",
        "",
        "## 20. Unresolved questions",
        "",
        "The audit isolates the invalid refinement acceptance but does not supply a "
        "general upstream proof or patch for every ODE/order/precision combination. "
        "The exact internal reason cached remainder-only evaluation diverges from a "
        "regenerated full Picard image remains an upstream algorithm question.",
        "",
        "## 21. Recommendation and decision",
        "",
        f"**Outcome {outcome}.** A valid three-way width ranking is not currently "
        "possible. Publish the corrected Torch-versus-DiffReach raw-semantic tables "
        "and the Flow* original-parity sanity result separately.",
        "",
        "## Correctness counts",
        "",
        "```json",
        json.dumps(checks["counts"], indent=2, sort_keys=True),
        "```",
        "",
        "## Repository provenance",
        "",
        "```json",
        json.dumps(repository, indent=2, sort_keys=True),
        "```",
        "",
        "## Reproduction",
        "",
        "```bash",
        "cd /srv/local/shengenli/torch_tm_flowpipe_three_way_repair",
        "export FLOWSTAR_ROOT=/srv/local/shengenli/flowstar_three_way_audit",
        "experiments/three_way_comparison_repair/run_smoke.sh",
        "experiments/three_way_comparison_repair/launch_background.sh",
        "```",
        "",
        "The tmux launcher prints the exact session, command, log, result path, "
        "progress command, and safe stop command.",
        "",
    ]
    return "\n".join(sections), executive


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    report, executive = generate(output)
    (output / "three_way_comparison_repair_report.md").write_text(
        report, encoding="utf-8"
    )
    (output / "executive_summary.md").write_text(executive, encoding="utf-8")
    print(output / "three_way_comparison_repair_report.md")


if __name__ == "__main__":
    main()
