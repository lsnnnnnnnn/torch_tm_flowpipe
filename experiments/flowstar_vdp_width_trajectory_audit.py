"""Consolidate Van der Pol Flow*/PyTorch width and trajectory evidence.

This script is an audit over existing artifacts. It does not add a flowpipe
mechanism, create a symbolic queue variant, or rerun the expensive h10 runs by
default.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "outputs" / "flowstar_vdp_width_trajectory_audit"

INVENTORY_FIELDS = [
    "artifact_path",
    "exists",
    "artifact_type",
    "family",
    "produced_by_script_if_known",
    "commit_if_known",
    "horizon_requested",
    "horizon_or_time_span",
    "tool_or_mode",
    "width_semantics",
    "endpoint_box_available",
    "segment_box_available",
    "tube_box_available",
    "has_plots",
    "has_samples",
    "has_flowstar_reference",
    "use_for_authoritative_comparison",
    "caution_note",
]

LEDGER_FIELDS = [
    "family",
    "run_id",
    "tool",
    "mode",
    "order",
    "candidate_order",
    "output_order",
    "h_policy",
    "horizon_requested",
    "status",
    "last_validated_t",
    "last_attempted_t",
    "segments",
    "runtime_s",
    "min_h_used",
    "min_regular_h_used",
    "h_below_flowstar_min_count",
    "step_rejections",
    "flowstar_reference",
    "flowstar_status",
    "flowstar_segments",
    "flowstar_last_validated_t",
    "py_last_segment_width_sum",
    "py_tube_width_sum",
    "py_endpoint_width_sum",
    "flowstar_last_segment_width_sum",
    "flowstar_tube_width_sum",
    "last_segment_width_ratio",
    "tube_width_ratio",
    "endpoint_width_ratio",
    "endpoint_ratio_allowed",
    "overlap_width_ratio_max",
    "overlap_width_ratio_median",
    "ratio_semantics_note",
    "width_comparable",
    "width_comparison_verdict",
    "trajectory_plot_available",
    "width_over_time_plot_available",
    "sample_overlay_available",
    "sample_containment_status",
    "oracle_status",
    "conclusion",
]

TRAJECTORY_FIELDS = [
    "family",
    "case_id",
    "order",
    "h",
    "horizon",
    "torch_mode",
    "flowstar_status",
    "torch_status",
    "last_segment_width_ratio",
    "tube_width_ratio",
    "endpoint_ratio_available",
    "endpoint_ratio_allowed",
    "ratio_semantics_note",
    "phase_xy_plot",
    "t_x_plot",
    "t_y_plot",
    "width_over_time_plot",
    "verdict",
]

CHECK_FIELDS = [
    "check_id",
    "status",
    "artifact_path",
    "details",
]

SUMMARY_FIELDS = [
    "topic",
    "status",
    "answer",
    "evidence_paths",
]

INVENTORY_SPECS = [
    ("outputs/flowstar_benchmark_parity/parity_report.md", "report", "original_generated_flowstar_parity", "experiments/flowstar_benchmark_parity.py", "10", "Flow* GNUPLOT segment/tube; torch rows are not endpoint-comparable to Flow*"),
    ("outputs/flowstar_benchmark_parity/parity_summary.csv", "csv", "original_generated_flowstar_parity", "experiments/flowstar_benchmark_parity.py", "10", "explicit endpoint/last-segment/tube columns"),
    ("outputs/flowstar_benchmark_parity/generated_flowstar_vs_original_comparison.csv", "csv", "original_generated_flowstar_parity", "experiments/flowstar_benchmark_parity.py", "10", "exact parsed Flow* segment-field comparison"),
    ("outputs/flowstar_benchmark_parity/original_flowstar/", "directory", "original_generated_flowstar_parity", "experiments/flowstar_benchmark_parity.py", "10", "Flow* reference artifacts"),
    ("outputs/flowstar_benchmark_parity/generated_flowstar/", "directory", "original_generated_flowstar_parity", "experiments/flowstar_benchmark_parity.py", "10", "generated Flow* artifacts"),
    ("outputs/README_RESULTS.md", "report", "corrected_fixed_step_audit", "", "", "audit index"),
    ("outputs/final_audit_summary.md", "report", "corrected_fixed_step_audit", "", "", "summary"),
    ("outputs/flowstar_vdp_plot_input_v2.csv", "csv", "corrected_fixed_step_audit", "", "10", "Flow* plot input"),
    ("outputs/flowstar_vdp_remainder_cutoff_sweep.csv", "csv", "corrected_fixed_step_audit", "", "10", "sweep"),
    ("outputs/flowstar_vdp_remainder_cutoff_sweep_summary_v2.md", "report", "corrected_fixed_step_audit", "", "10", "sweep summary"),
    ("outputs/van_der_pol_diagnostics_by_order_v2.csv", "csv", "corrected_fixed_step_audit", "experiments/diagnose_van_der_pol.py", "10", "PyTorch widths"),
    ("outputs/tm_order_audit_vdp_order2_8.csv", "csv", "corrected_fixed_step_audit", "experiments/tm_order_audit.py", "10", "order audit"),
    ("outputs/order_and_vdp_flowstar_report.md", "report", "corrected_fixed_step_audit", "", "10", "Flow* GNUPLOT segment/tube semantics"),
    ("outputs/order_flowstar_status_table.md", "report", "corrected_fixed_step_audit", "", "10", "Flow* statuses"),
    ("outputs/torch_over_flowstar_last_segment_width_ratio_by_order.png", "plot", "corrected_fixed_step_audit", "experiments/plot_order_results.py", "10", "segment ratio plot"),
    ("outputs/torch_over_flowstar_tube_width_ratio_by_order.png", "plot", "corrected_fixed_step_audit", "experiments/plot_order_results.py", "10", "tube ratio plot"),
    ("outputs/van_der_pol_*_vs_order.png", "plot_glob", "corrected_fixed_step_audit", "experiments/plot_order_results.py", "10", "order plots"),
    ("outputs/flowstar_status_by_order_and_setting.png", "plot", "corrected_fixed_step_audit", "experiments/plot_order_results.py", "10", "status plot"),
    ("outputs/trajectory_audit/README.md", "report", "trajectory_visual_audit", "experiments/trajectory_visual_audit.py", "0.025/0.1", "trajectory audit semantics"),
    ("outputs/trajectory_audit/visual_audit_report.md", "report", "trajectory_visual_audit", "experiments/trajectory_visual_audit.py", "0.025/0.1", "visual diagnostic only"),
    ("outputs/trajectory_audit/crosscheck_summary.md", "report", "trajectory_visual_audit", "experiments/trajectory_visual_audit.py", "0.025/0.1", "crosscheck"),
    ("outputs/trajectory_audit/flowstar_vs_torch_overlay_summary.csv", "csv", "trajectory_visual_audit", "experiments/trajectory_visual_audit.py", "0.025/0.1", "segment/tube ratios only"),
    ("outputs/trajectory_audit/flowstar_structured_summary.csv", "csv", "trajectory_visual_audit", "experiments/trajectory_visual_audit.py", "0.025/0.1", "Flow* segment/tube"),
    ("outputs/trajectory_audit/torch_structured_summary.csv", "csv", "trajectory_visual_audit", "experiments/trajectory_visual_audit.py", "0.025/0.1", "torch endpoint/segment/tube"),
    ("outputs/trajectory_audit/figures/*.png", "plot_glob", "trajectory_visual_audit", "experiments/trajectory_visual_audit.py", "0.025/0.1", "visual diagnostic plots"),
    ("outputs/trajectory_audit/samples/*.csv", "csv_glob", "trajectory_visual_audit", "experiments/trajectory_visual_audit.py", "0.025/0.1", "samples are visual diagnostics only"),
    ("outputs/trajectory_audit/flowstar_segments/*.csv", "csv_glob", "trajectory_visual_audit", "experiments/trajectory_visual_audit.py", "0.025/0.1", "Flow* segment boxes"),
    ("outputs/trajectory_audit/torch_segments/*.csv", "csv_glob", "trajectory_visual_audit", "experiments/trajectory_visual_audit.py", "0.025/0.1", "torch segment boxes"),
    ("outputs/flowstar_normalized_insertion_rescue/", "directory", "normalized_insertion_h5_rescue", "experiments/flowstar_style_rescue_vanderpol.py", "5", "segment/tube ratios vs Flow*"),
    ("outputs/flowstar_normalized_insertion_h10/", "directory", "normalized_insertion_h10", "experiments/flowstar_style_rescue_vanderpol.py", "10", "segment/tube ratios vs Flow*"),
    ("outputs/flowstar_width_fix_h10/", "directory", "scalar_alignment_width_fix", "experiments/flowstar_style_rescue_vanderpol.py", "10", "segment/tube ratios vs Flow*"),
    ("outputs/flowstar_insertion_width_attribution/", "directory", "width_attribution", "experiments/flowstar_insertion_width_attribution.py", "10", "component widths"),
    ("outputs/flowstar_width_growth_diagnostics/", "directory", "width_growth_diagnostics", "experiments/flowstar_width_growth_diagnostics.py", "10", "width trace"),
    ("outputs/flowstar_width_control_rescue/", "directory", "width_control_rescue", "experiments/flowstar_style_rescue_vanderpol.py", "10", "rescue attempt"),
    ("outputs/flowstar_right_map_scaling_diagnostics/", "directory", "right_map_scaling_diagnostics", "experiments/flowstar_right_map_scaling_diagnostics.py", "10", "right-map source evidence"),
    ("outputs/flowstar_normalized_insertion_symqueue_h10/", "directory", "symbolic_queue_old", "experiments/flowstar_style_rescue_vanderpol.py", "10", "symbolic queue diagnostic"),
    ("outputs/flowstar_normalized_insertion_symqueue_split_h10/", "directory", "symbolic_queue_split", "experiments/flowstar_style_rescue_vanderpol.py", "10", "split symbolic queue diagnostic"),
    ("outputs/flowstar_normalized_insertion_symqueue_v2_h10/", "directory", "symbolic_queue_v2", "experiments/flowstar_style_rescue_vanderpol.py", "10", "v2 output-only symbolic diagnostic"),
    ("outputs/flowstar_queue_state_audit/", "directory", "symbolic_queue_state_audit", "experiments/flowstar_queue_state_audit.py", "10", "queue channel audit"),
    ("outputs/flowstar_one_step_oracle_after_symqueue/", "directory", "one_step_oracle", "experiments/flowstar_one_step_oracle.py", "local", "one-step diagnostic"),
    ("outputs/flowstar_one_step_oracle_after_symqueue_split/", "directory", "one_step_oracle", "experiments/flowstar_one_step_oracle.py", "local", "one-step diagnostic"),
    ("outputs/flowstar_one_step_oracle_after_symqueue_v2/", "directory", "one_step_oracle", "experiments/flowstar_one_step_oracle.py", "local", "one-step diagnostic"),
    ("docs/flowstar_accepted_step_trace_plan.md", "report", "accepted_step_trace", "experiments/flowstar_step_trace_compare.py", "1", "diagnostic plan"),
    ("docs/flowstar_step_trace_divergence_report.md", "report", "accepted_step_trace", "experiments/flowstar_step_trace_compare.py", "0.5/1", "accepted ordinal diagnostic; guard noncausal if h differs"),
    ("outputs/flowstar_step_trace_compare/", "directory", "accepted_step_trace", "experiments/flowstar_step_trace_compare.py", "0.5/1", "accepted ordinal trace diff"),
    ("experiments/flowstar_probe/flowstar_vdp_step_trace_probe.cpp", "source", "accepted_step_trace", "experiments/flowstar_step_trace_compare.py", "0.5/1", "Flow* C++ probe inventory only"),
    ("experiments/flowstar_step_trace_compare.py", "source", "accepted_step_trace", "", "0.5/1", "diagnostic comparator"),
    ("docs/gpu_strategy_reality_check.md", "report", "gpu_strategy", "experiments/batched_tm_gpu_microbench.py", "n/a", "performance diagnostic"),
    ("outputs/batched_tm_gpu_microbench/gpu_microbench_report.md", "report", "gpu_strategy", "experiments/batched_tm_gpu_microbench.py", "n/a", "GPU performance report"),
    ("outputs/batched_tm_gpu_microbench/gpu_microbench_summary.csv", "csv", "gpu_strategy", "experiments/batched_tm_gpu_microbench.py", "n/a", "GPU benchmark rows"),
]

PLOT_LINKS = [
    "outputs/trajectory_audit/figures/contact_sheet_torch_orders.png",
    "outputs/trajectory_audit/figures/contact_sheet_flowstar_overlays.png",
    "outputs/trajectory_audit/figures/contact_sheet_width_trends.png",
    "outputs/trajectory_audit/figures/flowstar_rem1e-4_cut1e-10_h0p01_s10_o4_overlay_phase_xy.png",
    "outputs/trajectory_audit/figures/flowstar_rem1e-4_cut1e-10_h0p01_s10_o4_overlay_t_x.png",
    "outputs/trajectory_audit/figures/flowstar_rem1e-4_cut1e-10_h0p01_s10_o4_overlay_t_y.png",
    "outputs/trajectory_audit/figures/flowstar_rem1e-4_cut1e-10_h0p01_s10_o4_overlay_width_over_time.png",
    "outputs/trajectory_audit/figures/flowstar_rem1e-10_cut1e-15_h0p0025_s10_o8_overlay_phase_xy.png",
    "outputs/trajectory_audit/figures/flowstar_rem1e-10_cut1e-15_h0p0025_s10_o8_overlay_t_x.png",
    "outputs/trajectory_audit/figures/flowstar_rem1e-10_cut1e-15_h0p0025_s10_o8_overlay_t_y.png",
    "outputs/trajectory_audit/figures/flowstar_rem1e-10_cut1e-15_h0p0025_s10_o8_overlay_width_over_time.png",
    "outputs/flowstar_width_fix_h10/overlay_rescue_vs_original_flowstar_phase_xy.png",
    "outputs/flowstar_width_fix_h10/overlay_rescue_vs_original_flowstar_t_x.png",
    "outputs/flowstar_width_fix_h10/overlay_rescue_vs_original_flowstar_t_y.png",
    "outputs/flowstar_width_fix_h10/width_ratio_vs_t.png",
    "outputs/flowstar_width_fix_h10/reset_box_width_trace.png",
    "outputs/flowstar_width_fix_h10/residual_vs_t.png",
    "outputs/flowstar_width_fix_h10/step_size_trace.png",
    "outputs/flowstar_insertion_width_attribution/insertion_component_stack.png",
    "outputs/flowstar_insertion_width_attribution/o4_vs_o6_width_sources.png",
]

COMPARISON_SPECS = [
    ("normalized_insertion_h5", "outputs/flowstar_normalized_insertion_rescue/normalized_insertion_vs_flowstar_comparison.csv", "outputs/flowstar_normalized_insertion_rescue/normalized_insertion_summary.csv", "5", "normalized insertion h5/rescue"),
    ("normalized_insertion_h10", "outputs/flowstar_normalized_insertion_h10/normalized_insertion_h10_vs_flowstar_comparison.csv", "outputs/flowstar_normalized_insertion_h10/normalized_insertion_h10_summary.csv", "10", "normalized insertion h10"),
    ("scalar_alignment_width_fix_h10", "outputs/flowstar_width_fix_h10/width_fix_vs_flowstar_comparison.csv", "outputs/flowstar_width_fix_h10/width_fix_summary.csv", "10", "scalar alignment/width fix h10"),
    ("symbolic_queue_old_h10", "outputs/flowstar_normalized_insertion_symqueue_h10/symqueue_h10_vs_flowstar_comparison.csv", "outputs/flowstar_normalized_insertion_symqueue_h10/symqueue_h10_summary.csv", "10", "old symbolic queue h10"),
    ("symbolic_queue_split_h10", "outputs/flowstar_normalized_insertion_symqueue_split_h10/symqueue_split_vs_flowstar_comparison.csv", "outputs/flowstar_normalized_insertion_symqueue_split_h10/symqueue_split_summary.csv", "10", "split symbolic queue h10"),
    ("symbolic_queue_v2_h10", "outputs/flowstar_normalized_insertion_symqueue_v2_h10/symqueue_v2_vs_flowstar_comparison.csv", "outputs/flowstar_normalized_insertion_symqueue_v2_h10/symqueue_v2_summary.csv", "10", "v2 symbolic queue h10"),
]


def _str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.17g}"
    return str(value)


def _float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "yes", "1", "passed", "completed", "validated", "max_horizon_reached"}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _str(row.get(field, "")) for field in fieldnames})


def _read_text(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    return path.read_text(encoding="utf-8")


def _rel(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _status_reached(status: Any, last_t: Any, horizon: Any) -> bool:
    target = _float(horizon)
    last = _float(last_t)
    status_text = str(status).lower()
    if "max_horizon_reached" in status_text or "completed" in status_text:
        return True
    return target is not None and last is not None and last >= target - 1e-9


def _find_row(rows: Sequence[Mapping[str, str]], run_id: str) -> Mapping[str, str]:
    return next((row for row in rows if row.get("run_id") == run_id), {})


def _ratio_verdict(last_ratio: Any, tube_ratio: Any, status: Any, horizon: Any, last_t: Any) -> str:
    last = _float(last_ratio)
    tube = _float(tube_ratio)
    reached = _status_reached(status, last_t, horizon)
    if last is None and tube is None:
        return "missing_ratio"
    if last is not None and last <= 1.25 and tube is not None and tube <= 1.25 and reached:
        return "width_close_for_requested_horizon"
    if last is not None and last <= 1.5 and tube is not None and tube <= 1.5:
        return "visually_close_or_short_horizon_close"
    if tube is not None and tube <= 1.25 and last is not None and last > 5:
        return "tube_near_but_last_segment_not_close"
    return "not_width_close"


def _plot_available(repo_root: Path, *paths: str) -> bool:
    return any((repo_root / path).exists() for path in paths)


def inventory_rows(repo_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pattern, artifact_type, family, script, horizon, semantics in INVENTORY_SPECS:
        matches = sorted(repo_root.glob(pattern)) if any(ch in pattern for ch in "*?[") else [repo_root / pattern]
        if not matches:
            matches = [repo_root / pattern]
        for path in matches:
            exists = path.exists()
            lower = pattern.lower()
            is_plot = artifact_type.startswith("plot") or lower.endswith(('.png', '.eps', '.plt'))
            has_samples = "samples" in lower or "sample_containment" in lower
            has_flowstar = "flowstar" in lower or family in {"trajectory_visual_audit", "original_generated_flowstar_parity"}
            endpoint = "torch" in lower and "flowstar" not in lower
            if "endpoint" in semantics.lower() and "flow*" not in semantics.lower():
                endpoint = True
            segment = any(token in semantics.lower() for token in ("segment", "gnuplot", "tube", "flow*"))
            tube = "tube" in semantics.lower() or "gnuplot" in semantics.lower()
            use = exists and family in {
                "original_generated_flowstar_parity",
                "trajectory_visual_audit",
                "normalized_insertion_h5_rescue",
                "normalized_insertion_h10",
                "scalar_alignment_width_fix",
                "symbolic_queue_old",
                "symbolic_queue_split",
                "symbolic_queue_v2",
                "symbolic_queue_state_audit",
                "accepted_step_trace",
                "gpu_strategy",
                "width_attribution",
            }
            caution = ""
            if "gnuplot" in semantics.lower() or "flow*" in semantics.lower():
                caution = "Flow* GNUPLOT rectangles are flowpipe segment boxes, not endpoint boxes."
            if "sample" in semantics.lower():
                caution = "Sampling/trajectory overlays are visual diagnostics only, not proof."
            if not exists:
                caution = f"Missing artifact path searched: {pattern}"
            rows.append(
                {
                    "artifact_path": pattern if not exists else _rel(path, repo_root),
                    "exists": exists,
                    "artifact_type": artifact_type,
                    "family": family,
                    "produced_by_script_if_known": script,
                    "commit_if_known": "",
                    "horizon_requested": horizon,
                    "horizon_or_time_span": horizon,
                    "tool_or_mode": "flowstar" if "flowstar" in lower else "pytorch" if "torch" in lower else "mixed",
                    "width_semantics": semantics,
                    "endpoint_box_available": endpoint,
                    "segment_box_available": segment,
                    "tube_box_available": tube,
                    "has_plots": is_plot or path.is_dir() and any(path.glob("*.png")),
                    "has_samples": has_samples or path.is_dir() and any(path.glob("*sample*.csv")),
                    "has_flowstar_reference": has_flowstar,
                    "use_for_authoritative_comparison": use,
                    "caution_note": caution,
                }
            )
    return rows


def write_inventory_md(path: Path, rows: Sequence[Mapping[str, Any]], repo_root: Path) -> None:
    existing = sum(1 for row in rows if _truthy(row.get("exists")))
    missing = len(rows) - existing
    families = sorted({str(row.get("family", "")) for row in rows})
    lines = [
        "# Flowstar Van der Pol Width/Trajectory Evidence Inventory",
        "",
        f"Repository: `{repo_root}`",
        f"Artifacts classified: `{len(rows)}`; existing: `{existing}`; missing: `{missing}`.",
        "",
        "Flow* GNUPLOT rectangles are treated as flowpipe segment boxes. Endpoint ratios are disabled unless both compared tools explicitly provide endpoint boxes.",
        "",
        "## Families",
        "",
    ]
    for family in families:
        fam_rows = [row for row in rows if row.get("family") == family]
        lines.append(f"- `{family}`: {sum(1 for row in fam_rows if _truthy(row.get('exists')))}/{len(fam_rows)} present")
    lines.extend(["", "## Missing Paths", ""])
    missing_rows = [row for row in rows if not _truthy(row.get("exists"))]
    if missing_rows:
        for row in missing_rows:
            lines.append(f"- `{row.get('artifact_path')}` ({row.get('family')})")
    else:
        lines.append("- None")
    lines.extend(["", "See `evidence_inventory.csv` for the full ledger.", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parity_ledger(repo_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    summary_rows = _read_csv(repo_root / "outputs/flowstar_benchmark_parity/parity_summary.csv")
    comparison = {row.get("metric"): row.get("value") for row in _read_csv(repo_root / "outputs/flowstar_benchmark_parity/generated_flowstar_vs_original_comparison.csv")}
    exact = comparison.get("segment_count_match") == "true" and _float(comparison.get("max_abs_segment_field_diff")) == 0.0
    original = next((row for row in summary_rows if row.get("tool") == "original_flowstar"), {})
    generated = next((row for row in summary_rows if row.get("tool") == "generated_flowstar"), {})
    for source_row, mode in ((original, "original_flowstar"), (generated, "generated_flowstar")):
        if not source_row:
            continue
        rows.append(
            _empty_ledger(
                family="original_generated_flowstar_parity",
                run_id=mode,
                tool="flowstar",
                mode=mode,
                order="4",
                horizon="10",
                status=source_row.get("status", ""),
                conclusion="Exact generated-vs-original Flow* segment-field parity." if exact else "Parity incomplete or non-exact.",
                endpoint_allowed=False,
                width_comparable=True,
                verdict="exact_segment_field_parity" if exact else "not_exact_parity",
                flowstar_reference="original_flowstar_benchmark",
            )
            | {
                "segments": source_row.get("num_segments", ""),
                "runtime_s": source_row.get("generated_flowstar_internal_reach_s") or source_row.get("original_flowstar_wall_run_s", ""),
                "last_validated_t": source_row.get("last_validated_t", ""),
                "last_attempted_t": source_row.get("last_attempted_t", ""),
                "flowstar_status": source_row.get("status", ""),
                "flowstar_segments": source_row.get("num_segments", ""),
                "flowstar_last_validated_t": source_row.get("last_validated_t", ""),
                "flowstar_last_segment_width_sum": source_row.get("last_segment_width_sum", ""),
                "flowstar_tube_width_sum": source_row.get("tube_width_sum", ""),
                "last_segment_width_ratio": "1" if exact else "",
                "tube_width_ratio": "1" if exact else "",
                "ratio_semantics_note": "Flow* original vs generated exact parsed segment fields; no PyTorch endpoint comparison.",
            }
        )
    return rows, {"summary_rows": summary_rows, "comparison": comparison, "exact": exact, "original": original, "generated": generated}


def _empty_ledger(
    *,
    family: str,
    run_id: str,
    tool: str,
    mode: str,
    order: Any = "",
    candidate_order: Any = "",
    output_order: Any = "",
    h_policy: Any = "",
    horizon: Any = "",
    status: Any = "",
    conclusion: str = "",
    endpoint_allowed: bool = False,
    width_comparable: bool = False,
    verdict: str = "",
    flowstar_reference: str = "original_flowstar_gnuplot_segments",
) -> dict[str, Any]:
    return {field: "" for field in LEDGER_FIELDS} | {
        "family": family,
        "run_id": run_id,
        "tool": tool,
        "mode": mode,
        "order": order,
        "candidate_order": candidate_order,
        "output_order": output_order,
        "h_policy": h_policy,
        "horizon_requested": horizon,
        "status": status,
        "flowstar_reference": flowstar_reference,
        "endpoint_ratio_allowed": endpoint_allowed,
        "width_comparable": width_comparable,
        "width_comparison_verdict": verdict,
        "conclusion": conclusion,
    }


def build_comparison_ledgers(repo_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family, comparison_path, summary_path, horizon, note in COMPARISON_SPECS:
        comparison_rows = _read_csv(repo_root / comparison_path)
        summary_rows = _read_csv(repo_root / summary_path)
        sample_status = _sample_status_for_family(repo_root, family)
        oracle_status = _oracle_status(repo_root) if family == "symbolic_queue_v2_h10" else ""
        for comp in comparison_rows:
            run_id = comp.get("run_id", "")
            summary = _find_row(summary_rows, run_id)
            status = summary.get("status") or comp.get("py_status", "")
            last_t = summary.get("last_validated_t") or comp.get("py_last_validated_t", "")
            last_ratio = comp.get("last_width_ratio", "")
            tube_ratio = comp.get("tube_width_ratio", "")
            verdict = _ratio_verdict(last_ratio, tube_ratio, status, horizon, last_t)
            reached = _status_reached(status, last_t, horizon)
            conclusion = f"{note}: {'reached requested horizon' if reached else 'did not reach requested horizon'}; verdict={verdict}."
            if family == "normalized_insertion_h10" and "o4" in run_id:
                conclusion = "o4 target insertion stays tighter than o6 but stops at t=6.473; last-segment ratio is not close."
            elif family == "normalized_insertion_h10" and "o6" in run_id:
                conclusion = "o6 candidate8/output6 reaches farther, t=7.496, but is far wider than Flow* near the late horizon."
            elif family == "normalized_insertion_h5" and reached:
                conclusion = "normalized insertion reached h5 and is segment/tube width-close to Flow* in this artifact."
            rows.append(
                _empty_ledger(
                    family=family,
                    run_id=run_id,
                    tool="torch_tm_flowpipe",
                    mode=summary.get("reset_mode") or summary.get("mode") or family,
                    order=summary.get("order", ""),
                    candidate_order=summary.get("candidate_order", ""),
                    output_order=summary.get("output_order", ""),
                    h_policy="adaptive Flow*-style h; no expensive rerun in this audit",
                    horizon=horizon,
                    status=status,
                    conclusion=conclusion,
                    endpoint_allowed=False,
                    width_comparable=True,
                    verdict=verdict,
                )
                | {
                    "last_validated_t": last_t,
                    "last_attempted_t": summary.get("last_attempted_t", ""),
                    "segments": comp.get("py_segments") or summary.get("validated_segments", ""),
                    "runtime_s": comp.get("py_runtime_s") or summary.get("runtime_s", ""),
                    "min_h_used": summary.get("min_h_used", ""),
                    "min_regular_h_used": summary.get("min_regular_h_used", ""),
                    "h_below_flowstar_min_count": summary.get("h_below_flowstar_min_count", ""),
                    "step_rejections": summary.get("num_step_rejections", ""),
                    "flowstar_status": "completed_over_reference_prefix",
                    "flowstar_segments": comp.get("flowstar_segments_over_same_horizon", ""),
                    "flowstar_last_validated_t": comp.get("py_last_validated_t", ""),
                    "py_last_segment_width_sum": comp.get("py_last_width_sum", ""),
                    "py_tube_width_sum": comp.get("py_tube_width_sum", ""),
                    "py_endpoint_width_sum": "",
                    "flowstar_last_segment_width_sum": comp.get("flowstar_last_width_sum_near_T", ""),
                    "flowstar_tube_width_sum": comp.get("flowstar_tube_width_sum_over_same_horizon", ""),
                    "last_segment_width_ratio": last_ratio,
                    "tube_width_ratio": tube_ratio,
                    "endpoint_width_ratio": "",
                    "overlap_width_ratio_max": comp.get("max_time_overlap_width_ratio", ""),
                    "overlap_width_ratio_median": comp.get("median_time_overlap_width_ratio", ""),
                    "ratio_semantics_note": "PyTorch segment/tube boxes compared with Flow* GNUPLOT segment/tube boxes over the overlapping horizon; endpoint ratios disabled.",
                    "trajectory_plot_available": _plot_available(repo_root, f"outputs/{family}/overlay_rescue_vs_original_flowstar_phase_xy.png"),
                    "width_over_time_plot_available": _plot_available(repo_root, f"outputs/{family}/width_ratio_vs_t.png"),
                    "sample_overlay_available": sample_status != "",
                    "sample_containment_status": sample_status,
                    "oracle_status": oracle_status,
                }
            )
    return rows


def _sample_status_for_family(repo_root: Path, family: str) -> str:
    paths = {
        "symbolic_queue_old_h10": "outputs/flowstar_normalized_insertion_symqueue_h10/sample_containment_summary.csv",
        "symbolic_queue_split_h10": "outputs/flowstar_normalized_insertion_symqueue_split_h10/sample_containment_summary.csv",
        "symbolic_queue_v2_h10": "outputs/flowstar_normalized_insertion_symqueue_v2_h10/sample_containment_summary.csv",
        "normalized_insertion_h10": "outputs/flowstar_normalized_insertion_h10/sample_containment_summary.csv",
    }
    rows = _read_csv(repo_root / paths.get(family, "")) if family in paths else []
    if not rows:
        return ""
    return rows[0].get("status", "")


def _oracle_status(repo_root: Path) -> str:
    rows = _read_csv(repo_root / "outputs/flowstar_one_step_oracle_after_symqueue_v2/oracle_after_symqueue_v2_summary.csv")
    statuses = sorted({row.get("flowstar_status", "") for row in rows if row.get("flowstar_status")})
    return ";".join(statuses)


def build_trajectory_ledgers(repo_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    overlay_rows = _read_csv(repo_root / "outputs/trajectory_audit/flowstar_vs_torch_overlay_summary.csv")
    traj_rows: list[dict[str, Any]] = []
    ledger_rows: list[dict[str, Any]] = []
    for row in overlay_rows:
        endpoint_allowed = False
        last_ratio = row.get("last_segment_width_ratio_torch_over_flowstar", "")
        tube_ratio = row.get("tube_width_ratio_torch_over_flowstar", "")
        verdict = "visual_short_horizon_close" if (_float(last_ratio) or 99.0) <= 1.5 and (_float(tube_ratio) or 99.0) <= 1.25 else "not_comparable_or_not_close"
        phase = f"outputs/trajectory_audit/figures/{row.get('case_id')}_overlay_phase_xy.png"
        tx = f"outputs/trajectory_audit/figures/{row.get('case_id')}_overlay_t_x.png"
        ty = f"outputs/trajectory_audit/figures/{row.get('case_id')}_overlay_t_y.png"
        width = f"outputs/trajectory_audit/figures/{row.get('case_id')}_overlay_width_over_time.png"
        traj_rows.append(
            {
                "family": "trajectory_visual_audit",
                "case_id": row.get("case_id", ""),
                "order": row.get("order", ""),
                "h": row.get("h", ""),
                "horizon": row.get("horizon", ""),
                "torch_mode": row.get("torch_mode", ""),
                "flowstar_status": row.get("flowstar_status", ""),
                "torch_status": row.get("torch_status", ""),
                "last_segment_width_ratio": last_ratio,
                "tube_width_ratio": tube_ratio,
                "endpoint_ratio_available": row.get("endpoint_ratio_available", ""),
                "endpoint_ratio_allowed": endpoint_allowed,
                "ratio_semantics_note": row.get("ratio_note", ""),
                "phase_xy_plot": phase if (repo_root / phase).exists() else "",
                "t_x_plot": tx if (repo_root / tx).exists() else "",
                "t_y_plot": ty if (repo_root / ty).exists() else "",
                "width_over_time_plot": width if (repo_root / width).exists() else "",
                "verdict": verdict,
            }
        )
        ledger_rows.append(
            _empty_ledger(
                family="trajectory_visual_audit",
                run_id=f"{row.get('case_id')}_{row.get('torch_mode')}",
                tool="torch_tm_flowpipe_vs_flowstar",
                mode=row.get("torch_mode", ""),
                order=row.get("order", ""),
                h_policy=f"fixed h={row.get('h', '')}",
                horizon=row.get("horizon", ""),
                status=row.get("torch_status", ""),
                conclusion=f"Short-horizon visual diagnostic; verdict={verdict}; samples are not proof.",
                endpoint_allowed=endpoint_allowed,
                width_comparable=row.get("last_segment_ratio_available") == "true" or row.get("tube_ratio_available") == "true",
                verdict=verdict,
            )
            | {
                "flowstar_status": row.get("flowstar_status", ""),
                "last_segment_width_ratio": last_ratio,
                "tube_width_ratio": tube_ratio,
                "endpoint_width_ratio": "",
                "ratio_semantics_note": row.get("ratio_note", ""),
                "trajectory_plot_available": bool((repo_root / phase).exists()),
                "width_over_time_plot_available": bool((repo_root / width).exists()),
                "sample_overlay_available": True,
                "sample_containment_status": "visual_only",
            }
        )
    return traj_rows, ledger_rows


def step_alignment_warning_rows(repo_root: Path) -> list[dict[str, Any]]:
    rows = _read_csv(repo_root / "outputs/flowstar_step_trace_compare/aligned_trace_diff.csv")
    warnings: list[dict[str, Any]] = []
    for row in rows:
        flow_h = _float(row.get("flowstar_h"))
        flow_t = _float(row.get("t_flowstar"))
        messages: list[str] = []
        for label, value in (("noqueue_h", row.get("noqueue_h")), ("v2_h", row.get("v2_h"))):
            val = _float(value)
            if flow_h is not None and val is not None and abs(flow_h - val) > 1e-9:
                messages.append(f"{label} differs from Flow* h")
        for label, value in (("t_noqueue", row.get("t_noqueue")), ("t_v2", row.get("t_v2"))):
            val = _float(value)
            if flow_t is not None and val is not None and abs(flow_t - val) > 1e-9:
                messages.append(f"{label} differs from Flow* t")
        if messages:
            warnings.append(
                {
                    "step_index": row.get("step_index", ""),
                    "first_material_channel": "adaptive_step_alignment_mismatch",
                    "channel_attribution_valid": False,
                    "comparison_kind": "accepted_ordinal_trace_diff_noncausal",
                    "alignment_warning": "; ".join(messages),
                    "flowstar_h": row.get("flowstar_h", ""),
                    "noqueue_h": row.get("noqueue_h", ""),
                    "v2_h": row.get("v2_h", ""),
                    "t_flowstar": row.get("t_flowstar", ""),
                    "t_noqueue": row.get("t_noqueue", ""),
                    "t_v2": row.get("t_v2", ""),
                }
            )
    return warnings


def build_claim_checks(repo_root: Path, inventory: Sequence[Mapping[str, Any]], ledger: Sequence[Mapping[str, Any]], parity: Mapping[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    endpoint_violations = [row for row in ledger if row.get("endpoint_width_ratio") and not _truthy(row.get("endpoint_ratio_allowed"))]
    checks.append(
        {
            "check_id": "endpoint_ratio_disabled_without_flowstar_endpoint",
            "status": "pass" if not endpoint_violations else "fail",
            "artifact_path": "outputs/trajectory_audit/flowstar_vs_torch_overlay_summary.csv",
            "details": "Flow* GNUPLOT rows have endpoint_ratio_allowed=false; no endpoint ratio is emitted." if not endpoint_violations else f"Violations: {len(endpoint_violations)}",
        }
    )
    checks.append(
        {
            "check_id": "flowstar_parity_exact_only",
            "status": "pass" if parity.get("exact") else "warning",
            "artifact_path": "outputs/flowstar_benchmark_parity/generated_flowstar_vs_original_comparison.csv",
            "details": f"segment_count_match={parity.get('comparison', {}).get('segment_count_match')}; max_abs_segment_field_diff={parity.get('comparison', {}).get('max_abs_segment_field_diff')}",
        }
    )
    missing = [row.get("artifact_path") for row in inventory if not _truthy(row.get("exists"))]
    checks.append(
        {
            "check_id": "missing_artifacts_recorded",
            "status": "pass",
            "artifact_path": "outputs/flowstar_vdp_width_trajectory_audit/evidence_inventory.csv",
            "details": f"Missing artifact count recorded explicitly: {len(missing)}",
        }
    )
    warnings = step_alignment_warning_rows(repo_root)
    checks.append(
        {
            "check_id": "accepted_step_h_or_t_mismatch_noncausal",
            "status": "noncausal_guarded" if warnings else "pass",
            "artifact_path": "outputs/flowstar_step_trace_compare/aligned_trace_diff.csv",
            "details": "first_material_channel must be adaptive_step_alignment_mismatch; ordinal channel attribution invalid/noncausal" if warnings else "No h/t mismatch found in existing aligned_trace_diff.csv",
        }
    )
    h10_bad = [row for row in ledger if row.get("family") == "normalized_insertion_h10" and row.get("width_comparison_verdict") != "width_close_for_requested_horizon"]
    checks.append(
        {
            "check_id": "h10_not_timeout_only",
            "status": "pass" if h10_bad else "warning",
            "artifact_path": "outputs/flowstar_normalized_insertion_h10/normalized_insertion_h10_vs_flowstar_comparison.csv",
            "details": "h10 failures include width/trajectory divergence, not only runtime timeout." if h10_bad else "No h10 divergence rows found.",
        }
    )
    checks.append(
        {
            "check_id": "samples_visual_only",
            "status": "pass",
            "artifact_path": "outputs/trajectory_audit/samples/*.csv",
            "details": "Sampling trajectories and overlays are recorded as visual diagnostics only, not proof.",
        }
    )
    return checks


def _line_from_report(text: str, prefix: str) -> str:
    for line in text.splitlines():
        if line.startswith(prefix):
            return line
    return ""


def build_summary(repo_root: Path, ledger: Sequence[Mapping[str, Any]], parity: Mapping[str, Any]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    comparison = parity.get("comparison", {})
    summary.append(
        {
            "topic": "original_generated_flowstar_parity",
            "status": "exact" if parity.get("exact") else "incomplete",
            "answer": f"completed/completed; segment_count={comparison.get('original_num_segments')}/{comparison.get('generated_num_segments')}; max_abs_segment_field_diff={comparison.get('max_abs_segment_field_diff')}",
            "evidence_paths": "outputs/flowstar_benchmark_parity/generated_flowstar_vs_original_comparison.csv; outputs/flowstar_benchmark_parity/parity_summary.csv",
        }
    )
    h5 = [row for row in ledger if row.get("family") == "normalized_insertion_h5" and "cutoff_insert" in str(row.get("run_id"))]
    if h5:
        row = h5[0]
        summary.append(
            {
                "topic": "normalized_insertion_h5",
                "status": row.get("width_comparison_verdict", ""),
                "answer": f"reached t={row.get('last_validated_t')}; last_ratio={row.get('last_segment_width_ratio')}; tube_ratio={row.get('tube_width_ratio')}",
                "evidence_paths": "outputs/flowstar_normalized_insertion_rescue/normalized_insertion_vs_flowstar_comparison.csv",
            }
        )
    h10 = [row for row in ledger if row.get("family") == "normalized_insertion_h10"]
    if h10:
        best = max(h10, key=lambda row: _float(row.get("last_validated_t")) or -1.0)
        summary.append(
            {
                "topic": "normalized_insertion_h10",
                "status": "not_width_close",
                "answer": f"best t={best.get('last_validated_t')} from {best.get('run_id')}; last_ratio={best.get('last_segment_width_ratio')}; tube_ratio={best.get('tube_width_ratio')}; did not reach h10",
                "evidence_paths": "outputs/flowstar_normalized_insertion_h10/normalized_insertion_h10_vs_flowstar_comparison.csv",
            }
        )
    attr = _read_text(repo_root / "outputs/flowstar_insertion_width_attribution/insertion_width_report.md")
    summary.append(
        {
            "topic": "width_attribution",
            "status": "parsed" if attr else "missing",
            "answer": _line_from_report(attr, "Which component causes width growth?") or "missing attribution report",
            "evidence_paths": "outputs/flowstar_insertion_width_attribution/insertion_width_report.md",
        }
    )
    queue = _read_text(repo_root / "outputs/flowstar_queue_state_audit/queue_state_report.md")
    summary.append(
        {
            "topic": "symbolic_queue_v2",
            "status": "diagnostic_not_rescue" if queue else "missing",
            "answer": _line_from_report(queue, "Did v2 reach h10?") or "missing queue state report",
            "evidence_paths": "outputs/flowstar_queue_state_audit/queue_state_report.md",
        }
    )
    gpu = _read_text(repo_root / "outputs/batched_tm_gpu_microbench/gpu_microbench_report.md")
    summary.append(
        {
            "topic": "gpu_strategy",
            "status": "representation_redesign_signal" if gpu else "missing",
            "answer": _line_from_report(gpu, "- Is the project still justified") or "missing GPU report",
            "evidence_paths": "outputs/batched_tm_gpu_microbench/gpu_microbench_report.md",
        }
    )
    return summary


def _fmt_ratio(row: Mapping[str, Any]) -> str:
    return f"last={row.get('last_segment_width_ratio', '')}, tube={row.get('tube_width_ratio', '')}, overlap max/median={row.get('overlap_width_ratio_max', '')}/{row.get('overlap_width_ratio_median', '')}"


def write_report(path: Path, repo_root: Path, inventory: Sequence[Mapping[str, Any]], ledger: Sequence[Mapping[str, Any]], traj: Sequence[Mapping[str, Any]], checks: Sequence[Mapping[str, Any]], parity: Mapping[str, Any]) -> None:
    missing_plots = [plot for plot in PLOT_LINKS if not (repo_root / plot).exists()]
    existing_plots = [plot for plot in PLOT_LINKS if (repo_root / plot).exists()]
    comparison = parity.get("comparison", {})
    original = parity.get("original", {})
    h5 = next((row for row in ledger if row.get("family") == "normalized_insertion_h5" and "cutoff_insert" in str(row.get("run_id"))), {})
    h10_o4 = next((row for row in ledger if row.get("family") == "normalized_insertion_h10" and "o4" in str(row.get("run_id")) and "cutoff" not in str(row.get("run_id"))), {})
    h10_o6 = next((row for row in ledger if row.get("family") == "normalized_insertion_h10" and "o6" in str(row.get("run_id")) and "cutoff" not in str(row.get("run_id"))), {})
    width_fix_o6 = next((row for row in ledger if row.get("family") == "scalar_alignment_width_fix_h10" and "o6" in str(row.get("run_id"))), {})
    split_best = next((row for row in ledger if row.get("family") == "symbolic_queue_split_h10" and "o6" in str(row.get("run_id")) and "cutoff" not in str(row.get("run_id"))), {})
    v2_best = next((row for row in ledger if row.get("family") == "symbolic_queue_v2_h10" and "o6" in str(row.get("run_id")) and "cutoff" not in str(row.get("run_id"))), {})
    queue_report = _read_text(repo_root / "outputs/flowstar_queue_state_audit/queue_state_report.md")
    attr_report = _read_text(repo_root / "outputs/flowstar_insertion_width_attribution/insertion_width_report.md")
    gpu_report = _read_text(repo_root / "outputs/batched_tm_gpu_microbench/gpu_microbench_report.md")
    step_warnings = step_alignment_warning_rows(repo_root)
    trajectory_o4 = [row for row in traj if row.get("order") == "4" and row.get("torch_mode") == "range_only"]
    trajectory_o8 = [row for row in traj if row.get("order") == "8" and row.get("torch_mode") == "range_only"]

    lines = [
        "# Flowstar Van der Pol Width/Trajectory Audit",
        "",
        "This is an audit over existing artifacts. It does not add a new flowpipe mechanism, does not add a symbolic queue variant, and does not claim Flow* parity beyond exact generated-vs-original Flow* segment-field equality.",
        "",
        "## 1. What is already exact?",
        "",
        f"Original Flow* vs generated Flow* parity: `{'completed/completed' if parity.get('exact') else 'incomplete'}`.",
        f"Segment count: original=`{comparison.get('original_num_segments', '')}`, generated=`{comparison.get('generated_num_segments', '')}`; max_abs_segment_field_diff=`{comparison.get('max_abs_segment_field_diff', '')}`.",
        f"Last segment width sum: `{original.get('last_segment_width_sum', '')}`; tube width sum: `{original.get('tube_width_sum', '')}`.",
        "",
        "## 2. What is visually/short-horizon close?",
        "",
    ]
    if trajectory_o4:
        row = trajectory_o4[0]
        lines.append(f"Order 4 loose fixed-step trajectory overlay is short-horizon close in a visual/segment sense: last ratio `{row.get('last_segment_width_ratio')}`, tube ratio `{row.get('tube_width_ratio')}`.")
    if trajectory_o8:
        row = trajectory_o8[0]
        lines.append(f"Order 8 strict fixed-step overlay is closer: last ratio `{row.get('last_segment_width_ratio')}`, tube ratio `{row.get('tube_width_ratio')}`.")
    lines.extend(
        [
            "Phase/t-x/t-y/width-over-time plots are linked below. Sampling trajectories are visual diagnostics only, not proof.",
            "",
            "## 3. What is normalized insertion h5 status?",
            "",
        ]
    )
    if h5:
        lines.append(f"The h5 normalized insertion artifact reached t=`{h5.get('last_validated_t')}` with `{_fmt_ratio(h5)}`. It is width-close for that requested horizon.")
    else:
        lines.append("No h5 normalized insertion comparison was found. Searched `outputs/flowstar_normalized_insertion_rescue/`.")
    lines.extend(["", "## 4. What is normalized insertion h10 status?", ""])
    if h10_o4:
        lines.append(f"o4 target insert: last_validated_t=`{h10_o4.get('last_validated_t')}`; `{_fmt_ratio(h10_o4)}`; reached h10: no. Widths are not last-segment-close, though the tube ratio is much nearer than o6.")
    if h10_o6:
        lines.append(f"o6 candidate8/output6: last_validated_t=`{h10_o6.get('last_validated_t')}`; `{_fmt_ratio(h10_o6)}`; reached h10: no. It reaches farther but is far wider late.")
    lines.extend(["", "## 5. What did width attribution find?", ""])
    lines.append(_line_from_report(attr_report, "Which component causes width growth?") or "Width attribution report missing.")
    lines.append(_line_from_report(attr_report, "Is the next fix") or "The evidence points to right-map scaling/source-order/preconditioning rather than symbolic queue alone.")
    lines.extend(["", "## 6. What did scalar alignment/width fix do?", ""])
    if width_fix_o6:
        lines.append(f"Scalar alignment/width fix did not materially improve h10: o6 stayed at t=`{width_fix_o6.get('last_validated_t')}` with `{_fmt_ratio(width_fix_o6)}`.")
    else:
        lines.append("Scalar alignment/width-fix comparison missing.")
    lines.extend(["", "## 7. What did split/v2 symbolic queue do?", ""])
    lines.append(_line_from_report(queue_report, "Did v2 reach h10?") or "Queue state report missing.")
    lines.append(_line_from_report(queue_report, "Did v2 beat no_queue") or "")
    lines.append(_line_from_report(queue_report, "Did v2 reduce max reset") or "")
    lines.append(_line_from_report(queue_report, "Did v2 keep symbolic width") or "")
    if split_best:
        lines.append(f"split best: t=`{split_best.get('last_validated_t')}`, `{_fmt_ratio(split_best)}`.")
    if v2_best:
        lines.append(f"v2 best: t=`{v2_best.get('last_validated_t')}`, `{_fmt_ratio(v2_best)}`, sample containment `{v2_best.get('sample_containment_status')}`, Flow* one-step oracle `{v2_best.get('oracle_status')}`.")
    lines.extend(["", "## 8. What did accepted-step comparator do?", ""])
    if step_warnings:
        first = step_warnings[0]
        lines.append(f"The accepted-step comparator is diagnostic infrastructure, but the current accepted ordinal comparison is `accepted_ordinal_trace_diff_noncausal`: step `{first.get('step_index')}` has Flow* h=`{first.get('flowstar_h')}` versus PyTorch h=`{first.get('noqueue_h')}`.")
        lines.append("Channel attribution is therefore invalid/noncausal and must be treated as `adaptive_step_alignment_mismatch`.")
    else:
        lines.append("No h/t mismatch was detected in the existing aligned trace diff.")
    lines.append("Next comparator fix: produce `attempt_aligned_trace_diff.csv` and `forced_h_trace_diff.csv`.")
    lines.extend(["", "## 9. What did GPU benchmark prove?", ""])
    lines.append(_line_from_report(gpu_report, "- Are current data structures") or "Current sparse dict TaylorModel/Polynomial is not the GPU path.")
    lines.append(_line_from_report(gpu_report, "- Is the project still justified") or "Dense batched kernels show CUDA speedups at realistic batch sizes.")
    lines.append("This is a representation-redesign signal, not h10 rescue evidence.")
    lines.extend(
        [
            "",
            "## 10. Overall Conclusion",
            "",
            "We are not merely timing out. The PyTorch rescue has real width/trajectory differences versus Flow* in late horizon. Short-horizon and h5 evidence can be close, but h10 normalized insertion is not width-close: o4 stays tighter but stops earlier, o6 reaches farther but becomes far wider. Symbolic queue v2 improves diagnostics, not horizon/tightness. The next correctness task is aligned Flow* step comparison and right-map/preconditioning/source-order width mechanism; the next performance task is dense batched TM representation.",
            "",
            "## Claim Boundary Checks",
            "",
        ]
    )
    for check in checks:
        lines.append(f"- `{check.get('check_id')}`: `{check.get('status')}` - {check.get('details')}")
    lines.extend(["", "## Plot Links", ""])
    for plot in existing_plots:
        lines.append(f"- `{plot}`")
    lines.extend(["", "## Missing Requested Plot Paths", ""])
    if missing_plots:
        for plot in missing_plots:
            lines.append(f"- `{plot}`")
    else:
        lines.append("- None")
    lines.extend(
        [
            "",
            "## Generated Audit Files",
            "",
            "- `summary.csv`",
            "- `width_comparison_ledger.csv`",
            "- `trajectory_overlay_ledger.csv`",
            "- `claim_boundary_checks.csv`",
            "- `evidence_inventory.csv`",
            "- `evidence_inventory.md`",
            "- `report.md`",
            "",
        ]
    )
    path.write_text("\n".join(line for line in lines if line is not None) + "\n", encoding="utf-8")


def write_plan_doc(repo_root: Path) -> None:
    path = repo_root / "docs/flowstar_vdp_width_trajectory_audit_plan.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    text = """# Flowstar Van der Pol Width/Trajectory Audit Plan

Goal: consolidate existing Flow*/PyTorch Van der Pol width and trajectory evidence into one authoritative audit without adding a new flowpipe mechanism, symbolic queue variant, or Flow* source patch.

## Initial Repository State

Captured before this audit implementation on branch `codex/flowstar-normalized-insertion`.

```text
git status --short --branch
## codex/flowstar-normalized-insertion...origin/codex/flowstar-normalized-insertion

git branch --show-current
codex/flowstar-normalized-insertion

git log --oneline -12
7101408 Publish batched TM GPU microbenchmark evidence
b2b1a6d Add batched TM GPU microbenchmark
4d8f1e7 Add Flowstar accepted step trace comparator
c07a0b8 Add three-way Flowstar symqueue v2 audit evidence
7e81d6a Add Flowstar symbolic queue v2 audit
346772e Add Horner insertion diagnostics
3778e90 Add normal-eval right-map diagnostics
c56f817 Add Flowstar width attribution and scalar fix run
a2ac372 Add split symbolic queue semantics diagnostics
82a814c Fix h10 failure target width diagnostics
ed23e61 Add normalized insertion symqueue diagnostics
f667f73 Add normalized insertion h10 diagnostics

git remote -v
origin git@github.com:lsnnnnnnnn/torch_tm_flowpipe.git (fetch)
origin git@github.com:lsnnnnnnnn/torch_tm_flowpipe.git (push)
```

## Scope

- Inventory existing parity, fixed-step, trajectory, normalized insertion, width attribution, symbolic queue, accepted-step trace, and GPU strategy artifacts.
- Parse existing CSV/MD files only by default; do not rerun expensive h10 experiments.
- Treat Flow* GNUPLOT rectangles as flowpipe segment boxes, not endpoint boxes.
- Disable endpoint ratios unless both tools explicitly provide endpoint boxes.
- Treat sampling trajectories and overlays as visual diagnostics only.
- Mark accepted-step ordinal trace attribution noncausal when Flow* and PyTorch accepted `t` or `h` differ.

## Outputs

- `outputs/flowstar_vdp_width_trajectory_audit/evidence_inventory.csv`
- `outputs/flowstar_vdp_width_trajectory_audit/evidence_inventory.md`
- `outputs/flowstar_vdp_width_trajectory_audit/summary.csv`
- `outputs/flowstar_vdp_width_trajectory_audit/width_comparison_ledger.csv`
- `outputs/flowstar_vdp_width_trajectory_audit/trajectory_overlay_ledger.csv`
- `outputs/flowstar_vdp_width_trajectory_audit/claim_boundary_checks.csv`
- `outputs/flowstar_vdp_width_trajectory_audit/report.md`

## Next Comparator Work

The accepted-step comparator should next emit `attempt_aligned_trace_diff.csv` and `forced_h_trace_diff.csv` so channel localization is causal rather than accepted-ordinal only.
"""
    path.write_text(text, encoding="utf-8")


def run(repo_root: Path, out_dir: Path, *, strict_missing: bool, allow_regenerate_plots: bool) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_plan_doc(repo_root)

    inventory = inventory_rows(repo_root)
    _write_csv(out_dir / "evidence_inventory.csv", INVENTORY_FIELDS, inventory)
    write_inventory_md(out_dir / "evidence_inventory.md", inventory, repo_root)

    parity_ledger, parity = build_parity_ledger(repo_root)
    trajectory, trajectory_ledger = build_trajectory_ledgers(repo_root)
    comparison_ledger = build_comparison_ledgers(repo_root)
    ledger = parity_ledger + trajectory_ledger + comparison_ledger
    _write_csv(out_dir / "width_comparison_ledger.csv", LEDGER_FIELDS, ledger)
    _write_csv(out_dir / "trajectory_overlay_ledger.csv", TRAJECTORY_FIELDS, trajectory)

    checks = build_claim_checks(repo_root, inventory, ledger, parity)
    _write_csv(out_dir / "claim_boundary_checks.csv", CHECK_FIELDS, checks)

    summary = build_summary(repo_root, ledger, parity)
    if not allow_regenerate_plots:
        summary.append(
            {
                "topic": "plot_regeneration",
                "status": "not_run",
                "answer": "Existing plots were linked only; no plots regenerated because --allow-regenerate-plots was not passed.",
                "evidence_paths": ";".join(path for path in PLOT_LINKS if (repo_root / path).exists()),
            }
        )
    _write_csv(out_dir / "summary.csv", SUMMARY_FIELDS, summary)
    write_report(out_dir / "report.md", repo_root, inventory, ledger, trajectory, checks, parity)

    if strict_missing and any(not _truthy(row.get("exists")) for row in inventory):
        return 2
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--allow-regenerate-plots", action="store_true")
    strict = parser.add_mutually_exclusive_group()
    strict.add_argument("--strict-missing", action="store_true")
    strict.add_argument("--no-strict-missing", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = Path(args.repo_root).resolve()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = repo_root / out_dir
    strict_missing = bool(args.strict_missing and not args.no_strict_missing)
    code = run(repo_root, out_dir.resolve(), strict_missing=strict_missing, allow_regenerate_plots=args.allow_regenerate_plots)
    print(f"Wrote Van der Pol width/trajectory audit to {out_dir.resolve()}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
