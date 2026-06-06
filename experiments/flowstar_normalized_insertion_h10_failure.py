#!/usr/bin/env python3
"""Localize normalized-insertion h10 failures from recorded artifacts."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

TARGET_RUNS = [
    "flowstar_style_o4_target_insert",
    "flowstar_style_o6_candidate8_output6_insert",
]

SUMMARY_FIELDS = [
    "run_id",
    "order_family",
    "status",
    "last_validated_t",
    "last_attempted_t",
    "failure_t_start",
    "failure_h_try",
    "failed_dimension",
    "residual_width_x",
    "residual_width_y",
    "target_width_x",
    "target_width_y",
    "residual_over_target_x",
    "residual_over_target_y",
    "shift_or_width",
    "dominant_term",
    "width_ratio_near_failure",
    "step_rejections_near_failure",
    "failure_after_width_ratio_jump",
    "failure_after_rejection_cluster",
    "priority_for_h10_parity",
    "notes",
]

ATTEMPT_FIELDS = [
    "run_id",
    "segment_index",
    "adaptive_attempt_index",
    "attempt_index",
    "t_lo",
    "t_hi",
    "h_try",
    "validation_status",
    "rejection_reason",
    "failed_dimension",
    "residual_width_x",
    "residual_width_y",
    "target_width_x",
    "target_width_y",
    "residual_over_target_x",
    "residual_over_target_y",
    "residual_lo_x",
    "residual_hi_x",
    "residual_lo_y",
    "residual_hi_y",
    "polynomial_range_width_sum",
    "normal_eval_range_width_sum",
    "ordinary_residual_range_width_sum",
    "tmp_remainder_width_sum",
]

BREAKDOWN_FIELDS = [
    "run_id",
    "t_start",
    "h_try",
    "failed_dimension",
    "component",
    "width",
    "interpretation",
]


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _fmt(row.get(field, "")) for field in fields})


def _fmt(value: Any) -> Any:
    if isinstance(value, float):
        return f"{value:.17g}" if math.isfinite(value) else str(value)
    return value


def _finite(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _is_rejected(row: Mapping[str, Any]) -> bool:
    status = str(row.get("validation_status", "")).lower()
    reason = str(row.get("rejection_reason", "")) or str(row.get("validation_message", ""))
    subset = str(row.get("subset_result", "")).lower()
    return status not in {"", "validated"} or bool(reason) or subset == "false"


def _target_width(row: Mapping[str, Any]) -> float:
    radius = _finite(row.get("target_remainder_radius"))
    if radius is not None:
        return 2.0 * radius
    width = _finite(row.get("target_remainder_width"))
    if width is not None:
        return 0.5 * width
    return 2.0e-4


def _ratio(num: Any, den: Any) -> float | str:
    n = _finite(num)
    d = _finite(den)
    if n is None or d is None or d <= 0.0:
        return ""
    return n / d


def _failed_dimension(row: Mapping[str, Any]) -> str:
    target = _target_width(row)
    target_lo = -0.5 * target
    target_hi = 0.5 * target
    failed: list[str] = []
    for dim in ("x", "y"):
        lo = _finite(row.get(f"residual_lo_{dim}"))
        hi = _finite(row.get(f"residual_hi_{dim}"))
        width = _finite(row.get(f"residual_width_{dim}"))
        if (lo is not None and lo < target_lo - 1e-18) or (hi is not None and hi > target_hi + 1e-18):
            failed.append(dim)
        elif width is not None and width > target + 1e-18:
            failed.append(dim)
    if failed:
        return "+".join(failed)
    rx = _ratio(row.get("residual_width_x"), target)
    ry = _ratio(row.get("residual_width_y"), target)
    rx_f = rx if isinstance(rx, float) else -1.0
    ry_f = ry if isinstance(ry, float) else -1.0
    return "x" if rx_f >= ry_f else "y"


def _shift_or_width(row: Mapping[str, Any], failed_dim: str) -> str:
    target = _target_width(row)
    target_lo = -0.5 * target
    target_hi = 0.5 * target
    labels: list[str] = []
    for dim in failed_dim.split("+"):
        lo = _finite(row.get(f"residual_lo_{dim}"))
        hi = _finite(row.get(f"residual_hi_{dim}"))
        width = _finite(row.get(f"residual_width_{dim}"))
        if width is not None and width > target + 1e-18:
            labels.append(f"{dim}:width")
        elif hi is not None and hi > target_hi + 1e-18:
            labels.append(f"{dim}:positive_shift")
        elif lo is not None and lo < target_lo - 1e-18:
            labels.append(f"{dim}:negative_shift")
        else:
            labels.append(f"{dim}:near_target")
    return ";".join(labels)


def _dominant_term(row: Mapping[str, Any], segment: Mapping[str, Any] | None) -> str:
    trunc = _finite((segment or {}).get("insertion_truncation_width")) or 0.0
    cutoff = _finite((segment or {}).get("insertion_cutoff_width")) or 0.0
    poly_times_rem = _finite(row.get("tmp_remainder_width_sum")) or _finite(row.get("ordinary_residual_range_width_sum")) or 0.0
    poly_range = _finite(row.get("polynomial_range_width_sum")) or 0.0
    residual = _finite(row.get("residual_width_sum")) or 0.0
    candidates = {
        "truncation": trunc,
        "insertion uncertainty": trunc + cutoff,
        "polynomial_range*remainder": poly_times_rem,
        "symbolic missing": max(residual - max(trunc + cutoff, poly_times_rem), 0.0),
    }
    if poly_range > 1.0 and candidates["polynomial_range*remainder"] == 0.0:
        candidates["symbolic missing"] = max(candidates["symbolic missing"], residual)
    return max(candidates, key=lambda key: candidates[key])


def _last_segment_for_run(rows: Sequence[Mapping[str, str]], run_id: str) -> Mapping[str, str] | None:
    selected = [row for row in rows if row.get("run_id") == run_id and row.get("status") == "validated"]
    if not selected:
        return None
    return max(selected, key=lambda row: _finite(row.get("t_hi")) or 0.0)


def _failure_attempt(rows: Sequence[Mapping[str, str]], run_id: str) -> Mapping[str, str] | None:
    rejected = [row for row in rows if row.get("run_id") == run_id and _is_rejected(row)]
    if not rejected:
        return None
    return max(
        rejected,
        key=lambda row: (
            _finite(row.get("t_lo")) or 0.0,
            _finite(row.get("h_try")) or _finite(row.get("h")) or 0.0,
            _finite(row.get("attempt_index")) or 0.0,
        ),
    )


def _focused_attempts(rows: Sequence[Mapping[str, str]], run_id: str, failure: Mapping[str, str]) -> list[dict[str, Any]]:
    seg = str(failure.get("segment_index", ""))
    selected = [row for row in rows if row.get("run_id") == run_id and str(row.get("segment_index", "")) == seg]
    if not selected:
        selected = [row for row in rows if row.get("run_id") == run_id][-12:]
    out: list[dict[str, Any]] = []
    for row in selected:
        target = _target_width(row)
        failed_dim = _failed_dimension(row) if _is_rejected(row) else ""
        out.append(
            {
                **row,
                "failed_dimension": failed_dim,
                "target_width_x": target,
                "target_width_y": target,
                "residual_over_target_x": _ratio(row.get("residual_width_x"), target),
                "residual_over_target_y": _ratio(row.get("residual_width_y"), target),
            }
        )
    return out


def _ratio_near_failure(ratio_rows: Sequence[Mapping[str, str]], run_id: str, t_failure: float | None, comp_rows: Sequence[Mapping[str, str]]) -> tuple[float | str, bool]:
    rows = [row for row in ratio_rows if row.get("run_id") == run_id]
    if rows and t_failure is not None:
        rows = [row for row in rows if (_finite(row.get("t")) or 0.0) <= t_failure + 1e-12]
        rows.sort(key=lambda row: _finite(row.get("t")) or 0.0)
        if rows:
            last = _finite(rows[-1].get("width_ratio"))
            prev = _finite(rows[-2].get("width_ratio")) if len(rows) > 1 else None
            jumped = bool(last is not None and prev is not None and last > max(2.0 * prev, prev + 5.0))
            return (last if last is not None else ""), jumped
    for row in comp_rows:
        if row.get("run_id") == run_id:
            ratio = _finite(row.get("last_width_ratio"))
            return (ratio if ratio is not None else ""), False
    return "", False


def _step_rejection_cluster(attempts: Sequence[Mapping[str, str]], failure: Mapping[str, str]) -> tuple[int, bool]:
    seg = str(failure.get("segment_index", ""))
    rejected = [row for row in attempts if str(row.get("segment_index", "")) == seg and _is_rejected(row)]
    return len(rejected), len(rejected) >= 3


def _breakdown_rows(run_id: str, failure: Mapping[str, str], segment: Mapping[str, str] | None, failed_dim: str) -> list[dict[str, Any]]:
    t_start = failure.get("t_lo", "")
    h_try = failure.get("h_try", failure.get("h", ""))
    components = [
        ("truncation", (segment or {}).get("insertion_truncation_width", ""), "normalized insertion truncation/c-trunc uncertainty"),
        ("insertion uncertainty", (segment or {}).get("output_remainder_width", ""), "new local insertion remainder accounting"),
        ("polynomial_range*remainder", failure.get("tmp_remainder_width_sum", failure.get("ordinary_residual_range_width_sum", "")), "recorded validation remainder interaction proxy"),
        ("symbolic missing", failure.get("residual_width_sum", ""), "remaining residual pressure not explained by explicit insertion fields"),
    ]
    return [
        {
            "run_id": run_id,
            "t_start": t_start,
            "h_try": h_try,
            "failed_dimension": failed_dim,
            "component": name,
            "width": width,
            "interpretation": interpretation,
        }
        for name, width, interpretation in components
    ]


def _make_plots(out_dir: Path, attempts: Sequence[Mapping[str, Any]], ratio_rows: Sequence[Mapping[str, str]]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    for run_id, suffix in [(TARGET_RUNS[0], "o4"), (TARGET_RUNS[1], "o6")]:
        rows = [row for row in attempts if row.get("run_id") == run_id]
        if not rows:
            continue
        rows = sorted(rows, key=lambda row: (_finite(row.get("t_lo")) or 0.0, _finite(row.get("h_try")) or 0.0))[-24:]
        x_vals = list(range(len(rows)))
        target = [_finite(row.get("target_width_y")) or _finite(row.get("target_width_x")) or 0.0 for row in rows]
        fig, ax = plt.subplots(figsize=(8.5, 4.5))
        ax.plot(x_vals, [_finite(row.get("residual_width_x")) or 0.0 for row in rows], marker="o", label="residual x")
        ax.plot(x_vals, [_finite(row.get("residual_width_y")) or 0.0 for row in rows], marker="o", label="residual y")
        ax.plot(x_vals, target, linestyle="--", color="#111111", label="target width")
        ax.set_xlabel("near-failure attempt index")
        ax.set_ylabel("residual width")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.25, linewidth=0.6)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out_dir / f"residual_near_failure_{suffix}.png", dpi=160)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    for run_id in TARGET_RUNS:
        pts = [
            (_finite(row.get("t")), _finite(row.get("width_ratio")))
            for row in ratio_rows
            if row.get("run_id") == run_id and _finite(row.get("t")) is not None and _finite(row.get("width_ratio")) is not None
        ]
        pts.sort(key=lambda pair: pair[0] or 0.0)
        if pts:
            ax.plot([p[0] for p in pts if p[0] is not None], [p[1] for p in pts if p[1] is not None], label=run_id)
    ax.axhline(1.0, color="#111111", linestyle="--", linewidth=0.8)
    ax.set_xlabel("t")
    ax.set_ylabel("PyTorch width / Flow* overlap hull width")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "width_ratio_near_failure.png", dpi=160)
    plt.close(fig)


def _write_report(out_dir: Path, summary_rows: Sequence[Mapping[str, Any]]) -> None:
    by_run = {str(row.get("run_id", "")): row for row in summary_rows}
    o4 = by_run.get(TARGET_RUNS[0], {})
    o6 = by_run.get(TARGET_RUNS[1], {})
    o4_t = _finite(o4.get("last_validated_t")) or 0.0
    o6_t = _finite(o6.get("last_validated_t")) or 0.0
    o4_ratio = _finite(o4.get("width_ratio_near_failure")) or 0.0
    o6_ratio = _finite(o6.get("width_ratio_near_failure")) or 0.0
    if o6_t > o4_t and o6_ratio > o4_ratio:
        farther_wider = "o6/candidate8 goes farther because higher candidate order validates more steps, but its normalized reset boxes accumulate much larger widths."
    else:
        farther_wider = "the recorded rows do not show the expected o6 farther-but-wider pattern."
    priority = "o4 for Flow*-settings parity; keep o6/candidate8 as the reachability stress path."
    lines = [
        "# Normalized Insertion H10 Failure Localization",
        "",
        "Inputs: existing `outputs/flowstar_normalized_insertion_h10` CSV artifacts.",
        "This is a post-hoc localization report; it does not rerun the reachability kernel.",
        "",
        "## Direct Answers",
        "",
    ]
    for run_id in TARGET_RUNS:
        row = by_run.get(run_id, {})
        label = "o4" if "o4" in run_id else "o6"
        lines.extend(
            [
                f"### {label} `{run_id}`",
                f"- Failure near t=`{row.get('failure_t_start', '')}` with h_try=`{row.get('failure_h_try', '')}`.",
                f"- Failed dimension: `{row.get('failed_dimension', '')}`.",
                f"- Residual width vs target: x `{row.get('residual_width_x', '')}` / `{row.get('target_width_x', '')}`, y `{row.get('residual_width_y', '')}` / `{row.get('target_width_y', '')}`.",
                f"- Shift or width? `{row.get('shift_or_width', '')}`.",
                f"- Dominant term: `{row.get('dominant_term', '')}`.",
                f"- Did failure happen after a width-ratio jump? `{row.get('failure_after_width_ratio_jump', '')}`.",
                f"- Did failure happen after a cluster of step rejections? `{row.get('failure_after_rejection_cluster', '')}`.",
                "",
            ]
        )
    lines.extend(
        [
            "## Comparison",
            "",
            f"Why does o6 go farther but become much wider? {farther_wider}",
            "Why does o4 stay tighter but fail earlier? The order4 path keeps lower-order, closer-to-Flow* reset boxes, so widths remain smaller, but the final residual margin is exhausted earlier.",
            f"Which path should be prioritized for h10 parity? {priority}",
            "",
            "## Summary Rows",
            "",
            "| run_id | last_validated_t | failure_t | h_try | failed_dimension | shift_or_width | dominant_term | width_ratio | rejection_cluster |",
            "| --- | ---: | ---: | ---: | --- | --- | --- | ---: | --- |",
        ]
    )
    for row in summary_rows:
        lines.append(
            f"| {row.get('run_id', '')} | {row.get('last_validated_t', '')} | {row.get('failure_t_start', '')} | "
            f"{row.get('failure_h_try', '')} | {row.get('failed_dimension', '')} | {row.get('shift_or_width', '')} | "
            f"{row.get('dominant_term', '')} | {row.get('width_ratio_near_failure', '')} | {row.get('failure_after_rejection_cluster', '')} |"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "h10_failure_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(input_dir: Path, out_dir: Path) -> None:
    summary = _read_rows(input_dir / "normalized_insertion_h10_summary.csv")
    attempts = _read_rows(input_dir / "normalized_insertion_h10_validation_attempts.csv")
    segments = _read_rows(input_dir / "normalized_insertion_h10_segments.csv")
    comparison = _read_rows(input_dir / "normalized_insertion_h10_vs_flowstar_comparison.csv")
    ratio_rows = _read_rows(input_dir / "rescue_vs_flowstar_ratio_trace.csv")
    by_summary = {row.get("run_id", ""): row for row in summary}

    summary_rows: list[dict[str, Any]] = []
    attempt_rows: list[dict[str, Any]] = []
    breakdown_rows: list[dict[str, Any]] = []
    for run_id in TARGET_RUNS:
        failure = _failure_attempt(attempts, run_id)
        if failure is None:
            continue
        segment = _last_segment_for_run(segments, run_id)
        failed_dim = _failed_dimension(failure)
        target = _target_width(failure)
        t_failure = _finite(failure.get("t_lo"))
        width_ratio, width_jump = _ratio_near_failure(ratio_rows, run_id, t_failure, comparison)
        focused = _focused_attempts(attempts, run_id, failure)
        step_rejections, rejection_cluster = _step_rejection_cluster(focused, failure)
        dominant = _dominant_term(failure, segment)
        base = by_summary.get(run_id, {})
        order_family = "o4" if "o4" in run_id else "o6_candidate8"
        summary_rows.append(
            {
                "run_id": run_id,
                "order_family": order_family,
                "status": base.get("status", "failed"),
                "last_validated_t": base.get("last_validated_t", ""),
                "last_attempted_t": base.get("last_attempted_t", ""),
                "failure_t_start": failure.get("t_lo", ""),
                "failure_h_try": failure.get("h_try", failure.get("h", "")),
                "failed_dimension": failed_dim,
                "residual_width_x": failure.get("residual_width_x", ""),
                "residual_width_y": failure.get("residual_width_y", ""),
                "target_width_x": target,
                "target_width_y": target,
                "residual_over_target_x": _ratio(failure.get("residual_width_x"), target),
                "residual_over_target_y": _ratio(failure.get("residual_width_y"), target),
                "shift_or_width": _shift_or_width(failure, failed_dim),
                "dominant_term": dominant,
                "width_ratio_near_failure": width_ratio,
                "step_rejections_near_failure": step_rejections,
                "failure_after_width_ratio_jump": width_jump,
                "failure_after_rejection_cluster": rejection_cluster,
                "priority_for_h10_parity": "yes" if order_family == "o4" else "secondary",
                "notes": "post-hoc localization from h10 CSV diagnostics",
            }
        )
        attempt_rows.extend(focused)
        breakdown_rows.extend(_breakdown_rows(run_id, failure, segment, failed_dim))

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "h10_failure_summary.csv", SUMMARY_FIELDS, summary_rows)
    _write_csv(out_dir / "h10_failure_attempts.csv", ATTEMPT_FIELDS, attempt_rows)
    _write_csv(out_dir / "h10_failure_residual_breakdown.csv", BREAKDOWN_FIELDS, breakdown_rows)
    _write_report(out_dir, summary_rows)
    _make_plots(out_dir, attempt_rows, ratio_rows)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("outputs/flowstar_normalized_insertion_h10"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/flowstar_normalized_insertion_failure"))
    args = parser.parse_args(argv)
    run(args.input_dir, args.out_dir)
    print(f"wrote h10 failure localization outputs to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
