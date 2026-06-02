"""Summarize plant-only Flow* comparison CSV files.

The summary is intentionally conservative: it reports torch-vs-torch evidence
whenever Flow* is unavailable, and only reports torch/Flow* ratios for rows with
parsed Flow* boxes.  Sampling containment is labeled as a regression sanity
check, not a proof.
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping

CASE_KEY = tuple[str, str, str, str]

FLOWSTAR_BOX_NOTE = (
    "Flow* endpoint boxes were not available. Flow* GNUPLOT-derived last-segment "
    "and tube boxes were parsed for {n} completed cases. Torch-vs-Flow* ratios "
    "below use last-segment/tube widths, not endpoint widths."
)


def _read_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _float(v: Any) -> float | None:
    try:
        if v in (None, ""):
            return None
        x = float(v)
        if math.isfinite(x):
            return x
    except (TypeError, ValueError):
        return None
    return None


def _int(v: Any) -> int | None:
    x = _float(v)
    return None if x is None else int(x)


def _case_key(row: Mapping[str, str]) -> CASE_KEY:
    return (row["system"], row["h"], row["steps"], row["order"])


def _fmt(x: float | None, digits: int = 4) -> str:
    if x is None or not math.isfinite(x):
        return ""
    return f"{x:.{digits}g}"


def _truthy(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    return str(v).strip().lower() in {"1", "true", "yes", "y"}


def _ratio_table(rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    range_rows: dict[CASE_KEY, dict[str, str]] = {}
    dep_rows: dict[CASE_KEY, dict[str, str]] = {}
    for r in rows:
        if r.get("tool") != "torch_tm_flowpipe" or r.get("status") != "validated":
            continue
        if r.get("mode") == "range_only":
            range_rows[_case_key(r)] = r
        elif r.get("mode") == "dependency_preserving":
            dep_rows[_case_key(r)] = r

    by_system: dict[str, list[dict[str, Any]]] = defaultdict(list)
    per_case: list[dict[str, Any]] = []
    for key, dep in dep_rows.items():
        rng = range_rows.get(key)
        if rng is None:
            continue
        dep_w = _float(dep.get("final_width_sum"))
        rng_w = _float(rng.get("final_width_sum"))
        dep_t = _float(dep.get("runtime_s"))
        rng_t = _float(rng.get("runtime_s"))
        if dep_w is None or rng_w is None or rng_w <= 0:
            continue
        ratio = dep_w / rng_w
        runtime_ratio = dep_t / rng_t if dep_t is not None and rng_t and rng_t > 0 else None
        row = {
            "system": key[0],
            "h": key[1],
            "steps": int(float(key[2])),
            "order": int(float(key[3])),
            "range_width": rng_w,
            "dependency_width": dep_w,
            "width_ratio": ratio,
            "runtime_ratio": runtime_ratio,
            "range_failures": _int(rng.get("containment_failures")) or 0,
            "dependency_failures": _int(dep.get("containment_failures")) or 0,
        }
        per_case.append(row)
        by_system[key[0]].append(row)

    summary: list[dict[str, Any]] = []
    for system, items in sorted(by_system.items()):
        ratios = [i["width_ratio"] for i in items]
        runtime_ratios = [i["runtime_ratio"] for i in items if i["runtime_ratio"] is not None]
        summary.append({
            "system": system,
            "cases": len(items),
            "mean_width_ratio": mean(ratios) if ratios else None,
            "min_width_ratio": min(ratios) if ratios else None,
            "max_width_ratio": max(ratios) if ratios else None,
            "mean_runtime_ratio": mean(runtime_ratios) if runtime_ratios else None,
            "better_cases": sum(1 for r in ratios if r < 0.999),
            "worse_cases": sum(1 for r in ratios if r > 1.001),
            "equal_cases": sum(1 for r in ratios if 0.999 <= r <= 1.001),
            "sampling_failures": sum(i["range_failures"] + i["dependency_failures"] for i in items),
        })
    return summary, per_case


def _has_flowstar_last_or_tube(row: Mapping[str, str]) -> bool:
    return (_float(row.get("last_segment_width_sum")) is not None) or (_float(row.get("tube_width_sum")) is not None)


def _flowstar_parsed_completed_count(rows: list[dict[str, str]]) -> int:
    return sum(
        1
        for r in rows
        if r.get("tool") == "flowstar" and r.get("status") == "completed" and _has_flowstar_last_or_tube(r)
    )


def _flowstar_ratio_table(rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    flow: dict[CASE_KEY, list[dict[str, str]]] = defaultdict(list)
    torch: list[dict[str, str]] = []
    for r in rows:
        if r.get("tool") == "flowstar" and r.get("status") == "completed":
            flow[_case_key(r)].append(r)
        elif r.get("tool") == "torch_tm_flowpipe" and r.get("status") == "validated":
            torch.append(r)
    per_case: list[dict[str, Any]] = []
    by_label: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in torch:
        flow_rows = flow.get(_case_key(r), [])
        if not flow_rows:
            continue
        for f in flow_rows:
            specs = [
                ("last_segment", "last_segment_width_sum", True),
                ("tube", "tube_width_sum", True),
                ("endpoint", "endpoint_width_sum", _truthy(r.get("endpoint_box_available")) and _truthy(f.get("endpoint_box_available"))),
            ]
            for ratio_type, width_col, allowed in specs:
                if not allowed:
                    continue
                tw = _float(r.get(width_col))
                fw = _float(f.get(width_col))
                tt = _float(r.get("runtime_s"))
                ft = _float(f.get("flowstar_internal_reach_s") or f.get("runtime_s"))
                if tw is None or fw is None or fw <= 0:
                    continue
                item = {
                    "system": r["system"],
                    "ratio_type": ratio_type,
                    "torch_mode": r["mode"],
                    "flowstar_setting_label": f.get("setting_label") or "default",
                    "h": r["h"],
                    "steps": int(float(r["steps"])),
                    "order": int(float(r["order"])),
                    "torch_width": tw,
                    "flowstar_width": fw,
                    "torch_over_flowstar_ratio": tw / fw,
                    "torch_over_flowstar_runtime": tt / ft if tt is not None and ft and ft > 0 else None,
                }
                per_case.append(item)
                by_label[(r["system"], ratio_type, r["mode"], item["flowstar_setting_label"])].append(item)
    summary: list[dict[str, Any]] = []
    for (system, ratio_type, torch_mode, setting_label), items in sorted(by_label.items()):
        wr = [i["torch_over_flowstar_ratio"] for i in items]
        tr = [i["torch_over_flowstar_runtime"] for i in items if i["torch_over_flowstar_runtime"] is not None]
        summary.append({
            "system": system,
            "ratio_type": ratio_type,
            "torch_mode": torch_mode,
            "flowstar_setting_label": setting_label,
            "cases": len(items),
            "mean_width_ratio": mean(wr) if wr else None,
            "min_width_ratio": min(wr) if wr else None,
            "max_width_ratio": max(wr) if wr else None,
            "mean_runtime_ratio": mean(tr) if tr else None,
        })
    return summary, per_case


def _markdown_table(headers: Iterable[str], rows: Iterable[Iterable[Any]]) -> str:
    headers = list(headers)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines)


def generate_summary(csv_path: str | Path, out_path: str | Path | None = None) -> str:
    rows = _read_rows(csv_path)
    status_counts = Counter((r.get("tool", ""), r.get("mode", ""), r.get("status", "")) for r in rows)
    torch_summary, torch_cases = _ratio_table(rows)
    flow_summary, flow_cases = _flowstar_ratio_table(rows)
    parsed_flowstar_cases = _flowstar_parsed_completed_count(rows)

    lines: list[str] = []
    lines.append("# Plant-only Flow* comparison summary")
    lines.append("")
    lines.append(f"Source CSV: `{Path(csv_path).as_posix()}`")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("This is a plant-only comparison of polynomial ODE flowpipe boxes. It is not a full CROWN-Reach NNCS run and it does not compare raw Taylor-model coefficients.")
    lines.append("")
    lines.append("## Status counts")
    lines.append("")
    lines.append(_markdown_table(["tool", "mode", "status", "count"], ((k[0], k[1], k[2], v) for k, v in sorted(status_counts.items()))))
    lines.append("")
    lines.append("## Evidence that dependency-preserving propagation is useful")
    lines.append("")
    if torch_summary:
        lines.append(_markdown_table(
            ["system", "cases", "mean dep/range width", "min", "max", "mean runtime ratio", "better", "same", "worse", "sampling failures"],
            (
                [
                    r["system"], r["cases"], _fmt(r["mean_width_ratio"]), _fmt(r["min_width_ratio"]), _fmt(r["max_width_ratio"]),
                    _fmt(r["mean_runtime_ratio"]), r["better_cases"], r["equal_cases"], r["worse_cases"], r["sampling_failures"],
                ]
                for r in torch_summary
            ),
        ))
        best = sorted(torch_cases, key=lambda r: r["width_ratio"])[:5]
        worst = sorted(torch_cases, key=lambda r: r["width_ratio"], reverse=True)[:5]
        lines.append("")
        lines.append("Best dependency-preserving improvements:")
        lines.append("")
        lines.append(_markdown_table(
            ["system", "h", "steps", "order", "dep width", "range width", "ratio"],
            ([r["system"], r["h"], r["steps"], r["order"], _fmt(r["dependency_width"], 6), _fmt(r["range_width"], 6), _fmt(r["width_ratio"], 6)] for r in best),
        ))
        lines.append("")
        lines.append("Largest regressions or neutral cases:")
        lines.append("")
        lines.append(_markdown_table(
            ["system", "h", "steps", "order", "dep width", "range width", "ratio"],
            ([r["system"], r["h"], r["steps"], r["order"], _fmt(r["dependency_width"], 6), _fmt(r["range_width"], 6), _fmt(r["width_ratio"], 6)] for r in worst),
        ))
    else:
        lines.append("No paired validated torch rows were found.")
    lines.append("")
    lines.append("Interpretation: a dependency/range width ratio below 1 means the dependency-preserving endpoint box is tighter for the same plant, step size, horizon, and Taylor order. A ratio above 1 is a useful regression signal for nonlinear cases where term growth/remainder accumulation dominates.")
    lines.append("")
    lines.append("## Torch over Flow* ratios")
    lines.append("")
    if parsed_flowstar_cases:
        lines.append(FLOWSTAR_BOX_NOTE.format(n=parsed_flowstar_cases))
        lines.append("")
    if flow_summary:
        lines.append(_markdown_table(
            ["system", "ratio_type", "torch_mode", "flowstar_setting_label", "cases", "mean torch/Flow* width", "min", "max", "mean runtime ratio"],
            (
                [
                    r["system"], r["ratio_type"], r["torch_mode"], r["flowstar_setting_label"], r["cases"],
                    _fmt(r["mean_width_ratio"]), _fmt(r["min_width_ratio"]), _fmt(r["max_width_ratio"]), _fmt(r["mean_runtime_ratio"]),
                ]
                for r in flow_summary
            ),
        ))
        lines.append("")
        lines.append("Representative semantic ratio rows:")
        lines.append("")
        lines.append(_markdown_table(
            ["ratio_type", "torch_mode", "flowstar_setting_label", "h", "steps", "order", "torch_width", "flowstar_width", "torch_over_flowstar_ratio"],
            (
                [
                    r["ratio_type"], r["torch_mode"], r["flowstar_setting_label"], r["h"], r["steps"], r["order"],
                    _fmt(r["torch_width"], 6), _fmt(r["flowstar_width"], 6), _fmt(r["torch_over_flowstar_ratio"], 6),
                ]
                for r in sorted(flow_cases, key=lambda x: (x["ratio_type"], x["torch_mode"], x["flowstar_setting_label"], x["steps"], x["order"]))[:20]
            ),
        ))
    else:
        if parsed_flowstar_cases:
            lines.append("Parsed Flow* last-segment/tube boxes were found, but no matching validated torch rows had compatible semantic widths for ratio reporting.")
        else:
            lines.append("No parsed Flow* boxes were available. This usually means Flow* was not installed, the C++ benchmark did not compile/run, or the generated plot/range files could not be parsed. Torch-vs-torch evidence above is still valid, but torch-vs-Flow* numeric claims should not be made yet.")
    lines.append("")
    lines.append("## Validation note")
    lines.append("")
    lines.append("The `containment_failures` column is a sampling-based regression sanity check. It is not a formal proof. Formal soundness claims should be limited to the implemented Taylor-model validation assumptions and the current floating-point prototype limitations.")
    lines.append("")
    text = "\n".join(lines)
    if out_path is not None:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize torch/Flow* plant-only comparison CSV.")
    parser.add_argument("csv", nargs="?", default="outputs/flowstar_comparison.csv")
    parser.add_argument("--out", default=None, help="Markdown output path; default prints only")
    args = parser.parse_args()
    text = generate_summary(args.csv, args.out)
    if args.out is None:
        print(text)
    else:
        print(args.out)


if __name__ == "__main__":
    main()
