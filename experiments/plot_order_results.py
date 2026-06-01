#!/usr/bin/env python3
"""Generate order-sweep plots and a Flow* status table for Van der Pol."""
from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def read_rows(path: str | Path) -> list[dict[str, str]]:
    p = Path(path)
    if not p.exists():
        return []
    with p.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def to_float(v: Any) -> float | None:
    try:
        if v in (None, ""):
            return None
        x = float(v)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def to_int(v: Any) -> int | None:
    x = to_float(v)
    return int(x) if x is not None else None


def key_filter(row: dict[str, str], h: float, steps: int) -> bool:
    return row.get("system") == "van_der_pol" and to_float(row.get("h")) == h and to_int(row.get("steps")) == steps


def plot_metric_by_order(rows: list[dict[str, str]], metric: str, out: Path, ylabel: str, *, h: float, steps: int, torch_only: bool = False) -> None:
    series: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if not key_filter(r, h, steps):
            continue
        if torch_only and r.get("tool") == "flowstar":
            continue
        if r.get("tool") == "torch_tm_flowpipe" and r.get("status") != "validated":
            continue
        if r.get("tool") == "flowstar" and r.get("status") != "completed":
            continue
        order = to_int(r.get("order") or r.get("requested_order"))
        val = to_float(r.get(metric))
        if order is None or val is None:
            continue
        label = f"{r.get('tool', 'torch_tm_flowpipe')} {r.get('mode', '')}".strip()
        series[label][order].append(val)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    if series:
        for label, by_order in sorted(series.items()):
            xs = sorted(by_order)
            ys = [sum(by_order[x]) / len(by_order[x]) for x in xs]
            ax.plot(xs, ys, marker="o", label=label)
        ax.set_xlabel("requested order")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Van der Pol h={h:g}, steps={steps}")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "No matching parsed rows", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_torch_over_flowstar(rows: list[dict[str, str]], out: Path, *, h: float, steps: int) -> None:
    flow: dict[int, float] = {}
    for r in rows:
        if key_filter(r, h, steps) and r.get("tool") == "flowstar" and r.get("status") == "completed":
            order = to_int(r.get("order"))
            width = to_float(r.get("final_width_sum"))
            if order is not None and width and width > 0:
                flow[order] = width
    ratios: dict[str, dict[int, float]] = defaultdict(dict)
    for r in rows:
        if not key_filter(r, h, steps) or r.get("tool") != "torch_tm_flowpipe" or r.get("status") != "validated":
            continue
        order = to_int(r.get("order"))
        width = to_float(r.get("final_width_sum"))
        if order is None or width is None or order not in flow:
            continue
        ratios[r.get("mode", "torch")][order] = width / flow[order]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    if ratios:
        for label, by_order in sorted(ratios.items()):
            xs = sorted(by_order)
            ax.plot(xs, [by_order[x] for x in xs], marker="o", label=label)
        ax.axhline(1.0, linestyle="--", linewidth=1, color="black")
        ax.set_xlabel("requested order")
        ax.set_ylabel("torch final width sum / Flow* final width sum")
        ax.set_title(f"Van der Pol h={h:g}, steps={steps}")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "No completed parsed Flow* widths", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_dependency_ratio(rows: list[dict[str, str]], out: Path, *, h: float, steps: int) -> None:
    ranges: dict[int, float] = {}
    deps: dict[int, float] = {}
    for r in rows:
        if not key_filter(r, h, steps) or r.get("tool") != "torch_tm_flowpipe" or r.get("status") != "validated":
            continue
        order = to_int(r.get("order"))
        width = to_float(r.get("final_width_sum"))
        if order is None or width is None:
            continue
        if r.get("mode") == "range_only":
            ranges[order] = width
        elif r.get("mode") == "dependency_preserving":
            deps[order] = width
    xs = sorted(set(ranges) & set(deps))
    fig, ax = plt.subplots(figsize=(8, 4.8))
    if xs:
        ax.plot(xs, [deps[x] / ranges[x] for x in xs], marker="o")
        ax.axhline(1.0, linestyle="--", linewidth=1, color="black")
        ax.set_xlabel("requested order")
        ax.set_ylabel("dependency_preserving / range_only")
        ax.set_title(f"Van der Pol h={h:g}, steps={steps}")
    else:
        ax.text(0.5, 0.5, "No paired torch rows", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_status(rows: list[dict[str, str]], out: Path) -> None:
    counts: dict[int, Counter[str]] = defaultdict(Counter)
    for r in rows:
        if r.get("system") == "van_der_pol" and r.get("tool") == "flowstar":
            order = to_int(r.get("order"))
            if order is not None:
                counts[order][r.get("status", "")] += 1
    statuses = sorted({s for c in counts.values() for s in c})
    xs = sorted(counts)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    bottom = [0] * len(xs)
    if xs and statuses:
        for status in statuses:
            vals = [counts[x][status] for x in xs]
            ax.bar(xs, vals, bottom=bottom, label=status)
            bottom = [b + v for b, v in zip(bottom, vals)]
        ax.set_xlabel("requested order")
        ax.set_ylabel("Flow* case count")
        ax.set_title("Van der Pol Flow* status by order")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "No Flow* rows", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def write_status_table(rows: list[dict[str, str]], out: Path) -> None:
    vdp = [r for r in rows if r.get("system") == "van_der_pol" and r.get("tool") == "flowstar"]
    lines = ["| order | h | steps | status | final_width_sum | runtime_s | failure_reason |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for r in sorted(vdp, key=lambda x: (to_int(x.get("order")) or -1, to_float(x.get("h")) or -1, to_int(x.get("steps")) or -1)):
        reason = (r.get("failure_reason") or "").replace("|", "/")
        lines.append(f"| {r.get('order','')} | {r.get('h','')} | {r.get('steps','')} | {r.get('status','')} | {r.get('final_width_sum','')} | {r.get('runtime_s','')} | {reason} |")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Van der Pol order-sweep diagnostics.")
    parser.add_argument("--diagnostics-csv", default="outputs/van_der_pol_diagnostics_by_order.csv")
    parser.add_argument("--comparison-csv", default="outputs/flowstar_comparison_by_order.csv")
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument("--h", type=float, default=0.01)
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    diag_rows = read_rows(args.diagnostics_csv)
    for r in diag_rows:
        r.setdefault("tool", "torch_tm_flowpipe")
        if "requested_order" in r and "order" not in r:
            r["order"] = r["requested_order"]
    comp_rows = read_rows(args.comparison_csv)
    combined = comp_rows + diag_rows

    plot_metric_by_order(combined, "final_width_sum", out_dir / "van_der_pol_width_vs_order.png", "final width sum", h=args.h, steps=args.steps)
    plot_metric_by_order(combined, "runtime_s", out_dir / "van_der_pol_runtime_vs_order.png", "runtime (s)", h=args.h, steps=args.steps)
    plot_metric_by_order(diag_rows, "remainder_width_sum", out_dir / "van_der_pol_remainder_vs_order.png", "remainder width sum", h=args.h, steps=args.steps, torch_only=True)
    plot_torch_over_flowstar(comp_rows, out_dir / "torch_over_flowstar_width_ratio_by_order.png", h=args.h, steps=args.steps)
    plot_dependency_ratio(comp_rows, out_dir / "dependency_vs_range_ratio_by_order.png", h=args.h, steps=args.steps)
    plot_status(comp_rows, out_dir / "flowstar_status_by_order.png")
    write_status_table(comp_rows, out_dir / "order_flowstar_status_table.md")
    print(f"wrote plots and {out_dir / 'order_flowstar_status_table.md'}")


if __name__ == "__main__":
    main()
