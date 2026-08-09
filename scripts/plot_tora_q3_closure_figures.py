#!/usr/bin/env python3
"""Render deterministic aggregate TORA-Q3 closure figures."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


COLORS = {
    "baseline_native_k2": "#4c78a8",
    "k3_picard": "#f58518",
    "algorithm_aligned_q3": "#54a24b",
    "algorithm_aligned_h005_refresh1": "#e45756",
    "xiangru_native_q3": "#7b61a8",
}
LABELS = {
    "baseline_native_k2": "Torch K2",
    "k3_picard": "Torch K3",
    "algorithm_aligned_q3": "Torch aligned",
    "algorithm_aligned_h005_refresh1": "Torch aligned h=.05",
    "xiangru_native_q3": "Xiangru native",
}


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(
        path,
        format="svg",
        bbox_inches="tight",
        metadata={"Date": None, "Creator": "torch-tm-flowpipe aggregate plotter"},
    )
    plt.close(fig)
    # Matplotlib emits path commands with insignificant trailing spaces.
    # Normalize them so generated evidence passes the repository whitespace gate.
    normalized = "\n".join(
        line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()
    )
    path.write_text(normalized + "\n", encoding="utf-8")


def width_remainder_figure(native: Path, output: Path) -> None:
    sources = (
        ("Endpoint x4 width", native / "endpoint_width_over_time.csv"),
        ("Interval-remainder x4 width", native / "remainder_width_over_time.csv"),
    )
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 7.2), sharex=True)
    for axis, (title, source) in zip(axes, sources, strict=True):
        selected = [row for row in rows(source) if row["state"] == "x4"]
        for lane in LABELS:
            lane_rows = [
                row
                for row in selected
                if row["formal_lane"] == lane and float(row["physical_time"]) <= 5.0
            ]
            axis.plot(
                [float(row["physical_time"]) for row in lane_rows],
                [float(row["width_maximum"]) for row in lane_rows],
                label=LABELS[lane],
                color=COLORS[lane],
                linewidth=1.8,
            )
        axis.set_title(title)
        axis.set_ylabel("maximum width")
        axis.grid(alpha=0.25)
    axes[-1].set_xlabel("physical time (s); Torch curves stop at first property failure")
    axes[0].legend(ncol=2, fontsize=8)
    fig.suptitle("Native closed-loop width growth through the attempted T5 gate")
    fig.tight_layout()
    save(fig, output / "native_width_remainder_growth.svg")


def native_runtime_figure(native: Path, output: Path) -> None:
    selected = [
        row
        for row in rows(native / "runtime_breakdown.csv")
        if row["formal_lane"] != "xiangru_native_q3"
    ]
    lanes = [row["formal_lane"] for row in selected]
    components = {
        "controller build": [
            float(row["formal_prefix_controller_build_seconds"]) for row in selected
        ],
        "controller bound/compose": [
            float(row["formal_prefix_controller_bound_seconds"])
            + float(row["formal_prefix_controller_composition_seconds"])
            for row in selected
        ],
        "normalization": [
            float(row["formal_prefix_normalization_seconds"]) for row in selected
        ],
        "plant": [float(row["formal_prefix_plant_seconds"]) for row in selected],
    }
    fig, axis = plt.subplots(figsize=(9.0, 4.8))
    bottom = [0.0] * len(lanes)
    palette = ("#9ecae9", "#6baed6", "#fdcc8a", "#e34a33")
    for (name, values), color in zip(components.items(), palette, strict=True):
        axis.bar(range(len(lanes)), values, bottom=bottom, label=name, color=color)
        bottom = [left + value for left, value in zip(bottom, values, strict=True)]
    axis.set_xticks(range(len(lanes)), [LABELS[lane] for lane in lanes], rotation=12)
    axis.set_ylabel("measured accounted seconds")
    axis.set_title("Torch native formal-prefix runtime through first failure")
    axis.legend(fontsize=8)
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save(fig, output / "native_runtime_stage_breakdown.svg")


def common_control_runtime_figure(
    fused_path: Path,
    optimized_path: Path,
    baseline_path: Path,
    output: Path,
) -> None:
    fused = json.loads(fused_path.read_text(encoding="utf-8"))
    optimized = json.loads(optimized_path.read_text(encoding="utf-8"))
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    values = [
        float(fused["baseline_t20_seconds"]),
        float(optimized["common_control_t20"]["optimized_compiled_statistics"]["median_seconds"]),
        float(fused["common_control_t20"]["runtime"]["median_seconds"]),
        float(
            baseline["lanes"]["xiangru_matched_crown"][
                "steady_wall_statistics"
            ]["median_seconds"]
        ),
    ]
    labels = ("Torch frozen", "Torch prior optimized", "Torch fused", "Xiangru matched")
    fig, axis = plt.subplots(figsize=(8.5, 4.8))
    bars = axis.bar(labels, values, color=("#9ecae9", "#6baed6", "#54a24b", "#7b61a8"))
    axis.set_yscale("log")
    axis.set_ylabel("common-control T20 median seconds (log scale)")
    axis.set_title("Matched-stack common-control plant runtime")
    axis.grid(axis="y", which="both", alpha=0.25)
    for bar, value in zip(bars, values, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value * 1.12,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    fig.tight_layout()
    save(fig, output / "common_control_runtime_comparison.svg")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-dir", type=Path, required=True)
    parser.add_argument("--fused-summary", type=Path, required=True)
    parser.add_argument("--optimized-summary", type=Path, required=True)
    parser.add_argument("--baseline-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    matplotlib.rcParams["svg.hashsalt"] = "tora-q3-stage-parity-fused-20260809"
    width_remainder_figure(args.native_dir, output)
    native_runtime_figure(args.native_dir, output)
    common_control_runtime_figure(
        args.fused_summary,
        args.optimized_summary,
        args.baseline_summary,
        output,
    )
    print(json.dumps({"status": "PASS", "figure_count": 3}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
