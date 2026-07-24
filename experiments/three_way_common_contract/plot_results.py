#!/usr/bin/env python3
"""Generate the seven mandatory comparison figures."""
from __future__ import annotations

import argparse
import math
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PRIMARY_VARIANTS = {
    "torch_tm_flowpipe": "complete_total_degree_order_1",
    "diffreach": "affine_flag",
    "flowstar": "minimum_supported_fixed_order_2",
}
LABELS = {
    "torch_tm_flowpipe": "Torch TM",
    "diffreach": "DiffReach",
    "flowstar": "Flow*",
}
COLORS = {
    "torch_tm_flowpipe": "#377eb8",
    "diffreach": "#4daf4a",
    "flowstar": "#e41a1c",
}


def _primary(frame: pd.DataFrame) -> pd.DataFrame:
    mask = pd.Series(False, index=frame.index)
    for tool, variant in PRIMARY_VARIANTS.items():
        mask |= (frame["tool"] == tool) & (
            frame["tool_variant"] == variant
        )
    return frame[mask].copy()


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def one_step(output: Path, raw: pd.DataFrame) -> None:
    data = _primary(raw)
    data = data[
        (data["protocol"] == "one_step_common_input")
        & (data["interval_kind"] == "endpoint")
        & (data["step_index"] == 1)
        & data["system"].isin(["riccati", "harmonic"])
    ].copy()
    data["exact_inflation_ratio"] = pd.to_numeric(
        data["exact_inflation_ratio"], errors="coerce"
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)
    for axis, system in zip(axes, ("riccati", "harmonic")):
        subset = data[data["system"] == system]
        for tool in PRIMARY_VARIANTS:
            tool_data = (
                subset[subset["tool"] == tool]
                .groupby("h", as_index=False)["exact_inflation_ratio"]
                .max()
                .sort_values("h")
            )
            axis.plot(
                tool_data["h"],
                tool_data["exact_inflation_ratio"],
                marker="o",
                label=LABELS[tool],
                color=COLORS[tool],
            )
        axis.axhline(1.0, color="black", lw=1, ls="--")
        axis.set_title(system.replace("_", " ").title())
        axis.set_xlabel("step size h")
        axis.set_ylabel("endpoint width / exact width")
        axis.grid(alpha=0.25)
    axes[0].legend()
    fig.suptitle("One-step exact inflation ratio versus h")
    _save(fig, output / "plots" / "one_step_exact_inflation_ratio_vs_h.png")


def width_curves(
    output: Path,
    raw: pd.DataFrame,
    *,
    protocol: str,
    filename: str,
    title: str,
    primary_only: bool,
) -> None:
    data = _primary(raw) if primary_only else raw.copy()
    data = data[
        (data["protocol"] == protocol)
        & (data["interval_kind"] == "endpoint")
        & (data["step_index"] > 0)
        & (data["row_status"] == "validated")
    ].copy()
    data["width"] = pd.to_numeric(data["width"], errors="coerce")
    systems = ["riccati", "harmonic", "van_der_pol"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for axis, system in zip(axes, systems):
        subset = data[data["system"] == system]
        for keys, group in subset.groupby(
            ["tool", "tool_variant", "h", "state_index"]
        ):
            tool, variant, h, state = keys
            variant_suffix = (
                " quasi-Q"
                if variant == "default_restricted_quasi_quadratic"
                else ""
            )
            label = (
                f"{LABELS.get(tool, tool)}{variant_suffix}, "
                f"h={h:g}, x{int(state) + 1}"
            )
            axis.plot(
                group["time"],
                group["width"],
                label=label,
                color=COLORS.get(tool),
                alpha=0.85,
                lw=1.5,
            )
        axis.set_title(system.replace("_", " ").title())
        axis.set_xlabel("absolute time")
        axis.set_ylabel("endpoint width")
        axis.set_yscale("log")
        axis.grid(alpha=0.25)
        if len(subset):
            axis.legend(fontsize=6)
    fig.suptitle(title)
    _save(fig, output / "plots" / filename)


def common_bars(output: Path, common: pd.DataFrame) -> None:
    data = common[common["status"] == "validated"].copy()
    data["width"] = pd.to_numeric(data["width"], errors="coerce")
    data["category"] = data.apply(
        lambda row: (
            f"{row['system']}\nh={float(row['h']):g}, "
            f"t={float(row['checkpoint']):g}, x{int(row['state_index']) + 1}"
        ),
        axis=1,
    )
    categories = list(dict.fromkeys(data["category"]))
    x = np.arange(len(categories), dtype=float)
    width = 0.24
    fig, axis = plt.subplots(figsize=(max(12, 0.75 * len(categories)), 5.2))
    for offset, tool in enumerate(PRIMARY_VARIANTS):
        lookup = (
            data[data["tool"] == tool]
            .groupby("category")["width"]
            .first()
            .to_dict()
        )
        values = [lookup.get(category, np.nan) for category in categories]
        axis.bar(
            x + (offset - 1) * width,
            values,
            width,
            label=LABELS[tool],
            color=COLORS[tool],
        )
    axis.set_xticks(x)
    axis.set_xticklabels(categories, rotation=45, ha="right", fontsize=7)
    axis.set_yscale("log")
    axis.set_ylabel("endpoint width at common absolute time")
    axis.set_title("Common-time grouped widths (missing bars = validation_failed)")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    _save(fig, output / "plots" / "common_time_grouped_width_bars.png")


def failure_horizons(output: Path, failure: pd.DataFrame) -> None:
    data = failure[
        (failure["state_index"] == 0)
        & failure["protocol"].isin(
            ["multi_step_common_box_carry", "native_low_order"]
        )
    ].copy()
    data["failure_horizon_or_censor"] = pd.to_numeric(
        data["failure_horizon_or_censor"], errors="coerce"
    )
    data["label"] = data.apply(
        lambda row: (
            f"{LABELS.get(row['tool'], row['tool'])}"
            + (
                " quasi-Q"
                if row["tool_variant"]
                == "default_restricted_quasi_quadratic"
                else ""
            )
            + f"\n{row['protocol'].replace('_', ' ')}"
            + f"\n{row['system']}, h={float(row['h']):g}"
        ),
        axis=1,
    )
    fig, axis = plt.subplots(figsize=(max(12, len(data) * 0.55), 5.3))
    colors = [COLORS.get(tool, "#777777") for tool in data["tool"]]
    axis.bar(np.arange(len(data)), data["failure_horizon_or_censor"], color=colors)
    for index, (_, row) in enumerate(data.iterrows()):
        if bool(row["censored_at_requested_horizon"]):
            axis.text(
                index,
                row["failure_horizon_or_censor"],
                "≥",
                ha="center",
                va="bottom",
                fontsize=11,
            )
    axis.set_xticks(np.arange(len(data)))
    axis.set_xticklabels(data["label"], rotation=55, ha="right", fontsize=6.5)
    axis.set_ylabel("first failure time (≥ marks horizon-censored success)")
    axis.set_title("First native validation-failure horizon")
    axis.grid(axis="y", alpha=0.25)
    _save(fig, output / "plots" / "first_validation_failure_horizon.png")


def runtime(output: Path, runtime_frame: pd.DataFrame) -> None:
    data = runtime_frame.copy()
    for field in (
        "build_time_s",
        "jit_compile_time_s",
        "steady_runtime_per_step_s",
    ):
        data[field] = pd.to_numeric(data[field], errors="coerce").fillna(0.0)
    grouped = (
        data.groupby(["tool", "protocol"], as_index=False)[
            [
                "build_time_s",
                "jit_compile_time_s",
                "steady_runtime_per_step_s",
            ]
        ]
        .median()
        .sort_values(["protocol", "tool"])
    )
    grouped["label"] = grouped.apply(
        lambda row: (
            f"{LABELS.get(row['tool'], row['tool'])}\n"
            f"{row['protocol'].replace('_', ' ')}"
        ),
        axis=1,
    )
    x = np.arange(len(grouped))
    fields = [
        ("build_time_s", "Flow* build", "#984ea3"),
        ("jit_compile_time_s", "DiffReach JIT", "#ff7f00"),
        ("steady_runtime_per_step_s", "steady step", "#a6cee3"),
    ]
    fig, axis = plt.subplots(figsize=(max(11, len(grouped) * 0.9), 5.1))
    bottom = np.zeros(len(grouped))
    for field, label, color in fields:
        values = grouped[field].to_numpy()
        axis.bar(x, values, bottom=bottom, label=label, color=color)
        bottom += values
    axis.set_xticks(x)
    axis.set_xticklabels(grouped["label"], rotation=45, ha="right", fontsize=7)
    axis.set_yscale("log")
    axis.set_ylabel("seconds (median by tool/protocol)")
    axis.set_title("Runtime decomposition: build/JIT/steady execution")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    _save(fig, output / "plots" / "runtime_decomposition.png")


def semantics(output: Path, frame: pd.DataFrame) -> None:
    columns = [
        "tool",
        "tool_variant",
        "protocol",
        "local_order",
        "carried_representation",
        "reset_policy",
    ]
    data = frame[columns].copy()
    for column in columns:
        data[column] = data[column].astype(str).map(
            lambda value: "\n".join(
                textwrap.wrap(
                    value.replace("_", " "),
                    width=24,
                    break_long_words=False,
                )
            )
        )
    fig_height = max(4.5, 0.65 * len(data) + 1.5)
    fig, axis = plt.subplots(figsize=(15, fig_height))
    axis.axis("off")
    table = axis.table(
        cellText=data.values,
        colLabels=[
            "tool",
            "variant",
            "protocol",
            "local order/construction",
            "carried representation",
            "reset policy",
        ],
        cellLoc="left",
        colLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 2.3)
    axis.set_title("Semantics: local construction and carried representation", pad=18)
    _save(fig, output / "plots" / "semantics_table.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    (output / "plots").mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(output / "raw_results.csv")
    common = pd.read_csv(output / "common_time_summary.csv")
    failure = pd.read_csv(output / "failure_horizon_summary.csv")
    runtime_frame = pd.read_csv(output / "runtime_summary.csv")
    semantics_frame = pd.read_csv(output / "semantics_summary.csv")
    one_step(output, raw)
    width_curves(
        output,
        raw,
        protocol="multi_step_common_box_carry",
        filename="multi_step_common_box_carry_width_vs_time.png",
        title="Multi-step endpoint width under common componentwise-box carry",
        primary_only=True,
    )
    common_bars(output, common)
    failure_horizons(output, failure)
    runtime(output, runtime_frame)
    width_curves(
        output,
        raw,
        protocol="native_low_order",
        filename="native_low_order_width_curves.png",
        title="Native low-order supplementary endpoint-width curves",
        primary_only=False,
    )
    semantics(output, semantics_frame)


if __name__ == "__main__":
    main()
