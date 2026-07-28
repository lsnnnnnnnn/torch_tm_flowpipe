#!/usr/bin/env python3
"""Generate the mandatory repair figures in PNG and PDF."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (
    PROTOCOL_BOX,
    PROTOCOL_NATIVE,
    PROTOCOL_RAW,
    PROTOCOL_TUBE,
)

HERE = Path(__file__).resolve().parent


def _save(fig: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(base.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _empty(axis: plt.Axes, message: str) -> None:
    axis.text(0.5, 0.5, message, ha="center", va="center", transform=axis.transAxes)
    axis.set_axis_off()


def riccati_bounds(raw: pd.DataFrame, plots: Path) -> None:
    data = raw[
        (raw.system == "riccati")
        & (raw.protocol == PROTOCOL_BOX)
        & (raw.interval_kind == "endpoint_raw")
    ].copy()
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    if data.empty:
        _empty(axes[0], "No Riccati endpoint rows")
        _empty(axes[1], "No Riccati endpoint rows")
    else:
        for (tool, variant), group in data.groupby(["tool", "tool_variant"]):
            group = group[group.state_index == 0].sort_values("absolute_time")
            label = f"{tool}: {variant}"
            axes[0].plot(group.absolute_time, group.lower, marker=".", label=label)
            axes[1].plot(group.absolute_time, group.upper, marker=".", label=label)
        exact = data[data.state_index == 0].sort_values("absolute_time")
        exact = exact.drop_duplicates("absolute_time")
        axes[0].plot(exact.absolute_time, exact.exact_lower, "k--", label="exact")
        axes[1].plot(exact.absolute_time, exact.exact_upper, "k--", label="exact")
        axes[0].set_ylabel("lower bound")
        axes[1].set_ylabel("upper bound")
        axes[1].set_xlabel("absolute time")
        axes[0].legend(fontsize=6, ncol=2)
    fig.suptitle("Riccati raw endpoint bounds versus the analytic hull")
    _save(fig, plots / "riccati_lower_upper_vs_exact")


def flowstar_stock_reinjection(raw: pd.DataFrame, plots: Path) -> None:
    data = raw[
        (raw.tool == "flowstar")
        & (raw.system == "riccati")
        & (raw.protocol == PROTOCOL_BOX)
        & (raw.interval_kind == "endpoint_raw")
        & raw.tool_variant.isin(
            ["flowstar_stock", "flowstar_candidate_reinjection_diagnostic"]
        )
        & (raw.state_index == 0)
    ].copy()
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    if data.empty:
        for axis in axes:
            _empty(axis, "No stock/reinjection rows")
    else:
        for variant, group in data.groupby("tool_variant"):
            group = group.sort_values("absolute_time")
            axes[0].plot(group.absolute_time, group.lower, marker=".", label=variant)
            axes[1].plot(group.absolute_time, group.upper, marker=".", label=variant)
            axes[2].plot(
                group.absolute_time, group.remainder_width, marker=".", label=variant
            )
        axes[0].set_ylabel("lower")
        axes[1].set_ylabel("upper")
        axes[2].set_ylabel("remainder width")
        axes[2].set_xlabel("absolute time")
        axes[0].legend(fontsize=7)
    fig.suptitle("Flow* stock versus candidate-reinjection diagnostic")
    _save(fig, plots / "flowstar_stock_vs_candidate_reinjection")


def flowstar_parity(output: Path, plots: Path) -> None:
    path = output / "flowstar_original_parity.csv"
    data = pd.read_csv(path) if path.exists() else pd.DataFrame()
    fig, axis = plt.subplots(figsize=(9, 4.5))
    if data.empty:
        _empty(axis, "No original-parity schedule")
    else:
        for implementation, group in data.groupby("implementation"):
            group = group.sort_values("absolute_time")
            axis.plot(
                group.absolute_time,
                group.step_size,
                marker=".",
                markersize=2,
                label=implementation,
            )
        axis.set_xlabel("absolute time")
        axis.set_ylabel("accepted step size")
        axis.legend(fontsize=8)
    axis.set_title("Flow* original benchmark and identical-settings harnesses")
    _save(fig, plots / "flowstar_original_vs_generated_parity")


def torch_raw_tightened(raw: pd.DataFrame, plots: Path) -> None:
    data = raw[
        (raw.tool == "torch_tm_flowpipe")
        & (raw.protocol == PROTOCOL_RAW)
        & raw.interval_kind.isin(["endpoint_raw", "endpoint_tightened"])
    ]
    fig, axis = plt.subplots(figsize=(9, 5))
    if data.empty:
        _empty(axis, "No Torch endpoint audit rows")
    else:
        for (kind, system), group in data.groupby(["interval_kind", "system"]):
            group = group.groupby("h", as_index=False).width.mean().sort_values("h")
            axis.plot(group.h, group.width, marker="o", label=f"{system}: {kind}")
        axis.set_xlabel("h")
        axis.set_ylabel("mean component width")
        axis.set_xscale("log")
        axis.legend(fontsize=7, ncol=2)
    axis.set_title("Torch raw endpoint versus fixed-time tightened endpoint")
    _save(fig, plots / "torch_raw_vs_tightened_endpoint")


def inflation(raw: pd.DataFrame, plots: Path, *, protocol: str, kind: str, name: str) -> None:
    data = raw[
        (raw.protocol == protocol)
        & (raw.interval_kind == kind)
        & raw.inflation_ratio.notna()
        & (raw.analytic_reference_status == "passed")
    ]
    fig, axis = plt.subplots(figsize=(9, 5))
    if data.empty:
        _empty(axis, "No valid exact-reference inflation rows")
    else:
        for (tool, variant, system), group in data.groupby(
            ["tool", "tool_variant", "system"]
        ):
            group = group.groupby("h", as_index=False).inflation_ratio.mean()
            group = group.sort_values("h")
            axis.plot(
                group.h,
                group.inflation_ratio,
                marker="o",
                label=f"{tool}/{variant}/{system}",
            )
        axis.axhline(1.0, color="k", linestyle="--", linewidth=1)
        axis.set_xscale("log")
        axis.set_xlabel("h")
        axis.set_ylabel("width / exact width")
        axis.legend(fontsize=5, ncol=2)
    axis.set_title(name.replace("_", " "))
    _save(fig, plots / name)


def width_curves(
    raw: pd.DataFrame, plots: Path, *, protocol: str, name: str, title: str
) -> None:
    data = raw[
        (raw.protocol == protocol)
        & (raw.interval_kind == "endpoint_raw")
        & (raw.native_validation_status != "failed")
    ]
    systems = list(data.system.unique()) if not data.empty else []
    fig, axes = plt.subplots(max(1, len(systems)), 1, figsize=(10, 4 * max(1, len(systems))))
    axes = np.atleast_1d(axes)
    if not systems:
        _empty(axes[0], "No width-curve rows")
    for axis, system in zip(axes, systems):
        subset = data[data.system == system]
        for (tool, variant, state), group in subset.groupby(
            ["tool", "tool_variant", "state_index"]
        ):
            group = group.sort_values("absolute_time")
            axis.plot(
                group.absolute_time,
                group.width,
                marker=".",
                label=f"{tool}/{variant}/x{state}",
            )
        axis.set_title(system)
        axis.set_xlabel("absolute time")
        axis.set_ylabel("raw endpoint width")
        axis.legend(fontsize=5, ncol=2)
    fig.suptitle(title)
    _save(fig, plots / name)


def sensitivity(output: Path, plots: Path) -> None:
    path = output / "flowstar_parameter_sensitivity.csv"
    data = pd.read_csv(path) if path.exists() else pd.DataFrame()
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    if data.empty or "sensitivity_label" not in data:
        for axis in axes.flat:
            _empty(axis, "Full sensitivity sweep not run")
    else:
        data = data.copy()
        data["successful_horizon"] = data.h * data.completed_steps
        factors = [
            ("order=", "order"),
            ("candidate=", "candidate remainder"),
            ("cutoff=", "cutoff"),
            ("h=", "fixed step policy"),
        ]
        for axis, (prefix, title) in zip(axes.flat, factors):
            subset = data[data.sensitivity_label.astype(str).str.startswith(prefix)]
            if subset.empty:
                _empty(axis, f"No {title} sweep")
                continue
            axis.bar(
                subset.sensitivity_label.astype(str),
                subset.successful_horizon,
            )
            axis.tick_params(axis="x", rotation=45, labelsize=7)
            axis.set_ylabel("successful horizon")
            axis.set_title(title)
    fig.suptitle("Flow* fixed-configuration failure sensitivity")
    _save(fig, plots / "flowstar_successful_horizon_sensitivity")


def failure_chart(raw: pd.DataFrame, plots: Path) -> None:
    failures = raw[raw.interval_kind == "failure"].drop_duplicates(
        ["run_id", "failure_category"]
    )
    fig, axis = plt.subplots(figsize=(9, 4.5))
    if failures.empty:
        _empty(axis, "No failures in this run")
    else:
        counts = failures.failure_category.value_counts().sort_index()
        axis.bar(counts.index, counts.values)
        axis.tick_params(axis="x", rotation=35)
        axis.set_ylabel("failed configurations")
    axis.set_title("Structured Flow*/adapter failure categories")
    _save(fig, plots / "flowstar_failure_categories")


def runtime(raw: pd.DataFrame, plots: Path) -> None:
    data = raw[
        raw.interval_kind.isin(["tube", "endpoint_raw"])
    ].drop_duplicates("run_id")
    fig, axis = plt.subplots(figsize=(11, 5))
    if data.empty:
        _empty(axis, "No runtime rows")
    else:
        grouped = data.groupby("tool")[
            ["build_time_s", "warmup_time_s", "steady_runtime_s"]
        ].median(numeric_only=True)
        grouped.plot.bar(ax=axis)
        axis.set_yscale("symlog", linthresh=1e-6)
        axis.set_ylabel("seconds (median per configuration)")
        axis.legend(["compile/JIT", "first call", "steady step"])
        axis.tick_params(axis="x", rotation=0)
    axis.set_title("Runtime decomposition (not a combined ranking)")
    _save(fig, plots / "runtime_decomposition")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    plots = output / "plots"
    raw = pd.read_csv(output / "raw_results.csv", low_memory=False)
    for column in (
        "lower",
        "upper",
        "width",
        "exact_lower",
        "exact_upper",
        "inflation_ratio",
        "remainder_width",
        "build_time_s",
        "warmup_time_s",
        "steady_runtime_s",
        "h",
        "absolute_time",
    ):
        raw[column] = pd.to_numeric(raw[column], errors="coerce")
    riccati_bounds(raw, plots)
    flowstar_stock_reinjection(raw, plots)
    flowstar_parity(output, plots)
    torch_raw_tightened(raw, plots)
    inflation(
        raw,
        plots,
        protocol=PROTOCOL_TUBE,
        kind="tube",
        name="one_step_tube_inflation_vs_h",
    )
    inflation(
        raw,
        plots,
        protocol=PROTOCOL_RAW,
        kind="endpoint_raw",
        name="one_step_raw_endpoint_inflation_vs_h",
    )
    width_curves(
        raw,
        plots,
        protocol=PROTOCOL_BOX,
        name="common_box_raw_endpoint_width",
        title="Common-box raw endpoint carry",
    )
    width_curves(
        raw,
        plots,
        protocol=PROTOCOL_NATIVE,
        name="native_representation_width",
        title="Native carried representations (configuration-specific)",
    )
    sensitivity(output, plots)
    failure_chart(raw, plots)
    runtime(raw, plots)


if __name__ == "__main__":
    main()
