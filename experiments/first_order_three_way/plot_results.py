#!/usr/bin/env python3
"""Generate the benchmark's PNG/PDF visualization suite."""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/torch_tm_first_order_mpl")

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from common import iter_configurations, load_spec, output_dir_from_args

PRIMARY = "native_first_order_setting"
SUPPLEMENTAL = "supplementary_native_representations"
STYLE = {
    ("torch_tm_flowpipe", PRIMARY): ("Torch TM (dependency preserving)", "#1f77b4", "-", "o"),
    ("flowstar", PRIMARY): ("Flow* (fixed order 1)", "#2ca02c", "--", "s"),
    ("diffreach", PRIMARY): ("DiffReach affine-dynamics path", "#d62728", "-.", "^"),
    ("torch_tm_flowpipe", "strict_common_affine"): ("Torch TM strict affine", "#1f77b4", ":", "o"),
    ("flowstar", "strict_common_affine"): ("Flow* strict affine", "#2ca02c", ":", "s"),
    ("diffreach", "strict_common_affine"): ("DiffReach strict affine (unsupported)", "#d62728", ":", "^"),
    ("torch_tm_flowpipe", "supplementary_native_representations"): ("Torch TM range-only", "#17becf", "--", "D"),
    ("flowstar", "supplementary_native_representations"): ("Flow* fixed order 2 diagnostic", "#98df8a", "--", "s"),
    ("diffreach", "supplementary_native_representations"): ("DiffReach restricted quasi-quadratic", "#ff9896", "-", "v"),
}


def _save(fig: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    # Decimal h/T values are part of the stem; Path.with_suffix would treat
    # everything after the last decimal point as an extension and overwrite
    # distinct horizons/states.
    fig.savefig(Path(str(base) + ".png"), dpi=170, bbox_inches="tight")
    fig.savefig(Path(str(base) + ".pdf"), bbox_inches="tight")
    plt.close(fig)


def _select_config(frame: pd.DataFrame, system: str, h: float, horizon: float) -> pd.DataFrame:
    return frame[
        (frame["system"] == system)
        & np.isclose(frame["h"], h, rtol=0.0, atol=1e-12)
        & np.isclose(frame["horizon"], horizon, rtol=0.0, atol=1e-12)
    ]


def _style(tool: str, protocol: str) -> tuple[str, str, str, str]:
    return STYLE.get((tool, protocol), (f"{tool} / {protocol}", "#555555", "-", "o"))


def _through_first_failure(values: pd.DataFrame) -> pd.DataFrame:
    if values.empty:
        return values
    first = values.iloc[0]
    failure_time = pd.to_numeric(first.get("first_failure_time"), errors="coerce")
    if pd.notna(failure_time) and float(failure_time) > 0:
        return values[values["time"] <= float(failure_time) + 1.0e-12]
    return values


def _state_plots(
    raw: pd.DataFrame,
    references: pd.DataFrame,
    trajectories: pd.DataFrame,
    *,
    system: str,
    h: float,
    horizon: float,
    state_index: int,
    protocol: str,
    output: Path,
) -> None:
    config = _select_config(raw, system, h, horizon)
    ref = _select_config(references, system, h, horizon)
    traj = _select_config(trajectories, system, h, horizon)
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    state_traj = traj[traj["state_index"] == state_index]
    for _, values in list(state_traj.groupby("trajectory_id"))[:8]:
        ax.plot(values["time"], values["value"], color="0.65", lw=0.55, alpha=0.6, zorder=1)
    state_ref = ref[ref["state_index"] == state_index].sort_values("time")
    if not state_ref.empty:
        ax.fill_between(
            state_ref["time"], state_ref["lower"], state_ref["upper"],
            color="black", alpha=0.09, label="exact endpoint hull", zorder=1,
        )
        ax.plot(state_ref["time"], state_ref["lower"], color="black", lw=0.8)
        ax.plot(state_ref["time"], state_ref["upper"], color="black", lw=0.8)
    protocol_data = config[
        (config["protocol"] == protocol)
        & (config["state_index"] == state_index)
        & (config["interval_kind"] == "endpoint")
    ]
    for run_id, values in protocol_data.groupby("run_id"):
        del run_id
        values = _through_first_failure(
            values.dropna(subset=["lower", "upper"]).sort_values("time")
        )
        if values.empty:
            continue
        first = values.iloc[0]
        label, color, linestyle, marker = _style(first["tool"], first["protocol"])
        ax.fill_between(values["time"], values["lower"], values["upper"], color=color, alpha=0.17)
        ax.plot(values["time"], values["lower"], color=color, ls=linestyle, lw=1.2, marker=marker, markevery=max(1, len(values) // 12), ms=2.5)
        ax.plot(values["time"], values["upper"], color=color, ls=linestyle, lw=1.2, label=label)
    failures = config[config["protocol"] == protocol].drop_duplicates("run_id")
    for _, failure in failures.iterrows():
        failure_time = failure["first_failure_time"]
        if pd.notna(failure_time) and str(failure_time) != "" and float(failure_time) > 0:
            label, color, _, _ = _style(failure["tool"], failure["protocol"])
            ax.axvline(float(failure_time), color=color, ls=":", lw=1.0)
            ax.text(float(failure_time), ax.get_ylim()[1], " failure", color=color, fontsize=7, va="top")
    protocol_title = "primary" if protocol == PRIMARY else "supplemental"
    ax.set(
        title=f"{system}: state {state_index}, h={h:g}, T={horizon:g} ({protocol_title})",
        xlabel="time",
        ylabel="enclosure",
    )
    ax.grid(alpha=0.22)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=7, loc="best")
    _save(
        fig,
        output
        / f"enclosure_{protocol_title}_{system}_h{h:g}_T{horizon:g}_state{state_index}",
    )

    width_data = config[
        (config["protocol"] == protocol)
        & (config["state_index"] == state_index)
        & (config["interval_kind"] == "endpoint")
    ]
    for logarithmic in (False, True):
        fig, ax = plt.subplots(figsize=(8.4, 4.5))
        if not state_ref.empty:
            exact_width = state_ref["upper"] - state_ref["lower"]
            ax.plot(state_ref["time"], exact_width, color="black", lw=1.0, label="exact endpoint width")
        for _, values in width_data.groupby("run_id"):
            values = _through_first_failure(
                values.dropna(subset=["width"]).sort_values("time")
            )
            if values.empty:
                continue
            first = values.iloc[0]
            label, color, linestyle, marker = _style(first["tool"], first["protocol"])
            y = values["width"].clip(lower=np.finfo(float).tiny) if logarithmic else values["width"]
            ax.plot(
                values["time"], y, color=color, ls=linestyle, marker=marker,
                markevery=max(1, len(values) // 12), ms=2.5, lw=1.3, label=label,
            )
        if logarithmic:
            ax.set_yscale("log")
        ax.set(
            title=(
                f"{system}: endpoint width, state {state_index}, h={h:g}, "
                f"T={horizon:g} ({protocol_title})"
            ),
            xlabel="time", ylabel="width",
        )
        ax.grid(alpha=0.22, which="both")
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=7)
        suffix = "log" if logarithmic else "linear"
        _save(
            fig,
            output
            / (
                f"width_{suffix}_{protocol_title}_{system}_h{h:g}_"
                f"T{horizon:g}_state{state_index}"
            ),
        )


def _phase_plot(
    raw: pd.DataFrame,
    trajectories: pd.DataFrame,
    *,
    system: str,
    h: float,
    horizon: float,
    protocol: str,
    output: Path,
) -> None:
    config = _select_config(raw, system, h, horizon)
    traj = _select_config(trajectories, system, h, horizon)
    fig, ax = plt.subplots(figsize=(6.2, 6.0))
    for _, values in list(traj.groupby("trajectory_id"))[:10]:
        pivot = values.pivot_table(index="time", columns="state_index", values="value")
        if {0, 1}.issubset(pivot.columns):
            ax.plot(pivot[0], pivot[1], color="0.6", lw=0.55, alpha=0.55)
    endpoints = config[
        (config["protocol"] == protocol) & (config["interval_kind"] == "endpoint")
    ]
    for run_id, values in endpoints.groupby("run_id"):
        values = _through_first_failure(values.sort_values("time"))
        first = values.iloc[0]
        label, color, _, _ = _style(first["tool"], first["protocol"])
        by_step = values.pivot_table(
            index=["step_index", "time"], columns="state_index", values=["lower", "upper"]
        )
        if by_step.empty or ("lower", 0) not in by_step or ("lower", 1) not in by_step:
            continue
        stride = max(1, len(by_step) // 25)
        selected = by_step.iloc[::stride]
        for sequence, (_, row) in enumerate(selected.iterrows()):
            xlo, xhi = row[("lower", 0)], row[("upper", 0)]
            ylo, yhi = row[("lower", 1)], row[("upper", 1)]
            ax.add_patch(
                Rectangle(
                    (xlo, ylo), xhi - xlo, yhi - ylo,
                    facecolor=color, edgecolor=color, alpha=0.04 + 0.14 * (sequence + 1) / len(selected),
                    lw=0.55,
                )
            )
        centers_x = 0.5 * (by_step[("lower", 0)] + by_step[("upper", 0)])
        centers_y = 0.5 * (by_step[("lower", 1)] + by_step[("upper", 1)])
        ax.plot(centers_x, centers_y, color=color, lw=1.1, label=label)
    protocol_title = "primary" if protocol == PRIMARY else "supplemental"
    ax.set(
        title=(
            f"{system}: phase-plane enclosures, h={h:g}, T={horizon:g} "
            f"({protocol_title})"
        ),
        xlabel="state 0", ylabel="state 1",
    )
    ax.grid(alpha=0.2)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=7)
    _save(fig, output / f"phase_{protocol_title}_{system}_h{h:g}_T{horizon:g}")


def _bar_and_inflation(
    summary: pd.DataFrame,
    *,
    system: str,
    protocol: str,
    output: Path,
) -> None:
    all_data = summary[
        (summary["system"] == system)
        & (summary["protocol"] == protocol)
    ]
    data = all_data[all_data["status"] == "certified_ok"]
    if data.empty:
        return
    configs = sorted(
        {
            (float(row.h), float(row.horizon), int(row.state_index))
            for row in all_data.itertuples()
        }
    )
    labels = [f"h={h:g}\nT={t:g}\nx{s}" for h, t, s in configs]
    methods = [
        ("torch_tm_flowpipe", "#1f77b4"),
        ("flowstar", "#2ca02c"),
        ("diffreach", "#d62728"),
    ]
    x = np.arange(len(configs))
    width = 0.24
    fig, ax = plt.subplots(figsize=(max(9.0, len(configs) * 0.55), 5.0))
    for method_index, (tool, color) in enumerate(methods):
        values = []
        for h, horizon, state_index in configs:
            match = data[
                (data["tool"] == tool)
                & np.isclose(data["h"], h)
                & np.isclose(data["horizon"], horizon)
                & (data["state_index"] == state_index)
            ]
            values.append(float(match["final_endpoint_width"].iloc[0]) if not match.empty and pd.notna(match["final_endpoint_width"].iloc[0]) else np.nan)
        label, _, _, _ = _style(tool, protocol)
        ax.bar(x + (method_index - 1) * width, values, width, label=label, color=color, alpha=0.82)
    protocol_title = "primary" if protocol == PRIMARY else "supplemental"
    ax.set(
        title=f"{system}: final endpoint widths ({protocol_title})",
        ylabel="width",
        xticks=x,
        xticklabels=labels,
    )
    ax.tick_params(axis="x", labelsize=7)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(fontsize=7)
    ax.text(0.01, 0.98, "Missing bars denote unsupported/failed configurations.", transform=ax.transAxes, va="top", fontsize=7)
    _save(fig, output / f"final_width_bars_{protocol_title}_{system}")

    inflation = data.dropna(subset=["exact_inflation_ratio"])
    if inflation.empty:
        return
    fig, ax = plt.subplots(figsize=(max(8.0, len(configs) * 0.5), 4.8))
    for tool, color in methods:
        tool_data = inflation[inflation["tool"] == tool].sort_values(["horizon", "h", "state_index"])
        if tool_data.empty:
            continue
        label, _, linestyle, marker = _style(tool, protocol)
        xlabels = [
            f"{row.h:g}/{row.horizon:g}/x{int(row.state_index)}"
            for row in tool_data.itertuples()
        ]
        ax.plot(
            range(len(tool_data)), tool_data["exact_inflation_ratio"],
            color=color, ls=linestyle, marker=marker, label=label,
        )
        ax.set_xticks(range(len(tool_data)), xlabels, rotation=70, fontsize=7)
    ax.axhline(1.0, color="black", lw=0.8)
    ax.set(
        title=f"{system}: exact endpoint inflation ratio ({protocol_title})",
        xlabel="h / T / state",
        ylabel="enclosure width / exact width",
    )
    ax.grid(alpha=0.2)
    ax.legend(fontsize=7)
    _save(fig, output / f"inflation_ratio_{protocol_title}_{system}")


def _runtime_plots(summary: pd.DataFrame, output: Path) -> None:
    metrics = [
        ("build_time_s", "source/build or compile time"),
        ("warmup_time_s", "first-call / JIT-warmup time"),
        ("steady_runtime_median_s", "steady median runtime"),
    ]
    for protocol, protocol_title in (
        (PRIMARY, "primary"),
        (SUPPLEMENTAL, "supplemental"),
    ):
        runs = summary[
            (summary["protocol"] == protocol) & (summary["status"] == "certified_ok")
        ].drop_duplicates("run_id")
        for column, title in metrics:
            values = runs.dropna(subset=[column]).copy()
            values = values[values[column] >= 0]
            if values.empty:
                continue
            values["label"] = values.apply(
                lambda row: f"{row['tool']}\n{row['system']}\nh={row['h']:g},T={row['horizon']:g}",
                axis=1,
            )
            colors = values["tool"].map(
                {"torch_tm_flowpipe": "#1f77b4", "flowstar": "#2ca02c", "diffreach": "#d62728"}
            )
            fig, ax = plt.subplots(figsize=(max(10.0, len(values) * 0.34), 5.1))
            ax.bar(range(len(values)), values[column], color=colors, alpha=0.82)
            if bool((values[column] > 0).any()):
                ax.set_yscale("log")
                ylabel = "seconds (log scale)"
            else:
                ylabel = "seconds"
            ax.set(
                title=f"{title} ({protocol_title})",
                ylabel=ylabel,
                xticks=range(len(values)),
                xticklabels=values["label"],
            )
            ax.tick_params(axis="x", rotation=75, labelsize=6)
            ax.grid(axis="y", which="both", alpha=0.2)
            ax.text(
                0.01, 0.98,
                "CPU-only run on this host; compile/JIT/steady quantities are intentionally separated.",
                transform=ax.transAxes, va="top", fontsize=7,
            )
            _save(fig, output / f"runtime_{protocol_title}_{column}")


def _failure_plot(summary: pd.DataFrame, output: Path) -> None:
    data = summary[summary["system"] == "van_der_pol"].drop_duplicates("run_id")
    if data.empty:
        return
    grouped = (
        data.groupby(["tool", "protocol"], as_index=False)["successful_horizon"].max()
        .sort_values(["protocol", "tool"])
    )
    labels = [f"{row.tool}\n{row.protocol}" for row in grouped.itertuples()]
    colors = grouped["tool"].map(
        {"torch_tm_flowpipe": "#1f77b4", "flowstar": "#2ca02c", "diffreach": "#d62728"}
    )
    fig, ax = plt.subplots(figsize=(max(8.0, len(grouped) * 1.0), 4.8))
    ax.bar(range(len(grouped)), grouped["successful_horizon"], color=colors, alpha=0.82)
    ax.set(
        title="Van der Pol: maximum successfully validated horizon",
        ylabel="successful horizon",
        xticks=range(len(grouped)),
        xticklabels=labels,
    )
    ax.tick_params(axis="x", rotation=35, labelsize=7)
    ax.grid(axis="y", alpha=0.2)
    _save(fig, output / "failure_horizon_van_der_pol")


def _semantics_figure(summary: pd.DataFrame, output: Path) -> None:
    rows = (
        summary.drop_duplicates(["tool", "protocol", "retained_basis"])
        [["tool", "protocol", "retained_basis", "effective_max_degree", "truncate_to_affine", "nonzero_Lt"]]
        .sort_values(["protocol", "tool"])
    )
    if rows.empty:
        return
    display = rows.copy()
    display["retained_basis"] = display["retained_basis"].map(lambda value: str(value)[:58])
    fig, ax = plt.subplots(figsize=(13.0, 0.55 * len(display) + 1.8))
    ax.axis("off")
    table = ax.table(
        cellText=display.values,
        colLabels=["tool", "protocol", "retained basis", "effective degree", "affine flag", "final Lt"],
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.35)
    ax.set_title("Measured/requested representation semantics", pad=12)
    _save(fig, output / "representation_semantics")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", default=str(HERE / "benchmark_spec.yaml"))
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    spec = load_spec(args.spec)
    output_dir = output_dir_from_args(args.output_dir)
    plots = output_dir / "plots"
    raw = pd.read_csv(output_dir / "raw_results.csv")
    summary = pd.read_csv(output_dir / "run_summary.csv")
    references = pd.read_csv(output_dir / "references.csv")
    trajectories = pd.read_csv(output_dir / "trajectories.csv")
    numeric_columns = [
        "h", "horizon", "state_index", "step_index", "time", "lower", "upper", "width",
        "first_failure_time", "successful_horizon",
    ]
    for frame in (raw, references, trajectories):
        for column in numeric_columns:
            if column in frame:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in [
        "h", "horizon", "state_index", "final_endpoint_width", "exact_inflation_ratio",
        "build_time_s", "warmup_time_s", "steady_runtime_median_s", "successful_horizon",
    ]:
        if column in summary:
            summary[column] = pd.to_numeric(summary[column], errors="coerce")
    configurations = (
        raw[["system", "h", "horizon"]]
        .drop_duplicates()
        .sort_values(["system", "h", "horizon"])
        .to_dict("records")
    )
    for config in configurations:
        system = str(config["system"])
        h, horizon = float(config["h"]), float(config["horizon"])
        for state_index in range(len(spec["systems"][system]["state_names"])):
            for protocol in (PRIMARY, SUPPLEMENTAL):
                _state_plots(
                    raw,
                    references,
                    trajectories,
                    system=system,
                    h=h,
                    horizon=horizon,
                    state_index=state_index,
                    protocol=protocol,
                    output=plots,
                )
        if len(spec["systems"][system]["state_names"]) == 2:
            for protocol in (PRIMARY, SUPPLEMENTAL):
                _phase_plot(
                    raw,
                    trajectories,
                    system=system,
                    h=h,
                    horizon=horizon,
                    protocol=protocol,
                    output=plots,
                )
    for system in spec["systems"]:
        for protocol in (PRIMARY, SUPPLEMENTAL):
            _bar_and_inflation(
                summary,
                system=system,
                protocol=protocol,
                output=plots,
            )
    _runtime_plots(summary, plots)
    _failure_plot(summary, plots)
    _semantics_figure(summary, plots)
    print(f"generated {len(list(plots.glob('*.png')))} PNG and {len(list(plots.glob('*.pdf')))} PDF plots")


if __name__ == "__main__":
    main()
