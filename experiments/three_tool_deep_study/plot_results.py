#!/usr/bin/env python3
"""Generate the eighteen mandatory deep-study figures."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
COLORS = {
    "torch_tm_flowpipe": "#d95f02",
    "torch_common_engine": "#d95f02",
    "diffreach": "#1b9e77",
    "flowstar": "#7570b3",
}


def _read(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _f(value: Any, default: float = math.nan) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _finish(
    figure: plt.Figure,
    path: Path,
    title: str,
    *,
    legend: bool = True,
) -> None:
    figure.suptitle(title, fontsize=12)
    if legend:
        handles: list[Any] = []
        labels: list[str] = []
        for axis in figure.axes:
            axis_handles, axis_labels = axis.get_legend_handles_labels()
            for handle, label in zip(axis_handles, axis_labels):
                if label and label not in labels:
                    handles.append(handle)
                    labels.append(label)
        if handles:
            figure.legend(
                handles,
                labels,
                loc="lower center",
                ncol=min(4, len(handles)),
                fontsize=7,
            )
            figure.subplots_adjust(bottom=0.17)
    figure.tight_layout(rect=(0, 0.06 if legend else 0, 1, 0.94))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _empty(axis: plt.Axes, message: str = "No eligible data") -> None:
    axis.text(0.5, 0.5, message, ha="center", va="center")
    axis.set_axis_off()


def _system_axes() -> tuple[plt.Figure, dict[str, plt.Axes]]:
    systems = ["riccati", "harmonic", "coupled_quadratic", "van_der_pol"]
    figure, axes = plt.subplots(2, 2, figsize=(10, 7))
    return figure, {
        system: axis for system, axis in zip(systems, axes.ravel())
    }


def _group_max(
    rows: Iterable[Mapping[str, Any]], fields: tuple[str, ...], value: str
) -> dict[tuple[str, ...], float]:
    grouped: dict[tuple[str, ...], list[float]] = defaultdict(list)
    for row in rows:
        number = _f(row.get(value))
        if math.isfinite(number):
            grouped[
                tuple(str(row.get(field, "")) for field in fields)
            ].append(number)
    return {key: max(values) for key, values in grouped.items()}


def plot_one_step(
    rows: list[dict[str, Any]], plots: Path, kind: str, index: int
) -> None:
    figure, axes = _system_axes()
    for system, axis in axes.items():
        selected = [
            row
            for row in rows
            if row.get("system") == system
            and row.get("interval_kind") == kind
        ]
        grouped = _group_max(
            selected, ("tool", "variant", "h"), "width"
        )
        series: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(
            list
        )
        for (tool, variant, h), width in grouped.items():
            series[(tool, variant)].append((_f(h), width))
        for (tool, variant), values in sorted(series.items()):
            values.sort()
            axis.plot(
                [item[0] for item in values],
                [item[1] for item in values],
                marker="o",
                ms=3,
                color=COLORS.get(tool),
                alpha=0.75,
                label=f"{tool}:{variant}",
            )
        axis.set_title(system)
        axis.set_xlabel("h")
        axis.set_ylabel("max component width")
        axis.set_yscale("log")
        if not series:
            _empty(axis)
    label = "tube" if kind == "tube" else "raw endpoint"
    _finish(
        figure,
        plots / f"{index:02d}_one_step_{kind}_width_vs_h.png",
        f"One-step {label} width versus h",
    )


def plot_inflation(rows: list[dict[str, Any]], plots: Path) -> None:
    figure, axes_array = plt.subplots(1, 2, figsize=(10, 4))
    for system, axis in zip(("riccati", "harmonic"), axes_array):
        selected = [
            row
            for row in rows
            if row.get("system") == system
            and row.get("interval_kind") == "endpoint_raw"
            and math.isfinite(_f(row.get("exact_inflation_ratio")))
        ]
        series: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for row in selected:
            label = f"{row.get('tool')}:{row.get('variant')}"
            series[label].append(
                (_f(row.get("h")), _f(row.get("exact_inflation_ratio")))
            )
        for label, values in sorted(series.items()):
            values.sort()
            tool = label.split(":", 1)[0]
            axis.plot(
                [x for x, _ in values],
                [y for _, y in values],
                marker="o",
                ms=3,
                label=label,
                color=COLORS.get(tool),
                alpha=0.75,
            )
        axis.axhline(1.0, color="black", lw=0.8, ls="--")
        axis.set_title(system)
        axis.set_xlabel("h")
        axis.set_ylabel("enclosure width / exact width")
        if not series:
            _empty(axis)
    _finish(
        figure,
        plots / "03_exact_inflation_ratios.png",
        "Exact endpoint inflation ratios",
    )


def plot_carry(
    rows: list[dict[str, Any]], plots: Path, protocol: str, index: int
) -> None:
    figure, axes = _system_axes()
    for system, axis in axes.items():
        selected = [
            row
            for row in rows
            if row.get("system") == system
            and row.get("protocol") == protocol
            and row.get("interval_kind") in {"endpoint_raw", "endpoint"}
        ]
        grouped = _group_max(
            selected, ("tool", "variant", "h", "time"), "width"
        )
        series: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for (tool, variant, h, time_value), width in grouped.items():
            series[f"{tool}:{variant}:h={h}"].append(
                (_f(time_value), width)
            )
        for label, values in sorted(series.items()):
            values.sort()
            axis.plot(
                [x for x, _ in values],
                [y for _, y in values],
                label=label,
                color=COLORS.get(label.split(":", 1)[0]),
                alpha=0.8,
            )
        axis.set_title(system)
        axis.set_xlabel("absolute time")
        axis.set_ylabel("max component width")
        axis.set_yscale("log")
        if not series:
            _empty(axis)
    slug = "affine" if "affine" in protocol else "box"
    _finish(
        figure,
        plots / f"{index:02d}_common_{slug}_carry_width_vs_time.png",
        f"Common {slug} carry width versus time",
    )


def plot_carry_comparison(
    affine: list[dict[str, Any]],
    box: list[dict[str, Any]],
    plots: Path,
) -> None:
    def values(rows: list[dict[str, Any]]) -> dict[tuple[str, str], float]:
        return _group_max(rows, ("tool", "system"), "width")

    av, bv = values(affine), values(box)
    labels = sorted(set(av) & set(bv))
    ratios = [bv[key] / av[key] if av[key] else math.nan for key in labels]
    figure, axis = plt.subplots(figsize=(10, 4))
    if labels:
        axis.bar(
            np.arange(len(labels)),
            ratios,
            color=[COLORS.get(tool, "#777777") for tool, _ in labels],
        )
        axis.set_xticks(
            np.arange(len(labels)),
            [f"{tool}\n{system}" for tool, system in labels],
            rotation=35,
            ha="right",
        )
        axis.axhline(1.0, color="black", lw=0.8)
        axis.set_ylabel("box width / affine width")
        axis.set_yscale("log")
    else:
        _empty(axis)
    _finish(
        figure,
        plots / "06_affine_vs_box_carry.png",
        "Wrapping loss from common box carry",
        legend=False,
    )


def plot_native_low(rows: list[dict[str, Any]], plots: Path) -> None:
    figure, axes = _system_axes()
    for system, axis in axes.items():
        selected = [
            row
            for row in rows
            if row.get("system") == system
            and row.get("interval_kind") == "endpoint_raw"
        ]
        grouped = _group_max(
            selected, ("tool", "variant", "time"), "width"
        )
        series: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for (tool, variant, time_value), width in grouped.items():
            series[f"{tool}:{variant}"].append((_f(time_value), width))
        for label, values in sorted(series.items()):
            values.sort()
            axis.plot(
                [x for x, _ in values],
                [y for _, y in values],
                label=label,
                color=COLORS.get(label.split(":", 1)[0]),
            )
        axis.set_title(system)
        axis.set_xlabel("time")
        axis.set_ylabel("max component width")
        axis.set_yscale("log")
        if not series:
            _empty(axis)
    _finish(
        figure,
        plots / "07_native_low_order_width_curves.png",
        "Native low-order width curves (bases differ)",
    )


def plot_pareto(rows: list[dict[str, Any]], plots: Path) -> None:
    figure, axes = _system_axes()
    for system, axis in axes.items():
        selected = [
            row
            for row in rows
            if row.get("system") == system
            and math.isfinite(_f(row.get("width_at_evaluation_time")))
            and _f(row.get("steady_full_configuration_time_s")) > 0
            and str(
                row.get("primary_numerical_eligible", "true")
            ).lower()
            == "true"
        ]
        for row in selected:
            tool = str(row.get("tool"))
            axis.scatter(
                _f(row["steady_full_configuration_time_s"]),
                _f(row["width_at_evaluation_time"]),
                color=COLORS.get(tool),
                marker="*" if str(row.get("width_runtime_pareto")).lower() == "true" else "o",
                s=55,
                alpha=0.8,
                label=tool,
            )
        axis.set_title(system)
        axis.set_xlabel("steady full-config runtime (s)")
        axis.set_ylabel("width at labeled time")
        axis.set_xscale("log")
        axis.set_yscale("log")
        if not selected:
            _empty(axis)
    _finish(
        figure,
        plots / "08_native_practical_width_runtime_pareto.png",
        "Native practical tradeoffs (stars = within-tool Pareto only)",
    )


def plot_horizon_runtime(rows: list[dict[str, Any]], plots: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 5))
    selected = [
        row
        for row in rows
        if _f(row.get("steady_full_configuration_time_s")) > 0
        and math.isfinite(_f(row.get("successful_horizon")))
    ]
    for row in selected:
        tool = str(row.get("tool"))
        axis.scatter(
            _f(row["steady_full_configuration_time_s"]),
            _f(row["successful_horizon"]),
            color=COLORS.get(tool),
            marker=(
                "x"
                if str(
                    row.get("primary_numerical_eligible", "true")
                ).lower()
                != "true"
                else "o"
            ),
            label=tool,
            alpha=0.75,
        )
    if selected:
        axis.set_xscale("log")
        axis.set_xlabel("steady full-config runtime (s)")
        axis.set_ylabel("successful horizon")
    else:
        _empty(axis)
    _finish(
        figure,
        plots / "09_successful_horizon_vs_runtime.png",
        "Successful horizon versus runtime (x = numerical-ineligible audit row)",
    )


def plot_decomposition(rows: list[dict[str, Any]], plots: Path) -> None:
    selected = [
        row
        for row in rows
        if row.get("interval_kind") == "endpoint_raw"
    ][:36]
    figure, axis = plt.subplots(figsize=(11, 5))
    if selected:
        labels = [
            f"{row.get('tool')}:{row.get('variant')}:{row.get('system')}"
            for row in selected
        ]
        poly = [_f(row.get("polynomial_range_width"), 0.0) for row in selected]
        rem = [
            _f(row.get("independent_interval_remainder_width"), 0.0)
            for row in selected
        ]
        residual = [
            _f(row.get("unattributed_dependency_or_reset_width"), 0.0)
            for row in selected
        ]
        x = np.arange(len(selected))
        axis.bar(x, poly, label="polynomial range")
        axis.bar(x, rem, bottom=poly, label="independent remainder")
        axis.bar(
            x,
            residual,
            bottom=np.asarray(poly) + np.asarray(rem),
            label="dependency/reset residual",
        )
        axis.set_xticks(x, labels, rotation=75, ha="right", fontsize=6)
        axis.set_ylabel("width contribution")
        axis.set_yscale("log")
    else:
        _empty(axis)
    _finish(
        figure,
        plots / "10_polynomial_remainder_decomposition.png",
        "Polynomial/remainder width decomposition",
    )


def plot_monomials(rows: list[dict[str, Any]], plots: Path) -> None:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        encoded = row.get("monomial_families")
        if not encoded:
            continue
        try:
            families = json.loads(encoded)
        except (TypeError, json.JSONDecodeError):
            continue
        for family, count in families.items():
            counts[str(row.get("tool"))][family] += int(count)
    tools = sorted(counts)
    families = sorted(
        {family for value in counts.values() for family in value}
    )
    figure, axis = plt.subplots(figsize=(9, 5))
    if tools and families:
        bottom = np.zeros(len(tools))
        for family in families:
            values = np.asarray([counts[tool][family] for tool in tools])
            axis.bar(tools, values, bottom=bottom, label=family)
            bottom += values
        axis.set_ylabel("term occurrences over one-step sweep")
    else:
        _empty(axis)
    _finish(
        figure,
        plots / "11_monomial_family_support.png",
        "Monomial-family support by tool",
    )


def _ablation_bars(
    rows: list[dict[str, Any]],
    plots: Path,
    filename: str,
    title: str,
    tool: str,
    value_field: str = "width",
) -> None:
    selected = [row for row in rows if row.get("tool") == tool]
    grouped = _group_max(
        selected, ("variant", "system"), value_field
    )
    figure, axis = plt.subplots(figsize=(10, 5))
    if grouped:
        labels = [f"{variant}\n{system}" for variant, system in grouped]
        values = list(grouped.values())
        axis.bar(np.arange(len(values)), values, color=COLORS.get(tool))
        axis.set_xticks(
            np.arange(len(values)), labels, rotation=55, ha="right", fontsize=7
        )
        axis.set_yscale("log")
        axis.set_ylabel(value_field.replace("_", " "))
    else:
        _empty(axis)
    _finish(figure, plots / filename, title, legend=False)


def plot_matched(rows: list[dict[str, Any]], plots: Path) -> None:
    grouped = _group_max(rows, ("basis", "h"), "width")
    figure, axis = plt.subplots(figsize=(8, 5))
    if grouped:
        series: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for (basis, h), width in grouped.items():
            series[basis].append((_f(h), width))
        for basis, values in sorted(series.items()):
            values.sort()
            axis.plot(
                [x for x, _ in values],
                [y for _, y in values],
                marker="o",
                label=basis,
            )
        axis.set_xlabel("h")
        axis.set_ylabel("max endpoint width")
        axis.set_yscale("log")
    else:
        _empty(axis)
    _finish(
        figure,
        plots / "15_matched_basis_results.png",
        "Matched-basis B1/B_DR/B2/B3 in one engine",
    )


def plot_defect(rows: list[dict[str, Any]], plots: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 5))
    selected = [
        row
        for row in rows
        if _f(row.get("defect_norm_inf")) > 0
        and _f(row.get("native_certified_radius")) > 0
    ]
    for row in selected:
        tool = str(row.get("tool"))
        axis.scatter(
            _f(row["native_certified_radius"]),
            _f(row["defect_norm_inf"]),
            color=COLORS.get(tool),
            label=tool,
            alpha=0.75,
        )
    if selected:
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel("native independent-remainder radius")
        axis.set_ylabel("common polynomial defect infinity bound")
    else:
        _empty(axis)
    _finish(
        figure,
        plots / "16_common_defect_vs_native_remainder.png",
        "Common defect versus native remainder",
    )


def plot_runtime(rows: list[dict[str, Any]], plots: Path) -> None:
    figure, axis = plt.subplots(figsize=(10, 5))
    if rows:
        labels = [
            f"{row.get('tool')}:{row.get('system')}:{row.get('variant')}"
            for row in rows
        ]
        build = [_f(row.get("compile_or_jit_time_s"), 0.0) for row in rows]
        steady = [
            _f(row.get("steady_full_configuration_time_s"), 0.0)
            for row in rows
        ]
        x = np.arange(len(rows))
        axis.bar(x, build, label="build/JIT/first")
        axis.bar(x, steady, bottom=build, label="steady full configuration")
        axis.set_xticks(x, labels, rotation=70, ha="right", fontsize=6)
        axis.set_yscale("log")
        axis.set_ylabel("seconds")
    else:
        _empty(axis)
    _finish(
        figure,
        plots / "17_runtime_decomposition.png",
        "Runtime decomposition",
    )


def plot_failures(rows: list[dict[str, Any]], plots: Path) -> None:
    counts = Counter(
        row.get("failure_category") or "uncategorized" for row in rows
    )
    figure, axis = plt.subplots(figsize=(8, 4))
    if counts:
        labels, values = zip(*counts.most_common())
        axis.bar(labels, values, color="#666666")
        axis.tick_params(axis="x", rotation=45)
        axis.set_ylabel("count")
    else:
        axis.bar(["no primary failures"], [0], color="#66a61e")
        axis.set_ylim(0, 1)
        axis.set_ylabel("count")
    _finish(
        figure,
        plots / "18_failure_categories.png",
        "Failure-category chart",
        legend=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    plots = output / "plots"
    one_step = _read(output / "one_step_summary.csv")
    controlled = _read(output / "controlled_raw.csv")
    native_low = _read(output / "native_low_order_summary.csv")
    pareto = _read(output / "native_pareto_summary.csv")
    components = _read(output / "component_ablation.csv")
    matched = _read(output / "matched_basis_summary.csv")
    defect = _read(output / "defect_summary.csv")
    runtime = _read(output / "runtime_summary.csv")
    failures = _read(output / "failure_summary.csv")

    plot_one_step(one_step, plots, "tube", 1)
    plot_one_step(one_step, plots, "endpoint_raw", 2)
    plot_inflation(one_step, plots)
    plot_carry(controlled, plots, "common_affine_carry", 4)
    plot_carry(controlled, plots, "common_box_carry", 5)
    plot_carry_comparison(
        _read(output / "affine_carry_summary.csv"),
        _read(output / "box_carry_summary.csv"),
        plots,
    )
    plot_native_low(native_low, plots)
    plot_pareto(pareto, plots)
    plot_horizon_runtime(pareto, plots)
    plot_decomposition(components, plots)
    plot_monomials(one_step, plots)
    native_all = _read(output / "native_raw.csv")
    _ablation_bars(
        native_all,
        plots,
        "12_torch_reset_order_ablation.png",
        "Torch reset/order ablation",
        "torch_tm_flowpipe",
    )
    _ablation_bars(
        native_all,
        plots,
        "13_diffreach_affine_quasi_symbolic_ablation.png",
        "DiffReach affine/quasi/symbolic ablation",
        "diffreach",
    )
    flow_rows = _read(output / "flowstar_component_ablation.csv")
    _ablation_bars(
        flow_rows,
        plots,
        "14_flowstar_order_step_qr_symbolic_refinement_ablation.png",
        "Flow* order/step/QR/symbolic/refinement ablation",
        "flowstar",
        value_field="endpoint_max_width",
    )
    plot_matched(matched, plots)
    plot_defect(defect, plots)
    plot_runtime(runtime, plots)
    plot_failures(failures, plots)
    generated = sorted(path.name for path in plots.glob("*.png"))
    if len(generated) != 18:
        raise RuntimeError(f"expected 18 plots, generated {len(generated)}")
    print(json.dumps({"plots": generated}, indent=2))


if __name__ == "__main__":
    main()
