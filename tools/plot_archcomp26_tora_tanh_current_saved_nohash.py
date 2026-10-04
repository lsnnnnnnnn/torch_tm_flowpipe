#!/usr/bin/env python3
"""Draw current P3 and historical author TORA tanh ranges; never run a solver."""

import argparse
import csv
import importlib.util
import json
from pathlib import Path
import time


CURRENT = "tora_reach_tanh_official2026_mat_u11_workingp3_full500_20261002_001"
HISTORY = "tora_reach_tanh_historical_fourway_saved_20261002"
ARMS = (("current_p3", "Current working P3", "#6a3d9a", "-"),
        ("huan", "Historical Huan", "#d95f02", "--"),
        ("xiangru", "Historical Xiangru", "#009e73", ":"),
        ("native", "Historical native", "#1f77b4", "-."),
        ("old_p3", "Historical P3 (different engine)", "#777777", "--"))
TARGET = {"x1": [-.1, .2], "x2": [-.9, -.6]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    history, current = args.root / HISTORY, args.root / CURRENT
    # Reuse the preserved read-only parser, without calling its exporter or main.
    spec = importlib.util.spec_from_file_location("tanh_saved_parser", history / "plot_saved_ranges.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    paths = {key: (current / "ranges.bin" if key == "current_p3"
                   else history / "raw" / f"{key}_ranges.bin") for key, *_ in ARMS}
    ranges = {key: module.read_ranges(path) for key, path in paths.items()}
    result = json.loads((current / "RESULT.json").read_text())
    if (result["status"] != "completed_full_numerical_horizon" or not result["full_horizon_completed"]
            or result["observed_substeps"] != 500 or result["accepted_substeps"] != 500):
        raise ValueError("Current P3 receipt does not record the full numeric horizon")
    start = json.loads((current / "START.json").read_text())
    initial = {f"x{i + 1}": b for i, b in enumerate(start["contract"]["initial_box"][:4])}
    if (start["contract"]["internal_control"] != "u = 11 * f(x) + 0"
            or start["contract"]["ode_substeps"] != 500
            or initial != {"x1": [-.77, -.75], "x2": [-.45, -.43],
                           "x3": [.51, .54], "x4": [-.3, -.28]}):
        raise ValueError("Unexpected named current TORA contract")
    old_audit = json.loads((history / "AUDIT.json").read_text())
    if old_audit["raw_records_per_method"] != 500:
        raise ValueError("Unexpected historical range audit")
    if args.output_dir.exists():
        raise ValueError("Choose a new output directory; archived plots are preserved")
    args.output_dir.mkdir(parents=True)
    geometry = {
        "schema": "tora-tanh-current-and-historical-saved-v1",
        "contract": "official ReLU^3/tanh, u=11f, full four-state initial box",
        "initial_at_t0": initial, "target_at_T5_only": TARGET,
        "units": {"t": "s", "state": "not declared in saved plot contract; no conversion"},
        "step_s": .01, "steps": 500,
        "series": [{"key": key, "label": label, "source": str(paths[key]),
                    "generation": "2026-10-02 current working P3" if key == "current_p3" else "historical",
                    "color": color, "style": style, "records": ranges[key]}
                   for key, label, color, style in ARMS],
        "main_methods": [key for key, *_ in ARMS[:4]],
        "historical_p3_role": "generation comparison only; not the main P3 method",
    }
    (args.output_dir / "geometry.json").write_text(json.dumps(geometry, indent=2, allow_nan=False) + "\n")
    with (args.output_dir / "saved_bounds.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["method", "generation", "state", "step", "t_start", "t_end",
                         "tube_lo", "tube_hi", "tube_width", "endpoint_lo", "endpoint_hi", "endpoint_width", "source"])
        for series in geometry["series"]:
            for step, states in enumerate(series["records"], 1):
                for state, bounds in enumerate(states, 1):
                    lo, hi = bounds["tube"]
                    elo, ehi = bounds["endpoint"]
                    writer.writerow([series["key"], series["generation"], f"x{state}", step,
                                     (step - 1) * .01, step * .01, lo, hi, hi - lo,
                                     elo, ehi, ehi - elo, series["source"]])
    with (args.output_dir / "absolute_widths.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["method", "state", "terminal_endpoint_lo", "terminal_endpoint_hi", "terminal_endpoint_width",
                         "last_tube_lo", "last_tube_hi", "last_tube_width", "full_tube_lo", "full_tube_hi", "full_tube_width"])
        for key, *_ in ARMS:
            for state in range(4):
                elo, ehi = ranges[key][-1][state]["endpoint"]
                lo, hi = ranges[key][-1][state]["tube"]
                full_lo = min(r[state]["tube"][0] for r in ranges[key])
                full_hi = max(r[state]["tube"][1] for r in ranges[key])
                writer.writerow([key, f"x{state + 1}", elo, ehi, ehi - elo, lo, hi, hi - lo,
                                 full_lo, full_hi, full_hi - full_lo])
    validation_s = time.perf_counter() - started
    render_started = time.perf_counter()
    render(geometry, args.output_dir)
    endpoint_in_target = {key: all(TARGET[f"x{i+1}"][0] <= ranges[key][-1][i]["endpoint"][0]
                                             <= ranges[key][-1][i]["endpoint"][1] <= TARGET[f"x{i+1}"][1]
                                  for i in (0, 1)) for key, *_ in ARMS}
    receipt = {
        "status": "saved_data_validation_passed", "solver_executed": False,
        "source_files": [{"path": str(path), "bytes": path.stat().st_size} for path in paths.values()],
        "current_receipt": str(current / "RESULT.json"), "current_contract": str(current / "START.json"),
        "historical_receipt_audit": str(history / "AUDIT.json"),
        "main_range_records": 2000, "supplementary_historical_p3_records": 500,
        "csv_rows": 10000, "raw_record_bytes": 152,
        "validated": ["each source has 500 unique ordered records", "complete 0 to 5 s grid",
                      "finite ordered tube/endpoint", "endpoint within same-step tube", "no interpolation"],
        "initial_layer": "t=0 only", "target_layer": "T=5 only; sufficient numerical endpoint observation for within-5s reach",
        "endpoint_in_target": endpoint_in_target,
        "huan_xiangru_direct_saved_values_equal": ranges["huan"] == ranges["xiangru"],
        "property_and_certificate": "Current P3 and historical GPU receipts have no author property verdict; historical native author endpoint checker is VERIFIED. No independent floating-point NNCS certificate.",
        "validation_and_export_s": validation_s, "rendering_s": time.perf_counter() - render_started,
        "outputs": [p.name for p in sorted(args.output_dir.iterdir())],
    }
    (args.output_dir / "AUDIT.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": "plotted_saved_data", "output": str(args.output_dir), "records": 2500}))


def render(g, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series = {row["key"]: row for row in g["series"]}
    edges = [i * .01 for i in range(501)]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.4), gridspec_kw={"width_ratios": [3, 1.8]})
    for dim in (0, 1):
        name = f"x{dim+1}"
        ax, diff = axes[dim]
        for row in g["series"][:4]:
            for side in (0, 1):
                values = [r[dim]["tube"][side] for r in row["records"]]
                ax.stairs(values, edges, baseline=None, color=row["color"], linestyle=row["style"],
                          linewidth=1.4, label=row["label"] if side == 0 else None)
        ax.plot([0, 0], g["initial_at_t0"][name], color="black", linewidth=4, label="Initial set (t=0)")
        ax.plot([5, 5], TARGET[name], color="#b2188b", linewidth=4, label="Target (T=5 only)")
        ax.set(xlim=(-.025, 5.025), xlabel="t (s)", ylabel=name, title=f"{name}: saved whole-step tube")
        ax.grid(alpha=.2)
        if dim == 0:
            ax.legend(fontsize=8, ncol=2, loc="upper left")
        reference = series["huan"]["records"][-1][dim]["endpoint"]
        for index, row in enumerate(g["series"][:4]):
            for side, marker in enumerate(("o", "^")):
                value = row["records"][-1][dim]["endpoint"][side] - reference[side]
                diff.plot(value, index, marker=marker, color=row["color"], markersize=7)
        diff.axvline(0, color=".4", linestyle=":")
        diff.set(yticks=range(4), yticklabels=[r["label"] for r in g["series"][:4]],
                 ylim=(3.5, -.5), xlabel="Endpoint bound minus Huan", title=f"T=5 {name}: circle lower, triangle upper")
        diff.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        diff.grid(axis="x", alpha=.2)
    fig.suptitle("TORA reach-tanh: current P3 + three historical author methods · official u=11f", fontsize=14)
    fig.subplots_adjust(left=.065, right=.975, top=.91, bottom=.15, wspace=.4, hspace=.35)
    fig.text(.065, .035, "All four methods have 500 saved 0.01 s intervals. Huan/Xiangru coincide; no missing interval is filled.\n"
             "Initial is shown at t=0; target only at T=5. Mixed generations and saved axis boxes imply neither independent proof nor speed ranking.\n"
             "Historical P3 is reserved for the separate generation-comparison figure. State units were not declared in the saved plot contract.", fontsize=8)
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"tora_tanh_current_fourway.{suffix}", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for dim in (0, 1):
        for key in ("current_p3", "old_p3"):
            row = series[key]
            for side in (0, 1):
                axes[0, dim].stairs([r[dim]["tube"][side] for r in row["records"]], edges,
                                    baseline=None, color=row["color"], linestyle=row["style"],
                                    label=row["label"] if side == 0 else None)
        axes[0, dim].plot([0, 0], g["initial_at_t0"][f"x{dim+1}"], color="black", linewidth=4)
        axes[0, dim].plot([5, 5], TARGET[f"x{dim+1}"], color="#b2188b", linewidth=4)
        axes[0, dim].set(title=f"x{dim+1}: P3 generations, saved tubes", ylabel=f"x{dim+1}")
        axes[0, dim].legend(fontsize=8)
        delta = [(new[dim]["tube"][1] - new[dim]["tube"][0]) -
                 (old[dim]["tube"][1] - old[dim]["tube"][0])
                 for new, old in zip(series["current_p3"]["records"], series["old_p3"]["records"])]
        axes[1, dim].stairs(delta, edges, baseline=None, color="#6a3d9a")
        axes[1, dim].axhline(0, color=".4", linestyle=":")
        axes[1, dim].set(title=f"x{dim+1}: current minus historical tube width", ylabel="Width difference", xlabel="t (s)")
        axes[1, dim].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    for ax in axes.flat:
        ax.grid(alpha=.2)
        ax.set_xlim(-.025, 5.025)
    fig.suptitle("TORA reach-tanh P3 generation comparison · supplementary, no timing inference", fontsize=14)
    fig.text(.07, .025, "Old P3 uses engine_linear_leaf_v2; current P3 is the 2026-10-02 working-P3 run. Same named model/initial box.\n"
             "Black t=0: initial set; magenta T=5: target. Saved numerical boxes; no independent NNCS certificate.", fontsize=8)
    fig.tight_layout(rect=(.02, .07, .98, .95))
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"tora_tanh_p3_generations.{suffix}", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
