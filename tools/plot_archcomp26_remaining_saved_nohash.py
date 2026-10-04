#!/usr/bin/env python3
"""Render missing report projections from frozen saved ranges; no solver calls."""

import argparse
import csv
import json
import math
import struct
import time
from pathlib import Path


METHODS = ("Huan", "Xiangru", "ours/P3", "Flow* native")
COLORS = ("#D55E00", "#009E73", "#0072B2", "#CC79A7")
STYLES = ("--", ":", "-", "-.")


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_ranges(path, states, steps, h):
    if path.suffix == ".bin":
        record = struct.Struct("<QQd" + "d" * (4 * len(states)))
        raw = path.read_bytes()
        if len(raw) != steps * record.size:
            raise ValueError(f"Unexpected saved record count: {path}")
        rows = []
        for expected, values in enumerate(record.iter_unpack(raw), 1):
            lane, step, actual_h = values[:3]
            if lane != 0 or step != expected or abs(actual_h - h) > 1e-9:
                raise ValueError(f"Unexpected saved lane/step/h: {path}, {expected}")
            pairs = [values[3 + 4 * j:7 + 4 * j] for j in range(len(states))]
            rows.append({"substep": step, "tube": [p[:2] for p in pairs],
                         "endpoint": [p[2:] for p in pairs]})
    else:
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if len(rows) != steps:
            raise ValueError(f"Unexpected saved row count: {path}")
        for expected, row in enumerate(rows, 1):
            if row["substep"] != expected or row.get("accepted") is not True:
                raise ValueError(f"Nonaccepted or missing step: {path}, {expected}")
    for row in rows:
        for kind in ("tube", "endpoint"):
            if len(row[kind]) != len(states):
                raise ValueError(f"State count differs: {path}")
            for lo, hi in row[kind]:
                if not (math.isfinite(lo) and math.isfinite(hi) and lo <= hi):
                    raise ValueError(f"Invalid saved interval: {path}")
    return rows


def saved_tables(root, definitions):
    geometry, widths, sources = [], [], []
    for instance, states, steps, h, paths in definitions:
        for method, relative in zip(METHODS, paths):
            path = root / relative
            rows = read_ranges(path, states, steps, h)
            sources.append({"instance": instance, "method": method,
                            "source": relative, "bytes": path.stat().st_size,
                            "records": len(rows), "states": list(states),
                            "record_presence_is_not_acceptance_certificate": True})
            for j, state in enumerate(states):
                for row in rows:
                    lo, hi = row["tube"][j]
                    elo, ehi = row["endpoint"][j]
                    geometry.append({"instance": instance, "method": method,
                                     "state": state, "step": row["substep"],
                                     "t_start": (row["substep"] - 1) * h,
                                     "t_end": row["substep"] * h,
                                     "tube_lo": lo, "tube_hi": hi,
                                     "endpoint_lo": elo, "endpoint_hi": ehi,
                                     "source": relative})
                elo, ehi = rows[-1]["endpoint"][j]
                lo = min(row["tube"][j][0] for row in rows)
                hi = max(row["tube"][j][1] for row in rows)
                widths.append({"instance": instance, "method": method, "state": state,
                               "time": steps * h, "endpoint_lo": elo,
                               "endpoint_hi": ehi, "endpoint_width": ehi - elo,
                               "whole_tube_lo": lo, "whole_tube_hi": hi,
                               "whole_tube_width": hi - lo,
                               "initial_boxes": 1, "source": relative})
    return geometry, widths, sources


def draw_time(plt, output, geometry, instance, state, horizon, initial, region=None):
    fig, ax = plt.subplots(figsize=(10.8, 6.0))
    for method, color, style in zip(METHODS, COLORS, STYLES):
        rows = [row for row in geometry if (row["instance"], row["method"], row["state"])
                == (instance, method, state)]
        edges = [rows[0]["t_start"], *[row["t_end"] for row in rows]]
        lower, upper = ([row["tube_lo"] for row in rows], [row["tube_hi"] for row in rows])
        ax.stairs(upper, edges, baseline=lower, fill=True, color=color, alpha=0.065)
        ax.stairs(lower, edges, baseline=None, color=color, linestyle=style, label=method, linewidth=1.25)
        ax.stairs(upper, edges, baseline=None, color=color, linestyle=style, linewidth=1.25)
    ax.vlines(0, *initial, color="black", linewidth=3.0, label="Initial interval at t=0", zorder=6)
    if region is not None:
        ax.axhspan(*region, color="#B30000", alpha=0.09,
                   label="Unsafe-box projection (full 6D test is separate)")
    limits = [*initial, *(region or ())]
    for row in geometry:
        if (row["instance"], row["state"]) == (instance, state):
            limits.extend((row["tube_lo"], row["tube_hi"]))
    low, high = min(limits), max(limits)
    margin = .05 * (high - low)
    ax.set_ylim(low - margin, high + margin)
    ax.set(xlim=(-horizon * .012, horizon), xlabel="Time (s)",
           ylabel=state + (" (m)" if instance == "docking" else ""),
           title=f"{instance.title()}: four methods, saved whole-step {state} tubes")
    ax.grid(alpha=.22)
    ax.legend(fontsize=8, loc="best")
    caption = ("Attitude: the shaded x4 interval is only a projection of the six-dimensional unsafe box.\n"
               "Overlap in this projection does not establish an unsafe trajectory. No independent NNCS certificate."
               if instance == "attitude" else
               "Docking: safety couples position and velocity; no valid one-dimensional safe band exists on sx.\n"
               "All four nonlinear-property outcomes remain UNKNOWN. No independent NNCS certificate.")
    fig.text(.5, .02, caption, ha="center", va="bottom", fontsize=8)
    fig.tight_layout(rect=(0, .08, 1, 1))
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"{instance}_fourway_t_{state}.{suffix}", dpi=180)
    plt.close(fig)


def draw_quad(plt, root, output):
    from matplotlib.patches import Rectangle
    path = root / "quad_paper_fourway_saved_20261002/terminal_12states_fourway.csv"
    with path.open(newline="") as stream:
        source_rows = list(csv.DictReader(stream))
    selected = []
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.5), gridspec_kw={"width_ratios": [1, 1.2]})
    for ax in axes:
        ax.axhspan(.94, 1.06, color="#56B4E9", alpha=.16, label="Height target band at T=5 only")
    for method, source_method, color, style in zip(METHODS,
            ("Huan", "Xiangru", "ours/P3 driver", "Flow* native"), COLORS, STYLES):
        coords = {}
        for state in ("x1", "x3"):
            matches = [r for r in source_rows if r["method"] == source_method and r["state"] == state]
            if len(matches) != 1:
                raise ValueError(f"Missing or duplicate QUAD endpoint: {source_method}, {state}")
            row = matches[0]
            lo, hi = float(row["endpoint_lo"]), float(row["endpoint_hi"])
            if not (math.isfinite(lo) and math.isfinite(hi) and lo <= hi):
                raise ValueError("Invalid QUAD endpoint")
            coords[state] = (lo, hi)
            selected.append({"method": method, "state": state, "time": 5.0,
                             "endpoint_lo": lo, "endpoint_hi": hi,
                             "source_object": row["source_object"], "source": row["source"]})
        x, z = coords["x1"], coords["x3"]
        for ax in axes:
            ax.add_patch(Rectangle((x[0], z[0]), x[1]-x[0], z[1]-z[0],
                                   fill=False, edgecolor=color, linestyle=style,
                                   linewidth=1.7, label=method))
    axes[0].add_patch(Rectangle((-.4, -.4), .8, .8, fill=False,
                               edgecolor="black", linewidth=1.5, label="Initial box at t=0"))
    for ax in axes:
        ax.set(xlim=(-3.65, 3.65), xlabel="x1 (m)", ylabel="x3 (m)")
        ax.grid(alpha=.2)
    axes[0].set(ylim=(-.48, 1.14), title="Initial box and T=5 endpoint boxes")
    axes[1].set(ylim=(.935, 1.065), title="Same endpoint boxes, height magnified")
    axes[0].legend(loc="lower left", fontsize=7)
    fig.suptitle("QUAD paper equations: x1-x3 box projections", fontsize=14)
    fig.text(.5, .015, "Box projections, not certified octagons. Huan/Xiangru endpoints coincide; no intermediate path is inferred.\n"
             "The T=5 height band does not check the full reach-and-remain property. P3 uses its driver endpoint.",
             ha="center", va="bottom", fontsize=8)
    fig.tight_layout(rect=(0, .09, 1, .96))
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"quad_paper_fourway_endpoint_x1_x3.{suffix}", dpi=180)
    plt.close(fig)
    write_csv(output / "quad_endpoint_geometry.csv", selected)
    return {"source_csv": str(path), "records": selected,
            "geometry_class": "axis_aligned_endpoint_boxes", "time": 5,
            "initial_box": {"x1": [-.4, .4], "x3": [-.4, .4]},
            "target": {"x3": [.94, 1.06], "display_time": "T=5 endpoint only"}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    start = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    campaign = "attitude_corrected_fourway_campaign_20261002_001/"
    definitions = [
        ("attitude", tuple(f"x{i}" for i in range(1, 7)), 60, .05,
         [campaign + f"first00_{m}/payload/ranges.bin" for m in ("huan", "xiangru", "ours_p3")]
         + [campaign + "first00_native/ranges.bin"]),
        ("docking", ("sx", "sy", "vx", "vy"), 400, .1,
         [f"docking_{m}_full40_001/detail/ranges.jsonl" for m in ("huan", "xiangru", "p3")]
         + ["native_docking_full40_001/ranges.bin"]),
    ]
    geometry, widths, sources = saved_tables(args.root, definitions)
    write_csv(args.output_dir / "saved_geometry.csv", geometry)
    write_csv(args.output_dir / "absolute_widths.csv", widths)
    reduced = time.perf_counter()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    draw_time(plt, args.output_dir, geometry, "attitude", "x4", 3, (-.75, -.74), (-.7, -.6))
    draw_time(plt, args.output_dir, geometry, "docking", "sx", 40, (70, 106))
    quad = draw_quad(plt, args.root, args.output_dir)
    result = {"status": "rendered_from_saved_data", "sources": sources,
              "geometry_rows": len(geometry), "width_rows": len(widths), "quad": quad,
              "initial_layers": {"attitude_x4": [-.75, -.74], "docking_sx": [70, 106]},
              "interpolation": "none; each complete saved tube drawn as a constant step band",
              "geometry_reduction_seconds": reduced - start,
              "python_plot_seconds": time.perf_counter() - reduced,
              "numerical_solver_seconds": 0, "new_benchmark_attempts": 0,
              "no_content_digest_computed": True, "matlab_files_created": 0,
              "independent_nncs_certificate": False}
    (args.output_dir / "geometry.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("status", "geometry_rows", "width_rows", "python_plot_seconds")}))


if __name__ == "__main__":
    main()
