#!/usr/bin/env python3
"""Plot historical-versus-new NAV saved x/y tube projections and widths."""

import argparse
import csv
import json
import math
import time
from pathlib import Path


SUITES = (("nav-standard", "nav_standard_fourway_saved_20261002", "Standard"),
          ("nav-robust", "nav_robust_fourway_saved_20261002", "Robust"))
METHODS = ("ours", "huan", "xiangru", "flowstar_native")
P3_METHOD = "working_p3"
LABELS = {"ours": "old GPU ours", "huan": "old Huan",
          "xiangru": "old Xiangru", "flowstar_native": "Flow* native",
          P3_METHOD: "new working P3"}
COLORS = {"ours": "#0072B2", "huan": "#E69F00",
          "xiangru": "#009E73", "flowstar_native": "#CC79A7",
          P3_METHOD: "#6A3D9A"}
STYLES = {"ours": "-", "huan": "--", "xiangru": ":", "flowstar_native": "-.",
          P3_METHOD: (0, (5, 2))}
CAPTION = ("Standard native is a new 2026-10-02 run; other curves are historical. "
           "Old GPU ours is not the current working P3. Saved tube projections only; "
           "the joint obstacle [1,2]^2 was checked separately per saved box. "
           "No end-to-end NNCS certificate or speed ranking.")


def load(root, working_p3_csv=None, robust_working_p3_csv=None):
    curves = {}
    for instance, folder, title in SUITES:
        source = root / folder
        audit = json.loads((source / "SOURCE_AUDIT.json").read_text())
        if audit["instance"] != instance or tuple(row["method"] for row in audit["methods"]) != METHODS:
            raise ValueError(f"NAV four-method source audit differs: {source}")
        with (source / "xy_saved_curves.csv").open(newline="") as stream:
            for row in csv.DictReader(stream):
                if row["instance"] != instance or row["method"] not in METHODS or row["state"] not in ("x", "y"):
                    raise ValueError(f"Unexpected historical source row: {source}")
                key = (instance, row["method"], row["state"])
                curves.setdefault(key, []).append({field: float(row[field]) for field in
                    ("t_start", "t_end", "tube_lo", "tube_hi", "tube_union_width",
                     "endpoint_lo", "endpoint_hi", "endpoint_union_width",
                     "partition_mean_tube_width", "partition_max_tube_width",
                     "partition_mean_endpoint_width", "partition_max_endpoint_width")})
        for method in METHODS:
            for state in ("x", "y"):
                rows = curves.get((instance, method, state), [])
                if len(rows) != 600 or any(abs(row["t_end"] - i * 0.01) > 1e-12
                                            for i, row in enumerate(rows, 1)):
                    raise ValueError(f"NAV saved curve incomplete: {instance}, {method}, {state}")
    for instance, source_csv in (("nav-standard", working_p3_csv),
                                 ("nav-robust", robust_working_p3_csv)):
        if source_csv is None:
            continue
        with source_csv.open(newline="") as stream:
            for row in csv.DictReader(stream):
                if (row["instance"], row["method"], row["source_generation"]) != (
                    instance, P3_METHOD, "2026-10-02 new working P3"):
                    raise ValueError(f"Unexpected current P3 source row: {source_csv}")
                if row["state"] not in ("x", "y"):
                    raise ValueError(f"Unexpected current P3 state: {row['state']}")
                curves.setdefault((instance, P3_METHOD, row["state"]), []).append(
                    {field: float(row[field]) for field in (
                        "t_start", "t_end", "tube_lo", "tube_hi", "tube_union_width",
                        "endpoint_lo", "endpoint_hi", "endpoint_union_width",
                        "partition_mean_tube_width", "partition_max_tube_width",
                        "partition_mean_endpoint_width", "partition_max_endpoint_width")})
    for key, rows in curves.items():
        if len(rows) != 600:
            raise ValueError(f"Incomplete saved grid: {key}")
        for i, row in enumerate(rows):
            if (not all(map(math.isfinite, row.values()))
                    or abs(row["t_start"] - i * .01) > 1e-12
                    or abs(row["t_end"] - (i + 1) * .01) > 1e-12
                    or not row["tube_lo"] <= row["endpoint_lo"] <= row["endpoint_hi"] <= row["tube_hi"]
                    or abs(row["tube_union_width"] - (row["tube_hi"] - row["tube_lo"])) > 1e-12):
                raise ValueError(f"Invalid saved interval/time: {key}, step {i + 1}")
    return curves


def draw(root, output, working_p3_csv=None, robust_working_p3_csv=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curves = load(root, working_p3_csv, robust_working_p3_csv)
    output.mkdir(parents=True, exist_ok=True)
    for view, filename, title in (("tubes", "nav_saved_xy_tubes", "NAV saved flowpipe x/y projections"),
                                  ("widths", "nav_saved_xy_union_width", "NAV saved x/y tube union widths")):
        fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), sharex=True, sharey=True,
                                 constrained_layout=False)
        for r, (instance, _, suite_title) in enumerate(SUITES):
            for c, state in enumerate(("x", "y")):
                ax = axes[r, c]
                methods = ((P3_METHOD, "huan", "xiangru", "flowstar_native")
                           if (instance, P3_METHOD, state) in curves else METHODS)
                for method in methods:
                    rows = curves[(instance, method, state)]
                    times = [rows[0]["t_start"]] + [row["t_end"] for row in rows]
                    label = LABELS[method]
                    color = COLORS[method]
                    if view == "tubes":
                        lower = [row["tube_lo"] for row in rows] + [rows[-1]["tube_lo"]]
                        upper = [row["tube_hi"] for row in rows] + [rows[-1]["tube_hi"]]
                        ax.fill_between(times, lower, upper, step="post", color=color,
                                        alpha=0.075 if working_p3_csv is not None else 0.085)
                        ax.step(times, lower, where="post", color=color, linestyle=STYLES[method],
                                linewidth=1.25 if method == P3_METHOD else 0.9, label=label)
                        ax.step(times, upper, where="post", color=color, linestyle=STYLES[method],
                                linewidth=1.25 if method == P3_METHOD else 0.9)
                    else:
                        ax.step(times, [row["tube_union_width"] for row in rows] + [rows[-1]["tube_union_width"]], where="post",
                                color=color, linestyle=STYLES[method], linewidth=1.5, label=label)
                if view == "tubes":
                    ax.plot([0, 0], [2.9, 3.1], color="black", linewidth=4, label="Initial set (t=0)")
                    ax.plot([6, 6], [-.5, .5], color="#b2188b", linewidth=4, label="Target (T=6 only)")
                    ax.axhspan(1, 2, color="0.5", alpha=.07, hatch="//", label="Obstacle projection only")
                ax.set_title(f"{suite_title}: {state}")
                ax.set_xlim(-.03, 6.03)
                ax.set_xlabel("Time (s)")
                ax.set_ylabel(f"{state} saved tube" if view == "tubes" else f"{state} union width")
                ax.grid(alpha=0.25, linewidth=0.5)
                if r == 0 and c == 0:
                    ax.legend(loc="best", fontsize=8, framealpha=0.9)
        fig.suptitle(title, fontsize=14, y=0.98)
        caption = ("Current P3 is shown where supplied; Huan/Xiangru are historical. Standard native is new; robust native is historical.\n"
                   "Old GPU ours appears only when no current P3 is supplied. Each tube spans its saved step; initial is t=0 and target only T=6.\n"
                   "Obstacle shading is a one-coordinate projection of [1,2]^2; it does not decide joint avoidance. No proof or speed ranking.")
        fig.text(0.5, 0.015, caption, ha="center", va="bottom", fontsize=7,
                 wrap=True)
        fig.tight_layout(rect=(0.02, 0.055, 0.98, 0.95))
        if working_p3_csv is not None:
            filename += "_with_working_p3"
        fig.savefig(output / f"{filename}.png", dpi=200)
        fig.savefig(output / f"{filename}.pdf")
        plt.close(fig)
    print(json.dumps({"status": "plotted", "output": str(output)}))


def draw_details(root, output, working_p3_csv=None, robust_working_p3_csv=None):
    """Show coincident methods separately and magnify width differences."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curves = load(root, working_p3_csv, robust_working_p3_csv)
    output.mkdir(parents=True, exist_ok=True)
    for instance, _, title in SUITES:
        fig, axes = plt.subplots(2, 4, figsize=(17, 7.5), sharex=True,
                                 sharey="row")
        for c, method in enumerate(METHODS):
            for r, state in enumerate(("x", "y")):
                ax = axes[r, c]
                rows = curves[(instance, method, state)]
                times = [rows[0]["t_start"]] + [row["t_end"] for row in rows]
                lower = [row["tube_lo"] for row in rows] + [rows[-1]["tube_lo"]]
                upper = [row["tube_hi"] for row in rows] + [rows[-1]["tube_hi"]]
                ax.fill_between(times, lower, upper, step="post", color=COLORS[method], alpha=0.23)
                ax.step(times, lower, where="post", color=COLORS[method], linewidth=1.25)
                ax.step(times, upper, where="post", color=COLORS[method], linewidth=1.25)
                ax.set_xlim(-.03, 6.03)
                ax.set_title(LABELS[method] + (" (new)" if instance == "nav-standard" and method == "flowstar_native" else " (historical)"),
                             fontsize=10)
                ax.set_xlabel("Time (s)")
                if c == 0:
                    ax.set_ylabel(f"{state} saved tube")
                ax.grid(alpha=0.25, linewidth=0.5)
        fig.suptitle(f"NAV {title}: one saved x/y tube projection per method", fontsize=15)
        fig.text(0.5, 0.015, "All four methods have 600 saved steps. Huan and Xiangru bounds coincide exactly; old GPU ours is not working P3.\n"
                 "Each row shares one y-axis. x/y projections and shading are numerical saved intervals, not an independent NNCS proof.",
                 ha="center", va="bottom", fontsize=8)
        fig.tight_layout(rect=(0.015, 0.06, 0.99, 0.95))
        name = f"nav_{title.lower()}_each_method_xy_tubes"
        fig.savefig(output / f"{name}.png", dpi=180)
        fig.savefig(output / f"{name}.pdf")
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
    for r, (instance, _, title) in enumerate(SUITES):
        for c, state in enumerate(("x", "y")):
            ax = axes[r, c]
            native = curves[(instance, "flowstar_native", state)]
            huan = curves[(instance, "huan", state)]
            xiangru = curves[(instance, "xiangru", state)]
            if any(a["tube_lo"] != b["tube_lo"] or a["tube_hi"] != b["tube_hi"]
                   for a, b in zip(huan, xiangru)):
                raise ValueError(f"Huan/Xiangru saved bounds differ: {instance}, {state}")
            methods = [("ours", "old GPU ours"),
                       ("huan", "old Huan = old Xiangru")]
            if (instance, P3_METHOD, state) in curves:
                methods.append((P3_METHOD, "new working P3"))
            for method, label in methods:
                rows = curves[(instance, method, state)]
                ax.step([rows[0]["t_start"]] + [row["t_end"] for row in rows],
                        [row["tube_union_width"] - ref["tube_union_width"]
                         for row, ref in zip(rows, native)] +
                        [rows[-1]["tube_union_width"] - native[-1]["tube_union_width"]], where="post",
                        color=COLORS[method], linewidth=1.5, label=label)
            ax.axhline(0, color="0.3", linewidth=0.8, linestyle=":", label="native reference")
            ax.set_title(f"{title}: {state} union-width difference")
            ax.set_xlim(-.03, 6.03)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Method minus native width")
            ax.grid(alpha=0.25, linewidth=0.5)
            if r == 0 and c == 0:
                ax.legend(fontsize=8)
    fig.suptitle("NAV saved tube widths: differences magnified", fontsize=15)
    caption = (("Working P3 for each supplied variant and standard native are new; robust native is historical. Old GPU ours is a different engine.\n"
                if working_p3_csv is not None else
                "Standard native is new; robust native and all GPU curves are historical.\n") +
               "Differences are saved numerical tube projections; native zero is a reference. No proof or speed ranking follows.")
    fig.text(0.5, 0.015, caption,
             ha="center", va="bottom", fontsize=8)
    fig.tight_layout(rect=(0.015, 0.065, 0.99, 0.95))
    name = "nav_each_method_width_difference_vs_native"
    if working_p3_csv is not None:
        name += "_with_working_p3"
    fig.savefig(output / f"{name}.png", dpi=180)
    fig.savefig(output / f"{name}.pdf")
    plt.close(fig)
    print(json.dumps({"status": "plotted_details", "output": str(output)}))


def export_geometry(args, curves, validation_s):
    sources = [args.root / folder / "xy_saved_curves.csv" for _, folder, _ in SUITES]
    sources += [p for p in (args.working_p3_csv, args.robust_working_p3_csv) if p is not None]
    series = [{"instance": instance, "method": method, "state": state, "rows": rows}
              for (instance, method, state), rows in curves.items()]
    geometry = {
        "schema": "nav-saved-projection-20261004-v1",
        "sources": [str(p) for p in sources], "series": series,
        "units": {"t": "s", "x_y": "not declared in saved plot contract; no conversion"},
        "initial_at_t0": {"x": [2.9, 3.1], "y": [2.9, 3.1]},
        "target_at_T6_only": {"x": [-.5, .5], "y": [-.5, .5]},
        "all_time_joint_unsafe_box": {"x": [1, 2], "y": [1, 2]},
        "projection_note": "Shaded coordinate projections cannot decide joint obstacle intersection.",
        "qualification": "Saved numerical bounds; mixed execution generations; no independent NNCS certificate or speed ranking.",
    }
    (args.output_dir / "geometry.json").write_text(json.dumps(geometry, indent=2, allow_nan=False) + "\n")
    fields = ["instance", "method", "state", "step"] + list(next(iter(curves.values()))[0])
    with (args.output_dir / "saved_bounds.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for (instance, method, state), rows in curves.items():
            for step, row in enumerate(rows, 1):
                writer.writerow(dict(instance=instance, method=method, state=state, step=step, **row))
    receipt = {
        "status": "saved_data_validation_passed", "solver_executed": False,
        "source_files": [{"path": str(p), "bytes": p.stat().st_size} for p in sources],
        "source_rows": sum(len(rows) for rows in curves.values()),
        "series_count": len(curves), "steps_per_series": 600,
        "validated": ["finite ordered tube and endpoint", "endpoint within same-step tube",
                      "complete 0 to 6 s grid", "tube widths match bounds", "no missing-step interpolation"],
        "layers": {"initial": "t=0 only", "target": "T=6 only",
                   "unsafe": "all-time coordinate projection of joint [1,2]^2"},
        "validation_s": validation_s,
    }
    (args.output_dir / "AUDIT.json").write_text(json.dumps(receipt, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--details-only", action="store_true",
                        help="Add separate method and magnified difference figures without replacing the original plots")
    parser.add_argument("--working-p3-csv", type=Path,
                        help="Add independently scanned current P3 standard CSV")
    parser.add_argument("--robust-working-p3-csv", type=Path,
                        help="Add existing current P3 robust CSV; no solver is invoked")
    args = parser.parse_args()
    started = time.perf_counter()
    curves = load(args.root, args.working_p3_csv, args.robust_working_p3_csv)
    validation_s = time.perf_counter() - started
    if args.output_dir.exists():
        raise ValueError("Choose a new output directory to preserve previous plot artifacts")
    args.output_dir.mkdir(parents=True)
    export_geometry(args, curves, validation_s)
    rendering_started = time.perf_counter()
    if args.details_only:
        draw_details(args.root, args.output_dir, args.working_p3_csv, args.robust_working_p3_csv)
    else:
        draw(args.root, args.output_dir, args.working_p3_csv, args.robust_working_p3_csv)
        if args.working_p3_csv is not None:
            draw_details(args.root, args.output_dir, args.working_p3_csv, args.robust_working_p3_csv)
    receipt_path = args.output_dir / "AUDIT.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["rendering_s"] = time.perf_counter() - rendering_started
    receipt["outputs"] = [p.name for p in sorted(args.output_dir.iterdir())]
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
