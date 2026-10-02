#!/usr/bin/env python3
"""Plot historical-versus-new NAV saved x/y tube projections and widths."""

import argparse
import csv
import json
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


def load(root, working_p3_csv=None):
    curves = {}
    for instance, folder, title in SUITES:
        source = root / folder
        audit = json.loads((source / "SOURCE_AUDIT.json").read_text())
        if audit["instance"] != instance or tuple(row["method"] for row in audit["methods"]) != METHODS:
            raise ValueError(f"NAV four-method source audit differs: {source}")
        with (source / "xy_saved_curves.csv").open(newline="") as stream:
            for row in csv.DictReader(stream):
                key = (instance, row["method"], row["state"])
                curves.setdefault(key, []).append({field: float(row[field]) for field in
                    ("t_end", "tube_lo", "tube_hi", "tube_union_width")})
        for method in METHODS:
            for state in ("x", "y"):
                rows = curves.get((instance, method, state), [])
                if len(rows) != 600 or any(abs(row["t_end"] - i * 0.01) > 1e-12
                                            for i, row in enumerate(rows, 1)):
                    raise ValueError(f"NAV saved curve incomplete: {instance}, {method}, {state}")
    if working_p3_csv is not None:
        with working_p3_csv.open(newline="") as stream:
            for row in csv.DictReader(stream):
                if (row["instance"], row["method"], row["source_generation"]) != (
                    "nav-standard", P3_METHOD, "2026-10-02 new working P3"):
                    raise ValueError(f"Unexpected current P3 source row: {working_p3_csv}")
                key = ("nav-standard", P3_METHOD, row["state"])
                if row["state"] not in ("x", "y"):
                    raise ValueError(f"Unexpected current P3 state: {row['state']}")
                curves.setdefault(key, []).append({field: float(row[field]) for field in
                    ("t_end", "tube_lo", "tube_hi", "tube_union_width")})
        for state in ("x", "y"):
            rows = curves.get(("nav-standard", P3_METHOD, state), [])
            if len(rows) != 600 or any(abs(row["t_end"] - i * 0.01) > 1e-12
                                        for i, row in enumerate(rows, 1)):
                raise ValueError(f"Current P3 saved curve incomplete: {state}")
    return curves


def draw(root, output, working_p3_csv=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curves = load(root, working_p3_csv)
    output.mkdir(parents=True, exist_ok=working_p3_csv is not None)
    for view, filename, title in (("tubes", "nav_saved_xy_tubes", "NAV saved flowpipe x/y projections"),
                                  ("widths", "nav_saved_xy_union_width", "NAV saved x/y tube union widths")):
        fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), sharex=True,
                                 constrained_layout=False)
        for r, (instance, _, suite_title) in enumerate(SUITES):
            for c, state in enumerate(("x", "y")):
                ax = axes[r, c]
                methods = METHODS + ((P3_METHOD,) if working_p3_csv is not None and instance == "nav-standard" else ())
                for method in methods:
                    rows = curves[(instance, method, state)]
                    times = [row["t_end"] for row in rows]
                    label = LABELS[method]
                    color = COLORS[method]
                    if view == "tubes":
                        lower = [row["tube_lo"] for row in rows]
                        upper = [row["tube_hi"] for row in rows]
                        ax.fill_between(times, lower, upper, color=color,
                                        alpha=0.075 if working_p3_csv is not None else 0.085)
                        ax.plot(times, lower, color=color, linestyle=STYLES[method],
                                linewidth=1.25 if method == P3_METHOD else 0.9, label=label)
                        ax.plot(times, upper, color=color, linestyle=STYLES[method],
                                linewidth=1.25 if method == P3_METHOD else 0.9)
                    else:
                        ax.plot(times, [row["tube_union_width"] for row in rows],
                                color=color, linestyle=STYLES[method], linewidth=1.5, label=label)
                ax.set_title(f"{suite_title}: {state}")
                ax.set_xlim(0, 6)
                ax.set_xlabel("Time (s)")
                ax.set_ylabel(f"{state} saved tube" if view == "tubes" else f"{state} union width")
                ax.grid(alpha=0.25, linewidth=0.5)
                if r == 0 and c == 0:
                    ax.legend(loc="best", fontsize=8, framealpha=0.9)
        fig.suptitle(title, fontsize=14, y=0.98)
        caption = ("Standard: historical ours/Huan/Xiangru plus new native and new working P3; "
                   "robust: four historical methods. Coincident curves are saved results. "
                   "Old ours is not working P3. Saved x/y projections only; no proof or speed ranking."
                   if working_p3_csv is not None else CAPTION)
        fig.text(0.5, 0.015, caption, ha="center", va="bottom", fontsize=7,
                 wrap=True)
        fig.tight_layout(rect=(0.02, 0.055, 0.98, 0.95))
        if working_p3_csv is not None:
            filename += "_with_working_p3"
        fig.savefig(output / f"{filename}.png", dpi=200)
        fig.savefig(output / f"{filename}.pdf")
        plt.close(fig)
    print(json.dumps({"status": "plotted", "output": str(output)}))


def draw_details(root, output, working_p3_csv=None):
    """Show coincident methods separately and magnify width differences."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curves = load(root, working_p3_csv)
    output.mkdir(parents=True, exist_ok=True)
    for instance, _, title in SUITES:
        fig, axes = plt.subplots(2, 4, figsize=(17, 7.5), sharex=True,
                                 sharey="row")
        for c, method in enumerate(METHODS):
            for r, state in enumerate(("x", "y")):
                ax = axes[r, c]
                rows = curves[(instance, method, state)]
                times = [row["t_end"] for row in rows]
                lower = [row["tube_lo"] for row in rows]
                upper = [row["tube_hi"] for row in rows]
                ax.fill_between(times, lower, upper, color=COLORS[method], alpha=0.23)
                ax.plot(times, lower, color=COLORS[method], linewidth=1.25)
                ax.plot(times, upper, color=COLORS[method], linewidth=1.25)
                ax.set_xlim(0, 6)
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
            if working_p3_csv is not None and instance == "nav-standard":
                methods.append((P3_METHOD, "new working P3"))
            for method, label in methods:
                rows = curves[(instance, method, state)]
                ax.plot([row["t_end"] for row in rows],
                        [row["tube_union_width"] - ref["tube_union_width"]
                         for row, ref in zip(rows, native)],
                        color=COLORS[method], linewidth=1.5, label=label)
            ax.axhline(0, color="0.3", linewidth=0.8, linestyle=":", label="native reference")
            ax.set_title(f"{title}: {state} union-width difference")
            ax.set_xlim(0, 6)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Method minus native width")
            ax.grid(alpha=0.25, linewidth=0.5)
            if r == 0 and c == 0:
                ax.legend(fontsize=8)
    fig.suptitle("NAV saved tube widths: differences magnified", fontsize=15)
    caption = (("Standard native and working P3 are new; robust native is historical. Old GPU ours is a different engine.\n"
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--details-only", action="store_true",
                        help="Add separate method and magnified difference figures without replacing the original plots")
    parser.add_argument("--working-p3-csv", type=Path,
                        help="Add independently scanned new working P3 to standard only; preserve original four-method plots")
    args = parser.parse_args()
    if args.details_only:
        draw_details(args.root, args.output_dir, args.working_p3_csv)
    else:
        draw(args.root, args.output_dir, args.working_p3_csv)
        if args.working_p3_csv is not None:
            draw_details(args.root, args.output_dir, args.working_p3_csv)


if __name__ == "__main__":
    main()
