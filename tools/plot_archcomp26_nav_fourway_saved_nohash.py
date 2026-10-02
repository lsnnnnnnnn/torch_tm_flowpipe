#!/usr/bin/env python3
"""Plot historical-versus-new NAV saved x/y tube projections and widths."""

import argparse
import csv
import json
from pathlib import Path


SUITES = (("nav-standard", "nav_standard_fourway_saved_20261002", "Standard"),
          ("nav-robust", "nav_robust_fourway_saved_20261002", "Robust"))
METHODS = ("ours", "huan", "xiangru", "flowstar_native")
LABELS = {"ours": "old GPU ours", "huan": "old Huan",
          "xiangru": "old Xiangru", "flowstar_native": "Flow* native"}
COLORS = {"ours": "#0072B2", "huan": "#E69F00",
          "xiangru": "#009E73", "flowstar_native": "#CC79A7"}
STYLES = {"ours": "-", "huan": "--", "xiangru": ":", "flowstar_native": "-."}
CAPTION = ("Standard native is a new 2026-10-02 run; other curves are historical. "
           "Old GPU ours is not the current working P3. Saved tube projections only; "
           "the joint obstacle [1,2]^2 was checked separately per saved box. "
           "No end-to-end NNCS certificate or speed ranking.")


def load(root):
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
    return curves


def draw(root, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curves = load(root)
    output.mkdir(parents=True, exist_ok=False)
    for view, filename, title in (("tubes", "nav_saved_xy_tubes", "NAV saved flowpipe x/y projections"),
                                  ("widths", "nav_saved_xy_union_width", "NAV saved x/y tube union widths")):
        fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), sharex=True,
                                 constrained_layout=False)
        for r, (instance, _, suite_title) in enumerate(SUITES):
            for c, state in enumerate(("x", "y")):
                ax = axes[r, c]
                for method in METHODS:
                    rows = curves[(instance, method, state)]
                    times = [row["t_end"] for row in rows]
                    label = LABELS[method]
                    color = COLORS[method]
                    if view == "tubes":
                        lower = [row["tube_lo"] for row in rows]
                        upper = [row["tube_hi"] for row in rows]
                        ax.fill_between(times, lower, upper, color=color, alpha=0.085)
                        ax.plot(times, lower, color=color, linestyle=STYLES[method],
                                linewidth=0.9, label=label)
                        ax.plot(times, upper, color=color, linestyle=STYLES[method], linewidth=0.9)
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
        fig.text(0.5, 0.015, CAPTION, ha="center", va="bottom", fontsize=7,
                 wrap=True)
        fig.tight_layout(rect=(0.02, 0.055, 0.98, 0.95))
        fig.savefig(output / f"{filename}.png", dpi=200)
        fig.savefig(output / f"{filename}.pdf")
        plt.close(fig)
    print(json.dumps({"status": "plotted", "output": str(output)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    draw(args.root, args.output_dir)


if __name__ == "__main__":
    main()
