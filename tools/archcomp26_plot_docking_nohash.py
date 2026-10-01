#!/usr/bin/env python3
"""Plot saved Docking q upper bounds; this does not run a solver or verifier."""

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path


RUNS = (
    ("Huan", "docking_huan_full40_001", "#1f77b4", "-"),
    ("Xiangru", "docking_xiangru_full40_001", "#d95f02", "--"),
    ("P3", "docking_p3_full40_001", "#1b9e77", "-"),
)


def read_series(root):
    series = []
    for label, name, color, style in RUNS:
        directory = root / name
        outer = json.loads((directory / "RESULT.json").read_text())
        inner = json.loads((directory / "detail/RESULT.json").read_text())
        if (outer.get("status") != "completed" or inner.get("status") != "completed"
                or inner.get("accepted_substeps") != 400 or inner.get("safety_events") != 400
                or inner.get("checker_verdict") != "UNKNOWN_REPORTED_BY_AUTHOR_CHECKER"):
            raise ValueError(f"{name}: incomplete or unexpected Docking outcome")
        path = directory / "detail/safety.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if len(rows) != 400:
            raise ValueError(f"{name}: expected 400 saved substeps")
        bounds = []
        for step, row in enumerate(rows, 1):
            pair = row.get("q_interval")
            if (row.get("substep") != step or not isinstance(pair, list) or len(pair) != 2
                    or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in pair)
                    or pair[0] > pair[1]):
                raise ValueError(f"{name}: invalid saved q interval at step {step}")
            bounds.append(pair)
        series.append({"label": label, "run": name, "color": color, "style": style,
                       "source": str(path.relative_to(root)), "source_bytes": path.stat().st_size,
                       "outer_wall_s": outer["wall_s"], "accepted_substeps": 400,
                       "checker_verdict": inner["checker_verdict"], "q_intervals": bounds})
    return series


def read_native(root):
    directory = root / "native_docking_full40_001"
    outer = json.loads((directory / "RESULT.json").read_text())
    log = (directory / "native.log").read_text()
    if (outer.get("status") != "failed" or outer.get("exit_code") != 2
            or "COMPLETED_PERIODS 40/40\n" not in log
            or "FLOWPIPE_SEGMENTS 400\nUNKNOWN\n" not in log):
        raise ValueError("native Docking outcome or completion changed")
    source = directory / "safety.tsv"
    with source.open(newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if len(rows) != 400:
        raise ValueError("native Docking: expected 400 saved substeps")
    bounds = []
    for step, row in enumerate(rows, 1):
        if (int(row["period"]) != (step - 1) // 10
                or int(row["local_substep"]) != (step - 1) % 10 + 1
                or int(row["global_substep"]) != step):
            raise ValueError(f"native Docking: invalid time index at step {step}")
        pair = [float(row["q_lower"]), float(row["q_upper"])]
        if not all(math.isfinite(x) for x in pair) or pair[0] > pair[1]:
            raise ValueError(f"native Docking: invalid q interval at step {step}")
        bounds.append(pair)
    return {"label": "Flow* native", "run": directory.name, "color": "#9467bd", "style": "-.",
            "source": str(source.relative_to(root)), "source_bytes": source.stat().st_size,
            "outer_wall_s": outer["wall_s"], "accepted_substeps": 400,
            "checker_verdict": "UNKNOWN_REPORTED_BY_NATIVE_CHECKER", "q_intervals": bounds}


def matlab_vector(values):
    return "[" + " ".join(format(value, ".17g") for value in values) + "]"


def write_matlab(series, path):
    lines = ["% Saved Docking q upper bounds. Safe requires q <= 0 at every time.",
             "% q upper > 0 means this box check is Unknown, not a counterexample.",
             "% Inputs are the adjacent geometry JSON; no reachability solver runs here.",
             "t = 0:0.1:40; figure('Color','w'); hold on; grid on;",
             "patch([0 40 40 0],[-0.25 -0.25 0 0],[0.75 0.9 0.75],'FaceAlpha',0.35,'EdgeColor','none','DisplayName','Safe region (q <= 0)');"]
    for index, item in enumerate(series, 1):
        upper = [pair[1] for pair in item["q_intervals"]]
        lines += [f"q{index} = {matlab_vector(upper)};",
                  f"stairs(t, [q{index} q{index}(end)], '{item['style']}', 'LineWidth', 1.2, 'DisplayName', '{item['label']}');"]
    lines += ["plot(0,-0.007355828533536612,'ko','DisplayName','Initial box');",
              "plot([0 40],[0 0],'k:','LineWidth',1.3,'DisplayName','q = 0');",
              "xlabel('t (s)'); ylabel('Upper bound on q');",
              "title('Docking full initial box: saved q upper bounds');",
              "legend('Location','northwest'); xlim([0 40]); hold off;"]
    path.write_text("\n".join(lines) + "\n")


def render(series, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    edges = [i / 10 for i in range(401)]
    fig, ax = plt.subplots(figsize=(11.4, 6.4))
    inset = ax.inset_axes([0.13, 0.53, 0.39, 0.34])
    ax.axhspan(-0.25, 0, color="#a8dca8", alpha=0.42, label=r"Safe region ($q\leq0$)")
    inset.axhspan(-0.035, 0, color="#a8dca8", alpha=0.42)
    for item in series:
        upper = [pair[1] for pair in item["q_intervals"]]
        plotted = upper + [upper[-1]]
        for axes, line_width in ((ax, 1.4), (inset, 1.1)):
            axes.step(edges, plotted, where="post", color=item["color"],
                      linestyle=item["style"], linewidth=line_width,
                      label=item["label"] if axes is ax else None)
    for axes in (ax, inset):
        axes.axhline(0, color="black", linestyle=":", linewidth=1.1)
        axes.plot([0], [-0.007355828533536612], "ko", markersize=4,
                  label="Initial box" if axes is ax else None)
        axes.grid(alpha=0.2)
    ax.set_xlim(0, 40)
    ax.set_ylim(-0.25, max(pair[1] for item in series for pair in item["q_intervals"]) * 1.04)
    ax.set_xlabel("t (s)")
    ax.set_ylabel(r"Upper bound on $q=\|v\|-0.2-0.002054\|s\|$")
    ax.set_title("Docking, full initial box: saved all-time constraint bounds")
    ax.legend(loc="upper left", bbox_to_anchor=(0.54, 0.98), fontsize=9)
    inset.set_xlim(0, 1)
    inset.set_ylim(-0.035, 0.19)
    inset.set_title("First second", fontsize=9)
    inset.tick_params(labelsize=8)
    methods = len(series)
    footer = ("All 400 substeps completed per saved run; q upper > 0 means Unknown, not a trajectory counterexample.\n"
              "Huan/Xiangru upper bounds coincide; one full process per method.\n"
              + ("Source: each run's detail/safety.jsonl and native safety.tsv; " if methods == 4 else
                 "Source: each run's detail/safety.jsonl; ")
              + "plotted tube boxes lose coordinate correlation. No end-to-end NN certificate.")
    fig.subplots_adjust(left=0.10, right=0.97, top=0.91, bottom=0.21)
    fig.text(0.10, 0.055, footer, fontsize=8, va="bottom")
    fig.savefig(path.with_suffix(".png"), dpi=180)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--include-native", action="store_true")
    args = parser.parse_args()
    series = read_series(args.results_root)
    if series[0]["q_intervals"] != series[1]["q_intervals"]:
        raise ValueError("Huan/Xiangru coincidence claim changed")
    if args.include_native:
        series.append(read_native(args.results_root))
    output = args.output_prefix
    output.parent.mkdir(parents=True, exist_ok=True)
    geometry = {"schema": "archcomp26-docking-saved-q-geometry-v2",
                "contract": "2026 Docking, single complete initial box, 40 periods x 1s, 400 x 0.1s tubes",
                "property": "q=||v||-0.2-0.002054||s|| <= 0 for all t in [0,40]",
                "step_s": 0.1, "initial_q_interval": [-0.5079082336541202, -0.007355828533536612],
                "native_initial_q_interval": [-0.50790823365411986, -0.0073558285335368345] if args.include_native else None,
                "series": series,
                "qualification": "saved conservative axis-aligned tube-box intervals; positive q upper is Unknown, not falsification"}
    output.with_suffix(".geometry.json").write_text(json.dumps(geometry, indent=2) + "\n")
    write_matlab(series, output.with_suffix(".m"))
    render(series, output)
    output.with_suffix(".render.json").write_text(json.dumps({
        "schema": "archcomp26-docking-saved-q-render-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "geometry": output.with_suffix(".geometry.json").name,
        "outputs": [output.with_suffix(s).name for s in (".png", ".pdf", ".m")],
        "matlab_execution": "not run; script generated from saved intervals",
        "input_binding": "paths, sizes, schemas, row counts, acceptance and adjacent outcomes; no content digest",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
