#!/usr/bin/env python3
"""Draw saved 2026 paper-QUAD x3 boxes; never run a solver or a digest."""

import csv
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
EVIDENCE = HERE.parent
NATIVE = EVIDENCE / "native_quad_paper_full50_001" / "pooled_x3_1000steps.csv"
NATIVE_SCAN = EVIDENCE / "native_quad_paper_full50_001" / "SCAN.json"
P3 = EVIDENCE / "quad_paper_p3_nohash_v1/full50_001/data/observations.jsonl"
HUAN = EVIDENCE / "quad_paper_huan_full50_001/parity/metrics.json"
XIANGRU = EVIDENCE / "quad_paper_xiangru_v1/full50_001/data/metrics.json"
PREFIX = HERE / "quad_paper_fourway_t_x3_pooled_tube"
COLORS = {"Native": "#1f77b4", "Huan": "#d95f02",
          "Xiangru": "#7570b3", "ours/P3": "#1b9e77"}
H = 0.005
STEPS = 1000


def source(path):
    return {"path": str(path.resolve()), "bytes": path.stat().st_size}


def interval(pair, context):
    if (len(pair) != 2 or not all(math.isfinite(x) for x in pair)
            or pair[0] > pair[1]):
        raise ValueError(f"invalid interval: {context}: {pair}")
    return [float(pair[0]), float(pair[1])]


def native_rows():
    scan = json.loads(NATIVE_SCAN.read_text(encoding="utf-8"))
    if (scan.get("record_count") != 1024000
            or scan.get("expected_record_count") != 1024000
            or scan.get("complete_unique_ordered_grid") is not True
            or any(scan.get("errors", {}).values())
            or len(scan.get("per_step", [])) != STEPS):
        raise ValueError("native raw-range scan does not establish a complete 1024×1000 grid")
    with NATIVE.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != STEPS:
        raise ValueError(f"native pooled file has {len(rows)} steps, expected 1000")
    tubes, endpoints = [], []
    for step, row in enumerate(rows, 1):
        if int(row["step"]) != step:
            raise ValueError(f"native step {step} missing or out of order")
        if abs(float(row["t_start"]) - (step - 1) * H) > 1e-10:
            raise ValueError(f"native step {step} start time mismatches")
        if abs(float(row["t_end"]) - step * H) > 1e-10:
            raise ValueError(f"native step {step} end time mismatches")
        tubes.append(interval([float(row["tube_x3_lo"]), float(row["tube_x3_hi"])],
                              f"native tube {step}"))
        endpoints.append(interval([float(row["endpoint_x3_lo"]),
                                   float(row["endpoint_x3_hi"])],
                                  f"native endpoint {step}"))
        scan_row = scan["per_step"][step - 1]
        state = scan_row["states"][2]
        if (scan_row["step"] != step or state["state"] != 3
                or tubes[-1] != [state["tube_lo"], state["tube_hi"]]
                or endpoints[-1] != [state["endpoint_lo"], state["endpoint_hi"]]):
            raise ValueError(f"native pooled CSV differs from raw-range scan at step {step}")
    return tubes, endpoints


def p3_rows():
    tubes, endpoints = [], []
    with P3.open(encoding="utf-8") as handle:
        for step, line in enumerate(handle, 1):
            row = json.loads(line)
            if (row.get("substep") != step or row.get("accepted_count") != 1024
                    or row.get("status_counts") != {"0": 1024}
                    or row.get("rejected_lanes") != []):
                raise ValueError(f"P3 step {step} not 1024 accepted lanes")
            union = row.get("tube_endpoint_union_12x4")
            if not isinstance(union, list) or len(union) != 12 or len(union[2]) != 4:
                raise ValueError(f"P3 step {step} lacks saved x3 union")
            tubes.append(interval(union[2][:2], f"P3 tube {step}"))
            endpoints.append(interval(union[2][2:], f"P3 endpoint {step}"))
    if len(tubes) != STEPS:
        raise ValueError(f"P3 observer has {len(tubes)} steps, expected 1000")
    return tubes, endpoints


def endpoint_only(path, label):
    metrics = json.loads(path.read_text(encoding="utf-8"))
    if (metrics.get("B") != 1024 or metrics.get("steps") != 50
            or metrics.get("substeps") != 20 or metrics.get("broken") != 0):
        raise ValueError(f"{label}: terminal metric coverage mismatch")
    return interval(metrics["final_hull"]["x3"], f"{label} final x3")


def build():
    native_tubes, native_endpoints = native_rows()
    p3_tubes, p3_endpoints = p3_rows()
    huan = endpoint_only(HUAN, "Huan")
    xiangru = endpoint_only(XIANGRU, "Xiangru")
    if huan != xiangru:
        raise ValueError("Huan/Xiangru final bounds unexpectedly differ; revise caption")
    return {
        "schema": "archcomp26-quad-paper-fourway-saved-x3-v1",
        "contract": "2026 paper-equation QUAD, full 1024-box initial partition, T=5",
        "time_step_s": H, "steps": STEPS, "initial_x3": [-0.4, 0.4],
        "target_x3_at_t5_only": [0.94, 1.06],
        "series": [
            {"method": "Native", "view": "1024-box pooled whole-step x3 tube",
             "source": source(NATIVE), "raw_range_scan": source(NATIVE_SCAN),
             "tube": native_tubes,
             "endpoint_per_step": native_endpoints,
             "terminal_endpoint": native_endpoints[-1],
             "range_record_status": "ranges.bin stores no accepted flag; adjacent RESULT and log are separate"},
            {"method": "Huan", "view": "terminal x3 endpoint only",
             "source": source(HUAN), "terminal_endpoint": huan},
            {"method": "Xiangru", "view": "terminal x3 endpoint only",
             "source": source(XIANGRU), "terminal_endpoint": xiangru},
            {"method": "ours/P3", "view": "1024-box pooled whole-step x3 tube",
             "source": source(P3), "tube": p3_tubes,
             "endpoint_per_step": p3_endpoints,
             "terminal_endpoint": p3_endpoints[-1],
             "range_record_status": "1024 accepted, status 0, no rejected lanes in every saved observer row"},
        ],
        "qualification": (
            "Native and P3 curves are axis-aligned single-coordinate projections of saved boxes, "
            "not native Flow* octagons. Huan and Xiangru have no saved stepwise x3 "
            "ranges; only T=5 endpoints are drawn. Target [0.94,1.06] "
            "applies at T=5 only. Adjacent outcomes are not content-bound to ranges. "
            "No independent end-to-end floating-point NNCS certificate or speed ranking."),
    }


def write_csv(data):
    with PREFIX.with_suffix(".geometry.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("method", "record_type", "step", "t_start_s", "t_end_s",
                         "tube_lo", "tube_hi", "endpoint_lo", "endpoint_hi"))
        for item in data["series"]:
            if "tube" not in item:
                lo, hi = item["terminal_endpoint"]
                writer.writerow((item["method"], "terminal_endpoint_only", 1000, 5, 5,
                                 "", "", format(lo, ".17g"), format(hi, ".17g")))
                continue
            for step, (tube, endpoint) in enumerate(zip(item["tube"], item["endpoint_per_step"]), 1):
                writer.writerow((item["method"], "whole_step_tube_and_endpoint", step,
                                 format((step - 1) * H, ".17g"),
                                 format(step * H, ".17g"),
                                 *(format(x, ".17g") for x in tube + endpoint)))


def vector(values):
    return "[" + " ".join(format(x, ".17g") for x in values) + "]"


def write_matlab(data):
    lines = [
        "% 2026 paper QUAD. Saved axis-aligned x3 box projections; no solver is run.",
        "% Native and P3 have 1000 whole-step tube unions. Huan/Xiangru: T=5 endpoint only.",
        "% The [0.94,1.06] target applies at T=5, never as an all-time Safe band.",
        "% Native octagon support directions are not recoverable from ranges.bin.",
        "% This MATLAB script is generated but has not been executed in MATLAB/Octave.",
        "figure('Color','w'); subplot(1,4,1:3); hold on; grid on;",
        "t = " + vector([step * H for step in range(STEPS + 1)]) + ";",
    ]
    for index, item in enumerate(data["series"]):
        if "tube" not in item:
            continue
        color = [int(COLORS[item["method"]][k:k+2], 16) / 255 for k in (1, 3, 5)]
        n = index + 1
        style = "--" if item["method"] == "Native" else "-"
        lines.extend([
            f"lo{n} = {vector([pair[0] for pair in item['tube']])};",
            f"hi{n} = {vector([pair[1] for pair in item['tube']])};",
            f"c{n} = {vector(color)};",
            f"for k=1:{STEPS}",
            f"  patch([t(k) t(k+1) t(k+1) t(k)], [lo{n}(k) lo{n}(k) hi{n}(k) hi{n}(k)], c{n}, 'FaceAlpha', 0.045, 'EdgeColor', 'none');",
            "end",
            f"stairs(t, [lo{n} lo{n}(end)], 'Color', c{n}, 'LineStyle', '{style}', 'LineWidth', 1.1, 'DisplayName', '{item['method']} pooled tube');",
            f"stairs(t, [hi{n} hi{n}(end)], 'Color', c{n}, 'LineStyle', '{style}', 'LineWidth', 1.1, 'HandleVisibility', 'off');",
        ])
    lines.extend([
        "plot([0 0], [-0.4 0.4], 'ko-', 'LineWidth', 2, 'DisplayName', 'Initial x3 box');",
        "plot([5 5], [0.94 1.06], 'm-', 'LineWidth', 5, 'DisplayName', 'T=5 target');",
        "xlabel('t (s)'); ylabel('x_3'); xlim([0 5]);",
        "title('2026 paper QUAD: saved whole-step tubes'); legend('Location','best');",
        "subplot(1,4,4); hold on; grid on;",
        "patch([0.94 1.06 1.06 0.94], [0.5 0.5 4.5 4.5], [0.96 0.83 0.94], 'FaceAlpha', 0.25, 'EdgeColor', 'none');",
    ])
    for y, item in enumerate(data["series"], 1):
        lo, hi = item["terminal_endpoint"]
        color = [int(COLORS[item["method"]][k:k+2], 16) / 255 for k in (1, 3, 5)]
        lines.extend([f"plot([{lo:.17g} {hi:.17g}], [{y} {y}], '-', 'Color', {vector(color)}, 'LineWidth', 4);",
                      f"plot([{lo:.17g} {hi:.17g}], [{y} {y}], 'o', 'Color', {vector(color)});"])
    lines.extend([
        "set(gca,'YTick',1:4,'YTickLabel',{'Native','Huan','Xiangru','ours/P3'});",
        "ylim([0.5 4.5]); set(gca,'YDir','reverse'); xlabel('x_3 at T=5');",
        "title('Endpoint unions'); hold off;",
    ])
    PREFIX.with_suffix(".m").write_text("\n".join(lines) + "\n", encoding="utf-8")


def render(data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax, terminal) = plt.subplots(1, 2, figsize=(13.8, 6.8),
                                       gridspec_kw={"width_ratios": [3.7, 1.3]})
    edges = [step * H for step in range(STEPS + 1)]
    for item in data["series"]:
        if "tube" not in item:
            continue
        lo = [pair[0] for pair in item["tube"]]
        hi = [pair[1] for pair in item["tube"]]
        color = COLORS[item["method"]]
        ax.fill_between(edges, lo + lo[-1:], hi + hi[-1:], step="post",
                        color=color, alpha=0.075)
        style = "--" if item["method"] == "Native" else "-"
        ax.step(edges, lo + lo[-1:], where="post", color=color, linewidth=1.2,
                linestyle=style,
                label=item["method"] + " 1024-box tube")
        ax.step(edges, hi + hi[-1:], where="post", color=color, linewidth=1.2,
                linestyle=style)
    ax.plot([0, 0], [-0.4, 0.4], "ko-", linewidth=2, label="Initial x3 box")
    ax.plot([5, 5], [0.94, 1.06], color="#b2188b", linewidth=5,
            solid_capstyle="round", label="T=5 endpoint target")
    ax.set(xlim=(0, 5), xlabel="t (s)", ylabel=r"$x_3$",
           title="Saved 1024-box whole-step $t$–$x_3$ tube projection")
    ax.grid(alpha=0.18)
    ax.legend(loc="upper left", fontsize=8)

    terminals = [item["terminal_endpoint"] for item in data["series"]]
    target = data["target_x3_at_t5_only"]
    left = min(target[0], *(pair[0] for pair in terminals))
    right = max(target[1], *(pair[1] for pair in terminals))
    pad = max(0.025, 0.15 * (right - left))
    terminal.axvspan(*target, color="#b2188b", alpha=0.12,
                     label="T=5 target")
    for y, item in enumerate(data["series"], 1):
        lo, hi = item["terminal_endpoint"]
        color = COLORS[item["method"]]
        terminal.plot([lo, hi], [y, y], color=color, linewidth=4,
                      solid_capstyle="round")
        terminal.plot([lo, hi], [y, y], "o", color=color, markersize=5)
    terminal.set(xlim=(left - pad, right + pad), ylim=(4.5, 0.5),
                 xlabel=r"$x_3$ at $T=5$", yticks=range(1, 5),
                 yticklabels=[item["method"] for item in data["series"]],
                 title="Four terminal unions")
    terminal.grid(axis="x", alpha=0.18)
    terminal.legend(loc="lower right", fontsize=8)
    fig.suptitle("ARCH-COMP 2026 paper-equation QUAD — saved intervals, full initial partition")
    fig.subplots_adjust(left=0.07, right=0.98, top=0.87, bottom=0.23, wspace=0.35)
    fig.text(0.07, 0.065,
             "Native and P3: per-step unions of all 1,024 boxes. Native ranges contain x3 box bounds, not Flow* octagon geometry.\n"
             "Huan/Xiangru: only T=5 endpoint hulls were saved, so no intermediate tube is drawn. Their endpoint intervals coincide.\n"
             "[0.94,1.06] is a T=5 target only. Adjacent outcomes are separate from ranges; no end-to-end NNCS proof or speed ranking.",
             fontsize=8.0, va="bottom")
    fig.savefig(PREFIX.with_suffix(".png"), dpi=180)
    fig.savefig(PREFIX.with_suffix(".pdf"))
    plt.close(fig)


def main():
    data = build()
    PREFIX.with_suffix(".geometry.json").write_text(
        json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8")
    write_csv(data)
    write_matlab(data)
    render(data)
    print(json.dumps({item["method"]: item["terminal_endpoint"] for item in data["series"]},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
