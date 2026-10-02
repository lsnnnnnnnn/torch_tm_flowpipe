#!/usr/bin/env python3
"""Compare four existing h=0.05 TORA remain saved ranges; no solver or digest."""

import csv
import json
import math
from pathlib import Path
import struct


HERE = Path(__file__).resolve().parent
RUNS = HERE.parent
ROOT = HERE.parents[4]
ROW = struct.Struct("<QQd16d")
METHODS = (
    ("ours/P3", "p3_tora_remain_h005_probe_20261002/full20_001", "#1b9e77", "-"),
    ("Huan", "author_tora_remain_h005_probe_20261002/huan_full20_firstreject_001", "#d95f02", "-"),
    ("Xiangru", "author_tora_remain_h005_probe_20261002/xiangru_full20_firstreject_001", "#7570b3", ":"),
    ("Flow* native", "native_tora_remain_h005_probe_20261002/full20_001", "#1f77b4", "--"),
)
STEPS, LANES, H = 400, 12, 0.05
PREFIX = HERE / "tora_remain_h005_fourway_t_x4"


def obj(path):
    return json.loads(path.read_text(encoding="utf-8"))


def descriptor(path):
    return {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size}


def raw_ranges(path):
    raw = path.read_bytes()
    if len(raw) != LANES * STEPS * ROW.size:
        raise ValueError(f"{path}: expected 4,800 native-layout records")
    records = {}
    for lane, step, h, *flat in ROW.iter_unpack(raw):
        if (not 0 <= lane < LANES or not 1 <= step <= STEPS or (lane, step) in records
                or not math.isclose(h, H, rel_tol=0, abs_tol=1e-12)):
            raise ValueError(f"{path}: invalid lane/step/h {lane}/{step}/{h}")
        tube, endpoint = [], []
        for state in range(4):
            lo, hi, elo, ehi = flat[4*state:4*state+4]
            if (not all(math.isfinite(x) for x in (lo, hi, elo, ehi))
                    or not -2 <= lo <= elo <= ehi <= hi <= 2):
                raise ValueError(f"{path}: invalid or unsafe x{state+1} at lane {lane}, step {step}")
            tube.append([lo, hi])
            endpoint.append([elo, ehi])
        records[lane, step] = (tube, endpoint)
    if len(records) != LANES * STEPS:
        raise ValueError(f"{path}: incomplete lane/step grid")
    return records


def derive(records):
    tube_union, terminal_union = [], []
    mean_width, max_width = [], []
    for state in range(4):
        all_tubes = [records[lane, step][0][state]
                     for lane in range(LANES) for step in range(1, STEPS+1)]
        ends = [records[lane, STEPS][1][state] for lane in range(LANES)]
        widths = [hi-lo for lo, hi in ends]
        tube_union.append([min(x[0] for x in all_tubes), max(x[1] for x in all_tubes)])
        terminal_union.append([min(x[0] for x in ends), max(x[1] for x in ends)])
        mean_width.append(sum(widths)/LANES)
        max_width.append(max(widths))
    x4_by_step = []
    for step in range(1, STEPS+1):
        values = [records[lane, step][0][3] for lane in range(LANES)]
        x4_by_step.append([min(x[0] for x in values), max(x[1] for x in values)])
    return tube_union, terminal_union, mean_width, max_width, x4_by_step


def method(label, name, color, style):
    run = RUNS / name
    native = label == "Flow* native"
    payload = run if native else run / "payload"
    range_path = payload / "ranges.bin"
    records = raw_ranges(range_path)
    tube, terminal, mean, maximum, x4 = derive(records)
    result = obj(run / "RESULT.json")
    if result.get("status") != "completed" or result.get("exit_code") != 0:
        raise ValueError(f"{name}: outer result incomplete")
    scan_path = run / ("INDEPENDENT_SCAN.json" if native else "INDEPENDENT_INTERVAL_SCAN.json")
    scan = obj(scan_path)
    if native:
        log = (run / "native.log").read_text(encoding="utf-8")
        if (scan.get("complete_grid") is not True or scan.get("range_records") != LANES*STEPS
                or scan.get("all_saved_components_finite_ordered_endpoint_contained_and_safe") is not True
                or scan.get("rpc_count") != 20 or scan.get("native_checker_label") != "VERIFIED"
                or [int(line.split()[1]) for line in log.splitlines() if line.startswith("Step ")] != list(range(20))
                or scan.get("saved_tube_union") != tube or scan.get("final_endpoint_union") != terminal):
            raise ValueError(f"{name}: saved ranges differ from native scan or run log")
        checker = "native author checker printed VERIFIED"
        acceptance = "binary ranges lack accepted flag; adjacent exit 0, 20 periods, 20 RPC"
    else:
        inner = obj(payload / "RESULT.json")
        obs = [json.loads(line) for line in (payload / "observations.jsonl").read_text().splitlines()]
        if (inner.get("status") != "completed" or inner.get("accepted_lane_substeps") != LANES*STEPS
                or len(obs) != STEPS or any(o.get("substep") != step or o.get("accepted_count") != LANES
                                            or o.get("rejected_lanes") != [] for step, o in enumerate(obs, 1))
                or scan.get("accepted_lane_substeps") != LANES*STEPS
                or scan.get("all_accepted_and_saved_tube_safe_prefix_substeps") != STEPS
                or scan.get("accepted_observed_tube_union") != tube
                or scan.get("last_observed_accepted_endpoint_union") != terminal):
            raise ValueError(f"{name}: saved ranges differ from author result/observations/scan")
        checker = "no explicit checker output; saved tubes inside Safe"
        acceptance = "12/12 accepted and no rejected lanes in each of 400 adjacent observations"
    return {"label": label, "run": name, "color": color, "line_style": style,
            "ranges": descriptor(range_path), "outer_result": descriptor(run / "RESULT.json"),
            "scan": descriptor(scan_path), "acceptance_evidence": acceptance,
            "checker_label": checker, "tube_union": tube, "terminal_endpoint_union": terminal,
            "terminal_per_box_mean_width": mean, "terminal_per_box_max_width": maximum,
            "tube_x4_union_per_step": x4}


def build():
    series = [method(*values) for values in METHODS]
    if series[1]["tube_x4_union_per_step"] != series[2]["tube_x4_union_per_step"]:
        raise ValueError("Huan/Xiangru x4 saved-tube equality claim changed")
    return {"schema": "archcomp26-tora-remain-h005-fourway-saved-nohash-v1",
            "contract": "2026 TORA remain, h=0.05 supplemental numerical profile; 12 boxes; 20 x 1 s periods",
            "safe_physical_box": [[-2, 2]]*4, "step_s": H, "steps": STEPS, "lanes": LANES,
            "view": "per-step union of 12 saved whole-step axis-aligned x4 tubes; four-state endpoint/tube widths derived from all records",
            "series": series,
            "qualification": "h=0.05 supplemental profile, separate from h=0.1 main table; saved numeric boxes and adjacent outcomes do not establish independent end-to-end floating-point NNCS soundness or stable timing rank"}


def write_csv(geometry):
    path = HERE / "terminal_and_tube_fourway.csv"
    fields = ("method", "state", "step_s", "steps", "lanes", "endpoint_lo", "endpoint_hi",
              "endpoint_union_width", "endpoint_mean_width_per_box", "endpoint_max_width_per_box",
              "full_tube_union_lo", "full_tube_union_hi", "full_tube_union_width", "source_run")
    with path.open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()
        for item in geometry["series"]:
            for state in range(4):
                lo, hi = item["terminal_endpoint_union"][state]
                tlo, thi = item["tube_union"][state]
                writer.writerow({"method": item["label"], "state": f"x{state+1}", "step_s": H,
                                 "steps": STEPS, "lanes": LANES, "endpoint_lo": repr(lo),
                                 "endpoint_hi": repr(hi), "endpoint_union_width": repr(hi-lo),
                                 "endpoint_mean_width_per_box": repr(item["terminal_per_box_mean_width"][state]),
                                 "endpoint_max_width_per_box": repr(item["terminal_per_box_max_width"][state]),
                                 "full_tube_union_lo": repr(tlo), "full_tube_union_hi": repr(thi),
                                 "full_tube_union_width": repr(thi-tlo), "source_run": item["run"]})


def write_matlab():
    lines = ["% 2026 TORA remain h=0.05 supplemental saved-box visualization; no solver run.",
             "% Reads adjacent geometry JSON. MATLAB execution was not part of generation.",
             "base = fileparts(mfilename('fullpath'));",
             "g = jsondecode(fileread(fullfile(base,'tora_remain_h005_fourway_t_x4.geometry.json')));",
             "t = 0:g.step_s:g.steps*g.step_s; figure('Color','w'); hold on; grid on;",
             "patch([0 20 20 0],[-2 -2 2 2],[0.83 0.92 0.84],'FaceAlpha',0.25,'EdgeColor','none','DisplayName','Safe x4 band');",
             "for j = 1:numel(g.series)",
             "  s = g.series(j); b = s.tube_x4_union_per_step; c = sscanf(s.color(2:end),'%2x%2x%2x')'/255;",
             "  stairs(t,[b(:,1);b(end,1)],'Color',c,'LineStyle',s.line_style,'LineWidth',1.2,'DisplayName',s.label);",
             "  stairs(t,[b(:,2);b(end,2)],'Color',c,'LineStyle',s.line_style,'LineWidth',1.2,'HandleVisibility','off');",
             "end",
             "plot([0 0],[0.5 0.6],'ko-','DisplayName','Initial x4');",
             "xlabel('t (s)'); ylabel('x4'); title('TORA remain h=0.05: four saved whole-step x4 tubes');",
             "xlim([0 20]); ylim([-2.1 2.1]); legend('Location','best'); hold off;"]
    PREFIX.with_suffix(".m").write_text("\n".join(lines)+"\n", encoding="utf-8")


def render(geometry):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax, terminal) = plt.subplots(1, 2, figsize=(12.5, 6.1),
                                       gridspec_kw={"width_ratios": [3.3, 1.4]})
    ax.axhspan(-2, 2, color="#cde8d0", alpha=0.35, label="Safe x4 band")
    ax.axhline(-2, color="#558a5d", linestyle="--", linewidth=0.8)
    ax.axhline(2, color="#558a5d", linestyle="--", linewidth=0.8)
    edges = [step*H for step in range(STEPS+1)]
    for row, item in enumerate(geometry["series"]):
        bounds = item["tube_x4_union_per_step"]
        lo, hi = [x[0] for x in bounds], [x[1] for x in bounds]
        ax.fill_between(edges, lo+[lo[-1]], hi+[hi[-1]], step="post",
                        color=item["color"], alpha=0.04)
        ax.step(edges, lo+[lo[-1]], where="post", color=item["color"],
                linestyle=item["line_style"], linewidth=1.2, label=item["label"])
        ax.step(edges, hi+[hi[-1]], where="post", color=item["color"],
                linestyle=item["line_style"], linewidth=1.2)
        end_lo, end_hi = item["terminal_endpoint_union"][3]
        terminal.hlines(row, end_lo, end_hi, color=item["color"], linewidth=4)
        terminal.plot([end_lo, end_hi], [row, row], "o", color=item["color"], markersize=5)
    ax.plot([0, 0], [0.5, 0.6], "ko-", linewidth=1.5, label="Initial x4")
    ax.set(xlim=(0, 20), ylim=(-2.1, 2.1), xlabel="t (s)", ylabel=r"$x_4$",
           title="Saved whole-step x4 tube, 12-box union")
    ax.grid(alpha=0.18)
    ax.legend(loc="lower left", fontsize=8, ncol=2)
    terminal.set(yticks=range(4), yticklabels=[x["label"] for x in geometry["series"]],
                 ylim=(3.45, -0.45), xlim=(-0.44, 0.08), xlabel=r"$x_4$",
                 title="T=20 endpoint union")
    terminal.grid(axis="x", alpha=0.18)
    fig.suptitle("TORA remain · h=0.05 supplemental numerical profile", fontsize=13)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.89, bottom=0.20, wspace=0.30)
    fig.text(0.08, 0.07,
             "All four methods: 12 × 400 saved numeric ranges to T=20, all tube boxes inside [-2,2]^4. Huan/Xiangru x4 bounds coincide.\n"
             "Native author checker: VERIFIED; P3/Huan/Xiangru: no explicit checker line. Axis boxes only; no independent NNCS proof.\n"
             "This h=0.05 profile is separate from the fixed h=0.1 main table; single-run times are not ranked.",
             fontsize=7.8, va="bottom")
    fig.savefig(PREFIX.with_suffix(".png"), dpi=180)
    fig.savefig(PREFIX.with_suffix(".pdf"))
    plt.close(fig)


def main():
    geometry = build()
    (PREFIX.with_suffix(".geometry.json")).write_text(
        json.dumps(geometry, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")
    write_csv(geometry)
    write_matlab()
    render(geometry)
    print(PREFIX)


if __name__ == "__main__":
    main()
