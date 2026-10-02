#!/usr/bin/env python3
"""Plot the four saved TORA reach-sigmoid tube projections, without rerunning a solver."""

import json
import math
from pathlib import Path
import struct


HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent
ROOT = HERE.parents[4]
ROW = struct.Struct("<QQd16d")
STEPS = 500
H = 0.01
TARGET = {"x1": [-0.1, 0.2], "x2": [-0.9, -0.6]}
RUNS = (
    ("Huan", "tora_reach_sigmoid_official2026_mat_u11_full500_huan_002", "#d95f02", "-"),
    ("Xiangru", "tora_reach_sigmoid_official2026_mat_u11_xiangru_full500_001", "#7570b3", ":"),
    ("ours/P3", "tora_reach_sigmoid_official2026_mat_u11_p3_full500_001", "#1b9e77", "-"),
    ("Flow* native", "native_tora_reach_sigmoid_u11_full10_001", "#1f77b4", "--"),
)
PREFIX = HERE / "tora_reach_sigmoid_u11_fourway_saved"


def obj(path):
    return json.loads(path.read_text(encoding="utf-8"))


def source(path):
    return {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size}


def ranges(path):
    raw = path.read_bytes()
    if len(raw) != STEPS * ROW.size:
        raise ValueError(f"{path}: expected {STEPS} complete four-state records")
    tubes = {"x1": [], "x2": []}
    last = None
    for index, (lane, step, h, *values) in enumerate(ROW.iter_unpack(raw), 1):
        if lane != 0 or step != index or not math.isclose(h, H, abs_tol=1e-12):
            raise ValueError(f"{path}: lane, step, or h mismatch at {index}")
        last = []
        for state in range(4):
            lo, hi, end_lo, end_hi = values[4 * state:4 * state + 4]
            if (not all(math.isfinite(v) for v in (lo, hi, end_lo, end_hi))
                    or lo > hi or end_lo > end_hi or lo > end_lo or end_hi > hi):
                raise ValueError(f"{path}: invalid tube/endpoint at step {index}, state {state + 1}")
            if state < 2:
                tubes[f"x{state + 1}"].append([lo, hi])
            last.append([end_lo, end_hi])
    return tubes, last


def build_geometry():
    series = []
    for label, name, color, style in RUNS:
        run = RESULTS / name
        result = obj(run / "RESULT.json")
        start = obj(run / "START.json")
        scan_path = run / ("RANGE_SCAN.json" if label == "Flow* native" else "INDEPENDENT_INTERVAL_SCAN.json")
        scan = obj(scan_path)
        tubes, endpoint = ranges(run / "ranges.bin")
        if label == "Flow* native":
            log_path = run / "native.log"
            log = log_path.read_text(encoding="utf-8")
            if (result.get("status") != "completed" or result.get("exit_code") != 0
                    or scan.get("complete_grid") is not True or scan.get("records") != STEPS
                    or scan.get("nonfinite_records") != 0 or scan.get("reversed_component_intervals") != 0
                    or scan.get("endpoint_outside_same_step_tube_components") != 0
                    or "Step 9\nVERIFIED\n" not in log
                    or scan.get("final_endpoint_union") != endpoint
                    or start.get("contract_label") != "tora-reach-official2026-mat-u11f"):
                raise ValueError(f"{name}: native outcome/scan does not match saved ranges")
            property_status = "native author endpoint checker printed VERIFIED"
            acceptance = "native binary has no accepted field; complete 500-record grid and adjacent exit 0"
            observation = None
        else:
            observation = run / "observations.jsonl"
            rows = [json.loads(line) for line in observation.read_text(encoding="utf-8").splitlines()]
            contract = start.get("contract", {})
            if (result.get("status") != "completed_full_numerical_horizon"
                    or result.get("full_horizon_completed") is not True
                    or result.get("accepted_substeps") != STEPS
                    or result.get("property_evaluated") is not False
                    or len(rows) != STEPS
                    or any(row.get("substep") != i or row.get("accepted") is not True
                           or row.get("interval_valid") is not True for i, row in enumerate(rows, 1))
                    or scan.get("valid_saved_substeps") != STEPS
                    or scan.get("full_numerical_horizon_completed") is not True
                    or scan.get("endpoint_outside_same_step_tube_components") != 0
                    or [scan["last_saved_endpoint_by_state"][f"x{i}"] for i in range(1, 5)] != endpoint
                    or contract.get("internal_control") != "u = 11 * f(x) + 0"
                    or contract.get("external_control") != "output_scale = 1; output_offset = 0; dx4 = u1"
                    or contract.get("ode_substeps") != STEPS):
                raise ValueError(f"{name}: author outcome/contract/scan does not match saved ranges")
            property_status = "property checker not run; saved endpoint inclusion only"
            acceptance = "all 500 adjacent observations accepted=true and interval_valid=true"
        for i, state in enumerate(("x1", "x2")):
            a, b = endpoint[i]
            target_lo, target_hi = TARGET[state]
            if not target_lo <= a <= b <= target_hi:
                raise ValueError(f"{name}: final {state} does not lie in target")
        series.append({
            "label": label, "color": color, "line_style": style, "run": name,
            "range_source": source(run / "ranges.bin"), "run_start": source(run / "START.json"),
            "adjacent_result": source(run / "RESULT.json"), "adjacent_scan": source(scan_path),
            "adjacent_observations": source(observation) if observation else None,
            "adjacent_native_log": source(log_path) if label == "Flow* native" else None,
            "acceptance_evidence": acceptance, "property_status": property_status,
            "tube_x1": tubes["x1"], "tube_x2": tubes["x2"],
            "endpoint_T5_x1": endpoint[0], "endpoint_T5_x2": endpoint[1],
        })
    if series[0]["tube_x1"] != series[1]["tube_x1"] or series[0]["tube_x2"] != series[1]["tube_x2"]:
        raise ValueError("saved Huan/Xiangru tube coincidence changed")
    return {
        "schema": "archcomp26-tora-reach-sigmoid-fourway-saved-tube-nohash-v1",
        "contract": "2026 official four-layer sigmoid model, u=11f, four-state TORA, full initial box",
        "initial_box_x1_x2": [[-0.77, -0.75], [-0.45, -0.43]],
        "target_at_T5_x1_x2": TARGET, "step_s": H, "steps": STEPS,
        "view": "saved axis-aligned whole-step tube x1/x2, plus T=5 endpoint bound offsets against Huan",
        "series": series,
        "qualification": "500 numeric steps per method; endpoint inclusion is sufficient numeric evidence for within-5s target, not an independent end-to-end floating-point NNCS certificate; correlations and octagon supports unavailable",
        "source_binding": "raw record layout, 500-step grid, finite ordered intervals, endpoint containment, adjacent observations/results/scans; no content digest",
    }


def write_matlab(path):
    lines = [
        "% Four-method saved TORA reach-sigmoid intervals; no solver run here.",
        "% Reads adjacent geometry JSON; requires MATLAB with jsondecode. Not executed by generator.",
        "base = fileparts(mfilename('fullpath'));",
        "g = jsondecode(fileread(fullfile(base, 'tora_reach_sigmoid_u11_fourway_saved.geometry.json')));",
        "t = 0:g.step_s:g.steps*g.step_s; figure('Color','w','Position',[100 100 1300 760]);",
        "for dim = 1:2",
        "  subplot(2,2,2*dim-1); hold on; grid on;",
        "  if dim == 1, key = 'tube_x1'; target = g.target_at_T5_x1_x2.x1; else, key = 'tube_x2'; target = g.target_at_T5_x1_x2.x2; end",
        "  plot([5 5],target,'m-','LineWidth',4,'DisplayName','T=5 target');",
        "  for j = 1:numel(g.series)",
        "    s = g.series(j); b = s.(key); c = sscanf(s.color(2:end),'%2x%2x%2x')'/255;",
        "    for k = 1:g.steps, patch([t(k) t(k+1) t(k+1) t(k)], [b(k,1) b(k,1) b(k,2) b(k,2)], c, 'FaceAlpha',0.035,'EdgeColor','none'); end",
        "    stairs(t,[b(:,1);b(end,1)],'Color',c,'LineStyle',s.line_style,'LineWidth',1.2,'DisplayName',s.label);",
        "    stairs(t,[b(:,2);b(end,2)],'Color',c,'LineStyle',s.line_style,'LineWidth',1.2,'HandleVisibility','off');",
        "  end",
        "  xlabel('t (s)'); ylabel(sprintf('x%d',dim)); xlim([0 5]); title(sprintf('Saved whole-step tube: x%d',dim));",
        "  if dim == 1, legend('Location','best'); end",
        "  subplot(2,2,2*dim); hold on; grid on;",
        "  if dim == 1, endpoint_key = 'endpoint_T5_x1'; else, endpoint_key = 'endpoint_T5_x2'; end",
        "  ref = g.series(1).(endpoint_key);",
        "  for j = 1:numel(g.series)",
        "    s = g.series(j); e = s.(endpoint_key); c = sscanf(s.color(2:end),'%2x%2x%2x')'/255;",
        "    plot((e(1)-ref(1))*1e4,j,'o','Color',c,'MarkerFaceColor',c,'MarkerSize',7);",
        "    plot((e(2)-ref(2))*1e4,j,'^','Color',c,'MarkerFaceColor',c,'MarkerSize',7);",
        "  end",
        "  xline(0,'k:'); yticks(1:numel(g.series)); yticklabels({g.series.label}); ylim([0.5 numel(g.series)+0.5]);",
        "  xlabel('Endpoint bound offset vs Huan (10^{-4})'); title(sprintf('T=5 x%d bounds: circle lower, triangle upper',dim));",
        "end",
        "sgtitle('TORA reach-sigmoid, official u=11f: saved numeric tubes');",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render(g):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13.7, 8.3),
                             gridspec_kw={"width_ratios": [3.25, 1.55]})
    edges = [i * H for i in range(STEPS + 1)]
    for dim, key in enumerate(("x1", "x2")):
        ax, offset = axes[dim]
        lo_target, hi_target = TARGET[key]
        ax.plot([5, 5], [lo_target, hi_target], color="#b2188b", linewidth=5,
                solid_capstyle="round", label="T=5 target")
        for item in g["series"]:
            bounds = item[f"tube_{key}"]
            lower = [row[0] for row in bounds]
            upper = [row[1] for row in bounds]
            ax.fill_between(edges, lower + lower[-1:], upper + upper[-1:], step="post",
                            color=item["color"], alpha=0.035)
            ax.step(edges, lower + lower[-1:], where="post", color=item["color"],
                    linestyle=item["line_style"], linewidth=1.25, label=item["label"])
            ax.step(edges, upper + upper[-1:], where="post", color=item["color"],
                    linestyle=item["line_style"], linewidth=1.25)
        ax.set(xlim=(0, 5.01), xlabel="t (s)", ylabel=rf"${key}$",
               title=rf"Saved whole-step ${key}$ tube (500 × 0.01 s)")
        ax.grid(alpha=0.18)
        if dim == 0:
            ax.legend(loc="upper left", ncol=3, fontsize=8)
        ref = g["series"][0][f"endpoint_T5_{key}"]
        for row, item in enumerate(g["series"]):
            terminal = item[f"endpoint_T5_{key}"]
            for value, baseline, marker in zip(terminal, ref, ("o", "^")):
                offset.plot((value - baseline) * 1e4, row, marker=marker,
                            color=item["color"], markersize=7)
        offset.axvline(0, color="black", linestyle=":", linewidth=1)
        offset.set(yticks=range(4), yticklabels=[s["label"] for s in g["series"]],
                   ylim=(3.45, -0.45), xlabel=r"Endpoint bound offset vs Huan ($10^{-4}$)",
                   title=rf"$T=5$ ${key}$: ○ lower, △ upper")
        offset.grid(axis="x", alpha=0.2)
    fig.suptitle("TORA reach-sigmoid · official 2026 model, $u=11f$ · full initial box",
                 fontsize=14, y=0.975)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.90, bottom=0.19,
                        hspace=0.38, wspace=0.28)
    fig.text(0.08, 0.075,
             "Four full numeric T=5 histories; Huan and Xiangru saved tube bounds coincide. Magenta marks target only at T=5.\n"
             "All saved endpoints lie in target. Author Huan/Xiangru/P3 property checkers were not run; native author endpoint checker: VERIFIED.\n"
             "Axis-aligned saved boxes; adjacent outcomes are not content-bound. No independent end-to-end floating-point NNCS certificate.",
             fontsize=8, va="bottom")
    fig.savefig(PREFIX.with_suffix(".png"), dpi=180)
    fig.savefig(PREFIX.with_suffix(".pdf"))
    plt.close(fig)


def main():
    g = build_geometry()
    PREFIX.with_suffix(".geometry.json").write_text(
        json.dumps(g, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    write_matlab(PREFIX.with_suffix(".m"))
    render(g)
    print(PREFIX)


if __name__ == "__main__":
    main()
