#!/usr/bin/env python3
"""Draw four existing Unicycle whole-step tubes and T=10 endpoints; no solver."""

import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
CONTRACT = HERE.parents[1]
sys.path.insert(0, str(CONTRACT))
import audit_saved_endpoint_window_nohash as saved  # noqa: E402


PREFIX = HERE / "unicycle_paper_speed_fourway_saved"
COLORS = ("#1b9e77", "#d95f02", "#7570b3", "#1f77b4")
METHODS = (("p3", "ours/P3", "-"), ("huan", "Huan", "-"),
           ("xiangru", "Xiangru", ":"), ("flowstar_native", "Flow* native", "--"))
TARGET = saved.TARGET
STEPS, H = saved.STEPS, saved.H


def build():
    previous = json.loads((CONTRACT / "SAVED_ENDPOINT_WINDOW_AUDIT_20261002.json").read_text())
    indexed = {entry["method"]: entry for entry in previous["methods"]}
    series = []
    for (method, label, style), color in zip(METHODS, COLORS):
        data, source = saved.native() if method == "flowstar_native" else saved.author(method)
        current = saved.audit(method, data, source)
        if current != indexed[method]:
            raise ValueError(f"{method}: saved-window audit differs from raw ranges")
        last = current["last_saved_endpoint"]
        contained = all(TARGET[i][0] <= last[i][0] <= last[i][1] <= TARGET[i][1]
                        for i in range(4))
        if contained != (method in ("p3", "flowstar_native")):
            raise ValueError(f"{method}: T=10 target classification changed")
        series.append({
            "method": method, "label": label, "color": color, "line_style": style,
            "source_range": source["ranges"], "source_bytes": source["bytes"],
            "acceptance_evidence": source["acceptance"],
            "first_full_endpoint_target_t_s": current["first_saved_endpoint_target_t_s"],
            "saved_full_endpoint_target_step_spans": current["saved_endpoint_target_step_spans"],
            "terminal_physical_endpoint": last, "terminal_full_target_contained": contained,
            "terminal_property_label": ("native author endpoint checker printed VERIFIED" if method == "flowstar_native"
                                        else "author property UNKNOWN" if method in ("huan", "xiangru")
                                        else "saved numeric endpoint sufficient; no author VERIFIED line"),
            "tube_x1": [tube[0] for endpoint, tube in data],
            "tube_x3": [tube[2] for endpoint, tube in data],
        })
    if (series[1]["tube_x1"] != series[2]["tube_x1"]
            or series[1]["tube_x3"] != series[2]["tube_x3"]):
        raise ValueError("Huan/Xiangru saved x1/x3 tube equality changed")
    return {
        "schema": "archcomp26-unicycle-paper-speed-fourway-saved-plot-nohash-v1",
        "contract": "2026 paper Unicycle: w constant per trajectory, only speed derivative; complete initial box",
        "step_s": H, "steps": STEPS, "full_horizon_s": 10,
        "target_at_T10_by_physical_state": {f"x{i+1}": list(pair) for i, pair in enumerate(TARGET)},
        "view": "whole-step x1/x3 axis-aligned tube (four methods on each common axis); T=10 x3/x4 endpoint intervals separately",
        "saved_window_audit": str((CONTRACT / "SAVED_ENDPOINT_WINDOW_AUDIT_20261002.json").relative_to(HERE.parents[6])),
        "series": series,
        "qualification": "T=10 endpoint inclusion is a sufficient saved-numeric observation for reach within 10 s; no inclusion at saved endpoints does not prove unreachability; no independent end-to-end floating-point NNCS proof"}


def write_matlab():
    lines = [
        "% Saved 2026 Unicycle paper-speed-constant-w four-way figure. No solver is run.",
        "% Reads adjacent geometry JSON. MATLAB execution was not part of generation.",
        "base = fileparts(mfilename('fullpath'));",
        "g = jsondecode(fileread(fullfile(base,'unicycle_paper_speed_fourway_saved.geometry.json')));",
        "t = 0:g.step_s:g.full_horizon_s; figure('Color','w','Position',[100 100 1350 800]);",
        "for j = 1:2",
        "  if j == 1, key = 'tube_x1'; state = 1; else, key = 'tube_x3'; state = 3; end",
        "  subplot(2,2,2*j-1); hold on; grid on;",
        "  target = g.target_at_T10_by_physical_state.(sprintf('x%d',state));",
        "  plot([10 10],target,'m-','LineWidth',4,'DisplayName','T=10 target');",
        "  for k = 1:numel(g.series)",
        "    s = g.series(k); b = s.(key); c = sscanf(s.color(2:end),'%2x%2x%2x')'/255;",
        "    for q = 1:g.steps, patch([t(q) t(q+1) t(q+1) t(q)], [b(q,1) b(q,1) b(q,2) b(q,2)], c, 'FaceAlpha',0.035,'EdgeColor','none'); end",
        "    stairs(t,[b(:,1);b(end,1)],'Color',c,'LineStyle',s.line_style,'LineWidth',1.2,'DisplayName',s.label);",
        "    stairs(t,[b(:,2);b(end,2)],'Color',c,'LineStyle',s.line_style,'LineWidth',1.2,'HandleVisibility','off');",
        "  end",
        "  xlabel('t (s)'); ylabel(sprintf('x%d',state)); xlim([0 10]);",
        "  title(sprintf('Saved whole-step x%d tube',state)); if j == 1, legend('Location','best'); end",
        "end",
        "for j = 1:2",
        "  if j == 1, state = 3; else, state = 4; end",
        "  subplot(2,2,2*j); hold on; grid on;",
        "  target = g.target_at_T10_by_physical_state.(sprintf('x%d',state));",
        "  patch([target(1) target(2) target(2) target(1)],[0.5 0.5 4.5 4.5],[0.83 0.92 0.84],'FaceAlpha',0.35,'EdgeColor','none');",
        "  for k = 1:numel(g.series)",
        "    s = g.series(k); e = s.terminal_physical_endpoint(state,:); c = sscanf(s.color(2:end),'%2x%2x%2x')'/255;",
        "    plot(e,[k k],'-o','Color',c,'LineWidth',3,'MarkerFaceColor',c);",
        "  end",
        "  plot([target(1) target(1)],[0.5 4.5],'k--');",
        "  yticks(1:numel(g.series)); yticklabels({g.series.label}); ylim([0.5 4.5]);",
        "  if state == 4, xlim([-0.35 -0.18]); xticks([-0.35 -0.30 -0.25 -0.20]); end",
        "  xlabel(sprintf('T=10 endpoint x%d',state)); title(sprintf('Target x%d: [%g,%g]',state,target));",
        "end",
        "sgtitle('Unicycle paper-speed constant-w: saved four-method numeric comparison');",
    ]
    PREFIX.with_suffix(".m").write_text("\n".join(lines) + "\n", encoding="utf-8")


def render(geometry):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13.7, 8.5),
                             gridspec_kw={"width_ratios": [3.1, 1.45]})
    edges = [i * H for i in range(STEPS+1)]
    for row, state in enumerate((1, 3)):
        ax = axes[row, 0]
        target = TARGET[state-1]
        ax.plot([10, 10], target, color="#b2188b", linewidth=5,
                solid_capstyle="round", label="T=10 target")
        for item in geometry["series"]:
            bounds = item[f"tube_x{state}"]
            lower, upper = [x[0] for x in bounds], [x[1] for x in bounds]
            ax.fill_between(edges, lower+[lower[-1]], upper+[upper[-1]], step="post",
                            color=item["color"], alpha=0.035)
            ax.step(edges, lower+[lower[-1]], where="post", color=item["color"],
                    linestyle=item["line_style"], linewidth=1.25, label=item["label"])
            ax.step(edges, upper+[upper[-1]], where="post", color=item["color"],
                    linestyle=item["line_style"], linewidth=1.25)
        ax.set(xlim=(0, 10.01), xlabel="t (s)", ylabel=rf"$x_{state}$",
               title=rf"Saved whole-step $x_{state}$ tube (500 × 0.02 s)")
        ax.grid(alpha=0.18)
        if row == 0:
            ax.legend(loc="upper right", ncol=3, fontsize=8)
    for row, state in enumerate((3, 4)):
        ax = axes[row, 1]
        target = TARGET[state-1]
        ax.axvspan(*target, color="#cfe8d1", alpha=0.60, label="T=10 target")
        ax.axvline(target[0], color="#54875d", linestyle="--", linewidth=1)
        for index, item in enumerate(geometry["series"]):
            lo, hi = item["terminal_physical_endpoint"][state-1]
            ax.hlines(index, lo, hi, color=item["color"], linewidth=4)
            ax.plot([lo, hi], [index, index], "o", color=item["color"], markersize=5)
        ax.set(yticks=range(4), yticklabels=[s["label"] for s in geometry["series"]],
               ylim=(3.45, -0.45), xlabel=rf"$x_{state}$",
               title=rf"$T=10$ $x_{state}$ endpoint; target [{target[0]:g},{target[1]:g}]")
        ax.set_xlim((-.13, .08) if state == 3 else (-.35, -.18))
        if state == 4:
            ax.set_xticks((-.35, -.30, -.25, -.20))
        ax.grid(axis="x", alpha=0.18)
    fig.suptitle("Unicycle · 2026 paper RHS, constant $w$ in speed derivative · full initial box",
                 fontsize=13, y=0.975)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.90, bottom=0.18,
                        hspace=0.40, wspace=0.30)
    fig.text(0.07, 0.067,
             "Magenta target segments apply only at T=10; right panels show the saved endpoint box, separately from whole-step tubes.\n"
             "P3/native T=10 full endpoint boxes are inside target. Huan/Xiangru x3 and x4 extend below target: saved criterion Unknown, not unreachable.\n"
             "Huan/Xiangru x1/x3 saved bounds coincide. Axis boxes, adjacent run outcomes; no independent end-to-end floating-point NNCS proof.",
             fontsize=7.8, va="bottom")
    fig.savefig(PREFIX.with_suffix(".png"), dpi=180)
    fig.savefig(PREFIX.with_suffix(".pdf"))
    plt.close(fig)


def main():
    geometry = build()
    PREFIX.with_suffix(".geometry.json").write_text(
        json.dumps(geometry, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")
    write_matlab()
    render(geometry)
    print(PREFIX)


if __name__ == "__main__":
    main()
