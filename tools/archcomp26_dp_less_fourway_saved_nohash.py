#!/usr/bin/env python3
"""Compare saved full-run DP less intervals; never launch a solver."""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from archcomp26_scan_dp_ranges_nohash import DTYPE, scan


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "docs/evidence/results/archcomp26_20261001"
OUT = BASE / "dp_less_fourway_split4_20261002"
STATES = ("theta1", "theta2", "theta1_dot", "theta2_dot")
METHODS = ("native", "Huan", "Xiangru", "ours/P3")
COLORS = ("#1f77b4", "#d95f02", "#7570b3", "#1b9e77")
FILES = {
    "native": BASE / "native_dp_less_full20_001/ranges.bin",
    "Huan": BASE / "author_dp_less_v1/huan_full20_001/payload/ranges.bin",
    "Xiangru": BASE / "author_dp_less_v1/xiangru_full20_001/payload/ranges.bin",
    "ours/P3": BASE / "dp_p3_affine_split4_v1/full225_smoke_001/attempt/data/observations.jsonl",
}


def load_binary(path):
    # Existing independent scanner checks exact 225 x 100 lane/step coverage.
    audit = scan(path, 225, 100, 0.01, -1.7, 2.0, 100)
    rows = np.fromfile(path, dtype=DTYPE)
    rows = rows[np.lexsort((rows["lane"], rows["step"]))]
    assert len(rows) == 22500 and audit["tube_inside_safe_box"]
    return rows["bounds"].reshape(100, 225, 4, 4), audit


def load_p3(path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 100
    bounds = np.empty((100, 225, 4, 4), dtype=np.float64)
    for step, row in enumerate(rows, 1):
        assert row["substep"] == step
        for name, expected in (("accepted", True), ("tube_endpoint_valid", True),
                               ("whole_tube_inside_safe_box", True), ("status", 0)):
            assert len(row[name]) == 225 and all(value == expected for value in row[name])
        tube = np.asarray(row["tube"], dtype=np.float64)
        endpoint = np.asarray(row["endpoint"], dtype=np.float64)
        assert tube.shape == endpoint.shape == (225, 4, 2)
        bounds[step - 1, :, :, 0:2] = tube
        bounds[step - 1, :, :, 2:4] = endpoint
    assert np.isfinite(bounds).all()
    assert (bounds[:, :, :, 0] <= bounds[:, :, :, 1]).all()
    assert (bounds[:, :, :, 2] <= bounds[:, :, :, 3]).all()
    assert (bounds[:, :, :, 2] >= bounds[:, :, :, 0]).all()
    assert (bounds[:, :, :, 3] <= bounds[:, :, :, 1]).all()
    assert (bounds[:, :, :, 0] >= -1.7).all() and (bounds[:, :, :, 1] <= 2.0).all()
    result = json.loads(path.with_name("RESULT.json").read_text(encoding="utf-8"))
    assert result["status"] == "completed" and result["completed_substeps"] == 100
    assert result["accepted_lane_substeps"] == 22500 and result["all_lanes_accepted"]
    assert result["end_to_end_strict_certificate"] is False
    return bounds, {"records": 22500, "tube_inside_safe_box": True}


def write_csv(data):
    with (OUT / "endpoint_stats.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("method", "state", "T1_endpoint_union_lo", "T1_endpoint_union_hi",
                         "T1_endpoint_union_width", "T1_per_box_width_mean", "T1_per_box_width_max",
                         "all_time_tube_union_lo", "all_time_tube_union_hi"))
        for method in METHODS:
            bounds = data[method]
            for dim, state in enumerate(STATES):
                endpoint = bounds[-1, :, dim, 2:4]
                lo, hi = float(endpoint[:, 0].min()), float(endpoint[:, 1].max())
                widths = endpoint[:, 1] - endpoint[:, 0]
                tube = bounds[:, :, dim, 0:2]
                writer.writerow((method, state, lo, hi, hi - lo, float(widths.mean()),
                                 float(widths.max()), float(tube[:, :, 0].min()),
                                 float(tube[:, :, 1].max())))
    with (OUT / "tube_union.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("method", "state", "step", "t_start", "t_end", "tube_lo", "tube_hi"))
        for method in METHODS:
            for dim, state in enumerate(STATES):
                bounds = data[method][:, :, dim, 0:2]
                for step in range(1, 101):
                    writer.writerow((method, state, step, (step - 1) / 100, step / 100,
                                     float(bounds[step - 1, :, 0].min()),
                                     float(bounds[step - 1, :, 1].max())))


def plot(data):
    edges = np.arange(101) / 100
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8), sharex=True)
    for dim, ax in enumerate(axes.flat):
        for method, color in zip(METHODS, COLORS):
            bounds = data[method][:, :, dim, 0:2]
            lo, hi = bounds[:, :, 0].min(axis=1), bounds[:, :, 1].max(axis=1)
            ax.fill_between(edges, np.r_[lo, lo[-1]], np.r_[hi, hi[-1]],
                            step="post", color=color, alpha=0.045)
            style = "--" if method == "Xiangru" else "-"
            ax.step(edges, np.r_[lo, lo[-1]], where="post", color=color,
                    linestyle=style, linewidth=1.1, label=method)
            ax.step(edges, np.r_[hi, hi[-1]], where="post", color=color,
                    linestyle=style, linewidth=1.1)
        visible_y = ax.get_ylim()
        ax.axhspan(-1.7, 2.0, color="#d8eedb", alpha=0.55, zorder=-10,
                   label="Safe [-1.7, 2]")
        ax.set_ylim(visible_y)
        ax.plot([0, 0], [1, 1.3], "ko-", markersize=3, linewidth=2,
                label="Initial [1, 1.3] at t=0")
        ax.set_title(STATES[dim])
        ax.set_ylabel("state interval")
        ax.grid(alpha=0.2)
        ax.set_xlim(0, 1)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.115),
               ncol=6, fontsize=8)
    for ax in axes[1]:
        ax.set_xlabel("t (s)")
    fig.suptitle("Double Pendulum less robust: saved 225-box tube unions, four methods")
    fig.text(0.07, 0.025,
             "Green denotes the visible part of Safe [-1.7,2]; its bounds extend beyond each cropped y-axis. "
             "Black segment: initial [1,1.3] at t=0.\n"
             "Each step unions 225 saved axis-aligned tube boxes. Huan and Xiangru bounds coincide; "
             "native acceptance is absent from ranges.bin.\n"
             "One run per method; box projection, not trajectories or an independent NNCS certificate.",
             fontsize=8)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.91, bottom=0.21, hspace=0.25, wspace=0.18)
    fig.savefig(OUT / "fourway_tube_union.png", dpi=180)
    fig.savefig(OUT / "fourway_tube_union.pdf")
    plt.close(fig)


def write_matlab():
    (OUT / "fourway_tube_union.m").write_text("""% Plot saved four-method DP less tube unions. This script runs no solver.
% Requires tube_union.csv in the same directory; generated here, not run in MATLAB.
T = readtable(fullfile(fileparts(mfilename('fullpath')), 'tube_union.csv'));
methods = {'native', 'Huan', 'Xiangru', 'ours/P3'};
states = {'theta1', 'theta2', 'theta1_dot', 'theta2_dot'};
colors = [31 119 180; 217 95 2; 117 112 179; 27 158 119] / 255;
figure('Color', 'w');
for d = 1:4
    subplot(2, 2, d); hold on; grid on;
    V = T(strcmp(T.state, states{d}), :);
    yBounds = [min(V.tube_lo), max(V.tube_hi)];
    pad = 0.05 * (yBounds(2) - yBounds(1));
    ylim([yBounds(1)-pad, yBounds(2)+pad]);
    patch([0 1 1 0], [-1.7 -1.7 2 2], [0.85 0.93 0.86], ...
          'FaceAlpha', 0.55, 'EdgeColor', 'none', ...
          'DisplayName', 'Safe [-1.7, 2]');
    for m = 1:4
        R = T(strcmp(T.method, methods{m}) & strcmp(T.state, states{d}), :);
        assert(height(R) == 100 && all(R.step == (1:100)'));
        x = [0; R.t_end]; lo = [R.tube_lo; R.tube_lo(end)];
        hi = [R.tube_hi; R.tube_hi(end)];
        lineStyle = '-'; if m == 3, lineStyle = '--'; end
        stairs(x, lo, 'Color', colors(m, :), 'LineStyle', lineStyle, ...
               'DisplayName', methods{m});
        stairs(x, hi, 'Color', colors(m, :), 'LineStyle', lineStyle, ...
               'HandleVisibility', 'off');
    end
    plot([0 0], [1 1.3], 'ko-', 'MarkerSize', 3, 'LineWidth', 2, ...
         'DisplayName', 'Initial [1, 1.3] at t=0');
    title([states{d} ' | Safe [-1.7, 2] (cropped)']);
    xlabel('t (s)'); ylabel('state interval'); xlim([0 1]);
    if d == 1, legend('Location', 'best'); end
end
sgtitle('Double Pendulum less robust: saved 225-box tube unions');
""", encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = {}
    for method, path in FILES.items():
        assert path.is_file(), path
        data[method], audit = load_p3(path) if method == "ours/P3" else load_binary(path)
        assert audit["records"] == 22500 and audit["tube_inside_safe_box"]
        if method in ("Huan", "Xiangru"):
            child = json.loads(path.with_name("RESULT.json").read_text(encoding="utf-8"))
            assert child["status"] == "completed" and child["expected_substeps"] == 100
            assert child["accepted_lane_substeps"] == 22500
    native_dir = FILES["native"].parent
    native_result = json.loads((native_dir / "RESULT.json").read_text(encoding="utf-8"))
    assert native_result["status"] == "completed"
    assert "VERIFIED" in (native_dir / "native.log").read_text(encoding="utf-8").splitlines()
    assert np.array_equal(data["Huan"], data["Xiangru"])
    write_csv(data)
    plot(data)
    write_matlab()
    print(f"{OUT}: four methods x 225 boxes x 100 steps; all intervals ordered")


if __name__ == "__main__":
    main()
