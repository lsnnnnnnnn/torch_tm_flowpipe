#!/usr/bin/env python3
"""Plot the four saved two-physical-state pendulum runs; never call a solver."""

import csv
import json
import struct
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "docs/evidence/results/archcomp26_20261001"
OUT = BASE / "sp_two_state_fourway_saved_20261002"
RUNS = {
    "P3": BASE / "single_pendulum_two_state_p3_full20_001",
    "Huan": BASE / "single_pendulum_two_state_huan_full20_001",
    "Xiangru": BASE / "single_pendulum_two_state_xiangru_full20_001",
    "Flow* native": BASE / "native_sp_two_state_full20_001",
}
COLORS = {"P3": "#0072B2", "Huan": "#D55E00", "Xiangru": "#CC79A7", "Flow* native": "#009E73"}
STATES = ("x1", "x2")
RECORD = struct.Struct("<QQd8d")


def read_gpu(run):
    result = json.loads((run / "data/RESULT.json").read_text())
    assert result["status"] == "completed" and result["completed_substeps"] == 100
    assert result["all_substeps_accepted"] and result["metrics_broken"] == 0
    records = [json.loads(line) for line in (run / "data/ranges.jsonl").read_text().splitlines()]
    assert len(records) == 100
    bounds = np.empty((100, 2, 4), dtype=float)
    for step, row in enumerate(records, 1):
        assert row["substep"] == step and row["accepted"]
        for dim in range(2):
            bounds[step - 1, dim] = (*row["tube"][dim], *row["endpoint"][dim])
    return bounds


def read_native(run):
    result = json.loads((run / "RESULT.json").read_text())
    assert result["status"] == "completed" and result["exit_code"] == 0
    log = (run / "native.log").read_text().splitlines()
    assert "COMPLETED_PERIODS 20/20" in log and "VERIFIED" in log
    raw = (run / "ranges.bin").read_bytes()
    assert len(raw) == 100 * RECORD.size
    bounds = np.empty((100, 2, 4), dtype=float)
    for step, record in enumerate(RECORD.iter_unpack(raw), 1):
        lane, saved_step, h, *values = record
        assert lane == 0 and saved_step == step and abs(h - 0.01) < 1e-12
        bounds[step - 1] = np.asarray(values).reshape(2, 4)
    return bounds


def validate(bounds):
    assert bounds.shape == (100, 2, 4) and np.isfinite(bounds).all()
    assert (bounds[:, :, 0] <= bounds[:, :, 1]).all()
    assert (bounds[:, :, 2] <= bounds[:, :, 3]).all()
    assert (bounds[:, :, 0] <= bounds[:, :, 2]).all()
    assert (bounds[:, :, 3] <= bounds[:, :, 1]).all()
    assert (bounds[50:, 0, 0] >= 0).all() and (bounds[50:, 0, 1] <= 1).all()


def write_tables(data):
    with (OUT / "saved_tubes.csv").open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("method", "state", "step", "t_start", "t_end", "tube_lo", "tube_hi", "endpoint_lo", "endpoint_hi"))
        for method, bounds in data.items():
            for step in range(100):
                for dim, state in enumerate(STATES):
                    writer.writerow((method, state, step + 1, step / 100, (step + 1) / 100,
                                     *map(float, bounds[step, dim])))
    with (OUT / "endpoint_and_window_widths.csv").open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("method", "state", "T1_endpoint_lo", "T1_endpoint_hi", "T1_endpoint_width",
                         "window_tube_lo", "window_tube_hi", "window_endpoint_width_mean",
                         "window_endpoint_width_max"))
        for method, bounds in data.items():
            for dim, state in enumerate(STATES):
                final = bounds[-1, dim, 2:4]
                window = bounds[50:, dim]
                widths = window[:, 3] - window[:, 2]
                writer.writerow((method, state, *map(float, final), float(final[1] - final[0]),
                                 float(window[:, 0].min()), float(window[:, 1].max()),
                                 float(widths.mean()), float(widths.max())))


def draw(data):
    edges = np.arange(101) / 100
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.8))
    for dim, state in enumerate(STATES):
        full, detail = axes[0, dim], axes[1, dim]
        for method, bounds in data.items():
            color = COLORS[method]
            style = "--" if method == "Xiangru" else "-"
            tube = bounds[:, dim, :2]
            for col in range(2):
                full.step(edges, np.r_[tube[:, col], tube[-1, col]], where="post",
                          color=color, ls=style, lw=1.35, label=method if col == 0 else None)
            endpoint = bounds[:, dim, 2:4] - data["Huan"][:, dim, 2:4]
            for col in range(2):
                detail.step(edges[1:], endpoint[:, col], where="post",
                            color=color, ls="--" if col else style, lw=1.35)
        full.set_title(f"{state}: all 100 saved tube intervals")
        detail.set_title(f"{state}: endpoint difference from Huan, t = 0.8–1")
        detail.set_xlim(0.8, 1.0)
        detail.set_xlabel("time (s)")
        full.set_ylabel(state)
        detail.set_ylabel(f"Δ{state} endpoint")
        for ax in (full, detail):
            ax.grid(alpha=0.2)
        full.set_xlim(0, 1)
        detail.axhline(0, color="0.4", lw=0.8)
        if dim == 0:
            full.axhspan(0, 1, color="#DDEDDD", alpha=0.5, zorder=-5)
            full.axvline(0.5, color="0.5", ls=":", lw=0.8)
    axes[0, 0].legend(ncol=2, fontsize=8, loc="lower left")
    fig.suptitle("Single Pendulum, named two-physical-state profile: saved four-method bounds")
    fig.text(0.07, 0.025,
             "Huan and Xiangru saved bounds coincide exactly. Lower panels subtract Huan bounds; solid = lower, dashed = upper.\n"
             "Shading marks x1 target [0,1], checked on t = 0.5–1. Official three-state MATLAB identity remains unresolved. "
             "Axis projections are not an independent NNCS certificate.", fontsize=8)
    fig.subplots_adjust(left=0.095, right=0.98, top=0.91, bottom=0.145, hspace=0.32, wspace=0.20)
    fig.savefig(OUT / "fourway_saved_bounds.png", dpi=180)
    fig.savefig(OUT / "fourway_saved_bounds.pdf")
    plt.close(fig)


def write_matlab():
    (OUT / "fourway_saved_bounds.m").write_text("""% Saved Single Pendulum two-state boxes; no solver or property checker is run.
% Generated from saved_tubes.csv. This example has not been run in MATLAB.
T = readtable(fullfile(fileparts(mfilename('fullpath')), 'saved_tubes.csv'));
methods = {'P3', 'Huan', 'Xiangru', 'Flow* native'};
states = {'x1', 'x2'};
colors = [0 114 178; 213 94 0; 204 121 167; 0 158 115] / 255;
figure('Color', 'w');
for d = 1:2
    subplot(2,2,d); hold on; grid on;
    if d == 1
        patch([0 1 1 0], [0 0 1 1], [0.87 0.93 0.87], ...
              'FaceAlpha', .5, 'EdgeColor', 'none');
        xline(.5, ':', 'Color', [.5 .5 .5]);
    end
    for m = 1:4
        R = T(strcmp(T.method, methods{m}) & strcmp(T.state, states{d}), :);
        assert(height(R) == 100 && all(R.step == (1:100)'));
        style = '-'; if m == 3, style = '--'; end
        stairs([0; R.t_end], [R.tube_lo; R.tube_lo(end)], ...
               'Color', colors(m,:), 'LineStyle', style, 'DisplayName', methods{m});
        stairs([0; R.t_end], [R.tube_hi; R.tube_hi(end)], ...
               'Color', colors(m,:), 'LineStyle', style, 'HandleVisibility', 'off');
    end
    xlim([0 1]); xlabel('time (s)'); ylabel(states{d});
    title([states{d} ' saved whole-step tube']);
    if d == 1, legend('Location', 'best'); end
    subplot(2,2,d+2); hold on; grid on;
    H = T(strcmp(T.method, 'Huan') & strcmp(T.state, states{d}), :);
    for m = 1:4
        R = T(strcmp(T.method, methods{m}) & strcmp(T.state, states{d}), :);
        stairs(R.t_end, R.endpoint_lo-H.endpoint_lo, 'Color', colors(m,:));
        stairs(R.t_end, R.endpoint_hi-H.endpoint_hi, ...
               'Color', colors(m,:), 'LineStyle', '--');
    end
    xlim([.8 1]); xlabel('time (s)'); ylabel(['delta ' states{d}]);
    title([states{d} ' endpoint minus Huan (lower solid / upper dashed)']);
end
sgtitle('Single Pendulum, named two-physical-state profile');
""", encoding="utf-8")


def main():
    data = {method: read_native(run) if method == "Flow* native" else read_gpu(run)
            for method, run in RUNS.items()}
    for bounds in data.values():
        validate(bounds)
    assert np.array_equal(data["Huan"], data["Xiangru"])
    OUT.mkdir(exist_ok=True)
    write_tables(data)
    draw(data)
    write_matlab()
    print(f"{OUT}: 4 methods × 100 saved steps × 2 physical states")


if __name__ == "__main__":
    main()
