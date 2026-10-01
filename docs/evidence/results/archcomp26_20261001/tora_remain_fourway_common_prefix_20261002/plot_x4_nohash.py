#!/usr/bin/env python3
"""Draw saved 2026 TORA remain t–x4 tube unions; no solver or digest."""

import csv
import json
from pathlib import Path

import scan_saved_ranges as saved


HERE = Path(__file__).resolve().parent
OUT = HERE / "plots"
PREFIX = OUT / "tora_remain_2026_fourway_t_x4_saved_tube"
LABELS = {"native": "Flow* native", "huan": "Huan", "xiangru": "Xiangru",
          "ours_p3": "ours/P3"}
COLORS = {"native": "#1f77b4", "huan": "#d95f02", "xiangru": "#7570b3",
          "ours_p3": "#1b9e77"}
STEPS = {"native": 200, "huan": 184, "xiangru": 184, "ours_p3": 200}


def descriptor(path):
    return {"path": str(path.resolve()), "bytes": path.stat().st_size}


def x4_by_step(records, end_step):
    result = []
    for step in range(1, end_step + 1):
        rows = [records[lane, step][12:16] for lane in range(12)]
        result.append([min(row[0] for row in rows), max(row[1] for row in rows)])
    return result


def build_geometry():
    with (HERE / "RANGES.csv").open(newline="", encoding="utf-8") as handle:
        rows = {(row["method"], row["window"]): row for row in csv.DictReader(handle)
                if row["physical_state"] == "x4"}
    if len(rows) != 8:
        raise ValueError("expected x4 common-prefix and full-horizon rows for four methods")
    series = []
    for method, run_dir in saved.SOURCES.items():
        records = saved.load(method, run_dir)
        end_step = STEPS[method]
        saved.scan(records, end_step)  # full saved-box safety only for native/P3
        bounds = x4_by_step(records, end_step)
        common = rows[method, "common_prefix"]
        common_bounds = x4_by_step(records, 184)
        assert common["status"] == "all_12_accepted_and_saved_tubes_safe"
        assert [min(x[0] for x in common_bounds), max(x[1] for x in common_bounds)] == [
            float(common["tube_lo"]), float(common["tube_hi"])]
        full = rows[method, "full_horizon"]
        if end_step == 200:
            assert full["status"] == "all_12_accepted_and_saved_tubes_safe"
            assert [min(x[0] for x in bounds), max(x[1] for x in bounds)] == [
                float(full["tube_lo"]), float(full["tube_hi"])]
        else:
            assert full["status"] == "UNKNOWN_no_full_initial_set_result"
            assert not full["tube_lo"] and not full["endpoint_lo"]
            audit = saved.read_json(run_dir / "INDEPENDENT_INTERVAL_SCAN.json")
            assert audit["first_saved_tube_outside_safe_substep"] == 185
            assert audit["first_rejected_substep"] == 190
        raw = run_dir / ("ranges.bin" if method == "native" else "payload/ranges.bin")
        result = run_dir / ("RESULT.json" if method == "native" else "payload/RESULT.json")
        config = run_dir / ("START.json" if method == "native" else "payload/START.json")
        observed = None if method == "native" else run_dir / "payload/observations.jsonl"
        outcome = saved.read_json(result)
        assert outcome["status"] == ("incomplete" if end_step == 184 else "completed")
        if method == "ours_p3":
            assert outcome["accepted_lane_substeps"] == 2400
            assert outcome["all_lanes_accepted_and_saved_tubes_safe_prefix_substeps"] == 200
        series.append({
            "method": method, "label": LABELS[method], "color": COLORS[method],
            "range_source": descriptor(raw), "run_configuration": descriptor(config),
            "adjacent_run_result_unbound": descriptor(result),
            "observations": descriptor(observed) if observed else None,
            "qualified_substeps": end_step, "qualified_until_s": end_step / 10,
            "tube_x4_union_per_step": bounds,
            "qualified_tube_x4_union": [min(x[0] for x in bounds), max(x[1] for x in bounds)],
            "status": ("record-complete; acceptance absent from native ranges.bin; adjacent checker VERIFIED"
                       if method == "native" else
                       "2400/2400 accepted and saved boxes safe; no explicit final checker line"
                       if method == "ours_p3" else
                       "first 184 steps: all 12 accepted and saved boxes safe; later UNKNOWN"),
        })
    assert series[1]["tube_x4_union_per_step"] == series[2]["tube_x4_union_per_step"]
    return {
        "schema": "archcomp26-tora-remain-t-x4-saved-tube-nohash-v1",
        "contract": "2026 TORA remain, 12 initial boxes, T=20, h=0.1",
        "view": "per-step union of 12 saved whole-step axis-aligned x4 tubes",
        "initial_x4_interval": [0.5, 0.6], "all_time_safe_x4_band": [-2, 2],
        "full_horizon_s": 20, "common_qualified_prefix_s": 18.4,
        "huan_xiangru_later_status": {
            "first_saved_tube_outside_safe_step": 185,
            "first_saved_tube_outside_safe_interval_s": [18.4, 18.5],
            "first_rejected_step": 190,
            "first_rejected_interval_s": [18.9, 19.0],
            "accepted_lane_substeps": 2357, "expected_lane_substeps": 2400,
            "author_checker": "Unknown.",
            "full_initial_set_T20_endpoint": None,
        },
        "source_comparison": descriptor(HERE / "RANGES.csv"),
        "series": series,
        "qualification": "saved box projection only; adjacent outcomes unbound; no independent end-to-end floating-point NNCS certificate",
    }


def matlab_vector(values):
    return "[" + " ".join(format(value, ".17g") for value in values) + "]"


def write_matlab(geometry):
    lines = ["% 2026 TORA remain: saved t-x4 whole-step box unions only.",
             "% Huan/Xiangru stop at t=18.4; no T=20 continuation is plotted.",
             "% Source paths and status are in adjacent geometry JSON. MATLAB not run by generator.",
             "figure('Color','w'); hold on; grid on;",
             "patch([0 20 20 0], [-2 -2 2 2], [0.86 0.94 0.87], 'FaceAlpha', 0.3, 'EdgeColor', 'none');",
             "plot([0 20], [2 2], 'g--', 'LineWidth', 1);",
             "plot([0 20], [-2 -2], 'g--', 'LineWidth', 1);"]
    for index, item in enumerate(geometry["series"], 1):
        bounds = item["tube_x4_union_per_step"]
        edges = [step / 10 for step in range(len(bounds) + 1)]
        rgb = [int(item["color"][k:k + 2], 16) / 255 for k in (1, 3, 5)]
        style = ":" if item["method"] == "xiangru" else "-"
        lines += [f"t{index} = {matlab_vector(edges)};",
                  f"lo{index} = {matlab_vector([x[0] for x in bounds])};",
                  f"hi{index} = {matlab_vector([x[1] for x in bounds])};",
                  f"c{index} = {matlab_vector(rgb)};",
                  f"for k = 1:{len(bounds)}",
                  f"  patch([t{index}(k) t{index}(k+1) t{index}(k+1) t{index}(k)], [lo{index}(k) lo{index}(k) hi{index}(k) hi{index}(k)], c{index}, 'FaceAlpha', 0.05, 'EdgeColor', 'none');",
                  "end",
                  f"stairs(t{index}, [lo{index} lo{index}(end)], 'Color', c{index}, 'LineStyle', '{style}', 'LineWidth', 1.2, 'DisplayName', '{item['label']}');",
                  f"stairs(t{index}, [hi{index} hi{index}(end)], 'Color', c{index}, 'LineStyle', '{style}', 'LineWidth', 1.2, 'HandleVisibility', 'off');"]
    lines += ["plot([18.4 18.4], [-2.2 2.2], 'k:', 'LineWidth', 1.3, 'DisplayName', 'H/X qualified end');",
              "plot([0 0], [0.5 0.6], 'ko-', 'LineWidth', 2, 'DisplayName', 'Initial x4');",
              "xlabel('t (s)'); ylabel('x_4'); xlim([0 20]); ylim([-2.2 2.2]);",
              "title('2026 TORA remain: 12-box saved whole-step x_4 tube');",
              "legend('Location','southwest'); hold off;"]
    PREFIX.with_suffix(".m").write_text("\n".join(lines) + "\n", encoding="utf-8")


def render(geometry):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12.2, 6.7))
    ax.axhspan(-2, 2, color="#d9eedc", alpha=0.5, label="All-time Safe x4 band")
    ax.axhline(-2, color="#367d47", linestyle="--", linewidth=1)
    ax.axhline(2, color="#367d47", linestyle="--", linewidth=1)
    ax.axvspan(18.4, 20, color="#999999", alpha=0.12)
    for item in geometry["series"]:
        bounds = item["tube_x4_union_per_step"]
        edges = [step / 10 for step in range(len(bounds) + 1)]
        lo, hi = [x[0] for x in bounds], [x[1] for x in bounds]
        color = item["color"]
        ax.fill_between(edges, lo + lo[-1:], hi + hi[-1:], step="post",
                        color=color, alpha=0.045)
        style = ":" if item["method"] == "xiangru" else "-"
        ax.step(edges, lo + lo[-1:], where="post", color=color,
                linestyle=style, linewidth=1.25, label=item["label"])
        ax.step(edges, hi + hi[-1:], where="post", color=color,
                linestyle=style, linewidth=1.25)
    ax.axvline(18.4, color="#333333", linestyle=":", linewidth=1.3,
               label="Huan/Xiangru qualified end")
    ax.plot([0, 0], [0.5, 0.6], "ko-", linewidth=2, label="Initial x4 box")
    ax.text(18.55, -2.10, "H/X\nUNKNOWN", fontsize=8, color="#444444", va="bottom")
    ax.set(xlim=(0, 20), ylim=(-2.2, 2.2), xlabel="t (s)", ylabel=r"$x_4$",
           title="2026 TORA remain — saved 12-box whole-step tube projection")
    ax.grid(alpha=0.18)
    ax.legend(loc="lower left", fontsize=8, ncol=2)
    fig.subplots_adjust(left=0.09, right=0.97, top=0.90, bottom=0.24)
    fig.text(0.09, 0.075,
             "Native/P3: 12 boxes × 200 steps to T=20. Huan/Xiangru: qualified all-12 accepted/safe tube only through t=18.4; plotted bounds coincide.\n"
             "For Huan/Xiangru, step 185 first saved tube out of Safe; step 190 first rejection; checker Unknown. No qualified T=20 endpoint.\n"
             "Axis-aligned saved tube unions; native ranges lack accepted field. Adjacent outcomes are unbound; no end-to-end NNCS certificate.",
             fontsize=7.8, va="bottom")
    fig.savefig(PREFIX.with_suffix(".png"), dpi=180)
    fig.savefig(PREFIX.with_suffix(".pdf"))
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    geometry = build_geometry()
    PREFIX.with_suffix(".geometry.json").write_text(
        json.dumps(geometry, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8")
    write_matlab(geometry)
    render(geometry)
    print(PREFIX)


if __name__ == "__main__":
    main()
