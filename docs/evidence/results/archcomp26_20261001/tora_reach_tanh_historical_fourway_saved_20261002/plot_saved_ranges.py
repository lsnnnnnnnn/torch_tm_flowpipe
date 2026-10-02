#!/usr/bin/env python3
"""Audit and draw archived TORA reach-tanh ranges; never invoke a solver."""

import csv
import gzip
import json
import math
from pathlib import Path
import struct


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
ARCHIVE = REPO.parents[2] / "results" / "archcomp_review_20260923"
ROW = struct.Struct("<QQd16d")
STEPS = 500
H = 0.01
TARGET = {"x1": (-0.1, 0.2), "x2": (-0.9, -0.6)}
PREFIX = HERE / "tora_reach_tanh_u11_historical_fourway_saved"
SOURCES = (
    ("Huan", "huan", "tora_relu_tanh_huan", "#d95f02", "-"),
    ("Xiangru", "xiangru", "tora_relu_tanh_xiangru", "#7570b3", ":"),
    ("Historical P3", "old_p3", "tora_relu_tanh_ours", "#1b9e77", "-."),
    ("Flow* native", "native", "tora_relu_tanh", "#1f77b4", "--"),
)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def source_path(name):
    return HERE / "raw" / f"{name}_ranges.bin"


def read_ranges(path):
    binary = path.read_bytes()
    if len(binary) != STEPS * ROW.size:
        raise ValueError(f"{path}: expected {STEPS} complete four-state records")
    records = []
    for expected, unpacked in enumerate(ROW.iter_unpack(binary), 1):
        lane, step, h, *values = unpacked
        if lane != 0 or step != expected or not math.isclose(h, H, rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"{path}: invalid lane/step/time grid at record {expected}")
        states = []
        for state in range(4):
            tube_lo, tube_hi, end_lo, end_hi = values[4 * state:4 * state + 4]
            if (not all(map(math.isfinite, (tube_lo, tube_hi, end_lo, end_hi)))
                    or not tube_lo <= end_lo <= end_hi <= tube_hi):
                raise ValueError(f"{path}: invalid saved bounds at step {expected}, x{state + 1}")
            states.append({"tube": [tube_lo, tube_hi], "endpoint": [end_lo, end_hi]})
        records.append(states)
    return records


def audit_receipts():
    receipts = {}
    gpu_root = ARCHIVE / "evidence_v1" / "suite_v1"
    for label, name, folder, _, _ in SOURCES[:3]:
        path = gpu_root / folder / "result.json"
        result = read_json(path)
        if (result.get("status") != "completed" or result.get("complete") is not True
                or result.get("returncode") != 0
                or any(result.get(key) != STEPS for key in ("attempted_steps", "accepted_lane_steps", "expected_steps"))
                or len(result.get("steps", [])) != STEPS
                or any(row.get("step") != index or row.get("accepted") != [True] or row.get("status") != [0]
                       for index, row in enumerate(result["steps"], 1))
                or result.get("end_to_end_strict_certificate") is not False):
            raise ValueError(f"{name}: archived result does not show an accepted full numerical horizon")
        engine = result.get("engine", {}).get("root", "")
        if name == "old_p3" and not engine.endswith("engine_linear_leaf_v2"):
            raise ValueError("Historical P3 engine identity changed")
        receipts[name] = {
            "path": str(path), "status": result["status"], "complete": result["complete"],
            "returncode": result["returncode"], "attempted_steps": result["attempted_steps"],
            "accepted_lane_steps": result["accepted_lane_steps"], "engine_root": engine,
            "property_label": "no property verdict in this result",
        }
    path = ARCHIVE / "evidence_v2" / "native_matched" / "tora_relu_tanh" / "result.json"
    result = read_json(path)
    log_path = path.with_name("native.log")
    log_lines = log_path.read_text(encoding="utf-8").splitlines()
    expected_lines = [f"Step {i}" for i in range(10)] + ["VERIFIED"]
    if (result.get("status") != "completed" or result.get("complete") is not True
            or result.get("returncode") != 0 or result.get("control_periods_started") != 10
            or result.get("range_records") != STEPS or result.get("ranges_bytes") != STEPS * ROW.size
            or log_lines[:11] != expected_lines
            or result.get("end_to_end_strict_certificate") is not False):
        raise ValueError("Native result/log does not show a full numerical horizon and author checker label")
    receipts["native"] = {
        "path": str(path), "native_log": str(log_path), "status": result["status"],
        "complete": result["complete"], "returncode": result["returncode"],
        "control_periods_started": 10, "range_records": STEPS,
        "property_label": "author endpoint checker printed VERIFIED",
    }
    return receipts


def compare_archived_trajectory(ranges):
    """Compare every archived interval with the previously saved numeric CSV."""
    path = ARCHIVE / "evidence_v2" / "width_trajectory_tora_relu_tanh_matched.csv.gz"
    arms = {"huan": "huan", "xiangru": "xiangru", "ours": "old_p3"}
    count = 0
    seen = set()
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            arm = row["arm"]
            if arm not in arms:
                raise ValueError(f"Unexpected archive arm {arm}")
            step = int(row["step"])
            state = int(row["coordinate"]) - 1
            view = row["view"]
            if not 1 <= step <= STEPS or not 0 <= state < 4 or view not in ("tube", "endpoint"):
                raise ValueError("Unexpected archive trajectory grid")
            key = (arm, step, state, view)
            if key in seen:
                raise ValueError(f"Duplicate archive trajectory row: {key}")
            seen.add(key)
            if not math.isclose(float(row["time"]), step * H, rel_tol=0, abs_tol=1e-12):
                raise ValueError("Archive trajectory time mismatch")
            expected = ranges[arms[arm]][step - 1][state][view]
            native = ranges["native"][step - 1][state][view]
            for field, value in (("union_lo", expected[0]), ("union_hi", expected[1]),
                                 ("native_union_lo", native[0]), ("native_union_hi", native[1])):
                if not math.isclose(float(row[field]), value, rel_tol=0, abs_tol=1e-12):
                    raise ValueError(f"Archive numeric interval mismatch: {arm}, {step}, {state}, {view}, {field}")
            count += 1
    if count != 3 * 4 * 2 * STEPS:
        raise ValueError(f"Archive trajectory has {count} rows")
    return {"path": str(path), "compared_rows": count,
            "compared_interval_bounds": count * 4, "maximum_allowed_difference": 1e-12}


def write_endpoint_csv(ranges):
    path = HERE / "terminal_fourway_absolute_T5.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("method", "state", "lower", "upper", "absolute_width", "saved_range_source"))
        for label, name, _, _, _ in SOURCES:
            for state, bounds in enumerate(ranges[name][-1], 1):
                lo, hi = bounds["endpoint"]
                writer.writerow((label, f"x{state}", repr(lo), repr(hi), repr(hi - lo),
                                 str(source_path(name).relative_to(REPO))))
    return path


def geometry(ranges):
    series = []
    for label, name, _, color, style in SOURCES:
        final = ranges[name][-1]
        series.append({
            "label": label, "source": str(source_path(name).relative_to(REPO)),
            "color": color, "line_style": style,
            "tube_x1": [item[0]["tube"] for item in ranges[name]],
            "tube_x2": [item[1]["tube"] for item in ranges[name]],
            "endpoint_T5_x1": final[0]["endpoint"],
            "endpoint_T5_x2": final[1]["endpoint"],
        })
    return {"contract": "official ReLU^3/tanh, u=11f, four-state TORA, full initial box",
            "step_s": H, "steps": STEPS, "target_at_T5_x1_x2": TARGET, "series": series}


def write_matlab():
    path = PREFIX.with_suffix(".m")
    lines = [
        "% Historical TORA reach-tanh saved ranges; drawing only, no solver invocation.",
        "% MATLAB script reads the adjacent geometry JSON. It has not been executed here.",
        "base = fileparts(mfilename('fullpath'));",
        "g = jsondecode(fileread(fullfile(base, 'tora_reach_tanh_u11_historical_fourway_saved.geometry.json')));",
        "t = 0:g.step_s:g.steps*g.step_s; figure('Color','w','Position',[100 100 1300 760]);",
        "for dim = 1:2",
        "  subplot(2,2,2*dim-1); hold on; grid on;",
        "  if dim == 1, key = 'tube_x1'; target = g.target_at_T5_x1_x2.x1; else, key = 'tube_x2'; target = g.target_at_T5_x1_x2.x2; end",
        "  plot([5 5],target,'m-','LineWidth',5,'DisplayName','T=5 target');",
        "  for j = 1:numel(g.series)",
        "    s = g.series(j); b = s.(key); c = sscanf(s.color(2:end),'%2x%2x%2x')'/255;",
        "    stairs(t,[b(:,1);b(end,1)],'Color',c,'LineStyle',s.line_style,'LineWidth',1.6,'DisplayName',s.label);",
        "    stairs(t,[b(:,2);b(end,2)],'Color',c,'LineStyle',s.line_style,'LineWidth',1.6,'HandleVisibility','off');",
        "  end",
        "  xlabel('t (s)'); ylabel(sprintf('x%d',dim)); xlim([0 5]); title(sprintf('Saved whole-step tube: x%d',dim));",
        "  if dim == 1, legend('Location','best'); end",
        "  subplot(2,2,2*dim); hold on; grid on;",
        "  if dim == 1, endpoint_key = 'endpoint_T5_x1'; else, endpoint_key = 'endpoint_T5_x2'; end",
        "  ref = g.series(1).(endpoint_key);",
        "  for j = 1:numel(g.series)",
        "    s = g.series(j); e = s.(endpoint_key); c = sscanf(s.color(2:end),'%2x%2x%2x')'/255;",
        "    plot((e(1)-ref(1))*1e5,j,'o','Color',c,'MarkerFaceColor',c,'MarkerSize',7);",
        "    plot((e(2)-ref(2))*1e5,j,'^','Color',c,'MarkerFaceColor',c,'MarkerSize',7);",
        "  end",
        "  xline(0,'k:'); yticks(1:numel(g.series)); yticklabels({g.series.label}); ylim([0.5 numel(g.series)+0.5]);",
        "  xlabel('Endpoint bound offset vs Huan (10^{-5})'); title(sprintf('T=5 x%d bounds: circle lower, triangle upper',dim));",
        "end",
        "sgtitle('Historical TORA reach-tanh, official u=11f: saved numeric tubes');",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


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
            ax.step(edges, lower + lower[-1:], where="post", color=item["color"],
                    linestyle=item["line_style"], linewidth=1.5, label=item["label"])
            ax.step(edges, upper + upper[-1:], where="post", color=item["color"],
                    linestyle=item["line_style"], linewidth=1.5)
        ax.set(xlim=(0, 5.01), xlabel="t (s)", ylabel=rf"${key}$",
               title=rf"Saved whole-step ${key}$ tube (500 × 0.01 s)")
        ax.grid(alpha=0.18)
        if dim == 0:
            ax.legend(loc="upper left", ncol=3, fontsize=8)
        ref = g["series"][0][f"endpoint_T5_{key}"]
        for row, item in enumerate(g["series"]):
            terminal = item[f"endpoint_T5_{key}"]
            for value, baseline, marker in zip(terminal, ref, ("o", "^")):
                offset.plot((value - baseline) * 1e5, row, marker=marker,
                            color=item["color"], markersize=7)
        offset.axvline(0, color="black", linestyle=":", linewidth=1)
        offset.set(yticks=range(4), yticklabels=[s["label"] for s in g["series"]],
                   ylim=(3.45, -0.45), xlabel=r"Endpoint bound offset vs Huan ($10^{-5}$)",
                   title=rf"$T=5$ ${key}$: ○ lower, △ upper")
        offset.grid(axis="x", alpha=0.2)
    fig.suptitle("Historical TORA reach-tanh · official ReLU³/tanh, $u=11f$ · full initial box",
                 fontsize=14, y=0.975)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.90, bottom=0.19,
                        hspace=0.38, wspace=0.28)
    fig.text(0.08, 0.075,
             "Four archived full numeric T=5 histories; Huan and Xiangru saved tube bounds coincide. Magenta marks target only at T=5.\n"
             "All saved endpoints lie in target. GPU result receipts have no property verdict; native author endpoint checker: VERIFIED.\n"
             "Historical P3 used engine_linear_leaf_v2. Axis-aligned boxes do not establish an independent end-to-end NNCS certificate.",
             fontsize=8, va="bottom")
    fig.savefig(PREFIX.with_suffix(".png"), dpi=180)
    fig.savefig(PREFIX.with_suffix(".pdf"))
    plt.close(fig)


def main():
    ranges = {name: read_ranges(source_path(name)) for _, name, _, _, _ in SOURCES}
    receipts = audit_receipts()
    trajectory = compare_archived_trajectory(ranges)
    identical = all(ranges["huan"][i] == ranges["xiangru"][i] for i in range(STEPS))
    if not identical:
        raise ValueError("Expected archived Huan and Xiangru numeric bounds to coincide")
    inclusion = {}
    for label, name, _, _, _ in SOURCES:
        last = ranges[name][-1]
        inclusion[name] = all(TARGET[f"x{i}"][0] <= last[i-1]["endpoint"][0]
                              <= last[i-1]["endpoint"][1] <= TARGET[f"x{i}"][1]
                              for i in (1, 2))
    if not all(inclusion.values()):
        raise ValueError("An archived saved endpoint is outside the stated T=5 target")
    write_endpoint_csv(ranges)
    g = geometry(ranges)
    PREFIX.with_suffix(".geometry.json").write_text(
        json.dumps(g, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    (HERE / "AUDIT.json").write_text(json.dumps({
        "raw_records_per_method": STEPS, "raw_record_bytes": ROW.size,
        "all_record_grids_valid": True, "all_intervals_finite_ordered_and_endpoints_within_same_step_tube": True,
        "huan_xiangru_all_saved_bounds_identical": identical,
        "all_four_saved_T5_x1_x2_endpoints_inside_target": inclusion,
        "archived_trajectory_comparison": trajectory, "result_receipts": receipts,
        "provenance_limit": "Copied raw ranges and adjacent result receipts are compared numerically; no content binding is claimed.",
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_matlab()
    render(g)
    print("Saved historical four-method TORA reach-tanh artifacts to", HERE)


if __name__ == "__main__":
    main()
