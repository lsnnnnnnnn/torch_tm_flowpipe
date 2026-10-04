"""Render two frozen four-method prefixes; never solve, hash, or write MATLAB."""
from contextlib import ExitStack
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import struct
from unittest.mock import patch

BASE = Path(__file__).resolve().parent.parent
H = 0.005
COLORS = {"P3": "#16846d", "Huan": "#db731b", "Xiangru": "#7658a4", "Native": "#2773ad"}
STYLES = {"P3": "-", "Huan": "-", "Xiangru": (0, (5, 3)), "Native": (0, (2, 2))}
SOURCE_BYTES = {}


def read(path):
    path = BASE / path
    raw = path.read_bytes()
    SOURCE_BYTES[path] = raw
    return raw


def frame(step, lanes, names):
    stats = []
    outside, gap = 0, 0.0
    for state, name in enumerate(names):
        bounds = [lane[state] for lane in lanes]
        assert all(len(item) == 4 and all(map(math.isfinite, item))
                   and item[0] <= item[1] and item[2] <= item[3] for item in bounds)
        for lo, hi, elo, ehi in bounds:
            outside += int(elo < lo) + int(ehi > hi)
            gap = max(gap, lo - elo, ehi - hi)
        lo, hi = min(item[0] for item in bounds), max(item[1] for item in bounds)
        elo, ehi = min(item[2] for item in bounds), max(item[3] for item in bounds)
        tw, ew = [item[1] - item[0] for item in bounds], [item[3] - item[2] for item in bounds]
        stats.append(dict(state=name, tube_lo=lo, tube_hi=hi, tube_union_width=hi-lo,
                          endpoint_lo=elo, endpoint_hi=ehi, endpoint_union_width=ehi-elo,
                          tube_per_box_width_mean=math.fsum(tw)/len(tw),
                          tube_per_box_width_max=max(tw),
                          endpoint_per_box_width_mean=math.fsum(ew)/len(ew),
                          endpoint_per_box_width_max=max(ew)))
    return dict(step=step, t_start_s=(step-1)*H, t_end_s=step*H, lanes=len(lanes),
                states=stats, endpoint_outside_tube_components=outside,
                endpoint_outside_tube_max=gap)


def binary(path, names, expected_lanes, stored_states):
    record = struct.Struct(f"<QQd{4*stored_states}d")
    raw = read(path)
    assert len(raw) % record.size == 0
    grouped = {}
    for values in record.iter_unpack(raw):
        lane, step, h, *bounds = values
        assert h == H and 0 <= lane < expected_lanes and step >= 1
        assert all(math.isfinite(value) for value in bounds)
        states = [bounds[4*i:4*i+4] for i in range(len(names))]
        group = grouped.setdefault(step, {})
        assert lane not in group
        group[lane] = states
    assert sorted(grouped) == list(range(1, max(grouped)+1))
    frames = []
    for step, values in sorted(grouped.items()):
        assert sorted(values) == list(range(expected_lanes))
        frames.append(frame(step, [values[i] for i in range(expected_lanes)], names))
    return frames, None


def jsonl(path, names, expected_lanes):
    frames, rejected = [], None
    for step, line in enumerate(read(path).splitlines(), 1):
        row = json.loads(line)
        assert row["substep"] == step and rejected is None
        if "t_interval" in row:
            assert all(math.isclose(a, b, rel_tol=0, abs_tol=1e-12)
                       for a, b in zip(row["t_interval"], [(step-1)*H, step*H]))
        if row["accepted"] is False:
            assert row.get("tube") is None and row.get("endpoint") is None
            rejected = step
            continue
        if expected_lanes == 1:
            assert row["accepted"] is True
            tubes, endpoints = [row["tube"]], [row["endpoint"]]
        else:
            assert row["accepted"] == [True]*expected_lanes
            assert row["status"] == [0]*expected_lanes
            assert row["tube_endpoint_valid"] == [True]*expected_lanes
            tubes, endpoints = row["tube"], row["endpoint"]
        assert len(tubes) == len(endpoints) == expected_lanes
        lanes = [[tubes[lane][i]+endpoints[lane][i] for i in range(len(names))]
                 for lane in range(expected_lanes)]
        frames.append(frame(step, lanes, names))
    return frames, rejected


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def render(data, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    balancing = data["benchmark"] == "Balancing raw4"
    fig = plt.figure(figsize=(13.2, 9.6))
    grid = fig.add_gridspec(3, 2, height_ratios=[1, 1, .28])
    axes = [fig.add_subplot(grid[i//2, i%2]) for i in range(4)]
    common = data["common_numerical_steps"]
    end = common*H
    for i, ax in enumerate(axes):
        if not balancing:
            ax.axhspan(-1.5, 1.5, color="#9cc5a2", alpha=.18)
            ax.axvspan(end, .4, facecolor="#eeeeee", alpha=.7)
            ax.axvline(.3, color="#555555", lw=.8, ls=":")
        for method, item in data["series"].items():
            selected = item["frames"][:common]
            edges = [0]+[row["t_end_s"] for row in selected]
            lows = [row["states"][i]["tube_lo"] for row in selected]
            highs = [row["states"][i]["tube_hi"] for row in selected]
            ax.fill_between(edges, lows+lows[-1:], highs+highs[-1:],
                            step="post", color=COLORS[method], alpha=.04)
            ax.stairs(lows, edges, baseline=None, color=COLORS[method],
                      linestyle=STYLES[method], linewidth=1.5, label=method)
            ax.stairs(highs, edges, baseline=None, color=COLORS[method],
                      linestyle=STYLES[method], linewidth=1.5)
        ax.plot([0, 0], data["initial_box"][i], "k-", linewidth=3, label="Initial box")
        ax.set(xlim=(0, end if balancing else .4), xlabel="t (s)", ylabel=data["state_names"][i])
        ax.grid(alpha=.18)
        if not balancing:
            ax.set_ylim(-1.65, 1.65)
    handles, labels = axes[0].get_legend_handles_labels()
    if not balancing:
        handles.append(Patch(facecolor="#9cc5a2", alpha=.3))
        labels.append("Safe band for all t in [0, 0.4]")
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .94),
               ncol=6 if not balancing else 5, frameon=False, fontsize=10)
    timeline = fig.add_subplot(grid[2, :])
    for y, (method, item) in enumerate(data["series"].items()):
        timeline.plot([0, len(item["frames"])*H], [y, y], color=COLORS[method], lw=4)
        timeline.plot(len(item["frames"])*H, y, "|", color=COLORS[method], ms=9)
    timeline.set(yticks=range(4), yticklabels=list(data["series"]), xlim=(0, data["horizon_s"]),
                 xlabel="Requested full horizon (s); blank after each saved prefix")
    timeline.invert_yaxis()
    if balancing:
        timeline.axvspan(8, 10, color="#b2188b", alpha=.12)
        timeline.text(9, 1.5, "Target window 8 < t <= 10\nx1, x3, x4 in [-0.001, 0.001]\nNo saved data in this window",
                      ha="center", va="center", fontsize=8)
    timeline.grid(axis="x", alpha=.18)
    fig.suptitle(data["title"], fontsize=14, y=.985)
    fig.subplots_adjust(left=.075, right=.98, top=.87, bottom=.16, hspace=.39, wspace=.2)
    note = ("Common numerical prefix: 83 steps to 0.415 s. The future target is not a safety band on this prefix.\n"
            "Raw four-state controller profile; the paper's five-feature controller remains unresolved. All four full-horizon attempts stopped early."
            if balancing else
            "All panels draw the common numerical prefix: 64 steps to 0.32 s, 225 boxes per step. Common saved Safe prefix ends at 0.30 s (dotted).\n"
            "Steps 61-64 retain the saved band crossings; gray region has no four-way common data. Huan/Xiangru/P3 save through 0.36 s; Native through 0.32 s.")
    fig.text(.075, .035, note+"\nWhole-step axis-aligned box unions; no interpolation, independent NNCS certificate, trajectory counterexample, or full-horizon timing claim.",
             fontsize=8.3, va="bottom")
    for suffix in (".png", ".pdf"):
        fig.savefig(output / (data["stem"]+suffix), dpi=190)
    plt.close(fig)


def produce(config, output_root=None):
    relative = Path(config.pop("directory"))
    output = output_root / relative.parts[0] if output_root else BASE / relative
    output.mkdir(parents=True, exist_ok=False)
    files = config.pop("files")
    methods = {}
    for method, (path, kind, expected_steps) in files.items():
        frames, rejected = (binary(path, config["state_names"], config["lanes"], kind)
                            if isinstance(kind, int) else jsonl(path, config["state_names"], config["lanes"]))
        assert len(frames) == expected_steps
        methods[method] = dict(source_path=str(BASE/path), source_bytes=len(SOURCE_BYTES[BASE/path]),
                               frames=frames, rejected_step=rejected,
                               unobserved_from_step=len(frames)+1,
                               planned_final_endpoint=None,
                               endpoint_outside_tube_components=sum(row["endpoint_outside_tube_components"] for row in frames),
                               endpoint_outside_tube_max=max(row["endpoint_outside_tube_max"] for row in frames))
    assert min(len(item["frames"]) for item in methods.values()) == config["common_numerical_steps"]
    config.update(series=methods, geometry_kind="pooled axis-aligned saved whole-step tubes and endpoints",
                  content_digest_policy="none computed", missing_policy="no interpolation; absent planned rows stay blank",
                  property_verdict="no independent end-to-end NNCS certificate")
    if config["benchmark"] == "DP more":
        for method, item in methods.items():
            crossing = [row["step"] for row in item["frames"]
                        if any(s["tube_lo"] < -1.5 or s["tube_hi"] > 1.5 for s in row["states"])]
            assert crossing[0] == 61
            item["first_saved_tube_band_crossing"] = crossing[0]
        for method, suffix in (("Native", "native_dp_more_full20_001/COMMON60_SCAN_NOHASH.json"),
                               ("Huan", "author_dp_more_v1/huan_full20_001/COMMON60_SCAN.json"),
                               ("Xiangru", "author_dp_more_v1/xiangru_full20_001/COMMON60_SCAN.json")):
            prior = json.loads(read(suffix))
            states = methods[method]["frames"][59]["states"]
            assert [[s["endpoint_lo"],s["endpoint_hi"]] for s in states] == prior["terminal_endpoint_union"]
    fields = list(next(iter(methods.values()))["frames"][0]["states"][0])
    rows, widths = [], []
    for method, item in methods.items():
        for step in range(1, config["planned_steps"]+1):
            present = step <= len(item["frames"])
            for i, state in enumerate(config["state_names"]):
                values = item["frames"][step-1]["states"][i] if present else {k: "" for k in fields}
                rows.append(dict(method=method, step=step, t_start_s=(step-1)*H, t_end_s=step*H,
                                 recorded=present, in_common_display=present and step<=config["common_numerical_steps"],
                                 status="saved" if present else "rejected" if step==item["rejected_step"] else "unobserved",
                                 **{**values,"state":state}))
        scopes = [("common_numerical_endpoint", config["common_numerical_steps"]),
                  ("last_saved_endpoint",len(item["frames"])), ("requested_final_endpoint",None)]
        if config["benchmark"] == "DP more":
            scopes.insert(1,("common_saved_safe_endpoint",60))
        for scope, step in scopes:
            for i, state in enumerate(config["state_names"]):
                values = item["frames"][step-1]["states"][i] if step else {k:"" for k in fields}
                prefix = [row["states"][i] for row in item["frames"][:step]] if step else []
                widths.append(dict(method=method, scope=scope, step=step or "", t_s=step*H if step else config["horizon_s"],
                                   **{**values,"state":state},
                                   prefix_tube_lo=min(s["tube_lo"] for s in prefix) if prefix else "",
                                   prefix_tube_hi=max(s["tube_hi"] for s in prefix) if prefix else ""))
    (output/"geometry.json").write_text(json.dumps(config, indent=2, allow_nan=False)+"\n")
    write_csv(output/"saved_bounds.csv",rows)
    write_csv(output/"absolute_widths.csv",widths)
    render(config,output)
    audit = dict(benchmark=config["benchmark"],common_numerical_steps=config["common_numerical_steps"],
                 recorded_steps={m:len(v["frames"]) for m,v in methods.items()},
                 saved_bounds_rows=len(rows),absolute_width_rows=len(widths),
                 interval_repairs=0,matlab_files=0,solver_runs=0,
                 endpoint_tube_anomalies={m:{k:v[k] for k in ("endpoint_outside_tube_components","endpoint_outside_tube_max")} for m,v in methods.items()})
    (output/"AUDIT.json").write_text(json.dumps(audit,indent=2)+"\n")
    print(json.dumps(audit))
    return output


def main(output_root=None):
    bal = dict(directory="balancing_raw4_fourway_saved_20261004_001/output",
               benchmark="Balancing raw4", stem="balancing_raw4_fourway_common_prefix",
               title="Balancing fixed repository raw4  Four methods on their common saved prefix",
               state_names=["x1","x2","x3","x4"],initial_box=[[-.1,.1],[-.05,.05],[-.1,.1],[-.05,.05]],
               horizon_s=10,planned_steps=2000,lanes=1,common_numerical_steps=83,
               property={"role":"target","states":["x1","x3","x4"],"bounds":[-.001,.001],"time":"8 < t <= 10", "paper_profile_window":"[8,10]; separate controller identity"},
               files={"P3":("balancing_fixed_raw4_p3_full500_001/payload/ranges.jsonl","jsonl",86),
                      "Huan":("balancing_fixed_raw4_huan/balancing_fixed_raw4_huan_full500_001/ranges.jsonl","jsonl",98),
                      "Xiangru":("balancing_fixed_raw4_xiangru_20261002/balancing_fixed_raw4_xiangru_full500_001/ranges.jsonl","jsonl",98),
                      "Native":("native_balancing_raw4_20261002/native_balancing_raw4_full500_001/ranges.bin",5,83)})
    dp = dict(directory="dp_more_fourway_saved_20261004_001",benchmark="DP more",stem="dp_more_fourway_common_prefix",
              title="Double Pendulum more robust  Four methods on their common saved numerical prefix",
              state_names=["theta1","theta2","theta1_dot","theta2_dot"],initial_box=[[1,1.3]]*4,
              horizon_s=.4,planned_steps=80,lanes=225,common_numerical_steps=64,common_saved_safe_steps=60,
              property={"role":"safe","states":"all four physical states","bounds":[-1.5,1.5],"time":"all t in [0,0.4]"},
              files={"P3":("dp_more_p3_affine_split4_full20_20261003_001/attempt/data/observations.jsonl","jsonl",72),
                     "Huan":("author_dp_more_v1/huan_full20_001/payload/ranges.bin",4,72),
                     "Xiangru":("author_dp_more_v1/xiangru_full20_001/payload/ranges.bin",4,72),
                     "Native":("native_dp_more_full20_001/ranges.bin",4,64)})
    outputs = [produce(config, output_root) for config in (bal,dp)]
    for path, before in SOURCE_BYTES.items():
        assert path.read_bytes() == before
    for output in outputs:
        assert not list(output.glob("*.m"))
        (output/"SOURCE_BYTE_COMPARISON.json").write_text(json.dumps({"policy":"direct byte equality; no digest", "sources":[{"path":str(p),"bytes":len(b),"unchanged_after":True} for p,b in SOURCE_BYTES.items()]},indent=2)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, help="new root for both replay outputs")
    args = parser.parse_args()
    def forbidden(*_args, **_kwargs):
        raise AssertionError("content digest computation prohibited")
    with ExitStack() as guard:
        for name in (*hashlib.algorithms_guaranteed,"new","file_digest"):
            guard.enter_context(patch.object(hashlib,name,forbidden))
        main(args.output_root)
