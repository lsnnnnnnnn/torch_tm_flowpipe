"""Redraw the small set of current review figures from normalized derived tables."""
from __future__ import annotations

import csv
import gzip
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "research-review-v1"
import matplotlib.pyplot as plt
import numpy as np

PALETTE = {"cpu":"#156c8a", "gpu-range":"#bd5a16", "flowstar":"#3b8f68"}
LABELS = {"cpu":"CPU reference", "gpu-range":"GPU range in solver", "flowstar":"native Flow* object"}
PLANTS = {"van_der_pol":"Van der Pol", "brusselator":"Brusselator"}
CHANNELS = (("endpoint","x"),("endpoint","y"),("tube","x"),("tube","y"))


def rows(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def save(fig, path, description):
    fig.tight_layout(rect=(0, .06, 1, .96))
    fig.text(.01, .01, description, fontsize=7, color="#333333")
    fig.savefig(path, format="svg", metadata={"Description":description, "Date":None})
    plt.close(fig)
    # Matplotlib emits many SVG path lines with a cosmetic trailing blank.
    # Normalize those so checked-in figures have a stable text representation.
    path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines())+"\n",
                    encoding="utf-8")


def bounds_figures(curves, out, provenance):
    for plant, short in (("van_der_pol","vdp"),("brusselator","brusselator")):
        for coordinate in ("x","y"):
            fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
            for axis, view in zip(axes, ("endpoint","tube")):
                for lane in ("flowstar","cpu","gpu-range"):
                    chosen = [r for r in curves if r["plant"]==plant and r["view"]==view
                              and r["coordinate"]==coordinate and r["lane"]==lane]
                    t = np.array([float(r["time"]) for r in chosen])
                    lo = np.array([float.fromhex(r["lo_hex"]) for r in chosen])
                    hi = np.array([float.fromhex(r["hi_hex"]) for r in chosen])
                    axis.fill_between(t, lo, hi, color=PALETTE[lane], alpha=.12)
                    axis.plot(t, lo, color=PALETTE[lane], lw=.8, label=LABELS[lane]+" lower")
                    axis.plot(t, hi, color=PALETTE[lane], lw=.8, linestyle="--",
                              label=LABELS[lane]+" upper")
                axis.set_ylabel(f"{view} {coordinate} enclosure")
                axis.grid(alpha=.2)
            axes[-1].set_xlabel("time (sum of actual binary64 steps)")
            axes[0].legend(ncol=3, fontsize=7, loc="best")
            fig.suptitle(f"{PLANTS[plant]} full B1 fixed horizon: {coordinate} bounds, 1000 accepted steps")
            save(fig, out / f"{short}_{coordinate}_bounds.svg",
                 "Source: results/review/bounds/curves.csv.gz; provenance.json identifies each immutable run. "
                 "CPU/Flow* complete saved models were re-observed with the current strict CPU range path; "
                 "GPU bounds come from each accepted step. No whole-solver formal proof.")


def ratio_figures(ratios, out):
    for plant, short in (("van_der_pol","vdp"),("brusselator","brusselator")):
        fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        for axis, denominator in zip(axes, ("flowstar","cpu")):
            data = []
            for view, coordinate in CHANNELS:
                chosen = [r for r in ratios if r["plant"]==plant and r["numerator"]=="gpu-range"
                          and r["denominator"]==denominator and r["view"]==view
                          and r["coordinate"]==coordinate and r["width_ratio"]!=""]
                t = [float(r["time"]) for r in chosen]
                value = [float(r["width_ratio"]) for r in chosen]
                data.extend(value)
                axis.plot(t, value, lw=1, label=f"{view}-{coordinate}")
            if data and min(data)>0 and max(data)/min(data)>50:
                axis.set_yscale("log")
                axis.set_ylabel("width ratio (log)")
            else:
                axis.set_ylabel("width ratio")
            axis.axhline(1, color="#444444", lw=.7)
            axis.set_title(f"GPU-range width / {LABELS[denominator]} width")
            axis.legend(ncol=4, fontsize=8)
            axis.grid(alpha=.2)
        axes[-1].set_xlabel("time (actual binary64 step sum)")
        fig.suptitle(f"{PLANTS[plant]} four-channel width relationships, complete B1 horizon")
        save(fig, out / f"{short}_width_ratios.svg",
             "Source: results/review/bounds/width_ratios.csv.gz. Zero/near-zero denominators are omitted "
             "from ratios and retained as absolute differences in the CSV; ratios do not prove containment.")


def mechanism_figure(data, out):
    vdp = [r for r in data if r["plant"]=="van_der_pol" and r["horizon"]=="T6p32"]
    bruss = [r for r in data if r["plant"]=="brusselator"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    x = np.arange(2)
    for j, lane in enumerate(("c2","c3","flowstar")):
        record = next(r for r in vdp if r["lane"]==lane)
        axes[0].bar(x + (j-1)*.22,
                    [float(record["endpoint_x_width"]),float(record["endpoint_y_width"])],
                    width=.21,label={"c2":"legacy","c3":"linear history","flowstar":"Flow*"}[lane])
    axes[0].set_xticks(x,["endpoint-x","endpoint-y"])
    axes[0].set_ylabel("width at 632-step frozen boundary")
    axes[0].set_title("VDP: cross-step history (fixed historical contract)")
    axes[0].legend(fontsize=8)
    lanes=("legacy","C4 refined","stock Flow*")
    accepted=[int(next(r for r in bruss if r["lane"]==lane)["accepted"]) for lane in lanes]
    axes[1].bar(lanes, accepted, color=["#a34d47","#156c8a","#3b8f68"])
    axes[1].set_ylabel("accepted fixed steps")
    axes[1].set_ylim(0,1100)
    axes[1].set_title("Brusselator: acceptance-after refinement (SR1000)")
    axes[1].grid(axis="y",alpha=.2)
    fig.suptitle("Two distinct historical mechanism tests; no cross-contract version ranking")
    save(fig, out / "mechanism_comparisons.svg",
         "Sources: artifacts/runs/vdp_c3_cross_step_causal_closure_20260827/RESULT.json; "
         "brusselator_sr1000_c4_closure_20260828/CLOSURE_RESULT.json; "
         "brusselator_live_range_c5_20260828/RESULT.json. Older arithmetic versions have scoped limitations.")


def cpu_figure(data, out):
    fig, axes=plt.subplots(1,2,figsize=(11,4.5))
    for axis, optimization in zip(axes,("prepared remainder replay",
                                        "ordered tensor range after prepared replay")):
        chosen=[r for r in data if r["optimization"]==optimization]
        x=np.arange(len(chosen))
        baseline=[float(r["baseline_s"]) for r in chosen]
        candidate=[float(r["candidate_s"]) for r in chosen]
        axis.bar(x-.18,baseline,width=.35,label="paired denominator",color="#8a8a8a")
        axis.bar(x+.18,candidate,width=.35,label="candidate",color="#156c8a")
        axis.set_xticks(x,[r["plant"] for r in chosen])
        axis.set_ylabel("solve seconds (one complete pair)")
        axis.set_title(optimization)
        axis.grid(axis="y",alpha=.2)
        axis.legend(fontsize=8)
    fig.suptitle("CPU paired execution time; second denominator already includes first optimization")
    save(fig,out/"cpu_pairings.svg",
         "Source: results/review/cpu_pairings.csv, recomputed from the two frozen RESULT.json files. "
         "Each complete pair has n=1; no compounded stable speedup is claimed.")


def gpu_figure(data,out):
    fig,axes=plt.subplots(1,2,figsize=(11,5.5))
    for axis,field,title in zip(axes,("local_speedup","online_speedup"),
                                ("local operator/request ratio","complete online prefix ratio")):
        chosen=[r for r in data if r[field]!=""]
        labels=[("VDP" if r["plant"]=="van_der_pol" else "Bruss")+" / "+
                r["layer"].split(" ")[0] for r in chosen]
        values=[float(r[field]) for r in chosen]
        axis.barh(labels,values,color=["#bd5a16" if r["plant"]=="brusselator" else "#156c8a"
                                       for r in chosen])
        axis.axvline(1,color="#333333",lw=.7)
        axis.set_xlabel("paired ratio; >1 favors GPU route")
        axis.set_title(title)
        axis.grid(axis="x",alpha=.2)
    fig.suptitle("CUDA route: local acceleration versus actual online effect")
    save(fig,out/"gpu_local_vs_online.svg",
         "Source: results/review/gpu_route.csv and registered immutable RESULT.json files. "
         "Offline batch, online service, packet, and resident observations use different contracts. "
         "Resident full B1 T10/T20 and a whole GPU engine were not run.")


def matched_figure(data,out):
    fig,axis=plt.subplots(figsize=(9,4.5))
    x=np.arange(len(data))
    flowstar=[float(r["flowstar_process_wall_s"]) for r in data]
    resident=[float(r["candidate_median_wall_s"]) for r in data]
    axis.bar(x-.18,flowstar,width=.35,label="Flow* one invocation",color="#3b8f68")
    axis.bar(x+.18,resident,width=.35,label="resident 5-run median",color="#bd5a16")
    axis.set_xticks(x,["Van der Pol","Brusselator"])
    axis.set_yscale("log")
    axis.set_ylabel("wall seconds (log; all observed times positive)")
    axis.legend()
    axis.grid(axis="y",alpha=.2)
    fig.suptitle("Matched B32 × 20-step workload: engineering wall-time scale")
    save(fig,out/"matched_flowstar_scale.svg",
         "Source: artifacts/runs/resident_tm_block_20260914T032650Z/matched_flowstar.csv. "
         "32 distinct 8x4 boxes; Flow* n=1 per plant, resident n=5. Main parameters and CPU resource match; "
         "internal validation/history algorithms differ.")


def plot_all(out):
    out=Path(out).resolve()
    results=out.parent
    needed=[results/"bounds/curves.csv.gz",results/"bounds/width_ratios.csv.gz",
            results/"mechanisms.csv",results/"cpu_pairings.csv",results/"gpu_route.csv",
            results/"matched_flowstar.csv",results/"provenance.json"]
    missing=[str(path) for path in needed if not path.is_file()]
    if missing:
        raise FileNotFoundError("run summarize for this output root first; missing: "+", ".join(missing))
    out.mkdir(parents=True,exist_ok=True)
    curves=rows(needed[0]); ratios=rows(needed[1])
    if len(curves)!=24000 or len(ratios)!=24000:
        raise ValueError(f"unexpected complete curve/ratio rows: {len(curves)}, {len(ratios)}")
    bounds_figures(curves,out,needed[-1])
    ratio_figures(ratios,out)
    mechanism_figure(rows(needed[2]),out)
    cpu_figure(rows(needed[3]),out)
    gpu_figure(rows(needed[4]),out)
    matched_figure(rows(needed[5]),out)
    print(f"redrew 10 SVG figures in {out}",flush=True)
