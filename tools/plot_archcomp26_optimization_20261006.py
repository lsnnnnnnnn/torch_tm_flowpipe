#!/usr/bin/env python3
"""Rebuild October 6 selection charts from saved report data only.

No solver, checker, MATLAB, JIT or content digest. PDFs are vector ReportLab
drawings with a literal document ID; Poppler renders their matching PNGs.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess


def prohibited(*args, **kwargs):
    raise RuntimeError("content digest operation prohibited")


for name in (*hashlib.algorithms_guaranteed, "new", "file_digest"):
    if hasattr(hashlib, name):
        setattr(hashlib, name, prohibited)

from reportlab.graphics import renderPDF
from reportlab.graphics.shapes import Drawing, Line, PolyLine, Rect, String
from reportlab.lib import colors
from reportlab.pdfbase import pdfdoc


class NoDocumentSignature:
    """ReportLab's unused signature accumulator, with no hashing machinery."""
    def update(self, value):
        pass


# PDFDocument constructs this accumulator even when its ID is supplied.
# No update is processed, and the ID method never requests a digest.
pdfdoc.md5 = lambda *a, **kw: NoDocumentSignature()
pdfdoc.PDFDocument.ID = lambda self: b"\n[<30363036323032363030303030303030><30363036323032363030303030303030>]\n"

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "research/p3_speed_tightness_20261006"
FORMAL_DATA = ROOT / "docs/evidence/results/archcomp26_report_20261006"
OLD_WIDTHS = ROOT / "docs/evidence/results/archcomp26_report_20261005/widths"
METHODS = ("pytorch_gpu", "huan", "xiangru", "flowstar_native")
METHOD_LABEL = dict(zip(METHODS, ("Selected P3", "Huan", "Xiangru", "Native")))
PALETTE = dict(zip(METHODS, ("#0072B2", "#E69F00", "#CC79A7", "#009E73")))
LABELS = {"attitude-control-avoid": "Attitude", "docking-constraint": "Docking",
          "single-pendulum-reach": "Single Pendulum (two-state)", "nav-robust": "NAV robust",
          "nav-standard": "NAV standard", "tora-reach-sigmoid": "TORA sigmoid",
          "tora-reach-tanh": "TORA tanh", "tora-remain": "TORA remain",
          "double-pendulum-less-robust": "Double Pendulum less",
          "acc-safe-distance": "ACC", "unicycle-reach": "Unicycle", "quad-reach": "QUAD"}
BLACK = colors.HexColor("#17232D")
MUTED = colors.HexColor("#53626D")
GRID = colors.HexColor("#DAE1E6")


def read(path):
    return json.loads(path.read_text())


def rel(path):
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def text(drawing, x, y, value, size=10, bold=False, anchor="start", color=BLACK):
    drawing.add(String(x, y, value, fontName="Helvetica-Bold" if bold else "Helvetica",
                       fontSize=size, textAnchor=anchor, fillColor=color))


def nice_max(value):
    if value <= 0:
        return 1.0
    unit = 10**math.floor(math.log10(value))
    return math.ceil(value/unit*5)/5*unit


def selected_speed(data, selections):
    timing = read(data/"timing_index.json")
    rows = read(data/"selection_audit.json")
    if {r["benchmark"]: r["run_id"] for r in rows} != {k:v["run_id"] for k,v in selections.items()}:
        raise ValueError("selection_audit and selection.json disagree; rebuild report data first")
    result = []
    for row in rows:
        old, new = row["old_timings"]["driver_elapsed_s"], row["new_timings"]["driver_elapsed_s"]
        if any(not isinstance(v,(int,float)) or not math.isfinite(v) or v <= 0 for v in (old,new)):
            raise ValueError("every selected timing needs a positive driver duration")
        previous = timing["runs"][row["previous_selected_run"]]
        current = timing["runs"][row["new_selected_run"]]
        if current["horizon"]["complete_named_horizon"] is not True:
            raise ValueError("speed chart cannot display an incomplete candidate")
        if previous["timings"]["driver_elapsed_s"] != old or current["timings"]["driver_elapsed_s"] != new:
            raise ValueError("driver fields disagree with selection audit")
        result.append(dict(benchmark=row["benchmark"], previous_s=old, selected_s=new,
            change_percent=100*(new-old)/old, previous_run=row["previous_selected_run"],
            selected_run=row["new_selected_run"], width_policy=row["width_policy"],
            previous_source=previous["timing_sources"]["driver_elapsed_s"],
            selected_source=current["timing_sources"]["driver_elapsed_s"],
            previous_raw_receipts=previous.get("raw_receipts",[]),
            selected_raw_receipts=current.get("raw_receipts",[])))
    return sorted(result,key=lambda r: LABELS.get(r["benchmark"],r["benchmark"]))


def speed_drawing(rows):
    count = math.ceil(len(rows)/2)
    d = Drawing(900,150+count*140)
    text(d,450,d.height-28,"Current selection vs previous selection",17,True,"middle")
    text(d,450,d.height-49,"Internal driver time (seconds) | all October 6 selected complete runs",11,anchor="middle")
    for i,row in enumerate(rows):
        x,y = 35+(i%2)*440,d.height-92-(i//2)*140
        text(d,x,y,LABELS.get(row["benchmark"],row["benchmark"]),12,True)
        delta = row["change_percent"]
        change = f"{abs(delta):.1f}% {'more' if delta > 0 else 'less'} time" if delta else "Unchanged time"
        text(d,x+390,y,change,10,anchor="end")
        xmax = nice_max(max(row["previous_s"],row["selected_s"])*1.18)
        for j,(label,field,color) in enumerate((("Previous","previous_s","#8997A2"),("Selected","selected_s","#0072B2"))):
            cy = y-32-j*31
            text(d,x,cy+6,label,10)
            width = row[field]/xmax*245
            d.add(Rect(x+63,cy,width,20,strokeWidth=0,fillColor=colors.HexColor(color)))
            text(d,x+70+width,cy+5,f"{row[field]:.6f} s",9)
        note = "New saved geometry" if row["width_policy"] == "new_saved_ranges" else "Saved geometry unchanged"
        text(d,x+63,y-84,note,9,color=MUTED)
    text(d,450,35,"Each pair has its own scale. Exact recorded seconds are retained in FIGURE_SOURCES.json.",10,anchor="middle")
    text(d,450,18,"One sample per new run; width gains can cost time. Shared-host timings do not establish a stable speed ranking.",10,anchor="middle")
    return d


def width_series(data, selections, summary, sources):
    instances = ("tora-reach-sigmoid","tora-reach-tanh")
    series = {}
    base = data/"widths_long.csv"
    # Newer report-data builds provide a merged table. Earlier builds provide
    # an old table plus explicit selected numerical replacements.
    files = [base if base.is_file() else OLD_WIDTHS/"widths_long.csv"]
    replace = {k:v["run_id"] for k,v in selections.items()
               if k in instances and v["width_policy"] == "new_saved_ranges"}
    if not base.is_file():
        files.extend(data/(run+"_widths_long.csv") for run in replace.values())
    for number,path in enumerate(files):
        with path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                instance,method = row["instance_id"],row["method"]
                if instance not in instances or method not in METHODS:
                    continue
                if number == 0 and not base.is_file() and instance in replace and method == "pytorch_gpu":
                    continue
                if row["geometry"] not in ("endpoint","tube") or row["state"] not in ("x1","x2","x3","x4"):
                    continue
                if row["complete_initial_set"] != "True" or int(row["available_lanes"]) != int(row["expected_lanes"]):
                    raise ValueError("TORA selected plots require the complete initial set")
                lo,hi,w = (float(row[k]) for k in ("lo","hi","absolute_width"))
                if not all(math.isfinite(v) for v in (lo,hi,w)) or hi < lo or w != hi-lo:
                    raise ValueError("saved absolute width is inconsistent")
                key = instance,method,row["state"],row["geometry"]
                series.setdefault(key,[]).append(dict(step=int(row["step"]),t=float(row["t_end"]),width=w,
                    source_id=row["source_id"],source_locator=row["source_locator"],saved_object=row["saved_object"],
                    table=rel(path)))
    mapping = {(r["instance_id"],r["method"],r["state"]):r for r in summary}
    for instance in instances:
        for method in METHODS:
            for state in ("x1","x2","x3","x4"):
                for geometry in ("endpoint","tube"):
                    rows = series.get((instance,method,state,geometry),[])
                    rows.sort(key=lambda r:r["step"])
                    if [r["step"] for r in rows] != list(range(1,501)) or rows[-1]["t"] != 5:
                        raise ValueError(f"missing/duplicate complete 500-step series: {instance}/{method}/{state}/{geometry}")
                    sm = mapping[instance,method,state]
                    observed = rows[-1]["width"] if geometry == "endpoint" else max(r["width"] for r in rows)
                    expected = sm["common_endpoint_width"] if geometry == "endpoint" else sm["common_max_tube_width"]
                    if sm["common_time"] != 5 or observed != expected:
                        raise ValueError("selected full trajectory disagrees with current summary")
                    if any(r["source_id"] not in sources for r in rows):
                        raise ValueError("source mapping missing")
    return series


def width_drawing(instance,geometry,series):
    d = Drawing(900,780)
    name = LABELS[instance]
    phrase = "Endpoint absolute width" if geometry == "endpoint" else "Per-step tube absolute width"
    text(d,450,753,f"{name}: {phrase}",17,True,"middle")
    text(d,450,732,"Selected P3, Huan, Xiangru and native | complete 500-step horizon | state axes remain separate",10,anchor="middle")
    for i,method in enumerate(METHODS):
        x = 210+i*142
        d.add(Line(x,708,x+23,708,strokeColor=colors.HexColor(PALETTE[method]),strokeWidth=2,
                   strokeDashArray=[4,2] if method == "xiangru" else []))
        text(d,x+29,704,METHOD_LABEL[method],10)
    for i,state in enumerate(("x1","x2","x3","x4")):
        x,y = 64+(i%2)*438,462-(i//2)*330
        w,h = 360,188
        ymax = nice_max(max(r["width"] for method in METHODS for r in series[instance,method,state,geometry])*1.06)
        text(d,x,y+h+28,state+" | absolute width",12,True)
        for tick in range(5):
            cy = y+tick*h/4
            d.add(Line(x,cy,x+w,cy,strokeColor=GRID,strokeWidth=.5))
            text(d,x-8,cy-3,f"{ymax*tick/4:.4g}",9,anchor="end",color=MUTED)
        for tick in range(6):
            cx = x+tick*w/5
            d.add(Line(cx,y,cx,y-4,strokeColor=MUTED,strokeWidth=.6))
            text(d,cx,y-17,str(tick),9,anchor="middle",color=MUTED)
        d.add(Line(x,y,x,y+h,strokeColor=MUTED,strokeWidth=.7))
        d.add(Line(x,y,x+w,y,strokeColor=MUTED,strokeWidth=.7))
        for method in ("flowstar_native","huan","xiangru","pytorch_gpu"):
            points = [v for r in series[instance,method,state,geometry] for v in (x+r["t"]/5*w,y+r["width"]/ymax*h)]
            d.add(PolyLine(points,strokeColor=colors.HexColor(PALETTE[method]),
                strokeWidth=1.8 if method == "pytorch_gpu" else 1.2,
                strokeDashArray=[4,2] if method == "xiangru" else [],fillColor=None))
        text(d,x+w/2,y-33,"Step end time (s)",9,anchor="middle",color=MUTED)
        values = []
        for method in METHODS:
            points = series[instance,method,state,geometry]
            value = points[-1]["width"] if geometry == "endpoint" else max(r["width"] for r in points)
            values.append(f"{METHOD_LABEL[method]} {value:.6g}")
        text(d,x,y-51,("At 5 s: " if geometry == "endpoint" else "Max step: ")+" | ".join(values[:2]),8.5)
        text(d,x,y-65," | ".join(values[2:]),8.5)
    text(d,450,17,"Tube width is each step's upper minus lower, not the width of the all-time union. Width is not a soundness proof.",9,anchor="middle")
    return d


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,default=FORMAL_DATA,
                        help="Formal report root (timing/ and widths/), or a flat rebuilt report-data folder")
    parser.add_argument("--selection",type=Path,default=STUDY/"report_data/selection.json")
    parser.add_argument("--output",type=Path,default=ROOT/"docs/evidence/results/archcomp26_report_20261006/figures")
    parser.add_argument("--pdftoppm",type=Path)
    args = parser.parse_args()
    selections = read(args.selection)
    timing_data = args.data/"timing" if (args.data/"timing").is_dir() else args.data
    width_data = args.data/"widths" if (args.data/"widths").is_dir() else args.data
    speeds = selected_speed(timing_data,selections)
    summary = read(width_data/"summary.json")
    sources = {r["source_id"]:r for r in read(width_data/"sources.json")}
    series = width_series(width_data,selections,summary,sources)
    renderer = args.pdftoppm or shutil.which("pdftoppm") or Path("/Users/shengenli/.cache/codex-runtimes/codex-primary-runtime/dependencies/bin/override/pdftoppm")
    if not Path(renderer).is_file():
        raise FileNotFoundError("pdftoppm is required to render the exact produced PDFs")
    args.output.mkdir(parents=True,exist_ok=True)
    drawings = {"driver_selected_before_after":speed_drawing(speeds)}
    for instance in ("tora-reach-sigmoid","tora-reach-tanh"):
        for geometry in ("endpoint","tube"):
            drawings[instance.replace("tora-reach-","tora_")+f"_{geometry}_widths"] = width_drawing(instance,geometry,series)
    artifacts = []
    for name,drawing in drawings.items():
        pdf = args.output/(name+".pdf")
        renderPDF.drawToFile(drawing,str(pdf))
        subprocess.run([str(renderer),"-png","-singlefile","-r","165",str(pdf),str(args.output/name)],check=True,capture_output=True)
        artifacts.append(dict(pdf=rel(pdf),pdf_bytes=pdf.stat().st_size,
                              png=rel(args.output/(name+".png")),png_bytes=(args.output/(name+".png")).stat().st_size))
    plotted = []
    for key,rows in series.items():
        chosen = selections.get(key[0]) if key[1] == "pytorch_gpu" else None
        new_raw = STUDY/"results"/chosen["run_id"]/"candidate/data/ranges.bin" if chosen else None
        if new_raw is not None and not new_raw.is_file():
            raise FileNotFoundError("selected TORA candidate raw geometry must remain available")
        plotted.append(dict(instance=key[0],method=key[1],state=key[2],geometry=key[3],steps=len(rows),
            source_tables=sorted({r["table"] for r in rows}),saved_objects=sorted({r["saved_object"] for r in rows}),
            source_ids=sorted({r["source_id"] for r in rows}),
            raw_sources=[sources[k] for k in sorted({r["source_id"] for r in rows})],
            selected_p3=chosen,
            new_raw_geometry=dict(path=rel(new_raw),bytes=new_raw.stat().st_size) if new_raw else None))
    receipt = dict(selection_file=rel(args.selection),selection=selections,data_directory=rel(args.data),
        speed_rows=speeds,width_series=plotted,artifacts=artifacts,
        no_numerical_experiments=True,old_checker_executed=False,digest_operations=0,
        pdf_id="fixed literal; PDFDocument signature accumulator disabled without any digest",
        qualification="descriptive single samples; state units kept separate; saved widths are not independent NNCS certificates")
    (args.output/"FIGURE_SOURCES.json").write_text(json.dumps(receipt,indent=2,ensure_ascii=False,allow_nan=False)+"\n")
    print(json.dumps(dict(figures=len(artifacts),speed_pairs=len(speeds),width_series=len(plotted),output=str(args.output))))


if __name__ == "__main__":
    main()
