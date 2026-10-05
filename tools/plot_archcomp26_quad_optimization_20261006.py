#!/usr/bin/env python3
"""Plot an explicitly named complete QUAD candidate against saved references.

Reads saved pooled observers only. Huan/Xiangru have terminal markers, never
invented trajectories or tubes. No solver, old checker or digest is executed.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import re
import shutil
import subprocess

# Reuse the audited no-digest ReportLab setup and drawing primitives.
from plot_archcomp26_optimization_20261006 import (
    ROOT, STUDY, OLD_WIDTHS, BLACK, MUTED, GRID, colors, Drawing, Line,
    PolyLine, read, rel, renderPDF, text, nice_max,
)
from reportlab.graphics.shapes import Circle, Polygon, Rect

STATES = tuple(f"x{i}" for i in range(1, 13))
CURVES = ("old_p3", "new_p3", "flowstar_native")
COLORS = dict(old_p3="#73808C", new_p3="#0072B2", flowstar_native="#009E73",
              huan="#E69F00", xiangru="#CC79A7")
LABELS = dict(old_p3="Old P3", new_p3="New P3", flowstar_native="Native",
              huan="Huan", xiangru="Xiangru")
REFERENCE_CONFIG = ROOT / "docs/evidence/results/archcomp26_20261001/quad_paper_p3_nohash_v1/full50_001/data/config.yaml"


def contract(candidate_config):
    """Read the fixed published YAML form without importing any old runner."""
    def initial(path):
        saved = path.read_text()
        intervals = {name: [float(lo), float(hi)] for name,lo,hi in
                     re.findall(r"- name: (x\d+)\n  interval:\n  - ([^\n]+)\n  - ([^\n]+)", saved)}
        if set(intervals) != set(STATES):
            raise ValueError("configuration must retain the twelve named physical initial intervals")
        for field, expected in (("steps", 50), ("step_size", .1), ("ode_step_size", .005)):
            value = re.search(r"^"+field+r": ([^\n]+)$", saved, re.MULTILINE)
            if value is None or float(value.group(1)) != expected:
                raise ValueError("plot requires the saved paper full50 time contract")
        return intervals
    old, new = initial(REFERENCE_CONFIG), initial(candidate_config)
    if old != new:
        raise ValueError("candidate and reference initial intervals differ")
    return dict(initial_intervals=new, initial_sources=[rel(REFERENCE_CONFIG), rel(candidate_config)],
        target=dict(state="x3", time_s=5, interval=[.94, 1.06], scope="endpoint only",
                    source="tools/archcomp26_plot_saved_nohash.py::_quad.target_endpoint_interval"))


def interval(lo, hi):
    lo, hi = float(lo), float(hi)
    if not all(math.isfinite(x) for x in (lo, hi)) or lo > hi:
        raise ValueError("nonfinite or reversed saved interval")
    return dict(lo=lo, hi=hi, width=hi-lo)


def old_series(directory):
    sources = {r["source_id"]: r for r in read(directory / "sources.json")}
    series, markers, used = {}, {}, set()
    with (directory / "widths_long.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["instance_id"] != "quad-reach":
                continue
            method, obj = row["method"], row["saved_object"]
            curve = obj == "pooled_observer" and method in ("pytorch_gpu", "flowstar_native")
            marker = obj == "driver_final_hull" and method in ("huan", "xiangru")
            if not (curve or marker):
                continue
            if row["complete_initial_set"] != "True" or int(row["available_lanes"]) != 1024:
                raise ValueError("old QUAD series must cover every initial lane")
            if row["state"] not in STATES or row["geometry"] not in ("endpoint", "tube"):
                raise ValueError("unexpected QUAD state/geometry")
            value = interval(row["lo"], row["hi"])
            if value["width"] != float(row["absolute_width"]):
                raise ValueError("old saved width disagrees with bounds")
            value.update(step=int(row["step"]), t=float(row["t_end"]),
                         source_id=row["source_id"], source_locator=row["source_locator"],
                         saved_object=obj)
            used.add(row["source_id"])
            if marker:
                if row["geometry"] != "endpoint" or value["step"] != 1000 or value["t"] != 5:
                    raise ValueError("author data must remain a single terminal endpoint marker")
                key = method, row["state"]
                if key in markers:
                    raise ValueError("duplicate author terminal marker")
                markers[key] = value
            else:
                name = "old_p3" if method == "pytorch_gpu" else method
                series.setdefault((name, row["state"], row["geometry"]), []).append(value)
    for method in ("old_p3", "flowstar_native"):
        for state in STATES:
            for geometry in ("endpoint", "tube"):
                rows = series.get((method, state, geometry), [])
                rows.sort(key=lambda r: r["step"])
                if [r["step"] for r in rows] != list(range(1, 1001)) or rows[-1]["t"] != 5:
                    raise ValueError("reference trajectory is incomplete or duplicated")
    if set(markers) != {(m, s) for m in ("huan", "xiangru") for s in STATES}:
        raise ValueError("24 terminal author markers are required")
    return series, markers, [sources[s] for s in sorted(used)]


def candidate_series(run, series):
    candidate = run / "candidate"
    outer, result, start = (read(p) for p in (run / "RESULT.json", candidate / "RESULT.json", candidate / "START.json"))
    if outer.get("exit_code") != 0 or outer.get("status") != "completed" or result.get("exit_code") != 0:
        raise ValueError("candidate must have successful saved outer and inner receipts")
    if result.get("completed_substeps") != 1000 or result.get("accepted_lane_substeps") != 1024000 or start.get("mode") != "full50":
        raise ValueError("candidate must complete all 1024 lanes and 1000 steps; no prefix extrapolation")
    observation = candidate / "data/observations.jsonl"
    rows = [json.loads(line) for line in observation.read_text().splitlines() if line.strip()]
    if [r["substep"] for r in rows] != list(range(1, 1001)):
        raise ValueError("candidate observer is incomplete or duplicated")
    for row in rows:
        if row["accepted_count"] != 1024 or row["status_counts"] != {"0": 1024} or row["rejected_lanes"]:
            raise ValueError("candidate row does not have full accepted coverage")
        bounds = row["tube_endpoint_union_12x4"]
        if len(bounds) != 12 or any(len(b) != 4 for b in bounds):
            raise ValueError("expected 12 physical states in observer order")
        for state, b in zip(STATES, bounds):
            for geometry, offset in (("tube", 0), ("endpoint", 2)):
                value = interval(b[offset], b[offset+1])
                # Time is taken from the contract-matched reference, never from
                # a second free-standing interpretation of the solver clock.
                reference = series["old_p3", state, geometry][row["substep"]-1]
                value.update(step=row["substep"], t=reference["t"],
                             source_locator=f"jsonl_line:{row['substep']}", saved_object="pooled_observer")
                series.setdefault(("new_p3", state, geometry), []).append(value)
    comparison_path = candidate / "WIDTH_COMPARISON.json"
    if not comparison_path.is_file():
        comparison_path = candidate / "SAVED_COMPARISON.json"
    comparison = read(comparison_path)
    if comparison.get("completed_substeps") != 1000 or comparison.get("width_rows") != 24000:
        raise ValueError("saved comparison must cover the complete 24,000-row geometry")
    # Direct comparison with the saved comparison CSV, not an old checker run.
    compared = set()
    with (candidate / "ABSOLUTE_WIDTHS.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            key = row["state"], row["geometry"], int(row["substep"])
            if key in compared:
                raise ValueError("duplicate candidate comparison row")
            compared.add(key)
            for method, prefix in (("old_p3", "reference"), ("new_p3", "candidate")):
                value = series[method, key[0], key[1]][key[2]-1]
                if (value["lo"], value["hi"]) != (float(row[prefix+"_lower"]), float(row[prefix+"_upper"])):
                    raise ValueError("saved comparison differs from the actual plotted source bounds")
    if len(compared) != 24000:
        raise ValueError("candidate comparison has missing rows")
    extra_paths = [candidate/name for name in ("JOINT_QUALIFICATION.json", "CONTROL_RESULT.json",
                                               "TRIG_POWER_RESULT.json", "TRIG_POWER_GPU_GATE.json")
                   if (candidate/name).is_file()]
    return dict(outer=outer, result=result, start=start, comparison=comparison,
                supplemental_receipts={p.name:read(p) for p in extra_paths},
                source_files=[dict(path=rel(p), bytes=p.stat().st_size) for p in
                    (run/"RESULT.json", candidate/"RESULT.json", candidate/"START.json", observation,
                     comparison_path, candidate/"ABSOLUTE_WIDTHS.csv", *extra_paths)])


def marker(d, x, y, method):
    color = colors.HexColor(COLORS[method])
    if method == "huan":
        d.add(Circle(x, y, 4.3, strokeColor=color, strokeWidth=1.7, fillColor=None))
    else:
        d.add(Polygon([x, y+4.2, x-4.2, y-3.2, x+4.2, y-3.2],
                      strokeColor=color, strokeWidth=1.3, fillColor=None))


def drawing(geometry, series, markers, run_id, selected):
    d = Drawing(1230, 1330)
    phrase = "Endpoint absolute width" if geometry == "endpoint" else "Per-step tube absolute width"
    text(d, 615, 1304, f"QUAD: {phrase}", 21, True, "middle")
    text(d, 615, 1280, "12 physical states | 1024 initial boxes | complete 1000-step horizon | separate state scales", 12, anchor="middle")
    text(d, 615, 1259, f"New P3: {run_id} ({'current selected run' if selected else 'explicit numerical candidate; not selected'})", 11, anchor="middle")
    for i, method in enumerate(CURVES):
        x = 265+i*155
        d.add(Line(x, 1237, x+23, 1237, strokeColor=colors.HexColor(COLORS[method]),
                   strokeWidth=2.1, strokeDashArray=[5, 3] if method == "old_p3" else []))
        text(d, x+29, 1233, LABELS[method], 12)
    if geometry == "endpoint":
        for i, method in enumerate(("huan", "xiangru")):
            x = 747+i*162
            marker(d, x, 1237, method)
            text(d, x+11, 1233, LABELS[method]+" at 5 s only", 11)
    else:
        text(d, 756, 1233, "Huan / Xiangru: no saved tube data", 11, color=MUTED)
    for i, state in enumerate(STATES):
        x, y, w, h = 77+(i % 3)*405, 998-(i//3)*291, 315, 173
        maximum = max(r["width"] for m in CURVES for r in series[m, state, geometry])
        if geometry == "endpoint":
            maximum = max(maximum, *(markers[m, state]["width"] for m in ("huan", "xiangru")))
        ymax = nice_max(maximum*1.08)
        text(d, x, y+h+20, state+" | absolute width", 14, True)
        # Scientific multiplier keeps x12's subnormal numerical residue legible.
        exponent = math.floor(math.log10(ymax)) if ymax else 0
        scale = 10.0**exponent if exponent < -2 or exponent > 3 else 1.0
        if scale != 1:
            text(d, x+w, y+h+20, f"scale 1e{exponent:+d}", 10, anchor="end", color=MUTED)
        for tick in range(5):
            cy = y+tick*h/4
            d.add(Line(x, cy, x+w, cy, strokeColor=GRID, strokeWidth=.6))
            text(d, x-8, cy-3, f"{(ymax/scale)*tick/4:.3g}", 10, anchor="end", color=MUTED)
        for tick in range(6):
            cx = x+tick*w/5
            d.add(Line(cx, y, cx, y-4, strokeColor=MUTED, strokeWidth=.7))
            text(d, cx, y-17, str(tick), 10, anchor="middle", color=MUTED)
        d.add(Line(x, y, x, y+h, strokeColor=MUTED, strokeWidth=.7))
        d.add(Line(x, y, x+w, y, strokeColor=MUTED, strokeWidth=.7))
        for method in ("flowstar_native", "old_p3", "new_p3"):
            points = [v for r in series[method, state, geometry]
                      for v in (x+r["t"]/5*w, y+r["width"]/ymax*h)]
            d.add(PolyLine(points, strokeColor=colors.HexColor(COLORS[method]),
                          strokeWidth=1.9 if method == "new_p3" else 1.3,
                          strokeDashArray=[5, 3] if method == "old_p3" else [], fillColor=None))
        if geometry == "endpoint":
            for method in ("huan", "xiangru"):
                marker(d, x+w, y+markers[method, state]["width"]/ymax*h, method)
        text(d, x+w/2, y-33, "Step end time (s)", 10, anchor="middle", color=MUTED)
        values = {m: (series[m, state, geometry][-1]["width"] if geometry == "endpoint"
                      else max(r["width"] for r in series[m, state, geometry])) for m in CURVES}
        text(d, x, y-51, f"{'At 5 s' if geometry == 'endpoint' else 'Max step'}: old {values['old_p3']:.5g} | new {values['new_p3']:.5g}", 10)
        detail = f"Native {values['flowstar_native']:.5g}"
        if geometry == "endpoint":
            detail += f" | H {markers['huan',state]['width']:.5g} | X {markers['xiangru',state]['width']:.5g}"
        text(d, x, y-67, detail, 10)
    text(d, 615, 31, "Curves use saved pooled observers. Author markers use driver final hulls; these saved objects remain distinct.", 11, anchor="middle")
    text(d, 615, 13, "Tube widths are per step, not all-time union widths. Narrower is not containment or an independent NNCS certificate.", 11, anchor="middle")
    return d


def axes(d, x, y, w, h, xlim, ylim, xlabel, ylabel):
    def px(value):
        return x+(value-xlim[0])/(xlim[1]-xlim[0])*w
    def py(value):
        return y+(value-ylim[0])/(ylim[1]-ylim[0])*h
    for i in range(6):
        xv, yv = xlim[0]+i*(xlim[1]-xlim[0])/5, ylim[0]+i*(ylim[1]-ylim[0])/5
        d.add(Line(px(xv), y, px(xv), y+h, strokeColor=GRID, strokeWidth=.5))
        d.add(Line(x, py(yv), x+w, py(yv), strokeColor=GRID, strokeWidth=.5))
        text(d, px(xv), y-18, f"{xv:.3g}", 10, anchor="middle", color=MUTED)
        text(d, x-8, py(yv)-3, f"{yv:.3g}", 10, anchor="end", color=MUTED)
    d.add(Rect(x, y, w, h, strokeColor=MUTED, strokeWidth=.7, fillColor=None))
    text(d, x+w/2, y-37, xlabel, 11, anchor="middle")
    text(d, x, y+h+14, ylabel, 11)
    return px, py


def padded(lo, hi, fraction=.06):
    pad = (hi-lo)*fraction if hi != lo else max(abs(lo)*fraction, .1)
    return lo-pad, hi+pad


def geometry_drawing(kind, series, markers, run_id, selected, declared):
    initial = declared["initial_intervals"]
    d = Drawing(1230, 730)
    title = "QUAD: time-height saved tube bounds" if kind == "time_x3" else "QUAD: x1-x2 pooled coordinate-box projection"
    text(d, 615, 702, title, 21, True, "middle")
    text(d, 615, 678, f"New P3: {run_id} ({'current selected run' if selected else 'explicit numerical candidate; not selected'})", 11, anchor="middle")
    for i, method in enumerate(CURVES):
        x = 210+i*147
        d.add(Line(x, 652, x+23, 652, strokeColor=colors.HexColor(COLORS[method]), strokeWidth=2,
                   strokeDashArray=[5, 3] if method == "old_p3" else []))
        text(d, x+29, 648, LABELS[method], 11)
    text(d, 667, 648, "Black: initial set | Orange / pink: Huan / Xiangru terminal only", 11)
    if kind == "time_x3":
        bounds = [r for m in CURVES for r in series[m, "x3", "tube"]]
        ylim = padded(min(initial["x3"][0], *(r["lo"] for r in bounds)),
                      max(1.06, *(r["hi"] for r in bounds)))
        px, py = axes(d, 83, 115, 1060, 477, (0, 5), ylim, "Time (s)", "x3")
        for method in ("flowstar_native", "old_p3", "new_p3"):
            rows = series[method, "x3", "tube"]
            color = colors.HexColor(COLORS[method])
            for side in ("lo", "hi"):
                points, previous = [], 0.
                for row in rows:
                    points += [px(previous), py(row[side]), px(row["t"]), py(row[side])]
                    previous = row["t"]
                d.add(PolyLine(points, strokeColor=color, strokeWidth=1.8 if method == "new_p3" else 1.2,
                              strokeDashArray=[5, 3] if method == "old_p3" else [], fillColor=None))
        # Endpoint-only target: a vertical interval at T=5, never a time band.
        target = declared["target"]["interval"]
        d.add(Line(px(5), py(target[0]), px(5), py(target[1]), strokeColor=colors.HexColor("#7B3294"), strokeWidth=6))
        for method in ("huan", "xiangru"):
            value = markers[method, "x3"]
            d.add(Line(px(5), py(value["lo"]), px(5), py(value["hi"]),
                       strokeColor=colors.HexColor(COLORS[method]), strokeWidth=2))
            for side in ("lo", "hi"):
                marker(d, px(5), py(value[side]), method)
        d.add(Line(px(0), py(initial["x3"][0]), px(0), py(initial["x3"][1]), strokeColor=BLACK, strokeWidth=3))
        for end in initial["x3"]:
            d.add(Circle(px(0), py(end), 3, strokeColor=BLACK, fillColor=colors.white))
        text(d, 615, 52, "Purple segment: x3 in [0.94, 1.06] at T=5 only. No all-time safe band is asserted.", 12, anchor="middle")
        text(d, 615, 31, "Each tube bound is constant on its saved substep. Curves cover all 1000 steps; Huan/Xiangru supply only final hulls.", 11, anchor="middle")
    else:
        def limits(state):
            vals = [r for m in CURVES for r in series[m, state, "tube"]]
            vals += [markers[m, state] for m in ("huan", "xiangru")]
            return padded(min(initial[state][0], *(r["lo"] for r in vals)),
                          max(initial[state][1], *(r["hi"] for r in vals)), .09)
        xlim, ylim = limits("x1"), limits("x2")
        # Equal numeric scale and identical limits in every method panel.
        xmid, ymid = sum(xlim)/2, sum(ylim)/2
        span = max(xlim[1]-xlim[0], ylim[1]-ylim[0])
        xlim, ylim = (xmid-span/2, xmid+span/2), (ymid-span/2, ymid+span/2)
        for i, method in enumerate(CURVES):
            x, y, size = 74+i*407, 178, 323
            text(d, x+size/2, 568, LABELS[method]+" | 1000 saved tubes", 14, True, "middle")
            px, py = axes(d, x, y, size, size, xlim, ylim, "x1", "x2")
            base = colors.HexColor(COLORS[method])
            faint = colors.Color(base.red, base.green, base.blue, alpha=.22)
            for rx, ry in zip(series[method, "x1", "tube"], series[method, "x2", "tube"]):
                d.add(Rect(px(rx["lo"]), py(ry["lo"]), px(rx["hi"])-px(rx["lo"]),
                           py(ry["hi"])-py(ry["lo"]), strokeColor=faint, strokeWidth=.28, fillColor=None))
            # The initial box and final boxes remain actual coordinate rectangles.
            boxes = [(initial["x1"], initial["x2"], BLACK, [5, 3])]
            for other in (method, "huan", "xiangru"):
                rx = markers[other, "x1"] if other in ("huan", "xiangru") else series[other, "x1", "endpoint"][-1]
                ry = markers[other, "x2"] if other in ("huan", "xiangru") else series[other, "x2", "endpoint"][-1]
                boxes.append(([rx["lo"], rx["hi"]], [ry["lo"], ry["hi"]], colors.HexColor(COLORS[other]), [2, 2] if other == "xiangru" else []))
            for bx, by, color, dash in boxes:
                d.add(Rect(px(bx[0]), py(by[0]), px(bx[1])-px(bx[0]), py(by[1])-py(by[0]),
                           strokeColor=color, strokeWidth=1.8, strokeDashArray=dash, fillColor=None))
        text(d, 615, 96, "Each faint rectangle is [pooled x1 lower, upper] x [pooled x2 lower, upper] for one saved tube step.", 12, anchor="middle")
        text(d, 615, 73, "Black dashed: initial box. Solid method outline: its final endpoint box. Orange/pink: author final boxes only.", 11, anchor="middle")
        text(d, 615, 50, "All panels use the same coordinate scales. These axis-aligned boxes do not preserve per-lane or octagon correlation.", 11, anchor="middle")
    text(d, 615, 12, "Saved numerical interval geometry only; no independent NNCS certificate or inferred continuous trajectory.", 11, anchor="middle")
    return d


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-run", type=Path, required=True, help="Completed run folder containing candidate/ and its outer RESULT")
    parser.add_argument("--reference-data", type=Path, default=OLD_WIDTHS)
    parser.add_argument("--selection", type=Path, default=STUDY/"report_data/selection.json")
    parser.add_argument("--output", type=Path, default=ROOT/"docs/evidence/results/archcomp26_report_20261006/figures")
    parser.add_argument("--pdftoppm", type=Path)
    parser.add_argument("--validate-only", action="store_true", help="Read and validate sources without creating artifacts")
    args = parser.parse_args()
    run = args.candidate_run.resolve()
    series, markers, sources = old_series(args.reference_data)
    candidate = candidate_series(run, series)
    declared = contract(run / "candidate/data/config.yaml")
    selection = read(args.selection)
    selected = any(v["run_id"] == run.name for k,v in selection.items() if k == "quad-reach")
    if args.validate_only:
        print(json.dumps(dict(status="source_data_validated", curves=len(series), points_per_curve=1000,
                              author_endpoint_markers=len(markers), candidate_run=run.name, selected=selected)))
        return
    renderer = args.pdftoppm or shutil.which("pdftoppm") or Path("/Users/shengenli/.cache/codex-runtimes/codex-primary-runtime/dependencies/bin/override/pdftoppm")
    if not Path(renderer).is_file():
        raise FileNotFoundError("pdftoppm is required")
    args.output.mkdir(parents=True, exist_ok=True)
    artifacts = []
    drawings = {f"quad_{g}_widths": drawing(g, series, markers, run.name, selected) for g in ("endpoint", "tube")}
    drawings.update({f"quad_{g}": geometry_drawing(g, series, markers, run.name, selected, declared)
                     for g in ("time_x3", "x1_x2")})
    for name, graphic in drawings.items():
        pdf, png = args.output/(name+".pdf"), args.output/(name+".png")
        renderPDF.drawToFile(graphic, str(pdf))
        subprocess.run([str(renderer), "-png", "-singlefile", "-r", "165", str(pdf), str(args.output/name)], check=True, capture_output=True)
        artifacts.append(dict(pdf=rel(pdf), pdf_bytes=pdf.stat().st_size, png=rel(png), png_bytes=png.stat().st_size))
    metadata = [dict(method=k[0], state=k[1], geometry=k[2], points=len(rows),
                     saved_object="pooled_observer", terminal=rows[-1],
                     max_per_step_width=max(r["width"] for r in rows)) for k,rows in series.items()]
    receipt = dict(candidate_run=run.name, candidate=rel(run), selected=selected,
        selection_file=rel(args.selection), selection_entry=selection.get("quad-reach"),
        candidate_receipts=candidate, reference_data=rel(args.reference_data), reference_sources=sources,
        contract=declared,
        curve_series=metadata, author_endpoint_markers=[dict(method=k[0], state=k[1], **v) for k,v in markers.items()],
        missing="Huan/Xiangru stepwise endpoint trajectories and all tube data are unavailable; only terminal driver hull markers are plotted.",
        saved_object_boundary="Old/new P3 and native curves are pooled observers. Huan/Xiangru points are terminal driver hulls. The old P3 separate driver final hull is not substituted into its observer curve.",
        artifacts=artifacts, no_experiments=True, old_checker_executed=False, digest_operations=0,
        qualification="Numerical saved widths only; selection is recorded, never inferred from narrower output. No independent NNCS certificate.")
    (args.output/"QUAD_SOURCES.json").write_text(json.dumps(receipt, indent=2, allow_nan=False)+"\n")
    print(json.dumps(dict(figures=len(artifacts), curves=len(series), author_endpoint_markers=len(markers), selected=selected, output=str(args.output))))


if __name__ == "__main__":
    main()
