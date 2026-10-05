#!/usr/bin/env python3
"""Summarize saved candidate CSVs; no experiment, checker or digest operation."""
import csv
from collections import Counter
from fractions import Fraction
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RESULTS = HERE.parent / "results"


def read(path):
    return json.loads(path.read_text())


def summarize(run, quad=False, expected_steps=None):
    folder = RESULTS / run
    path = folder / "candidate/ABSOLUTE_WIDTHS.csv"
    data = list(csv.DictReader(path.open()))
    records = []
    for row in data:
        old_prefix, new_prefix = ("reference", "candidate") if quad else ("old", "new")
        old = Fraction(float(row[old_prefix + "_upper"])) - Fraction(float(row[old_prefix + "_lower"]))
        new = Fraction(float(row[new_prefix + "_upper"])) - Fraction(float(row[new_prefix + "_lower"]))
        delta = new - old
        assert delta == Fraction(row["width_difference_rational"])
        label = "narrower" if delta < 0 else "wider" if delta > 0 else "equal"
        assert label == row["classification"]
        records.append(dict(state=row["state"], geometry=row["geometry"],
            step=int(row["substep" if quad else "step"]),
            time_s=float(row["nominal_t_right" if quad else "nominal_t"]),
            old_width=float(old), new_width=float(new), delta=float(delta),
            delta_exact=str(delta), percent_change=float(delta / old * 100) if old else None,
            classification=label, subset=row["candidate_subset_reference"] == "True"))
    groups = []
    states = sorted({r["state"] for r in records}, key=lambda x: int(x[1:]))
    expected_steps = expected_steps or (40 if quad else 500)
    for state in states:
        for geometry in ("endpoint", "tube"):
            rows = [r for r in records if r["state"] == state and r["geometry"] == geometry]
            rows.sort(key=lambda r: r["step"])
            assert [r["step"] for r in rows] == list(range(1, expected_steps + 1))
            wider = [r for r in rows if r["classification"] == "wider"]
            groups.append(dict(state=state, geometry=geometry, steps=len(rows),
                counts=dict(Counter(r["classification"] for r in rows)),
                terminal=rows[-1], old_max_per_step_width=max(r["old_width"] for r in rows),
                new_max_per_step_width=max(r["new_width"] for r in rows),
                wider_steps=[r["step"] for r in wider],
                first_wider=wider[0] if wider else None,
                largest_absolute_increase=max(wider, key=lambda r: r["delta"]) if wider else None,
                largest_relative_increase=max(wider, key=lambda r: r["percent_change"]) if wider else None))
    comparison_file = folder / ("candidate/SAVED_COMPARISON.json" if quad else "candidate/WIDTH_COMPARISON.json")
    comparison = read(comparison_file)
    source_paths = [path, comparison_file, folder / "RESULT.json", folder / "candidate/data/metrics.json"]
    metrics = read(source_paths[-1])
    result = dict(run_id=run, saved_reference=comparison["reference"],
        sources=[dict(path=str(p.relative_to(ROOT)), bytes=p.stat().st_size) for p in source_paths],
        rows=len(records), counts=dict(Counter(r["classification"] for r in records)),
        subset_count=sum(r["subset"] for r in records), groups=groups,
        driver_elapsed_s=metrics["elapsed_s"], process_wall_s=read(folder / "RESULT.json")["wall_s"],
        independent_nncs_certificate=False, property_label_inherited=False)
    if quad:
        control_file = folder / "candidate/CONTROL_RESULT.json"
        result["control_receipt"] = read(control_file)
        result["sources"].append(dict(path=str(control_file.relative_to(ROOT)), bytes=control_file.stat().st_size))
        previous_id = "quad_paper_p3_private256_full1000_20261005_001" if expected_steps == 1000 else "quad_paper_p3_private256_20261005_001"
        previous_path = ROOT / "research/p3_speed_tightness_20261005/results" / previous_id / "run_001/candidate/data/metrics.json"
        previous = read(previous_path)["elapsed_s"]
        result["previous_driver_elapsed_s"] = previous
        result["driver_change_percent"] = 100 * (metrics["elapsed_s"] / previous - 1)
        result["previous_driver_source"] = str(previous_path.relative_to(ROOT))
        result["non_subset_records"] = [r for r in records if not r["subset"]]
    return result


def main():
    quad = summarize("quad_paper_anchored_controlv2_40_001", quad=True)
    tora = [summarize(f"tora_sigmoid_order{k}_cutoff1e8_full500_001") for k in (4, 6)]
    report = dict(scope="Read-only arithmetic summaries of saved candidate width CSVs and receipts",
        generated_on="2026-10-06", experiments_run=False, checkers_run=False, digest_operations=0,
        width_definition="Exact rational upper-lower of saved binary64 bounds; percentages use same-row reference width; no all-time union",
        quad_control_v2_40=quad, sigmoid_order_candidates=tora,
        decision="QUAD 40-step evidence warrants a new full-horizon diagnostic but is not promotion. Sigmoid orders 4 and 6 fail all-step no-width-increase selection criterion despite tighter terminal widths. No numerical tolerance erases their saved positive differences.")
    (HERE / "CANDIDATE_WIDTH_ASSESSMENT.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# Saved candidate width assessment", "", "This is a new arithmetic summary of saved CSVs and receipts. No experiment or numerical checker was run. Width is the exact difference of saved binary64 bounds; a smaller width is not an interval containment certificate. Each percentage is relative to the corresponding saved reference row.", "", "## QUAD anchored control v2: 40-step diagnostic", "",
        f"All {quad['rows']} saved state/geometry rows: {quad['counts'].get('narrower',0)} narrower, {quad['counts'].get('equal',0)} equal, {quad['counts'].get('wider',0)} wider; {quad['subset_count']} are subsets of their reference intervals. The 18 non-subsets are x10 endpoint rows, all narrower. All 24 terminal rows are subsets.", "",
        f"Driver {quad['previous_driver_elapsed_s']:.12g} → {quad['driver_elapsed_s']:.12g} s ({quad['driver_change_percent']:+.6f}%); candidate process {quad['process_wall_s']:.12g} s. These are single samples. The geometry reference is the original canonical batch2 saved output; the time reference is the saved private256 batch2 driver. Two additional CROWN calls yield four actual NN calls versus two base calls. Of 6144 control component rows, 3629 have a contracted remainder. Component timers are unsynchronized host timings and do not establish device cost attribution.", "",
        "Terminal time 0.2 s. Negative changes mean narrower.", "",
        "| State | Old endpoint width | New endpoint width | Endpoint change % | Tube change % |", "|---|---:|---:|---:|---:|"]
    for n in range(1, 13):
        rows = {g['geometry']: g['terminal'] for g in quad['groups'] if g['state'] == f'x{n}'}
        ep, tube = rows['endpoint'], rows['tube']
        lines.append(f"| x{n} | {ep['old_width']:.12g} | {ep['new_width']:.12g} | {ep['percent_change']:.9g} | {tube['percent_change']:.9g} |")
    x3 = next(g['terminal'] for g in quad['groups'] if g['state'] == 'x3' and g['geometry'] == 'endpoint')
    lines += ["", f"The height state x3 gains only {-x3['delta']:.12g} absolute width, or {-x3['percent_change']:.9g}%, at this short prefix. The largest terminal relative gain is x7 (about 0.128%). This supports a full numerical trial to measure accumulated behavior, not a claim of a significant height gain or full-horizon dominance. The same-controller floating CROWN enclosure assumption and missing independent NNCS certificate remain. QUAD is not promoted from this prefix.", "", "## Sigmoid order 4 and order 6", "", "Both use cutoff 1e-8 and compare with the saved order-3/cutoff-1e-6 reference. Reported early increases are retained exactly; they are not removed by a tolerance. Maximum absolute and relative increases may occur on different rows, so both row locations are given."]
    for result in tora:
        lines += ["", f"### {result['run_id']}", "",
            f"4000 rows: {result['counts'].get('narrower',0)} narrower, {result['counts'].get('equal',0)} equal, {result['counts'].get('wider',0)} wider; {result['subset_count']} subsets. Driver {result['driver_elapsed_s']:.12g} s; process {result['process_wall_s']:.12g} s.", "",
            "| State/object | Wider steps | Max absolute increase (step) | Relative % at that step | Max relative increase % (step) |", "|---|---|---:|---:|---:|"]
        for group in result['groups']:
            a, r = group['largest_absolute_increase'], group['largest_relative_increase']
            if a:
                steps = group['wider_steps']
                label = f"{steps[0]}–{steps[-1]} ({len(steps)})"
                lines.append(f"| {group['state']} {group['geometry']} | {label} | {a['delta']:.12g} ({a['step']}) | {a['percent_change']:.12g} | {r['percent_change']:.12g} ({r['step']}) |")
        lines += ["", "All terminal widths are smaller, but neither order candidate meets the all-state/all-step no-width-increase criterion. The JSON also retains all terminal widths, separate old/new maximum per-step widths, the first wider row, exact rational positive differences, and full wider-step lists."]
    lines += ["", "## Rebuild and provenance", "", "Run `python3 -B research/p3_speed_tightness_20261006/tightness/derive_candidate_width_assessment.py` from this repository. The adjacent JSON includes precise source paths and byte sizes. It reads existing comparison receipts and CSVs only; no checker or source experiment is invoked. All source files remain unchanged.", ""]
    (HERE / "CANDIDATE_WIDTH_ASSESSMENT.md").write_text("\n".join(lines))
    print(json.dumps({"quad_counts": quad['counts'], "driver_change_percent": quad['driver_change_percent'], "sigmoid": [{"run": r['run_id'], "counts": r['counts']} for r in tora]}))


if __name__ == "__main__":
    main()
