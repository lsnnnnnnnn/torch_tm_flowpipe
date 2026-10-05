#!/usr/bin/env python3
"""Derive a descriptive full QUAD assessment from saved CSVs and receipts."""
import json
from derive_candidate_width_assessment import HERE, summarize


def main():
    data = summarize("quad_paper_anchored_controlv2_full1000_001", quad=True, expected_steps=1000)
    data.update(scope="Read-only saved full1000 control-v2 numerical candidate; not the selected report run",
                generated_on="2026-10-06", experiments_run=False, old_checker_executed=False,
                digest_operations=0, primary_selected=False,
                qualification="Saved interval width is not containment. This full candidate and its single timing sample do not establish an independent NNCS certificate.")
    assert data["rows"] == 24000
    assert data["counts"] == {"narrower": 21990, "equal": 2002, "wider": 8}
    lines = ["# QUAD control v2 full-horizon saved assessment", "",
        "This report only reads the full candidate's saved CSV and receipts. It runs no numerical solver or old checker. The candidate is explicit and is not promoted to the report's primary selection here.", "",
        f"All 1000 steps, 1024 initial boxes and 12 physical states are represented by 24000 endpoint/tube rows: 21990 narrower, 2002 equal, 8 wider. {data['subset_count']} candidate intervals are subsets of their references; narrower is not automatically containment.", "",
        f"Saved internal driver time: {data['previous_driver_elapsed_s']:.12g} → {data['driver_elapsed_s']:.12g} s ({data['driver_change_percent']:+.6f}%). Candidate process wall time: {data['process_wall_s']:.12g} s. The time reference is the October 5 private256 full1000 run, while geometry is compared against the original canonical full50 saved observer. Single timings do not establish a stable ranking.", "",
        "Negative terminal changes mean narrower. Per-state units are kept separate.", "",
        "| State | Old endpoint width at 5 s | New endpoint width at 5 s | Endpoint change % | Tube at 5 s change % | Max per-step tube change % |", "|---|---:|---:|---:|---:|---:|"]
    for n in range(1, 13):
        groups = {g['geometry']:g for g in data['groups'] if g['state'] == f'x{n}'}
        ep, tube = groups['endpoint']['terminal'], groups['tube']['terminal']
        oldmax, newmax = groups['tube']['old_max_per_step_width'], groups['tube']['new_max_per_step_width']
        maxchange = 100*(newmax-oldmax)/oldmax if oldmax else 0
        lines.append(f"| x{n} | {ep['old_width']:.12g} | {ep['new_width']:.12g} | {ep['percent_change']:.9g} | {tube['percent_change']:.9g} | {maxchange:.9g} |")
    lines += ["", "Eight positive differences remain in their exact saved values; no tolerance hides them.", "",
        "| State/object | Step | Time s | Old width | New width | Absolute increase | Relative increase % |", "|---|---:|---:|---:|---:|---:|---:|"]
    # All positive rows are included in the saved non-subset list.
    wider = [r for r in data['non_subset_records'] if r['classification'] == 'wider']
    assert len(wider) == 8
    data['wider_records'] = wider
    for row in sorted(wider, key=lambda r:(r['step'],r['geometry'])):
        lines.append(f"| {row['state']} {row['geometry']} | {row['step']} | {row['time_s']:.12g} | {row['old_width']:.12g} | {row['new_width']:.12g} | {row['delta']:.12g} | {row['percent_change']:.12g} |")
    lines += ["", "All source paths, byte sizes, all 24 terminal rows, exact rational differences, old/new maximum per-step widths, wider step lists, and all non-subset rows are retained in the adjacent JSON. The control receipt retains actual additional CROWN call counts and per-refresh contraction statistics. Component wall times remain unsynchronized host measurements.", "", "Rebuild with `python3 -B research/p3_speed_tightness_20261006/tightness/derive_quad_full_width_assessment.py`.", ""]
    (HERE / "QUAD_CONTROL_FULL_WIDTH_ASSESSMENT.json").write_text(json.dumps(data, indent=2)+"\n")
    (HERE / "QUAD_CONTROL_FULL_WIDTH_ASSESSMENT.md").write_text("\n".join(lines))
    print(json.dumps(dict(counts=data['counts'], subset=data['subset_count'], driver=data['driver_elapsed_s'],
        driver_change_percent=data['driver_change_percent'], terminal=[g['terminal'] for g in data['groups'] if g['geometry']=='endpoint'],
        wider=wider), indent=2))


if __name__ == "__main__":
    main()
