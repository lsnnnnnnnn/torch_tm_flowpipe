#!/usr/bin/env python3
"""Combine fixed-method first-box probe and stopped sequential continuation."""

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


def load(path):
    return json.loads(path.read_text())


def ledger(directory):
    return [json.loads(line) for line in (HERE / directory / "BOX_AUDIT.jsonl").read_text().splitlines()
            if line.strip()]


def main():
    probe = load(HERE / "probe_lane0/RESULT.json")
    remainder = load(HERE / "remainder_1_1023/RESULT.json")
    firstbox = load(HERE / "FIRSTBOX_COMPARISON.json")
    crosscheck = load(HERE / "FIRSTBOX_RECEIPT_CROSSCHECK.json")
    blocker = load(HERE / "FIRST_UNDECIDED.json")
    rows = ledger("probe_lane0") + ledger("remainder_1_1023")
    assert [row["lane"] for row in rows] == list(range(7))
    assert all(row["strict_picard_substeps"] == 1000 and
               row["physical_outward_comparisons_inside"] == 20 and
               row["physical_unexpanded_binary64_comparisons_inside"] == 20
               for row in rows)
    assert firstbox["fixed_broad_inside"] == 27 and firstbox["fixed_comparison_count"] == 32
    assert crosscheck["all_passed"]
    assert probe["first_issue"] is None and remainder["first_issue"]["lane"] == 7
    assert blocker["failed_inequalities"] == ["x11_rate"]
    result = {
        "schema": "native-quad-fixed-decimal-firststep-prefix-nohash-v1",
        "source_boxes_total": 1024,
        "lanes_independently_contained": list(range(7)),
        "physical_composed_comparisons_inside": 140,
        "physical_composed_comparisons_total": 140,
        "physical_composed_inside_unexpanded_binary64": 140,
        "strict_picard_substeps_per_accepted_lane": 1000,
        "first_undecided_lane": 7,
        "first_undecided_reason": "fixed first-box x11 bootstrap rate limit 0.001 is below lane-7 exact saved-control bound 0.0010041811782920613",
        "first_undecided_picard_steps": 0,
        "remaining_unattempted_lanes": 1016,
        "prior_firstbox_broad_fixed_decimal_inside": 27,
        "prior_firstbox_broad_total": 32,
        "firstbox_saved_receipts_match_after_float32": True,
        "server_solver_or_crown_restarted": False,
        "production_gate": "CLOSED",
        "interpretation": "Seven source-defined boxes have corrected directed-Decimal plant first-step bounds inside saved outward-expanded composed observer intervals. Lane 7 stopped on a method bootstrap constant, not a Flow* containment failure. No conclusion for lanes 7-1023, later steps, CROWN/NN validity, Flow* floating parser, or reach-and-remain.",
    }
    (HERE / "RESULT.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in (
        "lanes_independently_contained", "physical_composed_comparisons_inside",
        "first_undecided_lane", "remaining_unattempted_lanes")}))


if __name__ == "__main__":
    main()
