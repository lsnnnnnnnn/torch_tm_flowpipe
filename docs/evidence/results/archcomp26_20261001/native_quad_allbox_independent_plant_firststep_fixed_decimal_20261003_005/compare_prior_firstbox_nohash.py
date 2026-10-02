#!/usr/bin/env python3
"""Rerun only the saved first-box interval readback with fixed arithmetic."""

import json
from pathlib import Path
import time

import interval_fixed_nohash as interval


HERE = Path(__file__).resolve().parent
OLD = HERE.parent / "native_quad_initial_recenter_repair_gate_20261003_002"


def main():
    begin = time.perf_counter()
    fixed = interval.run(OLD)
    elapsed = time.perf_counter() - begin
    (HERE / "FIRSTBOX_FIXED_32.json").write_text(json.dumps(fixed, indent=2) + "\n")
    prior = json.loads((OLD / "INDEPENDENT_AUDIT.json").read_text())
    def key(row):
        return row["kind"], row.get("coord", row.get("form"))
    before = {key(row): row for row in prior["comparisons"]}
    after = {key(row): row for row in fixed["comparisons"]}
    if len(before) != len(after) or set(before) != set(after):
        raise ValueError("first-box comparison keys changed")
    changed = [list(k) for k in before if before[k]["inside"] != after[k]["inside"]]
    summary = {
        "schema": "native-quad-firstbox-fixed-decimal-comparison-nohash-v1",
        "saved_firstbox_run": str(OLD),
        "fixed_elapsed_seconds": elapsed,
        "prior_broad_inside": sum(row["inside"] for row in before.values()),
        "fixed_broad_inside": sum(row["inside"] for row in after.values()),
        "prior_comparison_count": len(before),
        "fixed_comparison_count": len(after),
        "boolean_verdict_changed_keys": changed,
        "fixed_picard_strict_inclusion": fixed["picard_inclusion_passed"],
        "interpretation": "Saved first-box arithmetic requalified after exact Decimal sign and abs fixes; broad dependency-inconclusives remain for separate exact algebra, not plant counterexamples.",
    }
    (HERE / "FIRSTBOX_COMPARISON.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: summary[key] for key in (
        "fixed_elapsed_seconds", "prior_broad_inside", "fixed_broad_inside",
        "boolean_verdict_changed_keys")}))
    return 0 if summary["fixed_picard_strict_inclusion"] and not changed else 1


if __name__ == "__main__":
    raise SystemExit(main())
