#!/usr/bin/env python3
"""Audit saved first-step plant-containment ledgers without solver calls or digests."""

from fractions import Fraction as F
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / "native_quad_allbox_independent_plant_firststep_fixed_decimal_20261003_005"
LEDGERS = (
    PRIOR / "probe_lane0/BOX_AUDIT.jsonl",
    PRIOR / "remainder_1_1023/BOX_AUDIT.jsonl",
    HERE / "probe_lane7/BOX_AUDIT.jsonl",
    HERE / "remainder_8_1023/BOX_AUDIT.jsonl",
)
EXPECTED_COMPARISONS = ({("tube", i) for i in range(4)} |
                        {("endpoint", i) for i in range(4)} |
                        {("terminal_composed", i) for i in range(1, 13)})


def main():
    rows = []
    ledger_counts = {}
    for path in LEDGERS:
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open() as stream:
            batch = [json.loads(line) for line in stream if line.strip()]
        ledger_counts[str(path.relative_to(HERE.parent))] = len(batch)
        rows.extend(batch)

    if [ledger_counts[str(path.relative_to(HERE.parent))] for path in LEDGERS] != [1, 6, 1, 1016]:
        raise AssertionError("combined gate needs the complete 1+6+1+1016 saved lanes")
    continuation = json.loads((HERE / "remainder_8_1023/RESULT.json").read_text())
    if continuation["checked_boxes"] != 1016 or continuation["first_issue"] is not None:
        raise AssertionError("continuation did not complete cleanly")

    for expected_lane, row in enumerate(rows):
        lane = row["lane"]
        if lane != expected_lane or row["strict_picard_substeps"] != 1000:
            raise AssertionError(f"missing/duplicate lane or Picard step at {expected_lane}")
        bootstrap = row["strict_bootstrap"]
        required = {"x4", "x5", "x6", "x7", "x8"}
        if lane >= 7:
            required |= {"x10_rate", "x11_rate"}
        if set(bootstrap) != required or any(not F(a) < F(b) for a, b in bootstrap.values()):
            raise AssertionError(f"bootstrap inequality at lane {lane}")
        if lane < 7:
            controls = row["control_hull_exact"]
            inertia, h = F(54, 1000), F(5, 1000)
            if not (h * max(abs(F(v)) for v in controls[1]) / inertia < F(3, 2000) and
                    h * max(abs(F(v)) for v in controls[2]) / inertia < F(1, 1000)):
                raise AssertionError(f"fixed first-box rate at lane {lane}")
        elif F(row["adaptive_rates_exact"]["strict_margin"]) != F(1, 10**9):
            raise AssertionError(f"adaptive margin at lane {lane}")

        comparisons = row["comparisons"]
        keys = {(entry["kind"], entry["coord_or_form"]) for entry in comparisons}
        if (len(comparisons) != 20 or keys != EXPECTED_COMPARISONS or
                row["physical_outward_comparisons_inside"] != 20 or
                row["physical_unexpanded_binary64_comparisons_inside"] != 20):
            raise AssertionError(f"comparison grid/count at lane {lane}")
        for entry in comparisons:
            inner_lo, inner_hi = map(F, entry["bound_exact"])
            outer_lo, outer_hi = map(F, entry["saved_outward_exact"])
            if not (inner_lo <= inner_hi and outer_lo <= outer_hi and
                    outer_lo <= inner_lo and inner_hi <= outer_hi and
                    entry["inside_outward"] and entry["inside_unexpanded_binary64"]):
                raise AssertionError(f"first noncontained comparison at lane {lane}: {entry}")

    summary = {
        "schema": "native-quad-combined-ledger-audit-nohash-v1",
        "lane_count": len(rows),
        "lane_span": [0, len(rows) - 1] if rows else [],
        "ledger_counts": ledger_counts,
        "strict_picard_substeps_per_lane": 1000,
        "physical_composed_comparisons_inside": 20 * len(rows),
        "physical_composed_comparisons_inside_unexpanded_binary64": 20 * len(rows),
        "all_recorded_strict_bootstrap_inequalities_verified": True,
        "scope": "saved first control call, first h=0.005 paper-ODE plant step only",
        "production_gate": "CLOSED",
    }
    (HERE / "COMBINED_LEDGER_AUDIT.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
