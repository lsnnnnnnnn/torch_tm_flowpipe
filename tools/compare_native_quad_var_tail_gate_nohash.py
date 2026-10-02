"""Compare saved one-step QUAD intervals; no solver execution or digests."""

import csv
import json
import sys
from pathlib import Path


def rows(path, key_fields, interval_fields):
    with path.open(newline="") as stream:
        result = {}
        for row in csv.DictReader(stream):
            key = tuple(int(row[name]) for name in key_fields)
            if key in result:
                raise ValueError(f"duplicate key {key} in {path}")
            result[key] = {
                name: [float(row[lo]), float(row[hi])]
                for name, lo, hi in interval_fields
            }
    return result


def compare(original, skipped, repaired):
    if not original.keys() == skipped.keys() == repaired.keys():
        raise ValueError("saved row keys differ")
    output = []
    for key in sorted(repaired):
        for name in repaired[key]:
            a, b, c = (table[key][name] for table in (original, skipped, repaired))
            if any(lo > hi for lo, hi in (a, b, c)):
                raise ValueError(f"reversed interval at {key}/{name}")
            old_width, skip_width, fixed_width = (hi - lo for lo, hi in (a, b, c))
            output.append({
                "key": key,
                "interval": name,
                "original": a,
                "skip_refinement": b,
                "var_tail_repair": c,
                "width_original": old_width,
                "width_skip": skip_width,
                "width_repair": fixed_width,
                "skip_to_repair_width_ratio": skip_width / fixed_width if fixed_width else None,
                "repair_within_skip": b[0] <= c[0] and c[1] <= b[1],
                "strictly_narrower_than_skip": fixed_width < skip_width,
            })
    return output


def main():
    if len(sys.argv) != 5:
        raise SystemExit("usage: comparison.py original-dir skip-dir repair-dir output.json")
    original, skipped, repaired, output = map(Path, sys.argv[1:])
    terminal_fields = (("pre", "pre_lo", "pre_hi"),
                       ("composed", "composed_lo", "composed_hi"))
    octagon_fields = (("support", "lo", "hi"),)
    comparisons = {}
    for name, keys, fields in (("terminal_axes", ("coord",), terminal_fields),
                               ("octagon", ("step", "view", "form"), octagon_fields)):
        tables = [rows(root / "on" / f"{name}.csv", keys, fields)
                  for root in (original, skipped, repaired)]
        comparisons[name] = compare(*tables)
    all_rows = comparisons["terminal_axes"] + comparisons["octagon"]
    result = {
        "schema": "native-quad-var-tail-width-comparison-nohash-v1",
        "contract": "same first 2026-paper QUAD box, one CROWN call, h=0.005/order-2 step",
        "inputs": {"original": str(original), "skip_refinement": str(skipped),
                   "var_tail_repair": str(repaired)},
        "controller_rpc_equal_to_original": all(
            json.loads((root / "on" / "rpc.json").read_text()) ==
            json.loads((original / "on" / "rpc.json").read_text())
            for root in (skipped, repaired)),
        "all_repaired_intervals_within_skip": all(row["repair_within_skip"] for row in all_rows),
        "strictly_narrower_than_skip_count": sum(row["strictly_narrower_than_skip"] for row in all_rows),
        "interval_count": len(all_rows),
        "terminal_axes": comparisons["terminal_axes"],
        "octagon": comparisons["octagon"],
        "qualification": "one-box finite diagnostic; narrowing does not prove soundness or full-time reachability",
        "content_digest_policy": "none computed",
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in (
        "controller_rpc_equal_to_original", "all_repaired_intervals_within_skip",
        "strictly_narrower_than_skip_count", "interval_count")}))


if __name__ == "__main__":
    main()
