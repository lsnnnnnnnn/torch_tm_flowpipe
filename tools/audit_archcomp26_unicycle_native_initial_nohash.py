#!/usr/bin/env python3
"""Check the actual native Unicycle initial affine ledger with exact rationals."""

import argparse
import csv
from fractions import Fraction
import json
from pathlib import Path


def exact_hex(token):
    return Fraction.from_float(float.fromhex(token))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    boxes = json.loads((args.run / "initial_boxes.json").read_text())
    with (args.run / "initial_affine.tsv").open() as source:
        rows = list(csv.DictReader(source, delimiter="\t"))
    if len(boxes) != 1 or len(boxes[0]) != 8 or len(rows) != 8:
        raise ValueError("not one complete native Unicycle initial box")
    actual = []
    for index, row in enumerate(rows):
        if int(row["state"]) != index:
            raise ValueError("initial state order differs")
        lower = exact_hex(row["requested_lo"])
        upper = exact_hex(row["requested_hi"])
        if lower != Fraction.from_float(boxes[0][index][0]) or upper != Fraction.from_float(boxes[0][index][1]):
            raise ValueError("affine ledger does not match saved requested box")
        center = exact_hex(row["actual_center"])
        radius = exact_hex(row["actual_radius"])
        covers = radius >= 0 and center - radius <= lower and center + radius >= upper
        actual.append({"state": ["x1", "x2", "x3", "x4", "t", "u1", "u2", "w"][index],
                       "requested": [float(lower), float(upper)],
                       "affine_image": [float(center - radius), float(center + radius)],
                       "changed_from_original_constructor": row["changed"] == "1",
                       "covers_requested": covers})
    complete = all(row["covers_requested"] for row in actual)
    repaired = [row["state"] for row in actual if row["changed_from_original_constructor"]]
    result = {"schema": "archcomp26-unicycle-native-initial-fraction-nohash-v1",
              "run": str(args.run), "state_order": [row["state"] for row in actual],
              "all_8_initial_coordinates_covered": complete,
              "repaired_states": repaired, "coordinates": actual,
              "limit": "Checks recorded 53-bit actual initial affine coefficients; no NNCS or trajectory proof."}
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"all_8_initial_coordinates_covered": complete, "repaired_states": repaired}))
    return 0 if complete and repaired == ["x2"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
