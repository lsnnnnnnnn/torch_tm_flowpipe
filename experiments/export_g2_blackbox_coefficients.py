#!/usr/bin/env python3
"""Export project-core G2 coefficient tables for an independent oracle."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch

from torch_tm_flowpipe.g2_shared_column import (
    G2_SHARED_COLUMN_CANDIDATE,
    G2SharedColumnState,
    accepted_successor,
    commit_or_preserve,
    partition_source_terms,
    polynomial_table,
    rotate_current_to_retained,
)
from torch_tm_flowpipe.interval import Interval
from torch_tm_flowpipe.polynomial import Polynomial
from torch_tm_flowpipe.source_ledger import collapse_source_polynomial


def table(poly: Polynomial) -> list[list[object]]:
    return polynomial_table(poly)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    # Shared-column affine substitution into the VDP nonlinear monomial x^2 y.
    u = Polynomial.variable(0, 2)
    z = Polynomial.variable(1, 2)
    x = Polynomial.constant(1.0, 2) + u * 0.5 + z * 0.125
    y = Polynomial.constant(2.0, 2) - u * 0.25 + z * 0.125
    x2y = x * x * y

    # Fixed-6-variable rotation and mixed oldest/current retirement.
    base = Polynomial.variable(0, 6)
    oldest = Polynomial.variable(2, 6)
    current = Polynomial.variable(4, 6)
    mixed = (
        base * 0.5
        + current * 0.25
        + base * current * 0.125
        + oldest * current * 0.0625
        + oldest * oldest * 0.03125
    )
    collapse = collapse_source_polynomial(mixed, [Interval(-1.0, 1.0) for _ in range(6)], (2, 3))
    current_partition = partition_source_terms(collapse.retained, (4, 5))
    rotated = rotate_current_to_retained(current_partition.source_bearing, 2)

    # Complete degree-four ownership fixture.
    one_plus = Polynomial.constant(1.0, 2) + u + z
    fifth = one_plus * one_plus * one_plus * one_plus * one_plus
    kept, dropped = fifth.truncate(4)

    initial = G2SharedColumnState.initial(2)
    proposed = accepted_successor(
        initial,
        torch.tensor([[0.125, 0.25]], dtype=torch.float64),
        ("polynomial_truncation", "picard_residual"),
        retained_payload_sha256=initial.retained_payload_sha256,
        retained_active=(False, False),
    )
    rejected = commit_or_preserve(initial, proposed, accepted=False)

    payload = {
        "schema": "g2_project_blackbox_coefficients_v1",
        "candidate": G2_SHARED_COLUMN_CANDIDATE,
        "arithmetic": "binary64_dyadic_fixtures",
        "cases": {
            "canonical_merge": {
                "input_terms": [
                    [[1], "1/2"],
                    [[1], "1/4"],
                    [[1], "-1/8"],
                    [[0], "3/2"],
                ],
                "observed": table(
                    Polynomial.variable(0, 1) * 0.5
                    + Polynomial.variable(0, 1) * 0.25
                    - Polynomial.variable(0, 1) * 0.125
                    + Polynomial.constant(1.5, 1)
                ),
            },
            "affine_shared_column_x2y": {
                "x": [[[0, 0], "1"], [[1, 0], "1/2"], [[0, 1], "1/8"]],
                "y": [[[0, 0], "2"], [[1, 0], "-1/4"], [[0, 1], "1/8"]],
                "observed": table(x2y),
            },
            "oldest_current_retirement": {
                "input": table(mixed),
                "oldest_indices": [2, 3],
                "current_indices": [4, 5],
                "retained_after_collapse": table(collapse.retained),
                "collapsed_interval_hex": [
                    float(collapse.collapsed.lo.detach().cpu()).hex(),
                    float(collapse.collapsed.hi.detach().cpu()).hex(),
                ],
                "rotated_current": table(rotated),
            },
            "degree4_truncation": {
                "base": [[[0, 0], "1"], [[1, 0], "1"], [[0, 1], "1"]],
                "power": 5,
                "order": 4,
                "kept": table(kept),
                "dropped": table(dropped),
            },
            "retry_atomicity": {
                "accepted": False,
                "object_identity_preserved": rejected is initial,
                "before": initial.as_dict(),
                "after": rejected.as_dict(),
                "proposed_generation": proposed.generation,
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "PASS", "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
