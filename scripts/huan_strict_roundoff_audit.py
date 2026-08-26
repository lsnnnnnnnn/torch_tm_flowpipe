#!/usr/bin/env python3
"""Exact-rational witness/replay for symbolic-Phi strict roundoff."""

from __future__ import annotations

import argparse
from fractions import Fraction
import importlib
import inspect
import json
from pathlib import Path
import subprocess
import sys


MATRICES = [
    [[0.0, 0.0], [0.0, 0.0]],
    [[1.0000000000000002, 0.0], [-1.0, 0.0]],
    [[0.0, 0.0], [0.9999999999999999, 1.0000000000000002]],
]
J_COLUMNS = [[1048576.0, 0.0], [0.0, 0.0]]


def _mm(a, b):
    return [
        [sum((a[i][k] * b[k][j] for k in range(2)), Fraction()) for j in range(2)]
        for i in range(2)
    ]


def _mv(a, b):
    return [sum((a[i][k] * b[k] for k in range(2)), Fraction()) for i in range(2)]


def exact_result() -> list[Fraction]:
    phis = []
    js = []
    result = [Fraction(), Fraction()]
    for step, matrix in enumerate(MATRICES):
        exact_matrix = [[Fraction(value) for value in row] for row in matrix]
        for q in range(1, len(phis)):
            phis[q] = _mm(exact_matrix, phis[q])
        phis.append(exact_matrix)
        result = [Fraction(), Fraction()]
        for q in range(1, len(phis)):
            image = _mv(phis[q], js[q - 1])
            result = [x + y for x, y in zip(result, image, strict=True)]
        if step < len(J_COLUMNS):
            js.append([Fraction(value) for value in J_COLUMNS[step]])
    return result


def _head(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()


def run(engine_root: Path) -> dict[str, object]:
    sys.path.insert(0, str(engine_root.resolve() / "src"))
    torch = importlib.import_module("torch")
    iv = importlib.import_module("flowstar_gpu.interval")
    sr_mod = importlib.import_module("flowstar_gpu.symbolic_remainder")
    supports_strict = "strict" in inspect.signature(sr_mod.propagate).parameters
    parity = sr_mod.make_symbolic_remainder(1, 2, 8, "cpu")
    strict = sr_mod.make_symbolic_remainder(1, 2, 8, "cpu") if supports_strict else None
    parity_out = strict_out = None
    for step, matrix in enumerate(MATRICES):
        point = torch.tensor([matrix], dtype=torch.float64)
        parity_out = sr_mod.propagate(parity, point)
        if strict is not None:
            strict_out = sr_mod.propagate(
                strict, point, strict=True, phi_i_iv=iv.from_point(point)
            )
        if step < len(J_COLUMNS):
            j = torch.tensor(
                [[[value, value] for value in J_COLUMNS[step]]], dtype=torch.float64
            )
            parity.append_j(j)
            if strict is not None:
                strict.append_j(j)
    assert parity_out is not None
    exact = exact_result()

    def describe(out):
        intervals = out[0].tolist()
        contains = [
            Fraction(bounds[0]) <= value <= Fraction(bounds[1])
            for bounds, value in zip(intervals, exact, strict=True)
        ]
        return {"intervals": intervals, "contains_exact": contains, "all_contained": all(contains)}

    parity_desc = describe(parity_out)
    strict_desc = describe(strict_out) if strict_out is not None else None
    classification = (
        "D6_CONCRETE_UNDERENCLOSURE_WITNESS_FOUND"
        if not parity_desc["all_contained"] and strict_desc is None
        else "D6_REPAIRED_STRICT_CONTAINS_MINIMAL_WITNESS"
        if not parity_desc["all_contained"] and strict_desc and strict_desc["all_contained"]
        else "D6_WITNESS_NOT_REPRODUCED"
    )
    return {
        "schema": "torch_tm_flowpipe.huan_d6_minimal_witness/1",
        "engine_root": str(engine_root.resolve()),
        "engine_head": _head(engine_root),
        "matrices": MATRICES,
        "historical_j_columns": J_COLUMNS,
        "exact_result_fraction": [str(value) for value in exact],
        "exact_result_float": [float(value) for value in exact],
        "parity_or_legacy_point_phi": parity_desc,
        "strict_interval_phi": strict_desc,
        "classification": classification,
        "minimality": "greedy zero-deletion fixed point over a seeded q=3,n=2 witness; two Phi products and one nonzero J remain",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expect", choices=("under_enclosure", "repaired"), required=True)
    args = parser.parse_args()
    payload = run(args.engine_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    expected = (
        "D6_CONCRETE_UNDERENCLOSURE_WITNESS_FOUND"
        if args.expect == "under_enclosure"
        else "D6_REPAIRED_STRICT_CONTAINS_MINIMAL_WITNESS"
    )
    print(json.dumps({"classification": payload["classification"], "expected": expected}, sort_keys=True))
    return 0 if payload["classification"] == expected else 1


if __name__ == "__main__":
    raise SystemExit(main())
