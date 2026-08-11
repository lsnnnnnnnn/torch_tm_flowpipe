#!/usr/bin/env python3
"""Independently replay one R35 discarded monomial with MPFR rounding.

The fixture is deliberately bounded: it qualifies the R35 overflow route for
one exact-binary64 workload.  It does not turn the ordinary-float64 descriptor
lane into a generally outward-rounded implementation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from torch_tm_flowpipe.fixed_support import FixedSupportDescriptor, FixedSupportPolynomial


ORACLE_CPP = ROOT / "experiments" / "raw_remainder_mpfr_oracle.cpp"
RESULT_PATTERN = re.compile(
    r"^ORACLE_RESULT node=(\S+) precision_bits=(\d+) input_semantics=(\S+) "
    r"rounding=(\S+) lo_decimal=(\S+) lo_hex=(\S+) hi_decimal=(\S+) "
    r"hi_hex=(\S+)$"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _expand(value: float, toward: float, count: int) -> float:
    for _ in range(int(count)):
        value = math.nextafter(value, toward)
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)

    support = FixedSupportDescriptor.complete_total_degree(
        variable_names=("t", "xi0", "xi1"), order=4, local_time_index=0
    )
    left = torch.zeros((1, 1, support.num_slots), dtype=torch.float64)
    right = torch.zeros_like(left)
    left_value = float.fromhex("0x1.999999999999ap-4")  # binary64 0.1
    right_value = float.fromhex("0x1.3333333333333p-2")  # binary64 0.3
    left_exp = (0, 3, 0)
    right_exp = (0, 2, 0)
    product_exp = tuple(a + b for a, b in zip(left_exp, right_exp))
    left[..., support.slot(left_exp)] = left_value
    right[..., support.slot(right_exp)] = right_value
    box_lo = torch.tensor([[0.0, -1.0, -0.25]], dtype=torch.float64)
    box_hi = torch.tensor([[0.01, 1.0, 0.5]], dtype=torch.float64)
    _, ledger = FixedSupportPolynomial(left, support).mul_ctrunc(
        FixedSupportPolynomial(right, support), box_lo, box_hi
    )
    ordinary = ledger.as_dict()["discarded_product_monomials"]
    ordinary_lo = float(ordinary.lo[0, 0])
    ordinary_hi = float(ordinary.hi[0, 0])

    dag = output / "r35_overflow_interval_dag.tsv"
    dag.write_text(
        "\n".join(
            (
                "# exact-binary64 inputs; xi0^5 has natural interval [-1,1]",
                f"literal left_coefficient {left_value.hex()} {left_value.hex()}",
                f"literal right_coefficient {right_value.hex()} {right_value.hex()}",
                f"literal xi0_fifth {float(-1.0).hex()} {float(1.0).hex()}",
                "mul coefficient_product left_coefficient right_coefficient",
                "mul discarded_monomial coefficient_product xi0_fifth",
                "emit discarded_monomial",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    executable = output / "raw_remainder_mpfr_oracle"
    compile_command = [
        args.compiler,
        "-O2",
        "-std=c++11",
        str(ORACLE_CPP),
        "-lmpfr",
        "-lgmp",
        "-o",
        str(executable),
    ]
    compiled = subprocess.run(
        compile_command, cwd=ROOT, text=True, capture_output=True, check=False
    )
    (output / "mpfr_compile.stdout.log").write_text(compiled.stdout, encoding="utf-8")
    (output / "mpfr_compile.stderr.log").write_text(compiled.stderr, encoding="utf-8")
    if compiled.returncode != 0:
        raise RuntimeError("MPFR oracle compilation failed")
    completed = subprocess.run(
        [str(executable), str(dag)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    (output / "mpfr_oracle.stdout.log").write_text(completed.stdout, encoding="utf-8")
    (output / "mpfr_oracle.stderr.log").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError("MPFR oracle execution failed")
    match = next(
        (RESULT_PATTERN.match(line) for line in completed.stdout.splitlines() if RESULT_PATTERN.match(line)),
        None,
    )
    if match is None:
        raise RuntimeError("MPFR oracle did not emit the requested node")
    node, precision, semantics, rounding, lo_decimal, lo_hex, hi_decimal, hi_hex = match.groups()
    oracle_lo = float.fromhex(lo_hex)
    oracle_hi = float.fromhex(hi_hex)
    expanded_lo = _expand(ordinary_lo, -math.inf, 2)
    expanded_hi = _expand(ordinary_hi, math.inf, 2)
    direct = ordinary_lo <= oracle_lo and ordinary_hi >= oracle_hi
    expanded = expanded_lo <= oracle_lo and expanded_hi >= oracle_hi
    if node != "discarded_monomial" or not expanded:
        raise RuntimeError("two-ULP companion envelope failed MPFR containment")

    report = {
        "schema": "r35_mpfr_outward_remainder_replay_v1",
        "outcome": "R35_BOUNDED_MPFR_REPLAY_PASS",
        "support": {
            "name": support.name,
            "slot_count": support.num_slots,
            "sha256": support.support_sha256,
        },
        "fixture": {
            "left_exponent": list(left_exp),
            "right_exponent": list(right_exp),
            "product_exponent": list(product_exp),
            "product_is_overflow": product_exp not in support.exponents,
            "left_coefficient_hex": left_value.hex(),
            "right_coefficient_hex": right_value.hex(),
            "box_lo_hex": [value.hex() for value in box_lo[0].tolist()],
            "box_hi_hex": [value.hex() for value in box_hi[0].tolist()],
        },
        "ordinary_binary64_remainder": {
            "lo_decimal": ordinary_lo,
            "lo_hex": ordinary_lo.hex(),
            "hi_decimal": ordinary_hi,
            "hi_hex": ordinary_hi.hex(),
            "directly_contains_mpfr": direct,
        },
        "two_ulp_companion_envelope": {
            "lo_decimal": expanded_lo,
            "lo_hex": expanded_lo.hex(),
            "hi_decimal": expanded_hi,
            "hi_hex": expanded_hi.hex(),
            "contains_mpfr": expanded,
        },
        "mpfr_oracle": {
            "node": node,
            "precision_bits": int(precision),
            "input_semantics": semantics,
            "rounding": rounding,
            "lo_decimal": lo_decimal,
            "lo_hex": lo_hex,
            "hi_decimal": hi_decimal,
            "hi_hex": hi_hex,
            "source": str(ORACLE_CPP.relative_to(ROOT)),
            "source_sha256": _sha256(ORACLE_CPP),
            "input_sha256": _sha256(dag),
            "executable_sha256": _sha256(executable),
        },
        "scope": "one deterministic R35 discarded-product monomial with exact binary64 inputs",
        "does_not_prove": "universal outward soundness of ordinary fixed-support arithmetic",
    }
    _write_json(output / "summary.json", report)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--compiler", default="g++")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    print(json.dumps(run(parse_args(argv)), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
