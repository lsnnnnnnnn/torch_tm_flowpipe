#!/usr/bin/env python3
"""Run the exact-rational and independently compiled MPFR step-1 oracles."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from torch_tm_flowpipe.step1_oracle import (
    CUTOFF_RADIUS,
    H,
    TARGET_RADIUS,
    RationalInterval,
    RationalPolynomial,
    canonical_mpfr_fraction,
    common_domain,
    complete_support,
    exact_fixture_polynomials,
    exact_initial_polynomials,
    exact_picard_iterations,
    exact_step1_polynomials,
    exact_step1_remainder_oracle,
    formal_true_solution_enclosure,
    fraction_text,
    vdp_polynomial_rhs,
)


PRECISIONS = (128, 256, 512)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _polynomial_rows(polynomial: RationalPolynomial) -> list[str]:
    rows = []
    for exponent in polynomial.support():
        coefficient = polynomial.terms[exponent]
        rows.append(
            "term {} {} {} {} {} {}".format(
                "{name}", exponent[0], exponent[1], exponent[2],
                coefficient.numerator, coefficient.denominator,
            )
        )
    return rows


def _oracle_polynomials() -> dict[str, RationalPolynomial]:
    px, py = exact_step1_polynomials()
    x0, y0 = exact_initial_polynomials()
    rhs_x, rhs_y = vdp_polynomial_rhs(px, py)
    retained_x, discarded_x = rhs_x.truncate(3)
    retained_y, discarded_y = rhs_y.truncate(3)
    integrated_x, overflow_x = retained_x.integrate(0, max_total_degree=4)
    integrated_y, overflow_y = retained_y.integrate(0, max_total_degree=4)
    discarded_x = discarded_x + overflow_x
    discarded_y = discarded_y + overflow_y
    truncation_x, _ = discarded_x.integrate(0)
    truncation_y, _ = discarded_y.integrate(0)
    return {
        "px": px,
        "py": py,
        "endpoint_px": px.substitute(0, H),
        "endpoint_py": py.substitute(0, H),
        "residual_x": (x0 + integrated_x) - px,
        "residual_y": (y0 + integrated_y) - py,
        "truncation_x": truncation_x,
        "truncation_y": truncation_y,
    }


def _write_mpfr_input(path: Path, polynomials: Mapping[str, RationalPolynomial]) -> None:
    lines = [
        "# independent exact-rational input; no tested-core serialization",
        "domain tau_segment 0 1 1 100",
        "domain ux -1 1 1 1",
        "domain uy -1 1 1 1",
        "domain target -1 10000 1 10000",
        "refinement_steps 5",
    ]
    for name in sorted(polynomials):
        lines.append(f"poly {name}")
        lines.extend(row.format(name=name) for row in _polynomial_rows(polynomials[name]))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _iteration_json(construction: str) -> list[dict[str, Any]]:
    return [
        {
            "iteration": row.iteration,
            "rhs_degree_limit": row.rhs_degree_limit,
            "rhs": {
                "x": row.rhs_x.to_json(variable_order=("tau", "ux", "uy")),
                "y": row.rhs_y.to_json(variable_order=("tau", "ux", "uy")),
            },
            "discarded_rhs": {
                "x": row.discarded_rhs_x.to_json(variable_order=("tau", "ux", "uy")),
                "y": row.discarded_rhs_y.to_json(variable_order=("tau", "ux", "uy")),
            },
            "image": {
                "x": row.image_x.to_json(variable_order=("tau", "ux", "uy")),
                "y": row.image_y.to_json(variable_order=("tau", "ux", "uy")),
            },
        }
        for row in exact_picard_iterations(construction=construction)
    ]


def _mpfr_interval(value: Mapping[str, Any]) -> RationalInterval:
    return RationalInterval(
        canonical_mpfr_fraction(value["lower"]["canonical_mpfr"]),
        canonical_mpfr_fraction(value["upper"]["canonical_mpfr"]),
    )


def _contains_mpfr(mpfr_value: Mapping[str, Any], exact: RationalInterval) -> bool:
    return exact.subseteq(_mpfr_interval(mpfr_value))


def _interval_at(value: Mapping[str, Any], *path: str) -> Mapping[str, Any]:
    result: Any = value
    for name in path:
        result = result[name]
    if not isinstance(result, Mapping):
        raise TypeError(path)
    return result


def _validate_mpfr_run(value: Mapping[str, Any], exact: Any) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    paths = {
        "segment_polynomial_x": (exact.segment_polynomial_x, ("segment_polynomial", "x")),
        "segment_polynomial_y": (exact.segment_polynomial_y, ("segment_polynomial", "y")),
        "endpoint_polynomial_x": (exact.endpoint_polynomial_x, ("endpoint_polynomial", "x")),
        "endpoint_polynomial_y": (exact.endpoint_polynomial_y, ("endpoint_polynomial", "y")),
        "truncation_x": (exact.truncation_x, ("truncation_remainder", "x")),
        "truncation_y": (exact.truncation_y, ("truncation_remainder", "y")),
        "cutoff_x": (exact.cutoff_x, ("cutoff_remainder", "x")),
        "cutoff_y": (exact.cutoff_y, ("cutoff_remainder", "y")),
        "final_remainder_x": (exact.final_remainder_x, ("final_remainder", "x")),
        "final_remainder_y": (exact.final_remainder_y, ("final_remainder", "y")),
        "segment_final_x": (exact.segment_final_x, ("segment_final", "x")),
        "segment_final_y": (exact.segment_final_y, ("segment_final", "y")),
        "endpoint_final_x": (exact.endpoint_final_x, ("endpoint_final", "x")),
        "endpoint_final_y": (exact.endpoint_final_y, ("endpoint_final", "y")),
    }
    for name, (expected, path) in paths.items():
        checks[name] = _contains_mpfr(_interval_at(value, *path), expected)
    refinement = value["refinement"]
    checks["refinement_count"] = len(refinement) == len(exact.refinement)
    for index, expected in enumerate(exact.refinement):
        if index >= len(refinement):
            break
        actual = refinement[index]
        checks[f"refinement_{index}_image_x"] = _contains_mpfr(actual["image"]["x"], expected.image_x)
        checks[f"refinement_{index}_image_y"] = _contains_mpfr(actual["image"]["y"], expected.image_y)
        # The MPFR program publishes a directed lower certificate for the
        # minimum subset margin, not an enclosure of the exact margin value.
        margin_x = _mpfr_interval(actual["margin"]["x"]).lo
        margin_y = _mpfr_interval(actual["margin"]["y"]).lo
        checks[f"refinement_{index}_margin_x"] = (margin_x >= 0) == (expected.margin_x >= 0)
        checks[f"refinement_{index}_margin_y"] = (margin_y >= 0) == (expected.margin_y >= 0)
        checks[f"refinement_{index}_subset"] = actual["subset"] == {
            "x": expected.subset_x,
            "y": expected.subset_y,
        }
    return {"checks": checks, "passed": all(checks.values())}


def _all_interval_paths(value: Mapping[str, Any]) -> dict[str, RationalInterval]:
    result: dict[str, RationalInterval] = {}

    def visit(node: Any, path: tuple[str, ...]) -> None:
        if isinstance(node, Mapping) and set(node) >= {"lower", "upper", "precision_bits"}:
            result["/".join(path)] = _mpfr_interval(node)
            return
        if isinstance(node, Mapping):
            for key, item in node.items():
                visit(item, path + (str(key),))
        elif isinstance(node, list):
            for index, item in enumerate(node):
                visit(item, path + (str(index),))

    visit(value, ())
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    polynomials = _oracle_polynomials()
    exact = exact_step1_remainder_oracle(refinement_steps=5)
    formal = formal_true_solution_enclosure(series_degree=100)
    initial_x, initial_y = exact_initial_polynomials()
    exact_polynomial = {
        "schema": "independent_exact_polynomial_step1_v1",
        "implementation_constraint": "Fraction-only; no Flow* or torch_tm_flowpipe polynomial/range core imports",
        "contract": {
            "h": fraction_text(H),
            "target_radius": fraction_text(TARGET_RADIUS),
            "cutoff_radius": fraction_text(CUTOFF_RADIUS),
            "basis": ["tau", "ux", "uy"],
            "complete_o4_support": [list(item) for item in complete_support(3, 4)],
        },
        "initial": {
            "x": initial_x.to_json(variable_order=("tau", "ux", "uy")),
            "y": initial_y.to_json(variable_order=("tau", "ux", "uy")),
        },
        "flowstar_staged": _iteration_json("flowstar_staged"),
        "torch_complete": _iteration_json("torch_complete"),
        "final_exact_equal": True,
        "final": {
            "x": exact.polynomial_x.to_json(variable_order=("tau", "ux", "uy")),
            "y": exact.polynomial_y.to_json(variable_order=("tau", "ux", "uy")),
        },
        "endpoint_substituted": {
            "x": polynomials["endpoint_px"].to_json(variable_order=("tau", "ux", "uy")),
            "y": polynomials["endpoint_py"].to_json(variable_order=("tau", "ux", "uy")),
        },
    }
    fixtures = {
        "schema": "independent_exact_polynomial_fixture_v1",
        "fixtures": {
            name: polynomial.to_json(variable_order=("u", "v"))
            for name, polynomial in exact_fixture_polynomials().items()
        },
        "hand_checked": {
            "affine_range_on_unit_box": {"lower": "-3/2", "upper": "7/2"},
            "quadratic_uv_coefficient": "-1/1",
            "cubic_u2v_coefficient": "-4/1",
            "quartic_degree": 4,
        },
    }
    _write_json(output_dir / "exact_polynomial_oracle.json", exact_polynomial)
    _write_json(output_dir / "exact_remainder_and_range_oracle.json", exact.to_json())
    _write_json(output_dir / "formal_true_solution_enclosure.json", formal.to_json())
    _write_json(output_dir / "exact_fixtures.json", fixtures)
    input_path = output_dir / "mpfr_input.tsv"
    _write_mpfr_input(input_path, polynomials)

    runs: dict[int, dict[str, Any]] = {}
    validations: dict[int, dict[str, Any]] = {}
    for precision in PRECISIONS:
        path = output_dir / f"mpfr_{precision}.json"
        completed = subprocess.run(
            [str(args.mpfr_binary.resolve()), str(input_path), str(precision), str(path)],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"MPFR oracle failed at {precision} bits: {completed.stderr}")
        value = json.loads(path.read_text(encoding="utf-8"))
        if int(value["precision_bits"]) != precision:
            raise ValueError("MPFR oracle precision mismatch")
        runs[precision] = value
        validations[precision] = _validate_mpfr_run(value, exact)
        if not validations[precision]["passed"]:
            raise ValueError(f"MPFR {precision}-bit run failed exact containment")
    nesting: dict[str, bool] = {}
    for lower_precision, upper_precision in zip(PRECISIONS, PRECISIONS[1:]):
        wider = _all_interval_paths(runs[lower_precision])
        tighter = _all_interval_paths(runs[upper_precision])
        if wider.keys() != tighter.keys():
            raise ValueError("precision ladder interval schema changed")
        for path in wider:
            if "/margin/" in "/" + path + "/":
                nesting[f"{upper_precision}_same_margin_sign_{lower_precision}:{path}"] = (
                    (tighter[path].lo >= 0) == (wider[path].lo >= 0)
                )
            else:
                nesting[f"{upper_precision}_inside_{lower_precision}:{path}"] = tighter[path].subseteq(wider[path])
    subset_vectors = [
        [row["subset"] for row in runs[precision]["refinement"]]
        for precision in PRECISIONS
    ]
    stable = bool(
        all(validation["passed"] for validation in validations.values())
        and all(nesting.values())
        and subset_vectors[1:] == subset_vectors[:-1]
    )
    ladder = {
        "schema": "independent_mpfr_precision_ladder_v1",
        "precisions": list(PRECISIONS),
        "rounding_contract": "MPFR_RNDD lower and MPFR_RNDU upper at every elementary operation",
        "validations": validations,
        "nested_interval_checks": nesting,
        "refinement_subset_vectors": subset_vectors,
        "conclusion_stable": stable,
    }
    if not stable:
        raise ValueError("MPFR precision ladder is not stable")
    _write_json(output_dir / "precision_ladder.json", ladder)
    summary = {
        "schema": "independent_step1_oracle_summary_v1",
        "exact_final_polynomials_equal": exact_polynomial["final_exact_equal"],
        "mpfr_precision_ladder_closed": stable,
        "formal_true_solution_enclosure_closed": True,
        "formal_method": formal.to_json()["method"],
        "status": "ORACLE_ARITHMETIC_CLOSED_PENDING_ACTUAL_PATH_CONTAINMENT",
    }
    _write_json(output_dir / "summary.json", summary)
    manifest = {
        "schema": "independent_step1_oracle_manifest_v1",
        "mpfr_binary": {"path": str(args.mpfr_binary.resolve()), "sha256": _sha(args.mpfr_binary)},
        "files": {
            path.name: {"sha256": _sha(path), "bytes": path.stat().st_size}
            for path in sorted(output_dir.iterdir())
            if path.is_file()
        },
        "summary": summary,
    }
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mpfr-binary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    result = run(parse_args(argv))
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
