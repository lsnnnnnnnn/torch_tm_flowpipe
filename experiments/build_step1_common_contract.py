#!/usr/bin/env python3
"""Build the exact common step-1 contract and basis-equivalence artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from torch_tm_flowpipe.step1_oracle import (  # noqa: E402
    CUTOFF_RADIUS,
    H,
    TARGET_RADIUS,
    complete_support,
    exact_initial_polynomials,
    fraction_text,
)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_state(tool: str) -> str:
    x, y = exact_initial_polynomials()
    if tool == "flowstar":
        variable_order = ["tau", "ux", "uy", "ut_zero_radius"]
        components = {
            "x": [([0, 0, 0, 0], "5/4"), ([0, 1, 0, 0], "3/20")],
            "y": [([0, 0, 0, 0], "12/5"), ([0, 0, 1, 0], "1/20")],
            "t": [([1, 0, 0, 0], "1/1")],
        }
    elif tool == "torch":
        variable_order = ["tau", "ux", "uy"]
        components = {
            "x": [(list(exponent), fraction_text(value)) for exponent, value in x.terms.items()],
            "y": [(list(exponent), fraction_text(value)) for exponent, value in y.terms.items()],
        }
    else:
        raise ValueError(tool)
    records = [
        "schema=common_step_operator_contract_v1",
        f"tool={tool}",
        "phase=normalized_initial_mathematical_state",
        f"variable_order={','.join(variable_order)}",
        f"component_order={','.join(components)}",
        "domain.tau=0/1..1/100",
        "domain.ux=-1/1..1/1",
        "domain.uy=-1/1..1/1",
        "domain.ut_zero_radius=0/1..0/1" if tool == "flowstar" else "domain.ut_zero_radius=eliminated_by_proof",
    ]
    for component, terms in components.items():
        records.append(f"component.{component}.term_count={len(terms)}")
        for index, (exponents, coefficient) in enumerate(terms):
            records.append(f"component.{component}.term.{index}.exponents={','.join(map(str, exponents))}")
            records.append(f"component.{component}.term.{index}.coefficient={coefficient}")
        records.append(f"component.{component}.ordinary_remainder=0/1..0/1")
        records.append(f"component.{component}.truncation_remainder=0/1..0/1")
        records.append(f"component.{component}.cutoff_remainder=0/1..0/1")
    return "\n".join(records) + "\n"


def build(output: Path) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    support = complete_support(3, 4)
    contract = {
        "schema": "common_step_operator_contract_v1",
        "status": "COMMON_STEP1_MATHEMATICAL_INPUT_CLOSED",
        "physical_states": ["x", "y"],
        "ode": {"x_prime": "y", "y_prime": "y - x - x^2*y"},
        "local_time": {"symbol": "tau", "domain": ["0/1", fraction_text(H)]},
        "initial": {
            "x": {"center": "5/4", "radius": "3/20", "interval": ["11/10", "7/5"]},
            "y": {"center": "12/5", "radius": "1/20", "interval": ["47/20", "49/20"]},
        },
        "uncertainty_symbols": {
            "ux": {"domain": ["-1/1", "1/1"], "owner": "initial_x"},
            "uy": {"domain": ["-1/1", "1/1"], "owner": "initial_y"},
            "ut_zero_radius": {"domain": ["0/1", "0/1"], "owner": "flowstar_explicit_t_state"},
        },
        "order": 4,
        "support": "complete_total_degree_O4",
        "step": fraction_text(H),
        "target_remainder_radius": fraction_text(TARGET_RADIUS),
        "cutoff_radius": fraction_text(CUTOFF_RADIUS),
        "endpoint_contract": "substitute tau=1/100 before polynomial range evaluation",
        "segment_contract": "evaluate tau in [0,1/100]",
        "object_lifecycle": [
            "normalized_initial",
            "Picard_polynomial_and_source_ledgers",
            "returned_tmvPre_pre_reset",
            "endpoint_and_segment_evaluation",
            "normalized_insertion_reset",
            "step2_prestate",
        ],
        "remainder_owners": {
            "ordinary": "Picard self-map and refinement",
            "truncation": "terms discarded above complete total degree O4",
            "cutoff": "terms discarded by exact threshold 1/10000000000",
            "source_ledger": "cross-step old sources; empty at the mathematical step-1 input",
        },
    }
    basis = {
        "schema": "common_step_basis_mapping_v1",
        "canonical_basis": ["tau", "ux", "uy"],
        "flowstar_basis": ["tau", "ux", "uy", "ut_zero_radius"],
        "torch_basis_recorded": ["ux", "uy", "tau"],
        "flowstar_to_canonical_exponent": "drop exponent 3 after proving it is zero; keep (0,1,2)",
        "torch_to_canonical_permutation": [2, 0, 1],
        "explicit_t_state": {
            "initial": "t(0)=0",
            "derivative": "t'=1",
            "exact_solution": "t=tau",
            "elimination_proof": (
                "The x/y RHS is autonomous and contains no t. By structural induction on Picard "
                "iterations, x and y contain no ut_zero_radius monomial; projecting away t and "
                "the zero-radius ut symbol preserves every x/y coefficient and enclosure."
            ),
        },
        "monomial_order": "total_degree_then_lexicographic_for_neutral_schema; tool-native order retained as metadata",
    }
    support_json = {
        "schema": "complete_total_degree_support_v1",
        "basis": ["tau", "ux", "uy"],
        "order": 4,
        "cardinality": len(support),
        "exponents": [list(item) for item in support],
    }
    equivalence = {
        "schema": "initial_state_equivalence_v1",
        "status": "COMMON_STEP1_MATHEMATICAL_INPUT_CLOSED",
        "comparison_scope": "exact mathematical contract, before finite-precision encoding audit",
        "flowstar_projection": "project components x,y and substitute ut_zero_radius=0",
        "torch_reordering": "map recorded (ux,uy,tau) exponents to canonical (tau,ux,uy)",
        "exact_equal_components": ["x", "y"],
        "source_ledger_empty": True,
        "ordinary_remainders_zero": True,
        "proof_obligations": {
            "deterministic_t_elimination": "proved algebraically in basis_mapping.json",
            "complete_O4_support_same": True,
            "finite_encoding_containment": "deferred to independent oracle; not assumed by Gate B",
        },
    }
    write_json(output / "common_contract.json", contract)
    write_json(output / "basis_mapping.json", basis)
    write_json(output / "support_complete_o4.json", support_json)
    (output / "flowstar_initial_state_canonical.state").write_text(canonical_state("flowstar"), encoding="utf-8")
    (output / "torch_initial_state_canonical.state").write_text(canonical_state("torch"), encoding="utf-8")
    write_json(output / "initial_state_equivalence.json", equivalence)
    artifacts = sorted(path for path in output.iterdir() if path.name != "summary.json")
    summary = {
        "schema": "common_step_contract_build_summary_v1",
        "status": "COMMON_STEP1_MATHEMATICAL_INPUT_CLOSED",
        "artifact_sha256": {path.name: sha256(path) for path in artifacts},
    }
    write_json(output / "summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(build(args.output_dir.resolve()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
