#!/usr/bin/env python3
"""Execute every eligible Gate-E same-prestate cell and fail closed on mismatch."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from torch_tm_flowpipe.lossless_state_queue_schema import parse_file


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON object required: {path}")
    return value


def run(binary: Path, *args: str) -> dict[str, Any]:
    completed = subprocess.run(
        [str(binary), *args], text=True, capture_output=True, check=False
    )
    return {
        "argv": [str(binary), *args],
        "exit_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def first_torch_row(path: Path) -> Mapping[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return next(csv.DictReader(handle))


def audit(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    bridge = args.bridge_binary.resolve()
    fixtures = args.flowstar_fixtures.resolve()
    bridge_summary = read_json(fixtures / "summary.json")
    cross_summary = read_json(args.cross_language_summary.resolve())
    if (
        bridge_summary.get("status") != "SAME_PRESTATE_LOSSLESS_BRIDGE_AVAILABLE"
        or bridge_summary.get("canonical_byte_roundtrips_exact")
        != bridge_summary.get("fixture_count")
        or bridge_summary.get("next_step_roundtrips_exact")
        != bridge_summary.get("fixture_count")
        or cross_summary.get("status") != "SAME_PRESTATE_LOSSLESS_BRIDGE_AVAILABLE"
        or cross_summary.get("torch_flowstar_roundtrip_byte_exact") is not True
    ):
        raise RuntimeError("Gate D was not fully closed")

    flow_continuations: list[dict[str, Any]] = []
    for source_step, expected_step in ((1, 2), (99, 100), (100, 101)):
        source = fixtures / f"step_{source_step}_pre_reset.state"
        expected = fixtures / f"step_{expected_step}_pre_reset.state"
        actual = output / f"flowstar_step_{source_step}_to_{expected_step}.state"
        result = run(bridge, "continue", str(source), str(actual))
        exact = result["exit_code"] == 0 and actual.read_bytes() == expected.read_bytes()
        flow_continuations.append(
            {
                "source_step": source_step,
                "expected_step": expected_step,
                "crosses_q100_reset": source_step == 100,
                "run": result,
                "source_sha256": sha256(source),
                "expected_sha256": sha256(expected),
                "actual_sha256": sha256(actual) if actual.is_file() else None,
                "canonical_next_state_exact": exact,
            }
        )
        if not exact:
            raise RuntimeError(f"Flow* lossless continuation mismatch at step {source_step}")

    torch_state = args.torch_state.resolve()
    torch_records = parse_file(torch_state)
    torch_to_flow_output = output / "torch_state_flowstar_continuation.state"
    torch_to_flow = run(bridge, "continue", str(torch_state), str(torch_to_flow_output))
    expected_flowstar_refusal = (
        torch_to_flow["exit_code"] != 0
        and "schema/operator mismatch" in torch_to_flow["stderr"]
        and not torch_to_flow_output.exists()
    )
    if not expected_flowstar_refusal:
        raise RuntimeError("Flow* did not fail closed on the incompatible Torch state")

    flow_records = parse_file(fixtures / "step_1_pre_reset.state")
    flow_state_dimension = int(flow_records["state_dimension"])
    flow_variable_dimension = int(flow_records["variable_dimension"])
    flow_j_count = int(flow_records["queue.J_count"])
    flow_phi_count = int(flow_records["queue.Phi_L_count"])
    torch_state_dimension = int(torch_records["state_dimension"])
    torch_variable_dimension = int(torch_records["variable_dimension"])
    torch_j_count = int(torch_records["queue.J_count"])
    torch_phi_count = int(torch_records["queue.Phi_L_count"])
    torch_can_consume_flowstar_full_state = (
        flow_state_dimension == torch_state_dimension
        and flow_variable_dimension == torch_variable_dimension
        and flow_j_count == torch_j_count == 0
        and flow_phi_count == torch_phi_count == 0
    )
    if torch_can_consume_flowstar_full_state:
        raise RuntimeError("unexpected full-state compatibility; explicit Torch replay required")

    first_torch = first_torch_row(args.torch_segments.resolve())
    torch_native_accepted = first_torch.get("status") == "accepted"
    if not torch_native_accepted:
        raise RuntimeError("native Torch-on-Torch initial-step evidence is missing")

    matrix = [
        {
            "operator": "Flowstar",
            "prestate": "Flowstar",
            "eligibility": "ELIGIBLE_EXECUTED",
            "result": "EXACT_NEXT_STATE_AT_1_99_100",
            "raw": flow_continuations,
        },
        {
            "operator": "Torch",
            "prestate": "Flowstar",
            "eligibility": "INELIGIBLE_SCHEMA_OPERATOR_MISMATCH",
            "result": "NOT_RUN_FULL_QUEUE_CANNOT_BE_CONSUMED",
            "reason": (
                "The Torch operator has two state components and two normalized variables, "
                "and has no lossless consumer for Flow*'s x/y/t state, four-variable TM, or "
                "nonempty Phi_L/J queue. Dropping these objects is forbidden."
            ),
        },
        {
            "operator": "Flowstar",
            "prestate": "Torch",
            "eligibility": "INELIGIBLE_SCHEMA_OPERATOR_MISMATCH",
            "result": "EXECUTED_REFUSAL",
            "raw": torch_to_flow,
        },
        {
            "operator": "Torch",
            "prestate": "Torch",
            "eligibility": "ELIGIBLE_EXECUTED_NATIVE_DIAGNOSTIC",
            "result": "INITIAL_STEP_ACCEPTED",
            "source_status": first_torch.get("status"),
            "source_endpoint_widths": {
                "x": first_torch.get("endpoint_x_width"),
                "y": first_torch.get("endpoint_y_width"),
            },
        },
    ]
    result = {
        "schema": "flowstar_torch_same_prestate_operator_matrix_v1",
        "gate_d_lossless_serialization_roundtrip_closed": True,
        "dimensions": {
            "flowstar": {
                "state": flow_state_dimension,
                "variables": flow_variable_dimension,
                "J_step1": flow_j_count,
                "Phi_L_step1": flow_phi_count,
            },
            "torch": {
                "state": torch_state_dimension,
                "variables": torch_variable_dimension,
                "J": torch_j_count,
                "Phi_L": torch_phi_count,
            },
        },
        "matrix": matrix,
        "common_box_reboxing": False,
        "queue_dropped": False,
        "full_two_by_two_same_prestate_executed": False,
        "operator_attribution_closed": False,
        "first_bitwise_difference": "step_1_returned_raw_picard_coefficients",
        "first_beyond_roundoff_difference": (
            "step_1_published_endpoint_and_segment_widths; exact operator-stage boundary "
            "remains unresolved without the two cross-operator cells"
        ),
        "first_persistent_widening": (
            "cross-tool history is already different after step 1; no full-state replay "
            "separates incoming history from the local operator"
        ),
        "first_decision_relevant_difference": (
            "the accumulated history/operator difference makes Torch candidate 633 fail; "
            "no unique source line is causally closed"
        ),
        "status": "SOURCE_MECHANISM_CANDIDATES_LOCALIZED_CAUSAL_SPLIT_OPEN",
        "downstream_authorization": "NO_FIX_AUTHORIZED",
    }
    write_json(output / "summary.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge-binary", type=Path, required=True)
    parser.add_argument("--flowstar-fixtures", type=Path, required=True)
    parser.add_argument("--cross-language-summary", type=Path, required=True)
    parser.add_argument("--torch-state", type=Path, required=True)
    parser.add_argument("--torch-segments", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(audit(parse_args()), sort_keys=True, allow_nan=False))
