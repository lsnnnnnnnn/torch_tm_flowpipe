#!/usr/bin/env python3
"""Build the five-position lossless same-prestate operator matrix for Gate A."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch

from torch_tm_flowpipe import (
    DenseRangePolicy,
    FlowstarNormalFlowpipeState,
    PolynomialODE,
    TMVector,
    flowpipe_step_flowstar_style_adaptive,
    load_terminal_checkpoint,
    save_terminal_checkpoint,
)
from torch_tm_flowpipe.lossless_state_queue_schema import (
    export_torch_normal_state,
    parse_file,
)
from torch_tm_flowpipe.terminal_checkpoint import PAYLOAD_NAME, MANIFEST_NAME

sys.path.insert(0, str(ROOT / "experiments"))
from run_vdp_dense_backend import load_contract


POSITIONS = (
    (1, "step_1"),
    (2, "step_2"),
    (99, "before_T1"),
    (299, "before_T3"),
    (631, "before_T6p32"),
)


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tm_table(tmv: TMVector | None) -> list[dict[str, Any]] | None:
    if tmv is None:
        return None
    return [
        {
            "component": component,
            "terms": [
                [list(exponent), float(coefficient.detach().cpu()).hex()]
                for exponent, coefficient in sorted(model.polynomial.terms.items())
            ],
            "ordinary_remainder": {
                "lo_hex": float(model.remainder.lo.detach().cpu()).hex(),
                "hi_hex": float(model.remainder.hi.detach().cpu()).hex(),
            },
        }
        for component, model in enumerate(tmv)
    ]


def box_rows(tmv: TMVector | None) -> list[dict[str, Any]] | None:
    if tmv is None:
        return None
    return [
        {
            "component": index,
            "lo_hex": float(interval.lo.detach().cpu()).hex(),
            "hi_hex": float(interval.hi.detach().cpu()).hex(),
            "width": float(interval.width().detach().cpu()),
        }
        for index, interval in enumerate(tmv.range_box())
    ]


def ledger_rows(segment: Any) -> list[dict[str, Any]]:
    decomposition = segment.validated_remainder_decomposition
    if decomposition is None or not bool(torch.all(decomposition.contains_image)):
        raise RuntimeError("Torch accepted step lacks complete ledger containment")
    rows: list[dict[str, Any]] = []
    for category in decomposition.ledger.category_order:
        lo, hi = decomposition.ledger.entries[category]
        for component in range(lo.shape[1]):
            lower = float(lo[0, component].detach().cpu())
            upper = float(hi[0, component].detach().cpu())
            rows.append(
                {
                    "category": category,
                    "component": component,
                    "lo_hex": lower.hex(),
                    "hi_hex": upper.hex(),
                    "width": upper - lower,
                }
            )
    return rows


def replay(ode: PolynomialODE, current: TMVector, state: FlowstarNormalFlowpipeState) -> Any:
    return flowpipe_step_flowstar_style_adaptive(
        ode,
        current,
        h=0.01,
        h_min=0.01,
        h_max=0.01,
        order=4,
        target_remainder_radius=1e-4,
        cutoff_threshold=1e-10,
        max_validation_attempts=2,
        validation_eps=1e-12,
        validation_mode="flowstar_raw_remainder_compat",
        reset_mode="normalized_insertion",
        step_policy_mode="flowstar_compat",
        flowstar_normal_state=state,
        tm_backend="dense",
        dense_range_policy=DenseRangePolicy(
            method="adaptive_subdivision",
            max_depth=1,
            max_leaves=4,
            split_vars=(0, 1),
            trigger="proactive_depth1_on_named_contexts",
            named_contexts=("polynomial_truncation",),
        ),
    )


def run_process(argv: Sequence[str]) -> dict[str, Any]:
    completed = subprocess.run(list(argv), text=True, capture_output=True, check=False)
    return {
        "argv": list(argv),
        "exit_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def write_json(path: Path, value: Any) -> None:
    path.write_bytes(canonical_bytes(value))


def audit(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    bridge = args.flowstar_bridge.resolve()
    fixtures = args.flowstar_fixtures.resolve()
    if not bridge.is_file() or not fixtures.is_dir():
        raise FileNotFoundError("Flow* bridge or fixtures missing")
    bridge_summary = json.loads((fixtures / "summary.json").read_text(encoding="utf-8"))
    if bridge_summary.get("status") != "SAME_PRESTATE_LOSSLESS_BRIDGE_AVAILABLE":
        raise RuntimeError("Flow* native fixture round-trip gate is not closed")

    contract = load_contract()
    machine_contract = json.loads(
        (ROOT / "benchmarks/vdp_g2_shared_column_contract_20260815.json").read_text(encoding="utf-8")
    )
    ode = PolynomialODE.from_system_spec(contract["canonical_system_spec"])
    normal_state = FlowstarNormalFlowpipeState.from_exact_decimal_box(
        [("1.1", "1.4"), ("2.35", "2.45")], 4
    )
    current = normal_state.normalized_initial_tm(4)
    captured: dict[int, dict[str, Any]] = {}
    target_steps = {step for step, _ in POSITIONS}
    started = time.perf_counter()
    for accepted_step in range(1, max(target_steps) + 1):
        segment = replay(ode, current, normal_state)
        if segment.status != "validated" or segment.reset_tm is None or segment.flowstar_normal_state is None:
            raise RuntimeError(f"Torch legacy prefix failed at step {accepted_step}: {segment.message}")
        current = segment.reset_tm
        normal_state = segment.flowstar_normal_state
        if accepted_step not in target_steps:
            continue
        label = dict(POSITIONS)[accepted_step]
        cell_dir = output / label
        cell_dir.mkdir(parents=True)
        checkpoint_dir = cell_dir / "torch_prestate"
        manifest = save_terminal_checkpoint(
            checkpoint_dir,
            current=current,
            normal_state=normal_state,
            scheduler={
                "current_time_decimal": format(accepted_step * 0.01, ".17g"),
                "current_time_hex": float(accepted_step * 0.01).hex(),
                "h_decimal": "0.01",
                "h_hex": float(0.01).hex(),
                "accepted_steps": accepted_step,
            },
            contract=machine_contract,
            provenance={"producer": "torch_exact_decimal_fresh", "source_head": "WORKTREE"},
        )
        roundtrip_dir = cell_dir / "torch_prestate_roundtrip"
        loaded = load_terminal_checkpoint(
            checkpoint_dir,
            expected_contract=machine_contract,
            expected_order=4,
            expected_dtype="float64",
        )
        save_terminal_checkpoint(
            roundtrip_dir,
            current=loaded.current,
            normal_state=loaded.normal_state,
            scheduler=loaded.scheduler,
            contract=loaded.contract,
            provenance=loaded.provenance,
        )
        roundtrip_equal = all(
            (checkpoint_dir / name).read_bytes() == (roundtrip_dir / name).read_bytes()
            for name in (PAYLOAD_NAME, MANIFEST_NAME)
        )
        if not roundtrip_equal:
            raise RuntimeError(f"Torch canonical checkpoint round trip differs at {label}")
        owner_payload = {
            "schema": "torch_common_prestate_separate_owner_ledgers_v1",
            "label": label,
            "time_decimal": format(accepted_step * 0.01, ".17g"),
            "time_hex": float(accepted_step * 0.01).hex(),
            "h_decimal": "0.01",
            "h_hex": float(0.01).hex(),
            "ordinary_remainder_ledger": tm_table(current),
            "complete_o4_validated_owner_ledger": ledger_rows(segment),
            "ledgers_merged": False,
            "contains_unchanged_picard_image": True,
            "validator_target": [-1e-4, 1e-4],
            "range_contract": machine_contract["range"],
        }
        write_json(cell_dir / "torch_owner_ledgers.json", owner_payload)
        torch_state_path = cell_dir / "torch_flowstar_schema.state"
        export_summary = export_torch_normal_state(
            normal_state,
            fixtures / f"step_{accepted_step}_pre_reset.state",
            torch_state_path,
            local_time=accepted_step * 0.01,
            phase="pre_reset",
        )
        torch_next = replay(ode, current, normal_state)
        torch_output = {
            "schema": "torch_same_prestate_operator_output_v1",
            "label": label,
            "status": torch_next.status,
            "message": torch_next.message,
            "canonical_coefficients": {
                "segment": tm_table(torch_next.tm),
                "endpoint": tm_table(torch_next.endpoint_raw_tm),
                "next_boundary": tm_table(torch_next.reset_tm),
            },
            "raw_picard_image": torch_next.picard_image_remainder,
            "candidate_remainder": torch_next.candidate_remainder,
            "subset_margin": torch_next.subset_margin,
            "endpoint_raw_bounds": box_rows(torch_next.endpoint_raw_tm),
            "segment_tube_raw_bounds": box_rows(torch_next.tm),
            "complete_owner_ledger": ledger_rows(torch_next) if torch_next.status == "validated" else None,
        }
        write_json(cell_dir / "torch_on_torch_output.json", torch_output)
        captured[accepted_step] = {
            "label": label,
            "checkpoint_manifest": manifest,
            "checkpoint_byte_roundtrip_equal": roundtrip_equal,
            "torch_schema_export": export_summary,
            "torch_output_sha256": sha256(cell_dir / "torch_on_torch_output.json"),
        }

    matrix_rows: list[dict[str, Any]] = []
    for step, label in POSITIONS:
        cell_dir = output / label
        source_fixture = fixtures / f"step_{step}_pre_reset.state"
        copied_fixture = cell_dir / "flowstar_prestate.state"
        shutil.copy2(source_fixture, copied_fixture)
        roundtrip = cell_dir / "flowstar_prestate.roundtrip.state"
        next_state = cell_dir / "flowstar_on_flowstar_next.state"
        roundtrip_run = run_process((str(bridge), "roundtrip", str(copied_fixture), str(roundtrip)))
        continuation_run = run_process((str(bridge), "continue", str(copied_fixture), str(next_state)))
        roundtrip_equal = roundtrip_run["exit_code"] == 0 and copied_fixture.read_bytes() == roundtrip.read_bytes()
        if not roundtrip_equal or continuation_run["exit_code"] != 0:
            raise RuntimeError(f"Flow* native cell failed at {label}")
        flow_records = parse_file(copied_fixture)
        next_records = parse_file(next_state)
        torch_state_path = cell_dir / "torch_flowstar_schema.state"
        flow_on_torch = run_process((str(bridge), "continue", str(torch_state_path), str(cell_dir / "forbidden_flowstar_on_torch.state")))
        expected_refusal = flow_on_torch["exit_code"] != 0 and not (cell_dir / "forbidden_flowstar_on_torch.state").exists()
        if not expected_refusal:
            raise RuntimeError(f"Flow* failed to refuse incompatible Torch prestate at {label}")
        matrix_rows.extend(
            [
                {
                    "label": label,
                    "position_step": step,
                    "operator": "Flowstar",
                    "prestate": "Flowstar",
                    "status": "EXECUTED_NATIVE_LOSSLESS",
                    "prestate_sha256": sha256(copied_fixture),
                    "next_state_sha256": sha256(next_state),
                    "canonical_roundtrip_equal": roundtrip_equal,
                    "state_dimension": int(flow_records["state_dimension"]),
                    "variable_dimension": int(flow_records["variable_dimension"]),
                    "J_count": int(flow_records["queue.J_count"]),
                    "Phi_L_count": int(flow_records["queue.Phi_L_count"]),
                    "next_step": int(next_records["step"]),
                    "raw_picard_and_complete_owner_ledger": "UNAVAILABLE_IN_LOSSLESS_NATIVE_STATE_EXPORT",
                },
                {
                    "label": label,
                    "position_step": step,
                    "operator": "Torch",
                    "prestate": "Torch",
                    "status": "EXECUTED_NATIVE_LOSSLESS",
                    **captured[step],
                },
                {
                    "label": label,
                    "position_step": step,
                    "operator": "Torch",
                    "prestate": "Flowstar",
                    "status": "UNAVAILABLE_LOSSLESS_CROSS_OPERATOR_CELL",
                    "reason": "Torch has no lossless consumer for Flowstar's 3-state/4-variable TM plus nonempty Phi_L/J and distinct MPFR remainder objects; projection or zero-fill is forbidden",
                },
                {
                    "label": label,
                    "position_step": step,
                    "operator": "Flowstar",
                    "prestate": "Torch",
                    "status": "EXECUTED_FAIL_CLOSED_SCHEMA_REFUSAL",
                    "expected_refusal": expected_refusal,
                    "run": flow_on_torch,
                },
            ]
        )

    conclusion = "LOSSLESS_CROSS_OPERATOR_CELL_UNAVAILABLE__TOTAL_CAUSE_OPEN"
    summary = {
        "schema": "vdp_five_position_lossless_operator_matrix_v1",
        "conclusion": conclusion,
        "positions": [label for _, label in POSITIONS],
        "position_count": len(POSITIONS),
        "matrix_cell_count": len(matrix_rows),
        "flowstar_native_cells_executed": 5,
        "torch_native_cells_executed": 5,
        "torch_on_flowstar_cells_executed": 0,
        "flowstar_on_torch_fail_closed_refusals": 5,
        "full_four_cell_matrix_available": False,
        "total_cause_closed": False,
        "common_component_box_used": False,
        "queue_dropped": False,
        "flowstar_native_roundtrip_summary": bridge_summary,
        "contract": machine_contract,
        "runtime_s": time.perf_counter() - started,
        "matrix": matrix_rows,
    }
    write_json(output / "operator_matrix.json", summary)
    print(json.dumps({"conclusion": conclusion, "runtime_s": summary["runtime_s"]}, sort_keys=True))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flowstar-bridge", type=Path, required=True)
    parser.add_argument("--flowstar-fixtures", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    audit(parse_args())
