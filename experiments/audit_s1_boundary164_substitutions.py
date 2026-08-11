#!/usr/bin/env python3
"""Run the frozen boundary-164 real-prestate and component substitutions."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from torch_tm_flowpipe.batched_dense_tm import DenseRangePolicy
from torch_tm_flowpipe.flowpipe import (
    FlowstarNormalFlowpipeState,
    _tmvector_remainder_tensor,
    _tmvector_with_remainder_tensor,
    _tmvector_without_remainder,
)
from torch_tm_flowpipe.s1_boundary_attribution import tensor_hex
from torch_tm_flowpipe.structured_remainder import (
    StructuredRemainderState,
    materialize_structured_remainder,
)
from torch_tm_flowpipe.terminal_checkpoint import (
    TerminalCheckpoint,
    _encode_normal_state,
    _encode_tmvector,
    load_terminal_checkpoint,
    tmvector_hashes,
)


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "experiments/run_s1_prefix_complete_o4.py"
RUNNER_SPEC = importlib.util.spec_from_file_location(
    "run_s1_prefix_for_boundary164_substitutions",
    RUNNER_PATH,
)
assert RUNNER_SPEC is not None and RUNNER_SPEC.loader is not None
runner = importlib.util.module_from_spec(RUNNER_SPEC)
RUNNER_SPEC.loader.exec_module(runner)

EXPECTED_TIME = 4.738198114669049
EXPECTED_H = 0.03661680691961388
EXPECTED_Y_MARGINS = {
    "P0": 8.058292550874906e-6,
    "P1": -3.872231318094365e-6,
    "P2": -3.773875528686747e-6,
}


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")


def _state_sha(current, normal_state: FlowstarNormalFlowpipeState) -> str:
    payload = {
        "current": _encode_tmvector(current),
        "normal_state": _encode_normal_state(normal_state),
    }
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _policy() -> DenseRangePolicy:
    spec = runner.CONTRACT["dense_range_policy"]
    return DenseRangePolicy(
        method=spec["method"],
        max_depth=spec["max_depth"],
        max_leaves=spec["max_leaves"],
        split_vars=tuple(spec["split_vars"]),
        trigger=spec["trigger"],
        named_contexts=tuple(spec["named_contexts"]),
        variable_orders=tuple(tuple(row) for row in spec["variable_orders"]),
    )


def _load(path: Path) -> TerminalCheckpoint:
    return load_terminal_checkpoint(
        path,
        expected_contract=runner.CONTRACT,
        expected_order=4,
        expected_dtype="float64",
    )


def _total_remainder(
    normal_state: FlowstarNormalFlowpipeState,
) -> tuple[torch.Tensor, torch.Tensor]:
    structured = normal_state.structured_remainder_state
    if isinstance(structured, StructuredRemainderState):
        total = materialize_structured_remainder(structured)
        return total.lo, total.hi
    return _tmvector_remainder_tensor(normal_state.tmv_right)


def _ordinary_only(
    normal_state: FlowstarNormalFlowpipeState,
) -> FlowstarNormalFlowpipeState:
    total_lo, total_hi = _total_remainder(normal_state)
    right = _tmvector_with_remainder_tensor(
        _tmvector_without_remainder(normal_state.tmv_right),
        total_lo,
        total_hi,
    )
    return replace(
        normal_state,
        tmv_right=right,
        structured_remainder_state=None,
    )


def _with_donor_total(
    core: FlowstarNormalFlowpipeState,
    donor: FlowstarNormalFlowpipeState,
) -> FlowstarNormalFlowpipeState:
    donor_lo, donor_hi = _total_remainder(donor)
    right = _tmvector_with_remainder_tensor(
        _tmvector_without_remainder(core.tmv_right),
        donor_lo,
        donor_hi,
    )
    return replace(core, tmv_right=right, structured_remainder_state=None)


def _interval_record(lo: torch.Tensor, hi: torch.Tensor) -> dict[str, Any]:
    return {
        "units": "old normalized",
        "lo": lo.detach().cpu().tolist(),
        "hi": hi.detach().cpu().tolist(),
        "lo_hex": tensor_hex(lo),
        "hi_hex": tensor_hex(hi),
    }


def _run(
    name: str,
    *,
    lane: str,
    current,
    normal_state: FlowstarNormalFlowpipeState,
    set_relation: str,
    provenance: str,
) -> dict[str, Any]:
    before = _state_sha(current, normal_state)
    validator_input = normal_state.normalized_initial_tm(4)
    validator_input_hashes = tmvector_hashes(validator_input)
    diagnostics: list[dict[str, Any]] = []
    ode = runner.PolynomialODE.from_system_spec(
        runner.CONTRACT["canonical_system_spec"]
    )
    segment = runner._run_lane_step(
        ode,
        current,
        normal_state,
        lane=lane,
        h=EXPECTED_H,
        h_min=EXPECTED_H,
        h_max=EXPECTED_H,
        max_validation_attempts=1,
        policy=_policy(),
        diagnostics=diagnostics,
        diagnostics_context={
            "substitution": name,
            "segment_index": 164,
            "t_before": EXPECTED_TIME,
        },
    )
    after = _state_sha(current, normal_state)
    total_lo, total_hi = _total_remainder(normal_state)
    return {
        "schema": "torch_tm_flowpipe_s1_boundary164_substitution_v1",
        "name": name,
        "lane": lane,
        "provenance": provenance,
        "set_relation": set_relation,
        "diagnostic_only": name.startswith("H"),
        "time": EXPECTED_TIME,
        "time_hex": EXPECTED_TIME.hex(),
        "h": EXPECTED_H,
        "h_hex": EXPECTED_H.hex(),
        "order": 4,
        "target_remainder_radius": 1e-4,
        "cutoff_threshold": 1e-10,
        "validator": "flowstar_raw_remainder_compat",
        "max_validation_attempts": 1,
        "status": segment.status,
        "message": segment.message,
        "subset_margin": segment.subset_margin,
        "y_margin": float(segment.subset_margin[0][1]),
        "y_margin_hex": float(segment.subset_margin[0][1]).hex(),
        "center": list(normal_state.center),
        "center_hex": [float(value).hex() for value in normal_state.center],
        "scale": list(normal_state.scales),
        "scale_hex": [float(value).hex() for value in normal_state.scales],
        "right_polynomial_hashes": tmvector_hashes(
            _tmvector_without_remainder(normal_state.tmv_right)
        ),
        "materialized_total_remainder": _interval_record(total_lo, total_hi),
        "validator_input_hashes": validator_input_hashes,
        "prestate_sha256_before": before,
        "prestate_sha256_after": after,
        "prestate_unchanged": before == after,
        "first_validator_diagnostics": diagnostics,
    }


def _same_margin(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return left["y_margin_hex"] == right["y_margin_hex"]


def audit(
    l0_path: Path,
    l1_path: Path,
    l2_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    checkpoints = {
        "P0": _load(l0_path),
        "P1": _load(l1_path),
        "P2": _load(l2_path),
    }
    for name, checkpoint in checkpoints.items():
        scheduler = checkpoint.scheduler
        if (
            float(scheduler["current_time"]) != EXPECTED_TIME
            or float(scheduler["h_attempted"]) != EXPECTED_H
            or int(scheduler["accepted_segment_count"]) != 164
        ):
            raise RuntimeError(f"{name} checkpoint scheduler is not boundary-164 full-h")

    p0 = checkpoints["P0"]
    p1 = checkpoints["P1"]
    p2 = checkpoints["P2"]
    states = {
        "P0": ("L0", p0.current, p0.normal_state, "equal", "loaded L0 real prestate"),
        "P1": ("L1", p1.current, p1.normal_state, "equal", "loaded L1 real prestate"),
        "P2": ("L2", p2.current, p2.normal_state, "equal", "loaded L2 real prestate"),
        "H1": (
            "L0",
            p0.current,
            _with_donor_total(p0.normal_state, p1.normal_state),
            "diagnostic_only",
            "L0 center/scale/right polynomial plus L1 materialized total remainder",
        ),
        "H2": (
            "L0",
            p1.current,
            _with_donor_total(p1.normal_state, p0.normal_state),
            "diagnostic_only",
            "L1 center/scale/right polynomial plus L0 total remainder",
        ),
        "H3": (
            "L0",
            p1.current,
            _ordinary_only(p1.normal_state),
            "equal",
            "L1 represented remainder materialized into ordinary-only carrier without reboxing",
        ),
        "H4": (
            "L0",
            p2.current,
            _ordinary_only(p2.normal_state),
            "equal",
            "L2 represented remainder materialized into ordinary-only carrier without reboxing",
        ),
    }
    records = {
        name: _run(
            name,
            lane=lane,
            current=current,
            normal_state=normal_state,
            set_relation=relation,
            provenance=provenance,
        )
        for name, (lane, current, normal_state, relation, provenance) in states.items()
    }
    for name, expected in EXPECTED_Y_MARGINS.items():
        if records[name]["y_margin"] != expected:
            raise RuntimeError(
                f"BOUNDARY164_REPLAY_NOT_REPRODUCIBLE: {name} "
                f"{records[name]['y_margin']!r} != {expected!r}"
            )
    if not all(record["prestate_unchanged"] for record in records.values()):
        raise RuntimeError("boundary-164 diagnostic mutated a prestate")

    projection_checks = {
        "H1_matches_P0": _same_margin(records["H1"], records["P0"]),
        "H2_matches_P1": _same_margin(records["H2"], records["P1"]),
        "H3_matches_P1": _same_margin(records["H3"], records["P1"]),
        "H4_matches_P2": _same_margin(records["H4"], records["P2"]),
        "P0_H1_validator_input_equal": records["P0"]["validator_input_hashes"]
        == records["H1"]["validator_input_hashes"],
        "P1_H2_H3_validator_input_equal": records["P1"]["validator_input_hashes"]
        == records["H2"]["validator_input_hashes"]
        == records["H3"]["validator_input_hashes"],
        "P2_H4_validator_input_equal": records["P2"]["validator_input_hashes"]
        == records["H4"]["validator_input_hashes"],
    }
    if not all(projection_checks.values()):
        raise RuntimeError("boundary-164 substitution projection parity failed")

    p0_y = records["P0"]["y_margin"]
    p1_y = records["P1"]["y_margin"]
    total_loss = p1_y - p0_y
    center_equal = records["P0"]["center_hex"] == records["P1"]["center_hex"]
    if not center_equal:
        raise RuntimeError("boundary-164 center substitution requires an extra registered control")
    contributions = {
        "units": "validator y subset margin",
        "L0_to_L1_total_difference": total_loss,
        "center_contribution": 0.0,
        "scale_contribution": total_loss,
        "right_polynomial_contribution": 0.0,
        "total_remainder_contribution": 0.0,
        "validator_reduction_residual": 0.0,
        "interaction_remainder": 0.0,
        "method": (
            "exact equal centers plus exact validator-input hashes: normalized_initial_tm "
            "projects only center/scale; H1/H2/H3/H4 prove right-polynomial and "
            "remainder carrier invariance for the first validator"
        ),
    }
    summary = {
        "schema": "torch_tm_flowpipe_s1_boundary164_substitution_summary_v1",
        "checkpoint_sha256": {
            name: checkpoint.manifest["full_checkpoint_sha256"]
            for name, checkpoint in checkpoints.items()
        },
        "real_prestate_y_margins": {
            name: records[name]["y_margin"] for name in ("P0", "P1", "P2")
        },
        "hybrid_y_margins": {
            name: records[name]["y_margin"] for name in ("H1", "H2", "H3", "H4")
        },
        "projection_checks": projection_checks,
        "contributions": contributions,
        "formal_claim_use": "diagnostic_hybrids_excluded",
    }
    with (output_dir / "substitution_records.jsonl").open("w", encoding="utf-8") as handle:
        for name in ("P0", "P1", "P2", "H1", "H2", "H3", "H4"):
            handle.write(json.dumps(records[name], sort_keys=True) + "\n")
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--l0-checkpoint", type=Path, required=True)
    parser.add_argument("--l1-checkpoint", type=Path, required=True)
    parser.add_argument("--l2-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = audit(
        args.l0_checkpoint.resolve(),
        args.l1_checkpoint.resolve(),
        args.l2_checkpoint.resolve(),
        args.output_dir.resolve(),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
