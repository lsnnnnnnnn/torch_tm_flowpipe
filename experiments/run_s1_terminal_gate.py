#!/usr/bin/env python3
"""Run the historical terminal gate from candidate and canonical prestates."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from torch_tm_flowpipe.fixed_support_outward import OutwardIntervalTensor
from torch_tm_flowpipe.flowpipe import (
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
    _decode_normal_state,
    _decode_tmvector,
    _encode_normal_state,
    _encode_tmvector,
    load_terminal_checkpoint,
    tmvector_hashes,
)


ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = _load_module(
    "run_s1_prefix_for_terminal_gate",
    ROOT / "experiments/run_s1_prefix_complete_o4.py",
)
corrected = _load_module(
    "run_s1_corrected_for_terminal_gate",
    ROOT / "experiments/run_s1_corrected_frozen_prefix.py",
)

TERMINAL_TIME = 6.397083942944808
TERMINAL_H = 0.003623635847674574
H_MIN = 0.002


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _state_sha(current, normal_state) -> str:
    return _sha256(
        {
            "current": _encode_tmvector(current),
            "normal_state": _encode_normal_state(normal_state),
        }
    )


def _policy():
    spec = runner.CONTRACT["dense_range_policy"]
    return runner.DenseRangePolicy(
        method=spec["method"],
        max_depth=spec["max_depth"],
        max_leaves=spec["max_leaves"],
        split_vars=tuple(spec["split_vars"]),
        trigger=spec["trigger"],
        named_contexts=tuple(spec["named_contexts"]),
        variable_orders=tuple(tuple(row) for row in spec["variable_orders"]),
    )


def _materialize(normal_state):
    structured = normal_state.structured_remainder_state
    if not isinstance(structured, StructuredRemainderState):
        raise ValueError("T1 requires a native structured candidate prestate")
    total = materialize_structured_remainder(structured)
    right = _tmvector_with_remainder_tensor(
        _tmvector_without_remainder(normal_state.tmv_right),
        total.lo,
        total.hi,
    )
    return replace(
        normal_state,
        tmv_right=right,
        structured_remainder_state=None,
        diagnostics={
            **dict(normal_state.diagnostics or {}),
            "terminal_materialization": "ordinary_only_without_reboxing",
            "reset_mode": "normalized_insertion",
        },
    )


def _load_l0_snapshot(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("boundary", -1)) != 307:
        raise ValueError("historical L0 snapshot is not boundary 307")
    current = _decode_tmvector(payload["current"])
    normal_state = _decode_normal_state(
        payload["normal_state"],
        require_structured=False,
    )
    if runner._state_hash(current, normal_state) != payload["state_sha256"]:
        raise ValueError("historical L0 snapshot state hash mismatch")
    return current, normal_state, payload


def _total_record(normal_state) -> dict[str, Any]:
    structured = normal_state.structured_remainder_state
    if isinstance(structured, StructuredRemainderState):
        total = materialize_structured_remainder(structured)
        lo, hi = total.lo, total.hi
    else:
        lo, hi = _tmvector_remainder_tensor(normal_state.tmv_right)
    return {
        "units": "old normalized",
        "lo": lo.detach().cpu().tolist(),
        "hi": hi.detach().cpu().tolist(),
        "lo_hex": tensor_hex(lo),
        "hi_hex": tensor_hex(hi),
    }


def _run(name: str, lane: str, current, normal_state) -> tuple[dict[str, Any], Any]:
    before = _state_sha(current, normal_state)
    diagnostics: list[dict[str, Any]] = []
    ode = runner.PolynomialODE.from_system_spec(
        runner.CONTRACT["canonical_system_spec"]
    )
    segment = runner._run_lane_step(
        ode,
        current,
        normal_state,
        lane=lane,
        h=TERMINAL_H,
        h_min=H_MIN,
        h_max=runner.CONTRACT["h_max"],
        max_validation_attempts=2,
        policy=_policy(),
        diagnostics=diagnostics,
        diagnostics_context={
            "terminal_control": name,
            "segment_index": 307,
            "t_before": TERMINAL_TIME,
        },
    )
    after = _state_sha(current, normal_state)
    decision = (
        "accepted"
        if segment.status == "validated" and segment.reset_tm is not None
        else "rejected"
    )
    ledger = segment.validated_remainder_ledger
    typed = (
        {
            category: {
                "lo": ledger.entries[category][0].detach().cpu().tolist(),
                "hi": ledger.entries[category][1].detach().cpu().tolist(),
                "lo_hex": tensor_hex(ledger.entries[category][0]),
                "hi_hex": tensor_hex(ledger.entries[category][1]),
            }
            for category in ledger.category_order
        }
        if ledger is not None
        else {}
    )
    publication = {
        "applicable": lane == "B" and decision == "accepted",
        "endpoint": (
            bool(torch.all(segment.endpoint_publication_mask))
            if segment.endpoint_publication_mask is not None
            else None
        ),
        "tube": (
            bool(torch.all(segment.tube_publication_mask))
            if segment.tube_publication_mask is not None
            else None
        ),
        "endpoint_in_tube": (
            bool(
                (segment.flowstar_normal_stats or {}).get(
                    "structured_published_endpoint_in_tube"
                )
            )
            if lane == "B" and decision == "accepted"
            else None
        ),
    }
    candidate_gates = (
        corrected._candidate_gates(segment, 308)
        if lane == "B" and decision == "accepted"
        else None
    )
    record = {
        "schema": "torch_tm_flowpipe_s1_terminal_control_v1",
        "name": name,
        "lane": lane,
        "time": TERMINAL_TIME,
        "time_hex": TERMINAL_TIME.hex(),
        "h": TERMINAL_H,
        "h_hex": TERMINAL_H.hex(),
        "h_min": H_MIN,
        "h_min_hex": H_MIN.hex(),
        "order": 4,
        "target_remainder_radius": 1e-4,
        "cutoff_threshold": 1e-10,
        "validator": "flowstar_raw_remainder_compat",
        "prestate_sha256_before": before,
        "prestate_sha256_after": after,
        "prestate_unchanged": before == after,
        "center_hex": [float(value).hex() for value in normal_state.center],
        "scale_hex": [float(value).hex() for value in normal_state.scales],
        "right_polynomial_hashes": tmvector_hashes(
            _tmvector_without_remainder(normal_state.tmv_right)
        ),
        "materialized_total_remainder": _total_record(normal_state),
        "decision": decision,
        "status": segment.status,
        "message": segment.message,
        "returned_h": float(segment.h),
        "returned_h_hex": float(segment.h).hex(),
        "step_rejections": int(segment.step_rejections),
        "any_shrink": int(segment.step_rejections) > 0
        or float(segment.h).hex() != TERMINAL_H.hex(),
        "raw_remainder": (
            {
                "lo": segment.picard_image_remainder[0],
                "hi": segment.picard_image_remainder[1],
            }
            if segment.picard_image_remainder is not None
            else None
        ),
        "typed_categories": typed,
        "subset_margin": segment.subset_margin,
        "x_margin": (
            float(segment.subset_margin[0][0])
            if segment.subset_margin is not None
            else None
        ),
        "y_margin": (
            float(segment.subset_margin[0][1])
            if segment.subset_margin is not None
            else None
        ),
        "publication": publication,
        "candidate_gates": candidate_gates,
        "returned_state": (
            {
                "current": _encode_tmvector(segment.reset_tm),
                "normal_state": _encode_normal_state(segment.flowstar_normal_state),
                "state_sha256": _state_sha(
                    segment.reset_tm,
                    segment.flowstar_normal_state,
                ),
            }
            if decision == "accepted"
            else None
        ),
        "diagnostics": diagnostics,
    }
    return record, segment


def run(
    candidate_checkpoint: Path,
    l0_snapshot: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    candidate = load_terminal_checkpoint(
        candidate_checkpoint,
        expected_contract=runner.CONTRACT,
        expected_order=4,
        expected_dtype="float64",
    )
    if (
        float(candidate.scheduler["current_time"]) != TERMINAL_TIME
        or float(candidate.scheduler["h_attempted"]) != TERMINAL_H
        or int(candidate.scheduler["accepted_segment_count"]) != 307
    ):
        raise ValueError("candidate checkpoint is not the frozen terminal prestate")
    t1_state = _materialize(candidate.normal_state)
    l0_current, l0_state, l0_payload = _load_l0_snapshot(l0_snapshot)

    native_total = _total_record(candidate.normal_state)
    materialized_total = _total_record(t1_state)
    same_set_checks = {
        "center_hex_equal": [float(value).hex() for value in candidate.normal_state.center]
        == [float(value).hex() for value in t1_state.center],
        "scale_hex_equal": [float(value).hex() for value in candidate.normal_state.scales]
        == [float(value).hex() for value in t1_state.scales],
        "right_polynomial_equal": tmvector_hashes(
            _tmvector_without_remainder(candidate.normal_state.tmv_right)
        )
        == tmvector_hashes(_tmvector_without_remainder(t1_state.tmv_right)),
        "materialized_total_hex_equal": native_total["lo_hex"]
        == materialized_total["lo_hex"]
        and native_total["hi_hex"] == materialized_total["hi_hex"],
    }
    if not all(same_set_checks.values()):
        raise RuntimeError("T0/T1 represented-set equality construction failed")

    records: dict[str, dict[str, Any]] = {}
    records["T0"], _ = _run(
        "T0",
        "B",
        candidate.current,
        candidate.normal_state,
    )
    records["T1"], _ = _run(
        "T1",
        "L0",
        candidate.current,
        t1_state,
    )
    records["T2"], _ = _run("T2", "L0", l0_current, l0_state)
    with (output_dir / "terminal_controls.jsonl").open("w", encoding="utf-8") as handle:
        for name in ("T0", "T1", "T2"):
            handle.write(json.dumps(records[name], sort_keys=True) + "\n")

    t0 = records["T0"]
    contract_frozen = {
        "time": TERMINAL_TIME,
        "h": TERMINAL_H,
        "order": runner.CONTRACT["requested_order"],
        "target": runner.CONTRACT["target_remainder_radius"],
        "cutoff": runner.CONTRACT["cutoff"],
        "h_min": runner.CONTRACT["h_min"],
        "validator": runner.CONTRACT["validation_mode"],
    }
    contract_exact = contract_frozen == {
        "time": 6.397083942944808,
        "h": 0.003623635847674574,
        "order": 4,
        "target": 1e-4,
        "cutoff": 1e-10,
        "h_min": 0.002,
        "validator": "flowstar_raw_remainder_compat",
    }
    passed = bool(
        contract_exact
        and t0["decision"] == "accepted"
        and t0["returned_h_hex"] == TERMINAL_H.hex()
        and not t0["any_shrink"]
        and t0["prestate_unchanged"]
        and t0["candidate_gates"] is not None
        and t0["candidate_gates"]["passed"]
        and all(same_set_checks.values())
    )
    summary = {
        "schema": "torch_tm_flowpipe_s1_terminal_gate_v1",
        "outcome": (
            "CORRECTED_S1_TERMINAL_GATE_PASS"
            if passed
            else "CORRECTED_S1_REACHES_TERMINAL_BUT_DOES_NOT_CLOSE_IT"
        ),
        "passed": passed,
        "frozen_contract": contract_frozen,
        "frozen_contract_exact": contract_exact,
        "candidate_checkpoint_sha256": candidate.manifest[
            "full_checkpoint_sha256"
        ],
        "historical_l0_snapshot_sha256": hashlib.sha256(
            l0_snapshot.read_bytes()
        ).hexdigest(),
        "historical_l0_snapshot_state_sha256": l0_payload["state_sha256"],
        "T0_T1_set_relation": "equal",
        "T0_T1_same_set_checks": same_set_checks,
        "decisions": {
            name: {
                "decision": record["decision"],
                "returned_h_hex": record["returned_h_hex"],
                "step_rejections": record["step_rejections"],
                "x_margin": record["x_margin"],
                "y_margin": record["y_margin"],
                "prestate_unchanged": record["prestate_unchanged"],
            }
            for name, record in records.items()
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--historical-l0-snapshot", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run(
        args.candidate_checkpoint.resolve(),
        args.historical_l0_snapshot.resolve(),
        args.output_dir.resolve(),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
