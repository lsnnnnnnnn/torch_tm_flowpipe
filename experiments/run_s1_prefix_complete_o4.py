#!/usr/bin/env python3
"""Freeze and replay the authoritative 307-boundary complete-O4 schedule."""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import time
from typing import Any, Mapping, Sequence

import torch

from torch_tm_flowpipe import (
    FlowstarNormalFlowpipeState,
    Interval,
    TMVector,
    flowpipe_step_flowstar_style_adaptive,
    load_terminal_checkpoint,
    save_terminal_checkpoint,
    tmvector_hashes,
)
from torch_tm_flowpipe.batched_dense_tm import (
    DenseRangePolicy,
    REMAINDER_LEDGER_CATEGORIES,
)
from torch_tm_flowpipe.fixed_support_outward import OutwardIntervalTensor, outward_sum
from torch_tm_flowpipe.polynomial_ode import PolynomialODE
from torch_tm_flowpipe.structured_remainder import (
    ELIGIBLE_STRUCTURED_SOURCES,
    StructuredRemainderState,
    initialize_structured_remainder_state,
    materialize_structured_remainder,
)
from torch_tm_flowpipe.terminal_checkpoint import (
    _encode_normal_state,
    _encode_tmvector,
    _sha256_json,
)


ROOT = Path(__file__).resolve().parents[1]
FROZEN_T = 6.397083942944808
FROZEN_H = 0.003623635847674574
FROZEN_ACCEPTED = 307
OBSERVATION_BOUNDARIES = {0, 1, 2, 7, 8, 9, 15, 16, 20, 44, 100, 200, 306, 307}
SCHEDULE_SCHEMA = "torch_tm_flowpipe_frozen_complete_o4_schedule_v1"
PREFIX_SCHEMA = "torch_tm_flowpipe_s1_prefix_ledger_v1"

CONTRACT = {
    "canonical_system_spec": {
        "state_names": ["position", "velocity"],
        "initial_box": [[1.1, 1.4], [2.35, 2.45]],
        "rhs": [
            {"terms": [{"coefficient": 1.0, "powers": [0, 1]}]},
            {
                "terms": [
                    {"coefficient": 1.0, "powers": [0, 1]},
                    {"coefficient": -1.0, "powers": [1, 0]},
                    {"coefficient": -1.0, "powers": [2, 1]},
                ]
            },
        ],
    },
    "requested_order": 4,
    "dtype": "float64",
    "h_min": 0.002,
    "h_max": 0.1,
    "cutoff": 1e-10,
    "target_remainder_radius": 1e-4,
    "validation_mode": "flowstar_raw_remainder_compat",
    "step_policy_mode": "flowstar_compat",
    "dense_range_policy": {
        "method": "adaptive_subdivision",
        "max_depth": 1,
        "max_leaves": 4,
        "split_vars": [0, 1],
        "trigger": "proactive_depth1_on_named_contexts",
        "named_contexts": ["polynomial_truncation"],
        "variable_orders": [[0, 1, 2], [1, 0, 2], [2, 0, 1]],
    },
}


def _canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, OutwardIntervalTensor):
        return {"lo": _jsonable(value.lo), "hi": _jsonable(value.hi)}
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _float_record(text: str) -> dict[str, Any] | None:
    if text == "":
        return None
    value = float(text)
    return {"decimal": text, "hex": value.hex(), "value": value}


def freeze_schedule(source_csv: Path, destination: Path) -> dict[str, Any]:
    source_csv = source_csv.resolve()
    rows = list(csv.DictReader(source_csv.open(newline="", encoding="utf-8")))
    if len(rows) != FROZEN_ACCEPTED + 1:
        raise ValueError(f"expected 308 authoritative segment rows, received {len(rows)}")
    frozen_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if int(row["segment_index"]) != index:
            raise ValueError("authoritative segment indices are not contiguous")
        expected = "accepted" if index < FROZEN_ACCEPTED else "rejected"
        if row["status"] != expected:
            raise ValueError(f"authoritative decision mismatch at segment {index}")
        frozen_rows.append(
            {
                "attempt_index": index,
                "accepted_boundary_index_before": index,
                "accepted_boundary_index_after": index + 1 if expected == "accepted" else index,
                "expected_status": expected,
                "t_before": _float_record(row["t_lo"]),
                "t_after": _float_record(row["t_hi"]),
                "h_attempted": _float_record(row["h_attempted"]),
                "h_accepted": _float_record(row["h_accepted"]),
                "rejection_count_before_acceptance": int(row["step_rejections"]),
                "next_h": _float_record(row["next_h"]),
            }
        )
    terminal = frozen_rows[-1]
    if terminal["t_before"]["hex"] != FROZEN_T.hex() or terminal["h_attempted"]["hex"] != FROZEN_H.hex():
        raise ValueError("authoritative terminal time/step does not match the frozen contract")
    try:
        source_relative = str(source_csv.relative_to(ROOT))
    except ValueError:
        source_relative = str(source_csv)
    schedule = {
        "schema": SCHEDULE_SCHEMA,
        "accepted_boundary_count": FROZEN_ACCEPTED,
        "terminal_attempt_index": FROZEN_ACCEPTED,
        "source_artifact": source_relative,
        "source_artifact_sha256": _sha256(source_csv),
        "precision": (
            "CSV uses Python round-trip decimal spellings; every schedule scalar is also stored "
            "as float.hex() and the replay compares binary64 hex values."
        ),
        "contract_sha256": _sha256_json(CONTRACT),
        "rows": frozen_rows,
    }
    _atomic_json(destination, schedule)
    return schedule


def _state_hash(current: TMVector | list[Interval], normal_state: FlowstarNormalFlowpipeState | None) -> str:
    if isinstance(current, TMVector):
        current_value: Any = _encode_tmvector(current)
    else:
        current_value = [
            {"lo_hex": float(interval.lo).hex(), "hi_hex": float(interval.hi).hex()}
            for interval in current
        ]
    return _sha256_json(
        {
            "current": current_value,
            "normal_state": _encode_normal_state(normal_state) if normal_state is not None else None,
        }
    )


def _interval_record(value: OutwardIntervalTensor) -> dict[str, Any]:
    return {"lo": value.lo.detach().cpu().tolist(), "hi": value.hi.detach().cpu().tolist()}


def _state_identities(state: StructuredRemainderState) -> list[dict[str, int]]:
    rows: list[dict[str, int]] = []
    for batch in range(state.batch):
        for slot in range(state.capacity):
            if bool(state.active[batch, slot]):
                rows.append(
                    {
                        "batch": batch,
                        "slot": slot,
                        "boundary": int(state.source_boundary_index[batch, slot]),
                        "category_id": int(state.source_id[batch, slot]),
                        "occurrence": int(state.source_occurrence_index[batch, slot]),
                        "age": int(state.age[batch, slot]),
                    }
                )
    return rows


def _event_record(event: Any) -> dict[str, Any]:
    return {
        "reason": event.reason,
        "active_mask": _jsonable(event.active_mask),
        "accepted_boundary_index": event.accepted_boundary_index,
        "slot": _jsonable(event.slot),
        "source_category_id": _jsonable(event.source_category_id),
        "source_category": list(event.source_category),
        "source_boundary_index": _jsonable(event.source_boundary_index),
        "source_occurrence_index": _jsonable(event.source_occurrence_index),
        "age": _jsonable(event.age),
        "pre_propagation": {"lo": _jsonable(event.pre_propagation_lo), "hi": _jsonable(event.pre_propagation_hi)},
        "post_propagation": {"lo": _jsonable(event.post_propagation_lo), "hi": _jsonable(event.post_propagation_hi)},
        "materialized": {"lo": _jsonable(event.materialized_lo), "hi": _jsonable(event.materialized_hi)},
    }


def _materialize_every_boundary(state: StructuredRemainderState) -> tuple[StructuredRemainderState, int]:
    total = materialize_structured_remainder(state)
    count = int(state.active.sum().item())
    return replace(
        state,
        ordinary_rem_lo=total.lo,
        ordinary_rem_hi=total.hi,
        j_lo=torch.zeros_like(state.j_lo),
        j_hi=torch.zeros_like(state.j_hi),
        phi_lo=torch.zeros_like(state.phi_lo),
        phi_hi=torch.zeros_like(state.phi_hi),
        active=torch.zeros_like(state.active),
        age=torch.full_like(state.age, -1),
        source_id=torch.zeros_like(state.source_id),
        source_boundary_index=torch.full_like(state.source_boundary_index, -1),
        source_occurrence_index=torch.full_like(state.source_occurrence_index, -1),
    ), count


def _initialize_structured_lane(
    reset_mode: str = "normalized_insertion_structured_remainder_k16",
) -> tuple[TMVector, FlowstarNormalFlowpipeState]:
    if reset_mode not in {
        "normalized_insertion_structured_remainder_k16",
        "normalized_insertion_structured_total_delta_k16",
    }:
        raise ValueError("unknown structured reset mode")
    initial = [Interval(*bounds) for bounds in CONTRACT["canonical_system_spec"]["initial_box"]]
    normal = FlowstarNormalFlowpipeState.from_initial_box(initial, CONTRACT["requested_order"])
    structured = initialize_structured_remainder_state(1, len(normal.center))
    scale = torch.tensor([normal.scales], dtype=torch.float64)
    inverse = torch.where(scale == 0, torch.ones_like(scale), 1.0 / scale)
    structured = replace(structured, inverse_scale=inverse)
    normal = replace(
        normal,
        structured_remainder_state=structured,
        diagnostics={
            **dict(normal.diagnostics or {}),
            "reset_mode": reset_mode,
            "structured_initial_state": True,
        },
    )
    return normal.normalized_initial_tm(CONTRACT["requested_order"]), normal


def _write_snapshot(
    directory: Path,
    boundary: int,
    current: TMVector | list[Interval],
    normal_state: FlowstarNormalFlowpipeState | None,
    *,
    reason: str,
) -> None:
    payload = {
        "boundary": boundary,
        "reason": reason,
        "state_sha256": _state_hash(current, normal_state),
        "current": (
            _encode_tmvector(current)
            if isinstance(current, TMVector)
            else [
                {"lo_hex": float(interval.lo).hex(), "hi_hex": float(interval.hi).hex()}
                for interval in current
            ]
        ),
        "normal_state": _encode_normal_state(normal_state) if normal_state is not None else None,
    }
    _atomic_json(directory / f"boundary_{boundary:03d}_{reason}.json", payload)


def _append_jsonl(handle: Any, value: Mapping[str, Any]) -> None:
    handle.write(json.dumps(_jsonable(value), sort_keys=True) + "\n")
    handle.flush()


def _git_value(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def _run_lane_step(
    ode: PolynomialODE,
    current: TMVector | list[Interval],
    normal_state: FlowstarNormalFlowpipeState | None,
    *,
    lane: str,
    h: float,
    policy: DenseRangePolicy,
    diagnostics: list[dict[str, Any]],
    diagnostics_context: Mapping[str, Any],
    h_min: float | None = None,
    h_max: float | None = None,
    max_validation_attempts: int = 2,
    structured_allow_outward_renormalization: bool = True,
):
    structured_lane = lane in {"L1", "L2", "B"}
    total_delta_lane = lane == "B"
    return flowpipe_step_flowstar_style_adaptive(
        ode,
        current,
        h=float(h),
        h_min=CONTRACT["h_min"] if h_min is None else float(h_min),
        h_max=CONTRACT["h_max"] if h_max is None else float(h_max),
        order=CONTRACT["requested_order"],
        target_remainder_radius=CONTRACT["target_remainder_radius"],
        cutoff_threshold=CONTRACT["cutoff"],
        max_validation_attempts=int(max_validation_attempts),
        validation_eps=1e-12,
        validation_mode=CONTRACT["validation_mode"],
        reset_mode=(
            "normalized_insertion_structured_total_delta_k16"
            if total_delta_lane
            else "normalized_insertion_structured_remainder_k16"
            if structured_lane
            else "normalized_insertion"
        ),
        step_policy_mode=CONTRACT["step_policy_mode"],
        flowstar_normal_state=normal_state,
        right_map_center_mode="constant",
        right_map_range_mode="standard",
        tm_backend="dense",
        dense_device="cpu",
        dense_range_policy=policy,
        diagnostics=diagnostics,
        diagnostics_context=diagnostics_context,
        structured_allow_outward_renormalization=structured_allow_outward_renormalization,
    )


def _save_boundary_checkpoint(
    output_dir: Path,
    *,
    boundary: int,
    current: TMVector,
    normal_state: FlowstarNormalFlowpipeState,
    frozen: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    checkpoint_dir = output_dir / f"boundary_{boundary:03d}_checkpoint"
    manifest = save_terminal_checkpoint(
        checkpoint_dir,
        current=current,
        normal_state=normal_state,
        scheduler={
            "current_time": frozen["t_before"]["value"],
            "h_next": frozen["h_attempted"]["value"],
            "h_attempted": frozen["h_attempted"]["value"],
            "accepted_segment_count": int(boundary),
            "previous_rejection_count": sum(int(row["actual_rejections"]) for row in rows),
            "checkpoint_role": "frozen_proposed_step_prestate",
        },
        contract=CONTRACT,
        provenance=provenance,
    )
    loaded = load_terminal_checkpoint(
        checkpoint_dir,
        expected_contract=CONTRACT,
        expected_order=4,
        expected_dtype="float64",
    )
    roundtrip_dir = output_dir / f"boundary_{boundary:03d}_checkpoint_roundtrip"
    roundtrip_manifest = save_terminal_checkpoint(
        roundtrip_dir,
        current=loaded.current,
        normal_state=loaded.normal_state,
        scheduler=loaded.scheduler,
        contract=loaded.contract,
        provenance=loaded.provenance,
    )
    byte_stable = all(
        (checkpoint_dir / name).read_bytes() == (roundtrip_dir / name).read_bytes()
        for name in ("terminal_state.json", "terminal_state_manifest.json")
    )
    record = {
        "checkpoint_role": "frozen_proposed_step_prestate",
        "accepted_boundary_index": int(boundary),
        "schema": manifest["schema"],
        "byte_stable": byte_stable,
        "first_full_sha256": manifest["full_checkpoint_sha256"],
        "second_full_sha256": roundtrip_manifest["full_checkpoint_sha256"],
    }
    _atomic_json(output_dir / f"boundary_{boundary:03d}_checkpoint_roundtrip.json", record)
    if not byte_stable:
        raise RuntimeError(f"boundary {boundary} checkpoint did not round-trip exactly")
    return record


def _frozen_full_h_diagnostic(
    output_dir: Path,
    *,
    lane: str,
    attempt_index: int,
    current: TMVector,
    normal_state: FlowstarNormalFlowpipeState,
    frozen: Mapping[str, Any],
    ode: PolynomialODE,
    policy: DenseRangePolicy,
) -> dict[str, Any]:
    """Run exactly the first validator call at the frozen proposed h.

    This diagnostic never commits its returned state.  Setting h_min=h_max=h
    prevents the adaptive helper from publishing a half-step in this record.
    """
    pre_hash = _state_hash(current, normal_state)
    diagnostics: list[dict[str, Any]] = []
    proposed_h = float(frozen["h_attempted"]["value"])
    segment = _run_lane_step(
        ode,
        current,
        normal_state,
        lane=lane,
        h=proposed_h,
        h_min=proposed_h,
        h_max=proposed_h,
        max_validation_attempts=1,
        policy=policy,
        diagnostics=diagnostics,
        diagnostics_context={
            "segment_index": attempt_index,
            "t_before": frozen["t_before"]["value"],
            "lane": lane,
            "diagnostic_role": "frozen_full_h_first_validator",
        },
    )
    post_hash = _state_hash(current, normal_state)
    first_validator = next(
        (
            row
            for row in diagnostics
            if row.get("phase") == "remainder_validation"
            and int(row.get("attempt", -1)) == 1
        ),
        None,
    )
    if first_validator is None:
        raise RuntimeError(f"{lane} frozen full-h diagnostic did not expose validator attempt 1")
    if pre_hash != post_hash:
        raise RuntimeError(f"{lane} frozen full-h diagnostic mutated its prestate")
    decision = (
        "accepted"
        if segment.status == "validated" and segment.reset_tm is not None
        else "rejected"
    )
    result = {
        "schema": "torch_tm_flowpipe_s1_frozen_full_h_diagnostic_v1",
        "lane": lane,
        "attempt_index": int(attempt_index),
        "accepted_boundary_index": int(frozen["accepted_boundary_index_before"]),
        "t_before": frozen["t_before"],
        "h": frozen["h_attempted"],
        "max_validation_attempts": 1,
        "adaptive_shrink_allowed": False,
        "decision": decision,
        "segment_status": segment.status,
        "subset_margin": segment.subset_margin,
        "first_validator_diagnostic": _jsonable(first_validator),
        "all_diagnostics": _jsonable(diagnostics),
        "prestate_sha256": pre_hash,
        "poststate_sha256": post_hash,
        "prestate_byte_identity_preserved": True,
        "returned_state_committed": False,
    }
    _atomic_json(
        output_dir / f"boundary_{int(frozen['accepted_boundary_index_before']):03d}_full_h_diagnostic.json",
        result,
    )
    return result


def replay_lane(
    schedule: Mapping[str, Any],
    output_dir: Path,
    lane: str,
    *,
    max_boundaries: int | None = None,
) -> dict[str, Any]:
    if lane not in {"L0", "L1", "L2", "B"}:
        raise ValueError("lane must be L0, L1, L2, or B")
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FileExistsError(f"refusing non-empty lane output: {output_dir}")
    structured_lane = lane in {"L1", "L2", "B"}
    if structured_lane:
        current, normal_state = _initialize_structured_lane(
            "normalized_insertion_structured_total_delta_k16"
            if lane == "B"
            else "normalized_insertion_structured_remainder_k16"
        )
    else:
        current = [Interval(*bounds) for bounds in CONTRACT["canonical_system_spec"]["initial_box"]]
        normal_state = None
    policy_spec = CONTRACT["dense_range_policy"]
    policy = DenseRangePolicy(
        method=policy_spec["method"],
        max_depth=policy_spec["max_depth"],
        max_leaves=policy_spec["max_leaves"],
        split_vars=tuple(policy_spec["split_vars"]),
        trigger=policy_spec["trigger"],
        named_contexts=tuple(policy_spec["named_contexts"]),
        variable_orders=tuple(tuple(row) for row in policy_spec["variable_orders"]),
    )
    ode = PolynomialODE.from_system_spec(CONTRACT["canonical_system_spec"])
    snapshots = output_dir / "state_snapshots"
    _write_snapshot(snapshots, 0, current, normal_state, reason="mandatory")
    provenance = {
        "lane": lane,
        "branch": _git_value("branch", "--show-current"),
        "commit": _git_value("rev-parse", "HEAD"),
        "worktree_status": _git_value("status", "--short"),
        "schedule_sha256": _sha256_json(schedule),
        "source_artifact_sha256": schedule["source_artifact_sha256"],
        "contract_sha256": _sha256_json(CONTRACT),
        "torch_version": torch.__version__,
        "dtype": "float64",
        "device": "cpu",
    }
    _atomic_json(output_dir / "provenance.json", provenance)
    rows: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    first_full: int | None = None
    first_eviction: int | None = None
    largest_eviction = {"width": -1.0, "boundary": None}
    outcome = "S1_PREFIX_COMPLETE" if structured_lane else "HISTORICAL_BASELINE_REPLAY_COMPLETE"
    divergence: dict[str, Any] | None = None
    terminal_checkpoint_manifest: Mapping[str, Any] | None = None
    boundary_checkpoint_record: Mapping[str, Any] | None = None
    start = time.perf_counter()
    ledger_path = output_dir / "prefix_conservation.jsonl"
    event_path = output_dir / "prefix_source_events.jsonl"
    with ledger_path.open("w", encoding="utf-8") as ledger_handle, event_path.open("w", encoding="utf-8") as event_handle:
        for frozen in schedule["rows"]:
            attempt_index = int(frozen["attempt_index"])
            if max_boundaries is not None and attempt_index >= max_boundaries:
                outcome = "TEST_BOUNDARY_LIMIT_REACHED"
                break
            expected_status = frozen["expected_status"]
            if attempt_index == 164:
                if not isinstance(current, TMVector) or normal_state is None:
                    raise RuntimeError(f"{lane} boundary-164 prestate is incomplete")
                _write_snapshot(
                    snapshots,
                    164,
                    current,
                    normal_state,
                    reason="checkpoint_prestate",
                )
                boundary_checkpoint_record = _save_boundary_checkpoint(
                    output_dir,
                    boundary=164,
                    current=current,
                    normal_state=normal_state,
                    frozen=frozen,
                    rows=rows,
                    provenance=provenance,
                )
            if attempt_index == FROZEN_ACCEPTED and lane == "L2" and terminal_checkpoint_manifest is None:
                checkpoint_dir = output_dir / "terminal_checkpoint_v2"
                terminal_checkpoint_manifest = save_terminal_checkpoint(
                    checkpoint_dir,
                    current=current,
                    normal_state=normal_state,
                    scheduler={
                        "current_time": frozen["t_before"]["value"],
                        "h_next": frozen["h_attempted"]["value"],
                        "h_attempted": frozen["h_attempted"]["value"],
                        "accepted_segment_count": FROZEN_ACCEPTED,
                        "previous_rejection_count": sum(int(row["actual_rejections"]) for row in rows),
                    },
                    contract=CONTRACT,
                    provenance=provenance,
                )
                loaded = load_terminal_checkpoint(
                    checkpoint_dir,
                    expected_contract=CONTRACT,
                    expected_order=4,
                    expected_dtype="float64",
                )
                roundtrip_dir = output_dir / "terminal_checkpoint_v2_roundtrip"
                roundtrip_manifest = save_terminal_checkpoint(
                    roundtrip_dir,
                    current=loaded.current,
                    normal_state=loaded.normal_state,
                    scheduler=loaded.scheduler,
                    contract=loaded.contract,
                    provenance=loaded.provenance,
                )
                byte_stable = all(
                    (checkpoint_dir / name).read_bytes() == (roundtrip_dir / name).read_bytes()
                    for name in ("terminal_state.json", "terminal_state_manifest.json")
                )
                _atomic_json(
                    output_dir / "checkpoint_roundtrip.json",
                    {
                        "byte_stable": byte_stable,
                        "first_full_sha256": terminal_checkpoint_manifest["full_checkpoint_sha256"],
                        "second_full_sha256": roundtrip_manifest["full_checkpoint_sha256"],
                    },
                )
                if not byte_stable:
                    outcome = "S1_PREFIX_CONSERVATION_FAILED"
                    divergence = {"attempt_index": attempt_index, "reason": "v2_checkpoint_roundtrip_not_exact"}
                    break

            pre_hash = _state_hash(current, normal_state)
            pre_structured = (
                normal_state.structured_remainder_state
                if normal_state is not None and isinstance(normal_state.structured_remainder_state, StructuredRemainderState)
                else None
            )
            pre_materialized = materialize_structured_remainder(pre_structured) if pre_structured is not None else None
            full_h_diagnostic = None
            if attempt_index == 164:
                assert isinstance(current, TMVector) and normal_state is not None
                full_h_diagnostic = _frozen_full_h_diagnostic(
                    output_dir,
                    lane=lane,
                    attempt_index=attempt_index,
                    current=current,
                    normal_state=normal_state,
                    frozen=frozen,
                    ode=ode,
                    policy=policy,
                )
            diagnostics: list[dict[str, Any]] = []
            step_start = time.perf_counter()
            segment = _run_lane_step(
                ode,
                current,
                normal_state,
                lane=lane,
                h=frozen["h_attempted"]["value"],
                policy=policy,
                diagnostics=diagnostics,
                diagnostics_context={"segment_index": attempt_index, "t_before": frozen["t_before"]["value"], "lane": lane},
            )
            step_runtime = time.perf_counter() - step_start
            actual_status = "accepted" if segment.status == "validated" and segment.reset_tm is not None else "rejected"
            actual_h = float(segment.h)
            schedule_match = (
                actual_status == expected_status
                and (expected_status != "accepted" or actual_h.hex() == frozen["h_accepted"]["hex"])
                and int(segment.step_rejections) == int(frozen["rejection_count_before_acceptance"])
            )
            row: dict[str, Any] = {
                "schema": PREFIX_SCHEMA,
                "lane": lane,
                "attempt_index": attempt_index,
                "accepted_boundary_index_before": frozen["accepted_boundary_index_before"],
                "expected_status": expected_status,
                "actual_status": actual_status,
                "schedule_match": schedule_match,
                "t_before_hex": frozen["t_before"]["hex"],
                "h_attempted_hex": frozen["h_attempted"]["hex"],
                "h_actual_hex": actual_h.hex(),
                "actual_rejections": int(segment.step_rejections),
                "expected_rejections": int(frozen["rejection_count_before_acceptance"]),
                "prestate_sha256": pre_hash,
                "runtime_s": step_runtime,
                "message": segment.message,
                "subset_margin": segment.subset_margin,
                "committed_to_frozen_prefix": False,
                "frozen_full_h_diagnostic": full_h_diagnostic,
            }
            if not schedule_match:
                row["poststate_sha256"] = pre_hash
                row["frozen_proposed_step_decision"] = "rejected"
                if structured_lane:
                    _write_snapshot(
                        snapshots,
                        int(frozen["accepted_boundary_index_before"]),
                        current,
                        normal_state,
                        reason="first_divergence_prestate",
                    )
                if lane == "L2":
                    checkpoint_dir = output_dir / "final_common_prefix_checkpoint_v2"
                    terminal_checkpoint_manifest = save_terminal_checkpoint(
                        checkpoint_dir,
                        current=current,
                        normal_state=normal_state,
                        scheduler={
                            "current_time": frozen["t_before"]["value"],
                            "h_next": frozen["h_attempted"]["value"],
                            "h_attempted": frozen["h_attempted"]["value"],
                            "accepted_segment_count": int(frozen["accepted_boundary_index_before"]),
                            "previous_rejection_count": sum(int(item["actual_rejections"]) for item in rows),
                            "stop_reason": "first_frozen_proposed_step_rejection",
                        },
                        contract=CONTRACT,
                        provenance=provenance,
                    )
                    loaded = load_terminal_checkpoint(
                        checkpoint_dir,
                        expected_contract=CONTRACT,
                        expected_order=4,
                        expected_dtype="float64",
                    )
                    roundtrip_dir = output_dir / "final_common_prefix_checkpoint_v2_roundtrip"
                    roundtrip_manifest = save_terminal_checkpoint(
                        roundtrip_dir,
                        current=loaded.current,
                        normal_state=loaded.normal_state,
                        scheduler=loaded.scheduler,
                        contract=loaded.contract,
                        provenance=loaded.provenance,
                    )
                    byte_stable = all(
                        (checkpoint_dir / name).read_bytes() == (roundtrip_dir / name).read_bytes()
                        for name in ("terminal_state.json", "terminal_state_manifest.json")
                    )
                    _atomic_json(
                        output_dir / "checkpoint_roundtrip.json",
                        {
                            "checkpoint_role": "final_common_frozen_prefix_prestate",
                            "accepted_boundary_index": int(frozen["accepted_boundary_index_before"]),
                            "byte_stable": byte_stable,
                            "first_full_sha256": terminal_checkpoint_manifest["full_checkpoint_sha256"],
                            "second_full_sha256": roundtrip_manifest["full_checkpoint_sha256"],
                        },
                    )
                    if not byte_stable:
                        raise RuntimeError("final common-prefix v2 checkpoint did not round-trip exactly")
                rows.append(row)
                _append_jsonl(ledger_handle, row)
                divergence = {
                    "attempt_index": attempt_index,
                    "expected_status": expected_status,
                    "actual_status": actual_status,
                    "expected_h_accepted_hex": frozen["h_accepted"]["hex"] if frozen["h_accepted"] else None,
                    "actual_h_hex": actual_h.hex(),
                    "expected_rejections": frozen["rejection_count_before_acceptance"],
                    "actual_rejections": int(segment.step_rejections),
                    "message": segment.message,
                    "prestate_sha256": pre_hash,
                    "off_schedule_poststate_discarded": actual_status == "accepted",
                }
                outcome = (
                    "S1_PREFIX_REJECTS_BEFORE_TERMINAL"
                    if structured_lane and expected_status == "accepted"
                    else "SCHEDULE_DECISION_DIVERGENCE"
                )
                break
            if actual_status == "accepted":
                assert segment.reset_tm is not None and segment.flowstar_normal_state is not None
                next_current = segment.reset_tm
                next_normal = segment.flowstar_normal_state
                immediate_materialization_count = 0
                boundary = segment.structured_boundary_result
                if structured_lane:
                    if boundary is None or not bool(torch.all(boundary.accepted)):
                        schedule_match = False
                        row["schedule_match"] = False
                        row["integration_gate_failure"] = "missing_or_rejected_structured_boundary"
                    post_structured = next_normal.structured_remainder_state
                    assert isinstance(post_structured, StructuredRemainderState)
                    before_control = post_structured
                    if lane == "L1":
                        post_structured, immediate_materialization_count = _materialize_every_boundary(post_structured)
                        next_normal = replace(next_normal, structured_remainder_state=post_structured)
                    post_materialized = materialize_structured_remainder(post_structured)
                    identities_before = _state_identities(pre_structured) if pre_structured is not None else []
                    identities_after = _state_identities(post_structured)
                    identity_keys = {
                        (item["batch"], item["boundary"], item["category_id"], item["occurrence"])
                        for item in identities_after
                    }
                    unique_ownership = len(identity_keys) == len(identities_after)
                    typed = segment.validated_remainder_ledger.entries
                    typed_total = outward_sum(
                        [OutwardIntervalTensor(*typed[name]) for name in REMAINDER_LEDGER_CATEGORIES]
                    )
                    eligible = outward_sum(
                        [OutwardIntervalTensor(*typed[name]) for name in ELIGIBLE_STRUCTURED_SOURCES]
                    )
                    ineligible = outward_sum(
                        [
                            OutwardIntervalTensor(*typed[name])
                            for name in REMAINDER_LEDGER_CATEGORIES
                            if name not in ELIGIBLE_STRUCTURED_SOURCES
                        ]
                    )
                    finite = bool(
                        torch.all(torch.isfinite(post_materialized.lo))
                        and torch.all(torch.isfinite(post_materialized.hi))
                    )
                    publication = bool(
                        torch.all(segment.endpoint_publication_mask)
                        and torch.all(segment.tube_publication_mask)
                    )
                    conservation = bool(torch.all(boundary.conservation_mask))
                    source_decomposition = bool(torch.all(boundary.source_decomposition_mask))
                    row.update(
                        {
                            "ordinary_pre": (
                                {"lo": _jsonable(pre_structured.ordinary_rem_lo), "hi": _jsonable(pre_structured.ordinary_rem_hi)}
                                if pre_structured is not None
                                else None
                            ),
                            "ordinary_post": {"lo": _jsonable(post_structured.ordinary_rem_lo), "hi": _jsonable(post_structured.ordinary_rem_hi)},
                            "materialized_structured_pre": _interval_record(pre_materialized) if pre_materialized is not None else None,
                            "materialized_structured_post": _interval_record(post_materialized),
                            "validated_raw_compatible_image": {
                                "lo": segment.picard_image_remainder[0],
                                "hi": segment.picard_image_remainder[1],
                            },
                            "typed_source_sum": _interval_record(typed_total),
                            "decomposition_padding": {
                                "lo": _jsonable(segment.validated_remainder_decomposition.padding_lo),
                                "hi": _jsonable(segment.validated_remainder_decomposition.padding_hi),
                            },
                            "linear_structured_image": {"lo": _jsonable(boundary.propagated_symbolic_lo), "hi": _jsonable(boundary.propagated_symbolic_hi)},
                            "nonlinear_structured_residual": {"lo": _jsonable(boundary.nonlinear_residual_lo), "hi": _jsonable(boundary.nonlinear_residual_hi)},
                            "new_eligible_sources": _interval_record(eligible),
                            "ineligible_ordinary_sources": _interval_record(ineligible),
                            "evicted_contribution": {"lo": _jsonable(boundary.evicted_materialized_lo), "hi": _jsonable(boundary.evicted_materialized_hi)},
                            "published_endpoint_total": _interval_record(segment.endpoint_total_remainder),
                            "published_tube_total": _interval_record(segment.tube_total_remainder),
                            "source_identities_before": identities_before,
                            "source_identities_after": identities_after,
                            "active_columns_before": int(pre_structured.active.sum().item()) if pre_structured is not None else 0,
                            "active_columns_after": int(post_structured.active.sum().item()),
                            "event_count_after": int(post_structured.event_count.sum().item()),
                            "immediate_control_materialization_count": immediate_materialization_count,
                            "conservation_mask": conservation,
                            "source_decomposition_mask": source_decomposition,
                            "no_double_count_mask": unique_ownership,
                            "finite_mask": finite,
                            "endpoint_publication_mask": bool(torch.all(segment.endpoint_publication_mask)),
                            "tube_publication_mask": bool(torch.all(segment.tube_publication_mask)),
                            "accepted_mask": bool(torch.all(boundary.accepted)),
                        }
                    )
                    for event in boundary.source_events:
                        event_row = {"lane": lane, "boundary": post_structured.accepted_boundary_index, **_event_record(event)}
                        events.append(event_row)
                        _append_jsonl(event_handle, event_row)
                        if event.reason == "capacity_eviction" and bool(torch.any(event.active_mask)):
                            if first_eviction is None:
                                first_eviction = post_structured.accepted_boundary_index
                            width = float(torch.max(event.materialized_hi - event.materialized_lo).detach().cpu())
                            if width > largest_eviction["width"]:
                                largest_eviction = {"width": width, "boundary": post_structured.accepted_boundary_index}
                                _write_snapshot(
                                    snapshots,
                                    post_structured.accepted_boundary_index,
                                    next_current,
                                    next_normal,
                                    reason="largest_eviction_candidate",
                                )
                    if lane == "L2" and first_full is None and bool(torch.all(before_control.active)):
                        first_full = before_control.accepted_boundary_index
                    all_gates = conservation and source_decomposition and unique_ownership and finite and publication
                    if not all_gates:
                        outcome = "S1_PREFIX_CONSERVATION_FAILED"
                        schedule_match = False
                        row["schedule_match"] = False
                commit_boundary = outcome != "S1_PREFIX_CONSERVATION_FAILED"
                if commit_boundary:
                    current = next_current
                    normal_state = next_normal
                    row["committed_to_frozen_prefix"] = True
                    row["poststate_sha256"] = _state_hash(current, normal_state)
                    row["accepted_boundary_index_after"] = (
                        normal_state.structured_remainder_state.accepted_boundary_index
                        if structured_lane
                        else frozen["accepted_boundary_index_after"]
                    )
                else:
                    row["poststate_sha256"] = pre_hash
                    row["accepted_boundary_index_after"] = frozen["accepted_boundary_index_before"]
                boundary_after = int(row["accepted_boundary_index_after"])
                if commit_boundary and boundary_after in OBSERVATION_BOUNDARIES:
                    _write_snapshot(snapshots, boundary_after, current, normal_state, reason="mandatory")
                if commit_boundary and first_full == boundary_after:
                    _write_snapshot(snapshots, boundary_after, current, normal_state, reason="first_full_k16")
                if commit_boundary and first_eviction == boundary_after:
                    _write_snapshot(snapshots, boundary_after, current, normal_state, reason="first_eviction")
            else:
                row["poststate_sha256"] = pre_hash
                if expected_status == "accepted":
                    outcome = "S1_PREFIX_REJECTS_BEFORE_TERMINAL" if structured_lane else "BASELINE_REPLAY_DIVERGED"

            rows.append(row)
            _append_jsonl(ledger_handle, row)
            if outcome == "S1_PREFIX_CONSERVATION_FAILED":
                divergence = {
                    "attempt_index": attempt_index,
                    "reason": row.get("integration_gate_failure", "structured_conservation_or_publication_gate"),
                    "prestate_sha256": pre_hash,
                }
                break
            if expected_status == "rejected":
                break

    summary = {
        "schema": PREFIX_SCHEMA,
        "lane": lane,
        "outcome": outcome,
        "accepted_boundaries": sum(bool(row.get("committed_to_frozen_prefix")) for row in rows),
        "attempt_rows": len(rows),
        "terminal_prestate_reached": sum(bool(row.get("committed_to_frozen_prefix")) for row in rows) == FROZEN_ACCEPTED,
        "final_common_prefix_boundary": sum(bool(row.get("committed_to_frozen_prefix")) for row in rows),
        "first_full_k16_boundary": first_full,
        "first_eviction_boundary": first_eviction,
        "largest_eviction": largest_eviction if largest_eviction["boundary"] is not None else None,
        "first_schedule_or_decision_divergence": divergence,
        "runtime_s": time.perf_counter() - start,
        "checkpoint_full_sha256": (
            boundary_checkpoint_record["first_full_sha256"]
            if boundary_checkpoint_record is not None
            else terminal_checkpoint_manifest["full_checkpoint_sha256"]
            if terminal_checkpoint_manifest is not None
            else None
        ),
        "boundary_164_checkpoint": boundary_checkpoint_record,
        "provenance": provenance,
    }
    _atomic_json(output_dir / "summary.json", summary)
    _atomic_json(output_dir / "first_divergence.json", divergence or {"divergence": None})
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-segments", type=Path)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--lane", choices=("L0", "L1", "L2", "B"))
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--max-boundaries", type=int)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.source_segments is not None:
        schedule = freeze_schedule(args.source_segments, args.schedule)
    else:
        schedule = json.loads(args.schedule.read_text(encoding="utf-8"))
    if schedule.get("schema") != SCHEDULE_SCHEMA:
        raise ValueError("frozen schedule schema mismatch")
    if args.freeze_only:
        print(json.dumps({"schedule": str(args.schedule), "sha256": _sha256(args.schedule)}, sort_keys=True))
        return 0
    if args.output_dir is None or args.lane is None:
        raise ValueError("--output-dir and --lane are required for replay")
    summary = replay_lane(schedule, args.output_dir, args.lane, max_boundaries=args.max_boundaries)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
