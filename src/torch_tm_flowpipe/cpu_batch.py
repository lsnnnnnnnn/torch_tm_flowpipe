"""Independent-lane CPU batch semantics for polynomial-plant flowpipes.

This is deliberately an orchestration foundation, not a claim that the full
solver is one fused tensor kernel.  Every lane owns its Taylor model, ordinary
remainder, accepted-boundary SR queue, refinement outcome, rollback state, and
checkpoint.  A rejected lane freezes without changing any other lane.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .flowpipe import FlowpipeSegment, FlowstarNormalFlowpipeState
from .symbolic_remainder import accepted_boundary_sr_queue_sha256
from .terminal_checkpoint import (
    load_terminal_checkpoint,
    save_terminal_checkpoint,
    tmvector_hashes,
)
from .tm_vector import TMVector


CPU_BATCH_SCHEMA = "torch_tm_flowpipe.independent_cpu_plant_batch/1"
CPU_BATCH_CHECKPOINT_SCHEMA = "torch_tm_flowpipe.independent_cpu_plant_batch_checkpoint/1"


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    raise TypeError(f"CPU batch metadata is not JSON serializable: {type(value).__name__}")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


@dataclass(frozen=True)
class CPUPolynomialPlantLane:
    """One independently owned B1 solver state embedded in a CPU batch."""

    lane_id: str
    current: TMVector
    normal_state: FlowstarNormalFlowpipeState
    accepted_steps: int = 0
    rejected_steps: int = 0
    frozen: bool = False
    last_status: str = "initial"
    last_message: str = ""
    last_endpoint_hashes: Mapping[str, str] | None = None
    last_tube_hashes: Mapping[str, str] | None = None
    last_reset_hashes: Mapping[str, str] | None = None
    last_candidate_remainder: Any = None
    last_final_remainder: Any = None
    last_replay_calls: int = 0
    last_committed_replays: int = 0
    last_stop_reason: str = "not_run"

    def __post_init__(self) -> None:
        if not self.lane_id:
            raise ValueError("CPU batch lane_id must be nonempty")
        if self.accepted_steps < 0 or self.rejected_steps < 0:
            raise ValueError("CPU batch lane counters must be nonnegative")


@dataclass(frozen=True)
class CPUPolynomialPlantBatch:
    """A fixed-order collection of isolated CPU B1 lanes."""

    lanes: tuple[CPUPolynomialPlantLane, ...]
    cycle_index: int = 0
    schema: str = CPU_BATCH_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != CPU_BATCH_SCHEMA:
            raise ValueError("CPU batch schema mismatch")
        if not self.lanes:
            raise ValueError("CPU batch must contain at least one lane")
        lane_ids = tuple(lane.lane_id for lane in self.lanes)
        if len(set(lane_ids)) != len(lane_ids):
            raise ValueError("CPU batch lane ids must be unique")
        if self.cycle_index < 0:
            raise ValueError("CPU batch cycle_index must be nonnegative")

    @property
    def batch_size(self) -> int:
        return len(self.lanes)


CPULaneStep = Callable[[CPUPolynomialPlantLane], FlowpipeSegment]


def _stop_reason(counters: Mapping[str, Any]) -> str:
    names = (
        ("post_accept_stop_ratio_count", "stop_ratio"),
        ("post_accept_fixed_point_count", "fixed_point"),
        ("post_accept_failure_count", "refinement_failure"),
        ("post_accept_replay_cap_count", "replay_cap"),
    )
    selected = [label for key, label in names if int(counters.get(key, 0)) > 0]
    return selected[0] if len(selected) == 1 else ("none" if not selected else "multiple")


def advance_independent_cpu_batch(
    batch: CPUPolynomialPlantBatch,
    step_lane: CPULaneStep,
) -> CPUPolynomialPlantBatch:
    """Advance every non-frozen lane once using whole-lane atomic commit."""

    next_lanes: list[CPUPolynomialPlantLane] = []
    for lane in batch.lanes:
        if lane.frozen:
            next_lanes.append(lane)
            continue
        segment = step_lane(lane)
        counters = dict(segment.backend_counters or {})
        if (
            segment.status == "validated"
            and segment.reset_tm is not None
            and segment.flowstar_normal_state is not None
            and segment.endpoint_raw_tm is not None
        ):
            next_lanes.append(
                replace(
                    lane,
                    current=segment.reset_tm,
                    normal_state=segment.flowstar_normal_state,
                    accepted_steps=lane.accepted_steps + 1,
                    last_status=segment.status,
                    last_message=segment.message,
                    last_endpoint_hashes=tmvector_hashes(segment.endpoint_raw_tm),
                    last_tube_hashes=tmvector_hashes(segment.tm),
                    last_reset_hashes=tmvector_hashes(segment.reset_tm),
                    last_candidate_remainder=_jsonable(segment.candidate_remainder),
                    last_final_remainder=_jsonable(segment.picard_image_remainder),
                    last_replay_calls=int(counters.get("post_accept_replay_calls", 0)),
                    last_committed_replays=int(
                        counters.get("post_accept_committed_replays", 0)
                    ),
                    last_stop_reason=_stop_reason(counters),
                )
            )
        else:
            # The prior B1 state is retained byte-for-byte.  Rejection is a
            # lane-local terminal state for this fixed-schedule batch request.
            next_lanes.append(
                replace(
                    lane,
                    rejected_steps=lane.rejected_steps + 1,
                    frozen=True,
                    last_status=segment.status,
                    last_message=segment.message,
                    last_candidate_remainder=_jsonable(segment.candidate_remainder),
                    last_final_remainder=_jsonable(segment.picard_image_remainder),
                    last_replay_calls=int(counters.get("post_accept_replay_calls", 0)),
                    last_committed_replays=int(
                        counters.get("post_accept_committed_replays", 0)
                    ),
                    last_stop_reason=_stop_reason(counters),
                )
            )
    return CPUPolynomialPlantBatch(tuple(next_lanes), batch.cycle_index + 1)


def run_independent_cpu_batch(
    batch: CPUPolynomialPlantBatch,
    step_lane: CPULaneStep,
    *,
    cycles: int,
) -> CPUPolynomialPlantBatch:
    if int(cycles) < 0:
        raise ValueError("CPU batch cycles must be nonnegative")
    current = batch
    for _ in range(int(cycles)):
        current = advance_independent_cpu_batch(current, step_lane)
    return current


def cpu_batch_lane_fingerprint(lane: CPUPolynomialPlantLane) -> dict[str, Any]:
    queue = lane.normal_state.symbolic_queue
    queue_payload: dict[str, Any] | None = None
    if queue is not None:
        queue_payload = {
            "sha256": accepted_boundary_sr_queue_sha256(queue),
            "generation": queue.generation,
            "accepted_boundary_index": queue.accepted_boundary_index,
            "owner_generations": list(queue.owner_generations),
            "owner_boundary_indices": list(queue.owner_boundary_indices),
            "reset_count": queue.reset_count,
            "size": len(queue.J),
        }
    return {
        "lane_id": lane.lane_id,
        "state_step_index": lane.normal_state.step_index,
        "accepted_steps": lane.accepted_steps,
        "rejected_steps": lane.rejected_steps,
        "frozen": lane.frozen,
        "last_status": lane.last_status,
        "last_message": lane.last_message,
        "last_endpoint_hashes": _jsonable(lane.last_endpoint_hashes),
        "last_tube_hashes": _jsonable(lane.last_tube_hashes),
        "last_reset_hashes": _jsonable(lane.last_reset_hashes),
        "current_hashes": tmvector_hashes(lane.current),
        "last_candidate_remainder": _jsonable(lane.last_candidate_remainder),
        "last_final_remainder": _jsonable(lane.last_final_remainder),
        "last_replay_calls": lane.last_replay_calls,
        "last_committed_replays": lane.last_committed_replays,
        "last_stop_reason": lane.last_stop_reason,
        "queue": queue_payload,
    }


def cpu_batch_fingerprint(batch: CPUPolynomialPlantBatch) -> dict[str, Any]:
    lanes = [cpu_batch_lane_fingerprint(lane) for lane in batch.lanes]
    payload = {
        "schema": CPU_BATCH_SCHEMA,
        "batch_size": batch.batch_size,
        "cycle_index": batch.cycle_index,
        "lanes": lanes,
    }
    return {
        **payload,
        "sha256": hashlib.sha256(_canonical_json(payload)).hexdigest(),
    }


def save_cpu_batch_checkpoint(
    path: str | Path,
    batch: CPUPolynomialPlantBatch,
    *,
    contract: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> Path:
    """Save every lane independently plus a batch ownership manifest."""

    root = Path(path)
    root.mkdir(parents=True, exist_ok=False)
    lane_records: list[dict[str, Any]] = []
    for index, lane in enumerate(batch.lanes):
        relative = Path(f"lane_{index:04d}")
        save_terminal_checkpoint(
            root / relative,
            current=lane.current,
            normal_state=lane.normal_state,
            scheduler={
                "batch_cycle_index": batch.cycle_index,
                "lane_accepted_steps": lane.accepted_steps,
                "lane_rejected_steps": lane.rejected_steps,
                "lane_frozen": lane.frozen,
            },
            contract={**dict(contract), "cpu_batch_lane_id": lane.lane_id},
            provenance=dict(provenance),
        )
        lane_records.append(
            {
                "lane_id": lane.lane_id,
                "checkpoint": str(relative),
                "fingerprint": cpu_batch_lane_fingerprint(lane),
            }
        )
    manifest_body = {
        "schema": CPU_BATCH_CHECKPOINT_SCHEMA,
        "batch_schema": CPU_BATCH_SCHEMA,
        "cycle_index": batch.cycle_index,
        "batch_size": batch.batch_size,
        "contract": _jsonable(contract),
        "provenance": _jsonable(provenance),
        "lanes": lane_records,
    }
    manifest = {
        **manifest_body,
        "manifest_sha256": hashlib.sha256(_canonical_json(manifest_body)).hexdigest(),
    }
    (root / "cpu_batch_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return root


def load_cpu_batch_checkpoint(
    path: str | Path,
    *,
    expected_contract: Mapping[str, Any] | None = None,
) -> CPUPolynomialPlantBatch:
    """Load a batch and verify every lane against its immutable fingerprint."""

    root = Path(path)
    manifest = json.loads((root / "cpu_batch_manifest.json").read_text(encoding="utf-8"))
    manifest_hash = manifest.pop("manifest_sha256", None)
    if manifest.get("schema") != CPU_BATCH_CHECKPOINT_SCHEMA:
        raise ValueError("CPU batch checkpoint schema mismatch")
    if manifest_hash != hashlib.sha256(_canonical_json(manifest)).hexdigest():
        raise ValueError("CPU batch checkpoint manifest checksum mismatch")
    if expected_contract is not None and manifest.get("contract") != _jsonable(expected_contract):
        raise ValueError("CPU batch checkpoint contract mismatch")
    records = manifest.get("lanes")
    if not isinstance(records, list) or len(records) != int(manifest.get("batch_size", -1)):
        raise ValueError("CPU batch checkpoint lane count mismatch")
    lanes: list[CPUPolynomialPlantLane] = []
    for record in records:
        fingerprint = record.get("fingerprint")
        if not isinstance(fingerprint, dict):
            raise ValueError("CPU batch checkpoint lane fingerprint is missing")
        loaded = load_terminal_checkpoint(root / record["checkpoint"])
        lane = CPUPolynomialPlantLane(
            lane_id=str(record["lane_id"]),
            current=loaded.current,
            normal_state=loaded.normal_state,
            accepted_steps=int(fingerprint["accepted_steps"]),
            rejected_steps=int(fingerprint["rejected_steps"]),
            frozen=bool(fingerprint["frozen"]),
            last_status=str(fingerprint["last_status"]),
            last_message=str(fingerprint["last_message"]),
            last_endpoint_hashes=fingerprint.get("last_endpoint_hashes"),
            last_tube_hashes=fingerprint.get("last_tube_hashes"),
            last_reset_hashes=fingerprint.get("last_reset_hashes"),
            last_candidate_remainder=fingerprint.get("last_candidate_remainder"),
            last_final_remainder=fingerprint.get("last_final_remainder"),
            last_replay_calls=int(fingerprint["last_replay_calls"]),
            last_committed_replays=int(fingerprint["last_committed_replays"]),
            last_stop_reason=str(fingerprint["last_stop_reason"]),
        )
        if cpu_batch_lane_fingerprint(lane) != fingerprint:
            raise ValueError("CPU batch checkpoint lane fingerprint mismatch")
        lanes.append(lane)
    return CPUPolynomialPlantBatch(
        tuple(lanes),
        cycle_index=int(manifest["cycle_index"]),
    )

