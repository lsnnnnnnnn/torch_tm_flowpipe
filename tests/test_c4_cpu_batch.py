from __future__ import annotations

from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from torch_tm_flowpipe import (
    CPUPolynomialPlantBatch,
    CPUPolynomialPlantLane,
    FlowstarNormalFlowpipeState,
    cpu_batch_fingerprint,
    cpu_batch_lane_fingerprint,
    load_cpu_batch_checkpoint,
    run_independent_cpu_batch,
    save_cpu_batch_checkpoint,
)


def _lane(lane_id: str, shift: float = 0.0) -> CPUPolynomialPlantLane:
    state = FlowstarNormalFlowpipeState.from_exact_decimal_box(
        [(str(-0.1 + shift), str(0.1 + shift))],
        2,
    )
    return CPUPolynomialPlantLane(lane_id, state.normalized_initial_tm(2), state)


def _step(lane: CPUPolynomialPlantLane):
    if lane.lane_id == "reject":
        return SimpleNamespace(
            status="failed",
            message="designed rejection",
            backend_counters={"post_accept_replay_calls": 0},
            candidate_remainder=None,
            picard_image_remainder=None,
            reset_tm=None,
            flowstar_normal_state=None,
            endpoint_raw_tm=None,
            tm=lane.current,
        )
    state = replace(lane.normal_state, step_index=lane.normal_state.step_index + 1)
    return SimpleNamespace(
        status="validated",
        message="",
        backend_counters={
            "post_accept_replay_calls": 3,
            "post_accept_committed_replays": 3,
            "post_accept_stop_ratio_count": 1,
        },
        candidate_remainder=((0.0, 0.0),),
        picard_image_remainder=((0.0, 0.0),),
        reset_tm=lane.current,
        flowstar_normal_state=state,
        endpoint_raw_tm=lane.current,
        tm=lane.current,
    )


def test_cpu_batch_rejection_freezes_only_its_owned_lane() -> None:
    batch = CPUPolynomialPlantBatch(
        (_lane("accept_a"), _lane("reject", 0.25), _lane("accept_b", 0.5))
    )
    rejected_before = cpu_batch_lane_fingerprint(batch.lanes[1])
    result = run_independent_cpu_batch(batch, _step, cycles=2)
    assert [lane.accepted_steps for lane in result.lanes] == [2, 0, 2]
    assert [lane.rejected_steps for lane in result.lanes] == [0, 1, 0]
    assert [lane.frozen for lane in result.lanes] == [False, True, False]
    rejected_after = cpu_batch_lane_fingerprint(result.lanes[1])
    assert rejected_after["current_hashes"] == rejected_before["current_hashes"]
    assert rejected_after["state_step_index"] == rejected_before["state_step_index"]
    assert result.lanes[0].last_stop_reason == "stop_ratio"


def test_cpu_batch_chunk_partition_is_lane_exact() -> None:
    initial = tuple(_lane(f"lane_{index}", index / 20.0) for index in range(8))
    whole = run_independent_cpu_batch(CPUPolynomialPlantBatch(initial), _step, cycles=3)
    for chunk_size in (1, 2, 4):
        chunked = []
        for offset in range(0, 8, chunk_size):
            part = run_independent_cpu_batch(
                CPUPolynomialPlantBatch(initial[offset : offset + chunk_size]),
                _step,
                cycles=3,
            )
            chunked.extend(part.lanes)
        assert [cpu_batch_lane_fingerprint(lane) for lane in chunked] == [
            cpu_batch_lane_fingerprint(lane) for lane in whole.lanes
        ]


def test_cpu_batch_checkpoint_resume_and_manifest_tamper(tmp_path) -> None:
    initial = CPUPolynomialPlantBatch(tuple(_lane(f"lane_{index}") for index in range(2)))
    after_one = run_independent_cpu_batch(initial, _step, cycles=1)
    checkpoint = tmp_path / "batch_checkpoint"
    contract = {"plant": "unit_scalar", "dtype": "float64"}
    save_cpu_batch_checkpoint(
        checkpoint,
        after_one,
        contract=contract,
        provenance={"test": True},
    )
    loaded = load_cpu_batch_checkpoint(checkpoint, expected_contract=contract)
    resumed = run_independent_cpu_batch(loaded, _step, cycles=2)
    uninterrupted = run_independent_cpu_batch(initial, _step, cycles=3)
    assert cpu_batch_fingerprint(resumed) == cpu_batch_fingerprint(uninterrupted)

    manifest_path = checkpoint / "cpu_batch_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["cycle_index"] = 999
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        load_cpu_batch_checkpoint(checkpoint)

