import json
from pathlib import Path

import pytest

from torch_tm_flowpipe import (
    FlowstarNormalFlowpipeState,
    Interval,
    TMVector,
    flowpipe_step_flowstar_style_adaptive,
    load_terminal_checkpoint,
    save_terminal_checkpoint,
    tmvector_hashes,
)


CONTRACT = {
    "ode": ["y", "y-x-x*x*y"],
    "requested_order": 4,
    "dtype": "float64",
    "h_min": 0.002,
    "cutoff": 1e-10,
    "target_remainder_radius": 1e-4,
}


def _scheduler(time=0.0, h=0.01):
    return {
        "current_time": time,
        "h_next": h,
        "h_attempted": h,
        "accepted_segment_count": 0,
        "previous_rejection_count": 0,
    }


def _roundtrip(tmp_path: Path, current: TMVector, state: FlowstarNormalFlowpipeState, *, scheduler=None):
    first = tmp_path / "first"
    second = tmp_path / "second"
    manifest = save_terminal_checkpoint(
        first,
        current=current,
        normal_state=state,
        scheduler=scheduler or _scheduler(),
        contract=CONTRACT,
        provenance={"branch": "test", "commit": "0" * 40, "dtype": "float64", "device": "cpu"},
    )
    loaded = load_terminal_checkpoint(first, expected_contract=CONTRACT, expected_order=4, expected_dtype="float64")
    second_manifest = save_terminal_checkpoint(
        second,
        current=loaded.current,
        normal_state=loaded.normal_state,
        scheduler=loaded.scheduler,
        contract=loaded.contract,
        provenance=loaded.provenance,
    )
    assert (first / "terminal_state.json").read_bytes() == (second / "terminal_state.json").read_bytes()
    assert manifest["full_checkpoint_sha256"] == second_manifest["full_checkpoint_sha256"]
    assert tmvector_hashes(current) == tmvector_hashes(loaded.current)
    return first, loaded


def test_scalar_affine_checkpoint_roundtrip_is_bit_exact(tmp_path):
    state = FlowstarNormalFlowpipeState.from_initial_box([Interval(-0.25, 0.75)], order=4)
    precise = 0.12345678901234566
    state = FlowstarNormalFlowpipeState(
        tmv_pre=state.tmv_pre,
        tmv_right=state.tmv_right,
        domain=state.domain,
        center=[precise],
        scales=[precise / 3.0],
        step_index=state.step_index,
        diagnostics=state.diagnostics,
    )
    _, loaded = _roundtrip(tmp_path, state.normalized_initial_tm(4), state)
    assert loaded.normal_state.center[0].hex() == precise.hex()
    assert loaded.normal_state.scales[0].hex() == (precise / 3.0).hex()


def test_harmonic_oscillator_small_state_roundtrip(tmp_path):
    def harmonic(x, u=None):
        return TMVector([x[1], -x[0]])

    initial = [Interval(0.9, 1.1), Interval(-0.1, 0.1)]
    segment = flowpipe_step_flowstar_style_adaptive(
        harmonic,
        initial,
        h=0.01,
        h_min=0.002,
        h_max=0.1,
        order=4,
        target_remainder_radius=1e-4,
        cutoff_threshold=1e-10,
        validation_mode="flowstar_raw_remainder_compat",
        reset_mode="normalized_insertion",
        step_policy_mode="flowstar_compat",
        tm_backend="dense",
    )
    assert segment.status == "validated"
    assert segment.reset_tm is not None and segment.flowstar_normal_state is not None
    _roundtrip(tmp_path, segment.reset_tm, segment.flowstar_normal_state, scheduler=_scheduler(0.01, float(segment.next_h)))


def test_vdp_early_step_state_roundtrip(tmp_path):
    def vdp(x, u=None):
        return TMVector([x[1], x[1] - x[0] - x[0] * x[0] * x[1]])

    segment = flowpipe_step_flowstar_style_adaptive(
        vdp,
        [Interval(1.1, 1.4), Interval(2.35, 2.45)],
        h=0.01,
        h_min=0.002,
        h_max=0.1,
        order=4,
        target_remainder_radius=1e-4,
        cutoff_threshold=1e-10,
        validation_mode="flowstar_raw_remainder_compat",
        reset_mode="normalized_insertion",
        step_policy_mode="flowstar_compat",
        tm_backend="dense",
    )
    assert segment.status == "validated"
    assert segment.reset_tm is not None and segment.flowstar_normal_state is not None
    _roundtrip(tmp_path, segment.reset_tm, segment.flowstar_normal_state, scheduler=_scheduler(0.01, float(segment.next_h)))


def test_corrupted_payload_hash_is_rejected(tmp_path):
    state = FlowstarNormalFlowpipeState.from_initial_box([Interval(-1.0, 1.0)], order=4)
    directory, _ = _roundtrip(tmp_path, state.normalized_initial_tm(4), state)
    payload = directory / "terminal_state.json"
    payload.write_bytes(payload.read_bytes().replace(b'"step_index":0', b'"step_index":1', 1))
    with pytest.raises(ValueError, match="payload SHA256 mismatch"):
        load_terminal_checkpoint(directory)


def test_corrupted_manifest_full_hash_is_rejected(tmp_path):
    state = FlowstarNormalFlowpipeState.from_initial_box([Interval(-1.0, 1.0)], order=4)
    directory, _ = _roundtrip(tmp_path, state.normalized_initial_tm(4), state)
    path = directory / "terminal_state_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["full_checkpoint_sha256"] = "0" * 64
    path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="full SHA256 mismatch"):
        load_terminal_checkpoint(directory)


def test_contract_order_and_dtype_mismatches_fail_closed(tmp_path):
    state = FlowstarNormalFlowpipeState.from_initial_box([Interval(-1.0, 1.0)], order=4)
    directory, _ = _roundtrip(tmp_path, state.normalized_initial_tm(4), state)
    with pytest.raises(ValueError, match="contract mismatch"):
        load_terminal_checkpoint(directory, expected_contract={**CONTRACT, "cutoff": 0.0})
    with pytest.raises(ValueError, match="order mismatch"):
        load_terminal_checkpoint(directory, expected_order=5)
    with pytest.raises(ValueError, match="dtype mismatch"):
        load_terminal_checkpoint(directory, expected_dtype="float32")


def test_checkpoint_writer_refuses_nonempty_directory(tmp_path):
    state = FlowstarNormalFlowpipeState.from_initial_box([Interval(-1.0, 1.0)], order=4)
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    sentinel = occupied / "sentinel.json"
    sentinel.write_text(json.dumps({"preserve": True}), encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing non-empty"):
        save_terminal_checkpoint(
            occupied,
            current=state.normalized_initial_tm(4),
            normal_state=state,
            scheduler=_scheduler(),
            contract=CONTRACT,
            provenance={"branch": "test"},
        )
    assert sentinel.exists()
