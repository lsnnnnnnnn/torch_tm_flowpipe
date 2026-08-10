from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from torch_tm_flowpipe import (
    Interval,
    flowpipe_step_flowstar_style_adaptive,
    load_terminal_checkpoint,
    save_terminal_checkpoint,
)
from torch_tm_flowpipe.ode_examples import van_der_pol_ode
from torch_tm_flowpipe.structured_remainder import (
    StructuredRemainderState,
    materialize_structured_remainder,
    normal_interval_to_physical,
)
from torch_tm_flowpipe.terminal_checkpoint import (
    MANIFEST_NAME,
    PAYLOAD_NAME,
    SCHEMA_V2,
    _canonical_bytes,
    _sha256_bytes,
    _sha256_json,
)


CONTRACT = {
    "ode": ["y", "y-x-x*x*y"],
    "requested_order": 4,
    "dtype": "float64",
    "h_min": 0.002,
    "cutoff": 1e-10,
    "target_remainder_radius": 1e-4,
}
SCHEDULER = {
    "current_time": 0.005,
    "h_next": 0.005,
    "h_attempted": 0.005,
    "accepted_segment_count": 1,
    "previous_rejection_count": 0,
}


def _one_s1_step():
    segment = flowpipe_step_flowstar_style_adaptive(
        van_der_pol_ode,
        [Interval(1.1, 1.4), Interval(2.35, 2.45)],
        h=0.005,
        h_min=0.005,
        h_max=0.005,
        order=4,
        target_remainder_radius=1e-4,
        cutoff_threshold=1e-10,
        max_validation_attempts=2,
        validation_mode="flowstar_raw_remainder_compat",
        reset_mode="normalized_insertion_structured_remainder_k16",
        step_policy_mode="flowstar_compat",
        tm_backend="dense",
    )
    assert segment.status == "validated"
    return segment


def _continue_s1(current, normal_state, *, h=0.005):
    return flowpipe_step_flowstar_style_adaptive(
        van_der_pol_ode,
        current,
        h=h,
        h_min=h,
        h_max=h,
        order=4,
        target_remainder_radius=1e-4,
        cutoff_threshold=1e-10,
        max_validation_attempts=2,
        validation_mode="flowstar_raw_remainder_compat",
        reset_mode="normalized_insertion_structured_remainder_k16",
        step_policy_mode="flowstar_compat",
        tm_backend="dense",
        flowstar_normal_state=normal_state,
    )


def _write(directory: Path):
    segment = _one_s1_step()
    manifest = save_terminal_checkpoint(
        directory,
        current=segment.reset_tm,
        normal_state=segment.flowstar_normal_state,
        scheduler=SCHEDULER,
        contract=CONTRACT,
        provenance={"branch": "test", "commit": "1" * 40},
    )
    return segment, manifest


def _assert_structured_bit_equal(left: StructuredRemainderState, right: StructuredRemainderState):
    for name in left.__dataclass_fields__:
        lhs = getattr(left, name)
        rhs = getattr(right, name)
        if isinstance(lhs, torch.Tensor):
            assert torch.equal(lhs, rhs), name
        else:
            assert lhs == rhs, name


def _rewrite_semantically_corrupt(directory: Path, mutate):
    payload_path = directory / PAYLOAD_NAME
    manifest_path = directory / MANIFEST_NAME
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mutate(payload)
    structured = payload.get("normal_state", {}).get("structured_remainder")
    if isinstance(structured, dict):
        manifest["hashes"]["structured_remainder"] = {
            "component_sha256": _sha256_json(structured),
            "fields_sha256": {
                name: _sha256_json(encoded)
                for name, encoded in structured.get("fields", {}).items()
            },
        }
    payload_bytes = _canonical_bytes(payload)
    payload_sha = _sha256_bytes(payload_bytes)
    manifest["payload_sha256"] = payload_sha
    manifest["full_checkpoint_sha256"] = _sha256_json(
        {"payload_sha256": payload_sha, "hashes": manifest["hashes"]}
    )
    payload_path.write_bytes(payload_bytes)
    manifest_path.write_bytes(_canonical_bytes(manifest))


def test_active_s1_uses_v2_and_roundtrips_every_tensor_bit_exactly(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    segment, manifest = _write(first)
    assert manifest["schema"] == SCHEMA_V2
    loaded = load_terminal_checkpoint(
        first,
        expected_contract=CONTRACT,
        expected_order=4,
        expected_dtype="float64",
    )
    assert loaded.scheduler == SCHEDULER
    original = segment.flowstar_normal_state.structured_remainder_state
    restored = loaded.normal_state.structured_remainder_state
    assert isinstance(original, StructuredRemainderState)
    assert isinstance(restored, StructuredRemainderState)
    _assert_structured_bit_equal(original, restored)

    second_manifest = save_terminal_checkpoint(
        second,
        current=loaded.current,
        normal_state=loaded.normal_state,
        scheduler=loaded.scheduler,
        contract=loaded.contract,
        provenance=loaded.provenance,
    )
    assert (first / PAYLOAD_NAME).read_bytes() == (second / PAYLOAD_NAME).read_bytes()
    assert (first / MANIFEST_NAME).read_bytes() == (second / MANIFEST_NAME).read_bytes()
    assert manifest["full_checkpoint_sha256"] == second_manifest["full_checkpoint_sha256"]

    scale_original = torch.tensor([segment.flowstar_normal_state.scales], dtype=torch.float64)
    scale_restored = torch.tensor([loaded.normal_state.scales], dtype=torch.float64)
    original_total = materialize_structured_remainder(original)
    restored_total = materialize_structured_remainder(restored)
    original_endpoint = normal_interval_to_physical(
        original_total.lo, original_total.hi, forward_scale=scale_original
    )
    restored_endpoint = normal_interval_to_physical(
        restored_total.lo, restored_total.hi, forward_scale=scale_restored
    )
    assert torch.equal(original_endpoint.lo, restored_endpoint.lo)
    assert torch.equal(original_endpoint.hi, restored_endpoint.hi)

    original_next = _continue_s1(segment.reset_tm, segment.flowstar_normal_state)
    restored_next = _continue_s1(loaded.current, loaded.normal_state)
    assert original_next.status == restored_next.status == "validated"
    for name in (
        "endpoint_total_structured_remainder",
        "tube_total_structured_remainder",
        "endpoint_total_remainder",
        "tube_total_remainder",
    ):
        lhs = getattr(original_next, name)
        rhs = getattr(restored_next, name)
        assert torch.equal(lhs.lo, rhs.lo), name
        assert torch.equal(lhs.hi, rhs.hi), name

    rejected_snapshot = StructuredRemainderState(
        **{
            name: value.clone() if isinstance(value, torch.Tensor) else value
            for name, value in restored.__dict__.items()
        }
    )
    rejected = _continue_s1(loaded.current, loaded.normal_state, h=0.1)
    assert rejected.status == "failed"
    _assert_structured_bit_equal(rejected_snapshot, loaded.normal_state.structured_remainder_state)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload["normal_state"].pop("structured_remainder"), "missing its required S1"),
        (
            lambda payload: payload["normal_state"]["structured_remainder"]["fields"].pop("j_hi"),
            "field is missing: j_hi",
        ),
        (lambda payload: payload["normal_state"]["structured_remainder"].__setitem__("capacity", 15), "K mismatch"),
        (lambda payload: payload["normal_state"]["structured_remainder"].__setitem__("source_schema", "wrong"), "source schema mismatch"),
        (lambda payload: payload["normal_state"]["structured_remainder"].__setitem__("source_schema_version", 999), "source schema version mismatch"),
        (
            lambda payload: payload["normal_state"]["structured_remainder"]["fields"]["j_lo"].__setitem__("dtype", "float32"),
            "dtype mismatch",
        ),
        (
            lambda payload: payload["normal_state"]["structured_remainder"]["fields"]["j_lo"]["values_hex"].__setitem__(0, "inf"),
            "non-finite|nonfinite",
        ),
    ],
)
def test_v2_semantic_corruption_fails_closed(tmp_path, mutation, message):
    directory = tmp_path / "checkpoint"
    _write(directory)
    _rewrite_semantically_corrupt(directory, mutation)
    with pytest.raises(ValueError, match=message):
        load_terminal_checkpoint(directory)
