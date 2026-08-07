from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest
import torch

from experiments.profile_tora_q3_stages import aggregate_events
from torch_tm_flowpipe.batched_dense_tm import dense_validation_batch
from torch_tm_flowpipe.tora_q3 import (
    build_tora_q3_box_model,
    dense_tora_q3_dr_step,
    tora_b48_boxes,
)


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class FakeEvent:
    name: str
    cpu_parent: "FakeEvent | None" = None
    stack: list[str] = field(default_factory=list)
    input_shapes: list[object] = field(default_factory=list)
    self_cpu_time_total: float = 0.0


@pytest.mark.unit
def test_profiler_aggregation_attributes_stage_source_and_shapes() -> None:
    stage = FakeEvent("stage::remainder_picard_round_03")
    parent = FakeEvent("aten::all", cpu_parent=stage)
    source = ROOT.joinpath("src", "torch_tm_flowpipe", "batched_dense_tm.py")
    callsite = f"{source}(141): validate"
    events = [
        FakeEvent(
            "aten::item",
            cpu_parent=parent,
            stack=[callsite],
            input_shapes=[[]],
            self_cpu_time_total=2.0,
        ),
        FakeEvent(
            "aten::_local_scalar_dense",
            cpu_parent=parent,
            stack=[callsite],
            input_shapes=[[]],
            self_cpu_time_total=3.0,
        ),
        FakeEvent(
            "aten::to",
            cpu_parent=parent,
            stack=[callsite],
            input_shapes=[[48, 5]],
            self_cpu_time_total=1.0,
        ),
    ]
    result = aggregate_events(events, ROOT)
    assert result["totals"]["host_scalar_sync_estimate"] == 1
    assert result["totals"]["aten_to_count"] == 1
    assert result["stage_rows"][0]["stage"] == "remainder_picard_round_03"
    assert result["sync_rows"][0]["source_callsite"] == (
        "src/torch_tm_flowpipe/batched_dense_tm.py:141:validate"
    )
    assert result["to_rows"][0]["input_shapes"] == "[[48,5]]"


@pytest.mark.regression
def test_stage_instrumentation_on_off_is_bitwise_equivalent_cpu() -> None:
    state_lower, state_upper = tora_b48_boxes()
    control_lower = torch.tensor([9.8], dtype=torch.float64)
    control_upper = torch.tensor([10.2], dtype=torch.float64)
    base = build_tora_q3_box_model(
        state_lower[:1],
        state_upper[:1],
        control_lower,
        control_upper,
    )
    plain = dense_tora_q3_dr_step(
        base,
        capture_trace=False,
        profile_stages=False,
    )
    instrumented = dense_tora_q3_dr_step(
        base,
        capture_trace=False,
        profile_stages=True,
    )
    with dense_validation_batch():
        deferred = dense_tora_q3_dr_step(
            base,
            capture_trace=False,
            profile_stages=False,
        )
    assert plain.status == instrumented.status
    assert torch.equal(plain.accepted_by_leaf, instrumented.accepted_by_leaf)
    for left, right in (
        (plain.segment_tm.poly.coeffs, instrumented.segment_tm.poly.coeffs),
        (plain.segment_tm.rem_lo, instrumented.segment_tm.rem_lo),
        (plain.segment_tm.rem_hi, instrumented.segment_tm.rem_hi),
        (plain.endpoint_lower, instrumented.endpoint_lower),
        (plain.endpoint_upper, instrumented.endpoint_upper),
        (plain.tube_lower, instrumented.tube_lower),
        (plain.tube_upper, instrumented.tube_upper),
    ):
        assert torch.equal(left, right)
    for left, right in (
        (plain.segment_tm.poly.coeffs, deferred.segment_tm.poly.coeffs),
        (plain.segment_tm.rem_lo, deferred.segment_tm.rem_lo),
        (plain.segment_tm.rem_hi, deferred.segment_tm.rem_hi),
        (plain.endpoint_lower, deferred.endpoint_lower),
        (plain.endpoint_upper, deferred.endpoint_upper),
        (plain.tube_lower, deferred.tube_lower),
        (plain.tube_upper, deferred.tube_upper),
    ):
        assert torch.equal(left, right)
