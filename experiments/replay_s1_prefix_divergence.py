#!/usr/bin/env python3
"""Replay the first frozen-schedule S1 divergence from its exact v2 prestate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import torch

from torch_tm_flowpipe import flowpipe_step_flowstar_style_adaptive, load_terminal_checkpoint
from torch_tm_flowpipe.batched_dense_tm import DenseRangePolicy
from torch_tm_flowpipe.polynomial_ode import PolynomialODE

from run_s1_prefix_complete_o4 import CONTRACT


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def replay(checkpoint: Path, schedule_path: Path) -> dict[str, Any]:
    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    frozen = schedule["rows"][164]
    loaded = load_terminal_checkpoint(
        checkpoint,
        expected_contract=CONTRACT,
        expected_order=4,
        expected_dtype="float64",
    )
    if loaded.normal_state.structured_remainder_state.accepted_boundary_index != 164:
        raise ValueError("divergence replay checkpoint is not boundary 164")
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
    diagnostics: list[dict[str, Any]] = []
    start = time.perf_counter()
    segment = flowpipe_step_flowstar_style_adaptive(
        PolynomialODE.from_system_spec(CONTRACT["canonical_system_spec"]),
        loaded.current,
        h=frozen["h_attempted"]["value"],
        h_min=CONTRACT["h_min"],
        h_max=CONTRACT["h_max"],
        order=CONTRACT["requested_order"],
        target_remainder_radius=CONTRACT["target_remainder_radius"],
        cutoff_threshold=CONTRACT["cutoff"],
        max_validation_attempts=2,
        validation_eps=1e-12,
        validation_mode=CONTRACT["validation_mode"],
        reset_mode="normalized_insertion_structured_remainder_k16",
        step_policy_mode=CONTRACT["step_policy_mode"],
        flowstar_normal_state=loaded.normal_state,
        right_map_center_mode="constant",
        right_map_range_mode="standard",
        tm_backend="dense",
        dense_device="cpu",
        dense_range_policy=policy,
        diagnostics=diagnostics,
        diagnostics_context={"segment_index": 164, "t_before": frozen["t_before"]["value"], "lane": "L2"},
    )
    failed = [
        row
        for row in diagnostics
        if str(row.get("validation_status", "")).lower() == "failed"
    ]
    accepted = [
        row
        for row in diagnostics
        if str(row.get("validation_status", "")).lower() == "validated"
    ]
    if not failed or segment.step_rejections != 1:
        raise RuntimeError("expected the frozen proposed step to reject exactly once")
    if float(segment.h).hex() == frozen["h_accepted"]["hex"]:
        raise RuntimeError("divergence replay unexpectedly accepted the frozen proposed h")
    state = loaded.normal_state.structured_remainder_state
    return {
        "schema": "torch_tm_flowpipe_s1_prefix_divergence_replay_v1",
        "checkpoint_full_sha256": loaded.manifest["full_checkpoint_sha256"],
        "accepted_boundary_index": state.accepted_boundary_index,
        "active_columns": int(state.active.sum().item()),
        "event_count": int(state.event_count.sum().item()),
        "t_before": frozen["t_before"],
        "historical_h_accepted": frozen["h_accepted"],
        "historical_rejections": frozen["rejection_count_before_acceptance"],
        "s1_returned_h": {"value": float(segment.h), "hex": float(segment.h).hex()},
        "s1_rejections": int(segment.step_rejections),
        "s1_returned_status": segment.status,
        "off_schedule_poststate_published": False,
        "frozen_proposed_step_decision": "rejected",
        "first_failed_diagnostic": _jsonable(failed[0]),
        "accepted_after_shrink_diagnostic": _jsonable(accepted[-1]) if accepted else None,
        "all_attempt_diagnostics": _jsonable(diagnostics),
        "runtime_s": time.perf_counter() - start,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = replay(args.checkpoint, args.schedule)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
