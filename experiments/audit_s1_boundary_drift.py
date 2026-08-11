#!/usr/bin/env python3
"""Replay the frozen S1 controls and attribute exact boundary drift."""
from __future__ import annotations

import argparse
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from torch_tm_flowpipe import FlowstarNormalFlowpipeState, Interval, TMVector, tmvector_hashes
from torch_tm_flowpipe.batched_dense_tm import DenseRangePolicy, REMAINDER_LEDGER_CATEGORIES
from torch_tm_flowpipe.fixed_support_outward import OutwardIntervalTensor
from torch_tm_flowpipe.flowpipe import (
    _box_tensor,
    _tmvector_remainder_tensor,
    _tmvector_with_remainder_tensor,
    _tmvector_without_remainder,
)
from torch_tm_flowpipe.s1_boundary_attribution import (
    compare_binary64_scalar,
    compare_interval,
    tensor_hex,
)
from torch_tm_flowpipe.structured_remainder import (
    StructuredRemainderState,
    materialize_structured_remainder,
    normal_interval_to_physical,
    structured_column_contributions,
)
from torch_tm_flowpipe.terminal_checkpoint import _encode_normal_state, _encode_tmvector


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "experiments/run_s1_prefix_complete_o4.py"
RUNNER_SPEC = importlib.util.spec_from_file_location("run_s1_prefix_for_attribution", RUNNER_PATH)
assert RUNNER_SPEC is not None and RUNNER_SPEC.loader is not None
runner = importlib.util.module_from_spec(RUNNER_SPEC)
RUNNER_SPEC.loader.exec_module(runner)


CONTROL_NAMES = ("C0", "C1", "C2", "C3", "C4", "L2")
MILESTONES = (100, 150, 163, 164)


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


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_jsonable(row), sort_keys=True) + "\n")


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


def _interval_dict(value: OutwardIntervalTensor) -> dict[str, Any]:
    return {
        "lo": value.lo.detach().cpu().tolist(),
        "hi": value.hi.detach().cpu().tolist(),
        "lo_hex": tensor_hex(value.lo),
        "hi_hex": tensor_hex(value.hi),
        "width": (value.hi - value.lo).detach().cpu().tolist(),
    }


def _incoming_interval(value: Any | None) -> dict[str, Any] | None:
    if value is None:
        return None
    return _interval_dict(value)


def _state_record(
    control: str,
    boundary: int,
    time_value: float,
    current: TMVector,
    normal_state: FlowstarNormalFlowpipeState,
    incoming: Any | None,
    *,
    carrier_relation: str = "not_applicable",
) -> dict[str, Any]:
    structured = normal_state.structured_remainder_state
    q = _tmvector_without_remainder(normal_state.tmv_right)
    q_lo, q_hi = _box_tensor(q.range_box())
    if isinstance(structured, StructuredRemainderState):
        ordinary = OutwardIntervalTensor(
            structured.ordinary_rem_lo,
            structured.ordinary_rem_hi,
        )
        columns = structured_column_contributions(structured)
        structured_total = columns.sum(dim=1)
        total_remainder = materialize_structured_remainder(structured)
        live_columns = [
            {
                "slot": slot,
                "source_boundary_index": int(structured.source_boundary_index[0, slot]),
                "source_id": int(structured.source_id[0, slot]),
                "source_occurrence_index": int(structured.source_occurrence_index[0, slot]),
                "contribution": _interval_dict(
                    OutwardIntervalTensor(
                        columns.lo[:, slot, :],
                        columns.hi[:, slot, :],
                    )
                ),
            }
            for slot in range(structured.capacity)
            if bool(structured.active[0, slot])
        ]
        inverse_scale = structured.inverse_scale
        active_columns = int(structured.active.sum().item())
    else:
        ordinary_lo, ordinary_hi = _tmvector_remainder_tensor(normal_state.tmv_right)
        ordinary = OutwardIntervalTensor(ordinary_lo, ordinary_hi)
        structured_total = OutwardIntervalTensor.zeros_like(ordinary_lo)
        total_remainder = ordinary
        live_columns = []
        scale = torch.tensor([normal_state.scales], dtype=torch.float64)
        inverse_scale = torch.where(scale == 0, torch.ones_like(scale), 1.0 / scale)
        active_columns = 0
    total_normal = OutwardIntervalTensor(q_lo, q_hi).add(total_remainder)
    scale = torch.tensor([normal_state.scales], dtype=torch.float64)
    physical_delta = normal_interval_to_physical(
        total_normal.lo,
        total_normal.hi,
        forward_scale=scale,
    )
    center = torch.tensor([normal_state.center], dtype=torch.float64)
    total_physical = physical_delta.add(OutwardIntervalTensor.point(center))
    incoming_stats = dict(incoming.flowstar_normal_stats or {}) if incoming is not None else {}
    ledger = incoming.validated_remainder_ledger if incoming is not None else None
    source_intervals = (
        {
            category: _interval_dict(OutwardIntervalTensor(*ledger.entries[category]))
            for category in REMAINDER_LEDGER_CATEGORIES
        }
        if ledger is not None
        else {}
    )
    stage_record = (
        incoming.boundary_attribution_record.as_dict()
        if incoming is not None and incoming.boundary_attribution_record is not None
        else None
    )
    return {
        "schema": "torch_tm_flowpipe_s1_boundary_prestate_comparator_v1",
        "control": control,
        "boundary": int(boundary),
        "time": float(time_value),
        "time_hex": float(time_value).hex(),
        "current": _encode_tmvector(current),
        "normal_state": _encode_normal_state(normal_state),
        "current_hashes": tmvector_hashes(current),
        "right_polynomial_hashes": tmvector_hashes(q),
        "center": list(normal_state.center),
        "center_hex": [float(value).hex() for value in normal_state.center],
        "forward_scale": list(normal_state.scales),
        "forward_scale_hex": [float(value).hex() for value in normal_state.scales],
        "inverse_scale": inverse_scale.detach().cpu().tolist(),
        "inverse_scale_hex": tensor_hex(inverse_scale),
        "right_polynomial_range": _interval_dict(OutwardIntervalTensor(q_lo, q_hi)),
        "ordinary_remainder": _interval_dict(ordinary),
        "live_columns": live_columns,
        "active_columns": active_columns,
        "structured_total": _interval_dict(structured_total),
        "materialized_total": _interval_dict(total_remainder),
        "total_normalized_right_map": _interval_dict(total_normal),
        "total_physical_right_map": _interval_dict(total_physical),
        "endpoint_published_remainder": _incoming_interval(
            incoming.endpoint_total_remainder if incoming is not None else None
        ),
        "tube_published_remainder": _incoming_interval(
            incoming.tube_total_remainder if incoming is not None else None
        ),
        "raw_picard_candidate_remainder": (
            {
                "lo": incoming.picard_image_remainder[0],
                "hi": incoming.picard_image_remainder[1],
            }
            if incoming is not None and incoming.picard_image_remainder is not None
            else None
        ),
        "subset_margin": incoming.subset_margin if incoming is not None else None,
        "outward_renormalization_count": int(
            incoming_stats.get("structured_outward_renormalization_count", 0)
        ),
        "validated_decomposition_padding": (
            {
                "lo": incoming.validated_remainder_decomposition.padding_lo.detach().cpu().tolist(),
                "hi": incoming.validated_remainder_decomposition.padding_hi.detach().cpu().tolist(),
            }
            if incoming is not None and incoming.validated_remainder_decomposition is not None
            else None
        ),
        "source_ledger": source_intervals,
        "stage_ledger": stage_record,
        "carrier_same_set_relation": carrier_relation,
    }


def _exact_carrier_split_remerge(
    normal_state: FlowstarNormalFlowpipeState,
) -> tuple[FlowstarNormalFlowpipeState, str]:
    """C2: move interval endpoints through a carrier without arithmetic."""
    lo, hi = _tmvector_remainder_tensor(normal_state.tmv_right)
    carrier_lo = lo.clone()
    carrier_hi = hi.clone()
    remerged = _tmvector_with_remainder_tensor(
        normal_state.tmv_right,
        carrier_lo,
        carrier_hi,
    )
    relation = "equal" if torch.equal(lo, carrier_lo) and torch.equal(hi, carrier_hi) else "incomparable"
    return replace(normal_state, tmv_right=remerged), relation


def _initialize(control: str) -> tuple[TMVector, FlowstarNormalFlowpipeState]:
    if control in {"C3", "C4", "L2"}:
        return runner._initialize_structured_lane()
    initial = [
        Interval(*bounds)
        for bounds in runner.CONTRACT["canonical_system_spec"]["initial_box"]
    ]
    normal = FlowstarNormalFlowpipeState.from_initial_box(
        initial,
        runner.CONTRACT["requested_order"],
    )
    return normal.normalized_initial_tm(runner.CONTRACT["requested_order"]), normal


def replay_control(
    control: str,
    schedule: Mapping[str, Any],
    *,
    max_attempt_index: int = 164,
) -> dict[str, Any]:
    if control not in CONTROL_NAMES:
        raise ValueError(control)
    current, normal_state = _initialize(control)
    policy = _policy()
    ode = runner.PolynomialODE.from_system_spec(runner.CONTRACT["canonical_system_spec"])
    states = [_state_record(control, 0, 0.0, current, normal_state, None)]
    attempts: list[dict[str, Any]] = []
    status = (
        "completed_boundary_164_full_h"
        if int(max_attempt_index) >= 164
        else "test_attempt_limit_reached"
    )
    failure_boundary = None
    for frozen in schedule["rows"]:
        attempt_index = int(frozen["attempt_index"])
        if attempt_index > int(max_attempt_index):
            break
        diagnostics: list[dict[str, Any]] = []
        full_h = attempt_index == 164
        proposed_h = float(frozen["h_attempted"]["value"])
        segment = runner._run_lane_step(
            ode,
            current,
            normal_state,
            lane="L1" if control in {"C3", "C4"} else "L2" if control == "L2" else "L0",
            h=proposed_h,
            h_min=proposed_h if full_h else None,
            h_max=proposed_h if full_h else None,
            max_validation_attempts=1 if full_h else 2,
            structured_allow_outward_renormalization=control != "C3",
            policy=policy,
            diagnostics=diagnostics,
            diagnostics_context={
                "control": control,
                "segment_index": attempt_index,
                "t_before": frozen["t_before"]["value"],
            },
        )
        decision = (
            "accepted"
            if segment.status == "validated" and segment.reset_tm is not None
            else "rejected"
        )
        attempts.append(
            {
                "control": control,
                "attempt_index": attempt_index,
                "boundary_before": int(frozen["accepted_boundary_index_before"]),
                "t_before": float(frozen["t_before"]["value"]),
                "h_attempted": proposed_h,
                "h_attempted_hex": proposed_h.hex(),
                "full_h_first_validator_only": full_h,
                "decision": decision,
                "returned_h": float(segment.h),
                "returned_h_hex": float(segment.h).hex(),
                "step_rejections": int(segment.step_rejections),
                "subset_margin": segment.subset_margin,
                "message": segment.message,
                "diagnostics": diagnostics,
            }
        )
        if full_h:
            break
        expected_h = frozen["h_accepted"]
        schedule_match = (
            decision == "accepted"
            and expected_h is not None
            and float(segment.h).hex() == expected_h["hex"]
            and int(segment.step_rejections) == int(frozen["rejection_count_before_acceptance"])
        )
        if not schedule_match:
            status = "domain_gate_failure" if "normalized right-map total leaves" in segment.message else "control_rejected_before_milestone"
            failure_boundary = int(frozen["accepted_boundary_index_before"])
            break
        assert segment.reset_tm is not None and segment.flowstar_normal_state is not None
        next_current = segment.reset_tm
        next_normal = segment.flowstar_normal_state
        carrier_relation = "not_applicable"
        if control in {"C3", "C4"}:
            structured = next_normal.structured_remainder_state
            assert isinstance(structured, StructuredRemainderState)
            structured, _ = runner._materialize_every_boundary(structured)
            next_normal = replace(next_normal, structured_remainder_state=structured)
        elif control == "C2":
            next_normal, carrier_relation = _exact_carrier_split_remerge(next_normal)
        current, normal_state = next_current, next_normal
        boundary_after = int(frozen["accepted_boundary_index_after"])
        states.append(
            _state_record(
                control,
                boundary_after,
                float(frozen["t_after"]["value"]),
                current,
                normal_state,
                segment,
                carrier_relation=carrier_relation,
            )
        )
    return {
        "control": control,
        "status": status,
        "failure_boundary": failure_boundary,
        "states": states,
        "attempts": attempts,
    }


def _signature(record: Mapping[str, Any], field: str) -> Any:
    if field == "coefficient_hash":
        return record["right_polynomial_hashes"]["coefficient_sha256"]
    if field == "center":
        return record["center_hex"]
    if field == "scale":
        return record["forward_scale_hex"]
    if field == "materialized_total":
        return (record["materialized_total"]["lo_hex"], record["materialized_total"]["hi_hex"])
    if field == "physical_hull":
        return (
            record["total_physical_right_map"]["lo_hex"],
            record["total_physical_right_map"]["hi_hex"],
        )
    if field == "renormalization":
        return int(record["outward_renormalization_count"])
    raise KeyError(field)


def _state_map(result: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    return {int(row["boundary"]): row for row in result["states"]}


def _first_state_difference(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    field: str,
) -> int | None:
    left_map = _state_map(left)
    right_map = _state_map(right)
    for boundary in sorted(set(left_map) & set(right_map)):
        if _signature(left_map[boundary], field) != _signature(right_map[boundary], field):
            return boundary
    return None


def _first_attempt_margin_difference(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> int | None:
    left_map = {int(row["attempt_index"]): row for row in left["attempts"]}
    right_map = {int(row["attempt_index"]): row for row in right["attempts"]}
    for attempt in sorted(set(left_map) & set(right_map)):
        if left_map[attempt]["subset_margin"] != right_map[attempt]["subset_margin"]:
            return attempt
    return None


def _first_any_numeric_difference(left: Mapping[str, Any], right: Mapping[str, Any]) -> int | None:
    fields = ("coefficient_hash", "center", "scale", "materialized_total", "physical_hull")
    values = [
        value
        for value in (_first_state_difference(left, right, field) for field in fields)
        if value is not None
    ]
    return min(values) if values else None


def _component_values(interval: Mapping[str, Any], component: int) -> tuple[float, float]:
    lo = interval["lo"]
    hi = interval["hi"]
    while lo and isinstance(lo[0], list):
        lo = lo[0]
        hi = hi[0]
    return float(lo[component]), float(hi[component])


def _comparison_rows(results: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for left_name, right_name in (("C0", "C4"), ("C0", "L2"), ("C4", "L2")):
        left = _state_map(results[left_name])
        right = _state_map(results[right_name])
        for boundary in sorted(set(left) & set(right)):
            for component in range(2):
                for field in ("center", "forward_scale"):
                    scalar = compare_binary64_scalar(
                        left[boundary][field][component],
                        right[boundary][field][component],
                    )
                    rows.append(
                        {
                            "left": left_name,
                            "right": right_name,
                            "boundary": boundary,
                            "object": field,
                            "component": component,
                            **scalar,
                        }
                    )
                for field in (
                    "right_polynomial_range",
                    "ordinary_remainder",
                    "structured_total",
                    "materialized_total",
                    "total_normalized_right_map",
                    "total_physical_right_map",
                ):
                    left_lo, left_hi = _component_values(left[boundary][field], component)
                    right_lo, right_hi = _component_values(right[boundary][field], component)
                    rows.append(
                        {
                            "left": left_name,
                            "right": right_name,
                            "boundary": boundary,
                            "object": field,
                            "component": component,
                            **compare_interval(left_lo, left_hi, right_lo, right_hi),
                        }
                    )
    return rows


def _assert_known_margins(results: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    attempts = {
        name: {int(row["attempt_index"]): row for row in results[name]["attempts"]}
        for name in ("C0", "C4", "L2")
    }
    expected = {
        ("C0", 163): 2.60697659917348e-5,
        ("C4", 163): 1.7291650118437743e-5,
        ("L2", 163): 1.7363995494671766e-5,
        ("C0", 164): 8.058292550874906e-6,
        ("L2", 164): -3.773875528686747e-6,
    }
    observed: dict[str, float] = {}
    for (name, attempt), value in expected.items():
        actual = float(attempts[name][attempt]["subset_margin"][0][1])
        if actual != value:
            raise RuntimeError(
                f"BOUNDARY164_REPLAY_NOT_REPRODUCIBLE: {name} step {attempt}: {actual!r} != {value!r}"
            )
        observed[f"{name}_step_{attempt}_y_margin"] = actual
    observed["C4_step_164_y_margin"] = float(
        attempts["C4"][164]["subset_margin"][0][1]
    )
    return observed


def _exact_state_sequence(result: Mapping[str, Any]) -> list[str]:
    return [
        json.dumps(
            {"current": row["current"], "normal_state": row["normal_state"]},
            sort_keys=True,
            separators=(",", ":"),
        )
        for row in result["states"]
    ]


def _attempt_decision_sequence(result: Mapping[str, Any]) -> list[Any]:
    fields = (
        "attempt_index",
        "h_attempted_hex",
        "decision",
        "returned_h_hex",
        "step_rejections",
        "subset_margin",
        "message",
    )
    return [tuple(row[field] for field in fields) for row in result["attempts"]]


def audit(schedule_path: Path, output_dir: Path) -> dict[str, Any]:
    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=False)
    results = {name: replay_control(name, schedule) for name in CONTROL_NAMES}
    for name, result in results.items():
        _jsonl(output_dir / name / "boundary_records.jsonl", result["states"])
        _jsonl(output_dir / name / "attempt_records.jsonl", result["attempts"])
        _json(
            output_dir / name / "summary.json",
            {
                key: value
                for key, value in result.items()
                if key not in {"states", "attempts"}
            }
            | {
                "recorded_boundaries": len(result["states"]),
                "recorded_attempts": len(result["attempts"]),
            },
        )

    c0_signatures = _exact_state_sequence(results["C0"])
    c1_signatures = _exact_state_sequence(results["C1"])
    c2_signatures = _exact_state_sequence(results["C2"])
    c1_bit_exact = (
        c0_signatures == c1_signatures
        and _attempt_decision_sequence(results["C0"])
        == _attempt_decision_sequence(results["C1"])
    )
    c2_bit_exact = c0_signatures == c2_signatures and all(
        row["carrier_same_set_relation"] in {"not_applicable", "equal"}
        for row in results["C2"]["states"]
    )
    if not c1_bit_exact:
        raise RuntimeError("C1_DIAGNOSTIC_SIDE_EFFECT_STOP")
    known_margins = _assert_known_margins(results)
    first = {
        "first_coefficient_hash_difference_boundary": _first_state_difference(results["C0"], results["C4"], "coefficient_hash"),
        "first_center_difference_boundary": _first_state_difference(results["C0"], results["C4"], "center"),
        "first_scale_hex_difference_boundary": _first_state_difference(results["C0"], results["C4"], "scale"),
        "first_materialized_total_difference_boundary": _first_state_difference(results["C0"], results["C4"], "materialized_total"),
        "first_physical_hull_difference_boundary": _first_state_difference(results["C0"], results["C4"], "physical_hull"),
        "first_subset_margin_difference_attempt": _first_attempt_margin_difference(results["C0"], results["C4"]),
        "first_outward_renormalization_difference_boundary": _first_state_difference(results["C0"], results["C4"], "renormalization"),
        "first_L1_L2_numeric_difference_boundary": _first_any_numeric_difference(results["C4"], results["L2"]),
    }
    lane_aliases = {"L0": results["C0"], "L1": results["C4"], "L2": results["L2"]}
    for name, boundary_value in first.items():
        if boundary_value is None:
            continue
        boundary = int(boundary_value)
        snapshots = {}
        for lane, result in lane_aliases.items():
            state_rows = _state_map(result)
            snapshots[lane] = {
                str(index): state_rows.get(index)
                for index in (boundary - 1, boundary, boundary + 1)
                if index >= 0
            }
        _json(output_dir / "first_divergence_snapshots" / f"{name}.json", snapshots)

    causal_rows: list[dict[str, Any]] = []
    for control, result in results.items():
        state_rows = _state_map(result)
        for boundary in MILESTONES:
            row = state_rows.get(boundary)
            causal_rows.append(
                {
                    "control": control,
                    "boundary": boundary,
                    "status": "recorded" if row is not None else "not_reached_after_stop",
                    "control_status": result["status"],
                    "failure_boundary": result["failure_boundary"],
                    "coefficient_hash": _signature(row, "coefficient_hash") if row is not None else None,
                    "physical_hull": row["total_physical_right_map"] if row is not None else None,
                }
            )
    comparisons = _comparison_rows(results)
    _jsonl(output_dir / "prestate_comparisons.jsonl", comparisons)
    _jsonl(output_dir / "causal_ladder.jsonl", causal_rows)
    _json(output_dir / "first_divergence.json", first)
    summary = {
        "schema": "torch_tm_flowpipe_s1_boundary_drift_audit_v1",
        "controls": {
            name: {
                "status": result["status"],
                "failure_boundary": result["failure_boundary"],
                "recorded_boundaries": len(result["states"]),
            }
            for name, result in results.items()
        },
        "C1_bit_exact_C0": c1_bit_exact,
        "C2_bit_exact_C0": c2_bit_exact,
        "C2_same_set_relation": "equal" if c2_bit_exact else "incomparable",
        "known_margins": known_margins,
        "first_divergence": first,
    }
    _json(output_dir / "summary.json", summary)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = audit(args.schedule.resolve(), args.output_dir.resolve())
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
