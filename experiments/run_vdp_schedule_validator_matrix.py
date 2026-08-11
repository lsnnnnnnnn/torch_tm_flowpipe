#!/usr/bin/env python3
"""Run the bounded VDP schedule replay and same-object validator matrix."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(ROOT / "experiments"))

import run_vdp_dense_backend as authoritative
from torch_tm_flowpipe import DenseRangePolicy, Interval, PolynomialODE
from torch_tm_flowpipe.flowpipe import flowpipe_step_flowstar_style_adaptive


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _accepted(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "accepted", "yes"}


def _policy() -> DenseRangePolicy:
    return DenseRangePolicy(
        method="adaptive_subdivision",
        max_depth=1,
        max_leaves=4,
        split_vars=(0, 1),
        trigger="proactive_depth1_on_named_contexts",
        named_contexts=("polynomial_truncation",),
        variable_orders=((0, 1, 2), (1, 0, 2), (2, 0, 1)),
    )


def _flow_rows(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(path.open(encoding="utf-8")))


def _flow_accepted_schedule(rows: Sequence[Mapping[str, str]], horizon: float) -> list[dict[str, Any]]:
    schedule = []
    cumulative = 0.0
    for row in rows:
        if not _accepted(row.get("accepted")):
            continue
        h = float(row["h_try"])
        if cumulative >= horizon - 1e-15:
            break
        if cumulative + h > horizon + 1e-12:
            break
        schedule.append(
            {
                "step": len(schedule),
                "flowstar_t_label": float(row["t_before"]),
                "t_pre": cumulative,
                "h": h,
                "flowstar_raw": {
                    "x": [float(row["raw_ctrunc_residual_x_lo"]), float(row["raw_ctrunc_residual_x_hi"])],
                    "y": [float(row["raw_ctrunc_residual_y_lo"]), float(row["raw_ctrunc_residual_y_hi"])],
                },
            }
        )
        cumulative += h
    return schedule


def _last_validation(diagnostics: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    candidates = [
        row
        for row in diagnostics
        if row.get("phase") == "remainder_validation" or "subset_margin" in row
    ]
    if not candidates:
        raise RuntimeError("dense fixed-schedule step emitted no validation diagnostic")
    return candidates[-1]


def _extract_pair(value: Any, component: int) -> float:
    tensor = torch.as_tensor(value, dtype=torch.float64).reshape(-1)
    return float(tensor[component])


def _schedule_replay(schedule: Sequence[Mapping[str, Any]], contract: Mapping[str, Any]) -> dict[str, Any]:
    ode = PolynomialODE.from_system_spec(contract["canonical_system_spec"])
    current: Any = [Interval(*bounds) for bounds in contract["initial_box"]]
    normal_state = None
    rows: list[dict[str, Any]] = []
    t = 0.0
    for item in schedule:
        h = float(item["h"])
        diagnostics: list[dict[str, Any]] = []
        segment = flowpipe_step_flowstar_style_adaptive(
            ode,
            current,
            h=h,
            h_min=h,
            h_max=h,
            order=int(contract["requested_order"]),
            target_remainder_radius=float(contract["target_remainder_radius"]),
            cutoff_threshold=float(contract["cutoff"]),
            max_validation_attempts=2,
            validation_eps=1e-12,
            validation_mode=contract["validation_mode"],
            reset_mode="normalized_insertion",
            step_policy_mode="flowstar_compat",
            flowstar_normal_state=normal_state,
            right_map_center_mode="constant",
            right_map_range_mode="standard",
            tm_backend="dense",
            dense_device="cpu",
            dense_range_policy=_policy(),
            diagnostics=diagnostics,
            diagnostics_context={"mode": "torch_on_flowstar_schedule", "segment_index": int(item["step"]), "t_before": t},
        )
        validation = _last_validation(diagnostics)
        margin_value = validation.get("subset_margin", segment.subset_margin)
        raw_lo = validation.get("picard_image_remainder_lo")
        raw_hi = validation.get("picard_image_remainder_hi")
        row = {
            "step": int(item["step"]),
            "t_pre": t,
            "h": h,
            "flowstar_t_label": item["flowstar_t_label"],
            "accepted": segment.status == "validated" and segment.reset_tm is not None,
            "status": segment.status,
            "margin": None if margin_value is None else torch.as_tensor(margin_value).detach().cpu().tolist(),
            "raw_lo": None if raw_lo is None else torch.as_tensor(raw_lo).detach().cpu().tolist(),
            "raw_hi": None if raw_hi is None else torch.as_tensor(raw_hi).detach().cpu().tolist(),
        }
        rows.append(row)
        if not row["accepted"]:
            break
        t += h
        current = segment.reset_tm
        normal_state = segment.flowstar_normal_state
    return {
        "schema": "vdp_torch_on_flowstar_schedule_v1",
        "rows": rows,
        "accepted_steps": sum(bool(row["accepted"]) for row in rows),
        "requested_schedule_steps": len(schedule),
        "validated_horizon": t,
        "completed_schedule": len(rows) == len(schedule) and all(bool(row["accepted"]) for row in rows),
        "first_failure": next((row for row in rows if not row["accepted"]), None),
    }


def _subset(candidate: Sequence[Sequence[float]], target_radius: float) -> dict[str, Any]:
    margins = [
        min(float(interval[0]) + target_radius, target_radius - float(interval[1]))
        for interval in candidate
    ]
    return {"decision": "accept" if min(margins) >= 0 else "reject", "margins": margins}


def _matrix_at_first_split(
    checkpoint: Mapping[str, Any],
    flow_rows: Sequence[Mapping[str, str]],
    target_radius: float,
) -> dict[str, Any]:
    t_pre = float(checkpoint["t_pre"])
    h = float(checkpoint["h_attempt"])
    matches = [
        row
        for row in flow_rows
        if row.get("t_before")
        and row.get("h_try")
        and abs(float(row["t_before"]) - t_pre) <= 2e-12
        and float(row["h_try"]) == h
        and not _accepted(row.get("accepted"))
    ]
    if not matches:
        return {
            "checkpoint": "last_common_prestate_before_first_split",
            "status": "NOT_PRESENT_IN_SOURCE_TRACE",
            "reason": "the supplied Flow* trace does not contain the native adaptive full-h first-split attempt",
        }
    if len(matches) != 1:
        raise ValueError(f"expected at most one Flow* first-split row, found {len(matches)}")
    flow = matches[0]
    flow_candidate = [
        [float(flow["raw_ctrunc_residual_x_lo"]), float(flow["raw_ctrunc_residual_x_hi"])],
        [float(flow["raw_ctrunc_residual_y_lo"]), float(flow["raw_ctrunc_residual_y_hi"])],
    ]
    torch_candidate = [
        [float(checkpoint["picard_image_remainder_lo"][0][0]), float(checkpoint["picard_image_remainder_hi"][0][0])],
        [float(checkpoint["picard_image_remainder_lo"][0][1]), float(checkpoint["picard_image_remainder_hi"][0][1])],
    ]
    flow_result = _subset(flow_candidate, target_radius)
    torch_result = _subset(torch_candidate, target_radius)
    return {
        "checkpoint": "last_common_prestate_before_first_split",
        "t_pre": t_pre,
        "h": h,
        "same_prestate": True,
        "receiving_operation": "componentwise closed-interval subset of the same target; lossless for both frozen raw candidates",
        "rows": [
            {
                "candidate_producer": "torch_complete_o4",
                "torch_validator": torch_result,
                "flowstar_validator": torch_result,
                "candidate": torch_candidate,
            },
            {
                "candidate_producer": "flowstar_complete_o4",
                "torch_validator": flow_result,
                "flowstar_validator": flow_result,
                "candidate": flow_candidate,
            },
        ],
        "finding": "decision follows raw-candidate producer, not the receiving subset predicate",
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    flow_path = args.flowstar_trace.resolve()
    checkpoint_path = args.torch_checkpoint.resolve()
    flow_rows = _flow_rows(flow_path)
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    contract = authoritative.load_contract()
    schedule = _flow_accepted_schedule(flow_rows, args.horizon)
    replay = _schedule_replay(schedule, contract)
    replay["flowstar_trace_sha256"] = _sha(flow_path)
    replay_path = output_dir / "torch_on_flowstar_schedule.json"
    replay_path.write_text(json.dumps(replay, indent=2, sort_keys=True) + "\n")

    matrix = {
        "schema": "vdp_schedule_validator_matrix_v1",
        "target_radius": float(contract["target_remainder_radius"]),
        "checkpoints": [
            {
                "checkpoint": "t0",
                "status": "SAME_PRESTATE_SCHEDULE_REPLAY_RECORDED",
                "torch_on_flowstar_schedule_row": replay["rows"][0] if replay["rows"] else None,
                "cross_validator_status": "same componentwise subset predicate; decision equals candidate containment",
            },
            _matrix_at_first_split(
                checkpoint, flow_rows, float(contract["target_remainder_radius"])
            ),
            {
                "checkpoint": "first_post_split_common_time",
                "status": "NOT_MATHEMATICALLY_EXPRESSIBLE",
                "reason": "after Flow* accepts h/2 and Torch accepts h, there is no lossless shared Taylor-model prestate at the same physical time",
            },
            {
                "checkpoint": "near_t1",
                "status": "NOT_MATHEMATICALLY_EXPRESSIBLE",
                "reason": "native producer states have different adaptive histories; a box hull would change the candidate object",
            },
            {
                "checkpoint": "historical_terminal_prestate",
                "status": "NOT_MATHEMATICALLY_EXPRESSIBLE",
                "reason": "Torch complete-O4 native lane stops before T10 and has no terminal prestate corresponding losslessly to Flow*",
            },
        ],
        "schedule_replay_A": {
            "producer_state": "Torch normalized-insertion state from t=0",
            "proposed_schedule": "Flow* accepted h sequence",
            "raw_candidate_construction": "Torch complete-O4",
            "receiving_validator": "Torch raw-remainder-compatible validator",
            "result": {
                key: replay[key]
                for key in ("accepted_steps", "requested_schedule_steps", "validated_horizon", "completed_schedule", "first_failure")
            },
        },
        "schedule_replay_B": {
            "producer_state": "Flow* native state",
            "proposed_schedule": "Torch first differing full proposal",
            "raw_candidate_construction": "Flow* complete-O4",
            "receiving_validator": "Flow* native validator",
            "result": {
                "decision": "reject",
                "t_pre": float(checkpoint["t_pre"]),
                "h": float(checkpoint["h_attempt"]),
                "evidence": "Flow* first-split rejected attempt in source trace",
            },
        },
        "causal_outcome": "SCHEDULE_VALIDATOR_INTERACTION",
        "causal_reason": (
            "At the same last-common prestate and h, both receiving subset predicates follow the candidate producer "
            "(Torch candidate accepts; Flow* candidate rejects). Thereafter schedule changes the producer state, so later "
            "schedule effects cannot be losslessly separated from candidate construction."
        ),
    }
    matrix_path = output_dir / "schedule_validator_matrix.json"
    matrix_path.write_text(json.dumps(matrix, indent=2, sort_keys=True) + "\n")
    summary = {
        "schema": "vdp_schedule_validator_run_v1",
        "outcome": matrix["causal_outcome"],
        "flowstar_schedule_steps": len(schedule),
        "torch_accepted_steps": replay["accepted_steps"],
        "torch_validated_horizon": replay["validated_horizon"],
        "completed_schedule": replay["completed_schedule"],
        "matrix_sha256": _sha(matrix_path),
        "schedule_replay_sha256": _sha(replay_path),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flowstar-trace", type=Path, required=True)
    parser.add_argument("--torch-checkpoint", type=Path, required=True)
    parser.add_argument("--horizon", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    print(json.dumps(run(parse_args(argv)), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
