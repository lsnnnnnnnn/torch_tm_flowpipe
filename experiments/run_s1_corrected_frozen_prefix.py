#!/usr/bin/env python3
"""Replay S1 total-delta on the 307 historical accepted step sizes."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import time
from typing import Any, Mapping, Sequence

import torch

from torch_tm_flowpipe.structured_remainder import StructuredRemainderState
from torch_tm_flowpipe.terminal_checkpoint import (
    load_terminal_checkpoint,
    save_terminal_checkpoint,
)


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "experiments/run_s1_prefix_complete_o4.py"
RUNNER_SPEC = importlib.util.spec_from_file_location(
    "run_s1_prefix_for_corrected_gate",
    RUNNER_PATH,
)
assert RUNNER_SPEC is not None and RUNNER_SPEC.loader is not None
runner = importlib.util.module_from_spec(RUNNER_SPEC)
RUNNER_SPEC.loader.exec_module(runner)


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
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


def _git_value(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _decision(segment) -> str:
    return (
        "accepted"
        if segment.status == "validated" and segment.reset_tm is not None
        else "rejected"
    )


def _candidate_gates(segment, boundary: int) -> dict[str, Any]:
    state = segment.structured_state_after
    result = segment.structured_boundary_result
    record = segment.boundary_attribution_record
    stats = segment.flowstar_normal_stats or {}
    if not isinstance(state, StructuredRemainderState) or result is None or record is None:
        return {"passed": False, "failure": "candidate poststate payload missing"}
    stages = {stage.stage: stage for stage in record.stages}
    active_identities = [
        (
            int(state.source_boundary_index[0, slot]),
            int(state.source_id[0, slot]),
            int(state.source_occurrence_index[0, slot]),
        )
        for slot in range(state.capacity)
        if bool(state.active[0, slot])
    ]
    checks = {
        "candidate_name": stats.get("structured_candidate")
        == "normalized_insertion_structured_total_delta_k16",
        "image_contract": stats.get("structured_image_contract") == "total_delta",
        "raw_picard_target_unchanged": stats.get(
            "structured_raw_picard_target_changed"
        )
        is False,
            "source_ledger_raw_compatible": bool(
                segment.validated_remainder_decomposition is not None
                and torch.all(
                    segment.validated_remainder_decomposition.contains_image
                )
            ),
        "candidate_total_contains_canonical_target": bool(
            torch.all(stages["B13"].lo <= stages["B0"].lo)
            and torch.all(stages["B13"].hi >= stages["B0"].hi)
        ),
        "conservation": bool(torch.all(result.conservation_mask)),
        "source_decomposition": bool(torch.all(result.source_decomposition_mask)),
        "unique_live_source_ownership": len(active_identities)
        == len(set(active_identities)),
        "endpoint_publication": bool(
            segment.endpoint_publication_mask is not None
            and torch.all(segment.endpoint_publication_mask)
        ),
        "tube_publication": bool(
            segment.tube_publication_mask is not None
            and torch.all(segment.tube_publication_mask)
        ),
        "endpoint_contained_in_tube": bool(
            stats.get("structured_published_endpoint_in_tube")
        ),
        "right_map_domain": bool(stats.get("structured_total_self_map_containment")),
        "accepted_boundary_index": state.accepted_boundary_index == boundary,
        "no_fallback_branch": "fallback" not in stats,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "active_columns": int(state.active.sum().item()),
        "event_count": int(state.event_count.sum().item()),
    }


def _save_checkpoint(
    output_dir: Path,
    *,
    current,
    normal_state,
    schedule: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    last = schedule["rows"][306]
    terminal = schedule["rows"][307]
    first_dir = output_dir / "boundary_307_checkpoint"
    first = save_terminal_checkpoint(
        first_dir,
        current=current,
        normal_state=normal_state,
        scheduler={
            "current_time": last["t_after"]["value"],
            "h_next": terminal["h_attempted"]["value"],
            "h_attempted": terminal["h_attempted"]["value"],
            "accepted_segment_count": 307,
            "checkpoint_role": "corrected_frozen_accepted_prefix_prestate",
        },
        contract=runner.CONTRACT,
        provenance=provenance,
    )
    loaded = load_terminal_checkpoint(
        first_dir,
        expected_contract=runner.CONTRACT,
        expected_order=4,
        expected_dtype="float64",
    )
    second_dir = output_dir / "boundary_307_checkpoint_roundtrip"
    second = save_terminal_checkpoint(
        second_dir,
        current=loaded.current,
        normal_state=loaded.normal_state,
        scheduler=loaded.scheduler,
        contract=loaded.contract,
        provenance=loaded.provenance,
    )
    names = ("terminal_state.json", "terminal_state_manifest.json")
    byte_stable = all(
        (first_dir / name).read_bytes() == (second_dir / name).read_bytes()
        for name in names
    )
    return {
        "schema": first["schema"],
        "full_checkpoint_sha256": first["full_checkpoint_sha256"],
        "roundtrip_full_checkpoint_sha256": second["full_checkpoint_sha256"],
        "byte_stable": byte_stable,
    }


def run(
    schedule_path: Path,
    output_dir: Path,
    *,
    max_boundaries: int | None = None,
) -> dict[str, Any]:
    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    if schedule.get("accepted_boundary_count") != 307:
        raise ValueError("corrected frozen gate requires the 307-boundary schedule")
    output_dir.mkdir(parents=True, exist_ok=False)
    provenance = {
        "branch": _git_value("branch", "--show-current"),
        "commit": _git_value("rev-parse", "HEAD"),
        "worktree_status": _git_value("status", "--short"),
        "schedule": str(schedule_path.resolve()),
        "schedule_sha256": runner._sha256(schedule_path),
        "candidate": "normalized_insertion_structured_total_delta_k16",
        "replay_mode": "frozen_accepted_step",
    }
    current, normal_state = runner._initialize_structured_lane(
        "normalized_insertion_structured_total_delta_k16"
    )
    ode = runner.PolynomialODE.from_system_spec(
        runner.CONTRACT["canonical_system_spec"]
    )
    policy = _policy()
    accepted_rows = [
        row for row in schedule["rows"] if row["expected_status"] == "accepted"
    ]
    if max_boundaries is not None:
        accepted_rows = accepted_rows[: int(max_boundaries)]
    rows: list[dict[str, Any]] = []
    scheduler_divergences = 0
    outcome = "CORRECTED_S1_FROZEN_PREFIX_GO"
    failure: dict[str, Any] | None = None
    started = time.monotonic()
    rows_path = output_dir / "accepted_step_records.jsonl"
    with rows_path.open("w", encoding="utf-8") as handle:
        for frozen in accepted_rows:
            boundary = int(frozen["accepted_boundary_index_after"])
            prestate_sha = runner._state_hash(current, normal_state)

            attempted_diagnostics: list[dict[str, Any]] = []
            attempted = runner._run_lane_step(
                ode,
                current,
                normal_state,
                lane="B",
                h=float(frozen["h_attempted"]["value"]),
                policy=policy,
                diagnostics=attempted_diagnostics,
                diagnostics_context={
                    "mode": "historical_attempted_step_diagnostic",
                    "segment_index": int(frozen["attempt_index"]),
                    "t_before": float(frozen["t_before"]["value"]),
                    "boundary": boundary,
                },
                max_validation_attempts=2,
            )
            attempted_prestate_after = runner._state_hash(current, normal_state)
            attempted_decision = _decision(attempted)
            attempted_matches_historical = (
                attempted_decision == "accepted"
                and float(attempted.h).hex() == frozen["h_accepted"]["hex"]
                and int(attempted.step_rejections)
                == int(frozen["rejection_count_before_acceptance"])
            )
            if not attempted_matches_historical:
                scheduler_divergences += 1

            accepted_h = float(frozen["h_accepted"]["value"])
            fixed_diagnostics: list[dict[str, Any]] = []
            segment = runner._run_lane_step(
                ode,
                current,
                normal_state,
                lane="B",
                h=accepted_h,
                h_min=accepted_h,
                h_max=accepted_h,
                max_validation_attempts=2,
                policy=policy,
                diagnostics=fixed_diagnostics,
                diagnostics_context={
                    "mode": "frozen_accepted_step",
                    "segment_index": int(frozen["attempt_index"]),
                    "t_before": float(frozen["t_before"]["value"]),
                    "boundary": boundary,
                },
            )
            fixed_decision = _decision(segment)
            gates = (
                _candidate_gates(segment, boundary)
                if fixed_decision == "accepted"
                else {"passed": False, "failure": segment.message}
            )
            record = {
                "schema": "torch_tm_flowpipe_s1_corrected_accepted_step_v1",
                "boundary": boundary,
                "attempt_index": int(frozen["attempt_index"]),
                "t_before": frozen["t_before"],
                "t_after": frozen["t_after"],
                "prestate_sha256": prestate_sha,
                "historical_attempted_step": {
                    "h": frozen["h_attempted"],
                    "historical_rejections": int(
                        frozen["rejection_count_before_acceptance"]
                    ),
                    "candidate_decision": attempted_decision,
                    "candidate_returned_h": float(attempted.h),
                    "candidate_returned_h_hex": float(attempted.h).hex(),
                    "candidate_rejections": int(attempted.step_rejections),
                    "candidate_message": attempted.message,
                    "matches_historical_scheduler": attempted_matches_historical,
                    "prestate_unchanged": prestate_sha == attempted_prestate_after,
                    "diagnostics": attempted_diagnostics,
                },
                "frozen_accepted_step": {
                    "h": frozen["h_accepted"],
                    "h_min_hex": accepted_h.hex(),
                    "h_max_hex": accepted_h.hex(),
                    "max_validation_attempts": 2,
                    "decision": fixed_decision,
                    "returned_h_hex": float(segment.h).hex(),
                    "step_rejections": int(segment.step_rejections),
                    "message": segment.message,
                    "subset_margin": segment.subset_margin,
                    "diagnostics": fixed_diagnostics,
                },
                "candidate_gates": gates,
            }
            rows.append(record)
            handle.write(json.dumps(_jsonable(record), sort_keys=True) + "\n")
            handle.flush()
            if (
                attempted_prestate_after != prestate_sha
                or fixed_decision != "accepted"
                or float(segment.h).hex() != frozen["h_accepted"]["hex"]
                or int(segment.step_rejections) != 0
                or not gates["passed"]
            ):
                outcome = "CORRECTED_S1_REJECTS_BEFORE_TERMINAL"
                failure = record
                break
            current = segment.reset_tm
            normal_state = segment.flowstar_normal_state

    completed = len(rows)
    checkpoint = None
    if completed == 307 and outcome == "CORRECTED_S1_FROZEN_PREFIX_GO":
        checkpoint = _save_checkpoint(
            output_dir,
            current=current,
            normal_state=normal_state,
            schedule=schedule,
            provenance=provenance,
        )
        if not checkpoint["byte_stable"]:
            outcome = "CORRECTED_S1_REJECTS_BEFORE_TERMINAL"
            failure = {"failure": "boundary-307 checkpoint roundtrip is not exact"}
    elif max_boundaries is not None and completed == len(accepted_rows) and failure is None:
        outcome = "TEST_BOUNDARY_LIMIT_REACHED"

    summary = {
        "schema": "torch_tm_flowpipe_s1_corrected_frozen_prefix_v1",
        "outcome": outcome,
        "candidate": "normalized_insertion_structured_total_delta_k16",
        "replay_mode": "frozen_accepted_step",
        "accepted_step_count": completed if failure is None else completed - 1,
        "processed_row_count": completed,
        "scheduler_divergence_count": scheduler_divergences,
        "all_attempted_diagnostics_immutable": all(
            row["historical_attempted_step"]["prestate_unchanged"] for row in rows
        ),
        "all_candidate_gates_pass": all(
            row["candidate_gates"]["passed"] for row in rows
        ),
        "checkpoint": checkpoint,
        "failure": failure,
        "runtime_s": time.monotonic() - started,
        "provenance": provenance,
    }
    _write_json(output_dir / "summary.json", summary)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-boundaries", type=int)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run(
        args.schedule.resolve(),
        args.output_dir.resolve(),
        max_boundaries=args.max_boundaries,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
