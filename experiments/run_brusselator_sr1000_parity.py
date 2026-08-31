#!/usr/bin/env python3
"""Run the one frozen Torch SR1000 lane and compare its full prefix to stock Flow*."""

from __future__ import annotations

import argparse
from collections import deque
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import resource
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from torch_tm_flowpipe import (  # noqa: E402
    DenseRangePolicy,
    FlowstarNormalFlowpipeState,
    GENERIC_ACCEPTED_BOUNDARY_SYMBOLIC_REMAINDER,
    TMVector,
    accepted_boundary_sr_queue_sha256,
    flowpipe_step_flowstar_style_adaptive,
    save_terminal_checkpoint,
)
from torch_tm_flowpipe.ode_examples import brusselator_ode  # noqa: E402
from torch_tm_flowpipe.safety import intervals_are_finite  # noqa: E402
from torch_tm_flowpipe.terminal_checkpoint import MANIFEST_NAME, PAYLOAD_NAME  # noqa: E402

from experiments.run_brusselator_second_system_torch import (  # noqa: E402
    _advance_local_samples,
    _box_fields,
    _git,
    _jsonable,
    _owner_widths_ok,
    _sample_points,
    _sha256,
    _validation_passed,
    _write_json,
    _write_rows,
)


CONTRACT_PATH = ROOT / "benchmarks/brusselator_terminal_sr1000_contract.json"
CONTRACT = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
CONFIG = CONTRACT["sr1000"]
DECISION = CONTRACT["decision"]
STOCK_SEGMENTS = (
    ROOT
    / "artifacts/runs/brusselator_generic_core_validation_20260827"
    / "raw/flowstar/segments.csv"
)
DEFAULT_TERMINAL_REPLAY = Path(
    "/srv/local/shengenli/brusselator_sr100_terminal_replay_20260828/RESULT.json"
)
INITIAL_DECIMAL = tuple(tuple(pair) for pair in CONFIG["initial_decimal_box"])
INITIAL_FLOAT = tuple(tuple(float(item) for item in pair) for pair in INITIAL_DECIMAL)
ORDER = int(CONFIG["order"])
STEP = float(CONFIG["fixed_step_decimal"])
REQUESTED_STEPS = int(CONFIG["requested_steps"])
HORIZON = float(CONFIG["requested_horizon_decimal"])
REMAINDER_RADIUS = float(CONFIG["target_remainder_decimal"][1])
CUTOFF = float(CONFIG["cutoff_decimal"])
VALIDATION_EPS = float(CONFIG["validation_eps_decimal"])
QUEUE_CAPACITY = int(CONFIG["queue_capacity"])
MATERIAL_THRESHOLD = float(DECISION["material_absolute_threshold_decimal"])
PERSISTENCE = int(DECISION["material_persistence_consecutive_boundaries"])
BOUND_FIELDS = tuple(
    f"{prefix}_{component}_{bound}"
    for prefix in ("endpoint", "tube")
    for component in ("x", "y")
    for bound in ("lo", "hi")
)
CORE_PATHS = (
    "src/torch_tm_flowpipe/accepted_boundary_sr.py",
    "src/torch_tm_flowpipe/symbolic_remainder.py",
    "src/torch_tm_flowpipe/flowpipe.py",
    "src/torch_tm_flowpipe/terminal_checkpoint.py",
    "src/torch_tm_flowpipe/state_equality.py",
)


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _policy() -> DenseRangePolicy:
    policy = CONFIG["dense_range_policy"]
    return DenseRangePolicy(
        method=policy["method"],
        max_depth=int(policy["max_depth"]),
        max_leaves=int(policy["max_leaves"]),
        split_vars=tuple(int(item) for item in policy["split_vars"]),
        trigger=policy["trigger"],
        named_contexts=tuple(policy["named_contexts"]),
        variable_orders=((0, 1, 2), (1, 0, 2), (2, 0, 1)),
    )


def _read_stock() -> list[dict[str, str]]:
    if _sha256(STOCK_SEGMENTS) != CONTRACT["identity"]["flowstar_segments_sha256"]:
        raise RuntimeError("stock Flow* trace hash differs from the frozen contract")
    with STOCK_SEGMENTS.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != REQUESTED_STEPS:
        raise RuntimeError("stock Flow* trace is not the frozen 1000-step T20 run")
    for index, row in enumerate(rows, start=1):
        if int(row["step"]) != index or row["h_hex"] != STEP.hex():
            raise RuntimeError(f"stock Flow* trace schedule mismatch at boundary {index}")
    return rows


def _queue_fields(state: FlowstarNormalFlowpipeState, step: int) -> dict[str, Any]:
    queue = state.symbolic_queue
    if queue is None:
        return {"queue_present": False, "queue_accounting_ok": False}
    remainder = step % QUEUE_CAPACITY
    expected_owners = tuple(range(step - remainder + 1, step + 1)) if remainder else ()
    accounting = (
        queue.owner_schema == "accepted_boundary_sr_v1"
        and queue.max_size == QUEUE_CAPACITY
        and queue.generation == step
        and queue.accepted_boundary_index == step
        and len(queue.J) == remainder
        and queue.reset_count == step // QUEUE_CAPACITY
        and queue.owner_generations == expected_owners
        and queue.owner_boundary_indices == expected_owners
    )
    return {
        "queue_present": True,
        "queue_owner_schema": queue.owner_schema,
        "queue_capacity": queue.max_size,
        "queue_size": len(queue.J),
        "queue_generation": queue.generation,
        "queue_accepted_boundary": queue.accepted_boundary_index,
        "queue_reset_count": queue.reset_count,
        "queue_owner_generations": list(queue.owner_generations),
        "queue_owner_boundaries": list(queue.owner_boundary_indices),
        "queue_hash": accepted_boundary_sr_queue_sha256(queue),
        "queue_accounting_ok": accounting,
    }


def _step(
    current: TMVector,
    state: FlowstarNormalFlowpipeState,
    step: int,
    policy: DenseRangePolicy,
    *,
    validation_mode: str,
    lane_label: str,
    observer_mode: str = "full_evidence",
) -> tuple[Any, list[dict[str, Any]]]:
    diagnostics: list[dict[str, Any]] = []
    segment = flowpipe_step_flowstar_style_adaptive(
        brusselator_ode,
        current,
        h=STEP,
        h_min=STEP,
        h_max=STEP,
        order=ORDER,
        target_remainder_radius=REMAINDER_RADIUS,
        cutoff_threshold=CUTOFF,
        max_validation_attempts=int(CONFIG["max_validation_attempts"]),
        validation_eps=VALIDATION_EPS,
        validation_mode=validation_mode,
        reset_mode=CONFIG["reset_mode"],
        flowstar_normal_state=state,
        flowstar_symbolic_queue_max_size=QUEUE_CAPACITY,
        right_map_range_mode=CONFIG["right_map_range_mode"],
        right_map_center_mode=CONFIG["right_map_center_mode"],
        tm_backend="dense",
        dense_device=CONFIG["device"],
        dense_dtype=torch.float64,
        dense_range_policy=policy,
        diagnostics=diagnostics,
        diagnostics_context={
            "system": "brusselator",
            "lane": lane_label,
            "segment_index": step - 1,
            "t_before": (step - 1) * STEP,
        },
        dense_observer_mode=observer_mode,
    )
    return segment, diagnostics


def _checkpoint(
    directory: Path,
    *,
    current: TMVector,
    state: FlowstarNormalFlowpipeState,
    accepted_steps: int,
    provenance: Mapping[str, Any],
) -> Mapping[str, Any]:
    return save_terminal_checkpoint(
        directory,
        current=current,
        normal_state=state,
        scheduler={
            "current_time_hex": (accepted_steps * STEP).hex(),
            "h_hex": STEP.hex(),
            "accepted_steps": accepted_steps,
            "next_attempt": accepted_steps + 1,
        },
        contract=CONFIG,
        provenance=provenance,
    )


def _checkpoint_equal(left: Path, right: Path) -> bool:
    return all(
        (left / name).read_bytes() == (right / name).read_bytes()
        for name in (PAYLOAD_NAME, MANIFEST_NAME)
    )


def _stock_delta(row: Mapping[str, Any], stock: Mapping[str, str]) -> dict[str, Any]:
    deltas = {
        field: abs(float(row[field]) - float(stock[field])) for field in BOUND_FIELDS
    }
    differing = [
        field
        for field in BOUND_FIELDS
        if row[f"{field}_hex"] != stock[f"{field}_hex"]
    ]
    maximum = max(deltas.values())
    return {
        "stock_bitwise_differing_fields": differing,
        "stock_max_absolute_bound_delta": maximum,
        "stock_material_bound_difference": maximum > MATERIAL_THRESHOLD,
        **{f"stock_absolute_delta_{field}": value for field, value in deltas.items()},
    }


def _first_persistent_material(rows: Sequence[Mapping[str, Any]]) -> int | None:
    flags = [bool(row.get("stock_material_bound_difference")) for row in rows]
    for index in range(0, len(flags) - PERSISTENCE + 1):
        if all(flags[index : index + PERSISTENCE]):
            return int(rows[index]["step"])
    return None


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    replay_result = json.loads(args.terminal_replay.resolve().read_text(encoding="utf-8"))
    if (
        replay_result.get("status") != "C3_TERMINAL_ROLLBACK_CLOSED"
        or replay_result.get("rollback_proved") is not True
        or replay_result.get("terminal_attempt_count") != 1
    ):
        raise RuntimeError("SR1000 run is ordered after a closed one-attempt terminal replay")
    status_text = _git("status", "--short")
    diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD"], cwd=ROOT, check=True, capture_output=True
    ).stdout
    core_unchanged = subprocess.run(
        [
            "git",
            "diff",
            "--quiet",
            CONTRACT["identity"]["generic_core_commit"],
            "HEAD",
            "--",
            *CORE_PATHS,
        ],
        cwd=ROOT,
        check=False,
    ).returncode == 0
    command = {
        "schema": "torch_brusselator_sr1000_command_v1",
        "argv": sys.argv,
        "cwd": str(ROOT),
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("branch", "--show-current"),
        "worktree_status": status_text,
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "contract_sha256": _sha256(CONTRACT_PATH),
        "terminal_replay_result_sha256": _sha256(args.terminal_replay.resolve()),
        "generic_core_unchanged": core_unchanged,
        "validation_mode": args.validation_mode,
        "lane_label": args.lane_label,
    }
    _write_json(output / "command.json", command)
    if status_text:
        raise RuntimeError("SR1000 run requires a clean worktree")

    stock_rows = _read_stock()
    state = FlowstarNormalFlowpipeState.from_exact_decimal_box(INITIAL_DECIMAL, ORDER)
    current = state.normalized_initial_tm(ORDER)
    policy = _policy()
    samples = _sample_points()
    sample_endpoint_violations = 0
    sample_tube_violations = 0
    sample_solver_ok = True
    prefix_lo = [INITIAL_FLOAT[0][0], INITIAL_FLOAT[1][0]]
    prefix_hi = [INITIAL_FLOAT[0][1], INITIAL_FLOAT[1][1]]
    rows: list[dict[str, Any]] = []
    # Each requested boundary after step 1 also needs its immediately preceding
    # poststate so the accepted-boundary composition can be replayed from the
    # exact same live input.
    selected_checkpoint_steps = {1, 2, 3, 99, 100, 199, 200, 299, 300}
    selected_checkpoint_states: dict[
        int, tuple[TMVector, FlowstarNormalFlowpipeState]
    ] = {}
    rolling_checkpoint_states: deque[
        tuple[int, TMVector, FlowstarNormalFlowpipeState]
    ] = deque(maxlen=6)
    rejected = 0
    terminal_message = ""
    first_bitwise_step: int | None = None
    material_candidate_steps: list[int] = []
    provenance = {
        "producer": "torch_brusselator_sr1000",
        "commit": command["commit"],
        "contract_sha256": command["contract_sha256"],
    }
    started = time.perf_counter()
    diagnostics_path = output / "diagnostics.jsonl.gz"
    with gzip.open(diagnostics_path, "wt", encoding="utf-8", newline="\n") as diagnostic_handle:
        for step_index in range(1, REQUESTED_STEPS + 1):
            pre_current = current
            pre_state = state
            pre_queue_hash = (
                accepted_boundary_sr_queue_sha256(state.symbolic_queue)
                if state.symbolic_queue is not None
                else ""
            )
            step_started = time.perf_counter()
            segment, diagnostics = _step(
                current,
                state,
                step_index,
                policy,
                validation_mode=args.validation_mode,
                lane_label=args.lane_label,
            )
            step_runtime = time.perf_counter() - step_started
            for diagnostic_index, diagnostic in enumerate(diagnostics):
                diagnostic_handle.write(
                    json.dumps(
                        {
                            "step": step_index,
                            "diagnostic_index": diagnostic_index,
                            **_jsonable(diagnostic),
                        },
                        sort_keys=True,
                        allow_nan=False,
                    )
                    + "\n"
                )
            validation_passed, validation_records, validation_failures = _validation_passed(
                diagnostics
            )
            accepted = (
                segment.status == "validated"
                and segment.endpoint_raw_tm is not None
                and segment.reset_tm is not None
                and segment.flowstar_normal_state is not None
            )
            endpoint_box = []
            tube_box = []
            if accepted:
                try:
                    endpoint_box = segment.endpoint_raw_tm.range_box()
                    tube_box = segment.tm.range_box()
                    accepted = intervals_are_finite(endpoint_box) and intervals_are_finite(tube_box)
                except Exception as exc:
                    terminal_message = f"range evaluation failed: {exc}"
                    accepted = False
            row: dict[str, Any] = {
                "schema": "torch_brusselator_sr1000_segment_v1",
                "step": step_index,
                "t_before": (step_index - 1) * STEP,
                "t_after": step_index * STEP if accepted else (step_index - 1) * STEP,
                "h": STEP,
                "status": "accepted" if accepted else "rejected",
                "segment_status": segment.status,
                "message": segment.message,
                "validation_passed": validation_passed,
                "validation_record_count": validation_records,
                "validation_failure_count": validation_failures,
                "step_runtime_s": step_runtime,
                "queue_hash_before": pre_queue_hash,
                "endpoint_published": accepted,
            }
            if not accepted:
                rejected += 1
                terminal_message = terminal_message or segment.message or "fixed-step validation rejected"
                terminal_before = output / "terminal_checkpoint_before"
                terminal_after = output / "terminal_checkpoint_after"
                before_manifest = _checkpoint(
                    terminal_before,
                    current=pre_current,
                    state=pre_state,
                    accepted_steps=step_index - 1,
                    provenance=provenance,
                )
                caller_queue_hash_after = (
                    accepted_boundary_sr_queue_sha256(pre_state.symbolic_queue)
                    if pre_state.symbolic_queue is not None
                    else ""
                )
                after_manifest = _checkpoint(
                    terminal_after,
                    current=pre_current,
                    state=pre_state,
                    accepted_steps=step_index - 1,
                    provenance=provenance,
                )
                rollback = (
                    _checkpoint_equal(terminal_before, terminal_after)
                    and before_manifest["full_checkpoint_sha256"]
                    == after_manifest["full_checkpoint_sha256"]
                    and pre_queue_hash == caller_queue_hash_after
                )
                row.update(
                    {
                        "rollback_checkpoint_byte_equal": rollback,
                        "rollback_queue_unchanged": pre_queue_hash == caller_queue_hash_after,
                        "queue_hash_after_attempt": caller_queue_hash_after,
                    }
                )
                rows.append(row)
                _write_rows(output / "segments.csv", rows)
                break

            assert segment.endpoint_raw_tm is not None
            assert segment.reset_tm is not None
            assert segment.flowstar_normal_state is not None
            if step_index <= 10:
                samples, endpoint_violations, tube_violations, solver_ok = _advance_local_samples(
                    samples, endpoint_box, tube_box
                )
                sample_endpoint_violations += endpoint_violations
                sample_tube_violations += tube_violations
                sample_solver_ok &= solver_ok
                row["local_sample_endpoint_violations"] = endpoint_violations
                row["local_sample_tube_violations"] = tube_violations
            for dimension in range(2):
                prefix_lo[dimension] = min(prefix_lo[dimension], float(tube_box[dimension].lo))
                prefix_hi[dimension] = max(prefix_hi[dimension], float(tube_box[dimension].hi))
            row.update(_box_fields("endpoint", endpoint_box))
            row.update(_box_fields("tube", tube_box))
            row.update(
                {
                    "prefix_x_lo": prefix_lo[0],
                    "prefix_x_hi": prefix_hi[0],
                    "prefix_x_width": prefix_hi[0] - prefix_lo[0],
                    "prefix_y_lo": prefix_lo[1],
                    "prefix_y_hi": prefix_hi[1],
                    "prefix_y_width": prefix_hi[1] - prefix_lo[1],
                }
            )
            row.update(_queue_fields(segment.flowstar_normal_state, step_index))
            stats = dict(segment.flowstar_normal_stats or {})
            row["owner_widths_nonnegative_finite"] = _owner_widths_ok(stats)
            for key in (
                "accepted_boundary_sr_current_owner_width_sum",
                "accepted_boundary_sr_current_owner_width_sum_pre_cutoff",
                "accepted_boundary_sr_total_interval_image_width_sum",
                "accepted_boundary_sr_unscaled_roundoff_cutoff_owner_width_sum",
                "propagated_symbolic_width_sum",
                "insertion_truncation_width",
                "insertion_cutoff_width",
                "inserted_endpoint_width_sum",
                "normalized_reset_width_sum",
                "complete_carry_coefficient_sha256",
            ):
                row[key] = stats.get(key, "")
            delta = _stock_delta(row, stock_rows[step_index - 1])
            row.update(delta)
            if delta["stock_bitwise_differing_fields"] and first_bitwise_step is None:
                first_bitwise_step = step_index
            if delta["stock_material_bound_difference"]:
                previous_material = bool(rows and rows[-1].get("stock_material_bound_difference"))
                if not previous_material:
                    material_candidate_steps.append(step_index)
                    _checkpoint(
                        output / f"material_candidate_step_{step_index}_prestate",
                        current=pre_current,
                        state=pre_state,
                        accepted_steps=step_index - 1,
                        provenance=provenance,
                    )
            rows.append(row)
            current = segment.reset_tm
            state = segment.flowstar_normal_state
            if args.capture_c5_checkpoints:
                if step_index in selected_checkpoint_steps:
                    selected_checkpoint_states[step_index] = (current, state)
                rolling_checkpoint_states.append((step_index, current, state))
            if step_index % 25 == 0:
                _write_rows(output / "segments.csv", rows)

    runtime = time.perf_counter() - started
    _write_rows(output / "segments.csv", rows)
    checkpoint_records: list[dict[str, Any]] = []
    if args.capture_c5_checkpoints:
        for step_index, current_at_step, state_at_step in rolling_checkpoint_states:
            selected_checkpoint_states.setdefault(
                step_index, (current_at_step, state_at_step)
            )
        checkpoint_root = output / "accepted_checkpoints"
        for step_index in sorted(selected_checkpoint_states):
            current_at_step, state_at_step = selected_checkpoint_states[step_index]
            checkpoint_dir = checkpoint_root / f"accepted_step_{step_index:04d}"
            manifest = _checkpoint(
                checkpoint_dir,
                current=current_at_step,
                state=state_at_step,
                accepted_steps=step_index,
                provenance={
                    **provenance,
                    "capture_purpose": "brusselator_live_range_c5_same_object_exchange",
                },
            )
            checkpoint_records.append(
                {
                    "accepted_step": step_index,
                    "relative_directory": checkpoint_dir.relative_to(output).as_posix(),
                    "full_checkpoint_sha256": manifest["full_checkpoint_sha256"],
                }
            )
    accepted_rows = [row for row in rows if row["status"] == "accepted"]
    accepted_steps = len(accepted_rows)
    completed = accepted_steps == REQUESTED_STEPS and rejected == 0
    terminal_rollback = (
        not rows
        or rows[-1]["status"] != "rejected"
        or bool(rows[-1].get("rollback_checkpoint_byte_equal"))
    )
    owner_accounting = all(
        bool(row["queue_accounting_ok"]) and bool(row["owner_widths_nonnegative_finite"])
        for row in accepted_rows
    )
    certificate_checks = (
        bool(accepted_rows)
        and all(bool(row["validation_passed"]) for row in accepted_rows)
        and sample_solver_ok
        and sample_endpoint_violations == 0
        and sample_tube_violations == 0
        and owner_accounting
        and terminal_rollback
    )
    first_persistent = _first_persistent_material(accepted_rows)
    capacity_status = (
        DECISION["capacity_sufficient_status"]
        if certificate_checks and completed
        else DECISION["capacity_insufficient_status"]
        if certificate_checks
        else "SR1000_SOUNDNESS_GATE_FAILED"
    )
    summary = {
        "schema": "torch_brusselator_sr1000_summary_v1",
        "status": "completed" if completed else "stopped",
        "message": terminal_message,
        "capacity_reset_decision": capacity_status,
        "validation_mode": args.validation_mode,
        "lane_label": args.lane_label,
        "requested_steps": REQUESTED_STEPS,
        "accepted_steps": accepted_steps,
        "rejected_steps": rejected,
        "fixed_step": STEP,
        "requested_horizon": HORIZON,
        "completed_horizon": accepted_steps * STEP,
        "completed_requested_horizon": completed,
        "order": ORDER,
        "cutoff": CUTOFF,
        "target_remainder_radius": REMAINDER_RADIUS,
        "queue_capacity": QUEUE_CAPACITY,
        "queue_reset_count": state.symbolic_queue.reset_count if state.symbolic_queue else None,
        "owner_accounting_passed": owner_accounting,
        "terminal_rollback_passed": terminal_rollback,
        "sample_solver_ok": sample_solver_ok,
        "sample_endpoint_violations": sample_endpoint_violations,
        "sample_tube_violations": sample_tube_violations,
        "certificate_checks_passed": certificate_checks,
        "common_stock_prefix_steps": accepted_steps,
        "first_bitwise_stock_bound_difference_step": first_bitwise_step,
        "first_persistent_material_stock_bound_difference_step": first_persistent,
        "material_threshold": MATERIAL_THRESHOLD,
        "material_persistence_boundaries": PERSISTENCE,
        "material_candidate_prestate_steps": material_candidate_steps,
        "solver_wall_seconds": runtime,
        "peak_rss_bytes": _peak_rss_bytes(),
        "diagnostics_sha256": _sha256(diagnostics_path),
        "stock_segments_sha256": _sha256(STOCK_SEGMENTS),
        "commit": command["commit"],
        "branch": command["branch"],
        "tracked_diff_sha256": command["tracked_diff_sha256"],
        "worktree_dirty": bool(command["worktree_status"]),
        "contract_sha256": command["contract_sha256"],
        "c5_checkpoint_capture_enabled": bool(args.capture_c5_checkpoints),
        "accepted_checkpoint_records": checkpoint_records,
    }
    _write_json(output / "summary.json", summary)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--terminal-replay", default=DEFAULT_TERMINAL_REPLAY, type=Path)
    parser.add_argument(
        "--validation-mode",
        choices=(
            "flowstar_raw_remainder_compat",
            "flowstar_raw_remainder_compat_refined",
        ),
        default=CONFIG["validation_mode"],
    )
    parser.add_argument("--lane-label", default="torch_generic_sr1000")
    parser.add_argument(
        "--capture-c5-checkpoints",
        action="store_true",
        help=(
            "retain the fixed early checkpoints and the final six accepted states; "
            "serialization occurs after solver timing"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    summary = run(parse_args(argv))
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0 if summary["certificate_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
