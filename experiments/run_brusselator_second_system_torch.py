#!/usr/bin/env python3
"""Run one of the two pre-registered Torch Brusselator lanes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import resource
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.integrate import solve_ivp
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from torch_tm_flowpipe import (  # noqa: E402
    DenseRangePolicy,
    FlowstarNormalFlowpipeState,
    GENERIC_ACCEPTED_BOUNDARY_SYMBOLIC_REMAINDER,
    NORMALIZED_INSERTION_DEPENDENCY_PRESERVING,
    TMVector,
    accepted_boundary_sr_queue_sha256,
    flowpipe_step_flowstar_style_adaptive,
)
from torch_tm_flowpipe.ode_examples import brusselator_ode  # noqa: E402
from torch_tm_flowpipe.safety import intervals_are_finite  # noqa: E402


CONTRACT_PATH = ROOT / "SECOND_SYSTEM_CONTRACT.md"
INITIAL_DECIMAL = (("1.48", "1.52"), ("2.98", "3.02"))
INITIAL_FLOAT = ((1.48, 1.52), (2.98, 3.02))
ORDER = 6
STEP = 0.02
REQUESTED_STEPS = 1000
HORIZON = 20.0
REMAINDER_RADIUS = 1e-4
CUTOFF = 1e-10
QUEUE_CAPACITY = 100
WALL_CAP_SECONDS = 3600.0


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence):
        return [_jsonable(item) for item in value]
    if hasattr(value, "detach"):
        tensor = value.detach().cpu()
        return tensor.item() if tensor.numel() == 1 else tensor.tolist()
    return str(value)


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _box_fields(prefix: str, box: Sequence[Any]) -> dict[str, float]:
    row: dict[str, float] = {}
    width_sum = 0.0
    for name, interval in zip(("x", "y"), box):
        lo = float(interval.lo.detach().cpu())
        hi = float(interval.hi.detach().cpu())
        width = hi - lo
        row[f"{prefix}_{name}_lo"] = lo
        row[f"{prefix}_{name}_hi"] = hi
        row[f"{prefix}_{name}_width"] = width
        row[f"{prefix}_{name}_lo_hex"] = lo.hex()
        row[f"{prefix}_{name}_hi_hex"] = hi.hex()
        width_sum += width
    row[f"{prefix}_width_sum"] = width_sum
    return row


def _sample_points() -> list[tuple[float, float]]:
    x_lo, x_hi = INITIAL_FLOAT[0]
    y_lo, y_hi = INITIAL_FLOAT[1]
    x_mid = (x_lo + x_hi) / 2.0
    y_mid = (y_lo + y_hi) / 2.0
    return [
        (x_lo, y_lo),
        (x_lo, y_hi),
        (x_hi, y_lo),
        (x_hi, y_hi),
        (x_lo, y_mid),
        (x_hi, y_mid),
        (x_mid, y_lo),
        (x_mid, y_hi),
        (x_mid, y_mid),
    ]


def _rhs(_time: float, state: np.ndarray) -> tuple[float, float]:
    x_value, y_value = float(state[0]), float(state[1])
    xy = x_value * y_value
    return 1.0 + x_value * (xy - 4.0), x_value * (3.0 - xy)


def _contains(box: Sequence[Any], point: Sequence[float]) -> bool:
    return all(
        float(interval.lo.detach().cpu()) <= float(value) <= float(interval.hi.detach().cpu())
        for interval, value in zip(box, point)
    )


def _advance_local_samples(
    samples: Sequence[tuple[float, float]],
    endpoint_box: Sequence[Any],
    tube_box: Sequence[Any],
) -> tuple[list[tuple[float, float]], int, int, bool]:
    next_samples: list[tuple[float, float]] = []
    endpoint_violations = 0
    tube_violations = 0
    successful = True
    times = np.asarray((0.0, STEP / 4.0, STEP / 2.0, 3.0 * STEP / 4.0, STEP))
    for sample in samples:
        solution = solve_ivp(
            _rhs,
            (0.0, STEP),
            np.asarray(sample, dtype=np.float64),
            method="DOP853",
            t_eval=times,
            rtol=2.3e-14,
            atol=1e-15,
        )
        successful &= bool(solution.success and solution.y.shape == (2, len(times)))
        if not successful:
            endpoint_violations += 1
            tube_violations += 1
            next_samples.append(sample)
            continue
        for column in range(solution.y.shape[1]):
            tube_violations += int(not _contains(tube_box, solution.y[:, column]))
        endpoint = (float(solution.y[0, -1]), float(solution.y[1, -1]))
        endpoint_violations += int(not _contains(endpoint_box, endpoint))
        next_samples.append(endpoint)
    return next_samples, endpoint_violations, tube_violations, successful


def _validation_passed(diagnostics: Sequence[Mapping[str, Any]]) -> tuple[bool, int, int]:
    records = [row for row in diagnostics if "validation_status" in row]
    passed = [
        row
        for row in records
        if str(row.get("validation_status", "")).lower() in {"passed", "validated"}
    ]
    failed = [row for row in records if str(row.get("validation_status", "")).lower() == "failed"]
    return bool(passed), len(records), len(failed)


def _owner_widths_ok(stats: Mapping[str, Any]) -> bool:
    keys = [
        key
        for key in stats
        if "owner_width" in str(key) or "propagated_symbolic_width" in str(key)
    ]
    return bool(keys) and all(
        isinstance(stats[key], (int, float))
        and math.isfinite(float(stats[key]))
        and float(stats[key]) >= 0.0
        for key in keys
        if stats[key] != ""
    )


def _queue_fields(state: FlowstarNormalFlowpipeState, expected_step: int) -> dict[str, Any]:
    queue = state.symbolic_queue
    if queue is None:
        return {
            "queue_present": False,
            "queue_accounting_ok": False,
            "queue_hash": "",
        }
    remainder = expected_step % QUEUE_CAPACITY
    expected_size = remainder
    expected_reset_count = expected_step // QUEUE_CAPACITY
    first_owner = expected_step - remainder + 1
    expected_owners = tuple(range(first_owner, expected_step + 1)) if remainder else ()
    accounting = (
        queue.owner_schema == "accepted_boundary_sr_v1"
        and queue.max_size == QUEUE_CAPACITY
        and queue.generation == expected_step
        and queue.accepted_boundary_index == expected_step
        and len(queue.J) == expected_size
        and queue.reset_count == expected_reset_count
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


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(_jsonable(row.get(key)), sort_keys=True)
                    if isinstance(row.get(key), (list, tuple, dict))
                    else row.get(key, "")
                    for key in fields
                }
            )
    temporary.replace(path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    reset_mode = (
        NORMALIZED_INSERTION_DEPENDENCY_PRESERVING
        if args.lane == "torch_generic_no_queue"
        else GENERIC_ACCEPTED_BOUNDARY_SYMBOLIC_REMAINDER
    )
    status_text = _git("status", "--short")
    diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    command = {
        "argv": sys.argv,
        "cwd": str(ROOT),
        "lane": args.lane,
        "reset_mode": reset_mode,
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("branch", "--show-current"),
        "worktree_status": status_text,
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "contract_sha256": _sha256(CONTRACT_PATH),
    }
    _write_json(output / "command.json", command)
    if status_text:
        raise RuntimeError("the pre-registered Torch run requires a clean worktree")

    state = FlowstarNormalFlowpipeState.from_exact_decimal_box(INITIAL_DECIMAL, ORDER)
    current: TMVector = state.normalized_initial_tm(ORDER)
    policy = DenseRangePolicy(
        method="adaptive_subdivision",
        max_depth=1,
        max_leaves=4,
        split_vars=(0, 1),
        trigger="proactive_depth1_on_named_contexts",
        named_contexts=("polynomial_truncation",),
        variable_orders=((0, 1, 2), (1, 0, 2), (2, 0, 1)),
    )
    rows: list[dict[str, Any]] = []
    samples = _sample_points()
    sample_endpoint_violations = 0
    sample_tube_violations = 0
    sample_solver_ok = True
    prefix_lo = [INITIAL_FLOAT[0][0], INITIAL_FLOAT[1][0]]
    prefix_hi = [INITIAL_FLOAT[0][1], INITIAL_FLOAT[1][1]]
    started = time.perf_counter()
    terminal_message = ""
    rejected = 0

    for step_index in range(1, REQUESTED_STEPS + 1):
        if time.perf_counter() - started >= WALL_CAP_SECONDS:
            terminal_message = "wall-time cap reached before requested horizon"
            break
        diagnostics: list[dict[str, Any]] = []
        queue_before = state.symbolic_queue
        queue_hash_before = (
            accepted_boundary_sr_queue_sha256(queue_before) if queue_before is not None else ""
        )
        step_started = time.perf_counter()
        segment = flowpipe_step_flowstar_style_adaptive(
            brusselator_ode,
            current,
            h=STEP,
            h_min=STEP,
            h_max=STEP,
            order=ORDER,
            target_remainder_radius=REMAINDER_RADIUS,
            cutoff_threshold=CUTOFF,
            max_validation_attempts=2,
            validation_eps=1e-12,
            validation_mode="flowstar_raw_remainder_compat",
            reset_mode=reset_mode,
            flowstar_normal_state=state,
            flowstar_symbolic_queue_max_size=QUEUE_CAPACITY,
            right_map_range_mode="standard",
            right_map_center_mode="constant",
            tm_backend="dense",
            dense_device="cpu",
            dense_dtype=torch.float64,
            dense_range_policy=policy,
            diagnostics=diagnostics,
            diagnostics_context={
                "system": "brusselator",
                "lane": args.lane,
                "segment_index": step_index - 1,
                "t_before": (step_index - 1) * STEP,
            },
        )
        step_runtime = time.perf_counter() - step_started
        validation_passed, validation_records, validation_failures = _validation_passed(diagnostics)
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
            "schema": "torch_brusselator_second_system_segment_v1",
            "lane": args.lane,
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
            "queue_hash_before": queue_hash_before,
        }
        if not accepted:
            rejected += 1
            terminal_message = terminal_message or segment.message or "fixed-step validation rejected"
            candidate_state = segment.flowstar_normal_state
            candidate_queue = candidate_state.symbolic_queue if candidate_state is not None else None
            candidate_hash = (
                accepted_boundary_sr_queue_sha256(candidate_queue)
                if candidate_queue is not None
                else ""
            )
            row.update(
                {
                    "queue_hash_after_attempt": candidate_hash,
                    "rollback_queue_unchanged": candidate_hash == queue_hash_before,
                    "endpoint_published": False,
                }
            )
            rows.append(row)
            _write_rows(output / "segments.csv", rows)
            break

        assert segment.flowstar_normal_state is not None
        assert segment.reset_tm is not None
        assert endpoint_box and tube_box
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
                "endpoint_published": True,
            }
        )
        stats = dict(segment.flowstar_normal_stats or {})
        if args.lane == "torch_generic_sr100":
            row.update(_queue_fields(segment.flowstar_normal_state, step_index))
            row["owner_widths_nonnegative_finite"] = _owner_widths_ok(stats)
            for key in (
                "accepted_boundary_sr_current_owner_width_sum",
                "accepted_boundary_sr_current_owner_width_sum_pre_cutoff",
                "accepted_boundary_sr_total_interval_image_width_sum",
                "accepted_boundary_sr_unscaled_roundoff_cutoff_owner_width_sum",
                "propagated_symbolic_width_sum",
            ):
                row[key] = stats.get(key, "")
        else:
            row.update(
                {
                    "queue_present": segment.flowstar_normal_state.symbolic_queue is not None,
                    "queue_accounting_ok": segment.flowstar_normal_state.symbolic_queue is None,
                    "queue_hash": "",
                    "owner_widths_nonnegative_finite": True,
                }
            )
        rows.append(row)
        current = segment.reset_tm
        state = segment.flowstar_normal_state
        if step_index % 25 == 0:
            _write_rows(output / "segments.csv", rows)

    runtime = time.perf_counter() - started
    accepted_rows = [row for row in rows if row["status"] == "accepted"]
    accepted_steps = len(accepted_rows)
    completed = accepted_steps == REQUESTED_STEPS and rejected == 0
    certificate_checks = (
        bool(accepted_rows)
        and all(bool(row["validation_passed"]) for row in accepted_rows)
        and sample_solver_ok
        and sample_endpoint_violations == 0
        and sample_tube_violations == 0
        and all(bool(row["queue_accounting_ok"]) for row in accepted_rows)
        and all(bool(row["owner_widths_nonnegative_finite"]) for row in accepted_rows)
        and (
            not rows
            or rows[-1]["status"] != "rejected"
            or bool(rows[-1].get("rollback_queue_unchanged"))
        )
    )
    _write_rows(output / "segments.csv", rows)
    summary = {
        "schema": "torch_brusselator_second_system_summary_v1",
        "lane": args.lane,
        "reset_mode": reset_mode,
        "status": "completed" if completed else "stopped",
        "message": terminal_message,
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
        "queue_capacity": None if args.lane == "torch_generic_no_queue" else QUEUE_CAPACITY,
        "sample_solver_ok": sample_solver_ok,
        "sample_endpoint_violations": sample_endpoint_violations,
        "sample_tube_violations": sample_tube_violations,
        "certificate_checks_passed": certificate_checks,
        "owner_accounting_passed": all(
            bool(row["queue_accounting_ok"]) and bool(row["owner_widths_nonnegative_finite"])
            for row in accepted_rows
        ),
        "solver_wall_seconds": runtime,
        "peak_rss_bytes": _peak_rss_bytes(),
        "endpoint_semantics": "endpoint_raw_segment_substitution",
        "tube_semantics": "accepted_segment_tau_interval",
        "commit": command["commit"],
        "branch": command["branch"],
        "tracked_diff_sha256": command["tracked_diff_sha256"],
        "worktree_dirty": bool(command["worktree_status"]),
        "contract_sha256": command["contract_sha256"],
    }
    _write_json(output / "summary.json", summary)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lane",
        required=True,
        choices=("torch_generic_no_queue", "torch_generic_sr100"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        result = run(parse_args(argv))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0 if result["certificate_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
