#!/usr/bin/env python3
"""Canonical sparse/dense Van der Pol order-4 adaptive flowpipe runner."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from torch_tm_flowpipe import (
    DenseRangePolicy,
    FlowstarNormalFlowpipeState,
    Interval,
    PolynomialODE,
    TMVector,
    flowpipe_step_flowstar_style_adaptive,
    save_terminal_checkpoint,
)
from torch_tm_flowpipe.safety import intervals_are_finite

CANONICAL_CONFIG = ROOT / "benchmarks" / "canonical.yaml"
MATCHED_CONTRACT = ROOT / "benchmarks" / "three_tool_matched_contract.yaml"
CHECKPOINTS = (0.1, 0.5, 1.0, 4.0, 6.0, 7.5, 10.0)


def _read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} does not contain a mapping")
    return value


def load_contract() -> dict[str, Any]:
    canonical = _read_yaml(CANONICAL_CONFIG)
    matched = _read_yaml(MATCHED_CONTRACT)
    canonical_vdp = canonical["systems"]["van_der_pol"]
    matched_vdp = matched["systems"]["van_der_pol"]
    if canonical_vdp["initial_box"] != matched_vdp["initial_set"]:
        raise ValueError("authoritative VDP initial boxes disagree")
    if int(matched_vdp["requested_order"]) != 4:
        raise ValueError("authoritative VDP requested order is not four")
    native_step = matched_vdp["step_policy"]["native_flowstar"]
    contract = {
        "ode": matched_vdp["ode"],
        "initial_box": matched_vdp["initial_set"],
        "requested_order": int(matched_vdp["requested_order"]),
        "target_remainder_radius": max(abs(float(bound)) for interval in matched_vdp["remainder_initialization"] for bound in interval),
        "cutoff": float(matched_vdp["cutoff"]),
        "h_min": float(native_step["minimum"]),
        "h_max": float(native_step["maximum"]),
        "target_horizon": float(matched_vdp["horizons"]["target"]),
        "validation_mode": "flowstar_raw_remainder_compat",
        "reset_mode": "normalized_insertion",
        "step_policy_mode": "flowstar_compat",
        "dtype": "float64",
        "output_contract": matched["output_contract"],
        "source_files": [str(CANONICAL_CONFIG.relative_to(ROOT)), str(MATCHED_CONTRACT.relative_to(ROOT))],
        "canonical_system_spec": canonical_vdp,
    }
    expected_ode = ["y", "(1 - x^2) * y - x"]
    if contract["ode"] != expected_ode:
        raise ValueError(f"authoritative VDP ODE changed: {contract['ode']}")
    return contract


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


def _canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = sorted({str(key) for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        if fields:
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        field: json.dumps(_jsonable(row.get(field)), sort_keys=True)
                        if isinstance(row.get(field), (dict, list, tuple))
                        else row.get(field, "")
                        for field in fields
                    }
                )
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_jsonable(row), sort_keys=True) + "\n")
    temporary.replace(path)


def _git(args: Sequence[str]) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def _variable_orders(value: str) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(int(index) for index in order.split(",") if index.strip())
        for order in value.split(";")
        if order.strip()
    )


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _candidate_trace_hashes(trace: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [row for row in trace if row.get("phase") == "polynomial_picard"]
    if not rows:
        return {}
    terminal = rows[-1]
    return {
        "coefficient_sha256": terminal.get("coefficient_sha256"),
        "exponent_support_sha256": terminal.get("exponent_support_sha256"),
        "basis_hash": terminal.get("basis_hash"),
        "effective_degree": terminal.get("effective_degree"),
        "picard_iterations": int(terminal.get("iteration", len(rows))),
    }


def _rk4_step(point: Sequence[float], h: float) -> tuple[float, float]:
    def rhs(state: Sequence[float]) -> tuple[float, float]:
        x, y = state
        return y, y - x - x * x * y

    x, y = point
    k1 = rhs((x, y))
    k2 = rhs((x + 0.5 * h * k1[0], y + 0.5 * h * k1[1]))
    k3 = rhs((x + 0.5 * h * k2[0], y + 0.5 * h * k2[1]))
    k4 = rhs((x + h * k3[0], y + h * k3[1]))
    return (
        x + h * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0,
        y + h * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0,
    )


def _advance_sample(point: Sequence[float], h: float) -> tuple[float, float]:
    pieces = max(4, int(math.ceil(abs(h) / 5e-4)))
    dt = h / pieces
    out = tuple(float(value) for value in point)
    for _ in range(pieces):
        out = _rk4_step(out, dt)
    return out


def _samples(initial_box: Sequence[Sequence[float]]) -> list[tuple[float, float]]:
    (x_lo, x_hi), (y_lo, y_hi) = initial_box
    points = [(x_lo, y_lo), (x_lo, y_hi), (x_hi, y_lo), (x_hi, y_hi), ((x_lo + x_hi) / 2, (y_lo + y_hi) / 2)]
    generator = random.Random(20260804)
    points.extend((generator.uniform(x_lo, x_hi), generator.uniform(y_lo, y_hi)) for _ in range(16))
    return points


def _box_values(box: Sequence[Interval]) -> dict[str, float]:
    names = ("x", "y")
    row: dict[str, float] = {}
    for name, interval in zip(names, box):
        lo = float(interval.lo.detach().cpu())
        hi = float(interval.hi.detach().cpu())
        row[f"{name}_lo"] = lo
        row[f"{name}_hi"] = hi
        row[f"{name}_width"] = hi - lo
    row["width_sum"] = sum(row[f"{name}_width"] for name in names)
    return row


def _sample_violations(samples: Sequence[Sequence[float]], box: Sequence[Interval], tolerance: float = 1e-10) -> tuple[int, float]:
    count = 0
    maximum = 0.0
    for point in samples:
        for value, interval in zip(point, box):
            lo = float(interval.lo.detach().cpu())
            hi = float(interval.hi.detach().cpu())
            violation = max(lo - value - tolerance, value - hi - tolerance, 0.0)
            if violation > 0.0:
                count += 1
                maximum = max(maximum, violation)
    return count, maximum


def _failure_type(status: str, message: str, h_remaining: float, h_min: float) -> str:
    lower = message.lower()
    if status == "timeout":
        return "timeout_resource_exhaustion"
    if "non-finite" in lower or "nonfinite" in lower:
        return "nonfinite"
    if h_remaining < h_min - 1e-15 or "before h_min" in lower:
        return "minimum_step_reached"
    if "subset" in lower or "remainder" in lower or "validation" in lower:
        return "validation_rejected"
    return "unknown_requires_more_evidence"


def run(args: argparse.Namespace) -> dict[str, Any]:
    contract = load_contract()
    requested_horizon = float(args.horizon)
    if requested_horizon <= 0 or requested_horizon > contract["target_horizon"] + 1e-12:
        raise ValueError("horizon must be positive and no larger than the authoritative target")
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    diagnostic_factors = []
    if args.reset_mode != contract["reset_mode"]:
        diagnostic_factors.append(f"reset_mode={args.reset_mode}")
    if args.right_map_center_mode != "constant":
        diagnostic_factors.append(f"right_map_center_mode={args.right_map_center_mode}")
    if args.right_map_range_mode != "standard":
        diagnostic_factors.append(f"right_map_range_mode={args.right_map_range_mode}")
    config_snapshot = {
        "contract": contract,
        "requested_horizon": requested_horizon,
        "tm_backend": args.tm_backend,
        "device": args.device,
        "reset_mode": args.reset_mode,
        "right_map_center_mode": args.right_map_center_mode,
        "right_map_range_mode": args.right_map_range_mode,
        "diagnostic_factors": diagnostic_factors,
        "single_factor_diagnostic": len(diagnostic_factors) == 1,
        "save_terminal_checkpoint": bool(args.save_terminal_checkpoint),
        "dense_range_method": args.dense_range_method,
        "dense_range_trigger": args.dense_range_trigger,
        "dense_range_max_depth": int(args.dense_range_max_depth),
        "dense_range_max_leaves": int(args.dense_range_max_leaves),
        "dense_range_split_vars": [int(item) for item in args.dense_range_split_vars.split(",") if item.strip()],
        "dense_range_contexts": [item.strip() for item in args.dense_range_contexts.split(",") if item.strip()],
        "dense_range_variable_orders": [list(order) for order in _variable_orders(args.dense_range_variable_orders)],
    }
    (output_dir / "config_snapshot.yaml").write_text(yaml.safe_dump(config_snapshot, sort_keys=True), encoding="utf-8")
    command = {
        "argv": sys.argv,
        "cwd": str(ROOT),
        "branch": _git(["branch", "--show-current"]),
        "commit": _git(["rev-parse", "HEAD"]),
        "worktree_status": _git(["status", "--short"]),
        "tracked_diff_sha256": hashlib.sha256(
            subprocess.run(["git", "diff", "HEAD", "--binary"], cwd=ROOT, check=True, capture_output=True).stdout
        ).hexdigest(),
        "config_sha256": hashlib.sha256(json.dumps(config_snapshot, sort_keys=True).encode("utf-8")).hexdigest(),
    }
    _atomic_json(output_dir / "command.json", command)

    initial_box = [Interval(*bounds) for bounds in contract["initial_box"]]
    ode = PolynomialODE.from_system_spec(contract["canonical_system_spec"])
    dense_range_policy = DenseRangePolicy(
        method=args.dense_range_method,
        max_depth=0 if args.dense_range_method == "natural" else int(args.dense_range_max_depth),
        max_leaves=int(args.dense_range_max_leaves),
        split_vars=tuple(int(item) for item in args.dense_range_split_vars.split(",") if item.strip()),
        trigger=args.dense_range_trigger,
        named_contexts=tuple(item.strip() for item in args.dense_range_contexts.split(",") if item.strip()),
        variable_orders=_variable_orders(args.dense_range_variable_orders),
    )
    current: TMVector | list[Interval] = initial_box
    normal_state: FlowstarNormalFlowpipeState | None = None
    samples = _samples(contract["initial_box"])
    h_next = contract["h_max"]
    current_time = 0.0
    start = time.perf_counter()
    segment_rows: list[dict[str, Any]] = []
    attempt_rows: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    ledger_rows: list[dict[str, Any]] = []
    profile_rows: list[dict[str, Any]] = []
    range_trace_rows: list[dict[str, Any]] = []
    horner_stage_rows: list[dict[str, Any]] = []
    range_call_count = 0
    total_sample_violations = 0
    max_sample_violation = 0.0
    tube_lo = [math.inf, math.inf]
    tube_hi = [-math.inf, -math.inf]
    fallback_count = 0
    conversion_count = 0
    device_transfer_count = 0
    status = "completed"
    message = ""
    crossed: set[float] = set()

    while current_time < requested_horizon - 1e-12:
        elapsed = time.perf_counter() - start
        if elapsed >= float(args.wall_cap_s):
            status = "timeout"
            message = "wall-time cap reached before requested horizon"
            break
        remaining = requested_horizon - current_time
        if remaining < contract["h_min"] - 1e-15:
            status = "failed"
            message = "remaining horizon is below authoritative h_min; no clipped sub-minimum endpoint was published"
            break
        h_try = min(float(h_next), contract["h_max"], remaining)
        if 0.0 < remaining - h_try < contract["h_min"]:
            h_try = remaining
        diagnostics: list[dict[str, Any]] = []
        previous_rejection_count = sum(
            str(row.get("validation_status", "")).lower() == "failed" for row in attempt_rows
        )
        step_start = time.perf_counter()
        segment = flowpipe_step_flowstar_style_adaptive(
            ode,
            current,
            h=h_try,
            h_min=contract["h_min"],
            h_max=contract["h_max"],
            order=contract["requested_order"],
            target_remainder_radius=contract["target_remainder_radius"],
            cutoff_threshold=contract["cutoff"],
            max_validation_attempts=2,
            validation_eps=1e-12,
            validation_mode=contract["validation_mode"],
            reset_mode=args.reset_mode,
            step_policy_mode=contract["step_policy_mode"],
            flowstar_normal_state=normal_state,
            right_map_center_mode=args.right_map_center_mode,
            right_map_range_mode=args.right_map_range_mode,
            tm_backend=args.tm_backend,
            dense_device=args.device,
            dense_range_policy=dense_range_policy,
            diagnostics=diagnostics,
            diagnostics_context={"segment_index": len(segment_rows), "t_before": current_time, "mode": args.tm_backend},
        )
        step_wall = time.perf_counter() - step_start
        accepted = segment.status == "validated" and segment.endpoint_raw_tm is not None and segment.reset_tm is not None
        try:
            segment_box = segment.tm.range_box()
            endpoint_box = segment.endpoint_raw_tm.range_box() if segment.endpoint_raw_tm is not None else None
            finite = intervals_are_finite(segment_box) and endpoint_box is not None and intervals_are_finite(endpoint_box)
        except Exception as exc:
            segment_box = []
            endpoint_box = None
            finite = False
            segment.message = segment.message or f"range evaluation failed: {exc}"
        accepted = bool(accepted and finite)
        t_hi = current_time + float(segment.h) if accepted else current_time
        counters = dict(segment.backend_counters or {})
        fallback_count += int(counters.get("sparse_fallback_count", 0))
        conversion_count += int(counters.get("segment_boundary_conversions", 0))
        device_transfer_count += int(counters.get("device_transfer_count", 0))
        row = {
            "segment_index": len(segment_rows),
            "status": "accepted" if accepted else "rejected",
            "t_lo": current_time,
            "t_hi": t_hi,
            "h_attempted": h_try,
            "h_accepted": float(segment.h) if accepted else "",
            "requested_order": contract["requested_order"],
            "effective_degree": max((model.polynomial.degree() for model in segment.tm), default=0),
            "tau_index": segment.tau_index,
            "basis_hash": next((item.get("basis_hash") for item in (segment.backend_trace or []) if item.get("basis_hash")), ""),
            "backend_lane": segment.backend_lane,
            "validation_attempts": segment.validation_attempts,
            "step_rejections": segment.step_rejections,
            "next_h": segment.next_h if segment.next_h is not None else "",
            "raw_endpoint_published": endpoint_box is not None,
            "endpoint_tightening_applied": segment.endpoint_tightening_applied,
            "message": segment.message,
            **counters,
        }
        if segment_box:
            row.update({f"segment_{key}": value for key, value in _box_values(segment_box).items()})
        if endpoint_box is not None:
            row.update({f"endpoint_{key}": value for key, value in _box_values(endpoint_box).items()})
        for key, value in (segment.flowstar_normal_stats or {}).items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                row[f"carry_{key}"] = value
        segment_rows.append(row)
        for diagnostic in diagnostics:
            attempt_rows.append({"segment_index": len(segment_rows) - 1, "t_before": current_time, **_jsonable(diagnostic)})
        for trace_row in segment.backend_trace or []:
            if trace_row.get("phase") == "remainder_validation":
                ledger_rows.append({"segment_index": len(segment_rows) - 1, "t_before": current_time, **_jsonable(trace_row)})
            if trace_row.get("phase") in {"polynomial_range", "range_validation_lane", "range_fail_closed"}:
                compact_trace = dict(trace_row)
                stages = list(compact_trace.pop("horner_stages", []))
                if compact_trace.get("phase") == "polynomial_range":
                    stage_bytes = json.dumps(_jsonable(stages), sort_keys=True, separators=(",", ":")).encode("utf-8")
                    compact_trace["range_call_index"] = range_call_count
                    compact_trace["horner_stage_count"] = len(stages)
                    compact_trace["horner_stage_sha256"] = hashlib.sha256(stage_bytes).hexdigest()
                    horner_stage_rows.extend(
                        {
                            "segment_index": len(segment_rows) - 1,
                            "t_before": current_time,
                            "range_call_index": range_call_count,
                            "context": compact_trace.get("context"),
                            **dict(stage),
                        }
                        for stage in stages
                    )
                    range_call_count += 1
                range_trace_rows.append({"segment_index": len(segment_rows) - 1, "t_before": current_time, **_jsonable(compact_trace)})
        profile_rows.append({"segment_index": len(segment_rows) - 1, "t_before": current_time, "h_attempted": h_try, "total_wall_s": step_wall, "backend_lane": segment.backend_lane})

        if not accepted:
            if args.save_terminal_checkpoint:
                if not isinstance(current, TMVector) or normal_state is None:
                    raise RuntimeError("terminal checkpoint requires the canonical TMVector normal pre-state")
                checkpoint_dir = output_dir / "terminal_checkpoint"
                manifest = save_terminal_checkpoint(
                    checkpoint_dir,
                    current=current,
                    normal_state=normal_state,
                    scheduler={
                        "current_time": current_time,
                        "h_next": h_next,
                        "h_attempted": h_try,
                        "accepted_segment_count": len([item for item in segment_rows[:-1] if item.get("status") == "accepted"]),
                        "previous_rejection_count": previous_rejection_count,
                        "terminal_internal_step_rejections": int(segment.step_rejections),
                        "next_retry_h": segment.next_h,
                    },
                    contract=contract,
                    provenance={
                        "branch": command["branch"],
                        "commit": command["commit"],
                        "tracked_diff_sha256": command["tracked_diff_sha256"],
                        "config_sha256": command["config_sha256"],
                        "source_hashes": {
                            str(path.relative_to(ROOT)): _file_sha256(path)
                            for path in (CANONICAL_CONFIG, MATCHED_CONTRACT)
                        },
                        "dtype": contract["dtype"],
                        "device": args.device,
                    },
                )
                validation_rows = [item for item in (segment.backend_trace or []) if item.get("phase") == "remainder_validation"]
                terminal_reference = {
                    "attempted_h": h_try,
                    "t_before": current_time,
                    "accepted": accepted,
                    "status": segment.status,
                    "message": segment.message,
                    "validation_rejection_reason": (
                        validation_rows[-1].get("rejection_reason", "") if validation_rows else ""
                    ),
                    "candidate_hashes": _candidate_trace_hashes(segment.backend_trace or []),
                    "candidate_remainder": segment.candidate_remainder,
                    "picard_image_remainder": segment.picard_image_remainder,
                    "subset_margin": segment.subset_margin,
                    "backend_lane": segment.backend_lane,
                    "backend_counters": segment.backend_counters,
                    "backend_trace": segment.backend_trace,
                    "validation_rows": validation_rows,
                    "checkpoint_manifest_sha256": hashlib.sha256(
                        _canonical_json_bytes(manifest)
                    ).hexdigest(),
                }
                _atomic_json(checkpoint_dir / "terminal_reference.json", terminal_reference)
            status = "failed"
            message = segment.message or "dense/sparse flowpipe validation rejected"
            break
        assert endpoint_box is not None
        samples = [_advance_sample(point, float(segment.h)) for point in samples]
        violations, maximum = _sample_violations(samples, endpoint_box)
        total_sample_violations += violations
        max_sample_violation = max(max_sample_violation, maximum)
        for index, interval in enumerate(segment_box[:2]):
            tube_lo[index] = min(tube_lo[index], float(interval.lo.detach().cpu()))
            tube_hi[index] = max(tube_hi[index], float(interval.hi.detach().cpu()))
        current_time = t_hi
        current = segment.reset_tm
        normal_state = segment.flowstar_normal_state
        h_next = float(segment.next_h if segment.next_h is not None else min(float(segment.h) * 1.1, contract["h_max"]))
        for checkpoint in CHECKPOINTS:
            if checkpoint <= requested_horizon + 1e-12 and checkpoint not in crossed and current_time >= checkpoint - 1e-12:
                crossed.add(checkpoint)
                checkpoint_rows.append({"checkpoint": checkpoint, "last_validated_time": current_time, "segment_index": len(segment_rows) - 1, "status": "passed", "endpoint_width_sum": row.get("endpoint_width_sum", ""), "segment_width_sum": row.get("segment_width_sum", "")})

        _write_csv(output_dir / "segments.csv", segment_rows)
        _write_csv(output_dir / "attempts.csv", attempt_rows)
        _write_csv(output_dir / "checkpoints.csv", checkpoint_rows)
        _write_jsonl(output_dir / "remainder_ledger.jsonl", ledger_rows)
        _write_csv(output_dir / "profile.csv", profile_rows)
        _write_jsonl(output_dir / "range_trace.jsonl", range_trace_rows)
        _write_jsonl(output_dir / "horner_stage_trace.jsonl", horner_stage_rows)

    runtime = time.perf_counter() - start
    completed = status == "completed" and current_time >= requested_horizon - 1e-12
    if status == "completed" and not completed:
        status = "failed"
        message = message or "runner stopped before requested horizon"
    if total_sample_violations:
        status = "failed"
        message = message or "sample sanity violation"
        completed = False
    failure_type = "" if completed else _failure_type(status, message, requested_horizon - current_time, contract["h_min"])
    accepted_rows = [row for row in segment_rows if row.get("status") == "accepted"]
    last = accepted_rows[-1] if accepted_rows else {}
    summary = {
        "status": "completed" if completed else status,
        "failure_type": failure_type,
        "message": message,
        "requested_horizon": requested_horizon,
        "contract_target_horizon": contract["target_horizon"],
        "completed_horizon": current_time,
        "completed_requested_horizon": completed,
        "requested_order": contract["requested_order"],
        "h_min": contract["h_min"],
        "h_max": contract["h_max"],
        "cutoff": contract["cutoff"],
        "target_remainder_radius": contract["target_remainder_radius"],
        "tm_backend": args.tm_backend,
        "backend_lane": "hybrid_dense_core" if args.tm_backend == "dense" else "sparse_reference",
        "device": args.device,
        "reset_mode": args.reset_mode,
        "right_map_center_mode": args.right_map_center_mode,
        "right_map_range_mode": args.right_map_range_mode,
        "dense_range_method": args.dense_range_method,
        "dense_range_trigger": args.dense_range_trigger,
        "dense_range_max_depth": int(args.dense_range_max_depth),
        "dense_range_max_leaves": int(args.dense_range_max_leaves),
        "dense_range_contexts": list(dense_range_policy.named_contexts),
        "dense_range_variable_orders": [list(order) for order in dense_range_policy.variable_orders],
        "diagnostic_factors": diagnostic_factors,
        "single_factor_diagnostic": len(diagnostic_factors) == 1,
        "accepted_steps": sum(row["status"] == "accepted" for row in segment_rows),
        "rejected_step_records": sum(row["status"] == "rejected" for row in segment_rows),
        "rejected_attempts": sum(str(row.get("validation_status", "")).lower() == "failed" for row in attempt_rows),
        "fallback_count": fallback_count,
        "segment_boundary_conversion_count": conversion_count,
        "device_transfer_count": device_transfer_count,
        "range_subdivision_invocations": sum(int(row.get("range_subdivision_invocations", 0)) for row in segment_rows),
        "range_leaf_evaluations": sum(int(row.get("range_leaf_evaluations", 0)) for row in segment_rows),
        "sample_sanity_violations": total_sample_violations,
        "sample_sanity_max_violation": max_sample_violation,
        "sample_sanity_status": "passed" if total_sample_violations == 0 and segment_rows else "failed",
        "endpoint_repair_used": False,
        "endpoint_tightening_used": any(bool(row.get("endpoint_tightening_applied")) for row in segment_rows),
        "raw_endpoint": {key.removeprefix("endpoint_"): value for key, value in last.items() if key.startswith("endpoint_")},
        "last_segment": {key.removeprefix("segment_"): value for key, value in last.items() if key.startswith("segment_")},
        "full_tube": {"x_lo": tube_lo[0], "x_hi": tube_hi[0], "y_lo": tube_lo[1], "y_hi": tube_hi[1]} if segment_rows and math.isfinite(tube_lo[0]) else None,
        "runtime_s": runtime,
        "branch": command["branch"],
        "commit": command["commit"],
        "worktree_dirty": bool(command["worktree_status"]),
        "tracked_diff_sha256": command["tracked_diff_sha256"],
        "config_sha256": command["config_sha256"],
    }
    _write_csv(output_dir / "segments.csv", segment_rows)
    _write_csv(output_dir / "attempts.csv", attempt_rows)
    _write_csv(output_dir / "checkpoints.csv", checkpoint_rows)
    _write_jsonl(output_dir / "remainder_ledger.jsonl", ledger_rows)
    _write_csv(output_dir / "profile.csv", profile_rows)
    _write_jsonl(output_dir / "range_trace.jsonl", range_trace_rows)
    _write_jsonl(output_dir / "horner_stage_trace.jsonl", horner_stage_rows)
    _atomic_json(output_dir / "summary.json", summary)
    _atomic_json(
        output_dir / "decision.json",
        {
            "eligible_t10_completion": bool(completed and requested_horizon == contract["target_horizon"]),
            "highest_validated_horizon": current_time,
            "failure_type": failure_type,
            "backend_lane": summary["backend_lane"],
            "reset_mode": args.reset_mode,
            "diagnostic_factors": diagnostic_factors,
            "single_factor_diagnostic": summary["single_factor_diagnostic"],
            "fallback_count": fallback_count,
            "no_endpoint_repair": True,
            "no_hidden_inner_sparse_fallback": fallback_count == 0,
        },
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tm-backend", choices=("sparse", "dense"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--horizon", type=float, default=10.0)
    parser.add_argument("--wall-cap-s", type=float, default=1800.0)
    parser.add_argument(
        "--reset-mode",
        choices=("normalized_insertion", "normalized_insertion_symqueue_v2"),
        default="normalized_insertion",
    )
    parser.add_argument("--right-map-center-mode", choices=("constant", "range_midpoint"), default="constant")
    parser.add_argument("--right-map-range-mode", choices=("standard", "normal_eval"), default="standard")
    parser.add_argument(
        "--dense-range-method",
        choices=(
            "natural",
            "subdivision",
            "adaptive_subdivision",
            "horner_fixed",
            "horner_registered_best",
            "subdivision_then_horner",
            "horner_per_leaf",
        ),
        default="natural",
    )
    parser.add_argument(
        "--dense-range-trigger",
        choices=("always", "on_validation_failure", "proactive_depth1_on_named_contexts"),
        default="always",
    )
    parser.add_argument("--dense-range-max-depth", type=int, default=1)
    parser.add_argument("--dense-range-max-leaves", type=int, default=64)
    parser.add_argument("--dense-range-split-vars", default="0,1")
    parser.add_argument("--dense-range-contexts", default="")
    parser.add_argument(
        "--dense-range-variable-orders",
        default="0,1,2;1,0,2;2,0,1",
        help="semicolon-separated Horner variable permutations",
    )
    parser.add_argument(
        "--save-terminal-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="save the immutable pre-state and failed attempt reference on terminal rejection",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        summary = run(parse_args(argv))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["completed_requested_horizon"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
