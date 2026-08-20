#!/usr/bin/env python3
"""Canonical sparse/dense Van der Pol order-4 adaptive flowpipe runner."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import resource
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
    NORMALIZED_INSERTION_DEPENDENCY_PRESERVING,
    PolynomialODE,
    TMVector,
    flowpipe_step_flowstar_style_adaptive,
    save_terminal_checkpoint,
)
from torch_tm_flowpipe.safety import intervals_are_finite
from torch_tm_flowpipe.audit_trace import TransitionTraceWriter
from torch_tm_flowpipe.comparison_contract import vdp_identity_hashes

CANONICAL_CONFIG = ROOT / "benchmarks" / "canonical.yaml"
MATCHED_CONTRACT = ROOT / "benchmarks" / "three_tool_matched_contract.yaml"
CHECKPOINTS = (0.1, 0.5, 1.0, 4.0, 6.0, 7.5, 10.0)
EXACT_INITIAL_BOX_DECIMAL = [["1.1", "1.4"], ["2.35", "2.45"]]


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


def _write_trace_outputs(
    output_dir: Path,
    *,
    segment_rows: Sequence[Mapping[str, Any]],
    attempt_rows: Sequence[Mapping[str, Any]],
    checkpoint_rows: Sequence[Mapping[str, Any]],
    ledger_rows: Sequence[Mapping[str, Any]],
    refinement_rows: Sequence[Mapping[str, Any]],
    profile_rows: Sequence[Mapping[str, Any]],
    range_trace_rows: Sequence[Mapping[str, Any]],
    horner_stage_rows: Sequence[Mapping[str, Any]],
    owner_rows: Sequence[Mapping[str, Any]],
) -> None:
    _write_csv(output_dir / "segments.csv", segment_rows)
    _write_csv(output_dir / "attempts.csv", attempt_rows)
    _write_csv(output_dir / "checkpoints.csv", checkpoint_rows)
    _write_jsonl(output_dir / "remainder_ledger.jsonl", ledger_rows)
    _write_jsonl(output_dir / "refinement_ledger.jsonl", refinement_rows)
    _write_csv(output_dir / "profile.csv", profile_rows)
    _write_jsonl(output_dir / "range_trace.jsonl", range_trace_rows)
    _write_jsonl(output_dir / "horner_stage_trace.jsonl", horner_stage_rows)
    _write_jsonl(output_dir / "owner_ledger.jsonl", owner_rows)


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


def _float_hex(value: Any) -> str:
    if hasattr(value, "detach"):
        value = value.detach().cpu().item()
    return float(value).hex()


def _peak_rss_bytes() -> int:
    """Return the process high-water RSS in bytes on the supported Unix hosts."""
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Linux reports KiB; macOS reports bytes.  The evidence hosts are Linux,
    # but retaining the platform distinction keeps local reruns meaningful.
    return value if sys.platform == "darwin" else value * 1024


def _interval_payload(value: Interval) -> dict[str, str]:
    return {"lo_hex": _float_hex(value.lo), "hi_hex": _float_hex(value.hi)}


def _tmvector_payload(value: TMVector | Sequence[Interval]) -> dict[str, Any]:
    if isinstance(value, TMVector):
        return {
            "kind": "tmvector",
            "models": [
                {
                    "terms": [
                        {"exponent": list(exponent), "coefficient_hex": _float_hex(coefficient)}
                        for exponent, coefficient in sorted(model.polynomial.terms.items())
                    ],
                    "remainder": _interval_payload(model.remainder),
                    "domain": [_interval_payload(interval) for interval in model.domain],
                }
                for model in value
            ],
        }
    return {
        "kind": "interval_box",
        "box": [_interval_payload(interval) for interval in value],
    }


def _normal_state_payload(value: FlowstarNormalFlowpipeState | None) -> Any:
    if value is None:
        return None
    return {
        "tmv_pre": _tmvector_payload(value.tmv_pre),
        "tmv_right": _tmvector_payload(value.tmv_right),
        "domain": [_interval_payload(interval) for interval in value.domain],
        "center_hex": [_float_hex(item) for item in value.center],
        "scales_hex": [_float_hex(item) for item in value.scales],
        "step_index": int(value.step_index),
        "initial_remainders": (
            None
            if value.initial_remainders is None
            else [_interval_payload(interval) for interval in value.initial_remainders]
        ),
        "complete_initial_tm": (
            None
            if value.complete_initial_tm is None
            else _tmvector_payload(value.complete_initial_tm)
        ),
    }


def _state_sha256(
    value: TMVector | Sequence[Interval],
    normal_state: FlowstarNormalFlowpipeState | None = None,
) -> str:
    payload = {
        "current": _tmvector_payload(value),
        "normal_state": _normal_state_payload(normal_state),
    }
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _coefficient_sha256(value: TMVector | None) -> str | None:
    if value is None:
        return None
    payload = [
        [
            {"exponent": list(exponent), "coefficient_hex": _float_hex(coefficient)}
            for exponent, coefficient in sorted(model.polynomial.terms.items())
        ]
        for model in value
    ]
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


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
    if args.validation_mode is not None:
        contract = {**contract, "validation_mode": args.validation_mode}
    if args.initialization_contract == "exact_decimal_contract":
        contract = {
            **contract,
            "initial_box_exact_decimal": EXACT_INITIAL_BOX_DECIMAL,
        }
    contract = {**contract, "initialization_contract": args.initialization_contract}
    requested_horizon = float(args.horizon)
    if requested_horizon <= 0 or requested_horizon > contract["target_horizon"] + 1e-12:
        raise ValueError("horizon must be positive and no larger than the authoritative target")
    if int(args.trace_flush_every) < 0:
        raise ValueError("trace flush interval must be nonnegative")
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
    if contract["validation_mode"] != "flowstar_raw_remainder_compat":
        diagnostic_factors.append(f"validation_mode={contract['validation_mode']}")
    config_snapshot = {
        "contract": contract,
        "requested_horizon": requested_horizon,
        "fixed_step": args.fixed_step,
        "tm_backend": args.tm_backend,
        "device": args.device,
        "reset_mode": args.reset_mode,
        "right_map_center_mode": args.right_map_center_mode,
        "right_map_range_mode": args.right_map_range_mode,
        "validation_mode": contract["validation_mode"],
        "diagnostic_factors": diagnostic_factors,
        "single_factor_diagnostic": len(diagnostic_factors) == 1,
        "save_terminal_checkpoint": bool(args.save_terminal_checkpoint),
        "initialization_contract": args.initialization_contract,
        "trace_flush_every": int(args.trace_flush_every),
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

    transition_trace = (
        TransitionTraceWriter(
            args.transition_trace_dir,
            run_id="torch-authoritative-observation-20260806",
            source_commit=command["commit"],
        )
        if args.transition_trace_dir is not None
        else None
    )

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
    if args.initialization_contract == "exact_decimal_contract":
        if args.reset_mode not in {
            "normalized_insertion",
            NORMALIZED_INSERTION_DEPENDENCY_PRESERVING,
            "normalized_insertion_bounded_source_ledger_o4_g1",
            "normalized_insertion_bounded_shared_source_o4_g2",
        }:
            raise ValueError("exact-decimal matrix lane supports only frozen legacy/G1/G2 modes")
        normal_state = FlowstarNormalFlowpipeState.from_exact_decimal_box(
            [(row[0], row[1]) for row in EXACT_INITIAL_BOX_DECIMAL],
            contract["requested_order"],
        )
        if args.reset_mode == "normalized_insertion_bounded_source_ledger_o4_g1":
            normal_state = normal_state.with_bounded_source_g1(contract["requested_order"])
        elif args.reset_mode == "normalized_insertion_bounded_shared_source_o4_g2":
            normal_state = normal_state.with_g2_shared_columns(contract["requested_order"])
        current = normal_state.normalized_initial_tm(contract["requested_order"])
    samples = _samples(contract["initial_box"])
    fixed_step = None if args.fixed_step is None else float(args.fixed_step)
    if fixed_step is not None:
        if fixed_step <= 0.0:
            raise ValueError("fixed step must be positive")
        requested_steps = round(requested_horizon / fixed_step)
        if requested_steps <= 0 or requested_steps * fixed_step != requested_horizon:
            raise ValueError("requested horizon must be an exact integer fixed-step multiple")
    else:
        requested_steps = None
    h_next = fixed_step if fixed_step is not None else contract["h_max"]
    current_time = 0.0
    start = time.perf_counter()
    segment_rows: list[dict[str, Any]] = []
    attempt_rows: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    ledger_rows: list[dict[str, Any]] = []
    refinement_rows: list[dict[str, Any]] = []
    profile_rows: list[dict[str, Any]] = []
    range_trace_rows: list[dict[str, Any]] = []
    horner_stage_rows: list[dict[str, Any]] = []
    owner_rows: list[dict[str, Any]] = []
    range_call_count = 0
    total_sample_violations = 0
    max_sample_violation = 0.0
    tube_lo = [math.inf, math.inf]
    tube_hi = [-math.inf, -math.inf]
    fallback_count = 0
    conversion_count = 0
    device_transfer_count = 0
    host_to_device_s = 0.0
    dense_kernel_s = 0.0
    device_to_host_s = 0.0
    status = "completed"
    message = ""
    crossed: set[float] = set()
    trace_io_s = 0.0
    trace_write_count = 0

    def flush_trace_outputs() -> None:
        nonlocal trace_io_s, trace_write_count
        io_started = time.perf_counter()
        _write_trace_outputs(
            output_dir,
            segment_rows=segment_rows,
            attempt_rows=attempt_rows,
            checkpoint_rows=checkpoint_rows,
            ledger_rows=ledger_rows,
            refinement_rows=refinement_rows,
            profile_rows=profile_rows,
            range_trace_rows=range_trace_rows,
            horner_stage_rows=horner_stage_rows,
            owner_rows=owner_rows,
        )
        trace_io_s += time.perf_counter() - io_started
        trace_write_count += 1

    while current_time < requested_horizon - 1e-12:
        elapsed = time.perf_counter() - start
        if elapsed >= float(args.wall_cap_s):
            status = "timeout"
            message = "wall-time cap reached before requested horizon"
            break
        remaining = requested_horizon - current_time
        local_h_min = fixed_step if fixed_step is not None else contract["h_min"]
        local_h_max = fixed_step if fixed_step is not None else contract["h_max"]
        if remaining < local_h_min - 1e-15:
            status = "failed"
            message = "remaining horizon is below authoritative h_min; no clipped sub-minimum endpoint was published"
            break
        h_try = min(float(h_next), local_h_max, remaining)
        if fixed_step is not None:
            h_try = fixed_step
        elif 0.0 < remaining - h_try < local_h_min:
            h_try = remaining
        prestate_sha256 = _state_sha256(current, normal_state)
        prestate_center = (
            list(normal_state.center)
            if normal_state is not None
            else [
                (float(bounds[0]) + float(bounds[1])) / 2.0
                for bounds in contract["initial_box"]
            ]
        )
        prestate_scale = (
            list(normal_state.scales)
            if normal_state is not None
            else [
                (float(bounds[1]) - float(bounds[0])) / 2.0
                for bounds in contract["initial_box"]
            ]
        )
        diagnostics: list[dict[str, Any]] = []
        previous_rejection_count = sum(
            str(row.get("validation_status", "")).lower() == "failed" for row in attempt_rows
        )
        step_start = time.perf_counter()
        segment = flowpipe_step_flowstar_style_adaptive(
            ode,
            current,
            h=h_try,
            h_min=local_h_min,
            h_max=local_h_max,
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

        if transition_trace is not None:
            transition_trace.record_step(
                step=len(segment_rows),
                t_pre=current_time,
                current=current,
                previous_state=normal_state,
                segment=segment,
                diagnostics=diagnostics,
                accepted=accepted,
                attempted_h=h_try,
                order=contract["requested_order"],
            )
        t_hi = (
            (len([item for item in segment_rows if item.get("status") == "accepted"]) + 1)
            * fixed_step
            if accepted and fixed_step is not None
            else current_time + float(segment.h)
            if accepted
            else current_time
        )
        counters = dict(segment.backend_counters or {})
        fallback_count += int(counters.get("sparse_fallback_count", 0))
        conversion_count += int(counters.get("segment_boundary_conversions", 0))
        device_transfer_count += int(counters.get("device_transfer_count", 0))
        host_to_device_s += float(counters.get("host_to_device_s", 0.0))
        dense_kernel_s += float(counters.get("dense_kernel_s", 0.0))
        device_to_host_s += float(counters.get("device_to_host_s", 0.0))
        row = {
            "segment_index": len(segment_rows),
            "status": "accepted" if accepted else "rejected",
            "t_lo": current_time,
            "t_hi": t_hi,
            "t_lo_hex": float(current_time).hex(),
            "t_hi_hex": float(t_hi).hex(),
            "h_attempted": h_try,
            "h_accepted": float(segment.h) if accepted else "",
            "h_attempted_hex": float(h_try).hex(),
            "h_accepted_hex": float(segment.h).hex() if accepted else "",
            "schedule_kind": "fixed" if fixed_step is not None else "adaptive",
            "prestate_sha256": prestate_sha256,
            "prestate_center": prestate_center,
            "prestate_scale": prestate_scale,
            "retained_coefficient_sha256": _coefficient_sha256(
                segment.reset_tm if accepted else None
            ),
            "candidate_coefficient_sha256": _coefficient_sha256(segment.tm),
            "next_boundary_term_count": (
                sum(len(model.polynomial.terms) for model in segment.reset_tm)
                if accepted and segment.reset_tm is not None
                else ""
            ),
            "next_boundary_active_variables": (
                sorted(segment.reset_tm.active_variables())
                if accepted and segment.reset_tm is not None
                else []
            ),
            "raw_remainder": segment.candidate_remainder,
            "post_poly_diff_remainder": segment.picard_image_remainder,
            "target_margins": segment.subset_margin,
            "ordinary_symbolic_remainder_summary": {
                str(key): value
                for key, value in (segment.flowstar_normal_stats or {}).items()
                if "ordinary" in str(key) or "symbolic" in str(key)
            },
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
            segment_prefix = "segment" if accepted else "rejected_candidate_segment"
            row.update({f"{segment_prefix}_{key}": value for key, value in _box_values(segment_box).items()})
        if endpoint_box is not None:
            row.update({f"endpoint_{key}": value for key, value in _box_values(endpoint_box).items()})
        for key, value in (segment.flowstar_normal_stats or {}).items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                row[f"carry_{key}"] = value
        normal_stats = segment.flowstar_normal_stats or {}
        for owner_group in (
            "source_ledger_retired_owner_rows",
            "source_ledger_carried_ordinary_owner_rows",
            "source_ledger_dense_owner_rows",
            "source_ledger_insertion_owner_rows",
            "source_ledger_rebox_owner_rows",
            "g2_retired_owner_rows",
            "g2_carried_ordinary_owner_rows",
            "g2_dense_owner_rows",
            "g2_insertion_owner_rows",
            "g2_rebox_owner_rows",
        ):
            values = normal_stats.get(owner_group, [])
            if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
                continue
            owner_rows.extend(
                {
                    "segment_index": len(segment_rows),
                    "t_before": current_time,
                    "t_after": t_hi,
                    "reset_mode": args.reset_mode,
                    "owner_group": owner_group,
                    **_jsonable(value),
                }
                for value in values
                if isinstance(value, Mapping)
            )
        segment_rows.append(row)
        for diagnostic in diagnostics:
            attempt_rows.append({"segment_index": len(segment_rows) - 1, "t_before": current_time, **_jsonable(diagnostic)})
        for trace_row in segment.backend_trace or []:
            if trace_row.get("phase") == "remainder_validation":
                ledger_rows.append({"segment_index": len(segment_rows) - 1, "t_before": current_time, **_jsonable(trace_row)})
            if trace_row.get("phase") == "post_accept_refinement":
                refinement_rows.append(
                    {
                        "segment_index": len(segment_rows) - 1,
                        "t_before": current_time,
                        **_jsonable(trace_row),
                    }
                )
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
        row["stage_runtime_s"] = step_wall
        profile_rows.append({
            "segment_index": len(segment_rows) - 1,
            "t_before": current_time,
            "h_attempted": h_try,
            "total_wall_s": step_wall,
            "host_to_device_s": float(counters.get("host_to_device_s", 0.0)),
            "dense_kernel_s": float(counters.get("dense_kernel_s", 0.0)),
            "device_to_host_s": float(counters.get("device_to_host_s", 0.0)),
            "device_transfer_count": int(counters.get("device_transfer_count", 0)),
            "backend_lane": segment.backend_lane,
        })

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
        row.update(
            {
                "prefix_x_lo": tube_lo[0],
                "prefix_x_hi": tube_hi[0],
                "prefix_x_width": tube_hi[0] - tube_lo[0],
                "prefix_y_lo": tube_lo[1],
                "prefix_y_hi": tube_hi[1],
                "prefix_y_width": tube_hi[1] - tube_lo[1],
                "center": (
                    None
                    if segment.flowstar_normal_state is None
                    else segment.flowstar_normal_state.center
                ),
                "scale": (
                    None
                    if segment.flowstar_normal_state is None
                    else segment.flowstar_normal_state.scales
                ),
                "retained_center": (
                    None
                    if segment.flowstar_normal_state is None
                    else segment.flowstar_normal_state.center
                ),
                "retained_scale": (
                    None
                    if segment.flowstar_normal_state is None
                    else segment.flowstar_normal_state.scales
                ),
            }
        )
        current_time = (
            len([item for item in segment_rows if item.get("status") == "accepted"])
            * fixed_step
            if fixed_step is not None
            else t_hi
        )
        current = segment.reset_tm
        normal_state = segment.flowstar_normal_state
        h_next = (
            fixed_step
            if fixed_step is not None
            else float(
                segment.next_h
                if segment.next_h is not None
                else min(float(segment.h) * 1.1, contract["h_max"])
            )
        )
        for checkpoint in CHECKPOINTS:
            if checkpoint <= requested_horizon + 1e-12 and checkpoint not in crossed and current_time >= checkpoint - 1e-12:
                crossed.add(checkpoint)
                checkpoint_rows.append({"checkpoint": checkpoint, "last_validated_time": current_time, "segment_index": len(segment_rows) - 1, "status": "passed", "endpoint_width_sum": row.get("endpoint_width_sum", ""), "segment_width_sum": row.get("segment_width_sum", "")})

        if (
            int(args.trace_flush_every) > 0
            and len(segment_rows) % int(args.trace_flush_every) == 0
        ):
            flush_trace_outputs()

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
    flush_trace_outputs()
    summary = {
        "status": "completed" if completed else status,
        "failure_type": failure_type,
        "message": message,
        "requested_horizon": requested_horizon,
        "contract_target_horizon": contract["target_horizon"],
        "completed_horizon": current_time,
        "completed_requested_horizon": completed,
        "requested_order": contract["requested_order"],
        "support": "complete_total_degree_O4",
        "partition": "B1",
        "partition_count": 1,
        "contract_identity": vdp_identity_hashes(),
        "h_min": contract["h_min"],
        "h_max": contract["h_max"],
        "effective_h_min": fixed_step if fixed_step is not None else contract["h_min"],
        "effective_h_max": fixed_step if fixed_step is not None else contract["h_max"],
        "schedule": {
            "kind": "fixed" if fixed_step is not None else "adaptive",
            "h_decimal": None if fixed_step is None else format(fixed_step, ".17g"),
            "h_hex": None if fixed_step is None else fixed_step.hex(),
            "requested_steps": requested_steps,
            "adaptive_fallback_allowed": fixed_step is None,
        },
        "cutoff": contract["cutoff"],
        "target_remainder_radius": contract["target_remainder_radius"],
        "tm_backend": args.tm_backend,
        "backend_lane": "hybrid_dense_core" if args.tm_backend == "dense" else "sparse_reference",
        "device": args.device,
        "reset_mode": args.reset_mode,
        "initialization_contract": args.initialization_contract,
        "right_map_center_mode": args.right_map_center_mode,
        "right_map_range_mode": args.right_map_range_mode,
        "validation_mode": contract["validation_mode"],
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
        "host_to_device_s": host_to_device_s,
        "dense_kernel_s": dense_kernel_s,
        "device_to_host_s": device_to_host_s,
        "nonkernel_nontransfer_solver_s": max(
            0.0,
            runtime - host_to_device_s - dense_kernel_s - device_to_host_s,
        ),
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
        "trace_io_s": trace_io_s,
        "trace_write_count": trace_write_count,
        "trace_flush_every": int(args.trace_flush_every),
        "peak_rss_bytes": _peak_rss_bytes(),
        "peak_rss_source": "getrusage_RUSAGE_SELF_ru_maxrss",
        "branch": command["branch"],
        "commit": command["commit"],
        "worktree_dirty": bool(command["worktree_status"]),
        "tracked_diff_sha256": command["tracked_diff_sha256"],
        "config_sha256": command["config_sha256"],
    }
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
    if transition_trace is not None:
        transition_trace.close(result_summary=summary)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tm-backend", choices=("sparse", "dense"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--initialization-contract",
        choices=("binary64_literal_matched_contract", "exact_decimal_contract"),
        default="binary64_literal_matched_contract",
        help="keep legacy default unchanged; authoritative 20260815 matrix selects exact_decimal_contract",
    )
    parser.add_argument("--horizon", type=float, default=10.0)
    parser.add_argument("--fixed-step", type=float)
    parser.add_argument(
        "--trace-flush-every",
        type=int,
        default=1,
        help="rewrite cumulative traces every N steps; zero writes once at termination",
    )
    parser.add_argument("--wall-cap-s", type=float, default=1800.0)
    parser.add_argument("--transition-trace-dir", type=Path)
    parser.add_argument(
        "--reset-mode",
        choices=(
            "normalized_insertion",
            NORMALIZED_INSERTION_DEPENDENCY_PRESERVING,
            "normalized_insertion_complete_polynomial",
            "normalized_insertion_symqueue_v2",
            "normalized_insertion_horner",
            "normalized_insertion_horner_symqueue_v2",
            "normalized_insertion_structured_remainder_k16",
            "normalized_insertion_bounded_source_ledger_o4_g1",
            "normalized_insertion_bounded_shared_source_o4_g2",
        ),
        default="normalized_insertion",
    )
    parser.add_argument("--right-map-center-mode", choices=("constant", "range_midpoint"), default="constant")
    parser.add_argument("--right-map-range-mode", choices=("standard", "normal_eval"), default="standard")
    parser.add_argument(
        "--validation-mode",
        choices=(
            "flowstar_raw_remainder_compat",
            "flowstar_raw_remainder_compat_factorized_joint",
            "flowstar_raw_remainder_compat_factorized_joint_closure",
            "flowstar_raw_remainder_compat_factorized_joint_closure_refined",
        ),
        help="opt-in dense raw-RHS operator; omission preserves the frozen contract default",
    )
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
