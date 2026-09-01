#!/usr/bin/env python3
"""Profile the frozen C3+C4 reference without mixing solver and evidence I/O."""
from __future__ import annotations

import argparse
import cProfile
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import pstats
import resource
import statistics
import sys
import tempfile
import time
import tracemalloc
from typing import Any, Callable, Mapping, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = Path(os.environ.get("C4_PROFILE_CODE_ROOT", str(ROOT))).resolve()
SRC = CODE_ROOT / "src"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from experiments.run_brusselator_sr1000_parity import (  # noqa: E402
    INITIAL_DECIMAL,
    ORDER,
    _policy,
    _step,
)
from experiments.run_vdp_dense_backend import load_contract  # noqa: E402
from torch_tm_flowpipe import (  # noqa: E402
    DENSE_OBSERVER_FULL,
    DENSE_OBSERVER_LIGHTWEIGHT,
    DENSE_OBSERVER_NONE,
    DenseRangePolicy,
    FlowstarLikePolynomialPlantConfig,
    FlowstarNormalFlowpipeState,
    PolynomialODE,
    accepted_boundary_sr_queue_sha256,
    flowpipe_step_flowstar_style_adaptive,
    load_terminal_checkpoint,
    save_terminal_checkpoint,
    tmvector_hashes,
)


PROFILE_BUCKETS = (
    "polynomial Picard construction",
    "initial raw-remainder image",
    "post-accept remainder replays",
    "range bounding/subdivision",
    "polynomial multiplication/truncation/cutoff",
    "SR history propagation",
    "normalization/right-map/reset",
    "outward interval/roundoff accounting",
    "Python orchestration/allocation",
    "audit/serialization",
    "other",
)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = sorted({str(key) for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _queue_hash(state: FlowstarNormalFlowpipeState) -> str:
    if state.symbolic_queue is None:
        raise RuntimeError("formal reference state lost its accepted-boundary queue")
    return accepted_boundary_sr_queue_sha256(state.symbolic_queue)


def _snapshot(segment: Any) -> dict[str, Any]:
    if (
        segment.status != "validated"
        or segment.endpoint_raw_tm is None
        or segment.reset_tm is None
        or segment.flowstar_normal_state is None
    ):
        raise RuntimeError(f"reference step failed: {segment.status}: {segment.message}")
    counters = dict(segment.backend_counters or {})
    return {
        "endpoint": tmvector_hashes(segment.endpoint_raw_tm),
        "tube": tmvector_hashes(segment.tm),
        "reset": tmvector_hashes(segment.reset_tm),
        "queue_sha256": _queue_hash(segment.flowstar_normal_state),
        "queue_generation": segment.flowstar_normal_state.symbolic_queue.generation,
        "candidate_remainder": segment.candidate_remainder,
        "final_remainder": segment.picard_image_remainder,
        "post_accept_replay_calls": counters.get("post_accept_replay_calls", 0),
        "post_accept_committed_replays": counters.get("post_accept_committed_replays", 0),
        "post_accept_stop_ratio_count": counters.get("post_accept_stop_ratio_count", 0),
    }


def _run_brusselator_steps(
    count: int,
    observer_mode: str,
    *,
    checkpoint: Path | None = None,
    start_step: int = 1,
) -> tuple[Any, Any, Any, int]:
    if checkpoint is None:
        state = FlowstarNormalFlowpipeState.from_exact_decimal_box(INITIAL_DECIMAL, ORDER)
        current = state.normalized_initial_tm(ORDER)
    else:
        loaded = load_terminal_checkpoint(checkpoint, expected_order=ORDER, expected_dtype="float64")
        state = loaded.normal_state
        current = loaded.current
        if state.step_index != start_step - 1:
            raise ValueError("tail checkpoint step does not match --tail-start-step")
    segment = None
    accepted = 0
    for step in range(start_step, start_step + int(count)):
        segment, _diagnostics = _step(
            current,
            state,
            step,
            _policy(),
            validation_mode="flowstar_raw_remainder_compat_refined",
            lane_label=observer_mode,
            observer_mode=observer_mode,
        )
        if segment.status != "validated" or segment.reset_tm is None or segment.flowstar_normal_state is None:
            raise RuntimeError(f"Brusselator reference rejected step {step}: {segment.message}")
        current = segment.reset_tm
        state = segment.flowstar_normal_state
        accepted += 1
    assert segment is not None
    return current, state, segment, accepted


def _vdp_initial() -> tuple[PolynomialODE, Any, FlowstarNormalFlowpipeState]:
    contract = load_contract()
    ode = PolynomialODE.from_system_spec(contract["canonical_system_spec"])
    config = FlowstarLikePolynomialPlantConfig.van_der_pol()
    state = FlowstarNormalFlowpipeState.from_exact_decimal_box(config.initial_decimal_box, config.order)
    return ode, state.normalized_initial_tm(config.order), state


def _run_vdp_prefix(count: int, observer_mode: str) -> tuple[Any, Any, Any, int]:
    config = FlowstarLikePolynomialPlantConfig.van_der_pol()
    ode, current, state = _vdp_initial()
    policy = DenseRangePolicy(**config.range_policy_mapping)
    segment = None
    for step in range(1, int(count) + 1):
        segment = flowpipe_step_flowstar_style_adaptive(
            ode,
            current,
            h=0.01,
            h_min=0.01,
            h_max=0.01,
            order=config.order,
            target_remainder_radius=config.target_remainder_radius,
            cutoff_threshold=config.cutoff,
            max_validation_attempts=2,
            validation_eps=config.validation_epsilon,
            validation_mode=config.post_accept_refinement_mode,
            reset_mode=config.accepted_boundary_sr_mode,
            step_policy_mode="flowstar_compat",
            flowstar_normal_state=state,
            flowstar_symbolic_queue_max_size=config.accepted_boundary_sr_capacity,
            right_map_center_mode=config.right_map_center_mode,
            right_map_range_mode=config.right_map_range_mode,
            tm_backend="dense",
            dense_device="cpu",
            dense_dtype=torch.float64,
            dense_range_policy=policy,
            dense_observer_mode=observer_mode,
        )
        if segment.status != "validated" or segment.reset_tm is None or segment.flowstar_normal_state is None:
            raise RuntimeError(f"VDP reference rejected fixed prefix step {step}: {segment.message}")
        current = segment.reset_tm
        state = segment.flowstar_normal_state
    assert segment is not None
    return current, state, segment, int(count)


class _TensorResultCounter:
    """Count Python-visible torch tensor-producing API results.

    This intentionally avoids a low-level dispatch/profiler mode: those modes
    changed the one-step wall time by tens of times and would make the required
    100-step attribution windows operationally misleading.  The wrapped calls
    return their original results unchanged.
    """

    _NAMES = (
        "arange",
        "as_tensor",
        "cat",
        "empty",
        "empty_like",
        "full",
        "full_like",
        "maximum",
        "minimum",
        "nextafter",
        "ones",
        "ones_like",
        "stack",
        "tensor",
        "zeros",
        "zeros_like",
    )

    def __init__(self) -> None:
        self.count = 0
        self.logical_bytes = 0
        self._originals: dict[str, Any] = {}

    def _record(self, result: Any) -> Any:
        values = result if isinstance(result, (tuple, list)) else (result,)
        for value in values:
            if isinstance(value, torch.Tensor):
                self.count += 1
                self.logical_bytes += int(value.numel()) * int(value.element_size())
        return result

    def __enter__(self) -> "_TensorResultCounter":
        for name in self._NAMES:
            original = getattr(torch, name)
            self._originals[name] = original

            def wrapped(*args, _original=original, **kwargs):
                return self._record(_original(*args, **kwargs))

            setattr(torch, name, wrapped)
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        del exc_type, exc, traceback
        for name, original in self._originals.items():
            setattr(torch, name, original)
        self._originals.clear()


def _measure_allocations(
    action: Callable[[], tuple[Any, Any, Any, int]],
) -> tuple[int, int, int, int, tuple[Any, Any, Any, int]]:
    tracemalloc.start()
    before = tracemalloc.take_snapshot()
    tensor_counter = _TensorResultCounter()
    with tensor_counter:
        result = action()
    after = tracemalloc.take_snapshot()
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    count = sum(max(0, stat.count_diff) for stat in after.compare_to(before, "lineno"))
    return (
        count,
        int(peak),
        tensor_counter.count,
        tensor_counter.logical_bytes,
        result,
    )


def _observer_measurements(repeats: int, prefix_steps: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    snapshots: dict[str, Any] = {}
    for observer_mode in (DENSE_OBSERVER_NONE, DENSE_OBSERVER_LIGHTWEIGHT, DENSE_OBSERVER_FULL):
        runtimes: list[float] = []
        allocation_count = 0
        allocation_peak = 0
        tensor_result_count = 0
        tensor_result_bytes = 0
        final_result = None
        for repeat in range(int(repeats)):
            started = time.perf_counter()
            if repeat == 0:
                (
                    allocation_count,
                    allocation_peak,
                    tensor_result_count,
                    tensor_result_bytes,
                    final_result,
                ) = _measure_allocations(lambda: _run_brusselator_steps(prefix_steps, observer_mode))
            else:
                final_result = _run_brusselator_steps(prefix_steps, observer_mode)
            runtimes.append(time.perf_counter() - started)
        assert final_result is not None
        current, state, segment, accepted = final_result
        snapshot = _snapshot(segment)
        snapshots[observer_mode] = snapshot
        trace_payload = json.dumps(list(segment.backend_trace), sort_keys=True, separators=(",", ":")).encode("utf-8")
        serialization_started = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="c4-observer-serialize-") as temporary:
            artifact = Path(temporary) / "trace.json"
            artifact.write_bytes(trace_payload)
            artifact_sha = hashlib.sha256(artifact.read_bytes()).hexdigest()
        serialization_s = time.perf_counter() - serialization_started
        checkpoint_started = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="c4-observer-checkpoint-") as temporary:
            manifest = save_terminal_checkpoint(
                Path(temporary),
                current=current,
                normal_state=state,
                scheduler={"accepted_steps": accepted},
                contract=FlowstarLikePolynomialPlantConfig.brusselator().as_dict(),
                provenance={"producer": "profile_c4_reference_solver", "observer_neutral": True},
            )
        checkpoint_s = time.perf_counter() - checkpoint_started
        rows.append(
            {
                "lane": observer_mode,
                "prefix_steps": prefix_steps,
                "repeats": repeats,
                "solver_wall_median_s": statistics.median(runtimes),
                "solver_wall_min_s": min(runtimes),
                "solver_wall_max_s": max(runtimes),
                "artifact_serialization_s": serialization_s,
                "checkpoint_export_s": checkpoint_s,
                "trace_construction_s": 0.0,
                "trace_rows": len(segment.backend_trace),
                "trace_bytes": len(trace_payload),
                "trace_sha256": artifact_sha,
                "checkpoint_sha256": manifest["full_checkpoint_sha256"],
                "python_positive_allocation_count": allocation_count,
                "tracemalloc_peak_bytes": allocation_peak,
                "temporary_tensor_result_count": tensor_result_count,
                "temporary_tensor_logical_bytes": tensor_result_bytes,
                "temporary_tensor_measurement": "Python-visible torch tensor-producing API results; views and persistent outputs included",
                "peak_rss_bytes": _peak_rss_bytes(),
                "accepted_steps": accepted,
                "rejected_steps": 0,
                "endpoint_sha256": snapshot["endpoint"]["tmvector_sha256"],
                "tube_sha256": snapshot["tube"]["tmvector_sha256"],
                "final_remainder": json.dumps(snapshot["final_remainder"], separators=(",", ":")),
                "queue_sha256": snapshot["queue_sha256"],
                "refinement_replay_calls": snapshot["post_accept_replay_calls"],
                "refinement_stop_ratio_count": snapshot["post_accept_stop_ratio_count"],
            }
        )
    production = next(row for row in rows if row["lane"] == DENSE_OBSERVER_NONE)
    for row in rows:
        row["trace_construction_s"] = max(
            0.0,
            float(row["solver_wall_median_s"]) - float(production["solver_wall_median_s"]),
        )
    scientific_keys = (
        "endpoint",
        "tube",
        "reset",
        "queue_sha256",
        "queue_generation",
        "candidate_remainder",
        "final_remainder",
        "post_accept_replay_calls",
        "post_accept_committed_replays",
        "post_accept_stop_ratio_count",
    )
    equality = all(
        {key: snapshots[DENSE_OBSERVER_NONE][key] for key in scientific_keys}
        == {key: snapshots[mode][key] for key in scientific_keys}
        for mode in (DENSE_OBSERVER_LIGHTWEIGHT, DENSE_OBSERVER_FULL)
    )
    return rows, {"scientific_equality": equality, "snapshots": snapshots}


def _bucket(filename: str, function: str) -> str:
    name = function.lower()
    path = filename.lower()
    if "dense_polynomial_picard" in name:
        return "polynomial Picard construction"
    if "_dense_flowstar_raw_compat_image" in name:
        return "initial raw-remainder image"
    if "post_accept" in name or "refinement" in name:
        return "post-accept remainder replays"
    if "accepted_boundary_sr" in name or "symbolic_remainder.py" in path:
        return "SR history propagation"
    if any(token in name for token in ("range", "subdivision", "horner")):
        return "range bounding/subdivision"
    if any(token in name for token in ("mul_trunc", "cutoff", "truncate", "integrat")):
        return "polynomial multiplication/truncation/cutoff"
    if any(token in name for token in ("normalized_insertion", "right_map", "reset")):
        return "normalization/right-map/reset"
    if "interval.py" in path or any(token in name for token in ("_down", "_up", "nextafter", "roundoff")):
        return "outward interval/roundoff accounting"
    if any(token in path for token in ("checkpoint", "audit_trace")) or any(token in name for token in ("json", "serialize", "write")):
        return "audit/serialization"
    if path.startswith("~") or "<built-in" in path or any(token in name for token in ("__init__", "clone", "append", "list")):
        return "Python orchestration/allocation"
    return "other"


def _profile_action(
    window: str,
    action: Callable[[], tuple[Any, Any, Any, int]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], str, dict[str, Any]]:
    (
        allocation_count,
        allocation_peak,
        tensor_result_count,
        tensor_result_bytes,
        _warm_result,
    ) = _measure_allocations(action)
    profiler = cProfile.Profile()
    started = time.perf_counter()
    profiler.enable()
    current, state, segment, accepted = action()
    profiler.disable()
    wall = time.perf_counter() - started
    stats = pstats.Stats(profiler)
    hotspot_rows: list[dict[str, Any]] = []
    bucket_totals = {bucket: 0.0 for bucket in PROFILE_BUCKETS}
    calls: dict[str, int] = {
        "range-bound calls": 0,
        "RHS term-evaluation calls": 0,
        "Taylor multiplication/truncation calls": 0,
        "post-accept replay calls": int((segment.backend_counters or {}).get("post_accept_replay_calls", 0)),
        "SR prepare calls": 0,
        "SR propagate calls": 0,
        "SR commit calls": 0,
        "checkpoint/trace calls": 0,
    }
    for (filename, line, function), (primitive, total, exclusive, inclusive, _callers) in stats.stats.items():
        bucket = _bucket(filename, function)
        bucket_totals[bucket] += float(exclusive)
        hotspot_rows.append(
            {
                "window": window,
                "bucket": bucket,
                "filename": filename,
                "line": line,
                "function": function,
                "primitive_calls": primitive,
                "total_calls": total,
                "exclusive_wall_s": exclusive,
                "inclusive_wall_s": inclusive,
                "inclusive_fraction_of_solver": inclusive / wall if wall else 0.0,
            }
        )
        if function == "_range_for_terms_with_policy":
            calls["range-bound calls"] += total
        if filename.endswith("polynomial_ode.py") and function in {"__call__", "evaluate_canonical_factorized"}:
            calls["RHS term-evaluation calls"] += total
        if function in {"mul_trunc", "square_trunc"}:
            calls["Taylor multiplication/truncation calls"] += total
        if function == "prepare_accepted_boundary_sr":
            calls["SR prepare calls"] += total
        if function == "accepted_boundary_sr_queue_propagate":
            calls["SR propagate calls"] += total
        if function == "accepted_boundary_sr_queue_commit":
            calls["SR commit calls"] += total
    hotspot_rows.extend(
        {
            "window": window,
            "bucket": bucket,
            "filename": "<exclusive_bucket_total>",
            "line": 0,
            "function": "<exclusive_bucket_total>",
            "primitive_calls": "",
            "total_calls": "",
            "exclusive_wall_s": value,
            "inclusive_wall_s": value,
            "inclusive_fraction_of_solver": value / wall if wall else 0.0,
        }
        for bucket, value in bucket_totals.items()
    )
    call_rows = [
        {"window": window, "metric": metric, "call_count": count}
        for metric, count in calls.items()
    ]
    allocation_rows = [
        {
            "window": window,
            "python_positive_allocation_count": allocation_count,
            "tracemalloc_peak_bytes": allocation_peak,
            "temporary_tensor_result_count": tensor_result_count,
            "temporary_tensor_logical_bytes": tensor_result_bytes,
            "temporary_tensor_measurement": "Python-visible torch tensor-producing API results; views and persistent outputs included",
            "peak_rss_bytes": _peak_rss_bytes(),
        }
    ]
    stream = io.StringIO()
    pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats("cumulative").print_stats(180)
    summary = {
        "window": window,
        "solver_wall_s": wall,
        "accepted_steps": accepted,
        "snapshot": _snapshot(segment),
        "final_queue_sha256": _queue_hash(state),
        "final_current_sha256": tmvector_hashes(current)["tmvector_sha256"],
        "exclusive_bucket_seconds": bucket_totals,
    }
    return hotspot_rows, call_rows, allocation_rows, stream.getvalue(), summary


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    observer_rows, observer_summary = _observer_measurements(
        args.observer_repeats,
        args.observer_prefix_steps,
    )
    if not observer_summary["scientific_equality"]:
        raise RuntimeError("observer lanes changed the frozen scientific state")

    actions: list[tuple[str, Callable[[], tuple[Any, Any, Any, int]]]] = [
        (
            f"brusselator_steps_1_{args.brusselator_prefix_steps}",
            lambda: _run_brusselator_steps(
                args.brusselator_prefix_steps,
                DENSE_OBSERVER_LIGHTWEIGHT,
            ),
        ),
        (
            f"vdp_fixed_prefix_1_{args.vdp_prefix_steps}",
            lambda: _run_vdp_prefix(args.vdp_prefix_steps, DENSE_OBSERVER_LIGHTWEIGHT),
        ),
    ]
    if args.tail_checkpoint is not None:
        actions.append(
            (
                f"brusselator_steps_{args.tail_start_step}_{args.tail_start_step + args.tail_steps - 1}",
                lambda: _run_brusselator_steps(
                    args.tail_steps,
                    DENSE_OBSERVER_LIGHTWEIGHT,
                    checkpoint=args.tail_checkpoint,
                    start_step=args.tail_start_step,
                ),
            )
        )

    hotspots: list[dict[str, Any]] = []
    calls: list[dict[str, Any]] = []
    allocations: list[dict[str, Any]] = []
    flame_parts: list[str] = []
    windows: list[dict[str, Any]] = []
    for name, action in actions:
        h_rows, c_rows, a_rows, flame, summary = _profile_action(name, action)
        hotspots.extend(h_rows)
        calls.extend(c_rows)
        allocations.extend(a_rows)
        flame_parts.append(f"===== {name} =====\n{flame}")
        windows.append(summary)

    _write_csv(output / "production_vs_audit_overhead.csv", observer_rows)
    _write_csv(output / "hotspot_profile.csv", hotspots)
    _write_csv(output / "call_count_matrix.csv", calls)
    _write_csv(output / "allocation_profile.csv", allocations)
    (output / "flamegraph.txt").write_text("\n".join(flame_parts), encoding="utf-8")
    result = {
        "schema": "torch_tm_flowpipe.c4_reference_profile/1",
        "observer_scientific_equality": True,
        "observer_rows": observer_rows,
        "profile_windows": windows,
        "profile_bucket_semantics": "mutually_exclusive_cProfile_exclusive_time_by_function_class",
        "tail_checkpoint": None if args.tail_checkpoint is None else str(args.tail_checkpoint.resolve()),
        "profile_code_root": str(CODE_ROOT),
        "status": "PROFILE_COMPLETE",
    }
    _write_json(output / "profile_summary.json", result)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--observer-repeats", type=int, default=3)
    parser.add_argument("--observer-prefix-steps", type=int, default=1)
    parser.add_argument("--brusselator-prefix-steps", type=int, default=20)
    parser.add_argument("--vdp-prefix-steps", type=int, default=5)
    parser.add_argument("--tail-checkpoint", type=Path)
    parser.add_argument("--tail-start-step", type=int, default=996)
    parser.add_argument("--tail-steps", type=int, default=5)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    result = run(parse_args(argv))
    print(json.dumps({"status": result["status"], "windows": len(result["profile_windows"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
