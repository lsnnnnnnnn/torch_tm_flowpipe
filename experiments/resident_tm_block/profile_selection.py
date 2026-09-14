"""Measure the two bounded resident-TM candidates on real solver windows.

This diagnostic does not replace any arithmetic.  It wraps the existing dense
Picard/validation block and the accepted-boundary normal-composition block,
recording both thread CPU and mutually-overlapped wall intervals.  A context
marker attributes live range requests to composition without adding their
per-task Future waits together as wall time.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextvars import ContextVar
import json
from pathlib import Path
import statistics
import threading
import time
from typing import Any, Callable

import torch

from experiments.live_range_solver import runner
import torch_tm_flowpipe.accepted_boundary_sr as accepted_sr
import torch_tm_flowpipe.batched_dense_tm as dense
import torch_tm_flowpipe.flowpipe as flowpipe
import torch_tm_flowpipe.live_range_service as live_service
import torch_tm_flowpipe.polynomial as polynomial


_IN_COMPOSITION: ContextVar[bool] = ContextVar("resident_selection_in_composition", default=False)


def _union_ns(intervals: list[tuple[int, int]]) -> int:
    total = 0
    end = -1
    for start, stop in sorted(intervals):
        if stop < start:
            raise AssertionError("negative profiling interval")
        if start > end:
            total += stop - start
            end = stop
        elif stop > end:
            total += stop - end
            end = stop
    return total


class Recorder:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.rows: list[dict[str, Any]] = []
        self.counts: Counter[str] = Counter()

    def add(self, name: str, start: int, stop: int, cpu_ns: int, **extra: Any) -> None:
        with self.lock:
            self.rows.append(
                {
                    "name": name,
                    "thread": threading.get_ident(),
                    "start_ns": start,
                    "stop_ns": stop,
                    "wall_ns": stop - start,
                    "thread_cpu_ns": cpu_ns,
                    **extra,
                }
            )

    def increment(self, name: str, value: int = 1) -> None:
        with self.lock:
            self.counts[name] += int(value)

    def summarize(self, wall_s: float) -> dict[str, Any]:
        names = sorted({row["name"] for row in self.rows})
        result: dict[str, Any] = {}
        for name in names:
            selected = [row for row in self.rows if row["name"] == name]
            walls = [row["wall_ns"] / 1e9 for row in selected]
            result[name] = {
                "calls": len(selected),
                "wall_span_sum_s": sum(walls),
                "wall_interval_union_s": _union_ns(
                    [(row["start_ns"], row["stop_ns"]) for row in selected]
                ) / 1e9,
                "thread_cpu_sum_s": sum(row["thread_cpu_ns"] for row in selected) / 1e9,
                "call_wall_p50_s": statistics.median(walls),
                "call_wall_max_s": max(walls),
                "fraction_of_invocation_by_union": (
                    _union_ns([(row["start_ns"], row["stop_ns"]) for row in selected])
                    / 1e9
                    / wall_s
                    if wall_s
                    else 0.0
                ),
            }
        return {"sections": result, "counts": dict(self.counts)}


class Patches:
    def __init__(self, recorder: Recorder) -> None:
        self.recorder = recorder
        self.undo: list[tuple[Any, str, Any]] = []

    def replace(self, module: Any, name: str, replacement: Any) -> None:
        self.undo.append((module, name, getattr(module, name)))
        setattr(module, name, replacement)

    def timed(self, module: Any, name: str, label: str, *, composition: bool = False) -> None:
        original = getattr(module, name)

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            token = _IN_COMPOSITION.set(True) if composition else None
            start = time.perf_counter_ns()
            cpu = time.thread_time_ns()
            try:
                return original(*args, **kwargs)
            finally:
                self.recorder.add(
                    label,
                    start,
                    time.perf_counter_ns(),
                    time.thread_time_ns() - cpu,
                )
                if token is not None:
                    _IN_COMPOSITION.reset(token)

        self.replace(module, name, wrapped)

    def install(self) -> None:
        self.timed(flowpipe, "_flowpipe_step_from_tm_hybrid_dense", "dense_picard_validation")
        self.timed(flowpipe, "_flowstar_normalized_insertion_transition", "accepted_boundary_transition")
        self.timed(
            flowpipe,
            "insert_ctrunc_normal_dependency_preserving",
            "normal_composition",
            composition=True,
        )
        # flowpipe imported these functions directly, so patch its bindings.
        self.timed(flowpipe, "prepare_accepted_boundary_sr", "accepted_boundary_prepare")
        self.timed(flowpipe, "commit_accepted_boundary_sr", "accepted_boundary_commit")
        self.timed(dense, "sparse_tmvector_to_dense", "sparse_to_dense")
        self.timed(dense, "dense_to_sparse_tmvector", "dense_to_sparse")

        original_mul = polynomial.Polynomial.mul_truncate

        def mul_wrapped(instance: Any, *args: Any, **kwargs: Any) -> Any:
            if _IN_COMPOSITION.get():
                self.recorder.increment("composition_polynomial_mul_truncate")
            return original_mul(instance, *args, **kwargs)

        self.replace(polynomial.Polynomial, "mul_truncate", mul_wrapped)

        original_evaluate = live_service.RangeTask.evaluate

        def evaluate_wrapped(instance: Any, request: Any) -> Any:
            inside = _IN_COMPOSITION.get()
            start = time.perf_counter_ns()
            cpu = time.thread_time_ns()
            if inside:
                self.recorder.increment("composition_range_requests")
            try:
                return original_evaluate(instance, request)
            finally:
                if inside:
                    self.recorder.add(
                        "composition_range_future_wait_span",
                        start,
                        time.perf_counter_ns(),
                        time.thread_time_ns() - cpu,
                    )

        self.replace(live_service.RangeTask, "evaluate", evaluate_wrapped)

    def restore(self) -> None:
        for module, name, original in reversed(self.undo):
            setattr(module, name, original)


def _service_totals(groups: list[dict[str, Any]]) -> dict[str, Any]:
    intervals = [(int(row["start_ns"]), int(row["end_ns"])) for row in groups]
    return {
        "groups": len(groups),
        "requests": sum(int(row["size"]) for row in groups),
        "service_span_sum_s": sum((stop - start) for start, stop in intervals) / 1e9,
        "service_interval_union_s": _union_ns(intervals) / 1e9,
        "service_thread_cpu_sum_s": sum(int(row["thread_cpu_ns"]) for row in groups) / 1e9,
    }


def run_window(spec: dict[str, Any]) -> dict[str, Any]:
    recorder = Recorder()
    patches = Patches(recorder)
    patches.install()
    try:
        result, _events, _states = runner.run_case(
            spec["plant"],
            spec["ids"],
            spec["steps"],
            "Gp",
            run_id=spec["name"],
            checkpoint=spec.get("checkpoint"),
            retain_service_groups=True,
            retain_service_waits=False,
        )
    finally:
        patches.restore()
    if result["successful_tasks"] != len(spec["ids"]):
        raise RuntimeError(f"selection window failed: {spec['name']}: {result['task_statuses']}")
    summary = recorder.summarize(float(result["wall_s"]))
    return {
        "name": spec["name"],
        "plant": spec["plant"],
        "ids": spec["ids"],
        "steps": spec["steps"],
        "scope": result["scope"],
        "checkpoint": spec.get("checkpoint"),
        "wall_s": result["wall_s"],
        "process_cpu_s": result["cpu_s"],
        "accepted_lane_steps": result["accepted_lane_steps"],
        "successful_tasks": result["successful_tasks"],
        "service": _service_totals(result["groups"]),
        **summary,
    }


def default_specs(root: Path, *, quick: bool) -> list[dict[str, Any]]:
    parent = root / "artifacts/runs/live_gpu_packets_20260910T023603Z/full_horizon"
    if quick:
        return [
            {"name": "selection-vdp-B2x2", "plant": "van_der_pol", "ids": [0, 31], "steps": 2},
            {"name": "selection-bruss-B2x2", "plant": "brusselator", "ids": [0, 31], "steps": 2},
        ]
    from experiments.range_batch_device.common import read

    partition = read(runner.PARTITION)
    return [
        {"name": "selection-vdp-B8x6", "plant": "van_der_pol", "ids": partition["subsets"]["8"], "steps": 6},
        {"name": "selection-bruss-B8x6", "plant": "brusselator", "ids": partition["subsets"]["8"], "steps": 6},
        {"name": "selection-vdp-B32x4", "plant": "van_der_pol", "ids": partition["subsets"]["32"], "steps": 4},
        {"name": "selection-bruss-B32x4", "plant": "brusselator", "ids": partition["subsets"]["32"], "steps": 4},
        {
            "name": "selection-vdp-resume-step99-two",
            "plant": "van_der_pol",
            "ids": [0],
            "steps": 2,
            "checkpoint": str(parent / "van_der_pol-Gp/checkpoints/step_0099"),
        },
        {
            "name": "selection-vdp-resume-step100-two",
            "plant": "van_der_pol",
            "ids": [0],
            "steps": 2,
            "checkpoint": str(parent / "van_der_pol-Gp/checkpoints/step_0100"),
        },
        {
            "name": "selection-bruss-resume-step999-one",
            "plant": "brusselator",
            "ids": [0],
            "steps": 1,
            "checkpoint": str(parent / "brusselator-Gp/checkpoints/step_0999"),
        },
        {
            "name": "selection-bruss-resume-step1000-one",
            "plant": "brusselator",
            "ids": [0],
            "steps": 1,
            "checkpoint": str(parent / "brusselator-Gp/checkpoints/step_1000"),
        },
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    root = Path(__file__).resolve().parents[2]

    # The selection campaign uses one process and one initialized thread budget.
    # CUDA compilation/self-test is explicitly outside every measured window.
    with live_service.LiveRangeService("cuda", run_id="resident-selection-warmup", packet_mode=True):
        pass
    torch.cuda.reset_peak_memory_stats(0)
    rows = []
    for spec in default_specs(root, quick=args.quick):
        print(f"START {spec['name']}", flush=True)
        row = run_window(spec)
        rows.append(row)
        print(json.dumps({"name": row["name"], "wall_s": row["wall_s"]}), flush=True)
    payload = {
        "schema": "resident-tm-block-selection-profile-v1",
        "route": "Gp",
        "source": "fresh live solver windows; no saved answers used to advance",
        "timing_semantics": {
            "wall_interval_union_s": "mutually overlapping intervals are unioned",
            "wall_span_sum_s": "per-call spans; not an additive wall-time claim",
            "thread_cpu_sum_s": "thread CPU, excluding blocked Future wait",
            "cuda_events": "not used by this selection profiler",
        },
        "windows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
