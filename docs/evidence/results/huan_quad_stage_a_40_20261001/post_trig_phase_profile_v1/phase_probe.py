#!/usr/bin/env python3
"""Bounded old-author QUAD P3 phase profile via the existing no-digest runner."""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys
import time
import traceback


def read_json(path):
    return json.loads(path.read_text())


def load_runner(path):
    spec = importlib.util.spec_from_file_location("quad_old_p3_observer_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class PhaseProbe:
    def __init__(self, torch):
        self.torch = torch
        self.rows = []
        self.current = None
        self.depth = 0
        self.suppressed_capture_calls = 0

    def wrap(self, label, original):
        def call(*args, **kwargs):
            if self.current is None:
                return original(*args, **kwargs)
            if self.torch.cuda.is_current_stream_capturing():
                self.suppressed_capture_calls += 1
                return original(*args, **kwargs)
            if self.depth:
                counts = self.current["nested_call_counts"]
                counts[label] = counts.get(label, 0) + 1
                return original(*args, **kwargs)
            start = self.torch.cuda.Event(enable_timing=True)
            end = self.torch.cuda.Event(enable_timing=True)
            start.record()
            before = time.perf_counter()
            self.depth += 1
            try:
                return original(*args, **kwargs)
            finally:
                self.depth -= 1
                end.record()
                self.current["pending"].append((label, start, end, time.perf_counter() - before))

        return call

    def advance(self, original):
        def call(*args, **kwargs):
            if self.current is not None or self.depth:
                raise RuntimeError("overlapping advance profile")
            row = {"step": len(self.rows) + 1, "phases": [], "nested_call_counts": {}, "pending": []}
            self.current = row
            start = self.torch.cuda.Event(enable_timing=True)
            end = self.torch.cuda.Event(enable_timing=True)
            start.record()
            before = time.perf_counter()
            outcome = None
            try:
                outcome = original(*args, **kwargs)
                return outcome
            finally:
                end.record()
                # One synchronization after the original advance. The saved
                # observer arm already synchronizes to publish each step.
                self.torch.cuda.synchronize()
                row["advance_call_wall_s"] = time.perf_counter() - before
                row["advance_gpu_span_ms"] = float(start.elapsed_time(end))
                for label, first, last, cpu_wall in row.pop("pending"):
                    row["phases"].append({"phase": label, "cpu_call_wall_s": cpu_wall,
                                          "gpu_stream_span_ms": float(first.elapsed_time(last))})
                if outcome is not None:
                    accepted = outcome[1].detach().cpu()
                    row["accepted_lanes"] = int(accepted.sum().item())
                    row["all_lanes_accepted"] = bool(accepted.all().item())
                self.rows.append(row)
                self.current = None
                if outcome is not None and not row["all_lanes_accepted"]:
                    raise RuntimeError(f"first rejected lane at substep {row['step']}; stop")

        return call

    def install(self, prepared):
        torch, driver, se, host = prepared[:4]
        from flowstar_gpu import weighted_validation as weighted

        paths = [
            (se, "compose_s", "compose"),
            (se, "_structural_picard", "structural_picard"),
            (se, "_graphed_valid", "ordinary_validation"),
            (se, "_refine_dispatch", "ordinary_refinement"),
            (se, "_retry_self_map", "self_map_retry"),
            (weighted, "refine_accepted", "weighted_accepted"),
            (weighted, "validate_failed", "weighted_failed"),
            (weighted, "validate_recentered", "weighted_recentered"),
            (host.HostFactorLedger, "propagate", "sr_host_propagate"),
        ]
        for owner, name, label in paths:
            original = getattr(owner, name)
            if not callable(original):
                raise RuntimeError(f"phase path absent: {name}")
            setattr(owner, name, self.wrap(label, original))
        driver.advance_sparse = self.advance(driver.advance_sparse)
        self.prune = prepared[11]

    def summary(self):
        totals = defaultdict(lambda: {"calls": 0, "cpu_call_wall_s": 0.0,
                                      "gpu_stream_span_ms": 0.0})
        for row in self.rows:
            for phase in row["phases"]:
                item = totals[phase["phase"]]
                item["calls"] += 1
                item["cpu_call_wall_s"] += phase["cpu_call_wall_s"]
                item["gpu_stream_span_ms"] += phase["gpu_stream_span_ms"]
        prune_rows = list(self.prune.rows) if getattr(self, "prune", None) else []
        return {"completed_profiled_steps": len(self.rows),
                "all_profiled_steps_accepted": all(row.get("all_lanes_accepted") for row in self.rows),
                "advance_call_wall_s": sum(row["advance_call_wall_s"] for row in self.rows),
                "advance_gpu_stream_span_ms": sum(row["advance_gpu_span_ms"] for row in self.rows),
                "outer_phase_totals": dict(sorted(totals.items())),
                "graph_prune_steps": len(prune_rows),
                "graph_prune_wall_s": sum(row["elapsed_s"] for row in prune_rows),
                "graph_prune_nonempty_steps": [row["sequence"] + 1 for row in prune_rows
                                               if row["evicted_entries"]],
                "graph_prune_evicted_entries": sum(row["evicted_entries"] for row in prune_rows),
                "suppressed_calls_during_capture": self.suppressed_capture_calls,
                "timing_limit": "CUDA events are stream spans, not kernel sums; outer phases exclude uncovered advance work. One extra sync per advance perturbs timing. No full-run speed inference."}


def compare_direct(reference, output):
    import numpy as np

    fields = ("final_tube_12x2", "final_endpoint_12x2", "final_status")
    arrays = {}
    for name in fields:
        old = np.load(reference / (name + ".npy"), allow_pickle=False)
        new = np.load(output / (name + ".npy"), allow_pickle=False)
        arrays[name] = {"shape": list(new.shape), "dtype": str(new.dtype),
                        "direct_equal": bool(old.shape == new.shape and old.dtype == new.dtype
                                             and np.array_equal(old, new))}
    old_rows = [read_json_line(line) for line in (reference / "observations.jsonl").read_text().splitlines()]
    new_rows = [read_json_line(line) for line in (output / "observations.jsonl").read_text().splitlines()]
    rows_equal = old_rows == new_rows
    old_metrics = read_json(reference / "metrics.json")
    new_metrics = read_json(output / "metrics.json")
    metric_fields = ("B", "steps", "substeps", "broken", "ctrl_steps", "final_hull",
                     "final_hull_width_sum_mean", "order", "coupling")
    metrics_equal = {field: old_metrics[field] == new_metrics[field] for field in metric_fields}
    return {"reference": str(reference), "profiled": str(output), "arrays": arrays,
            "all_saved_observation_rows_direct_equal": rows_equal,
            "observation_count_reference": len(old_rows), "observation_count_profiled": len(new_rows),
            "metrics_direct_equal": metrics_equal,
            "pass": rows_equal and all(value["direct_equal"] for value in arrays.values())
                    and all(metrics_equal.values())}


def read_json_line(line):
    return json.loads(line)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-runner", type=Path, required=True)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("profile output already exists")
    base = load_runner(args.source_runner)
    if not all(part in args.source_runner.read_text() for part in
               ("def guard():", "hashlib.new = prohibited", "extension.load = extension.load_inline = prohibited",
                "torch.jit.script = python_script", "torch.compile = prohibited")):
        parser.error("source runner lacks the saved no-digest/no-JIT guards")
    if "snapshot.restore" in args.source_runner.read_text():
        parser.error("source runner enters old snapshot restore")
    original_prepare = base.prepare
    holder = {}

    def prepared(batch):
        result = original_prepare(batch)
        probe = PhaseProbe(result[0])
        probe.install(result)
        holder["probe"] = probe
        return result

    base.prepare = prepared
    started = time.perf_counter()
    status = "failed"
    error = None
    try:
        code = base.execute(argparse.Namespace(mode="observer_on", source_config=args.source_config,
                                               output=args.output))
        if code != 0:
            raise RuntimeError(f"base runner returned {code}")
        comparison = compare_direct(args.reference, args.output)
        status = "completed_and_direct_equal" if comparison["pass"] else "direct_mismatch"
    except BaseException as exc:
        error = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        comparison = None
    finally:
        record = {"schema": "quad-old-author-p3-post-trig-phase-profile-nohash-v1",
                  "status": status, "started_utc": datetime.now(timezone.utc).isoformat(),
                  "source_runner": str(args.source_runner), "source_config": str(args.source_config),
                  "source_reference": str(args.reference), "source_runner_bytes": args.source_runner.stat().st_size,
                  "profile_script_bytes": Path(__file__).stat().st_size,
                  "requested_steps": 40, "requested_boxes": 1024,
                  "profile": holder["probe"].summary() if "probe" in holder else None,
                  "profile_rows": holder["probe"].rows if "probe" in holder else [],
                  "comparison": comparison, "error": error,
                  "outer_wall_s": time.perf_counter() - started,
                  "full_horizon_inference": False, "end_to_end_certificate": False}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        (args.output.parent / "PHASE_PROFILE.json").write_text(json.dumps(record, indent=2,
                                                                            allow_nan=False) + "\n")
    print(json.dumps({"status": status, "steps": len(holder["probe"].rows)
                      if "probe" in holder else 0, "wall_s": record["outer_wall_s"]}))
    return 0 if status == "completed_and_direct_equal" else 1


if __name__ == "__main__":
    raise SystemExit(main())
