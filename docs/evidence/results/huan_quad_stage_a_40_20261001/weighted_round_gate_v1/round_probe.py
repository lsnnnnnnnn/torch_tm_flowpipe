#!/usr/bin/env python3
"""Observe the two existing early-weighted rounds in a fresh 40-step run."""

import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import time
import traceback

import numpy as np


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RoundProbe:
    def __init__(self, torch, output):
        self.torch, self.output = torch, output
        self.step = None
        self.active = None
        self.rows = []
        self.advance_calls = 0

    def install(self, prepared):
        torch, driver = prepared[:2]
        from flowstar_gpu import weighted_validation as weighted

        original_failed = weighted.validate_failed
        original_refine = weighted.refine_accepted
        original_advance = driver.advance_sparse

        def failed(*args, **kwargs):
            if self.active is None:
                return original_failed(*args, **kwargs)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            wall_start = time.perf_counter()
            output = original_failed(*args, **kwargs)
            end.record()
            torch.cuda.synchronize()
            self.active["rounds"].append({
                "accepted": output[0].detach().cpu().numpy().copy(),
                "new": output[1].detach().cpu().numpy().copy(),
                "info": output[2],
                "gpu_stream_span_ms": float(start.elapsed_time(end)),
                "synchronous_wall_s": time.perf_counter() - wall_start,
            })
            return output

        def refine(*args, **kwargs):
            if self.step is None or self.active is not None:
                raise RuntimeError("early-weighted call outside one numerical step")
            current, eligible = args[7:9]
            row = {
                "step": self.step,
                "before": current.detach().cpu().numpy().copy(),
                "eligible": eligible.detach().cpu().numpy().copy(),
                "rounds": [],
            }
            self.active = row
            try:
                result = original_refine(*args, **kwargs)
            finally:
                self.active = None
            row["final"] = result[0].detach().cpu().numpy().copy()
            running = row["before"].copy()
            details = []
            for index, item in enumerate(row["rounds"], 1):
                accepted, new = item["accepted"], item["new"]
                intersection = np.stack((np.maximum(running[..., 0], new[..., 0]),
                                         np.minimum(running[..., 1], new[..., 1])), -1)
                if np.any(accepted[:, None] & (intersection[..., 0] > intersection[..., 1])):
                    raise RuntimeError("diagnostic found disjoint accepted remainders")
                after = np.where(accepted[:, None, None], intersection, running)
                changed = np.any(running.view(np.uint64) != after.view(np.uint64), axis=(1, 2))
                first = np.flatnonzero(changed)[:3]
                details.append({
                    "round": index,
                    "accepted_count": int(accepted.sum()),
                    "accepted_mask": accepted.astype(np.uint8).tolist(),
                    "changed_lane_count": int(changed.sum()),
                    "changed_component_count": int(np.count_nonzero(running.view(np.uint64)
                                                                    != after.view(np.uint64))),
                    "first_changed_lanes": [{"lane": int(lane),
                                             "before": running[lane].tolist(),
                                             "after": after[lane].tolist()} for lane in first],
                    "gpu_stream_span_ms": item["gpu_stream_span_ms"],
                    "synchronous_wall_s": item["synchronous_wall_s"],
                    "validation_info": item["info"],
                })
                running = after
            row["details"] = details
            row["reconstruction_numeric_equal"] = bool(np.array_equal(running, row["final"]))
            row["reconstruction_bits_equal"] = bool(np.array_equal(
                running.view(np.uint64), row["final"].view(np.uint64)))
            self.rows.append(row)
            with (self.output / "ROUND_ROWS.jsonl").open("a") as stream:
                stream.write(json.dumps({"step": row["step"], "eligible_count": int(row["eligible"].sum()),
                                         "rounds": details,
                                         "reconstruction_numeric_equal": row["reconstruction_numeric_equal"],
                                         "reconstruction_bits_equal": row["reconstruction_bits_equal"]},
                                        allow_nan=False) + "\n")
            if not row["reconstruction_numeric_equal"]:
                raise RuntimeError(f"round reconstruction differs at step {self.step}")
            return result

        def advance(*args, **kwargs):
            self.advance_calls += 1
            self.step = self.advance_calls
            try:
                result = original_advance(*args, **kwargs)
            finally:
                self.step = None
            accepted = result[1].detach().cpu().numpy()
            if not bool(accepted.all()):
                (self.output / "FIRST_REJECT.json").write_text(json.dumps({
                    "step": self.advance_calls, "accepted_count": int(accepted.sum()),
                    "rejected_lanes": np.flatnonzero(~accepted).tolist()}, indent=2) + "\n")
                raise RuntimeError(f"first numerical rejection at step {self.advance_calls}")
            return result

        weighted.validate_failed = failed
        weighted.refine_accepted = refine
        driver.advance_sparse = advance

    def save_arrays(self):
        if not self.rows:
            return []
        rounds = [(row["step"], index, item) for row in self.rows
                  for index, item in enumerate(row["rounds"], 1)]
        arrays = {
            "STEP_CURRENT_BEFORE.npy": np.stack([row["before"] for row in self.rows]),
            "STEP_CURRENT_AFTER.npy": np.stack([row["final"] for row in self.rows]),
        }
        if rounds:
            arrays.update({
                "ROUND_STEP_INDEX.npy": np.asarray([(step, index) for step, index, _ in rounds], dtype=np.int16),
                "ROUND_ACCEPTED_MASK.npy": np.stack([item["accepted"] for _, _, item in rounds]),
                "ROUND_NEW_REMAINDER.npy": np.stack([item["new"] for _, _, item in rounds]),
            })
        for name, value in arrays.items():
            np.save(self.output / name, value, allow_pickle=False)
        return [{"name": name, "shape": list(value.shape), "dtype": str(value.dtype)}
                for name, value in arrays.items()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-runner", "source-config", "comparison-helper", "reference", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("new data directory must not already exist")
    source = args.source_runner.read_text()
    if not all(part in source for part in ("def guard():", "hashlib.new = prohibited",
                                           "extension.load = extension.load_inline = prohibited",
                                           "torch.jit.script = python_script", "torch.compile = prohibited")):
        parser.error("saved runner lacks no-digest/no-JIT guards")
    if "snapshot.restore" in source:
        parser.error("saved runner enters old snapshot restore")
    helpers = load("saved_phase_helpers", args.comparison_helper)
    base = helpers.load_runner(args.source_runner)
    original_prepare = base.prepare
    holder = {}

    def prepared(batch):
        result = original_prepare(batch)
        probe = RoundProbe(result[0], args.output)
        probe.install(result)
        holder["probe"] = probe
        return result

    base.prepare = prepared
    started_utc = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    status, comparison, error = "failed", None, None
    try:
        code = base.execute(argparse.Namespace(mode="observer_on", source_config=args.source_config,
                                               output=args.output))
        if code != 0:
            raise RuntimeError(f"saved runner returned {code}")
        comparison = helpers.compare_direct(args.reference, args.output)
        status = "completed_and_direct_equal" if comparison["pass"] else "direct_mismatch"
    except BaseException as exc:
        error = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
    finally:
        probe = holder.get("probe")
        arrays = probe.save_arrays() if probe else []
        record = {
            "schema": "quad-old-author-p3-weighted-round-gate-nohash-v1",
            "status": status, "started_utc": started_utc,
            "source_runner": str(args.source_runner), "source_config": str(args.source_config),
            "comparison_helper": str(args.comparison_helper), "reference": str(args.reference),
            "requested_steps": 40, "requested_boxes": 1024,
            "observed_steps": len(probe.rows) if probe else 0,
            "advance_calls": probe.advance_calls if probe else 0,
            "rounds_per_step": [len(row["rounds"]) for row in probe.rows] if probe else [],
            "round_array_files": arrays, "comparison": comparison, "error": error,
            "outer_wall_s": time.perf_counter() - started,
            "timing_limit": "CUDA events plus synchronization and diagnostic copies perturb timing; 40-step phase observation only",
            "full_horizon_inference": False, "end_to_end_certificate": False,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        (args.output.parent / "ROUND_PROBE.json").write_text(json.dumps(record, indent=2,
                                                                       allow_nan=False) + "\n")
    print(json.dumps({"status": status, "steps": record["observed_steps"],
                      "wall_s": record["outer_wall_s"]}))
    return 0 if status == "completed_and_direct_equal" else 1


if __name__ == "__main__":
    raise SystemExit(main())
