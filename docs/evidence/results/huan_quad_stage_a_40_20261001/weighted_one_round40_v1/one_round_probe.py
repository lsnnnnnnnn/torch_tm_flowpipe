#!/usr/bin/env python3
"""Isolated old-author QUAD P3 one-round, at most 40-step diagnostic."""

import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys
import time
import traceback


class FirstNumericalReject(RuntimeError):
    pass


def load_runner(path):
    spec = importlib.util.spec_from_file_location("quad_old_p3_observer_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load source runner: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def compare_saved(reference, output):
    import numpy as np

    old = rows(reference / "observations.jsonl")
    new = rows(output / "observations.jsonl")
    if len(new) > len(old):
        raise ValueError("variant saved more steps than the 40-step reference")
    changed = []
    max_abs_delta = 0.0
    for i, (a, b) in enumerate(zip(old, new), 1):
        if a["substep"] != b["substep"] or a["substep"] != i:
            raise ValueError(f"observation step mismatch at {i}")
        aa = np.asarray(a["tube_endpoint_union_12x4"], dtype=np.float64)
        bb = np.asarray(b["tube_endpoint_union_12x4"], dtype=np.float64)
        if aa.shape != bb.shape or aa.shape != (12, 4) or not np.isfinite(bb).all():
            raise ValueError(f"invalid saved union at {i}")
        if (bb[:, 0] > bb[:, 1]).any() or (bb[:, 2] > bb[:, 3]).any():
            raise ValueError(f"unordered saved union at {i}")
        if not np.array_equal(aa, bb):
            changed.append(i)
        max_abs_delta = max(max_abs_delta, float(np.max(np.abs(aa - bb))))
    result = {
        "reference_observed_steps": len(old),
        "variant_observed_steps": len(new),
        "common_step_all_accepted": all(a["accepted_count"] == b["accepted_count"] == 1024
                                    and not b["rejected_lanes"] for a, b in zip(old, new)),
        "changed_saved_union_steps": changed,
        "max_abs_saved_union_endpoint_delta": max_abs_delta,
    }
    if len(new) == 40:
        arrays = {}
        for name in ("final_tube_12x2", "final_endpoint_12x2"):
            a = np.load(reference / (name + ".npy"), allow_pickle=False)
            b = np.load(output / (name + ".npy"), allow_pickle=False)
            if a.shape != b.shape or a.shape != (1024, 12, 2) or not np.isfinite(b).all():
                raise ValueError(f"invalid final {name}")
            if (b[..., 0] > b[..., 1]).any():
                raise ValueError(f"unordered final {name}")
            arrays[name] = {
                "direct_equal": bool(np.array_equal(a.view(np.uint64), b.view(np.uint64))),
                "changed_interval_endpoints": int(np.count_nonzero(a.view(np.uint64) != b.view(np.uint64))),
                "reference_width_sum": float(np.sum(a[..., 1] - a[..., 0])),
                "variant_width_sum": float(np.sum(b[..., 1] - b[..., 0])),
                "max_abs_endpoint_delta": float(np.max(np.abs(a - b))),
            }
        a = np.load(reference / "final_status.npy", allow_pickle=False)
        b = np.load(output / "final_status.npy", allow_pickle=False)
        arrays["final_status"] = {"direct_equal": bool(np.array_equal(a, b)),
                                  "reference_counts": dict(zip(*np.unique(a, return_counts=True))),
                                  "variant_counts": dict(zip(*np.unique(b, return_counts=True)))}
        arrays["final_status"]["reference_counts"] = {
            str(int(k)): int(v) for k, v in arrays["final_status"]["reference_counts"].items()}
        arrays["final_status"]["variant_counts"] = {
            str(int(k)): int(v) for k, v in arrays["final_status"]["variant_counts"].items()}
        result["final_arrays"] = arrays
        old_metrics = json.loads((reference / "metrics.json").read_text())
        new_metrics = json.loads((output / "metrics.json").read_text())
        result["final_metrics"] = {
            "reference_hull": old_metrics["final_hull"],
            "variant_hull": new_metrics["final_hull"],
            "reference_mean_width_sum": old_metrics["final_hull_width_sum_mean"],
            "variant_mean_width_sum": new_metrics["final_hull_width_sum_mean"],
            "reference_broken": old_metrics["broken"],
            "variant_broken": new_metrics["broken"],
        }
    return result


class Probe:
    def __init__(self, output):
        self.output = output
        self.step = None
        self.advance_calls = 0
        self.round_rows = []
        self.first_reject = None

    def install(self, prepared):
        torch, driver = prepared[:2]
        from flowstar_gpu import weighted_validation as weighted

        original_refine = weighted.refine_accepted
        original_advance = driver.advance_sparse

        def one_round(*args, **kwargs):
            if self.step is None or "rounds" in kwargs:
                raise RuntimeError("unexpected weighted refinement call")
            result, info = original_refine(*args, rounds=1, **kwargs)
            if len(info["rounds"]) != 1:
                raise RuntimeError("weighted refinement did not execute one round")
            self.round_rows.append({"step": self.step, "eligible": int(info["eligible"]),
                                    "round": info["rounds"][0]})
            return result, info

        def advance(*args, **kwargs):
            self.advance_calls += 1
            self.step = self.advance_calls
            try:
                state, accepted = original_advance(*args, **kwargs)
                mask = accepted.detach().cpu().numpy()
                if not bool(mask.all()):
                    import numpy as np
                    self.first_reject = {"step": self.step, "accepted_count": int(mask.sum()),
                                         "rejected_lanes": np.flatnonzero(~mask).tolist()}
                    (self.output / "FIRST_REJECT.json").write_text(
                        json.dumps(self.first_reject, indent=2) + "\n")
                    raise FirstNumericalReject(f"first numerical refusal at step {self.step}")
                return state, accepted
            finally:
                self.step = None

        weighted.refine_accepted = one_round
        driver.advance_sparse = advance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-runner", type=Path, required=True)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("new output directory already exists")
    if not (args.reference / "RESULT.json").is_file():
        parser.error("old two-round reference is absent")
    source = args.source_runner.read_text()
    required = ("hashlib.new = prohibited", "extension.load = extension.load_inline = prohibited",
                "torch.jit.script = python_script", "torch.compile = prohibited")
    if not all(item in source for item in required) or "snapshot.restore" in source:
        parser.error("source runner does not match the isolated no-digest/no-JIT route")
    base = load_runner(args.source_runner)
    original_prepare = base.prepare
    holder = {}

    def prepared(batch):
        parts = original_prepare(batch)
        probe = Probe(args.output)
        probe.install(parts)
        holder["probe"] = probe
        return parts

    base.prepare = prepared
    started = time.perf_counter()
    status = "exception"
    error = None
    code = None
    comparison = None
    try:
        code = base.execute(argparse.Namespace(mode="observer_on", source_config=args.source_config,
                                               output=args.output))
        if code != 0:
            raise RuntimeError(f"base runner returned {code}")
        status = "completed_40_one_round"
    except FirstNumericalReject as exc:
        status = "first_numerical_reject"
        error = {"type": type(exc).__name__, "message": str(exc)}
    except BaseException as exc:
        error = {"type": type(exc).__name__, "message": str(exc),
                 "traceback": traceback.format_exc()}
    finally:
        if (args.output / "observations.jsonl").is_file():
            try:
                comparison = compare_saved(args.reference, args.output)
            except BaseException as exc:
                if status == "completed_40_one_round":
                    status = "comparison_failed"
                error = {"type": type(exc).__name__, "message": str(exc),
                         "traceback": traceback.format_exc()}
        probe = holder.get("probe")
        record = {"schema": "quad-old-author-p3-one-round40-nohash-v1",
                  "status": status, "started_utc": datetime.now(timezone.utc).isoformat(),
                  "source_runner": str(args.source_runner), "source_config": str(args.source_config),
                  "reference": str(args.reference), "requested_steps": 40, "requested_boxes": 1024,
                  "variant": "early weighted refinement rounds=1; original=2; all other runner settings fixed",
                  "observed_advance_calls": probe.advance_calls if probe else 0,
                  "round_rows": probe.round_rows if probe else [],
                  "first_numerical_reject": probe.first_reject if probe else None,
                  "base_exit_code": code, "comparison": comparison, "error": error,
                  "outer_wall_s": time.perf_counter() - started,
                  "full_1000_step_inference": False, "end_to_end_certificate": False,
                  "timing_qualification": "single short run; no full-horizon speed claim"}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        (args.output.parent / "ONE_ROUND_RESULT.json").write_text(
            json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": status,
                      "observed_advance_calls": record["observed_advance_calls"],
                      "wall_s": record["outer_wall_s"]}))
    return 0 if status == "completed_40_one_round" else 1


if __name__ == "__main__":
    raise SystemExit(main())
