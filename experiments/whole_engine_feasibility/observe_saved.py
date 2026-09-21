"""Observe every accepted candidate snapshot offline with exact composition."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import threading
import time

from .export import exact_composition, canonical_models
from .observe import observe_step

_FIELDS = ["plant", "lane", "step", "view", "coordinate", "h_hex",
           "t_start_exact", "t_end_exact", "time", "lo", "hi", "lo_hex",
           "hi_hex", "width", "observer"]


def _reject_constant(value):
    raise ValueError(f"nonfinite JSON constant: {value}")


def _read_json(text):
    return json.loads(text, parse_constant=_reject_constant)


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("snapshot coefficients must be finite JSON numbers")
    if not math.isfinite(value):
        raise ValueError("nonfinite snapshot number")


def _validate(record, expected_step, lane_ids, h):
    if type(record["step"]) is not int or record["step"] != expected_step:
        raise ValueError(f"noncontiguous accepted snapshot: expected step {expected_step}")
    if record["lane_ids"] != lane_ids or len(set(lane_ids)) != len(lane_ids):
        raise ValueError("lane IDs changed or are duplicated")
    if record["accepted"] != [True] * len(lane_ids) or any(
            type(flag) is not bool for flag in record["accepted"]):
        raise ValueError("snapshot contains an unaccepted lane")
    if float.fromhex(record["h_hex"]) != h:
        raise ValueError("fixed step changed")
    for name in ("pre", "tmv"):
        exponents = record[name + "_exponents"]
        if not exponents or any(
                len(e) != 3 or any(type(v) is not int or v < 0 for v in e)
                for e in exponents):
            raise ValueError("invalid support")
        if len(set(map(tuple, exponents))) != len(exponents):
            raise ValueError("duplicate support terms")
        if name == "tmv" and any(e[0] != 0 for e in exponents):
            raise ValueError("history map is not time free")
        coeffs, rems = record[name], record[name + "_rem"]
        if len(coeffs) != len(lane_ids) or len(rems) != len(lane_ids):
            raise ValueError("batch shape mismatch")
        for components, remainders in zip(coeffs, rems):
            if len(components) != 2 or len(remainders) != 2:
                raise ValueError("expected two state coordinates")
            for terms, pair in zip(components, remainders):
                if len(terms) != len(exponents) or len(pair) != 2:
                    raise ValueError("coefficient/support or remainder mismatch")
                for value in [*terms, *pair]:
                    _number(value)
                if pair[0] > pair[1]:
                    raise ValueError("inverted remainder")


def run(input_path, output):
    input_path, output = Path(input_path).resolve(), Path(output).resolve()
    source_path = input_path.with_name("summary.json")
    source = _read_json(source_path.read_text())
    lane_ids = source["case"]["lane_ids"]
    expected_counts = source["accepted_steps"]
    if len(expected_counts) != len(lane_ids) or any(
            type(n) is not int or n < 0 for n in expected_counts):
        raise ValueError("invalid source accepted counts")
    if len(set(expected_counts)) != 1:
        raise ValueError("this snapshot format requires whole-batch atomic steps")
    h = float(source["settings"]["step"])
    if not math.isfinite(h) or h <= 0:
        raise ValueError("invalid source fixed step")
    output.mkdir(parents=True, exist_ok=False)
    digest = hashlib.sha256()
    with input_path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    started = time.perf_counter()
    counts = [0] * len(lane_ids)
    report = {
        "schema": "whole-engine-exact-observation/1",
        "input": str(input_path), "input_sha256": digest.hexdigest(),
        "source_summary": str(source_path),
        "source_summary_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "plant": source["plant"], "lane_ids": lane_ids,
        "expected_accepted_steps": expected_counts, "observed_steps": counts,
        "solver_completed": source["completed"], "requested_steps": source["requested_steps"],
        "status": "running", "complete_observation": False,
        "model_records": 0, "bound_rows": 0,
        "max_export_seconds": 0.0, "max_terms_per_component": 0,
        "warnings": [], "observer": "EXACT_FRACTION_COMPLETE_TM",
        "composition": "full-degree exact Fraction interval-coefficient substitution; no truncation",
        "timing": "offline observation only; never solver timing",
    }

    def checkpoint():
        report["offline_wall_seconds"] = time.perf_counter() - started
        (output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    checkpoint()
    try:
        with gzip.open(input_path, "rt") as snapshots, \
             gzip.open(output / "models.jsonl.gz", "wt") as model_file, \
             (output / "bounds.csv").open("w", newline="") as bound_file:
            writer = csv.DictWriter(bound_file, fieldnames=_FIELDS)
            writer.writeheader()
            for expected_step, line in enumerate(snapshots, 1):
                record = _read_json(line)
                _validate(record, expected_step, lane_ids, h)
                if expected_step > expected_counts[0]:
                    raise ValueError("snapshot exceeds source accepted count")
                for index, lane in enumerate(lane_ids):
                    context = {"step": expected_step, "lane": lane}
                    timer = threading.Timer(10.0, lambda c=context: print(
                        json.dumps(dict(event="slow_exact_export_running", **c)), flush=True))
                    timer.daemon = True
                    timer.start()
                    export_start = time.perf_counter()
                    try:
                        components = exact_composition(
                            record["pre"][index], record["pre_rem"][index],
                            record["pre_exponents"], record["tmv"][index],
                            record["tmv_rem"][index], record["tmv_exponents"])
                        models = canonical_models(components, h)
                    finally:
                        timer.cancel()
                    elapsed = time.perf_counter() - export_start
                    terms = [len(component) for component in components]
                    report["max_export_seconds"] = max(report["max_export_seconds"], elapsed)
                    report["max_terms_per_component"] = max(report["max_terms_per_component"], *terms)
                    if elapsed > 10.0 or max(terms) > 50000:
                        warning = dict(context, export_seconds=elapsed, terms=terms,
                                       event="large_or_slow_exact_export", action="recorded_and_continued")
                        report["warnings"].append(warning)
                        print(json.dumps(warning), flush=True)
                        checkpoint()
                    rows = observe_step(models, plant=source["plant"], step=expected_step, h=h, lane=lane)
                    model_file.write(json.dumps(dict(
                        plant=source["plant"], step=expected_step, lane=lane,
                        export_seconds=elapsed, terms=terms, models=models),
                        separators=(",", ":"), allow_nan=False) + "\n")
                    writer.writerows(rows)
                    counts[index] += 1
                    report["model_records"] += 1
                    report["bound_rows"] += len(rows)
                if expected_step % 100 == 0:
                    model_file.flush()
                    bound_file.flush()
                    checkpoint()
                    print(json.dumps({"event": "observation_progress", "step": expected_step,
                                      "lanes": len(lane_ids), "bound_rows": report["bound_rows"],
                                      "offline_wall_seconds": report["offline_wall_seconds"]}), flush=True)
        if counts != expected_counts:
            raise ValueError(f"missing accepted snapshots: observed {counts}, expected {expected_counts}")
        if report["bound_rows"] != 4 * sum(expected_counts):
            raise ValueError("four-channel observation count mismatch")
        report["status"] = "complete"
        report["complete_observation"] = True
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        checkpoint()
        raise
    checkpoint()
    print(json.dumps({"event": "observation_complete", "observed_steps": counts,
                      "bound_rows": report["bound_rows"],
                      "offline_wall_seconds": report["offline_wall_seconds"]}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.input, args.output)


if __name__ == "__main__":
    main()
