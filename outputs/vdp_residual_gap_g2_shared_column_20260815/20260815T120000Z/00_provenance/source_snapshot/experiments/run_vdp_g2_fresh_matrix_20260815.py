#!/usr/bin/env python3
"""Run the frozen exact-decimal legacy/G1/G2 request matrix.

Each request is a new process and a new initial state.  Large per-step tables
are retained losslessly as deterministic gzip streams; summary and decision
files remain directly readable.  Matrix concurrency is for evidence latency,
not for performance claims.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import gzip
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "experiments/run_vdp_dense_backend.py"
MODES = {
    "legacy": "normalized_insertion",
    "g1": "normalized_insertion_bounded_source_ledger_o4_g1",
    "g2": "normalized_insertion_bounded_shared_source_o4_g2",
}
FIXED_HORIZONS = (0.1, 0.5, 1.0, 2.0, 3.0, 6.32)
NATIVE_HORIZONS = (1.0, 3.0, 6.0, 6.5, 7.5, 10.0)
TRACE_FILES = (
    "segments.csv",
    "attempts.csv",
    "checkpoints.csv",
    "profile.csv",
    "range_trace.jsonl",
    "horner_stage_trace.jsonl",
    "remainder_ledger.jsonl",
    "owner_ledger.jsonl",
)


def horizon_label(value: float) -> str:
    return f"T{format(value, 'g').replace('.', 'p')}"


def compress_deterministic(path: Path) -> Path:
    target = path.with_name(path.name + ".gz")
    with path.open("rb") as source, target.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=9, mtime=0) as sink:
            while chunk := source.read(1024 * 1024):
                sink.write(chunk)
    path.unlink()
    return target


def command(output: Path, mode: str, schedule: str, horizon: float, wall_cap_s: float) -> list[str]:
    argv = [
        sys.executable,
        str(RUNNER),
        "--output-dir",
        str(output),
        "--tm-backend",
        "dense",
        "--device",
        "cpu",
        "--initialization-contract",
        "exact_decimal_contract",
        "--horizon",
        format(horizon, "g"),
        "--trace-flush-every",
        "0",
        "--wall-cap-s",
        format(wall_cap_s, "g"),
        "--reset-mode",
        MODES[mode],
        "--dense-range-method",
        "adaptive_subdivision",
        "--dense-range-trigger",
        "proactive_depth1_on_named_contexts",
        "--dense-range-max-depth",
        "1",
        "--dense-range-max-leaves",
        "4",
        "--dense-range-split-vars",
        "0,1",
        "--dense-range-contexts",
        "polynomial_truncation",
    ]
    if schedule == "fixed":
        argv.extend(("--fixed-step", "0.01"))
    return argv


def one_request(spec: tuple[Path, str, str, float, float]) -> dict[str, Any]:
    root, mode, schedule, horizon, wall_cap_s = spec
    output = root / schedule / mode / horizon_label(horizon)
    output.mkdir(parents=True, exist_ok=False)
    argv = command(output, mode, schedule, horizon, wall_cap_s)
    started = time.perf_counter()
    completed = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True)
    (output / "matrix_stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (output / "matrix_stderr.txt").write_text(completed.stderr, encoding="utf-8")
    summary_path = output / "summary.json"
    if completed.returncode not in (0, 1) or not summary_path.is_file():
        raise RuntimeError(
            f"request failed without a scientific summary: {schedule}/{mode}/{horizon}; "
            f"returncode={completed.returncode}; stderr={completed.stderr[-1000:]}"
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    compressed = []
    for name in TRACE_FILES:
        path = output / name
        if path.is_file():
            compressed.append(str(compress_deterministic(path).relative_to(root)))
    return {
        "mode": mode,
        "reset_mode": MODES[mode],
        "schedule": schedule,
        "requested_horizon": horizon,
        "returncode": completed.returncode,
        "status": summary["status"],
        "completed_horizon": summary["completed_horizon"],
        "completed_requested_horizon": summary["completed_requested_horizon"],
        "accepted_steps": summary["accepted_steps"],
        "rejected_attempts": summary["rejected_attempts"],
        "runtime_s": summary["runtime_s"],
        "matrix_elapsed_s": time.perf_counter() - started,
        "peak_rss_bytes": summary["peak_rss_bytes"],
        "endpoint_x_width": (summary.get("raw_endpoint") or {}).get("x_width"),
        "endpoint_y_width": (summary.get("raw_endpoint") or {}).get("y_width"),
        "segment_x_width": (summary.get("last_segment") or {}).get("x_width"),
        "segment_y_width": (summary.get("last_segment") or {}).get("y_width"),
        "fallback_count": summary["fallback_count"],
        "active_variables": _last_csv_value(output / "segments.csv.gz", "next_boundary_active_variables"),
        "term_count": _last_csv_value(output / "segments.csv.gz", "next_boundary_term_count"),
        "message": summary["message"],
        "relative_output": str(output.relative_to(root)),
        "compressed_trace_files": compressed,
    }


def _last_csv_value(path: Path, field: str) -> str:
    if not path.is_file():
        return ""
    with gzip.open(path, "rt", newline="", encoding="utf-8") as handle:
        rows = csv.DictReader(handle)
        value = ""
        for row in rows:
            if row.get("status") == "accepted":
                value = row.get(field, "")
    return value


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("matrix produced no rows")
    keys = [key for key in rows[0] if key != "compressed_trace_files"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in keys} for row in rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--wall-cap-s", type=float, default=1800.0)
    parser.add_argument("--schedules", choices=("fixed", "native", "all"), default="all")
    args = parser.parse_args()
    root = args.output_root.resolve()
    if root.exists():
        raise FileExistsError(f"refusing existing matrix root: {root}")
    root.mkdir(parents=True)
    schedules = ("fixed", "native") if args.schedules == "all" else (args.schedules,)
    specs = [
        (root, mode, schedule, horizon, float(args.wall_cap_s))
        for schedule in schedules
        for mode in MODES
        for horizon in (FIXED_HORIZONS if schedule == "fixed" else NATIVE_HORIZONS)
    ]
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(args.jobs))) as pool:
        future_map = {pool.submit(one_request, spec): spec for spec in specs}
        for future in concurrent.futures.as_completed(future_map):
            row = future.result()
            rows.append(row)
            print(json.dumps({key: row[key] for key in (
                "schedule", "mode", "requested_horizon", "status", "completed_horizon"
            )}, sort_keys=True), flush=True)
    rows.sort(key=lambda row: (row["schedule"], row["mode"], row["requested_horizon"]))
    write_csv(root / "requests.csv", rows)
    result = {
        "schema": "vdp_g2_fresh_request_matrix_v1",
        "authoritative_lane": "CPU_float64_B1",
        "initialization": "exact_decimal_contract",
        "fixed_h": 0.01,
        "modes": MODES,
        "fixed_horizons": list(FIXED_HORIZONS),
        "native_horizons": list(NATIVE_HORIZONS),
        "request_count": len(rows),
        "matrix_concurrency": max(1, int(args.jobs)),
        "matrix_runtime_not_performance_evidence": True,
        "wall_runtime_s": time.perf_counter() - started,
        "rows": rows,
    }
    (root / "matrix.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "PASS", "requests": len(rows)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
