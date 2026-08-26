#!/usr/bin/env python3
"""Run the frozen fixed-step VDP contract in Huan parity/strict modes.

The native adaptive+symbolic lane is checked fail-closed.  The current engine
rejects that combination at configuration construction, so the runner records
an explicit contract-portability result instead of silently disabling the
symbolic queue or changing h_min/h_max.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


FIXED = {
    "step1": 0.01,
    "fixed_T1": 1.0,
    "fixed_T3": 3.0,
    "fixed_T6p32": 6.32,
}
ENGINE_HEAD = "b0ff55745d69205f3afb4dc8077b9ac1310bfff3"


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout.strip()


def _hash_tensor(torch: Any, tensor: Any) -> str:
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(row.get(key), sort_keys=True)
                    if isinstance(row.get(key), (dict, list))
                    else row.get(key, "")
                    for key in fields
                }
            )


def _box_channels(
    record: Any,
    mode: str,
    device: str,
    tables: Any,
    step: Any,
    iv: Any,
    poly: Any,
) -> dict[str, Any]:
    coeffs = record.pre_coeffs.to(device)
    remainder = record.pre_rem.to(device)
    tube = iv.add(poly.range_normal(coeffs, tables, step), remainder)
    if mode == "strict":
        endpoint_coeffs, endpoint_roundoff = poly.evaluate_time_end_with_roundoff(
            coeffs, tables, step
        )
        endpoint_remainder = iv.add(remainder, endpoint_roundoff)
    else:
        endpoint_coeffs = poly.evaluate_time_end(coeffs, tables, step)
        endpoint_roundoff = torch.zeros_like(remainder)  # set by run() module global
        endpoint_remainder = remainder
    endpoint = iv.add(
        poly.range_normal_spatial(endpoint_coeffs, tables), endpoint_remainder
    )
    return {
        "endpoint": endpoint.detach().cpu().tolist()[0],
        "segment_tube": tube.detach().cpu().tolist()[0],
        "endpoint_width": iv.width(endpoint).detach().cpu().tolist()[0],
        "segment_tube_width": iv.width(tube).detach().cpu().tolist()[0],
        "endpoint_roundoff": endpoint_roundoff.detach().cpu().tolist()[0],
    }


# Set after importing the selected source; used only in parity reporting above.
torch: Any = None


def run(engine_root: Path, gate_path: Path, output_root: Path, device: str) -> dict[str, Any]:
    global torch
    head = _git(engine_root, "rev-parse", "HEAD")
    dirty = bool(_git(engine_root, "status", "--porcelain"))
    gate = json.loads(gate_path.read_text())
    if head != ENGINE_HEAD or dirty:
        raise RuntimeError(f"Huan source mismatch: head={head}, dirty={dirty}")
    if not gate.get("phase_e_authorized") or gate.get("engine_head") != head:
        raise RuntimeError("Phase D gate does not authorize this Huan source")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"output root must be new or empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(engine_root / "src"))
    torch = importlib.import_module("torch")
    iv = importlib.import_module("flowstar_gpu.interval")
    poly = importlib.import_module("flowstar_gpu.polynomial")
    config = importlib.import_module("flowstar_gpu.config")
    flowpipe = importlib.import_module("flowstar_gpu.flowpipe")
    monomials = importlib.import_module("flowstar_gpu.monomials")

    rhs = ["y", "y - x - x^2*y"]
    names = ["x", "y"]
    boxes = torch.tensor(
        [[[1.1, 1.4], [2.35, 2.45]]], dtype=torch.float64, device=device
    )
    tables = monomials.build_tables(2, 4).to(device)
    step = poly.build_step_tables(tables, 0.01)
    runs: list[dict[str, Any]] = []
    all_trace: list[dict[str, Any]] = []

    for mode in ("parity", "strict"):
        for scenario, horizon in FIXED.items():
            trace: list[dict[str, Any]] = []
            settings = config.Settings(
                step=0.01,
                order=4,
                cutoff=1e-10,
                remainder_estimation=1e-4,
                sr_queue=100,
                mode=mode,
                device=device,
                max_refinement_steps=490,
                stop_ratio=0.99,
                refinement_callback=trace.append,
            )
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            started = time.perf_counter()
            result = flowpipe.reach(
                rhs, names, boxes, horizon, settings, record_tms=True
            )
            if device == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            records = result.records
            channels = (
                _box_channels(records[-1], mode, device, tables, step, iv, poly)
                if records
                else None
            )
            prefix_tube = None
            if records:
                tube_boxes = [
                    _box_channels(record, mode, device, tables, step, iv, poly)[
                        "segment_tube"
                    ]
                    for record in records
                ]
                prefix_tube = [
                    [
                        min(row[component][0] for row in tube_boxes),
                        max(row[component][1] for row in tube_boxes),
                    ]
                    for component in range(2)
                ]
            initial_rows = [row for row in trace if row["event"] == "initial_self_map"]
            final_rows = [row for row in trace if row["event"] == "final_remainder_owner"]
            failed_initial = next(
                (row for row in reversed(initial_rows) if not row["initial_self_map_ok"]),
                None,
            )
            row = {
                "tool": "huan_flowstar_gpu",
                "source_sha": head,
                "mode": mode,
                "scenario": scenario,
                "device": device,
                "requested_horizon": horizon,
                "completed_horizon": len(records) * 0.01,
                "accepted_steps": int(result.steps_completed[0]),
                "status_code": int(result.status[0]),
                "completed_requested_horizon": int(result.status[0]) == flowpipe.DONE,
                "rejected_attempts": sum(
                    not item["initial_self_map_ok"] for item in initial_rows
                ),
                "refinement_iterations": sum(item["iterations"] for item in final_rows),
                "runtime_s": elapsed,
                "peak_gpu_memory_bytes": (
                    int(torch.cuda.max_memory_allocated()) if device == "cuda" else None
                ),
                "settings": {
                    "rhs": rhs,
                    "initial_set": [[1.1, 1.4], [2.35, 2.45]],
                    "complete_total_degree_order": 4,
                    "fixed_step": 0.01,
                    "remainder_radius": 1e-4,
                    "cutoff": 1e-10,
                    "validation_epsilon": "engine-unmatched; loop threshold is 1e-12",
                    "symbolic_remainder_queue": 100,
                },
                "channels": channels,
                "prefix_tube": prefix_tube,
                "retained_candidate_polynomial_sha256": (
                    _hash_tensor(torch, records[0].pre_coeffs) if records else None
                ),
                "step1_remainder": (
                    records[0].pre_rem.tolist()[0] if records else None
                ),
                "first_self_map": initial_rows[0] if initial_rows else None,
                "failed_terminal_first_self_map": failed_initial,
                "roundoff_contribution_ledger": {
                    "strict_endpoint_roundoff": (
                        channels["endpoint_roundoff"] if channels else None
                    ),
                    "composition_and_preconditioning": (
                        "charged in ordinary remainder; category totals are not separately exposed"
                    ),
                    "parity_policy": (
                        "Flow* point-coefficient trust model" if mode == "parity" else None
                    ),
                },
            }
            runs.append(row)
            for attempt, item in enumerate(trace):
                all_trace.append(
                    {"mode": mode, "scenario": scenario, "attempt_order": attempt, **item}
                )
            _write_json(output_root / scenario / mode / "summary.json", row)

    native: dict[str, Any] = {}
    for mode in ("parity", "strict"):
        try:
            config.Settings(
                step=0.1,
                step_min=0.002,
                order=4,
                cutoff=1e-10,
                remainder_estimation=1e-4,
                sr_queue=100,
                mode=mode,
                device=device,
            )
        except NotImplementedError as exc:
            native[mode] = {
                "status": "NOT_RUN_CONTRACT_NOT_PORTABLE",
                "reason": str(exc),
                "requested_horizon": 10.0,
                "h_min": 0.002,
                "h_max": 0.1,
                "symbolic_remainder_queue": 100,
                "source_sha": head,
            }
        else:
            raise RuntimeError(
                "adaptive+symbolic contract unexpectedly became available; "
                "the preregistered runner requires explicit review before execution"
            )

    _write_csv(output_root / "fixed_horizon_matrix.csv", runs)
    _write_json(output_root / "native_terminal.json", native)
    with gzip.open(output_root / "refinement_ledgers.jsonl.gz", "wt", encoding="utf-8") as handle:
        for row in all_trace:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    step1_rows = [row for row in runs if row["scenario"] == "step1"]
    _write_csv(output_root / "step1_common_input.csv", step1_rows)
    result = {
        "schema": "torch_tm_flowpipe.huan_vdp_phase_e/1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "engine_head": head,
        "phase_d_gate_sha256": hashlib.sha256(gate_path.read_bytes()).hexdigest(),
        "fixed_runs": runs,
        "native": native,
        "primary_status": "HUAN_PROOF_CONTRACT_CLOSED__VDP_CONTRACT_NOT_PORTABLE",
        "throughput_phase": "NOT_RUN_THIS_ROUND_AFTER_PROOF_AND_VDP_SCOPE",
    }
    _write_json(output_root / "run_index.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-root", type=Path, required=True)
    parser.add_argument("--phase-d-gate", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    payload = run(
        args.engine_root.resolve(),
        args.phase_d_gate.resolve(),
        args.output_root.resolve(),
        args.device,
    )
    print(json.dumps({"primary_status": payload["primary_status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
