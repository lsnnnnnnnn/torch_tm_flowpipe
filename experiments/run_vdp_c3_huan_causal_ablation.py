#!/usr/bin/env python3
"""Run clean-source Huan fixed-step VDP queue ablations and causal ledgers."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Iterable


HUAN_BASE = "743f6205e6408072193ad76e940e7f15030e8d3c"
CHECKPOINTS = (1, 10, 50, 100, 200, 300, 400, 500, 600, 632)


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout.strip()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl_gz(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def _hash_tensors(torch: Any, tensors: Iterable[Any]) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        value = tensor.detach().cpu().contiguous()
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _record_hash(torch: Any, records: list[Any]) -> str:
    tensors = []
    for record in records:
        tensors.extend(
            (
                record.pre_coeffs,
                record.pre_rem,
                record.tmv_coeffs,
                record.tmv_rem,
                record.active_mask,
            )
        )
    return _hash_tensors(torch, tensors)


def _box_channels(
    torch: Any,
    record: Any,
    mode: str,
    tables: Any,
    step: Any,
    iv: Any,
    poly: Any,
) -> dict[str, Any]:
    device = tables.spatial_index.device
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
        endpoint_roundoff = torch.zeros_like(remainder)
        endpoint_remainder = remainder
    endpoint = iv.add(
        poly.range_normal_spatial(endpoint_coeffs, tables), endpoint_remainder
    )
    return {
        "endpoint": endpoint.detach().cpu().tolist()[0],
        "endpoint_width": iv.width(endpoint).detach().cpu().tolist()[0],
        "segment_tube": tube.detach().cpu().tolist()[0],
        "segment_tube_width": iv.width(tube).detach().cpu().tolist()[0],
        "strict_endpoint_roundoff": endpoint_roundoff.detach().cpu().tolist()[0],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    engine = args.engine_root.resolve()
    output = args.output_root.resolve()
    head = _git(engine, "rev-parse", "HEAD")
    status = _git(engine, "status", "--porcelain")
    if status:
        raise RuntimeError("Huan scientific source worktree is dirty")
    if head != args.expected_engine_head:
        raise RuntimeError(
            f"Huan engine mismatch: expected={args.expected_engine_head}, actual={head}"
        )
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", HUAN_BASE, head], cwd=engine
    ).returncode != 0:
        raise RuntimeError(f"instrumented engine does not descend from {HUAN_BASE}")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output root must be new or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(engine / "src"))
    torch = importlib.import_module("torch")
    iv = importlib.import_module("flowstar_gpu.interval")
    poly = importlib.import_module("flowstar_gpu.polynomial")
    config = importlib.import_module("flowstar_gpu.config")
    flowpipe = importlib.import_module("flowstar_gpu.flowpipe")
    monomials = importlib.import_module("flowstar_gpu.monomials")

    rhs = ["y", "y - x - x^2*y"]
    names = ["x", "y"]
    boxes = torch.tensor(
        [[[1.1, 1.4], [2.35, 2.45]]],
        dtype=torch.float64,
        device=args.device,
    )
    tables = monomials.build_tables(2, 4).to(args.device)
    step = poly.build_step_tables(tables, 0.01)
    full_checkpoints = tuple(
        sorted(set(args.full_checkpoints) | set(CHECKPOINTS))
    )
    rows: list[dict[str, Any]] = []

    for queue in args.queues:
        for mode in args.modes:
            causal: list[dict[str, Any]] = []
            refinement: list[dict[str, Any]] = []
            callback = args.callback == "on"
            settings = config.Settings(
                step=0.01,
                order=4,
                cutoff=1e-10,
                remainder_estimation=1e-4,
                sr_queue=queue,
                mode=mode,
                device=args.device,
                max_refinement_steps=490,
                stop_ratio=0.99,
                refinement_callback=refinement.append if callback else None,
                causal_callback=causal.append if callback else None,
                causal_full_checkpoints=full_checkpoints if callback else (),
            )
            if args.device == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            result = flowpipe.reach(
                rhs, names, boxes, args.horizon, settings, record_tms=True
            )
            if args.device == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            records = result.records
            checkpoints = {
                str(step_index): _box_channels(
                    torch, records[step_index - 1], mode, tables, step, iv, poly
                )
                for step_index in CHECKPOINTS
                if step_index <= len(records)
            }
            row = {
                "tool": "huan_flowstar_gpu",
                "base_source_sha": HUAN_BASE,
                "instrumented_source_sha": head,
                "engine_clean": True,
                "mode": mode,
                "symbolic_queue_capacity": queue,
                "callback": args.callback,
                "device": args.device,
                "requested_horizon": args.horizon,
                "completed_horizon": len(records) * 0.01,
                "accepted_steps": int(result.steps_completed[0]),
                "status_code": int(result.status[0]),
                "completed_requested_horizon": int(result.status[0]) == flowpipe.DONE,
                "rejected_attempts": sum(
                    row.get("event") == "initial_self_map"
                    and not row.get("initial_self_map_ok", False)
                    for row in refinement
                ),
                "runtime_s": elapsed,
                "published_snapshot_sha256": _record_hash(torch, records),
                "checkpoint_channels": checkpoints,
                "final_channels": (
                    _box_channels(
                        torch, records[-1], mode, tables, step, iv, poly
                    )
                    if records
                    else None
                ),
                "settings": {
                    "rhs": rhs,
                    "initial_set": [[1.1, 1.4], [2.35, 2.45]],
                    "complete_total_degree_order": 4,
                    "fixed_step": 0.01,
                    "ordinary_remainder": [-1e-4, 1e-4],
                    "cutoff": 1e-10,
                    "validation_epsilon": 1e-12,
                    "symbolic_queue_capacity": queue,
                },
                "causal_trace_rows": len(causal),
                "refinement_trace_rows": len(refinement),
            }
            variant = output / f"sr{queue}" / mode
            _write_json(variant / "summary.json", row)
            if callback:
                _write_jsonl_gz(variant / "causal_ledger.jsonl.gz", causal)
                _write_jsonl_gz(
                    variant / "refinement_ledger.jsonl.gz", refinement
                )
            rows.append(row)

    payload = {
        "schema": "torch_tm_flowpipe.vdp_c3_huan_causal_ablation/1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "cwd": os.getcwd(),
        "base_source_sha": HUAN_BASE,
        "instrumented_source_sha": head,
        "instrumentation_diff_sha256": hashlib.sha256(
            subprocess.run(
                ["git", "diff", "--binary", f"{HUAN_BASE}..{head}"],
                cwd=engine,
                check=True,
                stdout=subprocess.PIPE,
            ).stdout
        ).hexdigest(),
        "engine_clean": True,
        "runs": rows,
    }
    _write_json(output / "run_index.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-root", type=Path, required=True)
    parser.add_argument("--expected-engine-head", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--horizon", type=float, default=6.32)
    parser.add_argument("--queues", type=int, nargs="+", default=[100, 0])
    parser.add_argument("--modes", nargs="+", choices=("parity", "strict"), default=["parity", "strict"])
    parser.add_argument("--callback", choices=("on", "off"), default="on")
    parser.add_argument(
        "--full-checkpoints",
        type=int,
        nargs="*",
        default=[2, 3, 4, 5, 6, 7, 8, 9, 11, 20, 99, 101, 199, 201],
    )
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    result = run(parsed)
    print(
        json.dumps(
            {
                "run_count": len(result["runs"]),
                "instrumented_source_sha": result["instrumented_source_sha"],
            },
            sort_keys=True,
        )
    )
