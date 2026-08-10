#!/usr/bin/env python3
"""Run the prerefactor object-eager fixed-support baseline matrix."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _slug(value: float) -> str:
    return str(value).replace(".", "p")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _lanes(include_t10: bool) -> list[tuple[str, str, int, float]]:
    lanes: list[tuple[str, str, int, float]] = []
    batches = (1, 8, 64, 256, 512)
    for horizon in (0.1, 1.0):
        for index, batch in enumerate(batches):
            order = (("cpu", "cpu"), ("cuda_v100", "cuda:0"))
            if index % 2:
                order = tuple(reversed(order))
            for device_group, device in order:
                lanes.append((device_group, device, batch, horizon))
    if include_t10:
        lanes.extend(
            (
                ("cuda_v100", "cuda:0", 64, 10.0),
                ("cpu", "cpu", 64, 10.0),
            )
        )
    return lanes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--matrix-label", default="trace")
    parser.add_argument("--warm-runs", type=int, default=5)
    parser.add_argument("--include-t10", action="store_true")
    parser.add_argument("--timeout", type=float, default=3600.0)
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    source_root = args.source_root.resolve()
    matrix_root = run_root / "raw/fixed_object_baseline" / args.matrix_label
    child_env = os.environ.copy()
    child_env["PYTHONPATH"] = os.pathsep.join(
        value
        for value in (
            str(source_root / "src"),
            child_env.get("PYTHONPATH", ""),
        )
        if value
    )
    rows: list[dict[str, Any]] = []
    for position, (device_group, device, batch, horizon) in enumerate(
        _lanes(args.include_t10)
    ):
        lane = f"{device_group}_b{batch}_t{_slug(horizon)}"
        output_dir = matrix_root / lane
        summary_path = output_dir / "summary.json"
        process_path = output_dir / "process.json"
        command = [
            sys.executable,
            "experiments/run_vdp_fixed_support.py",
            "--output-dir",
            str(output_dir),
            "--horizon",
            str(horizon),
            "--step-size",
            "0.01",
            "--batch",
            str(batch),
            "--device",
            device,
            "--warm-runs",
            str(args.warm_runs),
            "--symbolic-window-size",
            "1000",
        ]
        if not summary_path.is_file():
            started = time.perf_counter()
            process = subprocess.run(
                command,
                cwd=source_root,
                env=child_env,
                capture_output=True,
                text=True,
                timeout=args.timeout,
            )
            wall_s = time.perf_counter() - started
            _write_json(
                process_path,
                {
                    "command": command,
                    "cwd_role": "source_root",
                    "exit_status": process.returncode,
                    "wall_s": wall_s,
                    "stdout": process.stdout,
                    "stderr": process.stderr,
                    "matrix_position": position,
                },
            )
            if process.returncode != 0:
                raise RuntimeError(f"baseline lane failed: {lane}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        timing = summary["cold_warm_core_process_runtime"]
        rows.append(
            {
                "lane": lane,
                "matrix_position": position,
                "device_group": device_group,
                "device": device,
                "batch": batch,
                "requested_horizon": horizon,
                "steps": int(round(horizon / 0.01)),
                "mode": "object_eager_trace",
                "cold_s": timing["cold_s"],
                "warm_s": timing["warm_s"],
                "warm_min_s": min(timing["warm_s"]),
                "warm_median_s": sorted(timing["warm_s"])[len(timing["warm_s"]) // 2],
                "warm_max_s": max(timing["warm_s"]),
                "peak_memory_bytes": summary["peak_memory_bytes"],
                "host_synchronizations": summary["host_synchronizations"],
                "solver_device_transfers": summary["solver_device_transfers"],
                "validated_horizon": summary["validated_horizon"],
                "completed": summary["completion_status"] == "completed",
                "source_sha": summary["source_sha"],
                "artifact": summary_path.relative_to(run_root).as_posix(),
            }
        )
        print(json.dumps(rows[-1], sort_keys=True), flush=True)
    _write_json(
        matrix_root / "object_trace_matrix.json",
        {
            "schema": "fixed_support_object_baseline_matrix_v1",
            "warm_runs": args.warm_runs,
            "trace_mode": True,
            "summary_mode_available": False,
            "matrix_label": args.matrix_label,
            "rows": rows,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
