#!/usr/bin/env python3
"""Evaluate only the preregistered 1/10/100/1000 compiled boundaries."""
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
BOUNDARIES = ((1, 600.0), (10, 180.0), (100, 90.0), (1000, 90.0))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-one", action="store_true", help="reuse the separately measured one-step boundary")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for boundary, timeout_s in BOUNDARIES:
        if boundary == 1 and args.skip_one:
            rows.append(
                {
                    "boundary": 1,
                    "status": "selected_separately_measured_boundary",
                    "timeout_s": timeout_s,
                }
            )
            continue
        lane = args.output_dir / f"boundary_{boundary}"
        cache = args.output_dir / f"cache_boundary_{boundary}"
        command = [
            sys.executable,
            "experiments/run_vdp_fixed_support_compiled.py",
            "--output-dir",
            str(lane),
            "--batch",
            "1",
            "--device",
            args.device,
            "--steps",
            str(boundary),
            "--boundary",
            str(boundary),
            "--warm-runs",
            "1",
            "--later-inputs",
            "5",
        ]
        env = os.environ.copy()
        env["TORCHINDUCTOR_CACHE_DIR"] = str(cache.resolve())
        started = time.perf_counter()
        timed_out = False
        try:
            process = subprocess.run(
                command,
                cwd=ROOT,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
            exit_status = process.returncode
            stdout, stderr = process.stdout, process.stderr
        except subprocess.TimeoutExpired as error:
            timed_out = True
            exit_status = 124
            stdout = (error.stdout or b"").decode() if isinstance(error.stdout, bytes) else (error.stdout or "")
            stderr = (error.stderr or b"").decode() if isinstance(error.stderr, bytes) else (error.stderr or "")
        wall_s = time.perf_counter() - started
        summary_path = lane / "summary.json"
        row = {
            "boundary": boundary,
            "device": args.device,
            "batch": 1,
            "logical_steps": boundary,
            "timeout_s": timeout_s,
            "wall_s": wall_s,
            "exit_status": exit_status,
            "timed_out": timed_out,
            "status": "completed" if exit_status == 0 else ("compile_timeout" if timed_out else "failed"),
            "summary_available": summary_path.is_file(),
            "command": command,
            "stdout": stdout,
            "stderr": stderr,
        }
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            row.update(
                {
                    "compile_execute_s": summary["compile_execute_s"],
                    "warm_s": summary["compiled_warm_s"],
                    "graph_break_count": summary["graph_break_count"],
                    "compiled_semantics": summary["compiled_semantics"],
                }
            )
        rows.append(row)
        _write_json(args.output_dir / f"boundary_{boundary}_process.json", row)
    artifact = {
        "schema": "compiled_boundary_feasibility_v1",
        "evaluated_boundaries": [boundary for boundary, _ in BOUNDARIES],
        "no_arbitrary_sweep": True,
        "selection": "one compiled logical step called from the Python summary loop",
        "rows": rows,
    }
    _write_json(args.output_dir / "summary.json", artifact)
    print(json.dumps(artifact, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
