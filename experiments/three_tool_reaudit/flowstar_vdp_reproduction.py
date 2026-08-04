#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any


NUMBER_START = set("+-0123456789.")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_plot_blocks(path: Path) -> list[dict[str, float]]:
    blocks: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] = []
    in_data = False
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if line.startswith("plot "):
            in_data = True
            continue
        if not in_data:
            continue
        if line == "e":
            if current:
                blocks.append(current)
            break
        if not line:
            if current:
                blocks.append(current)
                current = []
            continue
        if line[0] not in NUMBER_START:
            if current:
                blocks.append(current)
                current = []
            continue
        fields = line.split()
        try:
            current.append((float(fields[0]), float(fields[1])))
        except (IndexError, ValueError):
            if current:
                blocks.append(current)
                current = []
    return [
        {
            "t_lower": min(point[0] for point in block),
            "t_upper": max(point[0] for point in block),
            "value_lower": min(point[1] for point in block),
            "value_upper": max(point[1] for point in block),
        }
        for block in blocks
    ]


def run(flowstar_root: Path, output: Path, repetitions: int) -> dict[str, Any]:
    benchmark = flowstar_root / "benchmarks" / "continuous" / "vanderpol"
    executable = benchmark / "vanderpol"
    output.mkdir(parents=True, exist_ok=True)
    make_started = time.perf_counter()
    make = subprocess.run(
        ["make"], cwd=benchmark, text=True, capture_output=True, check=False
    )
    make_elapsed = time.perf_counter() - make_started
    (output / "make.stdout.txt").write_text(make.stdout, encoding="utf-8")
    (output / "make.stderr.txt").write_text(make.stderr, encoding="utf-8")
    if make.returncode:
        raise RuntimeError(f"official Flowstar make failed: {make.returncode}")

    records: list[dict[str, Any]] = []
    for repetition in range(repetitions):
        started_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        started = time.perf_counter()
        process = subprocess.run(
            [str(executable)],
            cwd=benchmark,
            text=True,
            capture_output=True,
            check=False,
        )
        wall = time.perf_counter() - started
        name = "cold" if repetition == 0 else f"steady_{repetition}"
        stdout = output / f"{name}.stdout.txt"
        stderr = output / f"{name}.stderr.txt"
        stdout.write_text(process.stdout, encoding="utf-8")
        stderr.write_text(process.stderr, encoding="utf-8")
        plots: dict[str, Any] = {}
        for state in ("x", "y"):
            source = benchmark / f"vanderpol_t_{state}.plt"
            destination = output / f"{name}_t_{state}.plt"
            if source.is_file():
                shutil.copy2(source, destination)
                blocks = parse_plot_blocks(destination)
                plots[state] = {
                    "path": destination.name,
                    "sha256": sha256_file(destination),
                    "segments": len(blocks),
                    "completed_horizon": blocks[-1]["t_upper"] if blocks else 0.0,
                    "last_segment": blocks[-1] if blocks else None,
                    "full_tube": {
                        "value_lower": min(row["value_lower"] for row in blocks),
                        "value_upper": max(row["value_upper"] for row in blocks),
                    }
                    if blocks
                    else None,
                }
        completed = (
            process.returncode == 0
            and "terminated due to the large overestimation" not in process.stdout
            and plots
            and all(
                abs(record["completed_horizon"] - 10.0) <= 1e-12
                for record in plots.values()
            )
        )
        records.append(
            {
                "phase": name,
                "started_utc": started_utc,
                "command": [str(executable)],
                "cwd": str(benchmark.resolve()),
                "exit_code": process.returncode,
                "wall_time_s": wall,
                "stdout": stdout.name,
                "stderr": stderr.name,
                "status": "completed" if completed else "incomplete_unknown",
                "requested_horizon": 10.0,
                "completed_horizon": min(
                    (record["completed_horizon"] for record in plots.values()),
                    default=0.0,
                ),
                "accepted_segments": min(
                    (record["segments"] for record in plots.values()), default=0
                ),
                "rejected_attempts": {
                    "availability": "unavailable",
                    "reason": "official stock program does not print adaptive rejected attempts",
                },
                "validation_status": "completed" if completed else "incomplete_unknown",
                "plots": plots,
            }
        )
    evidence = {
        "schema_version": 1,
        "backend": "official-stock",
        "execution_route": "official_program",
        "flowstar_root": str(flowstar_root.resolve()),
        "repository_sha": subprocess.run(
            ["git", "-C", str(flowstar_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip(),
        "source": {
            "path": str((benchmark / "vanderpol.cpp").resolve()),
            "sha256": sha256_file(benchmark / "vanderpol.cpp"),
        },
        "binary": {
            "path": str(executable.resolve()),
            "sha256": sha256_file(executable),
        },
        "library": {
            "path": str((flowstar_root / "flowstar-toolbox" / "libflowstar.a").resolve()),
            "sha256": sha256_file(flowstar_root / "flowstar-toolbox" / "libflowstar.a"),
        },
        "make": {
            "command": ["make"],
            "cwd": str(benchmark.resolve()),
            "exit_code": make.returncode,
            "wall_time_s": make_elapsed,
            "stdout": "make.stdout.txt",
            "stderr": "make.stderr.txt",
        },
        "model_contract": {
            "ode": ["y", "(1 - x^2) * y - x", "1"],
            "state_order": ["x", "y", "t"],
            "initial_set": [[1.1, 1.4], [2.35, 2.45], [0.0, 0.0]],
            "requested_horizon": 10.0,
            "order": 4,
            "adaptive_step": {"minimum": 0.002, "maximum": 0.1},
            "cutoff": [-1.0e-10, 1.0e-10],
            "candidate_remainder": [-1.0e-4, 1.0e-4],
            "precision_bits": 53,
            "symbolic_remainder_queue": 100,
        },
        "runs": records,
        "passed": len(records) == repetitions
        and repetitions >= 4
        and all(record["status"] == "completed" for record in records),
        "limitations": [
            "GNUplot boxes are last-segment/full-tube data, not fixed-time endpoints",
            "official program does not expose rejected adaptive attempts",
        ],
    }
    (output / "evidence.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flowstar-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=4)
    arguments = parser.parse_args()
    evidence = run(arguments.flowstar_root, arguments.output, arguments.repetitions)
    print(json.dumps({"passed": evidence["passed"], "runs": len(evidence["runs"])}))
    return 0 if evidence["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
