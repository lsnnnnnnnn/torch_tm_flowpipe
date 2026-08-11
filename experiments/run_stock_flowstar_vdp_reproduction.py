#!/usr/bin/env python3
"""Run a pinned stock Flow* VDP binary in an isolated artifact directory."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile
import time
from typing import Any, Sequence


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _plot_horizon(path: Path) -> float:
    highest = float("-inf")
    number = re.compile(r"^[ \t]*([-+0-9.eE]+)[ \t]+[-+0-9.eE]+")
    for line in path.read_text(encoding="utf-8").splitlines():
        match = number.match(line)
        if match:
            highest = max(highest, float(match.group(1)))
    if highest == float("-inf"):
        raise RuntimeError(f"no plot coordinates found in {path}")
    return highest


def _plot_segment_count(path: Path) -> int:
    """Count nonempty numeric blocks in an official GNUPLOT stream."""

    count = 0
    active = False
    in_data = False
    number = re.compile(r"^[ \t]*[-+0-9.]")
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("plot "):
            in_data = True
            continue
        if not in_data:
            continue
        if number.match(line):
            active = True
        elif active:
            count += 1
            active = False
    return count + int(active)


def _clean_build(
    source: Path,
    source_commit: str,
    output: Path,
) -> tuple[Path, dict[str, Any], tempfile.TemporaryDirectory[str]]:
    temporary = tempfile.TemporaryDirectory(prefix="flowstar-stock-vdp-")
    clone = Path(temporary.name) / "repo"
    cloned = subprocess.run(
        ["git", "clone", "--quiet", "--no-hardlinks", str(source), str(clone)],
        text=True,
        capture_output=True,
        check=False,
    )
    if cloned.returncode != 0:
        temporary.cleanup()
        raise RuntimeError("stock Flow* disposable clone failed")
    checked = subprocess.run(
        ["git", "checkout", "--quiet", source_commit],
        cwd=clone,
        text=True,
        capture_output=True,
        check=False,
    )
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=clone,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    if checked.returncode != 0 or head != source_commit:
        temporary.cleanup()
        raise RuntimeError("stock Flow* disposable clone commit mismatch")
    build_rows = []
    for name, directory in (
        ("toolbox", clone / "flowstar-toolbox"),
        ("official_vanderpol", clone / "benchmarks/continuous/vanderpol"),
    ):
        started = time.perf_counter()
        built = subprocess.run(
            ["make", "-j1"], cwd=directory, text=True, capture_output=True, check=False
        )
        elapsed = time.perf_counter() - started
        stdout = output / f"build_{name}.stdout.log"
        stderr = output / f"build_{name}.stderr.log"
        stdout.write_text(built.stdout, encoding="utf-8")
        stderr.write_text(built.stderr, encoding="utf-8")
        build_rows.append(
            {
                "target": name,
                "command": ["make", "-j1"],
                "exit_code": built.returncode,
                "wall_seconds": elapsed,
                "stdout_sha256": _sha(stdout),
                "stderr_sha256": _sha(stderr),
            }
        )
        if built.returncode != 0:
            temporary.cleanup()
            raise RuntimeError(f"stock Flow* clean build failed: {name}")
    binary = clone / "benchmarks/continuous/vanderpol/vanderpol"
    if not binary.is_file():
        temporary.cleanup()
        raise RuntimeError("stock Flow* clean build omitted official VDP binary")
    return binary, {"kind": "disposable_clean_clone", "steps": build_rows}, temporary


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    temporary: tempfile.TemporaryDirectory[str] | None = None
    if args.source is not None:
        binary, build, temporary = _clean_build(
            args.source.resolve(), args.source_commit, output
        )
    elif args.binary is not None:
        binary = args.binary.resolve()
        build = {"kind": "prebuilt_input", "steps": []}
    else:
        raise ValueError("one of --source or --binary is required")
    binary_sha256 = _sha(binary)
    try:
        started = time.perf_counter()
        completed = subprocess.run(
            [str(binary)], cwd=output, text=True, capture_output=True, check=False
        )
        elapsed = time.perf_counter() - started
    finally:
        if temporary is not None:
            temporary.cleanup()
    (output / "flowstar.stdout.log").write_text(completed.stdout)
    (output / "flowstar.stderr.log").write_text(completed.stderr)
    x_plot = output / "vanderpol_t_x.plt"
    y_plot = output / "vanderpol_t_y.plt"
    if completed.returncode != 0 or not x_plot.is_file() or not y_plot.is_file():
        raise RuntimeError("stock Flow* VDP reproduction failed")
    horizon = min(_plot_horizon(x_plot), _plot_horizon(y_plot))
    segment_counts = {
        "x": _plot_segment_count(x_plot),
        "y": _plot_segment_count(y_plot),
    }
    accepted_segments = min(segment_counts.values())
    core_match = re.search(
        r"(?:computation|Computation).*?([0-9]+(?:\.[0-9]+)?)\s*seconds",
        completed.stdout,
    )
    summary = {
        "schema": "stock_flowstar_vdp_reproduction_v1",
        "source_commit": args.source_commit,
        "binary_sha256": binary_sha256,
        "build": build,
        "binary_observer_status": "unmodified_stock",
        "model_sha256": args.model_sha256,
        "initial_set": [[1.1, 1.4], [2.35, 2.45]],
        "partition_count": 1,
        "representation": "complete_total_degree_O4",
        "validator": "native_flowstar_picard",
        "schedule": "native_adaptive",
        "horizon_requested": 10.0,
        "horizon_validated": horizon,
        "result_status": "completed" if horizon >= 10.0 else "partial",
        "accepted_segments": accepted_segments,
        "accepted_segments_by_plot": segment_counts,
        "reproduced_historical_290_segments": accepted_segments == 290,
        "segment_tube_available": True,
        "endpoint_available": False,
        "prefix_tube_available": True,
        "soundness_scope": "native build; ineligible after scalar-affine counterexample",
        "runtime_scope": {
            "process_wall_seconds": elapsed,
            "reported_core_seconds": None
            if core_match is None
            else float(core_match.group(1)),
        },
        "artifacts": {
            "x_plot_sha256": _sha(x_plot),
            "y_plot_sha256": _sha(y_plot),
            "stdout_sha256": _sha(output / "flowstar.stdout.log"),
            "stderr_sha256": _sha(output / "flowstar.stderr.log"),
        },
        "eligibility_status": "native_capability_only",
    }
    _write(output / "summary.json", summary)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--model-sha256", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    print(json.dumps(run(parse_args(argv)), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
