#!/usr/bin/env python3
"""Build and run the single pre-registered pinned Flow* Brusselator lane."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / "experiments/flowstar_probe/flowstar_brusselator_second_system.cpp"
CONTRACT = ROOT / "SECOND_SYSTEM_CONTRACT.md"
FLOWSTAR_COMMIT = "b85a3211748cb77b736fe4ad42ee02d8d2b81148"
BENCHMARK_RELATIVE = Path("benchmarks/continuous/brusselator/brusselator.cpp")
BENCHMARK_SHA256 = "b982f7c6f737e4b5e070942dc5fe01fa9d60e17a419a146d42444c71b5bf4f3b"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _run_logged(command: Sequence[str], cwd: Path, output: Path, label: str) -> dict[str, Any]:
    started = time.perf_counter()
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)
    wall = time.perf_counter() - started
    stdout = output / f"{label}.stdout.log"
    stderr = output / f"{label}.stderr.log"
    stdout.write_text(result.stdout, encoding="utf-8")
    stderr.write_text(result.stderr, encoding="utf-8")
    return {
        "command": list(command),
        "cwd": str(cwd),
        "exit_code": result.returncode,
        "wall_seconds": wall,
        "stdout_sha256": _sha256(stdout),
        "stderr_sha256": _sha256(stderr),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    source = args.source.resolve()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=source, check=True, capture_output=True, text=True
    ).stdout.strip() != FLOWSTAR_COMMIT:
        raise RuntimeError("Flow* source checkout is not the pre-registered commit")
    if _sha256(source / BENCHMARK_RELATIVE) != BENCHMARK_SHA256:
        raise RuntimeError("Flow* Brusselator benchmark source hash mismatch")
    compiler = shutil.which("g++-15")
    if compiler is None:
        raise FileNotFoundError("the frozen Flow* compiler g++-15 is unavailable")
    compiler = str(Path(compiler).resolve())
    compiler_version = subprocess.run(
        [compiler, "--version"], check=True, capture_output=True, text=True
    ).stdout.splitlines()[0]
    command = {
        "source": str(source),
        "source_commit": FLOWSTAR_COMMIT,
        "benchmark_relative": BENCHMARK_RELATIVE.as_posix(),
        "benchmark_sha256": BENCHMARK_SHA256,
        "driver_sha256": _sha256(DRIVER),
        "contract_sha256": _sha256(CONTRACT),
        "compiler": compiler,
        "compiler_sha256": _sha256(Path(compiler)),
        "compiler_version": compiler_version,
        "compatibility_flags": ["-fpermissive"],
    }
    _write_json(output / "command.json", command)

    with tempfile.TemporaryDirectory(prefix="flowstar-brusselator-second-") as temporary:
        clone = Path(temporary) / "repo"
        cloned = _run_logged(
            ["git", "clone", "--quiet", "--no-hardlinks", str(source), str(clone)],
            Path(temporary),
            output,
            "clone",
        )
        if cloned["exit_code"] != 0:
            raise RuntimeError("Flow* disposable clone failed")
        checked = _run_logged(
            ["git", "checkout", "--quiet", FLOWSTAR_COMMIT], clone, output, "checkout"
        )
        if checked["exit_code"] != 0:
            raise RuntimeError("Flow* disposable checkout failed")
        toolbox = clone / "flowstar-toolbox"
        built = _run_logged(
            ["make", "-j1", f"CXX={compiler} -fpermissive"],
            toolbox,
            output,
            "build_toolbox",
        )
        if built["exit_code"] != 0:
            raise RuntimeError("Flow* toolbox build failed")
        binary = Path(temporary) / "flowstar_brusselator_second_system"
        compile_command = [
            compiler,
            "-fpermissive",
            "-O3",
            "-std=c++11",
            "-I",
            str(toolbox),
            str(DRIVER),
            "-L",
            str(toolbox),
            "-L",
            "/usr/local/lib",
            "-o",
            str(binary),
            "-lflowstar",
            "-lmpfr",
            "-lgmp",
            "-lgsl",
            "-lgslcblas",
            "-lm",
            "-lglpk",
        ]
        compiled = _run_logged(compile_command, Path(temporary), output, "build_driver")
        if compiled["exit_code"] != 0 or not binary.is_file():
            raise RuntimeError("Flow* extraction driver build failed")
        segments = output / "segments.csv"
        native_summary = output / "native_summary.json"
        executed = _run_logged(
            [str(binary), str(segments), str(native_summary)],
            output,
            output,
            "run",
        )
        if executed["exit_code"] not in {0, 1} or not segments.is_file() or not native_summary.is_file():
            raise RuntimeError("Flow* Brusselator numerical run failed structurally")
        tracked_status = subprocess.run(
            ["git", "status", "--short", "--untracked-files=no"],
            cwd=clone,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if tracked_status:
            raise RuntimeError("Flow* build modified tracked source")

    summary = json.loads((output / "native_summary.json").read_text(encoding="utf-8"))
    summary.update(
        {
            "lane": "flowstar",
            "source_commit": FLOWSTAR_COMMIT,
            "benchmark_sha256": BENCHMARK_SHA256,
            "driver_sha256": command["driver_sha256"],
            "contract_sha256": command["contract_sha256"],
            "compiler": command["compiler_version"],
            "build": {"clone": cloned, "checkout": checked, "toolbox": built, "driver": compiled},
            "process_exit_code": executed["exit_code"],
            "process_wall_seconds": executed["wall_seconds"],
            "segments_sha256": _sha256(output / "segments.csv"),
            "source_tree_tracked_changes_after_build": "",
            "endpoint_semantics": "Flowpipe::intEvalNormal(step_end_exp_table)",
            "tube_semantics": "Flowpipe::intEvalNormal(step_exp_table)",
        }
    )
    _write_json(output / "summary.json", summary)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        result = run(parse_args(argv))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0 if result["completed_requested_horizon"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
