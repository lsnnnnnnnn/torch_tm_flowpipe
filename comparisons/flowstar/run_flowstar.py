"""Runners for Flow* comparison backends.

Current ``chenxin415/flowstar`` exposes the toolbox as a C++ static library.
The preferred runner therefore compiles a generated C++ benchmark against
``flowstar-toolbox/libflowstar.a`` and executes the resulting binary.  A legacy
stdin-based executable runner is retained for older Flow* parser builds.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class FlowstarRunResult:
    status: str
    runtime_s: float
    compile_s: float | str = ""
    run_s: float | str = ""
    stdout_path: Path | None = None
    stderr_path: Path | None = None
    returncode: int | None = None
    executable: str | None = None
    message: str = ""
    artifact_paths: list[Path] = field(default_factory=list)


def find_flowstar_executable(explicit: str | None = None) -> str | None:
    """Find an older Flow* executable that reads a model from stdin."""
    candidates: list[str] = []
    if explicit:
        candidates.append(explicit)
    if os.environ.get("FLOWSTAR_BIN"):
        candidates.append(os.environ["FLOWSTAR_BIN"])
    candidates.extend(["flowstar", "flowstar_2.1.0", "flowstar2", "Flowstar", "Flow*"])
    for cand in candidates:
        if not cand:
            continue
        path = shutil.which(cand) if os.path.basename(cand) == cand else cand
        if path and Path(path).exists():
            return str(path)
    return None


def find_flowstar_root(explicit: str | None = None) -> Path | None:
    """Find the current chenxin415/flowstar repository root."""
    candidates: list[str] = []
    if explicit:
        candidates.append(explicit)
    if os.environ.get("FLOWSTAR_ROOT"):
        candidates.append(os.environ["FLOWSTAR_ROOT"])
    for cand in candidates:
        if not cand:
            continue
        root = Path(cand).expanduser().resolve()
        if (root / "flowstar-toolbox" / "Continuous.h").exists():
            return root
        if root.name == "flowstar-toolbox" and (root / "Continuous.h").exists():
            return root.parent
    return None


def _write_completed_process(proc: subprocess.CompletedProcess[str], stdout_path: Path, stderr_path: Path) -> None:
    stdout_path.write_text(proc.stdout or "", encoding="utf-8")
    stderr_path.write_text(proc.stderr or "", encoding="utf-8")


def _append_completed_process(proc: subprocess.CompletedProcess[str], stdout_path: Path, stderr_path: Path) -> None:
    with stdout_path.open("a", encoding="utf-8") as f:
        f.write(proc.stdout or "")
    with stderr_path.open("a", encoding="utf-8") as f:
        f.write(proc.stderr or "")


def ensure_toolbox_library(flowstar_root: Path, *, build: bool, timeout_s: float | None, stdout_path: Path, stderr_path: Path) -> bool:
    lib = flowstar_root / "flowstar-toolbox" / "libflowstar.a"
    if lib.exists():
        return True
    if not build:
        stderr_path.write_text(f"libflowstar.a not found at {lib}; run make in flowstar-toolbox or enable build.\n", encoding="utf-8")
        return False
    try:
        proc = subprocess.run(
            ["make", "-C", str(flowstar_root / "flowstar-toolbox")],
            text=True,
            capture_output=True,
            timeout=timeout_s,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        stderr_path.write_text(f"failed to build Flow* toolbox library: {exc}\n", encoding="utf-8")
        return False
    _append_completed_process(proc, stdout_path, stderr_path)
    return proc.returncode == 0 and lib.exists()


def run_flowstar_toolbox(
    cpp_path: str | Path,
    *,
    flowstar_root: str | Path | None = None,
    output_dir: str | Path | None = None,
    timeout_s: float | None = None,
    build_lib: bool = True,
    compiler: str | None = None,
    run_executable: bool = True,
) -> FlowstarRunResult:
    """Compile and run a generated C++ benchmark with the Flow* toolbox."""
    cpp = Path(cpp_path)
    root = find_flowstar_root(str(flowstar_root) if flowstar_root is not None else None)
    out_dir = (Path(output_dir) if output_dir is not None else cpp.parent).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = out_dir / f"{cpp.stem}.stdout.txt"
    stderr_path = out_dir / f"{cpp.stem}.stderr.txt"
    stdout_path.write_text("", encoding="utf-8")
    stderr_path.write_text("", encoding="utf-8")

    if root is None:
        return FlowstarRunResult(
            status="skipped",
            runtime_s=0.0,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            message="Flow* toolbox root not found; set FLOWSTAR_ROOT or pass --flowstar-root",
        )

    if not ensure_toolbox_library(root, build=build_lib, timeout_s=timeout_s, stdout_path=stdout_path, stderr_path=stderr_path):
        return FlowstarRunResult(
            status="failed",
            runtime_s=0.0,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            message="Flow* toolbox library build/check failed",
        )

    exe = out_dir / cpp.stem
    cxx = compiler or os.environ.get("CXX", "g++")
    compile_cmd = [
        cxx,
        "-O3",
        "-w",
        "-std=c++11",
        "-I",
        str(root / "flowstar-toolbox"),
        "-I",
        "/usr/local/include",
        str(cpp),
        "-L",
        str(root / "flowstar-toolbox"),
        "-L",
        "/usr/local/lib",
        "-o",
        str(exe),
        "-lflowstar",
        "-lmpfr",
        "-lgmp",
        "-lgsl",
        "-lgslcblas",
        "-lm",
        "-lglpk",
    ]

    start = time.perf_counter()
    try:
        comp = subprocess.run(compile_cmd, text=True, capture_output=True, timeout=timeout_s, check=False)
    except subprocess.TimeoutExpired as exc:
        runtime = time.perf_counter() - start
        stdout_path.write_text(exc.stdout or "", encoding="utf-8")
        stderr_path.write_text(exc.stderr or "", encoding="utf-8")
        return FlowstarRunResult(status="compile_timeout", runtime_s=runtime, compile_s=runtime, stdout_path=stdout_path, stderr_path=stderr_path, message="Flow* benchmark compilation timed out")
    except OSError as exc:
        runtime = time.perf_counter() - start
        return FlowstarRunResult(status="compile_failed", runtime_s=runtime, compile_s=runtime, stdout_path=stdout_path, stderr_path=stderr_path, message=str(exc))
    _append_completed_process(comp, stdout_path, stderr_path)
    compile_elapsed = time.perf_counter() - start
    if comp.returncode != 0:
        return FlowstarRunResult(status="compile_failed", runtime_s=compile_elapsed, compile_s=compile_elapsed, stdout_path=stdout_path, stderr_path=stderr_path, returncode=comp.returncode, message="Flow* benchmark compilation failed")

    if not run_executable:
        runtime = time.perf_counter() - start
        artifacts = sorted(p for p in out_dir.glob(f"{cpp.stem}*") if p.is_file())
        return FlowstarRunResult(status="built", runtime_s=runtime, compile_s=compile_elapsed, stdout_path=stdout_path, stderr_path=stderr_path, returncode=comp.returncode, executable=str(exe), artifact_paths=artifacts)

    try:
        proc = subprocess.run([str(exe)], text=True, capture_output=True, timeout=timeout_s, cwd=str(out_dir), check=False)
    except subprocess.TimeoutExpired as exc:
        runtime = time.perf_counter() - start
        stdout_path.write_text((stdout_path.read_text(encoding="utf-8") if stdout_path.exists() else "") + (exc.stdout or ""), encoding="utf-8")
        stderr_path.write_text((stderr_path.read_text(encoding="utf-8") if stderr_path.exists() else "") + (exc.stderr or ""), encoding="utf-8")
        return FlowstarRunResult(status="timeout", runtime_s=runtime, compile_s=compile_elapsed, run_s=runtime - compile_elapsed, stdout_path=stdout_path, stderr_path=stderr_path, executable=str(exe), message="Flow* benchmark timed out")
    except OSError as exc:
        runtime = time.perf_counter() - start
        return FlowstarRunResult(status="run_failed", runtime_s=runtime, compile_s=compile_elapsed, run_s=runtime - compile_elapsed, stdout_path=stdout_path, stderr_path=stderr_path, executable=str(exe), message=str(exc))

    runtime = time.perf_counter() - start
    run_elapsed = runtime - compile_elapsed
    _append_completed_process(proc, stdout_path, stderr_path)
    artifacts = sorted(p for p in out_dir.glob(f"{cpp.stem}*") if p.is_file())
    status = "completed" if proc.returncode == 0 else "run_failed"
    msg = "" if proc.returncode == 0 else f"Flow* benchmark returned code {proc.returncode}"
    return FlowstarRunResult(status=status, runtime_s=runtime, compile_s=compile_elapsed, run_s=run_elapsed, stdout_path=stdout_path, stderr_path=stderr_path, returncode=proc.returncode, executable=str(exe), message=msg, artifact_paths=artifacts)


def run_flowstar_legacy_model(
    model_path: str | Path,
    *,
    flowstar_bin: str | None = None,
    output_dir: str | Path | None = None,
    timeout_s: float | None = None,
) -> FlowstarRunResult:
    """Run an older Flow* executable that reads a ``.model`` file from stdin."""
    model = Path(model_path)
    exe = find_flowstar_executable(flowstar_bin)
    if exe is None:
        return FlowstarRunResult(status="skipped", runtime_s=0.0, message="Flow* executable not found")
    out_dir = Path(output_dir) if output_dir is not None else model.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = out_dir / f"{model.stem}.stdout.txt"
    stderr_path = out_dir / f"{model.stem}.stderr.txt"

    start = time.perf_counter()
    try:
        proc = subprocess.run(
            [exe],
            input=model.read_text(encoding="utf-8"),
            text=True,
            capture_output=True,
            timeout=timeout_s,
            cwd=str(out_dir),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        runtime = time.perf_counter() - start
        stdout_path.write_text(exc.stdout or "", encoding="utf-8")
        stderr_path.write_text(exc.stderr or "", encoding="utf-8")
        return FlowstarRunResult(status="timeout", runtime_s=runtime, run_s=runtime, stdout_path=stdout_path, stderr_path=stderr_path, executable=exe, message="Flow* timed out")
    except OSError as exc:
        runtime = time.perf_counter() - start
        return FlowstarRunResult(status="failed", runtime_s=runtime, run_s=runtime, executable=exe, message=str(exc))

    runtime = time.perf_counter() - start
    stdout_path.write_text(proc.stdout, encoding="utf-8")
    stderr_path.write_text(proc.stderr, encoding="utf-8")
    artifacts = sorted(p for p in out_dir.glob(f"{model.stem}*") if p.is_file())
    status = "completed" if proc.returncode == 0 else "failed"
    msg = "" if proc.returncode == 0 else f"Flow* returned code {proc.returncode}"
    return FlowstarRunResult(status=status, runtime_s=runtime, run_s=runtime, stdout_path=stdout_path, stderr_path=stderr_path, returncode=proc.returncode, executable=exe, message=msg, artifact_paths=artifacts)


# Backwards-compatible alias for earlier callers.
def run_flowstar(
    model_path: str | Path,
    *,
    flowstar_bin: str | None = None,
    output_dir: str | Path | None = None,
    timeout_s: float | None = None,
) -> FlowstarRunResult:
    return run_flowstar_legacy_model(model_path, flowstar_bin=flowstar_bin, output_dir=output_dir, timeout_s=timeout_s)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Flow* on a generated input file.")
    parser.add_argument("input")
    parser.add_argument("--target", choices=["toolbox_cpp", "legacy_model"], default="toolbox_cpp")
    parser.add_argument("--flowstar-root", default=None)
    parser.add_argument("--flowstar-bin", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--timeout-s", type=float, default=None)
    parser.add_argument("--no-build-flowstar-lib", action="store_true")
    args = parser.parse_args()
    if args.target == "toolbox_cpp":
        result = run_flowstar_toolbox(
            args.input,
            flowstar_root=args.flowstar_root,
            output_dir=args.output_dir,
            timeout_s=args.timeout_s,
            build_lib=not args.no_build_flowstar_lib,
        )
    else:
        result = run_flowstar_legacy_model(args.input, flowstar_bin=args.flowstar_bin, output_dir=args.output_dir, timeout_s=args.timeout_s)
    print(result)


if __name__ == "__main__":
    main()
