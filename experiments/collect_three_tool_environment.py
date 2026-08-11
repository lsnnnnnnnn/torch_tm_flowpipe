#!/usr/bin/env python3
"""Collect arithmetic-relevant environment and executable provenance."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from typing import Any, Sequence

import torch


ARITHMETIC_ENVIRONMENT_VARIABLES = (
    "CUDA_VISIBLE_DEVICES",
    "CUBLAS_WORKSPACE_CONFIG",
    "NVIDIA_TF32_OVERRIDE",
    "JAX_ENABLE_X64",
    "JAX_DEFAULT_MATMUL_PRECISION",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "PYTORCH_CUDA_ALLOC_CONF",
    "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
)


def _sha(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _command(command: Sequence[str], *, cwd: Path | None = None) -> dict[str, Any]:
    completed = subprocess.run(
        list(command), cwd=cwd, text=True, capture_output=True, check=False
    )
    return {
        "command": list(command),
        "exit_code": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def _git(path: Path) -> dict[str, Any]:
    head = _command(("git", "rev-parse", "HEAD"), cwd=path)
    submodules = _command(("git", "submodule", "status", "--recursive"), cwd=path)
    status = _command(("git", "status", "--short"), cwd=path)
    return {
        "path": str(path),
        "head": head["stdout"] if head["exit_code"] == 0 else None,
        "worktree_status": status["stdout"] if status["exit_code"] == 0 else None,
        "submodules": (
            submodules["stdout"].splitlines()
            if submodules["exit_code"] == 0
            else {"error": submodules["stderr"]}
        ),
    }


def _cpu_model() -> str | None:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name") and ":" in line:
                return line.split(":", 1)[1].strip()
    return platform.processor() or None


def _jax(python: Path) -> dict[str, Any]:
    program = (
        "import json;"
        "\ntry:"
        "\n import jax,jaxlib;"
        "\n print(json.dumps({'available':True,'jax':jax.__version__,"
        "'jaxlib':jaxlib.__version__,'x64':bool(jax.config.x64_enabled),"
        "'devices':[str(v) for v in jax.devices()]}))"
        "\nexcept Exception as e:"
        "\n print(json.dumps({'available':False,'error':type(e).__name__+': '+str(e)}))"
    )
    env = dict(os.environ)
    env["JAX_ENABLE_X64"] = "true"
    completed = subprocess.run(
        [str(python), "-c", program],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError:
        value = {
            "available": False,
            "error": completed.stderr.strip() or completed.stdout.strip(),
        }
    value["probe_exit_code"] = completed.returncode
    value["probe_environment"] = {"JAX_ENABLE_X64": "true"}
    return value


def collect(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    repository = Path(__file__).resolve().parents[1]
    flowstar = args.flowstar_root.resolve()
    diffreach = args.diffreach_root.resolve()
    diffreach_python = args.diffreach_python.resolve()
    flowstar_binary = args.flowstar_binary.resolve()
    compiler = Path(shutil.which(args.compiler) or args.compiler).resolve()
    python = Path(sys.executable).resolve()
    gpu = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            gpu.append(
                {
                    "index": index,
                    "name": properties.name,
                    "capability": list(torch.cuda.get_device_capability(index)),
                    "total_memory_bytes": properties.total_memory,
                }
            )
    report = {
        "schema": "three_tool_arithmetic_environment_v1",
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": str(python),
            "executable_sha256": _sha(python),
        },
        "torch": {
            "version": torch.__version__,
            "cuda_build": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "cudnn_version": torch.backends.cudnn.version(),
            "default_dtype": str(torch.get_default_dtype()),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
        },
        "system": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_model": _cpu_model(),
            "cpu_count": os.cpu_count(),
            "libc": list(platform.libc_ver()),
            "ldd": _command(("ldd", "--version")),
            "gpu": gpu,
        },
        "compiler": {
            "path": str(compiler),
            "sha256": _sha(compiler),
            "version": _command((str(compiler), "--version")),
            "full_version": _command(
                (str(compiler), "-dumpfullversion", "-dumpversion")
            ),
        },
        "native_libraries": {
            "mpfr": _command(("pkg-config", "--modversion", "mpfr")),
            "gmp": _command(("pkg-config", "--modversion", "gmp")),
        },
        "jax": _jax(diffreach_python),
        "arithmetic_environment_variables": {
            name: os.environ.get(name) for name in ARITHMETIC_ENVIRONMENT_VARIABLES
        },
        "executables": {
            "flowstar_vanderpol": {
                "path": str(flowstar_binary),
                "sha256": _sha(flowstar_binary),
            },
            "diffreach_python": {
                "path": str(diffreach_python),
                "sha256": _sha(diffreach_python),
            },
        },
        "repositories": {
            "torch_tm_flowpipe": _git(repository),
            "flowstar": _git(flowstar),
            "diffreach": _git(diffreach),
        },
    }
    (output / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--flowstar-root", type=Path, required=True)
    parser.add_argument("--flowstar-binary", type=Path, required=True)
    parser.add_argument("--diffreach-root", type=Path, required=True)
    parser.add_argument("--diffreach-python", type=Path, required=True)
    parser.add_argument("--compiler", default="g++")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    print(json.dumps(collect(parse_args(argv)), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
