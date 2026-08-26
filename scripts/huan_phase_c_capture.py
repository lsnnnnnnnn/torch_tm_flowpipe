#!/usr/bin/env python3
"""Capture clean-source environment and build closure for flowstar-gpu."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
from typing import Sequence


def _run(command: Sequence[str], cwd: Path, env: dict[str, str]) -> dict[str, object]:
    result = subprocess.run(
        list(command), cwd=cwd, env=env, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    return {"command": list(command), "returncode": result.returncode, "output": result.stdout}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _render(records: list[dict[str, object]]) -> str:
    chunks: list[str] = []
    for record in records:
        chunks.append(
            "$ " + " ".join(record["command"]) + "\n"
            + str(record["output"])
            + f"[returncode={record['returncode']}]\n"
        )
    return "\n".join(chunks)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cuda-home", type=Path, required=True)
    parser.add_argument("--compiler-root", type=Path, required=True)
    args = parser.parse_args()

    engine = args.engine_root.resolve()
    output = args.output_root.resolve()
    py = engine / ".venv" / "bin" / "python"
    uv = engine / ".venv" / "bin" / "uv"
    cc = args.compiler_root / "bin" / "x86_64-conda-linux-gnu-gcc"
    cxx = args.compiler_root / "bin" / "x86_64-conda-linux-gnu-g++"
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "0",
            "CUDA_HOME": str(args.cuda_home),
            "CC": str(cc),
            "CXX": str(cxx),
            "CUDAHOSTCXX": str(cxx),
            "PATH": f"{args.compiler_root / 'bin'}:{args.cuda_home / 'bin'}:{env['PATH']}",
        }
    )

    environment_commands = [
        ["git", "remote", "-v"],
        ["git", "status", "--short", "--branch"],
        ["git", "rev-parse", "HEAD"],
        ["git", "submodule", "status", "--recursive"],
        ["git", "tag", "--list"],
        ["git", "diff", "--binary"],
        ["uname", "-a"],
        ["lscpu"],
        ["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version,compute_cap", "--format=csv,noheader"],
        [str(args.cuda_home / "bin" / "nvcc"), "--version"],
        [str(cxx), "--version"],
        [str(py), "--version"],
        [str(uv), "--version"],
        [str(py), "-m", "pip", "freeze"],
    ]
    env_records = [_run(command, engine, env) for command in environment_commands]
    lock = {
        name: _sha256(engine / name)
        for name in ("pyproject.toml", "uv.lock")
        if (engine / name).is_file()
    }
    env_header = {
        "schema": "torch_tm_flowpipe.huan_environment/1",
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "engine_root": str(engine),
        "platform": platform.platform(),
        "lock_sha256": lock,
        "build_environment": {key: env[key] for key in ("CUDA_VISIBLE_DEVICES", "CUDA_HOME", "CC", "CXX", "CUDAHOSTCXX")},
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "environment.txt").write_text(
        json.dumps(env_header, indent=2, sort_keys=True) + "\n" + _render(env_records),
        encoding="utf-8",
    )

    smoke = (
        "import json, torch; "
        "from flowstar_gpu import cuda_kernels as ck; "
        "print(json.dumps({'torch':torch.__version__,'torch_cuda':torch.version.cuda,"
        "'cuda_available':torch.cuda.is_available(),'device':torch.cuda.get_device_name(0),"
        "'compute_capability':torch.cuda.get_device_capability(0),'kernel_available':ck.available()}, sort_keys=True))"
    )
    build_commands = [
        [str(uv), "sync", "--active", "--frozen"],
        [str(py), "-c", smoke],
        [str(uv), "pip", "install", "ninja==1.13.0", "pyyaml==6.0.3"],
        [str(py), "-c", smoke],
    ]
    build_records = [_run(command, engine, env) for command in build_commands]
    build_header = {
        "schema": "torch_tm_flowpipe.huan_build/1",
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "engine_root": str(engine),
        "source_modified_by_audit": False,
        "project_lock_complete_for_tests_and_cuda_jit": False,
        "audit_overlay": ["ninja==1.13.0", "pyyaml==6.0.3"],
        "interpretation": "frozen sync removes both required packages and leaves kernels unavailable; the pinned audit overlay restores the shipped JIT without changing source or uv.lock",
        "returncodes": [record["returncode"] for record in build_records],
    }
    (output / "build.log").write_text(
        json.dumps(build_header, indent=2, sort_keys=True) + "\n" + _render(build_records),
        encoding="utf-8",
    )
    return 0 if all(record["returncode"] == 0 for record in env_records + build_records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
