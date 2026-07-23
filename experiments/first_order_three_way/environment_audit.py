#!/usr/bin/env python3
"""Capture the mandated repository, software, and hardware audit."""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from common import output_dir_from_args, utc_timestamp, write_json

REPOSITORIES = {
    "torch_original_checkout": Path("/srv/local/shengenli/torch_tm_flowpipe"),
    "torch_benchmark_worktree": Path("/srv/local/shengenli/torch_tm_flowpipe_first_order_bench"),
    "diffreach": Path("/srv/local/shengenli/DiffReach"),
    "flowstar": Path("/srv/local/shengenli/flowstar"),
}


def run(command: Sequence[str], *, cwd: Path | None = None) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            list(command),
            cwd=str(cwd) if cwd else None,
            text=True,
            capture_output=True,
            check=False,
            env={**os.environ},
        )
        return {
            "command": list(command),
            "cwd": str(cwd) if cwd else "",
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    except OSError as exc:
        return {
            "command": list(command),
            "cwd": str(cwd) if cwd else "",
            "returncode": None,
            "stdout": "",
            "stderr": f"{type(exc).__name__}: {exc}",
        }


def repo_audit(path: Path) -> dict[str, Any]:
    commands = {
        "status_short": ["git", "status", "--short"],
        "branches": ["git", "branch", "-avv"],
        "head": ["git", "rev-parse", "HEAD"],
        "log": ["git", "log", "--oneline", "--decorate", "--graph", "--all", "-n", "100"],
        "remotes": ["git", "remote", "-v"],
        "untracked": ["git", "ls-files", "--others", "--exclude-standard"],
    }
    return {name: run(command, cwd=path) for name, command in commands.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output_dir = output_dir_from_args(args.output_dir)
    software_commands = {
        "conda_env_list": ["conda", "env", "list"],
        "py11_conda_list": ["conda", "list", "-n", "py11"],
        "diffreach312_conda_list": ["conda", "list", "-n", "diffreach312"],
        "py11_python": ["conda", "run", "-n", "py11", "python", "--version"],
        "py11_torch": [
            "conda", "run", "-n", "py11", "python", "-c",
            (
                "import torch; print(torch.__version__); "
                "print('cuda_available', torch.cuda.is_available()); "
                "print('cuda_version', torch.version.cuda); "
                "print('devices', [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"
            ),
        ],
        "diffreach312_python_jax": [
            "conda", "run", "-n", "diffreach312", "python", "-c",
            (
                "import sys,jax; jax.config.update('jax_enable_x64', True); "
                "print(sys.version); print(jax.__version__); "
                "print('x64',jax.config.x64_enabled); "
                "print('backend',jax.default_backend()); print('devices',jax.devices())"
            ),
        ],
        "nvidia_smi": ["nvidia-smi"],
        "gcc": ["gcc", "--version"],
        "gxx": ["g++", "--version"],
        "lscpu": ["lscpu"],
        "memory": ["free", "-h"],
        "uname": ["uname", "-a"],
        "os_release": ["cat", "/etc/os-release"],
        "tmux_version": ["tmux", "-V"],
        "flowstar_library": [
            "file", "/srv/local/shengenli/flowstar/flowstar-toolbox/libflowstar.a"
        ],
        "flowstar_dependency_cache": ["ldconfig", "-p"],
    }
    audit = {
        "timestamp": utc_timestamp(),
        "phase0_observation": (
            "Read-only repository and branch inspection was performed before the benchmark "
            "worktree was created. This persisted audit repeats those commands in the isolated "
            "worktree so the result directory is self-contained."
        ),
        "repositories": {name: repo_audit(path) for name, path in REPOSITORIES.items()},
        "software_hardware": {
            name: run(command) for name, command in software_commands.items()
        },
        "paths": {
            "flowstar_library_exists": Path(
                "/srv/local/shengenli/flowstar/flowstar-toolbox/libflowstar.a"
            ).exists(),
            "tmux": shutil.which("tmux"),
            "python_executable_for_audit": sys.executable,
        },
        "platform": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "machine": platform.machine(),
        },
        "environment_decisions": {
            "torch_and_plotting": (
                "py11 (Python 3.11, torch 2.5.1+cu121); the benchmark specification "
                "selects float64 CPU batch-1 even when CUDA devices are visible"
            ),
            "diffreach": "diffreach312 (Python 3.12, CPU JAX 0.10.2, x64 enabled)",
            "flowstar": "system GCC/G++ 15.2 and existing static toolbox library",
            "diffreach_declared_install": (
                "The declared editable install was attempted and rejected by pip because "
                "jax2onnx>=0.10.1 requires equinox>=0.13.1 while every available "
                "immrax[cuda] requires equinox~=0.12.2. The plant adapter therefore uses "
                "the repository's analytic source path with a minimal CPU JAX environment."
            ),
        },
    }
    write_json(output_dir / "environment.json", audit)
    text: list[str] = [
        f"First-order three-way environment audit\nTimestamp: {audit['timestamp']}\n",
        audit["phase0_observation"],
        "",
    ]
    for repo_name, entries in audit["repositories"].items():
        text.append(f"===== repository: {repo_name} =====")
        for label, result in entries.items():
            text.extend(
                [
                    f"--- {label}: {' '.join(result['command'])}",
                    result["stdout"].rstrip(),
                    result["stderr"].rstrip(),
                    f"[returncode={result['returncode']}]",
                ]
            )
    for label, result in audit["software_hardware"].items():
        text.extend(
            [
                f"===== {label}: {' '.join(result['command'])} =====",
                result["stdout"].rstrip(),
                result["stderr"].rstrip(),
                f"[returncode={result['returncode']}]",
            ]
        )
    (output_dir / "environment.txt").write_text("\n".join(text) + "\n", encoding="utf-8")
    print(output_dir)


if __name__ == "__main__":
    main()
