#!/usr/bin/env python3
"""Capture the raw provenance required by the Xiangru q3 matched audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
from typing import Iterable, Sequence


REPOSITORIES = {
    "torch": Path("/srv/local/shengenli/torch_tm_flowpipe_xiangru_q3_audit"),
    "torch_user_worktree": Path("/srv/local/shengenli/torch_tm_flowpipe"),
    "xiangru": Path("/srv/local/shengenli/CROWN-Reach_Development"),
    "xiangru_reproduction": Path(
        "/srv/local/shengenli/CROWN-Reach_Development_native_27d2905"
    ),
    "flowstar": Path("/srv/local/shengenli/flowstar"),
}

TORCH_PYTHON = Path("/srv/local/shengenli/miniforge3/envs/py11/bin/python")
XIANGRU_PYTHON = Path("/srv/local/shengenli/native_envs/crownreach28/bin/python")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _run(argv: Sequence[str], *, cwd: Path | None = None) -> tuple[int, bytes, bytes]:
    try:
        completed = subprocess.run(
            list(argv), cwd=cwd, check=False, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, env=os.environ.copy(),
        )
    except FileNotFoundError as error:
        return 127, b"", f"{error}\n".encode()
    return completed.returncode, completed.stdout, completed.stderr


def _section(label: str, argv: Sequence[str], *, cwd: Path | None = None) -> bytes:
    code, stdout, stderr = _run(argv, cwd=cwd)
    header = [
        f"## {label}",
        f"cwd: {cwd.resolve() if cwd else Path.cwd().resolve()}",
        f"command: {shlex.join(argv)}",
        f"exit_code: {code}",
        "stdout:",
    ]
    payload = "\n".join(header).encode() + b"\n" + stdout
    if stdout and not stdout.endswith(b"\n"):
        payload += b"\n"
    payload += b"stderr:\n" + stderr
    if stderr and not stderr.endswith(b"\n"):
        payload += b"\n"
    return payload + b"\n"


def _capture(path: Path, commands: Iterable[tuple[str, Sequence[str], Path | None]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = b"".join(_section(label, argv, cwd=cwd) for label, argv, cwd in commands)
    path.write_bytes(payload)


def _git_commands(repo: Path, *, include_fetch: bool) -> list[tuple[str, Sequence[str], Path | None]]:
    commands: list[tuple[str, Sequence[str], Path | None]] = [
        ("pwd", ["pwd"], repo),
        ("status_short_branch", ["git", "status", "--short", "--branch"], repo),
        ("status_porcelain_v2", ["git", "status", "--porcelain=v2"], repo),
        ("head", ["git", "rev-parse", "HEAD"], repo),
        ("tree", ["git", "rev-parse", "HEAD^{tree}"], repo),
        ("recent_log", ["git", "log", "-15", "--oneline", "--decorate"], repo),
        ("branches", ["git", "branch", "-a", "-vv"], repo),
        ("remotes", ["git", "remote", "-v"], repo),
    ]
    if include_fetch:
        commands.append(("fetch_all_tags", ["git", "fetch", "--all", "--tags"], repo))
    commands.extend(
        [
            ("ls_remote_origin", ["git", "ls-remote", "origin"], repo),
            ("diff_check", ["git", "diff", "--check"], repo),
            ("worktree_list", ["git", "worktree", "list", "--porcelain"], repo),
        ]
    )
    return commands


def _python_environment(python: Path, label: str) -> list[tuple[str, Sequence[str], Path | None]]:
    script = (
        "import platform,sys,torch; "
        "print(sys.version); print(platform.platform()); print(torch.__version__); "
        "print(torch.version.cuda); print(torch.cuda.is_available()); "
        "print(torch.cuda.device_count()); print(torch.get_default_dtype())"
    )
    return [
        (f"{label}_python", [str(python), "--version"], None),
        (f"{label}_python_torch", [str(python), "-c", script], None),
        (f"{label}_pip_freeze", [str(python), "-m", "pip", "freeze"], None),
    ]


def _git_value(repo: Path, *args: str) -> str | None:
    code, stdout, _ = _run(["git", *args], cwd=repo)
    return stdout.decode(errors="replace").strip() if code == 0 else None


def _source_hashes() -> dict[str, object]:
    payload: dict[str, object] = {"repositories": {}, "selected_environment": {}}
    repositories = payload["repositories"]
    assert isinstance(repositories, dict)
    for name, repo in REPOSITORIES.items():
        code, diff, stderr = _run(["git", "diff", "HEAD", "--binary"], cwd=repo)
        repositories[name] = {
            "path": str(repo),
            "head": _git_value(repo, "rev-parse", "HEAD"),
            "tree": _git_value(repo, "rev-parse", "HEAD^{tree}"),
            "branch": _git_value(repo, "branch", "--show-current"),
            "status_porcelain_v2_sha256": _sha256_bytes(
                (_run(["git", "status", "--porcelain=v2"], cwd=repo)[1])
            ),
            "tracked_diff_sha256": _sha256_bytes(diff),
            "tracked_diff_exit_code": code,
            "tracked_diff_stderr": stderr.decode(errors="replace"),
        }
    selected = payload["selected_environment"]
    assert isinstance(selected, dict)
    for key in (
        "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "CUBLAS_WORKSPACE_CONFIG",
        "CONDA_DEFAULT_ENV", "CONDA_PREFIX", "PYTHONPATH",
    ):
        selected[key] = os.environ.get(key)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing non-empty provenance directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    _capture(
        output / "torch_repository_state.txt",
        [
            *_git_commands(REPOSITORIES["torch"], include_fetch=True),
            *_git_commands(REPOSITORIES["torch_user_worktree"], include_fetch=False),
        ],
    )
    _capture(
        output / "xiangru_repository_state.txt",
        [
            *_git_commands(REPOSITORIES["xiangru"], include_fetch=True),
            *_git_commands(REPOSITORIES["xiangru_reproduction"], include_fetch=False),
        ],
    )
    _capture(
        output / "flowstar_repository_state.txt",
        _git_commands(REPOSITORIES["flowstar"], include_fetch=True),
    )
    _capture(output / "torch_environment.txt", _python_environment(TORCH_PYTHON, "torch"))
    _capture(output / "xiangru_environment.txt", _python_environment(XIANGRU_PYTHON, "xiangru"))
    hardware_commands: list[tuple[str, Sequence[str], Path | None]] = [
        ("uname", ["uname", "-a"], None),
        ("lscpu", ["lscpu"], None),
        ("memory", ["free", "-h"], None),
        ("nvidia_smi", ["nvidia-smi"], None),
        ("nvidia_smi_query", ["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,driver_version,compute_mode", "--format=csv,noheader"], None),
        ("conda_env_list", ["conda", "env", "list"], None),
        ("gcc", ["gcc", "--version"], None),
        ("gxx", ["g++", "--version"], None),
        ("cmake", ["cmake", "--version"], None),
        ("nvcc", ["nvcc", "--version"], None),
        (
            "selected_thread_environment",
            [
                str(TORCH_PYTHON),
                "-c",
                (
                    "import os; "
                    "keys=('CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','MKL_NUM_THREADS',"
                    "'OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','CUBLAS_WORKSPACE_CONFIG'); "
                    "print('\\n'.join(f'{key}={os.environ.get(key)!r}' for key in keys))"
                ),
            ],
            None,
        ),
    ]
    _capture(output / "hardware.txt", hardware_commands)
    (output / "source_hashes.json").write_text(
        json.dumps(_source_hashes(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
