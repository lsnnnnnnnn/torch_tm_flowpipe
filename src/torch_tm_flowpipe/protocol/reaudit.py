from __future__ import annotations

import hashlib
import json
import os
import platform
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Mapping, Sequence

from .backend_identity import classify_flowstar_backend
from .schema import (
    ComparisonLane,
    FailureCategory,
    RUNTIME_BOUNDARY_VERSION,
    SoundnessLevel,
)


MANIFEST_SCHEMA_VERSION = "three-tool-reaudit-1.0.0"
PRIMARY_REQUIRED_FIELDS = (
    "backend",
    "lane",
    "completed_horizon",
    "requested_horizon",
    "validation_status",
    "soundness_level",
    "primary_eligible",
    "endpoint_semantics",
    "effective_support_sha256",
    "runtime_boundary",
    "backend_sha",
    "run_authority",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run(command: Sequence[str], *, cwd: Path | None = None) -> dict[str, Any]:
    process = subprocess.run(
        list(command),
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "command": list(command),
        "cwd": str(cwd) if cwd else None,
        "exit_code": process.returncode,
        "stdout": process.stdout,
        "stderr": process.stderr,
    }


def _git(root: Path, *arguments: str) -> str:
    result = _run(("git", "-C", str(root), *arguments))
    if result["exit_code"]:
        raise RuntimeError(result["stderr"] or "git command failed")
    return str(result["stdout"]).strip()


def repository_record(root: Path | None) -> dict[str, Any]:
    if root is None or not root.exists():
        return {"source_kind": "missing", "path": str(root) if root else None}
    canonical = root.resolve()
    if not (canonical / ".git").exists() and not _run(
        ("git", "-C", str(canonical), "rev-parse", "--git-dir")
    )["exit_code"] == 0:
        return {"source_kind": "archive", "path": str(canonical)}
    status = _git(canonical, "status", "--porcelain=v2", "--branch")
    submodules = _run(
        ("git", "-C", str(canonical), "submodule", "status", "--recursive")
    )
    return {
        "source_kind": "git_checkout",
        "path": str(canonical),
        "remote": _git(canonical, "remote", "get-url", "origin"),
        "branch": _git(canonical, "rev-parse", "--abbrev-ref", "HEAD"),
        "sha": _git(canonical, "rev-parse", "HEAD"),
        "dirty": any(
            line and not line.startswith("# branch.")
            for line in status.splitlines()
        ),
        "status_porcelain_v2": status,
        "submodules": (
            submodules["stdout"].splitlines()
            if submodules["exit_code"] == 0
            else {"error": submodules["stderr"]}
        ),
    }


def _memory_bytes() -> int | None:
    meminfo = Path("/proc/meminfo")
    if not meminfo.is_file():
        return None
    for line in meminfo.read_text(encoding="utf-8").splitlines():
        if line.startswith("MemTotal:"):
            return int(line.split()[1]) * 1024
    return None


def _cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def _version(command: Sequence[str]) -> dict[str, Any]:
    result = _run(command)
    return {
        "command": list(command),
        "exit_code": result["exit_code"],
        "version": (result["stdout"] or result["stderr"]).splitlines()[:3],
    }


def _gpu_record() -> dict[str, Any]:
    if shutil.which("nvidia-smi") is None:
        return {"available": False, "reason": "nvidia-smi missing"}
    result = _run(
        (
            "nvidia-smi",
            "--query-gpu=index,name,uuid,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        )
    )
    return {
        "available": result["exit_code"] == 0,
        "devices": result["stdout"].splitlines(),
        "error": result["stderr"],
    }


def _torch_record() -> dict[str, Any]:
    try:
        import torch
    except Exception as error:  # pragma: no cover - environment dependent
        return {"available": False, "error": repr(error)}
    return {
        "available": True,
        "version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "devices": [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ],
    }


def collect_manifest(
    *,
    run_id: str,
    repo_roots: Mapping[str, Path | None],
    benchmark_files: Sequence[Path],
    flowstar_binary: Path | None,
    environment: Mapping[str, str],
    started_utc: str,
    started_local: str,
) -> dict[str, Any]:
    repositories = {
        name: repository_record(root) for name, root in repo_roots.items()
    }
    flowstar_root = repo_roots.get("flowstar")
    flowstar_identity: dict[str, Any]
    try:
        flowstar_identity = (
            classify_flowstar_backend(flowstar_root, environment=environment).to_record()
            if flowstar_root is not None and flowstar_root.exists()
            else {"backend_class": "missing"}
        )
    except Exception as error:
        flowstar_identity = {
            "backend_class": "identity_error",
            "primary_eligible": False,
            "error": repr(error),
        }
    binary_record: dict[str, Any] = {"path": None}
    if flowstar_binary is not None and flowstar_binary.is_file():
        ldd = _run(("ldd", str(flowstar_binary)))
        binary_record = {
            "path": str(flowstar_binary.resolve()),
            "sha256": sha256_file(flowstar_binary),
            "ldd_exit_code": ldd["exit_code"],
            "ldd": ldd["stdout"].splitlines(),
            "ldd_stderr": ldd["stderr"],
        }
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": run_id,
        "started_utc": started_utc,
        "started_local": started_local,
        "host": {
            "hostname": platform.node(),
            "os": platform.platform(),
            "cpu_model": _cpu_model(),
            "cpu_count": os.cpu_count(),
            "ram_bytes": _memory_bytes(),
            "gpu": _gpu_record(),
        },
        "software": {
            "python": sys.version,
            "python_executable": sys.executable,
            "compiler": _version(("g++", "--version")),
            "cmake": _version(("cmake", "--version")),
            "torch": _torch_record(),
        },
        "repositories": repositories,
        "flowstar_backend_identity": flowstar_identity,
        "flowstar_binary": binary_record,
        "benchmark_files": [
            {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
            }
            for path in benchmark_files
            if path.is_file()
        ],
        "execution_contract": {
            "dtype": "float64 unless backend-native lane says otherwise",
            "device": "recorded per command/row",
            "thread_count": environment.get("OMP_NUM_THREADS", "unspecified"),
            "environment": {
                key: environment[key]
                for key in sorted(environment)
                if key.startswith(("FLOWSTAR_", "DIFFREACH_", "CUDA_", "OMP_"))
            },
        },
        "commands": [],
    }


def validate_manifest(manifest: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in (
        "schema_version",
        "run_id",
        "started_utc",
        "started_local",
        "host",
        "software",
        "repositories",
        "flowstar_backend_identity",
        "flowstar_binary",
        "benchmark_files",
        "execution_contract",
        "commands",
    ):
        if field not in manifest:
            errors.append(f"missing manifest field: {field}")
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        errors.append("manifest schema_version mismatch")
    repositories = manifest.get("repositories")
    if not isinstance(repositories, Mapping):
        errors.append("repositories must be a mapping")
    else:
        for name in ("torch_tm_flowpipe", "flowstar", "diffreach", "xiangru"):
            if name not in repositories:
                errors.append(f"missing repository record: {name}")
    return errors


def validate_primary_row(row: Mapping[str, Any]) -> list[str]:
    errors = [
        f"missing primary field: {field}"
        for field in PRIMARY_REQUIRED_FIELDS
        if field not in row or row.get(field) in (None, "")
    ]
    if errors:
        return errors
    if row.get("backend") in {"patched-audit", "torch-dense-prototype"}:
        errors.append("primary backend is categorically ineligible")
    if row.get("lane") not in {lane.value for lane in ComparisonLane}:
        errors.append("invalid lane")
    if row.get("soundness_level") not in {level.value for level in SoundnessLevel}:
        errors.append("invalid soundness_level")
    if row.get("run_authority") != "authoritative":
        errors.append("smoke/exploratory row cannot be primary")
    if row.get("validation_status") != FailureCategory.COMPLETED.value:
        errors.append("failed/incomplete row cannot be primary")
    try:
        requested = float(row["requested_horizon"])
        completed = float(row["completed_horizon"])
    except (TypeError, ValueError):
        errors.append("horizons must be numeric")
    else:
        if abs(requested - completed) > 1e-12 * max(1.0, abs(requested)):
            errors.append("requested horizon was not completed")
    if type(row.get("primary_eligible")) is not bool:
        errors.append("primary_eligible must be a boolean")
    elif row["primary_eligible"] is not True:
        errors.append("row is explicitly primary-ineligible")
    if row.get("endpoint_semantics") != "raw_endpoint":
        errors.append("primary width must use raw_endpoint")
    if row.get("runtime_boundary") != RUNTIME_BOUNDARY_VERSION:
        errors.append("runtime boundary mismatch")
    return errors


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )

