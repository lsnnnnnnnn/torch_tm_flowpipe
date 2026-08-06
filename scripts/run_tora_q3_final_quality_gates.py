#!/usr/bin/env python3
"""Run the frozen final quality commands and split private/public evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def file_digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--controller-trace", type=Path, required=True)
    parser.add_argument("--xiangru-root", type=Path, required=True)
    parser.add_argument("--crown-python", type=Path, required=True)
    parser.add_argument("--private-log", type=Path, required=True)
    parser.add_argument("--public-summary", type=Path, required=True)
    args = parser.parse_args()
    root = args.repository.resolve()
    crown_python = args.crown_python.expanduser().absolute()
    if not crown_python.is_file():
        raise FileNotFoundError(crown_python)
    conda = shutil.which("conda")
    if conda is None:
        raise RuntimeError("conda is unavailable")
    external_env = {
        **os.environ,
        "TORA_CONTROLLER_PATH": str(args.controller.resolve()),
        "TORA_CONTROLLER_TRACE_PATH": str(args.controller_trace.resolve()),
        "XIANGRU_ROOT": str(args.xiangru_root.resolve()),
    }
    site_query = subprocess.run(
        [
            str(crown_python),
            "-c",
            "import json, site; print(json.dumps(site.getsitepackages()))",
        ],
        cwd=root,
        text=True,
        capture_output=True,
        check=True,
    )
    crown_sites = json.loads(site_query.stdout)
    crown_env = {
        **external_env,
        "PYTHONPATH": os.pathsep.join(
            [
                str(root),
                str(root / "src"),
                *crown_sites,
                external_env.get("PYTHONPATH", ""),
            ]
        ),
    }
    commands: list[tuple[str, list[str], dict[str, str]]] = [
        (
            "editable_test_install",
            [conda, "run", "-n", "py11", "python", "-m", "pip", "install", "-e", ".[test]"],
            os.environ.copy(),
        ),
        (
            "full_pytest",
            [conda, "run", "-n", "py11", "pytest", "-q"],
            os.environ.copy(),
        ),
        (
            "external_integration_py11",
            [conda, "run", "-n", "py11", "pytest", "-q", "-m", "external_integration"],
            external_env,
        ),
        (
            "controller_external_environment_preflight",
            [
                str(crown_python),
                "-c",
                (
                    "import json, onnx, torch, auto_LiRPA; "
                    "print(json.dumps({'python': __import__('sys').version, "
                    "'onnx': onnx.__version__, 'torch': torch.__version__, "
                    "'auto_lirpa': getattr(auto_LiRPA, '__version__', 'unknown'), "
                    "'cuda': torch.version.cuda, "
                    "'gpu': torch.cuda.get_device_name(0)}))"
                ),
            ],
            crown_env,
        ),
        (
            "controller_external_integration_frozen_crown_env",
            [str(crown_python), "-m", "pytest", "-q", "tests/test_tora_controller.py", "-m", "external_integration"],
            crown_env,
        ),
        (
            "git_diff_check",
            ["git", "diff", "--check"],
            os.environ.copy(),
        ),
        (
            "git_status_short_branch",
            ["git", "status", "--short", "--branch"],
            os.environ.copy(),
        ),
    ]
    records: list[dict[str, Any]] = []
    for label, command, environment in commands:
        started = time.perf_counter()
        completed = subprocess.run(
            command,
            cwd=root,
            env=environment,
            text=True,
            capture_output=True,
        )
        records.append({
            "label": label,
            "command": command,
            "exit_code": completed.returncode,
            "wall_seconds": time.perf_counter() - started,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        })
    raw = json.dumps(records, indent=2, sort_keys=True) + "\n"
    args.private_log.parent.mkdir(parents=True, exist_ok=True)
    args.private_log.write_text(raw, encoding="utf-8")
    public_rows = [
        {
            "label": row["label"],
            "exit_code": row["exit_code"],
            "wall_seconds": row["wall_seconds"],
            "stdout_sha256": digest(row["stdout"]),
            "stderr_sha256": digest(row["stderr"]),
            "stdout_line_count": len(row["stdout"].splitlines()),
            "stderr_line_count": len(row["stderr"].splitlines()),
        }
        for row in records
    ]
    summary = {
        "schema": "tora_q3_final_quality_gates_v1",
        "status": "PASS" if all(row["exit_code"] == 0 for row in records) else "FAIL",
        "commands": public_rows,
        "private_raw_log_sha256": hashlib.sha256(raw.encode()).hexdigest(),
        "external_assets_supplied_by_environment": True,
        "external_source_hashes": {
            "controller": file_digest(args.controller.resolve()),
            "controller_trace": file_digest(args.controller_trace.resolve()),
        },
    }
    args.public_summary.parent.mkdir(parents=True, exist_ok=True)
    args.public_summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": summary["status"],
        "exit_codes": {row["label"]: row["exit_code"] for row in records},
    }))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
