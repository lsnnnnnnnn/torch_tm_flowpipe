#!/usr/bin/env python3
"""Run final portable, external-integration, and explicit GPU quality gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Any


RESULT_COUNTS = re.compile(r"(?P<count>\d+) (?P<kind>passed|skipped|deselected)")


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pytest_counts(output: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for match in RESULT_COUNTS.finditer(output):
        counts[match.group("kind")] = int(match.group("count"))
    return counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--controller-trace", type=Path, required=True)
    parser.add_argument("--xiangru-root", type=Path, required=True)
    parser.add_argument("--xiangru-stage-trace", type=Path, required=True)
    parser.add_argument("--torch-stage-trace", type=Path, required=True)
    parser.add_argument("--private-log", type=Path, required=True)
    parser.add_argument("--public-summary", type=Path, required=True)
    args = parser.parse_args()
    root = args.repository.resolve()
    # Preserve the environment entry point.  Resolving its symlink can bypass
    # the virtual environment and silently select the base interpreter.
    python = args.python.expanduser().absolute()
    if not python.is_file():
        raise FileNotFoundError(python)

    external_env = {
        **os.environ,
        "TORA_CONTROLLER_PATH": str(args.controller.resolve()),
        "TORA_CONTROLLER_TRACE_PATH": str(args.controller_trace.resolve()),
        "XIANGRU_ROOT": str(args.xiangru_root.resolve()),
        "TORA_XIANGRU_STAGE_TRACE_PATH": str(args.xiangru_stage_trace.resolve()),
        "TORA_TORCH_STAGE_TRACE_PATH": str(args.torch_stage_trace.resolve()),
    }
    commands: list[tuple[str, list[str], dict[str, str]]] = [
        (
            "editable_test_install",
            [str(python), "-m", "pip", "install", "-e", ".[test]"],
            os.environ.copy(),
        ),
        (
            "compileall",
            [
                str(python),
                "-m",
                "compileall",
                "-q",
                "src",
                "tests",
                "experiments",
                "scripts",
                "examples",
            ],
            os.environ.copy(),
        ),
        ("portable_full_pytest", [str(python), "-m", "pytest", "-q"], os.environ.copy()),
        (
            "external_integration",
            [str(python), "-m", "pytest", "-q", "-m", "external_integration"],
            external_env,
        ),
        (
            "gpu_focused",
            [
                str(python),
                "-m",
                "pytest",
                "-q",
                "tests/test_tora_algorithm_aligned_q3.py",
                "tests/test_tora_fused_kernel.py",
                "-m",
                "cuda",
            ],
            os.environ.copy(),
        ),
        (
            "readme_surface",
            [str(python), "scripts/check_readme_surface.py"],
            os.environ.copy(),
        ),
        ("git_diff_check", ["git", "diff", "--check"], os.environ.copy()),
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
        records.append(
            {
                "label": label,
                "command": command,
                "exit_code": completed.returncode,
                "wall_seconds": time.perf_counter() - started,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            }
        )

    environment_probe = subprocess.run(
        [
            str(python),
            "-c",
            (
                "import json, platform, torch; "
                "print(json.dumps({'python': platform.python_version(), "
                "'pytorch': torch.__version__, 'cuda_runtime': torch.version.cuda, "
                "'cuda_available': torch.cuda.is_available(), "
                "'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}))"
            ),
        ],
        cwd=root,
        text=True,
        capture_output=True,
        check=True,
    )
    environment = json.loads(environment_probe.stdout)
    raw = json.dumps(records, indent=2, sort_keys=True) + "\n"
    args.private_log.parent.mkdir(parents=True, exist_ok=True)
    args.private_log.write_text(raw, encoding="utf-8")

    rows = []
    for row in records:
        combined = f"{row['stdout']}\n{row['stderr']}"
        rows.append(
            {
                "label": row["label"],
                "exit_code": row["exit_code"],
                "wall_seconds": row["wall_seconds"],
                "stdout_sha256": digest(row["stdout"]),
                "stderr_sha256": digest(row["stderr"]),
                "result_counts": pytest_counts(combined),
            }
        )
    by_label = {row["label"]: row for row in rows}
    status = "PASS" if all(row["exit_code"] == 0 for row in rows) else "FAIL"
    summary = {
        "schema": "tora_q3_stage_parity_final_quality_gates_v2",
        "status": status,
        "compileall": (
            "PASS" if by_label["compileall"]["exit_code"] == 0 else "FAIL"
        ),
        "portable": {
            **by_label["portable_full_pytest"]["result_counts"],
            "status": (
                "PASS"
                if by_label["portable_full_pytest"]["exit_code"] == 0
                else "FAIL"
            ),
        },
        "external_integration": {
            **by_label["external_integration"]["result_counts"],
            "explicit_controller_and_stage_assets": True,
            "status": (
                "PASS"
                if by_label["external_integration"]["exit_code"] == 0
                else "FAIL"
            ),
        },
        "gpu_focused": {
            **by_label["gpu_focused"]["result_counts"],
            "environment": environment,
            "status": (
                "PASS"
                if by_label["gpu_focused"]["exit_code"] == 0
                and environment["cuda_available"]
                and by_label["gpu_focused"]["result_counts"].get("passed", 0) > 0
                else "FAIL"
            ),
        },
        "commands": rows,
        "private_raw_log_sha256": hashlib.sha256(raw.encode()).hexdigest(),
        "external_source_hashes": {
            "controller": file_digest(args.controller.resolve()),
            "controller_trace": file_digest(args.controller_trace.resolve()),
            "xiangru_stage_trace": file_digest(args.xiangru_stage_trace.resolve()),
            "torch_stage_trace": file_digest(args.torch_stage_trace.resolve()),
        },
        "raw_paths_in_public_record": False,
    }
    if summary["gpu_focused"]["status"] != "PASS":
        summary["status"] = "FAIL"
    args.public_summary.parent.mkdir(parents=True, exist_ok=True)
    args.public_summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": summary["status"], "environment": environment}))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
