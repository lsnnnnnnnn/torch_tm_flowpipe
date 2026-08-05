#!/usr/bin/env python3
"""Create the provenance closure for the VDP cross-step carry audit."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
FORMAL_SHA = "a1fb3527bb7c12ce23aa2fb49d66f6380c463c90"
PACKAGING_SHA = "2e4507220a631a21dbe5227a7f9a5201948aedde"
FLOWSTAR_SHA = "b85a3211748cb77b736fe4ad42ee02d8d2b81148"
R4_MANIFEST = ROOT / "evidence/vdp_terminal_range_closure/20260805T055556Z/manifest.json"
H1_MANIFEST = ROOT / "outputs/vdp_later_terminal_factorized_range/manifest.json"


def _run(command: Sequence[str], cwd: Path, *, check: bool = False) -> dict[str, Any]:
    completed = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    if check and completed.returncode:
        raise RuntimeError(f"command failed ({completed.returncode}): {' '.join(command)}\n{completed.stderr}")
    return {
        "command": list(command),
        "cwd": str(cwd),
        "exit_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def _write_json(path: Path, value: Any) -> None:
    _write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _git_paths_at(commit: str) -> list[str]:
    result = _run(["git", "ls-tree", "-r", "--name-only", commit], ROOT, check=True)
    return result["stdout"].splitlines()


def _numerical_paths(commit: str, *, packaging: bool) -> list[str]:
    tracked = _git_paths_at(commit)
    selected = [
        path
        for path in tracked
        if path.startswith("src/torch_tm_flowpipe/") and path.endswith(".py")
    ]
    required = {
        "benchmarks/canonical.yaml",
        "benchmarks/three_tool_matched_contract.yaml",
        "experiments/run_vdp_dense_backend.py",
        "experiments/replay_vdp_terminal_range.py",
        "experiments/run_vdp_later_terminal_factorized_range.py",
        "tests/test_dense_horner_range.py",
        "tests/test_vdp_later_terminal_factorized_range.py",
        "evidence/vdp_terminal_range_closure/20260805T055556Z/05_fresh_horizons/"
        "t6p5_proactive_d1_truncation/terminal_checkpoint/terminal_state.json",
        "evidence/vdp_terminal_range_closure/20260805T055556Z/05_fresh_horizons/"
        "t6p5_proactive_d1_truncation/terminal_checkpoint/terminal_state_manifest.json",
        "evidence/vdp_terminal_range_closure/20260805T055556Z/05_fresh_horizons/"
        "t6p5_proactive_d1_truncation/terminal_checkpoint/terminal_reference.json",
    }
    if packaging:
        required.update(
            {
                "experiments/package_vdp_later_terminal_factorized_range.py",
                "docs/VDP_LATER_TERMINAL_FACTORIZED_RANGE.md",
            }
        )
    missing = sorted(required - set(tracked))
    if missing:
        raise RuntimeError(f"required paths absent at {commit}: {missing}")
    return sorted(set(selected) | required)


def _hash_git_sources(commit: str, *, packaging: bool) -> dict[str, Any]:
    rows = []
    for path in _numerical_paths(commit, packaging=packaging):
        completed = subprocess.run(["git", "show", f"{commit}:{path}"], cwd=ROOT, capture_output=True)
        if completed.returncode:
            raise RuntimeError(completed.stderr.decode("utf-8", errors="replace"))
        rows.append({"path": path, "bytes": len(completed.stdout), "sha256": _sha256(completed.stdout)})
    return {
        "schema": "torch_tm_flowpipe_source_hashes_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "commit": commit,
        "source_kind": "immutable_git_objects",
        "dirty": False,
        "commands": [
            ["git", "ls-tree", "-r", "--name-only", commit],
            ["git", "show", f"{commit}:<path>"],
        ],
        "files": rows,
    }


def _manifest_entries(value: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    files = value.get("files", [])
    if not isinstance(files, list):
        raise ValueError("manifest files must be a list")
    yield from files


def _audit_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    root = path.parent
    checked = []
    failures = []
    for entry in _manifest_entries(value):
        relative = str(entry.get("path", ""))
        target = root / relative
        actual = target.read_bytes() if target.is_file() else None
        row = {
            "path": relative,
            "exists": actual is not None,
            "expected_bytes": entry.get("bytes"),
            "actual_bytes": len(actual) if actual is not None else None,
            "expected_sha256": entry.get("sha256"),
            "actual_sha256": _sha256(actual) if actual is not None else None,
        }
        row["matches"] = bool(
            actual is not None
            and row["actual_bytes"] == row["expected_bytes"]
            and row["actual_sha256"] == row["expected_sha256"]
        )
        checked.append(row)
        if not row["matches"]:
            failures.append(row)
    compressed_checked = []
    compressed_failures = []
    for entry in value.get("compressed_sources", []):
        relative = str(entry.get("stored", ""))
        target = root / relative
        actual = target.read_bytes() if target.is_file() else None
        row = {
            "stored": relative,
            "exists": actual is not None,
            "expected_bytes": entry.get("stored_bytes"),
            "actual_bytes": len(actual) if actual is not None else None,
            "expected_sha256": entry.get("stored_sha256"),
            "actual_sha256": _sha256(actual) if actual is not None else None,
            "compression": entry.get("compression"),
            "source_sha256": entry.get("source_sha256"),
        }
        row["matches"] = bool(
            actual is not None
            and row["actual_bytes"] == row["expected_bytes"]
            and row["actual_sha256"] == row["expected_sha256"]
        )
        compressed_checked.append(row)
        if not row["matches"]:
            compressed_failures.append(row)
    return {
        "manifest": str(path.relative_to(ROOT)),
        "manifest_sha256": _sha256(path.read_bytes()),
        "schema": value.get("schema"),
        "declared_commits": {
            key: value.get(key)
            for key in (
                "base_commit",
                "code_commit",
                "baseline_sha",
                "implementation_sha_used_by_formal_runs",
                "packaging_sha",
            )
            if key in value
        },
        "file_count": len(checked),
        "file_failures": failures,
        "compressed_count": len(compressed_checked),
        "compressed_failures": compressed_failures,
        "passed": not failures and not compressed_failures,
        "files": checked,
        "compressed_sources": compressed_checked,
    }


def _render_commands(results: Sequence[Mapping[str, Any]]) -> str:
    pieces = []
    for result in results:
        pieces.append(f"$ {' '.join(result['command'])}\n")
        pieces.append(str(result["stdout"]))
        if result["stderr"]:
            pieces.append("[stderr]\n" + str(result["stderr"]))
        pieces.append(f"[exit_code={result['exit_code']}]\n")
    return "".join(pieces)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--flowstar-root", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    repository_commands = [
        ["pwd"],
        ["git", "status", "--short", "--branch"],
        ["git", "status", "--porcelain=v2"],
        ["git", "rev-parse", "HEAD"],
        ["git", "log", "-15", "--oneline", "--decorate"],
        ["git", "branch", "-vv"],
        ["git", "remote", "-v"],
        ["git", "diff", "--check"],
    ]
    repository_results = [_run(command, ROOT) for command in repository_commands]
    _write_text(output / "repository_state.txt", _render_commands(repository_results))

    flowstar_root = args.flowstar_root.resolve()
    flowstar_commands = [
        ["git", "status", "--short", "--branch"],
        ["git", "status", "--porcelain=v2"],
        ["git", "rev-parse", "HEAD"],
        ["git", "log", "-8", "--oneline", "--decorate"],
        ["git", "diff", "--check"],
        ["git", "diff", "--", "flowstar-toolbox/TaylorModel.h"],
    ]
    flowstar_results = [_run(command, flowstar_root) for command in flowstar_commands]
    _write_text(output / "flowstar_state.txt", _render_commands(flowstar_results))

    torch_info = _run(
        [
            "conda",
            "run",
            "-n",
            "py11",
            "python",
            "-c",
            (
                "import json,platform,sys,torch;"
                "print(json.dumps({'python':sys.version,'pytorch':torch.__version__,"
                "'cuda_version':torch.version.cuda,'cuda_available':torch.cuda.is_available(),"
                "'gpu_count':torch.cuda.device_count(),'gpu_names':[torch.cuda.get_device_name(i) "
                "for i in range(torch.cuda.device_count())],'platform':platform.platform()},sort_keys=True))"
            ),
        ],
        ROOT,
        check=True,
    )
    environment = json.loads(torch_info["stdout"].strip())
    environment.update(
        {
            "schema": "torch_tm_flowpipe_vdp_cross_step_environment_v1",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "collector_python": sys.version,
            "collector_platform": platform.platform(),
            "flowstar_root": str(flowstar_root),
            "flowstar_exact_stock_commit": FLOWSTAR_SHA,
            "flowstar_build_compatibility_flag": "-fpermissive",
            "flowstar_build_compatibility_reason": (
                "exact stock TaylorModel.h assigns remainder in a const derivative method and GCC 15 rejects it"
            ),
            "environment_variables": {
                key: os.environ[key]
                for key in ("FLOWSTAR_ROOT", "CONDA_DEFAULT_ENV", "CONDA_PREFIX")
                if key in os.environ
            },
            "command": torch_info["command"],
        }
    )
    _write_json(output / "environment.json", environment)
    _write_json(output / "formal_run_source_hashes.json", _hash_git_sources(FORMAL_SHA, packaging=False))
    _write_json(output / "packaging_source_hashes.json", _hash_git_sources(PACKAGING_SHA, packaging=True))
    audits = [_audit_manifest(R4_MANIFEST), _audit_manifest(H1_MANIFEST)]
    _write_json(
        output / "existing_evidence_audit.json",
        {
            "schema": "torch_tm_flowpipe_existing_evidence_audit_v1",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "formal_numerical_source_commit": FORMAL_SHA,
            "packaging_report_commit": PACKAGING_SHA,
            "current_branch_head": _run(["git", "rev-parse", "HEAD"], ROOT, check=True)["stdout"].strip(),
            "manifests": audits,
            "passed": all(row["passed"] for row in audits),
        },
    )
    return 0 if all(row["passed"] for row in audits) else 1


if __name__ == "__main__":
    raise SystemExit(main())
