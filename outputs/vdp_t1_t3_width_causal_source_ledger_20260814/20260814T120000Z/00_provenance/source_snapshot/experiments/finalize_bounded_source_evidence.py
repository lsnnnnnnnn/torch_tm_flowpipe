#!/usr/bin/env python3
"""Finalize the compact VDP bounded-source evidence package."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACKAGE = (
    ROOT
    / "outputs/vdp_t1_t3_width_causal_source_ledger_20260814/20260814T120000Z"
)
FLOWSTAR = Path("/srv/local/shengenli/flowstar_step1_stage_oracle")
EXPECTED_FLOWSTAR_SHA = "b85a3211748cb77b736fe4ad42ee02d8d2b81148"
EXPECTED_BASE_SHA = "08dd34e44f7cfc3fb456bf947959304599f07451"

SNAPSHOT_PATHS = (
    "handoff.md",
    "docs/COMPLETE_O4_BOUNDED_SOURCE_LEDGER_CONTRACT_20260814.md",
    "docs/VDP_T1_T3_WIDTH_CAUSAL_REPORT_20260814.md",
    "src/torch_tm_flowpipe/__init__.py",
    "src/torch_tm_flowpipe/batched_dense_tm.py",
    "src/torch_tm_flowpipe/flowpipe.py",
    "src/torch_tm_flowpipe/source_ledger.py",
    "src/torch_tm_flowpipe/terminal_checkpoint.py",
    "experiments/audit_bounded_source_actual_consumers.py",
    "experiments/audit_bounded_source_terminal_prestate.py",
    "experiments/benchmark_bounded_source_ledger.py",
    "experiments/build_vdp_t1_t3_width_ledger.py",
    "experiments/export_torch_step1_stage_ledger.py",
    "experiments/finalize_bounded_source_evidence.py",
    "experiments/flowstar_probe/flowstar_vdp_stock_reach_driver.cpp",
    "experiments/run_vdp_dense_backend.py",
    "experiments/summarize_bounded_source_causal_runs.py",
    "experiments/tamper_test_bounded_source_evidence.py",
    "experiments/verify_bounded_source_evidence.py",
    "experiments/verify_bounded_source_ledger_oracle.py",
    "tests/test_bounded_source_ledger.py",
    "tests/test_step1_stage_oracle_audit.py",
)

COMMANDS = (
    {
        "gate": "A",
        "purpose": "exact-decimal step-1 consumer and containment checks",
        "command": "conda run -n py11 pytest -q tests/test_step1_stage_oracle_audit.py",
    },
    {
        "gate": "B",
        "purpose": "rebuild the 632-boundary raw width ledger",
        "command": "conda run -n py11 python experiments/build_vdp_t1_t3_width_ledger.py --help  # arguments and input hashes are recorded in 01_width_ledger/provenance.json",
    },
    {
        "gate": "C",
        "purpose": "actual next-Picard consumer interventions",
        "command": "conda run -n py11 python experiments/audit_bounded_source_actual_consumers.py --help  # frozen artifact arguments are recorded in 03_consumer_audit/consumer_audit.json",
    },
    {
        "gate": "C",
        "purpose": "candidate terminal frozen-prestate intervention",
        "command": "conda run -n py11 python experiments/audit_bounded_source_terminal_prestate.py --help",
    },
    {
        "gate": "E",
        "purpose": "independent complete-O4 micro-oracles",
        "command": "conda run -n py11 python experiments/verify_bounded_source_ledger_oracle.py --output outputs/vdp_t1_t3_width_causal_source_ledger_20260814/20260814T120000Z/02_contract_oracles/independent_oracle.json",
    },
    {
        "gate": "G",
        "purpose": "fresh fixed/native candidate and legacy requests",
        "command": "conda run -n py11 python experiments/run_vdp_dense_backend.py --help  # exact request matrix and outcomes are in 04_causal_runs/fresh_{fixed,native}_requests.csv",
    },
    {
        "gate": "G",
        "purpose": "join candidate traces with the frozen Flow* ledger",
        "command": "conda run -n py11 python experiments/summarize_bounded_source_causal_runs.py --help  # immutable inputs are summarized in 04_causal_runs/scientific_summary.json",
    },
    {
        "gate": "H",
        "purpose": "CPU/CUDA B1/B8/B64/B256/B512 and actual B1 timing",
        "command": "conda run -n py11 python experiments/benchmark_bounded_source_ledger.py --output-dir outputs/vdp_t1_t3_width_causal_source_ledger_20260814/20260814T120000Z/05_performance/kernel_and_b1",
    },
    {
        "gate": "I",
        "purpose": "bytecode compilation",
        "command": "conda run -n py11 python -m compileall -q src experiments tests",
    },
    {
        "gate": "I",
        "purpose": "focused regression suite",
        "command": "conda run -n py11 pytest -q tests/test_bounded_source_ledger.py tests/test_step1_stage_oracle_audit.py tests/test_terminal_checkpoint_v2.py tests/test_canonical_status_consistency.py --junitxml=outputs/vdp_t1_t3_width_causal_source_ledger_20260814/20260814T120000Z/06_tests/focused_pytest.xml",
    },
    {
        "gate": "I",
        "purpose": "complete regression suite",
        "command": "conda run -n py11 pytest -q --junitxml=outputs/vdp_t1_t3_width_causal_source_ledger_20260814/20260814T120000Z/06_tests/full_pytest.xml",
    },
)


def _run(*args: str, cwd: Path = ROOT) -> str:
    completed = subprocess.run(args, cwd=cwd, check=True, capture_output=True, text=True)
    return completed.stdout.strip()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _snapshot(package: Path) -> list[str]:
    destination = package / "00_provenance/source_snapshot"
    destination.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for relative in SNAPSHOT_PATHS:
        source = ROOT / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied.append(relative)
    goal = Path("/srv/local/shengenli/codex/goal_vdp_terminal.md")
    shutil.copy2(goal, destination / "goal_vdp_terminal.md")
    copied.append("goal_vdp_terminal.md")
    return copied


def finalize(package: Path, stage: str, scientific_sha: str | None, attestation_sha: str | None) -> dict[str, Any]:
    package = package.resolve()
    if not package.is_dir():
        raise FileNotFoundError(package)
    copied = _snapshot(package)
    branch = _run("git", "branch", "--show-current")
    head = _run("git", "rev-parse", "HEAD")
    flowstar_head = _run("git", "rev-parse", "HEAD", cwd=FLOWSTAR)
    if flowstar_head != EXPECTED_FLOWSTAR_SHA:
        raise ValueError(f"Flow* identity drift: {flowstar_head}")
    diff = subprocess.run(
        ["git", "diff", "--binary", EXPECTED_BASE_SHA], cwd=ROOT, check=True, capture_output=True
    ).stdout
    provenance = {
        "schema": "vdp_bounded_source_evidence_provenance_v1",
        "stage": stage,
        "branch": branch,
        "head_at_finalization": head,
        "expected_base_sha": EXPECTED_BASE_SHA,
        "flowstar_repo": str(FLOWSTAR),
        "flowstar_sha": flowstar_head,
        "scientific_sha": scientific_sha,
        "attestation_sha": attestation_sha,
        "source_diff_from_base_sha256": hashlib.sha256(diff).hexdigest(),
        "source_snapshot_files": copied,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
    }
    _json_write(package / "00_provenance/provenance.json", provenance)
    _json_write(
        package / "00_provenance/commands.json",
        {"schema": "vdp_bounded_source_commands_v1", "commands": COMMANDS},
    )

    excluded = {package / "manifest.json", package / "SHA256SUMS"}
    files = sorted(
        path for path in package.rglob("*") if path.is_file() and path not in excluded
    )
    records = {
        path.relative_to(package).as_posix(): {"bytes": path.stat().st_size, "sha256": _sha(path)}
        for path in files
    }
    total_bytes = sum(record["bytes"] for record in records.values())
    manifest = {
        "schema": "vdp_t1_t3_bounded_source_evidence_package_v1",
        "conclusion": "T1_T3_WIDTH_CAUSE_CLOSED__EARLY_GAP_IMPROVED__TERMINAL_STILL_OPEN",
        "candidate": "normalized_insertion_bounded_source_ledger_o4_g1",
        "stage": stage,
        "scientific_sha": scientific_sha,
        "attestation_sha": attestation_sha,
        "file_count": len(records),
        "total_bytes_excluding_manifest_and_sums": total_bytes,
        "under_25_mib": total_bytes < 25 * 1024 * 1024,
        "files": records,
    }
    _json_write(package / "manifest.json", manifest)
    sums = "".join(f"{record['sha256']}  {relative}\n" for relative, record in records.items())
    (package / "SHA256SUMS").write_text(sums, encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument("--stage", choices=("scientific_precommit", "attestation"), required=True)
    parser.add_argument("--scientific-sha")
    parser.add_argument("--attestation-sha")
    args = parser.parse_args()
    result = finalize(args.package, args.stage, args.scientific_sha, args.attestation_sha)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
