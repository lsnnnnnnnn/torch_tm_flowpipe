#!/usr/bin/env python3
"""Build/finalize the compact, self-contained VDP G2 evidence package."""
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

import torch


ROOT = Path(__file__).resolve().parents[1]
PARTIAL = "G2_MECHANISM_IMPROVED__PRODUCTION_GATE_NOT_MET"
TOTAL = "LOSSLESS_CROSS_OPERATOR_CELL_UNAVAILABLE__TOTAL_CAUSE_OPEN"
SNAPSHOT_PATHS = (
    "handoff.md",
    "benchmarks/vdp_g2_shared_column_contract_20260815.json",
    "docs/VDP_G1_CAUSAL_CLAIM_ERRATUM_20260815.md",
    "docs/VDP_T1_T3_RESIDUAL_CAUSAL_DECOMPOSITION_20260815.md",
    "docs/COMPLETE_O4_G2_SHARED_COLUMN_CONTRACT_20260815.md",
    "docs/VDP_G2_SHARED_COLUMN_RESULT_20260815.md",
    "src/torch_tm_flowpipe/__init__.py",
    "src/torch_tm_flowpipe/batched_dense_tm.py",
    "src/torch_tm_flowpipe/flowpipe.py",
    "src/torch_tm_flowpipe/g2_shared_column.py",
    "src/torch_tm_flowpipe/lossless_state_queue_schema.py",
    "src/torch_tm_flowpipe/source_ledger.py",
    "src/torch_tm_flowpipe/terminal_checkpoint.py",
    "experiments/audit_g1_owner_interventions_20260815.py",
    "experiments/audit_g1_terminal_owner_interventions_20260815.py",
    "experiments/audit_g2_checkpoint_resume_20260815.py",
    "experiments/audit_vdp_lossless_operator_matrix_20260815.py",
    "experiments/benchmark_g2_cpu_cuda_20260815.py",
    "experiments/build_vdp_g2_evidence_20260815.py",
    "experiments/export_g2_blackbox_coefficients.py",
    "experiments/flowstar_probe/flowstar_lossless_state_queue_bridge.cpp",
    "experiments/independent_g2_exact_oracle.py",
    "experiments/run_vdp_dense_backend.py",
    "experiments/run_vdp_g2_fresh_matrix_20260815.py",
    "experiments/run_vdp_g2_fresh_clone_acceptance_20260815.py",
    "experiments/summarize_vdp_g2_matrix_20260815.py",
    "experiments/tamper_test_vdp_g2_evidence_20260815.py",
    "experiments/verify_vdp_g2_evidence_20260815.py",
    "tests/test_batched_dense_runner_contract.py",
    "tests/test_g2_shared_column.py",
    "tests/test_vdp_g2_evidence_contract.py",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def copy_file(source: Path, target: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def copy_source_snapshot(package: Path) -> list[str]:
    destination = package / "00_provenance/source_snapshot"
    copied = []
    for relative in SNAPSHOT_PATHS:
        copy_file(ROOT / relative, destination / relative)
        copied.append(relative)
    copy_file(Path("/srv/local/shengenli/codex/goal_vdp_terminal.md"), destination / "goal_vdp_terminal.md")
    copied.append("goal_vdp_terminal.md")
    return copied


def copy_matrix_minimal(matrix_root: Path, target: Path) -> None:
    matrix = json.loads((matrix_root / "matrix.json").read_text(encoding="utf-8"))
    copy_file(matrix_root / "matrix.json", target / "matrix.json")
    copy_file(matrix_root / "requests.csv", target / "requests.csv")
    for row in matrix["rows"]:
        source = matrix_root / row["relative_output"]
        destination = target / "request_summaries" / row["relative_output"]
        for name in ("summary.json", "decision.json", "command.json", "config_snapshot.yaml", "matrix_stdout.txt", "matrix_stderr.txt"):
            path = source / name
            if path.is_file():
                copy_file(path, destination / name)
        if float(row["requested_horizon"]) == 10.0:
            checkpoint = source / "terminal_checkpoint"
            if checkpoint.is_dir():
                for path in checkpoint.iterdir():
                    if path.is_file():
                        copy_file(path, destination / "terminal_checkpoint" / path.name)


def initial_build(args: argparse.Namespace) -> None:
    package = args.package.resolve()
    if package.exists():
        raise FileExistsError(package)
    package.mkdir(parents=True)
    copied = copy_source_snapshot(package)
    copy_file(
        ROOT / "benchmarks/vdp_g2_shared_column_contract_20260815.json",
        package / "00_provenance/vdp_g2_shared_column_contract_20260815.json",
    )
    copy_file(args.gate_a / "operator_matrix.json", package / "01_gate_a/operator_matrix.json")
    for child in args.gate_a.iterdir():
        if child.is_dir():
            shutil.copytree(child, package / "01_gate_a/fixtures" / child.name)
    copy_file(args.gate_b_fixed, package / "02_gate_b/fixed_owner_interventions.json")
    copy_file(args.gate_b_terminal, package / "02_gate_b/terminal_owner_interventions.json")
    copy_file(args.blackbox, package / "03_oracle/blackbox.json")
    copy_file(args.oracle, package / "03_oracle/independent_oracle.json")
    shutil.copytree(args.resume_audit, package / "03_oracle/checkpoint_resume")
    copy_matrix_minimal(args.matrix_root, package / "04_matrix")
    for name in ("fixed_curve.csv.gz", "resource_curve.csv.gz", "ratio_crossings.csv", "scientific_summary.json"):
        copy_file(args.summary_dir / name, package / "04_matrix" / name)
    copy_file(args.performance / "performance.json", package / "05_performance/performance.json")
    for directory in ("full_solver_cpu_B1_T0p1", "full_solver_cuda_B1_T0p1"):
        source = args.performance / directory
        if source.is_dir():
            for name in ("summary.json", "profile.csv", "command.json", "config_snapshot.yaml"):
                if (source / name).is_file():
                    copy_file(source / name, package / "05_performance" / directory / name)
    for name in ("compileall.txt", "focused_pytest.xml", "full_pytest.xml", "focused_stdout.txt", "full_stdout.txt"):
        copy_file(args.tests_dir / name, package / "06_tests" / name)
    try:
        raw_matrix_path = args.matrix_root.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        raw_matrix_path = "external_diagnostic_only; compact critical fixtures copied below"
    write_json(
        package / "00_provenance/environment.json",
        {
            "schema": "vdp_g2_environment_v1",
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "cpu_count": os.cpu_count(),
            "authoritative_lane": "CPU_float64_B1",
        },
    )
    write_json(
        package / "00_provenance/commands.json",
        {
            "schema": "vdp_g2_commands_v1",
            "commands": [
                "conda run -n py11 python experiments/audit_vdp_lossless_operator_matrix_20260815.py --help",
                "conda run -n py11 python experiments/audit_g1_owner_interventions_20260815.py --help",
                "conda run -n py11 python experiments/audit_g1_terminal_owner_interventions_20260815.py --help",
                "conda run -n py11 python experiments/export_g2_blackbox_coefficients.py --help",
                "python experiments/independent_g2_exact_oracle.py --help",
                "conda run -n py11 python experiments/audit_g2_checkpoint_resume_20260815.py --help",
                "conda run -n py11 python experiments/run_vdp_g2_fresh_matrix_20260815.py --help",
                "conda run -n py11 python experiments/summarize_vdp_g2_matrix_20260815.py --help",
                "conda run -n py11 python experiments/benchmark_g2_cpu_cuda_20260815.py --help",
                "conda run -n py11 python -m compileall -q src experiments tests",
                "conda run -n py11 pytest -q [focused files] --junitxml=focused_pytest.xml",
                "conda run -n py11 pytest -q --junitxml=full_pytest.xml",
                "python experiments/verify_vdp_g2_evidence_20260815.py --package <package>",
            ],
        },
    )
    write_json(
        package / "00_provenance/provenance.json",
        {
            "schema": "vdp_g2_provenance_v1",
            "base_sha": "771948ef7592d5b5c81e35e36ba4aa067674821e",
            "previous_scientific_sha": "8ac2962bf691dd81ae5d06a9ea146bb011b7ec42",
            "branch": git("branch", "--show-current"),
            "head_at_build": git("rev-parse", "HEAD"),
            "source_diff_sha256": hashlib.sha256(
                subprocess.run(["git", "diff", "HEAD", "--binary"], cwd=ROOT, check=True, capture_output=True).stdout
            ).hexdigest(),
            "source_snapshot_files": copied,
            "raw_matrix_repository_path": raw_matrix_path,
            "raw_matrix_note": "full compressed per-request traces are tracked outside this compact package; all critical raw bounds are duplicated in fixed_curve.csv.gz",
        },
    )
    write_json(
        package / "07_fresh_clone/acceptance.json",
        {
            "schema": "vdp_g2_fresh_clone_acceptance_v1",
            "status": "PENDING_SCIENTIFIC_SHA",
            "git_status_porcelain_empty": False,
            "scientific_sha": None,
        },
    )


def finalize(
    package: Path,
    *,
    stage: str,
    scientific_sha: str | None,
    attestation_sha: str | None,
    tamper_required: bool,
) -> dict[str, Any]:
    package = package.resolve()
    if not package.is_dir():
        raise FileNotFoundError(package)
    if stage == "scientific_precommit":
        write_json(
            package / "07_fresh_clone/acceptance.json",
            {
                "schema": "vdp_g2_fresh_clone_acceptance_v1",
                "status": "PENDING_SCIENTIFIC_SHA",
                "git_status_porcelain_empty": False,
                "scientific_sha": None,
            },
        )
    excluded = {package / "manifest.json", package / "SHA256SUMS"}
    files = sorted(path for path in package.rglob("*") if path.is_file() and path not in excluded)
    records = {
        path.relative_to(package).as_posix(): {"bytes": path.stat().st_size, "sha256": sha(path)}
        for path in files
    }
    total = sum(row["bytes"] for row in records.values())
    manifest = {
        "schema": "vdp_g2_shared_column_evidence_package_v1",
        "stage": stage,
        "conclusion": PARTIAL,
        "total_cause_conclusion": TOTAL,
        "candidate": "normalized_insertion_bounded_shared_source_o4_g2",
        "scientific_sha": scientific_sha,
        "attestation_sha": attestation_sha,
        "tamper_required": tamper_required,
        "file_count": len(records),
        "total_bytes_excluding_manifest_and_sums": total,
        "under_25_mib": total < 25 * 1024 * 1024,
        "files": records,
    }
    write_json(package / "manifest.json", manifest)
    (package / "SHA256SUMS").write_text(
        "".join(f"{row['sha256']}  {relative}\n" for relative, row in records.items()),
        encoding="utf-8",
    )
    if not manifest["under_25_mib"]:
        raise ValueError("evidence package exceeds 25 MiB")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--finalize-only", action="store_true")
    parser.add_argument("--gate-a", type=Path)
    parser.add_argument("--gate-b-fixed", type=Path)
    parser.add_argument("--gate-b-terminal", type=Path)
    parser.add_argument("--blackbox", type=Path)
    parser.add_argument("--oracle", type=Path)
    parser.add_argument("--resume-audit", type=Path)
    parser.add_argument("--matrix-root", type=Path)
    parser.add_argument("--summary-dir", type=Path)
    parser.add_argument("--performance", type=Path)
    parser.add_argument("--tests-dir", type=Path)
    parser.add_argument("--stage", choices=("scientific_precommit", "attestation"), default="scientific_precommit")
    parser.add_argument("--scientific-sha")
    parser.add_argument("--attestation-sha")
    parser.add_argument("--tamper-required", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.finalize_only:
        required = (
            args.gate_a, args.gate_b_fixed, args.gate_b_terminal, args.blackbox,
            args.oracle, args.resume_audit, args.matrix_root, args.summary_dir,
            args.performance, args.tests_dir,
        )
        if any(value is None for value in required):
            raise ValueError("initial build requires every evidence input")
        initial_build(args)
    result = finalize(
        args.package,
        stage=args.stage,
        scientific_sha=args.scientific_sha,
        attestation_sha=args.attestation_sha,
        tamper_required=args.tamper_required,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
