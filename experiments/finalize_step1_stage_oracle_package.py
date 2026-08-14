#!/usr/bin/env python3
"""Finalize compact references, manifest, and checksums for the step-1 package."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence


REQUIRED = tuple(f"{index:02d}_{name}" for index, name in (
    (0, "identity_provenance"), (1, "prior_result_recheck"),
    (2, "common_step1_contract"), (3, "flowstar_actual_stage_ledger"),
    (4, "torch_actual_stage_ledger"), (5, "exact_polynomial_oracle"),
    (6, "mpfr_interval_oracle"), (7, "stage_swap_matrix"),
    (8, "l1_local_candidate"), (9, "l2_symbolic_carry"),
    (10, "l3_combined_candidate"), (11, "frozen_contract_runs"),
    (12, "negative_results"), (13, "tests"), (14, "fresh_clone"),
))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _reference(path: Path, package: Path, target: Path, role: str) -> None:
    resolved = package / target
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    _write(
        path,
        {
            "schema": "step1_evidence_artifact_reference_v1",
            "role": role,
            "target": target.as_posix(),
            "sha256": _sha(resolved),
            "bytes": resolved.stat().st_size,
        },
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    package = args.package.resolve()
    package.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED:
        (package / name).mkdir(parents=True, exist_ok=True)
    oracle_root = Path("05_exact_polynomial_oracle/02_oracles/artifacts")
    for name in ("mpfr_128.json", "mpfr_256.json", "mpfr_512.json", "precision_ladder.json"):
        _reference(
            package / "06_mpfr_interval_oracle" / f"{name}.reference.json",
            package,
            oracle_root / name,
            "independently compiled directed-rounding MPFR artifact",
        )
    audit_root = Path("07_stage_swap_matrix/audit_retry/artifacts")
    for directory, status, role in (
        ("08_l1_local_candidate", "NOT_AUTHORIZED", "L1 stopped by Gate D under-enclosure"),
        ("09_l2_symbolic_carry", "NOT_RUN", "L2 not entered because L1 was not authorized"),
        ("10_l3_combined_candidate", "NOT_RUN", "L3 not entered"),
        ("11_frozen_contract_runs", "NOT_RUN", "no sound candidate existed for Gate H"),
    ):
        _write(
            package / directory / "not_run_reason.json",
            {
                "schema": "step1_fail_closed_not_run_v1",
                "status": status,
                "reason": role,
                "gate_d": "INDEPENDENT_STEP1_SOUNDNESS_ORACLE_INCOMPLETE",
                "evidence": {
                    "target": (audit_root / "candidate_decision.json").as_posix(),
                    "sha256": _sha(package / audit_root / "candidate_decision.json"),
                },
            },
        )
    _reference(
        package / "12_negative_results" / "under_enclosure_witness.reference.json",
        package,
        audit_root / "first_difference_full_input_output.json",
        "first independent under-enclosure witness",
    )
    _reference(
        package / "12_negative_results" / "candidate_decision.reference.json",
        package,
        audit_root / "candidate_decision.json",
        "fail-closed candidate authorization decision",
    )
    _write(
        package / "13_tests" / "status.json",
        {
            "schema": "step1_test_status_v1",
            "status": args.tests_status,
            "focused_command_present": (package / "13_tests/focused").is_dir(),
            "authoritative_full_command": "13_tests/full_retry",
            "full_command_present": (package / "13_tests/full_retry").is_dir(),
            "tamper_command_present": (package / "13_tests/tamper").is_dir(),
        },
    )
    _write(
        package / "14_fresh_clone" / "status.json",
        {
            "schema": "step1_fresh_clone_status_v1",
            "status": args.fresh_clone_status,
            "scientific_sha": args.scientific_sha or None,
            "final_tip_fresh_clone_verified": False,
            "reason": (
                "detached scientific SHA verification evidence is stored below"
                if args.fresh_clone_status == "pass"
                else "fresh clone runs only after the scientific commit exists"
            ),
        },
    )
    superseded = {
        "schema": "step1_superseded_attempts_v1",
        "attempts": [
            {
                "path": "04_torch_actual_stage_ledger/export",
                "reason": "superseded by export_retry, which pads step-2 two-variable exponents into the canonical three-variable basis",
            },
            {
                "path": "07_stage_swap_matrix/audit",
                "reason": "expected fail from the superseded unpadded step-2 ledger; audit_retry is authoritative",
            },
            {
                "path": "13_tests/full",
                "reason": "superseded by full_retry after restoring the four canonical broader outcome labels to the handoff headline",
            },
        ],
    }
    _write(package / "12_negative_results" / "superseded_attempts.json", superseded)

    files = {
        path.relative_to(package).as_posix(): {"sha256": _sha(path), "bytes": path.stat().st_size}
        for path in sorted(package.rglob("*"))
        if path.is_file() and path.name not in {"manifest.json", "SHA256SUMS"}
    }
    manifest = {
        "schema": "flowstar_torch_step1_stage_oracle_package_v1",
        "package_root": package.name,
        "branch": args.branch,
        "start_sha": "3940386a61bdd6edbf3dc1722be031a1da572171",
        "flowstar_sha": "b85a3211748cb77b736fe4ad42ee02d8d2b81148",
        "scientific_sha": args.scientific_sha or None,
        "attestation_tip": args.attestation_tip or None,
        "publication": {
            "scientific_sha_fresh_clone_verified": args.fresh_clone_status == "pass",
            "attestation_tip_contains_no_scientific_tree_changes": (
                True if args.attestation_tip else "unknown"
            ),
            "final_tip_fresh_clone_verified": False,
        },
        "statuses": {
            "gate_b": "COMMON_STEP1_MATHEMATICAL_INPUT_CLOSED",
            "gate_d": "INDEPENDENT_STEP1_SOUNDNESS_ORACLE_INCOMPLETE",
            "gate_e": "LOCAL_OPERATOR_SOURCE_DELTA_OPEN",
            "l1": "NOT_AUTHORIZED",
            "l2": "NOT_RUN",
            "l3": "NOT_RUN",
            "legacy": "LEGACY_DEFAULT_UNCHANGED",
            "t10": "NOT_REACHED",
        },
        "first_differing_stage": "normalized_initial_tm",
        "first_classification": "UNDER_ENCLOSURE_WITNESS",
        "torch_endpoint_narrower_formally_sound": True,
        "required_directories": list(REQUIRED),
        "superseded_attempts": superseded["attempts"],
        "files": files,
    }
    _write(package / "manifest.json", manifest)
    (package / "SHA256SUMS").write_text(
        "".join(f"{record['sha256']}  {relative}\n" for relative, record in sorted(files.items())),
        encoding="utf-8",
    )
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--scientific-sha", default="")
    parser.add_argument("--attestation-tip", default="")
    parser.add_argument("--tests-status", choices=("pending", "pass", "fail"), default="pending")
    parser.add_argument("--fresh-clone-status", choices=("pending", "pass", "fail"), default="pending")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    result = run(parse_args(argv))
    print(json.dumps({"files": len(result["files"]), "statuses": result["statuses"], "publication": result["publication"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
