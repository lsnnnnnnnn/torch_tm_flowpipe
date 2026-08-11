#!/usr/bin/env python3
"""Fail-closed finalization for the three-tool evidence package.

Command completion and scientific truth are deliberately separate.  A runner
exit code validates only the runner envelope.  Every scientific outcome must
be parsed from a hashed artifact under a serializable contract in config.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from torch_tm_flowpipe.evidence_verification import (
    VerificationClaim,
    classify_private_path_matches,
    derive_command_claim,
    derive_scientific_summary_claim,
    validate_verification_document,
    verification_document,
)


REQUIRED_RUNNER_FILES = {
    "config.json",
    "summary.json",
    "stdout.log",
    "stderr.log",
    "command.txt",
    "exit_code.txt",
    "started_at.txt",
    "finished_at.txt",
    "timing.json",
    "artifact_index.json",
}

OUTCOME_DIMENSIONS = (
    "evidence_package_status",
    "raw_remainder_root_cause_status",
    "flowstar_torch_fixed_schedule_status",
    "diffreach_torch_operator_status",
    "diffreach_torch_full_horizon_status",
    "carry_semantics_status",
    "single_fix_status",
    "performance_status",
    "tightness_status",
    "formal_scope",
    "empirical_scope",
)

# This is a closed vocabulary.  Pending/partial values are retained because a
# fail-closed research run must be able to report nonclosure honestly.
OUTCOME_TAXONOMY = {
    "evidence_package_status": frozenset(
        {
            "EVIDENCE_PACKAGE_MISSING_STOP",
            "EVIDENCE_PACKAGE_REBUILT_PENDING_TRACKING",
            "EVIDENCE_PACKAGE_REBUILT_AND_TRACKED",
            "EVIDENCE_PACKAGE_SOURCE_MISMATCH_STOP",
        }
    ),
    "raw_remainder_root_cause_status": frozenset(
        {
            "RAW_REMAINDER_ROOT_CAUSE_CLOSED",
            "RAW_REMAINDER_ROOT_CAUSE_PENDING",
        }
    ),
    "flowstar_torch_fixed_schedule_status": frozenset(
        {
            "FLOWSTAR_TORCH_NATIVE_FULL_HORIZON_PAIRWISE_PARTIAL",
            "FLOWSTAR_TORCH_FIXED_SCHEDULE_T10_BOTH_COMPLETE",
            "FLOWSTAR_TORCH_FIXED_SCHEDULE_COMMON_PREFIX_ONLY",
            "FLOWSTAR_TORCH_FIXED_SCHEDULE_ENVIRONMENT_BLOCKED",
        }
    ),
    "diffreach_torch_operator_status": frozenset(
        {
            "DIFFREACH_TORCH_DR7_OPERATOR_EQUIVALENCE_CLOSED",
            "DIFFREACH_TORCH_DR7_OPERATOR_EQUIVALENCE_DIVERGED",
            "DIFFREACH_TORCH_DR7_OPERATOR_ENVIRONMENT_BLOCKED",
        }
    ),
    "diffreach_torch_full_horizon_status": frozenset(
        {
            "DIFFREACH_TORCH_DR7_FULL_HORIZON_PAIRWISE_PENDING",
            "DIFFREACH_TORCH_DR7_FULL_HORIZON_BIT_EXACT",
            "DIFFREACH_TORCH_DR7_FULL_HORIZON_ULP_BOUNDED",
            "DIFFREACH_TORCH_DR7_FULL_HORIZON_DIVERGED",
            "DIFFREACH_TORCH_DR7_FULL_HORIZON_ENVIRONMENT_BLOCKED",
        }
    ),
    "carry_semantics_status": frozenset(
        {
            "CARRY_COORDINATE_CONTRACT_BUG",
            "CARRY_REMAINDER_DOUBLE_COUNT",
            "CARRY_NONLINEAR_COMPOSITION_INTERVALIZATION",
            "CARRY_MISSING_SYMBOLIC_SEMANTICS",
            "CARRY_BINARY64_ROUNDOFF_ONLY",
            "CARRY_ROOT_CAUSE_MIXED_OR_UNRESOLVED",
        }
    ),
    "single_fix_status": frozenset(
        {
            "SINGLE_IMPLEMENTATION_FIX_PROMOTED",
            "SINGLE_IMPLEMENTATION_FIX_REJECTED",
            "NO_FIX_AUTHORIZED",
        }
    ),
    "performance_status": frozenset(
        {
            "MATCHED_WORKLOAD_TIMING_AVAILABLE",
            "MATCHED_WORKLOAD_TIMING_UNAVAILABLE",
        }
    ),
    "tightness_status": frozenset(
        {
            "COMMON_PREFIX_TIGHTNESS_AVAILABLE",
            "TIGHTNESS_COMPARISON_UNAVAILABLE",
        }
    ),
    "formal_scope": frozenset(
        {
            "no_new_formal_cross_tool_claim",
            "flowstar_formal_torch_empirical_common_prefix_only",
        }
    ),
    "empirical_scope": frozenset(
        {
            "one_step_operator_and_separate_native_capability_only",
            "fixed_schedule_common_prefix_and_full_horizon_dr7",
            "environment_blocked_no_cross_tool_closure",
        }
    ),
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _load_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON token {token} in {path}")
        ),
    )
    if not isinstance(value, Mapping):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def _field(value: Mapping[str, Any], dotted_path: str) -> Any:
    current: Any = value
    for part in dotted_path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise RuntimeError(f"required field is missing: {dotted_path}")
        current = current[part]
    return current


def _safe_file(root: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute():
        raise RuntimeError(f"package path must be relative: {relative}")
    resolved = (root / candidate).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise RuntimeError(f"package path escapes root: {relative}")
    if not resolved.is_file():
        raise RuntimeError(f"required evidence path is missing: {relative}")
    return resolved


def _reject_nonfinite_json(path: Path) -> None:
    _load_json(path)


def _runner_directories(run_root: Path) -> list[Path]:
    return sorted(
        path
        for path in run_root.rglob("*")
        if path.is_dir()
        and (path / "config.json").is_file()
        and (path / "command.txt").is_file()
    )


def _expected_artifact_rows(runner: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": path.relative_to(runner).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha(path),
        }
        for path in sorted(runner.rglob("*"))
        if path.is_file() and path.name != "artifact_index.json"
    ]


def _validate_artifact_index(runner: Path) -> None:
    index = _load_json(runner / "artifact_index.json")
    if index.get("schema") != "torch_tm_flowpipe_evidence_artifact_index_v1":
        raise RuntimeError(f"artifact index schema mismatch: {runner}")
    if index.get("root") != ".":
        raise RuntimeError(f"artifact index root mismatch: {runner}")
    rows = index.get("files")
    if rows != _expected_artifact_rows(runner):
        raise RuntimeError(f"artifact index coverage or SHA mismatch: {runner}")


def _validate_runner_envelope(
    runner: Path,
) -> tuple[Mapping[str, Any], Mapping[str, Any], int]:
    config = _load_json(runner / "config.json")
    summary = _load_json(runner / "summary.json")
    timing = _load_json(runner / "timing.json")
    if config.get("schema") != "torch_tm_flowpipe_evidence_runner_config_v1":
        raise RuntimeError(f"runner config schema mismatch: {runner}")
    if summary.get("schema") != "torch_tm_flowpipe_evidence_runner_summary_v1":
        raise RuntimeError(f"runner summary schema mismatch: {runner}")
    if timing.get("schema") != "torch_tm_flowpipe_evidence_timing_v1":
        raise RuntimeError(f"runner timing schema mismatch: {runner}")
    try:
        exit_code = int((runner / "exit_code.txt").read_text().strip())
    except ValueError as exc:
        raise RuntimeError(f"invalid runner exit code: {runner}") from exc
    expected = tuple(int(value) for value in config.get("expected_exit_codes", (0,)))
    expected_status = (
        "pass"
        if exit_code == 0
        else "qualified_expected_nonzero"
        if exit_code in expected
        else "fail"
    )
    checks = {
        "name": config.get("runner_name"),
        "status": expected_status,
        "exit_code": exit_code,
        "source_commit": config.get("source_commit"),
        "config_sha256": _sha(runner / "config.json"),
        "eligibility_status": config.get("eligibility_status"),
    }
    for field, expected_value in checks.items():
        if summary.get(field) != expected_value:
            raise RuntimeError(
                f"runner summary field mismatch for {field}: {runner}"
            )
    artifact_summary = runner / "artifacts" / "summary.json"
    expected_artifact = (
        {"path": "artifacts/summary.json", "sha256": _sha(artifact_summary)}
        if artifact_summary.is_file()
        else None
    )
    if summary.get("artifact_summary") != expected_artifact:
        raise RuntimeError(f"runner artifact summary SHA mismatch: {runner}")
    if not (runner / "started_at.txt").read_text().strip():
        raise RuntimeError(f"runner start timestamp is missing: {runner}")
    if not (runner / "finished_at.txt").read_text().strip():
        raise RuntimeError(f"runner finish timestamp is missing: {runner}")
    _validate_artifact_index(runner)
    return config, summary, exit_code


def _contract_list(config: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    raw = config.get("scientific_summary")
    if raw is None:
        return ()
    values = raw if isinstance(raw, list) else [raw]
    if not all(isinstance(value, Mapping) for value in values):
        raise RuntimeError("scientific_summary contract must be an object or list")
    return tuple(values)


def _validate_report_assertions(
    runner: Path,
    scientific: Mapping[str, Any],
    assertions: Any,
) -> None:
    if assertions is None:
        return
    if not isinstance(assertions, list):
        raise RuntimeError("report_assertions must be a list")
    for assertion in assertions:
        if not isinstance(assertion, Mapping):
            raise RuntimeError("report assertion must be an object")
        report = _safe_file(runner, str(assertion["path"]))
        match = re.search(str(assertion["pattern"]), report.read_text(encoding="utf-8"))
        if match is None or "value" not in match.groupdict():
            raise RuntimeError(f"report value not found: {report}")
        source_value = _field(scientific, str(assertion["summary_field"]))
        kind = str(assertion.get("type", "string"))
        report_value: Any = match.group("value")
        if kind == "float":
            report_value = float(report_value)
            source_value = float(source_value)
        elif kind == "int":
            report_value = int(report_value)
            source_value = int(source_value)
        elif kind != "string":
            raise RuntimeError(f"unsupported report assertion type: {kind}")
        if report_value != source_value:
            raise RuntimeError(
                f"report/JSON mismatch for {assertion['summary_field']}: "
                f"{report_value!r} != {source_value!r}"
            )


def _validate_scientific_semantics(
    *,
    dimension: str | None,
    outcome: str,
    scientific: Mapping[str, Any],
    profile: str | None,
    exit_code: int,
) -> None:
    if dimension is not None:
        if dimension not in OUTCOME_TAXONOMY:
            raise RuntimeError(f"unknown outcome dimension: {dimension}")
        if outcome not in OUTCOME_TAXONOMY[dimension]:
            raise RuntimeError(f"outcome {outcome!r} is invalid for {dimension}")

    if outcome == "DIFFREACH_TORCH_DR7_OPERATOR_EQUIVALENCE_CLOSED":
        required = {
            "scope": "one_step",
            "batch_size": 1,
            "operator_equality": True,
            "initial_mask_equality": True,
            "later_mask_equality": True,
            "endpoint_tube_equality": True,
            "comparison.kind": "cross_tool",
            "comparison.left_tool": "diffreach",
            "comparison.right_tool": "torch",
        }
        for field, expected in required.items():
            if _field(scientific, field) != expected:
                raise RuntimeError(f"operator closure prerequisite failed: {field}")

    if outcome in {
        "DIFFREACH_TORCH_DR7_FULL_HORIZON_BIT_EXACT",
        "DIFFREACH_TORCH_DR7_FULL_HORIZON_ULP_BOUNDED",
    }:
        required = {
            "scope": "full_horizon",
            "batch_size": 64,
            "steps": 1000,
            "initial_masks_all_true": True,
            "initial_mask_equality": True,
            "later_mask_equality": True,
            "j_phi_equality": True,
            "endpoint_tube_equality": True,
            "no_hidden_fallback": True,
            "comparison.kind": "cross_tool",
            "comparison.left_tool": "diffreach",
            "comparison.right_tool": "torch",
        }
        for field, expected in required.items():
            if _field(scientific, field) != expected:
                raise RuntimeError(f"full-horizon closure prerequisite failed: {field}")
        if outcome.endswith("ULP_BOUNDED"):
            if int(_field(scientific, "max_ulp")) > int(
                _field(scientific, "preregistered_max_ulp")
            ):
                raise RuntimeError("full-horizon ULP bound exceeded")

    if profile == "bridge_g3":
        if _field(scientific, "max_gate") != "G3":
            raise RuntimeError("G3 semantic profile used for another gate")
        if exit_code == 1:
            if outcome != "FIXED_SUPPORT_BRIDGE_BLOCKED":
                raise RuntimeError("G3 exit 1 requires blocked outcome")
            cells = _field(scientific, "cells")
            if not isinstance(cells, list):
                raise RuntimeError("G3 cells must be a list")
            reasons = [
                failure.get("reason")
                for cell in cells
                if isinstance(cell, Mapping)
                and isinstance((failure := cell.get("first_failure")), Mapping)
            ]
            if not reasons or any(
                reason != "initial_remainder_inclusion_failed" for reason in reasons
            ):
                raise RuntimeError("G3 failure reason is not preregistered")

    if profile == "true_clone":
        if outcome != "TRUE_FRESH_CLONE_PASS":
            raise RuntimeError("true-clone success profile requires pass outcome")
        clone_root = Path(str(_field(scientific, "clone_root"))).resolve()
        source_root = Path(str(_field(scientific, "source_worktree"))).resolve()
        if clone_root == source_root or clone_root.is_relative_to(source_root):
            raise RuntimeError("true-clone marker was produced from source worktree")
        if not bool(_field(scientific, "origin_clone")):
            raise RuntimeError("true-clone marker lacks origin clone evidence")
        if _field(scientific, "temporary_root_method") != "tempfile.mkdtemp":
            raise RuntimeError("true-clone marker lacks temporary-root evidence")
        if _field(scientific, "origin") != _field(scientific, "cloned_origin"):
            raise RuntimeError("true clone origin identity mismatch")
        if _field(scientific, "checked_out_sha") != _field(
            scientific, "expected_sha"
        ):
            raise RuntimeError("true clone checked out the wrong SHA")
        if _field(scientific, "clean_tree") is not True:
            raise RuntimeError("true clone is not clean")
        commands = _field(scientific, "commands")
        if not isinstance(commands, list):
            raise RuntimeError("true clone command evidence must be a list")
        by_name = {
            str(row.get("name")): row
            for row in commands
            if isinstance(row, Mapping)
        }
        for name in ("clone_origin", "checkout_exact_sha", "install"):
            if name not in by_name or by_name[name].get("exit_code") != 0:
                raise RuntimeError(f"true clone command did not pass: {name}")
        clone_command = by_name["clone_origin"].get("command")
        if not isinstance(clone_command, list) or clone_command[:3] != [
            "git",
            "clone",
            "--no-local",
        ]:
            raise RuntimeError("true clone command is not an origin clone")


def _derive_scientific_contract(
    runner: Path,
    relative: str,
    contract: Mapping[str, Any],
    exit_code: int,
    run_root: Path,
) -> tuple[VerificationClaim, str | None, str | None]:
    path = _safe_file(runner, str(contract["path"]))
    dimension = (
        None
        if contract.get("outcome_dimension") is None
        else str(contract["outcome_dimension"])
    )
    allowed = tuple(str(value) for value in contract.get("allowed_outcomes", ()))
    if not allowed:
        raise RuntimeError(f"scientific contract has no allowed outcomes: {relative}")
    if dimension is not None and not set(allowed).issubset(OUTCOME_TAXONOMY.get(dimension, ())):
        raise RuntimeError(f"scientific contract contains invalid outcomes: {relative}")
    result = derive_scientific_summary_claim(
        str(contract.get("claim_id", relative.replace("/", ".") + ".scientific")),
        path,
        expected_schema=(
            None if contract.get("schema") is None else str(contract["schema"])
        ),
        outcome_field=str(contract.get("outcome_field", "outcome")),
        allowed_outcomes=allowed,
        required_fields=dict(contract.get("required_fields", {})),
        scope=str(contract.get("scope", f"scientific artifact {relative}")),
        repository_root=run_root,
        limitations=tuple(str(value) for value in contract.get("limitations", ())),
    )
    if result.claim.status != "pass" or result.outcome is None:
        raise RuntimeError(
            f"scientific summary validation failed for {relative}: "
            + "; ".join(result.claim.limitations)
        )
    scientific = _load_json(path)
    for required in contract.get("required_paths", ()):
        _safe_file(runner, str(required))
    source_fields = contract.get("source_sha256_fields", {})
    if not isinstance(source_fields, Mapping):
        raise RuntimeError("source_sha256_fields must be an object")
    for source_path, field in source_fields.items():
        source = _safe_file(runner, str(source_path))
        if _field(scientific, str(field)) != _sha(source):
            raise RuntimeError(f"scientific source SHA mismatch: {source_path}")
    _validate_report_assertions(runner, scientific, contract.get("report_assertions"))
    _validate_scientific_semantics(
        dimension=dimension,
        outcome=result.outcome,
        scientific=scientific,
        profile=(
            None
            if contract.get("semantic_profile") is None
            else str(contract["semantic_profile"])
        ),
        exit_code=exit_code,
    )
    return result.claim, dimension, result.outcome


def validate_checksum_coverage(run_root: Path) -> None:
    """Verify that SHA256SUMS exactly covers every other package file."""

    root = Path(run_root).resolve()
    checksum_path = _safe_file(root, "SHA256SUMS")
    actual: dict[str, str] = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        digest, separator, relative = line.partition("  ")
        if not separator or relative in actual:
            raise RuntimeError("invalid or duplicate SHA256SUMS entry")
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise RuntimeError(f"invalid SHA256 digest for {relative}")
        actual[relative] = digest
    expected = {
        path.relative_to(root).as_posix(): _sha(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path != checksum_path
    }
    if actual != expected:
        raise RuntimeError("SHA256SUMS coverage or digest mismatch")
    for relative, digest in actual.items():
        if _sha(_safe_file(root, relative)) != digest:
            raise RuntimeError(f"SHA256SUMS verification failed: {relative}")


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    run_root = args.run_root.resolve()
    if not run_root.is_dir():
        raise FileNotFoundError(run_root)
    runners = _runner_directories(run_root)
    if not runners:
        raise RuntimeError("evidence package contains no runner protocol directories")
    claims: list[VerificationClaim] = []
    outcomes: dict[str, str] = {}
    source_commits: set[str] = set()
    for runner in runners:
        missing = REQUIRED_RUNNER_FILES - {path.name for path in runner.iterdir()}
        if missing:
            raise RuntimeError(
                f"incomplete runner protocol {runner}: {sorted(missing)}"
            )
        relative = runner.relative_to(run_root).as_posix()
        runner_config, _runner_summary, exit_code = _validate_runner_envelope(runner)
        source_commit = str(runner_config.get("source_commit", ""))
        if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
            raise RuntimeError(f"invalid runner source commit: {relative}")
        source_commits.add(source_commit)
        expected_exit_codes = tuple(
            int(value) for value in runner_config.get("expected_exit_codes", [0])
        )

        def evaluate_exit(
            _stdout: str,
            _stderr: str,
            actual_exit: int,
            *,
            expected: tuple[int, ...] = expected_exit_codes,
        ) -> tuple[str, Sequence[str]]:
            if actual_exit not in expected:
                return "fail", (f"unexpected exit code {actual_exit}",)
            if actual_exit == 0:
                return "pass", ()
            return "qualified", (
                f"expected fail-closed/noncompletion exit code {actual_exit}",
            )

        command_claim = derive_command_claim(
            relative.replace("/", "."),
            runner,
            scope=f"runner protocol {relative}",
            repository_root=run_root,
            evaluator=evaluate_exit,
        )
        if command_claim.status not in {"pass", "qualified"}:
            raise RuntimeError(f"runner command claim failed: {relative}")
        claims.append(command_claim)
        for contract in _contract_list(runner_config):
            scientific_claim, dimension, outcome = _derive_scientific_contract(
                runner, relative, contract, exit_code, run_root
            )
            claims.append(scientific_claim)
            if dimension is not None and outcome is not None:
                previous = outcomes.setdefault(dimension, outcome)
                if previous != outcome:
                    raise RuntimeError(
                        f"conflicting source-derived outcomes for {dimension}"
                    )

    if len(source_commits) != 1:
        raise RuntimeError("runner source commits disagree")
    require_complete = bool(getattr(args, "require_complete_outcomes", False))
    if require_complete and set(outcomes) != set(OUTCOME_DIMENSIONS):
        missing = sorted(set(OUTCOME_DIMENSIONS) - set(outcomes))
        extra = sorted(set(outcomes) - set(OUTCOME_DIMENSIONS))
        raise RuntimeError(
            f"source-derived outcome registry is incomplete; missing={missing}, extra={extra}"
        )

    verification = verification_document(claims)
    _write_json(run_root / "verification.json", verification)
    validate_verification_document(verification, source_root=run_root)

    for path in run_root.rglob("*.json"):
        _reject_nonfinite_json(path)
    path_scan = classify_private_path_matches(
        [
            path
            for path in run_root.rglob("*")
            if path.is_file()
            and path.suffix in {".json", ".txt", ".log", ".csv", ".tsv"}
        ],
        scan_root=run_root,
        private_prefix="/srv/local/shengenli",
        provenance_only=[
            path.relative_to(run_root).as_posix()
            for path in run_root.rglob("*")
            if path.is_file()
            and (
                path.name
                in {
                    "command.txt",
                    "command.json",
                    "stdout.log",
                    "stderr.log",
                    "config.json",
                    "timing.json",
                    "verification.json",
                    "terminal_state.json",
                }
                or "03_native_flowstar/scalar_affine_gate/"
                in path.relative_to(run_root).as_posix()
                or path.relative_to(run_root).as_posix()
                == "00_environment/probe/artifacts/run/summary.json"
                or path.relative_to(run_root).as_posix()
                == "04_native_diffreach/official_vdp/artifacts/run/summary.json"
            )
        ],
    )
    if path_scan["status"] == "fail":
        raise RuntimeError("unclassified private path in evidence package")

    source_commit = next(iter(source_commits))
    manifest = {
        "schema": "three_tool_full_horizon_pairwise_carry_package_v2",
        "run_id": run_root.name,
        "tested_source_sha": source_commit,
        "package_commit_sha": None,
        "delivery_audit_sha": None,
        "package_root": ".",
        "runner_count": len(runners),
        "runners": [path.relative_to(run_root).as_posix() for path in runners],
        "outcome_registry": outcomes,
        "private_path_audit": path_scan,
        "verification_sha256": _sha(run_root / "verification.json"),
    }
    _write_json(run_root / "manifest.json", manifest)

    checksum_path = run_root / "SHA256SUMS"
    files = sorted(
        path
        for path in run_root.rglob("*")
        if path.is_file() and path != checksum_path
    )
    checksum_path.write_text(
        "".join(
            f"{_sha(path)}  {path.relative_to(run_root).as_posix()}\n"
            for path in files
        ),
        encoding="utf-8",
    )
    validate_checksum_coverage(run_root)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--require-complete-outcomes", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    print(json.dumps(finalize(parse_args(argv)), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
