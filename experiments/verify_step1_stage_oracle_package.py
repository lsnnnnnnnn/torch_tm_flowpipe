#!/usr/bin/env python3
"""Recompute the negative-result step-1 package from raw ledgers and oracles."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


REQUIRED_DIRECTORIES = tuple(f"{index:02d}_{name}" for index, name in (
    (0, "identity_provenance"),
    (1, "prior_result_recheck"),
    (2, "common_step1_contract"),
    (3, "flowstar_actual_stage_ledger"),
    (4, "torch_actual_stage_ledger"),
    (5, "exact_polynomial_oracle"),
    (6, "mpfr_interval_oracle"),
    (7, "stage_swap_matrix"),
    (8, "l1_local_candidate"),
    (9, "l2_symbolic_carry"),
    (10, "l3_combined_candidate"),
    (11, "frozen_contract_runs"),
    (12, "negative_results"),
    (13, "tests"),
    (14, "fresh_clone"),
))


class VerificationError(ValueError):
    pass


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise VerificationError(f"invalid JSON {path}: {exc}") from exc


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise VerificationError(f"cannot load verifier dependency {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def _verify_checksums(package: Path) -> dict[str, Any]:
    sums = package / "SHA256SUMS"
    if not sums.is_file():
        raise VerificationError("missing SHA256SUMS")
    entries: dict[str, str] = {}
    for line_number, line in enumerate(sums.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            digest, relative = line.split("  ", 1)
        except ValueError as exc:
            raise VerificationError(f"malformed SHA256SUMS line {line_number}") from exc
        if relative in entries or len(digest) != 64:
            raise VerificationError(f"duplicate or malformed checksum at line {line_number}")
        path = package / relative
        if not path.is_file() or _sha(path) != digest:
            raise VerificationError(f"checksum mismatch: {relative}")
        entries[relative] = digest
    expected = {
        path.relative_to(package).as_posix()
        for path in package.rglob("*")
        if path.is_file() and path.name not in {"SHA256SUMS", "manifest.json"}
    }
    _assert(set(entries) == expected, "SHA256SUMS coverage mismatch")
    return {"entry_count": len(entries), "coverage_exact": True}


def _verify_manifest(package: Path) -> dict[str, Any]:
    manifest = _json(package / "manifest.json")
    _assert(manifest.get("schema") == "flowstar_torch_step1_stage_oracle_package_v1", "manifest schema mismatch")
    files = manifest.get("files")
    _assert(isinstance(files, dict), "manifest files missing")
    expected = {
        path.relative_to(package).as_posix()
        for path in package.rglob("*")
        if path.is_file() and path.name not in {"SHA256SUMS", "manifest.json"}
    }
    _assert(set(files) == expected, "manifest file coverage mismatch")
    for relative, record in files.items():
        path = package / relative
        _assert(record["sha256"] == _sha(path), f"manifest hash mismatch: {relative}")
        _assert(int(record["bytes"]) == path.stat().st_size, f"manifest size mismatch: {relative}")
    return manifest


def _selected_paths(package: Path) -> dict[str, Path]:
    return {
        "flow_ledger": package / "03_flowstar_actual_stage_ledger/03_process/artifacts/stage_ledger.json",
        "flow_summary": package / "03_flowstar_actual_stage_ledger/03_process/artifacts/summary.json",
        "flow_trace": package / "03_flowstar_actual_stage_ledger/02_actual_run/artifacts/instrumented_trace.jsonl",
        "flow_trace_gz": package / "03_flowstar_actual_stage_ledger/03_process/artifacts/raw_actual_trace.jsonl.gz",
        "flow_unobserved": package / "03_flowstar_actual_stage_ledger/02_actual_run/artifacts/unobserved.csv",
        "flow_observed": package / "03_flowstar_actual_stage_ledger/02_actual_run/artifacts/instrumented.csv",
        "torch_ledger": package / "04_torch_actual_stage_ledger/export_retry/artifacts/stage_ledger.json",
        "torch_summary": package / "04_torch_actual_stage_ledger/export_retry/artifacts/actual_path_summary.json",
        "exact": package / "05_exact_polynomial_oracle/02_oracles/artifacts/exact_remainder_and_range_oracle.json",
        "exact_poly": package / "05_exact_polynomial_oracle/02_oracles/artifacts/exact_polynomial_oracle.json",
        "formal": package / "05_exact_polynomial_oracle/02_oracles/artifacts/formal_true_solution_enclosure.json",
        "ladder": package / "05_exact_polynomial_oracle/02_oracles/artifacts/precision_ladder.json",
        "audit": package / "07_stage_swap_matrix/audit_retry/artifacts/actual_path_soundness.json",
        "first": package / "07_stage_swap_matrix/audit_retry/artifacts/first_difference_full_input_output.json",
        "swaps": package / "07_stage_swap_matrix/audit_retry/artifacts/stage_swap_matrix.json",
        "candidate": package / "07_stage_swap_matrix/audit_retry/artifacts/candidate_decision.json",
    }


def _evidence_exit_zero(package: Path, relative: str) -> None:
    root = package / relative
    _assert(root.is_dir(), f"missing evidence command {relative}")
    try:
        exit_code = int((root / "exit_code.txt").read_text(encoding="utf-8").strip())
    except Exception as exc:
        raise VerificationError(f"invalid exit code for {relative}: {exc}") from exc
    _assert(exit_code == 0, f"authoritative evidence command failed: {relative}")
    summary = _json(root / "summary.json")
    _assert(summary.get("exit_code") == 0, f"summary exit mismatch: {relative}")
    _assert(summary.get("status") == "pass", f"summary status mismatch: {relative}")


def _verify_test_and_publication_status(
    package: Path, repo: Path, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    _evidence_exit_zero(package, "13_tests/focused")
    _evidence_exit_zero(package, "13_tests/full_retry")
    test_status = _json(package / "13_tests/status.json")
    _assert(test_status.get("schema") == "step1_test_status_v1", "test status schema mismatch")
    _assert(
        test_status.get("authoritative_full_command") == "13_tests/full_retry",
        "authoritative full-test pointer mismatch",
    )
    if test_status.get("status") == "pass":
        _evidence_exit_zero(package, "13_tests/tamper")
        tamper = _json(package / "13_tests/tamper/artifacts/tamper_tests.json")
        _assert(tamper.get("passed") is True, "tamper tests did not pass")
        cases = tamper.get("cases", [])
        _assert(
            {row.get("case") for row in cases}
            == {"raw_number", "status", "file", "checksum"},
            "tamper case coverage mismatch",
        )
        _assert(
            all(row.get("rejected") is True and int(row.get("verifier_exit_code", 0)) != 0 for row in cases),
            "a tampered package was not rejected",
        )
    else:
        _assert(test_status.get("status") == "pending", "tests may only be pass or pending")

    publication = manifest.get("publication", {})
    fresh = _json(package / "14_fresh_clone/status.json")
    scientific_sha = manifest.get("scientific_sha")
    if fresh.get("status") == "pass":
        _evidence_exit_zero(package, "14_fresh_clone/acceptance")
        result = _json(package / "14_fresh_clone/acceptance/artifacts/result.json")
        _assert(result.get("status") == "pass", "fresh-clone result did not pass")
        _assert(scientific_sha and len(scientific_sha) == 40, "scientific SHA missing")
        _assert(result.get("scientific_sha") == scientific_sha, "fresh-clone scientific SHA mismatch")
        _assert(result.get("detached_head") == scientific_sha, "fresh-clone HEAD mismatch")
        for field in ("compileall", "focused_tests", "full_pytest", "package_verifier"):
            _assert(result.get(field) == "pass", f"fresh-clone {field} did not pass")
        _assert(result.get("git_status_porcelain_empty") is True, "fresh clone was dirty")
        _assert(publication.get("scientific_sha_fresh_clone_verified") is True, "publication fresh-clone flag mismatch")
    else:
        _assert(fresh.get("status") == "pending", "fresh clone may only be pass or pending")
        _assert(publication.get("scientific_sha_fresh_clone_verified") is False, "pending fresh clone marked verified")

    attestation = manifest.get("attestation_tip")
    if attestation:
        _assert(scientific_sha and len(scientific_sha) == 40, "attestation lacks scientific SHA")
        completed = subprocess.run(
            ["git", "diff", "--quiet", scientific_sha, attestation, "--", "src", "experiments", "tests"],
            cwd=repo,
            check=False,
        )
        _assert(completed.returncode == 0, "attestation changes a scientific tree")
        _assert(
            publication.get("attestation_tip_contains_no_scientific_tree_changes") is True,
            "attestation tree flag mismatch",
        )
    else:
        _assert(
            publication.get("attestation_tip_contains_no_scientific_tree_changes") == "unknown",
            "missing attestation must remain unknown",
        )
    _assert(publication.get("final_tip_fresh_clone_verified") is False, "final tip must not claim fresh-clone verification")
    return {
        "tests_status": test_status.get("status"),
        "fresh_clone_status": fresh.get("status"),
        "publication": publication,
    }


def verify(package: Path, repo: Path) -> dict[str, Any]:
    package = package.resolve()
    repo = repo.resolve()
    _assert(package.is_dir(), "package directory does not exist")
    for name in REQUIRED_DIRECTORIES:
        _assert((package / name).is_dir(), f"missing required directory {name}")
    checksums = _verify_checksums(package)
    manifest = _verify_manifest(package)
    selected = _selected_paths(package)
    for name, path in selected.items():
        _assert(path.is_file(), f"missing selected raw artifact {name}: {path}")

    audit_module = _load_module("step1_soundness_audit_for_verifier", repo / "experiments/audit_step1_soundness_and_swaps.py")
    oracle_module = _load_module("step1_oracle_runner_for_verifier", repo / "experiments/run_step1_independent_oracles.py")
    flow_rows = audit_module._load_ledger(selected["flow_ledger"])
    torch_rows = audit_module._load_ledger(selected["torch_ledger"])
    _assert(len(flow_rows) == 793, "Flow* ledger row count changed")
    _assert(len(torch_rows) == 16, "Torch ledger row count changed")

    flow_summary = _json(selected["flow_summary"])
    _assert(int(flow_summary["raw_step1_record_count"]) == 791, "Flow* raw step-1 count mismatch")
    _assert(selected["flow_unobserved"].read_bytes() == selected["flow_observed"].read_bytes(), "Flow* observer changed CSV")
    with gzip.open(selected["flow_trace_gz"], "rb") as handle:
        _assert(handle.read() == selected["flow_trace"].read_bytes(), "compressed Flow* raw trace mismatch")
    _assert(flow_summary["raw_trace_sha256"] == _sha(selected["flow_trace"]), "Flow* raw trace summary hash mismatch")
    torch_summary = _json(selected["torch_summary"])
    _assert(torch_summary["hook_read_only_equivalence"] is True, "Torch hook equivalence is not true")
    _assert(torch_summary["status"] == "validated", "Torch actual step did not validate")

    exact_poly = _json(selected["exact_poly"])
    _assert(
        exact_poly["flowstar_staged"] == oracle_module._iteration_json("flowstar_staged"),
        "stored Flow* exact iterations were not reproduced",
    )
    _assert(
        exact_poly["torch_complete"] == oracle_module._iteration_json("torch_complete"),
        "stored Torch exact iterations were not reproduced",
    )
    flow_final = exact_poly["flowstar_staged"][-1]["image"]
    torch_final = exact_poly["torch_complete"][-1]["image"]
    _assert(flow_final == torch_final, "exact fourth Picard images differ")
    _assert(len(flow_final["x"]["terms"]) == 13 and len(flow_final["y"]["terms"]) == 18, "exact final support changed")
    exact = _json(selected["exact"])
    formal = _json(selected["formal"])
    ladder = _json(selected["ladder"])
    _assert(ladder.get("conclusion_stable") is True, "stored precision ladder status is not closed")
    exact_object = _load_exact_object(repo)
    _assert(exact == exact_object.to_json(), "stored exact remainder/range oracle was not reproduced")
    _assert(formal == _load_formal_object(repo).to_json(), "stored formal truth enclosure was not reproduced")
    for precision in oracle_module.PRECISIONS:
        mpfr = _json(package / f"05_exact_polynomial_oracle/02_oracles/artifacts/mpfr_{precision}.json")
        validation = oracle_module._validate_mpfr_run(mpfr, exact_object)
        _assert(validation["passed"], f"MPFR {precision} no longer encloses exact oracle")

    flow_terms = audit_module._flowstar_initial(flow_rows)
    torch_terms = audit_module._torch_initial(torch_rows)
    flow_initial = audit_module._initial_witness("flowstar_pinned_actual", flow_terms, flowstar=True)
    torch_initial = audit_module._initial_witness("torch_complete_o4_legacy_production", torch_terms, flowstar=False)
    _assert(not flow_initial["contains_common_exact_input"], "Flow* under-enclosure witness disappeared")
    _assert(not torch_initial["contains_common_exact_input"], "Torch under-enclosure witness disappeared")
    first = _json(selected["first"])
    _assert(first["stage"] == "normalized_initial_tm", "stored first stage mismatch")
    _assert(first["classification"] == "UNDER_ENCLOSURE_WITNESS", "stored first classification mismatch")
    _assert(
        first["flowstar_full_input_output"]["components"]["x"]["missing_lower_gap"]
        == flow_initial["components"]["x"]["missing_lower_gap"],
        "Flow* witness gap was not recomputed",
    )
    _assert(
        first["torch_full_input_output"]["components"]["x"]["missing_upper_gap"]
        == torch_initial["components"]["x"]["missing_upper_gap"],
        "Torch witness gap was not recomputed",
    )

    flow_ranges = audit_module._range_checks("flowstar_pinned_actual", flow_rows, exact["range"], formal)
    torch_ranges = audit_module._range_checks("torch_complete_o4_legacy_production", torch_rows, exact["range"], formal)
    _assert(
        all(stage["classification"] == "ENCLOSURE_DIFFERENT_BOTH_SOUND" for tool in (flow_ranges, torch_ranges) for stage in tool["stages"].values()),
        "downstream formal range containment changed",
    )
    stored_audit = _json(selected["audit"])
    _assert(stored_audit["gate_d_status"] == "INDEPENDENT_STEP1_SOUNDNESS_ORACLE_INCOMPLETE", "Gate D status tampered")
    _assert(stored_audit["under_enclosure_witness_present"] is True, "Gate D witness status tampered")
    _assert(stored_audit["torch_endpoint_narrowness"]["formally_contains_true_solution"] is True, "Torch endpoint conclusion tampered")
    swaps = _json(selected["swaps"])
    _assert(swaps["status"] == "LOCAL_OPERATOR_SOURCE_DELTA_OPEN", "Gate E status tampered")
    _assert(len(swaps["cells"]) == 8 and all(not row["executed"] for row in swaps["cells"]), "stage-swap stop matrix tampered")
    candidate = _json(selected["candidate"])
    _assert(candidate["l1"] == "NOT_AUTHORIZED" and candidate["l2"] == candidate["l3"] == "NOT_RUN", "candidate status tampered")
    _assert(candidate["horner_status"] == "diagnostic_only", "Horner status tampered")

    statuses = manifest.get("statuses", {})
    expected_statuses = {
        "gate_b": "COMMON_STEP1_MATHEMATICAL_INPUT_CLOSED",
        "gate_d": "INDEPENDENT_STEP1_SOUNDNESS_ORACLE_INCOMPLETE",
        "gate_e": "LOCAL_OPERATOR_SOURCE_DELTA_OPEN",
        "l1": "NOT_AUTHORIZED",
        "l2": "NOT_RUN",
        "l3": "NOT_RUN",
        "legacy": "LEGACY_DEFAULT_UNCHANGED",
        "t10": "NOT_REACHED",
    }
    _assert(statuses == expected_statuses, "manifest status table mismatch")
    acceptance = _verify_test_and_publication_status(package, repo, manifest)
    for doc in (
        "EVIDENCE_LABEL_AND_PUBLICATION_SEMANTICS_20260813.md",
        "STEP1_COMMON_OPERATOR_CONTRACT_20260813.md",
        "FLOWSTAR_TORCH_STAGE_LEDGER_20260813.md",
        "INDEPENDENT_STEP1_SOUNDNESS_ORACLE_20260813.md",
        "LOCAL_OPERATOR_CAUSAL_CLOSURE_20260813.md",
        "SOUND_CANDIDATE_DECISION_20260813.md",
    ):
        _assert((repo / "docs" / doc).is_file(), f"missing required report {doc}")

    return {
        "schema": "flowstar_torch_step1_stage_oracle_verification_v1",
        "passed": True,
        "checksum_entry_count": checksums["entry_count"],
        "flowstar_ledger_rows": len(flow_rows),
        "torch_ledger_rows": len(torch_rows),
        "first_differing_stage": "normalized_initial_tm",
        "first_classification": "UNDER_ENCLOSURE_WITNESS",
        "exact_final_coefficients_equal": True,
        "flowstar_initial_contains_exact": False,
        "torch_initial_contains_exact": False,
        "downstream_ranges_formally_contain": True,
        "torch_endpoint_narrower_formally_sound": True,
        "gate_d": expected_statuses["gate_d"],
        "gate_e": expected_statuses["gate_e"],
        "candidate_eligibility": expected_statuses["l1"],
        "tests_status": acceptance["tests_status"],
        "fresh_clone_status": acceptance["fresh_clone_status"],
        "publication": acceptance["publication"],
    }


def _load_exact_object(repo: Path):
    path = repo / "src"
    import sys

    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    from torch_tm_flowpipe.step1_oracle import exact_step1_remainder_oracle

    return exact_step1_remainder_oracle(refinement_steps=5)


def _load_formal_object(repo: Path):
    path = repo / "src"
    import sys

    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    from torch_tm_flowpipe.step1_oracle import formal_true_solution_enclosure

    return formal_true_solution_enclosure(series_degree=100)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = verify(args.package, args.repo)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output_json is not None:
        args.output_json.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
