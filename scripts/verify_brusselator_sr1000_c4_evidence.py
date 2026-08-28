#!/usr/bin/env python3
"""Verify the frozen SR1000/operator-ledger/C4 closure package."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from torch_tm_flowpipe import load_terminal_checkpoint  # noqa: E402
from scripts.verify_brusselator_terminal_replay_evidence import (  # noqa: E402
    DEFAULT_PACKAGE as C3_PACKAGE,
    verify as verify_c3,
)


DEFAULT_PACKAGE = ROOT / "artifacts/runs/brusselator_sr1000_c4_closure_20260828"
BASELINE_COMMIT = "beb0daf310c360a28a0ecce04554a29bc30d0dbe"
C4_COMMIT = "26323929d6f4fee0893478f6927ae76c5129bf47"
FLOWSTAR_COMMIT = "b85a3211748cb77b736fe4ad42ee02d8d2b81148"
FLOWSTAR_SEGMENTS_SHA256 = "08e184e2b0a99be48417be8971ed6632eccec0630849787a1048b9962d15f567"
OBSERVER_PATCH_SHA256 = "e2f9186e1576502d45771dfc5c244998680a1947d5c5bde7b8251536d7ac813c"
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
CORE_PATHS = (
    "src/torch_tm_flowpipe/batched_dense_tm.py",
    "src/torch_tm_flowpipe/flowpipe.py",
    "src/torch_tm_flowpipe/__init__.py",
)


class ClosureEvidenceError(ValueError):
    """Raised when closure evidence is incomplete or inconsistent."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ClosureEvidenceError(f"cannot read JSON {path}: {exc}") from exc


def _gunzip(path: Path) -> bytes:
    try:
        with gzip.open(path, "rb") as handle:
            return handle.read()
    except (OSError, gzip.BadGzipFile) as exc:
        raise ClosureEvidenceError(f"cannot read gzip stream {path}: {exc}") from exc


def _csv_rows(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(_gunzip(path).decode("utf-8"))))


def _trace_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in _gunzip(path).decode("utf-8").splitlines()]


def _number(value: Mapping[str, str]) -> float:
    decimal = float(value["decimal"])
    hexadecimal = float.fromhex(value["hex"])
    if decimal != hexadecimal:
        raise ClosureEvidenceError(f"decimal/hex trace mismatch: {value}")
    return hexadecimal


def _interval(row: Mapping[str, Any], field: str) -> tuple[float, float]:
    value = row[field]
    return _number(value["lower"]), _number(value["upper"])


def _expected_core_patch() -> bytes:
    return subprocess.run(
        [
            "git",
            "diff",
            "--binary",
            BASELINE_COMMIT,
            C4_COMMIT,
            "--",
            *CORE_PATHS,
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout


def recompute(package: Path) -> dict[str, Any]:
    raw = package / "raw"
    contract = read_json(raw / "contract.json")
    command = read_json(raw / "sr1000/command.json")
    summary = read_json(raw / "sr1000/summary.json")
    torch_rows = _csv_rows(raw / "sr1000/segments.csv.gz")
    stock_path = raw / "flowstar/frozen_stock_segments.csv.gz"
    observed_path = raw / "flowstar/observed.csv.gz"
    unobserved_path = raw / "flowstar/unobserved.csv.gz"
    stock_bytes = _gunzip(stock_path)
    observed_bytes = _gunzip(observed_path)
    unobserved_bytes = _gunzip(unobserved_path)
    stock_rows = _csv_rows(stock_path)
    trace_rows = _trace_rows(raw / "flowstar/observed_trace.jsonl.gz")
    compose_rows = _trace_rows(raw / "flowstar/compose_probe_trace.jsonl.gz")
    compose = read_json(raw / "flowstar/compose_probe_result.json")
    c4_result = read_json(raw / "c4/RESULT.json")
    c4_gate = read_json(raw / "c4/same_input_gate.json")
    ledger = read_json(raw / "c4/operator_ledger.json")
    c4_manifest = read_json(raw / "c4/MANIFEST.json")
    provenance = read_json(raw / "c4/provenance.json")

    accepted = [row for row in torch_rows if row["status"] == "accepted"]
    rejected = [row for row in torch_rows if row["status"] == "rejected"]
    last_accepted = accepted[-1]
    terminal = rejected[-1]
    stock_sha = _sha256_bytes(stock_bytes)
    contract_sha = sha256(raw / "contract.json")
    checkpoint = load_terminal_checkpoint(raw / "same_input_prestate")
    c3_result, c3_errors = verify_c3(C3_PACKAGE)

    expected_search = contract["decision"]["operator_search_order"]
    stages = ledger["stages"]
    first_material = next(row for row in stages if row.get("material") is True)
    stage3 = stages[2]
    candidate_intermediates = [
        _interval(row, "interval")
        for row in trace_rows
        if row.get("stage") == "candidate_intermediate_ranges"
    ]
    degree_hooks = [
        _interval(row, "interval")
        for row in trace_rows
        if row.get("stage") == "operator_degree_truncation"
        and int(row["attempt_index"]) == 0
    ]
    cutoff_hooks = [
        _interval(row, "interval")
        for row in trace_rows
        if row.get("stage") == "operator_cutoff" and int(row["attempt_index"]) == 0
    ]
    traced_terminal_truncation = (
        candidate_intermediates[7],
        candidate_intermediates[14],
    )
    hook_terminal_truncation = (degree_hooks[4], degree_hooks[9])

    stock_first = stock_rows[0]
    compose_exact = all(
        compose[f"{prefix}_{component}"][f"{bound}_hex"]
        == stock_first[f"{prefix}_{component}_{bound}_hex"]
        for prefix in ("endpoint", "tube")
        for component in ("x", "y")
        for bound in ("lo", "hi")
    )
    gate_booleans = (
        c4_result["same_input_hashes_equal"],
        c4_result["baseline_reproduces_frozen_sr1000_step1"],
        c4_result["flowstar_observer_output_equivalent"],
        c4_gate["first_acceptance_identical_to_legacy"],
        c4_gate["retained_polynomial_equal"],
        c4_gate["all_commits_subset"],
        c4_gate["validated_decomposition_contains_image"],
        c4_gate["sample_solver_ok"],
        c4_gate["sample_endpoint_violations"] == 0,
        c4_gate["sample_tube_violations"] == 0,
    )
    stage5 = stages[4]
    recomputed_ratio = stage5["c4_stock_l1_error"] / stage5["baseline_stock_l1_error"]

    expected_patch = _expected_core_patch()
    recorded_patch = (raw / "c4/core.patch").read_bytes()
    c4_files = {
        name: {"sha256": sha256(raw / f"c4/{name}"), "bytes": (raw / f"c4/{name}").stat().st_size}
        for name in ("RESULT.json", "operator_ledger.json", "provenance.json", "same_input_gate.json")
    }
    checks = {
        "c3_terminal_rollback_closed": not c3_errors
        and c3_result is not None
        and c3_result["soundness_gate_passed"],
        "contract_identity": (
            contract["schema"] == "torch_tm_flowpipe.brusselator_terminal_sr1000_contract/1"
            and command["contract_sha256"] == summary["contract_sha256"] == contract_sha
            and contract["scope"]["c4_numeric_fix_budget"] == 1
        ),
        "sr1000_clean_frozen_execution": (
            command["commit"] == BASELINE_COMMIT
            and command["worktree_status"] == ""
            and command["tracked_diff_sha256"] == EMPTY_SHA256
            and command["generic_core_unchanged"] is True
            and summary["certificate_checks_passed"] is True
            and summary["owner_accounting_passed"] is True
            and summary["sample_solver_ok"] is True
            and summary["sample_endpoint_violations"] == 0
            and summary["sample_tube_violations"] == 0
        ),
        "sr1000_prefix_and_terminal_rejection": (
            len(accepted) == summary["accepted_steps"] == 357
            and [int(row["step"]) for row in accepted] == list(range(1, 358))
            and len(rejected) == summary["rejected_steps"] == 1
            and int(terminal["step"]) == 358
            and terminal["segment_status"] == "failed"
            and terminal["rollback_checkpoint_byte_equal"] == "True"
            and terminal["rollback_queue_unchanged"] == "True"
            and terminal["queue_hash_before"] == terminal["queue_hash_after_attempt"]
        ),
        "queue_capacity_not_reached_or_reset": (
            int(last_accepted["queue_capacity"]) == 1000
            and int(last_accepted["queue_size"]) == 357
            and int(last_accepted["queue_reset_count"]) == 0
            and summary["queue_reset_count"] == 0
            and summary["capacity_reset_decision"] == "NOT_SOLELY_QUEUE_RESET_CAPACITY"
            and summary["completed_requested_horizon"] is False
        ),
        "stock_prefix_available": (
            stock_sha == contract["identity"]["flowstar_segments_sha256"]
            == FLOWSTAR_SEGMENTS_SHA256
            and len(stock_rows) == 1000
            and [int(row["step"]) for row in stock_rows] == list(range(1, 1001))
        ),
        "flowstar_observer_output_equivalent": (
            stock_bytes == observed_bytes == unobserved_bytes
            and stock_sha == FLOWSTAR_SEGMENTS_SHA256
            and provenance["flowstar_observer_trace_is_output_equivalent"] is True
        ),
        "flowstar_trace_provenance": (
            len(trace_rows) == 1850
            and len(compose_rows) == provenance["compose_trace_record_count"] == 3228
            and all(row["source_commit"] == FLOWSTAR_COMMIT for row in trace_rows + compose_rows)
            and {int(row["accepted_step_index"]) for row in trace_rows + compose_rows} == {0}
            and sha256(raw / "flowstar/observer.patch") == OBSERVER_PATCH_SHA256
        ),
        "compose_probe_matches_stock": compose_exact
        and stages[5]["flowstar_compose_probe_matches_frozen_stock_bit_exact"],
        "operator_search_order": (
            [row["stage"] for row in stages] == expected_search[:6]
            and [row["search_index"] for row in stages] == list(range(1, 7))
        ),
        "first_material_divergence": (
            stages[0]["material"] is False
            and stages[1]["material"] is False
            and first_material is stage3
            and first_material["stage"] == c4_result["first_material_operator_divergence"]
            == "truncation_cutoff_owners"
            and first_material["search_index"] == c4_result["first_material_search_index"] == 3
            and first_material["max_degree_truncation_bound_delta"] > ledger["material_threshold"]
        ),
        "truncation_owner_trace_recomputed": (
            tuple(tuple(value) for value in stage3["flowstar_terminal_degree_truncation"])
            == traced_terminal_truncation
            == hook_terminal_truncation
            and cutoff_hooks
            and all(lo == 0.0 and hi == 0.0 for lo, hi in cutoff_hooks)
        ),
        "same_input_checkpoint": (
            checkpoint.manifest["full_checkpoint_sha256"]
            == c4_result["same_input_checkpoint_sha256"]
            == "c8dea2b07eed81f29e1ac395e3f2b1f7d2528ab9064992d18db0b05c0641bb0b"
        ),
        "same_input_c4_gate": (
            all(gate_booleans)
            and c4_gate == c4_result["same_input_gate"]
            and c4_gate["refinement_iterations"] == 8
            and c4_gate["stop_reason"] == "stop_ratio"
            and c4_gate["stock_remainder_error_ratio"] == recomputed_ratio
            and recomputed_ratio < 0.25
        ),
        "single_c4_fix_authorized": (
            c4_result["c4_status"] == "C4_FIX_AUTHORIZED"
            and c4_result["c4_fix_budget"] == c4_result["c4_numeric_fixes_authorized"] == 1
            and c4_result["full_c4_prefix_rerun_performed"] is False
            and c4_result["endpoint_range_semantics_is_later_not_first"] is True
        ),
        "c4_patch_bound_to_commit": (
            provenance["head"] == C4_COMMIT
            and provenance["baseline_commit"] == BASELINE_COMMIT
            and recorded_patch == expected_patch
            and _sha256_bytes(recorded_patch) == c4_result["c4_core_diff_sha256"]
        ),
        "c4_analysis_manifest": (
            c4_manifest["result"] == c4_result and c4_manifest["files"] == c4_files
        ),
    }
    ok = all(checks.values())
    return {
        "schema": "torch_tm_flowpipe.brusselator_sr1000_c4_closure/1",
        "status": "BRUSSELATOR_SR1000_OPERATOR_C4_CLOSED" if ok else "CLOSURE_FAILED_STOP",
        "checks": checks,
        "capacity_reset_decision": summary["capacity_reset_decision"],
        "torch_accepted_steps": len(accepted),
        "torch_terminal_rejection_step": int(terminal["step"]),
        "stock_accepted_steps": len(stock_rows),
        "first_material_operator_divergence": first_material["stage"],
        "first_material_max_delta": first_material["max_degree_truncation_bound_delta"],
        "c4_status": c4_result["c4_status"],
        "c4_mode": c4_result["c4_mode"],
        "c4_stock_remainder_error_ratio": c4_gate["stock_remainder_error_ratio"],
        "c4_commit": C4_COMMIT,
    }


def verify_checksums(package: Path) -> list[str]:
    checksum_path = package / "SHA256SUMS"
    if not checksum_path.is_file():
        return ["SHA256SUMS missing"]
    expected: dict[str, str] = {}
    errors: list[str] = []
    for line in checksum_path.read_text(encoding="ascii").splitlines():
        digest, separator, relative = line.partition("  ")
        if not separator or len(digest) != 64 or relative in expected:
            errors.append(f"malformed or duplicate checksum line: {line}")
        else:
            expected[relative] = digest
    actual = {
        path.relative_to(package).as_posix()
        for path in package.rglob("*")
        if path.is_file() and path != checksum_path
    }
    for relative in sorted(actual | set(expected)):
        path = package / relative
        if relative not in actual:
            errors.append(f"checksum target missing: {relative}")
        elif relative not in expected:
            errors.append(f"uncovered file: {relative}")
        elif sha256(path) != expected[relative]:
            errors.append(f"checksum mismatch: {relative}")
    return errors


def verify(package: Path) -> tuple[dict[str, Any] | None, list[str]]:
    errors = verify_checksums(package)
    try:
        recomputed = recompute(package)
        recorded = read_json(package / "CLOSURE_RESULT.json")
        if recomputed != recorded:
            errors.append("CLOSURE_RESULT.json does not match raw recomputation")
        false_checks = sorted(key for key, value in recomputed["checks"].items() if not value)
        if false_checks:
            errors.append(f"closure checks failed: {false_checks}")
    except (KeyError, IndexError, TypeError, ValueError, OSError, subprocess.SubprocessError) as exc:
        return None, errors + [str(exc)]
    return recomputed, errors


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=DEFAULT_PACKAGE)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    result, errors = verify(parse_args(argv).package.resolve())
    print(json.dumps({"ok": not errors, "errors": errors, "result": result}, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
