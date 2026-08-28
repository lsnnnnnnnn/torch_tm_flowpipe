#!/usr/bin/env python3
"""Verify the supplemental SR100 terminal replay and recompute the C3 status."""

from __future__ import annotations

import argparse
import csv
import hashlib
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
from scripts.analyze_brusselator_second_system import analyze  # noqa: E402


DEFAULT_PACKAGE = ROOT / "artifacts/runs/brusselator_sr100_terminal_replay_20260828"
ORIGINAL_PACKAGE = ROOT / "artifacts/runs/brusselator_generic_core_validation_20260827"
CONTRACT_PATH = ROOT / "benchmarks/brusselator_terminal_sr1000_contract.json"
BOUND_HEX_FIELDS = tuple(
    f"{prefix}_{component}_{bound}_hex"
    for prefix in ("endpoint", "tube")
    for component in ("x", "y")
    for bound in ("lo", "hi")
)
CORE_PATHS = (
    "src/torch_tm_flowpipe/accepted_boundary_sr.py",
    "src/torch_tm_flowpipe/symbolic_remainder.py",
    "src/torch_tm_flowpipe/flowpipe.py",
    "src/torch_tm_flowpipe/terminal_checkpoint.py",
    "src/torch_tm_flowpipe/state_equality.py",
)
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()


class TerminalReplayEvidenceError(ValueError):
    """Raised when the supplemental evidence is incomplete or inconsistent."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TerminalReplayEvidenceError(f"cannot read JSON {path}: {exc}") from exc


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _expected_prefix_signature() -> str:
    path = ORIGINAL_PACKAGE / "raw/torch_generic_sr100/segments.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("status") == "accepted"]
    signatures = [
        {
            "step": int(row["step"]),
            **{field: row[field] for field in BOUND_HEX_FIELDS},
            "queue_hash": row["queue_hash"],
        }
        for row in rows
    ]
    return hashlib.sha256(_canonical_bytes(signatures)).hexdigest()


def _old_result() -> dict[str, Any]:
    return analyze(
        ORIGINAL_PACKAGE / "raw",
        ORIGINAL_PACKAGE / "exact_fraction_2d.xml",
    )


def _core_unchanged(run_commit: str, generic_commit: str) -> bool:
    return subprocess.run(
        ["git", "diff", "--quiet", generic_commit, run_commit, "--", *CORE_PATHS],
        cwd=ROOT,
        check=False,
    ).returncode == 0


def recompute(package: Path) -> dict[str, Any]:
    contract = read_json(CONTRACT_PATH)
    terminal_contract = contract["terminal_replay"]
    command = read_json(package / "raw/command.json")
    replay = read_json(package / "raw/RESULT.json")
    before = package / "raw/checkpoint_before"
    after = package / "raw/checkpoint_after"
    checkpoint_files_equal = all(
        (before / name).read_bytes() == (after / name).read_bytes()
        for name in ("terminal_state.json", "terminal_state_manifest.json")
    )
    loaded_before = load_terminal_checkpoint(
        before,
        expected_contract=terminal_contract,
        expected_order=6,
        expected_dtype="float64",
    )
    loaded_after = load_terminal_checkpoint(
        after,
        expected_contract=terminal_contract,
        expected_order=6,
        expected_dtype="float64",
    )
    expected_prefix = _expected_prefix_signature()
    queue = loaded_before.normal_state.symbolic_queue
    queue_after = loaded_after.normal_state.symbolic_queue
    queue_contract = (
        queue is not None
        and queue_after is not None
        and queue.max_size == terminal_contract["queue_capacity"] == 100
        and queue.generation == queue.accepted_boundary_index == 355
        and queue.reset_count == 3
        and len(queue.J) == 55
        and queue.owner_generations == tuple(range(301, 356))
        and queue.owner_boundary_indices == tuple(range(301, 356))
    )
    replay_checks = {
        "contract_hash_matches": command.get("contract_sha256") == sha256(CONTRACT_PATH),
        "clean_tracked_run": (
            not command.get("worktree_status")
            and command.get("tracked_diff_sha256") == EMPTY_SHA256
        ),
        "generic_core_unchanged": _core_unchanged(
            str(command.get("commit")), contract["identity"]["generic_core_commit"]
        ),
        "published_prefix_recomputed": (
            replay.get("published_prefix_reconstructed_bit_exact") is True
            and replay.get("published_prefix_boundaries") == 355
            and replay.get("published_prefix_signature_sha256") == expected_prefix
        ),
        "exactly_one_terminal_attempt": (
            replay.get("terminal_attempt_count") == 1
            and replay.get("terminal_step") == 356
        ),
        "terminal_rejected_without_publication": (
            replay.get("terminal_segment_status") == "failed"
            and replay.get("terminal_rejected_without_publication") is True
        ),
        "checkpoint_files_byte_equal": checkpoint_files_equal,
        "checkpoint_full_hash_equal": (
            loaded_before.manifest["full_checkpoint_sha256"]
            == loaded_after.manifest["full_checkpoint_sha256"]
            == replay.get("checkpoint_full_sha256_before")
            == replay.get("checkpoint_full_sha256_after")
        ),
        "queue_metadata_unchanged": (
            replay.get("queue_before") == replay.get("queue_after") and queue_contract
        ),
        "runner_rollback_conclusion": (
            replay.get("rollback_proved") is True
            and replay.get("status") == "C3_TERMINAL_ROLLBACK_CLOSED"
        ),
    }
    old = _old_result()
    old_sr100 = old["lane_checks"]["torch_generic_sr100"]
    false_old_checks = sorted(key for key, value in old_sr100.items() if not value)
    recomputed_sr100 = dict(old_sr100)
    recomputed_sr100["rollback"] = all(replay_checks.values())
    # The raw runner's summary certificate was false solely because its terminal
    # row could not observe caller-owned state.  Recompute that aggregate from
    # every other raw check plus the supplemental checkpoint proof.
    recomputed_sr100["summary_certificate"] = all(
        value for key, value in recomputed_sr100.items() if key != "summary_certificate"
    )
    lane_checks = dict(old["lane_checks"])
    lane_checks["torch_generic_sr100"] = recomputed_sr100
    soundness = (
        old["exact_fraction_2d_test_passed"]
        and all(old["source_checks"].values())
        and all(old["flowstar_checks"].values())
        and all(all(checks.values()) for checks in lane_checks.values())
    )
    if not soundness:
        status = "C3_GENERICITY_SOUNDNESS_GATE_FAILED_STOP"
    elif old["material_gain"]["passed"]:
        status = "C3_GENERIC_CORE_VALIDATED__SECOND_SYSTEM_PRODUCTION_USEFUL"
    elif old["material_gain"]["eligible"]:
        status = "C3_GENERIC_CORE_VALIDATED__SECOND_SYSTEM_NO_MATERIAL_GAIN"
    else:
        status = "C3_GENERIC_POLYNOMIAL_PLANT_CORE_VALIDATED"
    return {
        "schema": "torch_tm_flowpipe.brusselator_terminal_replay_closure/1",
        "status": status,
        "soundness_gate_passed": soundness,
        "original_status": old["status"],
        "original_sr100_false_checks": false_old_checks,
        "supplemental_replay_checks": replay_checks,
        "recomputed_sr100_checks": recomputed_sr100,
        "material_gain": old["material_gain"],
        "accepted_steps": old["accepted_steps"],
        "native_horizons": old["native_horizons"],
        "terminal_replay_result_sha256": sha256(package / "raw/RESULT.json"),
        "original_result_sha256": sha256(ORIGINAL_PACKAGE / "RESULT.json"),
        "contract_sha256": sha256(CONTRACT_PATH),
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
        if not recomputed["soundness_gate_passed"]:
            errors.append("supplemental terminal replay did not close the C3 soundness gate")
    except (KeyError, TypeError, ValueError, OSError) as exc:
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
