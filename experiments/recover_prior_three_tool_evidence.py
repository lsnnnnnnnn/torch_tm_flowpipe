#!/usr/bin/env python3
"""Recover a compact, audited copy of the missing 20260811 three-tool package."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Iterable, Sequence


EXPECTED_SOURCE_COMMIT = "2cb647cd37b530aef12e2b627f48b9b1dcf9aa23"
RUNNER_PROTOCOL_FILES = (
    "config.json",
    "summary.json",
    "command.txt",
    "stdout.log",
    "stderr.log",
    "exit_code.txt",
    "started_at.txt",
    "finished_at.txt",
    "timing.json",
)
EXACT_ARTIFACTS = (
    "03_native_flowstar/scalar_affine_gate/artifacts/run/analytic_oracle.json",
    "03_native_flowstar/scalar_affine_gate/artifacts/run/first_containment_loss.json",
    "03_native_flowstar/scalar_affine_gate/artifacts/run/official_generated_parity.json",
    "05_native_torch_complete_o4/authoritative/artifacts/run/checkpoints.csv",
    "05_native_torch_complete_o4/authoritative/artifacts/run/profile.csv",
    "06_native_torch_fixed_dr7/diffreach_explicit_f64_replay/artifacts/run/replayed_fixture.json",
    "07_flowstar_torch_raw_remainder/expression_tree/artifacts/run/raw_remainder_expression_tree.json",
    "07_flowstar_torch_raw_remainder/independent_analysis/artifacts/run/raw_remainder_counterfactuals.json",
    "07_flowstar_torch_raw_remainder/independent_analysis/artifacts/run/raw_remainder_first_divergence.json",
    "07_flowstar_torch_raw_remainder/independent_analysis/artifacts/run/raw_remainder_independent_replay.json",
    "07_flowstar_torch_raw_remainder/independent_analysis/artifacts/run/raw_remainder_mpfr_input.tsv",
    "07_flowstar_torch_raw_remainder/independent_analysis/artifacts/run/raw_remainder_node_comparison.csv",
    "07_flowstar_torch_raw_remainder/probe_t1/artifacts/flowstar_trace.csv",
    "08_schedule_validator_matrix/flowstar_fixed_h001/artifacts/flowstar_trace.csv",
    "08_schedule_validator_matrix/adaptive_schedule/artifacts/run/schedule_validator_matrix.json",
    "08_schedule_validator_matrix/adaptive_schedule/artifacts/run/torch_on_flowstar_schedule.json",
    "08_schedule_validator_matrix/fixed_h001_matrix/artifacts/run/schedule_validator_matrix.json",
    "08_schedule_validator_matrix/fixed_h001_matrix/artifacts/run/torch_on_flowstar_schedule.json",
    "09_fixed_support_descriptor/r35_mpfr_remainder_replay/artifacts/run/r35_overflow_interval_dag.tsv",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _original_checksum_rows(source: Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    for line in (source / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, separator, relative = line.partition("  ")
        if not separator or relative in rows:
            raise RuntimeError("original SHA256SUMS is malformed")
        rows[relative] = digest
    for relative, expected in rows.items():
        path = source / relative
        if not path.is_file() or _sha(path) != expected:
            raise RuntimeError(f"original package checksum mismatch: {relative}")
    expected_paths = {
        path.relative_to(source).as_posix()
        for path in source.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    }
    if set(rows) != expected_paths:
        raise RuntimeError("original checksum coverage is incomplete")
    return rows


def _runner_directories(source: Path) -> list[Path]:
    return sorted(
        path
        for path in source.rglob("config.json")
        if (path.parent / "command.txt").is_file()
    )


def _selected_paths(source: Path) -> Iterable[Path]:
    runners = _runner_directories(source)
    if len(runners) != 36:
        raise RuntimeError(f"expected 36 historical runners, found {len(runners)}")
    for config in runners:
        runner = config.parent
        for name in RUNNER_PROTOCOL_FILES:
            path = runner / name
            if not path.is_file():
                raise RuntimeError(f"historical runner file is missing: {path}")
            yield path
    yield source / "verification.json"
    for path in sorted(source.glob("*/**/artifacts/run/summary.json")):
        yield path
    for path in sorted(source.glob("*/**/artifacts/fraction_replay.json")):
        yield path
    for relative in EXACT_ARTIFACTS:
        path = source / relative
        if not path.is_file():
            raise RuntimeError(f"selected historical artifact is missing: {relative}")
        yield path
    figures = source / "15_reports" / "causal_figures" / "artifacts" / "run"
    for path in sorted(figures.iterdir()):
        if path.is_file() and path.suffix in {".json", ".csv", ".svg"}:
            yield path


def recover(args: argparse.Namespace) -> dict[str, Any]:
    source = args.source.resolve()
    destination = args.destination.resolve()
    if destination.exists():
        raise FileExistsError(destination)
    if not source.is_dir():
        raise FileNotFoundError(source)
    original_rows = _original_checksum_rows(source)
    original_manifest = json.loads((source / "manifest.json").read_text())
    if original_manifest.get("source_commit") != EXPECTED_SOURCE_COMMIT:
        raise RuntimeError("historical package source commit mismatch")
    destination.mkdir(parents=True)
    selected = sorted(
        set(_selected_paths(source)),
        key=lambda path: path.relative_to(source).as_posix(),
    )
    for path in selected:
        relative = path.relative_to(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    originals = destination / "historical_original"
    originals.mkdir()
    shutil.copy2(source / "manifest.json", originals / "manifest.json")
    shutil.copy2(source / "SHA256SUMS", originals / "SHA256SUMS")

    manifest = {
        "schema": "three_tool_historical_compact_recovery_v1",
        "run_id": destination.name,
        "recovery_status": "HISTORICAL_PACKAGE_RECOVERED_COMPACT_AUDITED",
        "source_commit": EXPECTED_SOURCE_COMMIT,
        "runner_count": 36,
        "package_root": ".",
        "original_package": {
            "file_count_excluding_SHA256SUMS": len(original_rows),
            "bytes": sum(
                (source / relative).stat().st_size for relative in original_rows
            )
            + (source / "SHA256SUMS").stat().st_size,
            "all_original_checksums_verified": True,
            "manifest": "historical_original/manifest.json",
            "manifest_sha256": _sha(source / "manifest.json"),
            "SHA256SUMS": "historical_original/SHA256SUMS",
            "SHA256SUMS_sha256": _sha(source / "SHA256SUMS"),
        },
        "compact_recovery": {
            "selected_source_file_count": len(selected),
            "preserves_all_runner_command_evidence": True,
            "preserves_original_verification_at_root": True,
            "preserves_key_scientific_summaries_and_small_raw_tables": True,
            "omits_compiled_binaries": True,
            "omits_npz": True,
            "omits_large_jsonl": True,
            "omits_large_bridge_ladder_arrays_but_preserves_gate_summaries": True,
        },
        "historical_claim_corrections": {
            "original_outcomes_are_archival_not_current": True,
            "diffreach_torch_operator_status": "DIFFREACH_TORCH_DR7_OPERATOR_EQUIVALENCE_CLOSED",
            "diffreach_torch_full_horizon_status": "DIFFREACH_TORCH_DR7_FULL_HORIZON_PAIRWISE_PENDING",
            "historical_14_fresh_clone_label": "CURRENT_WORKTREE_CHECKS_NOT_TRUE_CLONE",
        },
        "outcome_registry_at_recovery": {
            "evidence_package_status": "EVIDENCE_PACKAGE_REBUILT_PENDING_TRACKING",
            "raw_remainder_root_cause_status": "RAW_REMAINDER_ROOT_CAUSE_CLOSED",
            "flowstar_torch_fixed_schedule_status": "FLOWSTAR_TORCH_NATIVE_FULL_HORIZON_PAIRWISE_PARTIAL",
            "diffreach_torch_operator_status": "DIFFREACH_TORCH_DR7_OPERATOR_EQUIVALENCE_CLOSED",
            "diffreach_torch_full_horizon_status": "DIFFREACH_TORCH_DR7_FULL_HORIZON_PAIRWISE_PENDING",
            "carry_semantics_status": "CARRY_ROOT_CAUSE_MIXED_OR_UNRESOLVED",
            "single_fix_status": "NO_FIX_AUTHORIZED",
            "performance_status": "MATCHED_WORKLOAD_TIMING_UNAVAILABLE",
            "tightness_status": "TIGHTNESS_COMPARISON_UNAVAILABLE",
            "formal_scope": "no_new_formal_cross_tool_claim",
            "empirical_scope": "one_step_operator_and_separate_native_capability_only",
        },
    }
    _write_json(destination / "manifest.json", manifest)
    (destination / "RECOVERY_README.md").write_text(
        "# Compact historical recovery\n\n"
        "The server-local 20260811T100304Z package was found and every entry in "
        "its original SHA256SUMS was verified before recovery. This tracked copy "
        "retains all 36 runner command envelopes, the original verification, key "
        "scientific summaries, and small raw tables.\n\n"
        "The original manifest and checksum inventory are preserved under "
        "`historical_original/`. Their broad DiffReach/Torch full-horizon and "
        "`14_fresh_clone` labels are historical records, not current claims. Large "
        "NPZ, compiled binaries, bridge arrays, and JSONL traces are deliberately "
        "omitted from the compact tracked recovery.\n",
        encoding="utf-8",
    )
    checksum = destination / "SHA256SUMS"
    files = sorted(
        path
        for path in destination.rglob("*")
        if path.is_file() and path != checksum
    )
    checksum.write_text(
        "".join(
            f"{_sha(path)}  {path.relative_to(destination).as_posix()}\n"
            for path in files
        ),
        encoding="utf-8",
    )
    manifest["compact_recovery"]["tracked_file_count_excluding_SHA256SUMS"] = len(
        files
    )
    manifest["compact_recovery"]["tracked_bytes_excluding_SHA256SUMS"] = sum(
        path.stat().st_size for path in files
    )
    # Update metadata once, then regenerate the checksum inventory.
    _write_json(destination / "manifest.json", manifest)
    files = sorted(
        path
        for path in destination.rglob("*")
        if path.is_file() and path != checksum
    )
    checksum.write_text(
        "".join(
            f"{_sha(path)}  {path.relative_to(destination).as_posix()}\n"
            for path in files
        ),
        encoding="utf-8",
    )
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    print(json.dumps(recover(parse_args(argv)), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
