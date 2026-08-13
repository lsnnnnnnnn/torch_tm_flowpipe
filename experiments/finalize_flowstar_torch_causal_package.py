#!/usr/bin/env python3
"""Compact and index the complete Flow*--Torch causal-closure evidence package."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any


REQUIRED_DIRECTORIES = tuple(f"{index:02d}_{name}" for index, name in enumerate((
    "identity_provenance",
    "baseline_reproduction",
    "evidence_label_corrections",
    "stock_clean_outputs",
    "stock_instrumented_outputs",
    "copied_probe_equivalence",
    "flowstar_queue_factorial",
    "torch_horner_queue_factorial",
    "step1_step2_attribution",
    "lossless_schema_roundtrip",
    "same_prestate_2x2",
    "source_ledger_micro_oracles",
    "candidate_l1_l2_l3",
    "tests",
    "final_clone",
)))


SUPERSEDED = {
    "00_identity_provenance/copied_probe_build": "superseded by copied_probe_exact_build",
    "00_identity_provenance/flowstar_instrumented_build": "initial build missed a required header; retry passed",
    "00_identity_provenance/lossless_bridge_build": "superseded by explicit-degree retry",
    "00_identity_provenance/lossless_bridge_build_retry": "superseded by final guarded continuation build",
    "02_evidence_label_corrections/baseline_audit_initial_failed": "candidate step field indexing error fixed",
    "02_evidence_label_corrections/baseline_audit_scale_index_failed": "Flow* row/produced-boundary indexing error fixed",
    "05_copied_probe_equivalence/three_way_audit_initial_failed": "stock/copied step field mapping fixed",
    "05_copied_probe_equivalence/three_way_audit_hex_format_failed": "equivalent hex spellings normalized by binary64 value",
    "06_flowstar_queue_factorial/q1": "runner expected-exit setting superseded by q1_qualified; raw CSV is exact",
    "06_flowstar_queue_factorial/q2": "runner expected-exit setting superseded by q2_qualified; raw CSV is exact",
    "06_flowstar_queue_factorial/q10": "runner expected-exit setting superseded by q10_qualified; raw CSV is exact",
    "07_torch_horner_queue_factorial/T-D0_superseded_checkpoint_setting": "terminal checkpoint cannot encode diagnostic queue",
    "07_torch_horner_queue_factorial/T-H0_superseded_checkpoint_setting": "uniform no-checkpoint factorial rerun used",
    "07_torch_horner_queue_factorial/T-DQ_superseded_checkpoint_setting": "terminal checkpoint cannot encode diagnostic queue",
    "07_torch_horner_queue_factorial/T-HQ_superseded_checkpoint_setting": "terminal checkpoint cannot encode diagnostic queue",
    "08_step1_step2_attribution/causal_analysis_initial_failed": "polynomial trace path corrected",
    "08_step1_step2_attribution/causal_analysis_pre_field_correction": "step-1 segment/endpoint polynomial labels corrected",
    "09_lossless_schema_roundtrip/flowstar_fixtures": "term total-degree initialization corrected in guarded import",
    "09_lossless_schema_roundtrip/cross_language": "Python import path corrected",
    "09_lossless_schema_roundtrip/cross_language_retry": "zero exponent sentinel corrected",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def command(argv: list[str], cwd: Path | None = None) -> dict[str, Any]:
    completed = subprocess.run(argv, cwd=cwd, text=True, capture_output=True, check=False)
    return {
        "argv": argv,
        "cwd": str(cwd) if cwd is not None else None,
        "exit_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def deterministic_gzip(path: Path) -> Path:
    output = path.with_name(path.name + ".gz")
    with path.open("rb") as source, output.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", compresslevel=9, fileobj=raw, mtime=0) as target:
            shutil.copyfileobj(source, target, length=1024 * 1024)
    path.unlink()
    return output


def artifact_rows(root: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "artifact_index.json"
    ]


def rebuild_runner_indices(package: Path) -> int:
    count = 0
    for config in sorted(package.rglob("config.json")):
        runner = config.parent
        if not all((runner / name).is_file() for name in ("summary.json", "command.txt", "timing.json")):
            continue
        write_json(
            runner / "artifact_index.json",
            {
                "schema": "torch_tm_flowpipe_evidence_artifact_index_v1",
                "root": ".",
                "files": artifact_rows(runner),
            },
        )
        count += 1
    return count


def qualification(relative: str) -> str:
    if relative.startswith("00_identity_provenance/"):
        return "provenance/build/raw protocol"
    if relative.startswith(("03_stock", "04_stock", "05_copied")):
        return "actual-path empirical equivalence"
    if relative.startswith("06_flowstar"):
        return "deterministic actual-path single-factor ablation"
    if relative.startswith("07_torch"):
        return "diagnostic deterministic factorial; not a sound candidate"
    if relative.startswith("09_lossless"):
        return "canonical exact roundtrip and negative-test evidence"
    if relative.startswith("10_same"):
        return "same-producer exact continuation plus fail-closed operator mismatch"
    if relative.startswith(("11_source", "12_candidate")):
        return "not run because prerequisite gate remained open"
    if relative.startswith(("13_tests", "14_final")):
        return "software/publication verification"
    return "derived scientific evidence"


def finalize(package: Path, repository: Path, flowstar_clean: Path, flowstar_instrumented: Path) -> None:
    package = package.resolve()
    repository = repository.resolve()
    for directory in REQUIRED_DIRECTORIES:
        (package / directory).mkdir(parents=True, exist_ok=True)

    removed: list[dict[str, str]] = []
    for relative, reason in SUPERSEDED.items():
        target = package / relative
        if target.exists():
            if not target.is_dir():
                raise ValueError(f"expected superseded directory: {target}")
            shutil.rmtree(target)
            removed.append({"path": relative, "reason": reason})
    write_json(
        package / "00_identity_provenance/superseded_attempts.json",
        {
            "schema": "causal_closure_superseded_attempts_v1",
            "removed_generated_attempts": removed,
            "unique_scientific_raw_evidence_removed": False,
        },
    )

    compressed: list[dict[str, Any]] = []
    for path in sorted(package.rglob("*")):
        if (
            path.is_file()
            and path.suffix in {".csv", ".jsonl"}
            and path.stat().st_size >= 1_000_000
        ):
            before = path.stat().st_size
            output = deterministic_gzip(path)
            compressed.append(
                {
                    "path": output.relative_to(package).as_posix(),
                    "uncompressed_bytes": before,
                    "compressed_bytes": output.stat().st_size,
                    "sha256": sha256(output),
                }
            )
    write_json(
        package / "00_identity_provenance/deterministic_compression.json",
        {
            "schema": "causal_closure_deterministic_compression_v1",
            "gzip_mtime": 0,
            "files": compressed,
        },
    )

    instrumentation_diff = command(["git", "diff", "--binary"], flowstar_instrumented)
    if instrumentation_diff["exit_code"] != 0 or not instrumentation_diff["stdout"]:
        raise RuntimeError("cannot capture final Flow* instrumentation diff")
    (package / "00_identity_provenance/flowstar_instrumentation_final.diff").write_text(
        instrumentation_diff["stdout"], encoding="utf-8"
    )
    provenance = {
        "schema": "flowstar_causal_closure_final_provenance_v1",
        "torch_start_sha": "cdda27bf2c0e7f72e135edbfd2b2ba10a8c5f96d",
        "torch_branch": "codex/flowstar-torch-causal-mechanism-closure-20260813",
        "torch_status": command(["git", "status", "--short", "--branch"], repository),
        "flowstar_clean": {
            "root": str(flowstar_clean),
            "sha": command(["git", "rev-parse", "HEAD"], flowstar_clean),
            "tracked_diff": command(["git", "diff", "--exit-code"], flowstar_clean),
            "status": command(["git", "status", "--short", "--branch"], flowstar_clean),
        },
        "flowstar_instrumented": {
            "root": str(flowstar_instrumented),
            "sha": command(["git", "rev-parse", "HEAD"], flowstar_instrumented),
            "diff_sha256": sha256(package / "00_identity_provenance/flowstar_instrumentation_final.diff"),
            "status": command(["git", "status", "--short", "--branch"], flowstar_instrumented),
        },
        "binaries": [],
    }
    binary_paths = [
        flowstar_clean / "flowstar-toolbox/libflowstar.a",
        flowstar_instrumented / "flowstar-toolbox/libflowstar.a",
        package / "00_identity_provenance/flowstar_stock_driver_build/artifacts/flowstar_vdp_stock_reach_driver",
        package / "00_identity_provenance/flowstar_instrumented_driver_build/artifacts/flowstar_vdp_instrumented_reach_driver",
        package / "00_identity_provenance/copied_probe_exact_build/artifacts/flowstar_vdp_copied_probe_exact",
        package / "00_identity_provenance/lossless_bridge_build_continue/artifacts/flowstar_lossless_state_queue_bridge",
    ]
    for path in binary_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        record: dict[str, Any] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "file": command(["file", str(path)]),
        }
        if path.suffix != ".a":
            record["ldd"] = command(["ldd", str(path)])
        provenance["binaries"].append(record)
    write_json(package / "00_identity_provenance/final_provenance.json", provenance)

    write_json(
        package / "11_source_ledger_micro_oracles/not_run_reason.json",
        {
            "schema": "causal_closure_not_run_v1",
            "gate": "F",
            "status": "SOURCE_LEDGER_ORACLE_INCOMPLETE",
            "executed": False,
            "reason": (
                "Gate E did not close the full cross-operator same-prestate 2x2: Torch cannot "
                "consume the complete Flow* x/y/t plus Phi_L/J state, and Flow* rejects the "
                "two-component Torch state. Queue dropping and common-box reboxing are forbidden."
            ),
            "prerequisite_status": "SOURCE_MECHANISM_CANDIDATES_LOCALIZED_CAUSAL_SPLIT_OPEN",
            "micro_oracles_completed": [],
        },
    )
    write_json(
        package / "12_candidate_l1_l2_l3/not_run_reason.json",
        {
            "schema": "causal_closure_not_run_v1",
            "gate": "G",
            "status": "NO_FIX_AUTHORIZED",
            "executed": False,
            "reason": "Gate F is SOURCE_LEDGER_ORACLE_INCOMPLETE.",
            "candidate_implemented": False,
            "legacy_default_changed": False,
            "L1": "NOT_RUN",
            "L2": "NOT_RUN",
            "L3": "NOT_RUN",
        },
    )
    final_clone = package / "14_final_clone/status.json"
    if not final_clone.exists():
        write_json(
            final_clone,
            {
                "schema": "causal_closure_publication_attestation_v1",
                "status": "PENDING_SCIENTIFIC_COMMIT_AND_FRESH_CLONE",
                "scientific_sha": None,
                "publication_tip": None,
            },
        )

    statuses = [
        "BASELINE_CONCLUSIONS_REPRODUCED",
        "FLOWSTAR_WIDTH_MINIMUM_POSITIVE_NOT_NUMERICALLY_NEAR_ZERO",
        "STOCK_COPIED_PROBE_EQUIVALENCE_CLOSED",
        "CAUSAL_FACTOR_SPLIT_PARTIAL",
        "SAME_PRESTATE_LOSSLESS_BRIDGE_AVAILABLE",
        "SOURCE_MECHANISM_CANDIDATES_LOCALIZED_CAUSAL_SPLIT_OPEN",
        "SOURCE_LEDGER_ORACLE_INCOMPLETE",
        "NO_FIX_AUTHORIZED",
    ]
    verification = {
        "schema": "flowstar_torch_causal_closure_verification_v1",
        "status": "PASS_SCIENTIFIC_PREPUBLICATION",
        "scientific_statuses": statuses,
        "scientific_outcome_uses_process_exit_code": False,
        "gates": {
            "A": "BASELINE_CONCLUSIONS_REPRODUCED",
            "B": "STOCK_COPIED_PROBE_EQUIVALENCE_CLOSED",
            "C": "CAUSAL_FACTOR_SPLIT_PARTIAL",
            "D": "SAME_PRESTATE_LOSSLESS_BRIDGE_AVAILABLE",
            "E": "SOURCE_MECHANISM_CANDIDATES_LOCALIZED_CAUSAL_SPLIT_OPEN",
            "F": "SOURCE_LEDGER_ORACLE_INCOMPLETE",
            "G": "NO_FIX_AUTHORIZED",
        },
        "publication": read_json(final_clone),
    }
    write_json(package / "verification.json", verification)
    runner_count = rebuild_runner_indices(package)

    content_files = sorted(
        path
        for path in package.rglob("*")
        if path.is_file() and path.name not in {"manifest.json", "SHA256SUMS"}
    )
    manifest = {
        "schema": "flowstar_torch_causal_closure_manifest_v1",
        "package_root": package.name,
        "torch_start_sha": "cdda27bf2c0e7f72e135edbfd2b2ba10a8c5f96d",
        "scientific_commit": None,
        "publication_tip": None,
        "flowstar_sha": "b85a3211748cb77b736fe4ad42ee02d8d2b81148",
        "runner_count": runner_count,
        "required_directories": list(REQUIRED_DIRECTORIES),
        "scientific_statuses": statuses,
        "outputs": [
            {
                "path": path.relative_to(package).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "qualification": qualification(path.relative_to(package).as_posix()),
            }
            for path in content_files
        ],
    }
    write_json(package / "manifest.json", manifest)
    checksum_files = sorted(
        path for path in package.rglob("*") if path.is_file() and path.name != "SHA256SUMS"
    )
    (package / "SHA256SUMS").write_text(
        "".join(
            f"{sha256(path)}  {path.relative_to(package).as_posix()}\n"
            for path in checksum_files
        ),
        encoding="utf-8",
    )


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--flowstar-clean", type=Path, required=True)
    parser.add_argument("--flowstar-instrumented", type=Path, required=True)
    args = parser.parse_args()
    finalize(args.package, args.repository, args.flowstar_clean, args.flowstar_instrumented)


if __name__ == "__main__":
    main()
