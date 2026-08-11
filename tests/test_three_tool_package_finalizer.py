from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

from experiments.finalize_three_tool_evidence_package import finalize
from experiments.run_evidence_command import run
from torch_tm_flowpipe.evidence_verification import validate_verification_document


ROOT = Path(__file__).resolve().parents[1]


def test_top_level_package_builder_is_directly_executable() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "experiments/build_three_tool_evidence_package.py"),
            "--help",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--run-root" in completed.stdout


def test_finalizer_derives_claims_and_root_relative_hashes(tmp_path) -> None:
    root = tmp_path / "20260811T000000Z"
    runner = root / "00_environment" / "probe"
    assert (
        run(
            argparse.Namespace(
                output_dir=runner,
                name="probe",
                source_commit="a" * 40,
                config_json="{}",
                cwd=tmp_path,
                eligibility_status="environment_only",
                timing_eligibility="not_a_benchmark",
                expected_exit_codes=(0,),
                command=[sys.executable, "-c", "print('ok')"],
            )
        )
        == 0
    )
    manifest = finalize(
        argparse.Namespace(
            run_root=root,
            outcomes_json=json.dumps(
                {
                    "evidence": "pass",
                    "raw_remainder": "RAW_REMAINDER_ROOT_CAUSE_CLOSED",
                }
            ),
        )
    )
    assert manifest["runner_count"] == 1
    verification = json.loads((root / "verification.json").read_text())
    claims = validate_verification_document(verification, source_root=root)
    assert claims[0].status == "pass"
    lines = (root / "SHA256SUMS").read_text().splitlines()
    assert lines
    assert all(not line.split("  ", 1)[1].startswith("/") for line in lines)


def test_finalizer_is_byte_deterministic_for_frozen_runner_inputs(tmp_path: Path) -> None:
    first = tmp_path / "first" / "frozen-run"
    runner = first / "00_environment" / "probe"
    assert (
        run(
            argparse.Namespace(
                output_dir=runner,
                name="probe",
                source_commit="b" * 40,
                config_json="{}",
                cwd=tmp_path,
                eligibility_status="environment_only",
                timing_eligibility="not_a_benchmark",
                expected_exit_codes=(0,),
                command=[sys.executable, "-c", "print('frozen')"],
            )
        )
        == 0
    )
    second = tmp_path / "second" / "frozen-run"
    shutil.copytree(first, second)
    arguments = json.dumps({"evidence": "EVIDENCE_INTEGRITY_PASS"})
    finalize(argparse.Namespace(run_root=first, outcomes_json=arguments))
    finalize(argparse.Namespace(run_root=second, outcomes_json=arguments))
    assert (first / "manifest.json").read_bytes() == (second / "manifest.json").read_bytes()
    assert (first / "verification.json").read_bytes() == (second / "verification.json").read_bytes()
    assert (first / "SHA256SUMS").read_bytes() == (second / "SHA256SUMS").read_bytes()


def test_finalizer_classifies_diffreach_interpreter_path_as_provenance(
    tmp_path: Path,
) -> None:
    root = tmp_path / "20260811T000000Z"
    runner = root / "04_native_diffreach" / "official_vdp"
    assert (
        run(
            argparse.Namespace(
                output_dir=runner,
                name="stock_diffreach",
                source_commit="c" * 40,
                config_json="{}",
                cwd=tmp_path,
                eligibility_status="native_capability_only",
                timing_eligibility="not_a_benchmark",
                expected_exit_codes=(0,),
                command=[sys.executable, "-c", "print('ok')"],
            )
        )
        == 0
    )
    summary = runner / "artifacts" / "run" / "summary.json"
    summary.parent.mkdir(parents=True)
    summary.write_text(
        json.dumps(
            {
                "python_executable": {
                    "invoked_path": "/srv/local/shengenli/pinned/bin/python"
                }
            }
        ),
        encoding="utf-8",
    )
    manifest = finalize(
        argparse.Namespace(run_root=root, outcomes_json=json.dumps({"evidence": "pass"}))
    )
    audit = manifest["private_path_audit"]
    assert audit["status"] == "qualified"
    assert all(row["category"] == "provenance_only" for row in audit["matches"])
    assert any(
        row["path"]
        == "04_native_diffreach/official_vdp/artifacts/run/summary.json"
        for row in audit["matches"]
    )
