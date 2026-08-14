#!/usr/bin/env python3
"""Prove raw-number, status, file, and checksum tampering fail verification."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence


def _run(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=False)


def _refinalize(copy: Path, repo: Path) -> None:
    manifest = json.loads((copy / "manifest.json").read_text(encoding="utf-8"))
    test_status_path = copy / "13_tests/status.json"
    fresh_status_path = copy / "14_fresh_clone/status.json"
    test_status = (
        json.loads(test_status_path.read_text(encoding="utf-8")).get("status", "pending")
        if test_status_path.is_file()
        else "pending"
    )
    fresh_status = (
        json.loads(fresh_status_path.read_text(encoding="utf-8")).get("status", "pending")
        if fresh_status_path.is_file()
        else "pending"
    )
    command = [
        sys.executable, "experiments/finalize_step1_stage_oracle_package.py",
        "--package", str(copy),
        "--branch", manifest["branch"],
        "--tests-status", test_status,
        "--fresh-clone-status", fresh_status,
    ]
    if manifest.get("scientific_sha"):
        command.extend(("--scientific-sha", manifest["scientific_sha"]))
    if manifest.get("attestation_tip"):
        command.extend(("--attestation-tip", manifest["attestation_tip"]))
    completed = _run(command, repo)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr)


def run(args: argparse.Namespace) -> dict[str, Any]:
    package = args.package.resolve()
    repo = args.repo.resolve()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    cases: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="step1-package-tamper-") as temporary:
        root = Path(temporary)
        for case in ("raw_number", "status", "file", "checksum"):
            target = root / case
            shutil.copytree(package, target)
            # The evidence runner may create its own output files inside the
            # source package before this process starts.  Establish a clean,
            # self-consistent copied baseline so checksum rejection below is
            # attributable to the requested mutation rather than stale sums.
            _refinalize(target, repo)
            if case == "raw_number":
                path = target / "04_torch_actual_stage_ledger/export_retry/artifacts/stage_ledger.json"
                value = json.loads(path.read_text(encoding="utf-8"))
                coefficient = value["rows"][0]["payload"]["components"][0]["terms"][0]["coefficient"]
                coefficient["hex"] = "0x1.0000000000000p+0"
                path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                _refinalize(target, repo)
            elif case == "status":
                path = target / "07_stage_swap_matrix/audit_retry/artifacts/candidate_decision.json"
                value = json.loads(path.read_text(encoding="utf-8"))
                value["l1"] = "SOUND_LOCAL_OPERATOR_CANDIDATE_L1"
                path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                _refinalize(target, repo)
            elif case == "file":
                (target / "03_flowstar_actual_stage_ledger/03_process/artifacts/stage_ledger.json").unlink()
                _refinalize(target, repo)
            elif case == "checksum":
                path = target / "SHA256SUMS"
                text = path.read_text(encoding="utf-8")
                path.write_text(("0" if text[0] != "0" else "1") + text[1:], encoding="utf-8")
            command = [
                sys.executable, "experiments/verify_step1_stage_oracle_package.py",
                "--package", str(target),
                "--repo", str(repo),
            ]
            completed = _run(command, repo)
            cases.append(
                {
                    "case": case,
                    "verifier_exit_code": completed.returncode,
                    "rejected": completed.returncode != 0,
                    "stderr_tail": completed.stderr[-2000:],
                }
            )
    passed = all(row["rejected"] for row in cases)
    result = {
        "schema": "step1_package_tamper_tests_v1",
        "cases": cases,
        "passed": passed,
    }
    (output / "tamper_tests.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "summary.json").write_text(
        json.dumps({"schema": "step1_package_tamper_summary_v1", "passed": passed}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not passed:
        raise ValueError("one or more tampered packages passed verification")
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    result = run(parse_args(argv))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
