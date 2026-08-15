#!/usr/bin/env python3
"""Detached GitHub-clone acceptance for the frozen G2 scientific SHA."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any, Sequence
import xml.etree.ElementTree as ET


DEFAULT_REPOSITORY = "https://github.com/lsnnnnnnnn/torch_tm_flowpipe.git"
FOCUSED = (
    "tests/test_g2_shared_column.py",
    "tests/test_bounded_source_ledger.py",
    "tests/test_terminal_checkpoint_v2.py",
    "tests/test_batched_dense_runner_contract.py",
)


def run(argv: Sequence[str], cwd: Path) -> dict[str, Any]:
    started = time.perf_counter()
    completed = subprocess.run(list(argv), cwd=cwd, capture_output=True, text=True)
    result = {
        "argv": list(argv),
        "exit_code": completed.returncode,
        "runtime_s": time.perf_counter() - started,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    if completed.returncode != 0:
        raise RuntimeError(f"fresh-clone command failed: {argv}\n{completed.stdout}\n{completed.stderr}")
    return result


def counts(path: Path) -> dict[str, int]:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    return {
        key: sum(int(suite.attrib.get(key, 0)) for suite in suites)
        for key in ("tests", "failures", "errors", "skipped")
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scientific-sha", required=True)
    parser.add_argument("--package-relative", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repository", default=DEFAULT_REPOSITORY)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    commands = []
    with tempfile.TemporaryDirectory(prefix="vdp-g2-fresh-clone-") as temporary:
        temporary_root = Path(temporary)
        clone = temporary_root / "repo"
        commands.append(run(("git", "clone", "--no-checkout", args.repository, str(clone)), temporary_root))
        commands.append(run(("git", "checkout", "--detach", args.scientific_sha), clone))
        head = run(("git", "rev-parse", "HEAD"), clone)["stdout"].strip()
        if head != args.scientific_sha:
            raise RuntimeError("detached checkout SHA mismatch")
        commands.append(run((sys.executable, "-m", "compileall", "-q", "src", "experiments", "tests"), clone))
        focused_xml = temporary_root / "focused.xml"
        commands.append(run((sys.executable, "-m", "pytest", "-q", *FOCUSED, f"--junitxml={focused_xml}"), clone))
        full_xml = temporary_root / "full.xml"
        commands.append(run((sys.executable, "-m", "pytest", "-q", f"--junitxml={full_xml}"), clone))
        blackbox = temporary_root / "blackbox.json"
        oracle = temporary_root / "oracle.json"
        commands.append(run((sys.executable, "experiments/export_g2_blackbox_coefficients.py", "--output", str(blackbox)), clone))
        commands.append(run((sys.executable, "experiments/independent_g2_exact_oracle.py", "--input", str(blackbox), "--output", str(oracle)), clone))
        commands.append(run((sys.executable, "experiments/verify_vdp_g2_evidence_20260815.py", "--package", str(clone / args.package_relative)), clone))
        porcelain = run(("git", "status", "--porcelain"), clone)["stdout"]
        result = {
            "schema": "vdp_g2_fresh_clone_acceptance_v1",
            "status": "PASS",
            "repository": args.repository,
            "scientific_sha": args.scientific_sha,
            "detached_head": head,
            "compileall_passed": True,
            "focused": counts(focused_xml),
            "full": counts(full_xml),
            "independent_oracle": json.loads(oracle.read_text(encoding="utf-8")),
            "package_verifier_passed": True,
            "git_status_porcelain_empty": porcelain == "",
            "commands": commands,
        }
        if not result["git_status_porcelain_empty"]:
            raise RuntimeError(f"fresh clone dirty after acceptance: {porcelain!r}")
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "scientific_sha": result["scientific_sha"],
        "focused": result["focused"],
        "full": result["full"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
