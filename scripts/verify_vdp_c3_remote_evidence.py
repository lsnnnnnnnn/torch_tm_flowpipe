#!/usr/bin/env python3
"""Recompute and verify the committed VDP C3 evidence package.

The verifier intentionally derives every scientific gate, test count, and
reported status from the raw CSV/JSON/JUnit XML stored in the package.  The
only fixed inputs are the review contract and its pinned source identities.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACKAGE = ROOT / "artifacts/runs/vdp_c3_cross_step_causal_closure_20260827"
EMPTY_DIFF_SHA256 = hashlib.sha256(b"").hexdigest()
CHANNELS = ("endpoint_x", "endpoint_y", "tube_x", "tube_y")
WIDTH_PATHS = (
    ("raw_endpoint", "x_width", "x_lo", "x_hi"),
    ("raw_endpoint", "y_width", "y_lo", "y_hi"),
    ("last_segment", "x_width", "x_lo", "x_hi"),
    ("last_segment", "y_width", "y_lo", "y_hi"),
)


class EvidenceError(ValueError):
    """Raised when raw evidence is malformed rather than merely gate-failing."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"cannot read JSON {path}: {exc}") from exc


def read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    except (OSError, csv.Error) as exc:
        raise EvidenceError(f"cannot read CSV {path}: {exc}") from exc


def finite_float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise EvidenceError(f"{label} is not numeric: {value!r}") from exc
    if not math.isfinite(result):
        raise EvidenceError(f"{label} is nonfinite: {value!r}")
    return result


def integer(value: Any, label: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise EvidenceError(f"{label} is not an integer: {value!r}") from exc
    return result


def close(left: Any, right: Any, tolerance: float) -> bool:
    return abs(finite_float(left, "left value") - finite_float(right, "right value")) <= tolerance


def torch_widths(summary: Mapping[str, Any], tolerance: float, label: str) -> list[float]:
    widths: list[float] = []
    for owner, width_key, lo_key, hi_key in WIDTH_PATHS:
        block = summary.get(owner)
        if not isinstance(block, Mapping):
            raise EvidenceError(f"{label}.{owner} is missing")
        width = finite_float(block.get(width_key), f"{label}.{owner}.{width_key}")
        lo = finite_float(block.get(lo_key), f"{label}.{owner}.{lo_key}")
        hi = finite_float(block.get(hi_key), f"{label}.{owner}.{hi_key}")
        if width < 0 or abs(width - (hi - lo)) > tolerance:
            raise EvidenceError(f"{label}.{owner}.{width_key} is inconsistent with hi-lo")
        widths.append(width)
    return widths


def flowstar_widths(row: Mapping[str, str], label: str) -> list[float]:
    widths: list[float] = []
    for prefix in ("endpoint_x", "endpoint_y", "segment_x", "segment_y"):
        lo = finite_float(row.get(f"{prefix}_lo"), f"{label}.{prefix}_lo")
        hi = finite_float(row.get(f"{prefix}_hi"), f"{label}.{prefix}_hi")
        if hi < lo:
            raise EvidenceError(f"{label}.{prefix} has inverted bounds")
        widths.append(hi - lo)
    return widths


def _junit_tree(path: Path) -> tuple[ET.Element, list[ET.Element], list[ET.Element]]:
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as exc:
        raise EvidenceError(f"cannot read JUnit XML {path}: {exc}") from exc
    if root.tag not in {"testsuite", "testsuites"}:
        raise EvidenceError(f"unexpected JUnit root element: {root.tag}")
    suites = [root] if root.tag == "testsuite" else list(root.findall(".//testsuite"))
    cases = list(root.findall(".//testcase"))
    if root.tag == "testsuite":
        cases = list(root.findall(".//testcase"))
    return root, suites, cases


def _junit_case_ids(path: Path) -> list[str]:
    _, _, cases = _junit_tree(path)
    return sorted(
        "::".join(
            filter(
                None,
                (case.get("file", ""), case.get("classname", ""), case.get("name", "")),
            )
        )
        for case in cases
    )


def parse_junit(path: Path) -> dict[str, Any]:
    root, suites, cases = _junit_tree(path)
    case_ids = sorted(
        "::".join(
            filter(
                None,
                (case.get("file", ""), case.get("classname", ""), case.get("name", "")),
            )
        )
        for case in cases
    )
    duplicate_ids = len(case_ids) != len(set(case_ids))
    failures = sum(case.find("failure") is not None for case in cases)
    errors = sum(case.find("error") is not None for case in cases)
    skipped = sum(case.find("skipped") is not None for case in cases)
    count_consistent = True
    for suite in suites:
        suite_cases = list(suite.findall("./testcase"))
        if suite.get("tests") is not None:
            count_consistent &= integer(suite.get("tests"), "JUnit suite tests") == len(suite_cases)
        for key, tag in (("failures", "failure"), ("errors", "error"), ("skipped", "skipped")):
            if suite.get(key) is not None:
                observed = sum(case.find(tag) is not None for case in suite_cases)
                count_consistent &= integer(suite.get(key), f"JUnit suite {key}") == observed
    if root.tag == "testsuites":
        for key, observed in (
            ("tests", len(cases)), ("failures", failures), ("errors", errors), ("skipped", skipped)
        ):
            if root.get(key) is not None:
                count_consistent &= integer(root.get(key), f"JUnit root {key}") == observed
    total = len(cases)
    return {
        "total": total,
        "passed": total - failures - errors - skipped,
        "failed": failures,
        "errors": errors,
        "skipped": skipped,
        "case_ids_sha256": canonical_sha256(case_ids),
        "counts_consistent": count_consistent,
        "duplicate_case_ids": duplicate_ids,
    }


def verify_checksums(package: Path) -> list[str]:
    errors: list[str] = []
    checksum_path = package / "SHA256SUMS"
    if not checksum_path.is_file():
        return ["SHA256SUMS missing"]
    expected: dict[str, str] = {}
    for line in checksum_path.read_text(encoding="ascii").splitlines():
        digest, separator, relative = line.partition("  ")
        if (
            not separator
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or not relative
            or relative in expected
        ):
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


def verify_manifest(package: Path) -> list[str]:
    errors: list[str] = []
    manifest = read_json(package / "MANIFEST.json")
    if manifest.get("schema") != "torch_tm_flowpipe.vdp_c3_remote_evidence_manifest/1":
        return ["manifest schema mismatch"]
    rows = manifest.get("raw_files")
    if not isinstance(rows, list) or not rows:
        return ["manifest raw_files missing"]
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            errors.append("manifest raw-file row is not an object")
            continue
        relative = row.get("path")
        if not isinstance(relative, str) or not relative.startswith("raw/") or relative in seen:
            errors.append(f"invalid or duplicate manifest raw path: {relative!r}")
            continue
        seen.add(relative)
        path = package / relative
        if not path.is_file():
            errors.append(f"manifest raw file missing: {relative}")
            continue
        if integer(row.get("size"), f"manifest size {relative}") != path.stat().st_size:
            errors.append(f"manifest size mismatch: {relative}")
        if row.get("sha256") != sha256(path):
            errors.append(f"manifest hash mismatch: {relative}")
    actual = {
        path.relative_to(package).as_posix()
        for path in (package / "raw").rglob("*")
        if path.is_file()
    }
    if actual != seen:
        errors.append("manifest raw-file coverage mismatch")
    return errors


def _source_gate(
    package: Path,
    contract: Mapping[str, Any],
    observed_torch_shas: Mapping[str, set[str]],
) -> tuple[bool, dict[str, Any]]:
    expected = contract["source_shas"]
    huan = read_json(package / "raw/source/huan_run_index.json")
    flowstar = read_json(package / "raw/native/flowstar/summary.json")
    observed = {
        "c3_scientific": sorted(observed_torch_shas["c3"]),
        "c2_scientific": sorted(observed_torch_shas["c2"]),
        "huan_repaired": huan.get("engine_head"),
        "flowstar": flowstar.get("source_commit"),
    }
    gate = (
        observed["c3_scientific"] == [expected["c3_scientific"]]
        and observed["c2_scientific"] == [expected["c2_scientific"]]
        and observed["huan_repaired"] == expected["huan_repaired"]
        and observed["flowstar"] == expected["flowstar"]
    )
    return gate, observed


def recompute(package: Path) -> dict[str, Any]:
    contract = read_json(package / "EVIDENCE_CONTRACT.json")
    if contract.get("schema") != "torch_tm_flowpipe.vdp_c3_remote_evidence_contract/1":
        raise EvidenceError("evidence contract schema mismatch")
    gate_contract = contract.get("gates")
    horizons = contract.get("fixed_horizons")
    if not isinstance(gate_contract, Mapping) or not isinstance(horizons, Mapping):
        raise EvidenceError("gate or horizon contract missing")
    width_tolerance = finite_float(gate_contract.get("width_tolerance"), "width tolerance")
    horizon_tolerance = finite_float(gate_contract.get("horizon_tolerance"), "horizon tolerance")
    recovery_min = finite_float(gate_contract.get("t6_recovery_min"), "T6 recovery minimum")
    runtime_max = finite_float(gate_contract.get("runtime_ratio_max"), "runtime ratio maximum")
    if min(width_tolerance, horizon_tolerance, recovery_min, runtime_max) < 0:
        raise EvidenceError("gate thresholds must be nonnegative")

    fixed: dict[str, Any] = {}
    observed_torch_shas: dict[str, set[str]] = {"c2": set(), "c3": set()}
    completed_gate = True
    for label, requested_value in horizons.items():
        requested = finite_float(requested_value, f"fixed horizon {label}")
        flow_base = package / f"raw/fixed/flowstar/{label}"
        flow_summary = read_json(flow_base / "summary.json")
        flow_rows = read_csv(flow_base / "stock.csv")
        if not flow_rows:
            raise EvidenceError(f"Flow* {label} stock.csv is empty")
        flow_accepted = integer(flow_summary.get("accepted_steps"), f"Flow* {label} accepted")
        flow_requested = integer(flow_summary.get("requested_steps"), f"Flow* {label} requested")
        final_flow = flow_rows[-1]
        flow_complete = (
            len(flow_rows) == flow_accepted == flow_requested
            and integer(final_flow.get("step"), f"Flow* {label} final step") == flow_accepted
            and close(final_flow.get("t_after"), requested, horizon_tolerance)
            and integer(final_flow.get("status_code"), f"Flow* {label} status")
            == integer(flow_summary.get("result_status_code"), f"Flow* {label} summary status")
        )
        record: dict[str, Any] = {
            "requested_horizon": requested,
            "flowstar": {
                "widths": dict(zip(CHANNELS, flowstar_widths(final_flow, f"Flow* {label}"))),
                "accepted": flow_accepted,
                "completed": flow_complete,
            },
        }
        for lane in ("c2", "c3"):
            base = package / f"raw/fixed/torch_{lane}/{label}"
            summary = read_json(base / "summary.json")
            command = read_json(base / "command.json")
            summary_sha = summary.get("commit")
            command_sha = command.get("commit")
            if isinstance(summary_sha, str):
                observed_torch_shas[lane].add(summary_sha)
            if isinstance(command_sha, str):
                observed_torch_shas[lane].add(command_sha)
            lane_complete = (
                summary.get("status") == "completed"
                and summary.get("completed_requested_horizon") is True
                and close(summary.get("completed_horizon"), requested, horizon_tolerance)
                and integer(summary.get("accepted_steps"), f"Torch {lane} {label} accepted") == flow_accepted
                and integer(summary.get("rejected_attempts"), f"Torch {lane} {label} rejected") == 0
                and summary_sha == command_sha
                and command.get("tracked_diff_sha256") == EMPTY_DIFF_SHA256
                and command.get("worktree_status") == ""
            )
            record[lane] = {
                "widths": dict(
                    zip(CHANNELS, torch_widths(summary, width_tolerance, f"Torch {lane} {label}"))
                ),
                "runtime_s": finite_float(summary.get("runtime_s"), f"Torch {lane} {label} runtime"),
                "accepted": integer(summary.get("accepted_steps"), f"Torch {lane} {label} accepted"),
                "rejected": integer(summary.get("rejected_attempts"), f"Torch {lane} {label} rejected"),
                "completed": lane_complete,
                "source_sha": summary_sha,
            }
            completed_gate &= lane_complete
        completed_gate &= flow_complete
        fixed[label] = record

    no_regression: dict[str, bool] = {}
    for label in gate_contract.get("no_regression_horizons", []):
        if label not in fixed:
            raise EvidenceError(f"no-regression horizon is absent: {label}")
        no_regression[label] = all(
            fixed[label]["c3"]["widths"][channel]
            <= fixed[label]["c2"]["widths"][channel] + width_tolerance
            for channel in CHANNELS
        )

    recovery_label = str(gate_contract.get("recovery_horizon"))
    if recovery_label not in fixed:
        raise EvidenceError(f"recovery horizon is absent: {recovery_label}")
    recovery: dict[str, float] = {}
    for channel in CHANNELS:
        c2 = fixed[recovery_label]["c2"]["widths"][channel]
        c3 = fixed[recovery_label]["c3"]["widths"][channel]
        flowstar = fixed[recovery_label]["flowstar"]["widths"][channel]
        denominator = c2 - flowstar
        if denominator <= 0:
            raise EvidenceError(f"nonpositive recovery denominator for {channel}")
        recovery[channel] = (c2 - c3) / denominator
    recovery_gate = all(value >= recovery_min for value in recovery.values())

    runtime_ratio: dict[str, float] = {}
    for label, record in fixed.items():
        c2_runtime = record["c2"]["runtime_s"]
        c3_runtime = record["c3"]["runtime_s"]
        if c2_runtime <= 0 or c3_runtime <= 0:
            raise EvidenceError(f"nonpositive runtime at {label}")
        runtime_ratio[label] = c3_runtime / c2_runtime
    runtime_gate = all(value <= runtime_max for value in runtime_ratio.values())

    native: dict[str, Any] = {}
    native_gate = True
    for lane in ("c2", "c3"):
        base = package / f"raw/native/torch_{lane}"
        summary = read_json(base / "summary.json")
        command = read_json(base / "command.json")
        attempts = read_csv(base / "attempts.csv")
        segments = read_csv(base / "segments.csv")
        accepted_attempts = sum(row.get("validation_status") == "validated" for row in attempts)
        rejected_attempts = sum(row.get("validation_status") == "failed" for row in attempts)
        accepted_segments = [row for row in segments if row.get("status") == "accepted"]
        step_rejections = sum(integer(row.get("step_rejections"), "segment step_rejections") for row in segments)
        if not accepted_segments:
            raise EvidenceError(f"native {lane} has no accepted segment")
        completed_horizon = finite_float(summary.get("completed_horizon"), f"native {lane} horizon")
        last_accepted_horizon = finite_float(accepted_segments[-1].get("t_hi"), f"native {lane} last t_hi")
        summary_accepted = integer(summary.get("accepted_steps"), f"native {lane} accepted")
        summary_rejected = integer(summary.get("rejected_attempts"), f"native {lane} rejected")
        internal_consistency = (
            summary_accepted == accepted_attempts == len(accepted_segments)
            and summary_rejected == rejected_attempts == step_rejections
            and close(completed_horizon, last_accepted_horizon, horizon_tolerance)
            and summary.get("commit") == command.get("commit")
            and command.get("tracked_diff_sha256") == EMPTY_DIFF_SHA256
            and command.get("worktree_status") == ""
        )
        source_sha = summary.get("commit")
        if isinstance(source_sha, str):
            observed_torch_shas[lane].add(source_sha)
        if isinstance(command.get("commit"), str):
            observed_torch_shas[lane].add(command["commit"])
        native[lane] = {
            "completed_horizon": completed_horizon,
            "completed_requested_horizon": summary.get("completed_requested_horizon"),
            "accepted": accepted_attempts,
            "rejected": rejected_attempts,
            "runtime_s": finite_float(summary.get("runtime_s"), f"native {lane} runtime"),
            "status": summary.get("status"),
            "source_sha": source_sha,
            "raw_counts_consistent": internal_consistency,
        }
        expected = contract["native_expectations"][lane]
        lane_gate = (
            internal_consistency
            and close(completed_horizon, expected["completed_horizon"], horizon_tolerance)
            and summary.get("completed_requested_horizon") is expected["completed_requested_horizon"]
            and accepted_attempts == integer(expected["accepted"], f"expected {lane} accepted")
            and rejected_attempts == integer(expected["rejected"], f"expected {lane} rejected")
            and summary.get("status") == expected["status"]
        )
        native_gate &= lane_gate

    flowstar_native = read_json(package / "raw/native/flowstar/summary.json")
    flowstar_expected = contract["native_expectations"]["flowstar"]
    flowstar_native_gate = (
        close(flowstar_native.get("horizon_validated"), flowstar_expected["completed_horizon"], horizon_tolerance)
        and integer(flowstar_native.get("accepted_segments"), "Flow* native accepted")
        == integer(flowstar_expected["accepted"], "expected Flow* accepted")
        and flowstar_native.get("result_status") == flowstar_expected["status"]
    )
    native["flowstar"] = {
        "completed_horizon": finite_float(flowstar_native.get("horizon_validated"), "Flow* native horizon"),
        "accepted": integer(flowstar_native.get("accepted_segments"), "Flow* native accepted"),
        "status": flowstar_native.get("result_status"),
        "source_sha": flowstar_native.get("source_commit"),
    }
    native_gate &= flowstar_native_gate

    source_gate, source_observed = _source_gate(package, contract, observed_torch_shas)
    tests = parse_junit(package / "raw/tests/pytest.xml")
    tests_gate = (
        tests["total"] > 0
        and tests["failed"] == 0
        and tests["errors"] == 0
        and tests["counts_consistent"]
        and not tests["duplicate_case_ids"]
    )

    gates = {
        "fixed_runs_complete": completed_gate,
        "T1_T3_no_regression": bool(no_regression) and all(no_regression.values()),
        "T6p32_recovery": recovery_gate,
        "runtime": runtime_gate,
        "native_horizon_and_counts": native_gate,
        "source_shas": source_gate,
        "tests": tests_gate,
    }
    highest_status = (
        contract["statuses"]["passed"]
        if all(gates.values())
        else contract["statuses"]["failed_stop"]
    )
    return {
        "schema": "torch_tm_flowpipe.vdp_c3_remote_evidence_result/1",
        "highest_status": highest_status,
        "gates": gates,
        "no_regression_by_horizon": no_regression,
        "t6_recovery_by_channel": recovery,
        "runtime_ratio_c3_over_c2": runtime_ratio,
        "fixed": fixed,
        "native": native,
        "source_shas_observed": source_observed,
        "source_shas_expected": contract["source_shas"],
        "tests": tests,
    }


def verify_git_commits(contract: Mapping[str, Any], repository: Path) -> list[str]:
    errors: list[str] = []
    if not (repository / ".git").exists():
        return [f"Git metadata missing at {repository}"]
    for key in ("c3_scientific", "c3_assembly", "c2_scientific", "c2_package"):
        source_sha = contract["source_shas"].get(key)
        result = subprocess.run(
            ["git", "cat-file", "-e", f"{source_sha}^{{commit}}"],
            cwd=repository,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode:
            errors.append(f"pinned Torch commit unavailable: {key}={source_sha}")
    return errors


def rerun_tests(package: Path, expected: Mapping[str, Any], repository: Path) -> list[str]:
    errors: list[str] = []
    with tempfile.TemporaryDirectory(prefix="vdp-c3-evidence-tests-") as temporary:
        junit = Path(temporary) / "pytest.xml"
        command = [sys.executable, "-m", "pytest", "-q", f"--junitxml={junit}"]
        environment = os.environ.copy()
        environment["PATH"] = os.pathsep.join(
            (str(Path(sys.executable).resolve().parent), environment.get("PATH", ""))
        )
        environment["PYTHONPATH"] = os.pathsep.join(
            (str(repository / "src"), environment.get("PYTHONPATH", ""))
        )
        result = subprocess.run(
            command,
            cwd=repository,
            env=environment,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return [f"fresh pytest failed with exit code {result.returncode}"]
        observed = parse_junit(junit)
        expected_ids = set(_junit_case_ids(package / "raw/tests/pytest.xml"))
        observed_ids = set(_junit_case_ids(junit))
    if observed["failed"] or observed["errors"]:
        errors.append("fresh pytest contains a failed or errored testcase")
    missing = sorted(expected_ids - observed_ids)
    if missing:
        errors.append(
            f"fresh pytest is missing {len(missing)} recorded testcase identities: {missing[:3]}"
        )
    if observed["total"] < expected["total"]:
        errors.append(
            f"fresh pytest test count regressed: {observed['total']} < {expected['total']}"
        )
    return errors


def verify(
    package: Path,
    *,
    repository: Path = ROOT,
    check_git: bool = True,
    run_tests: bool = False,
) -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    if not package.is_dir():
        return None, [f"package directory missing: {package}"]
    try:
        errors.extend(verify_checksums(package))
        errors.extend(verify_manifest(package))
        contract = read_json(package / "EVIDENCE_CONTRACT.json")
        if check_git:
            errors.extend(verify_git_commits(contract, repository))
        recomputed = recompute(package)
        claimed = read_json(package / "RESULT.json")
        if claimed != recomputed:
            errors.append("RESULT.json does not equal the raw-evidence recomputation")
        if run_tests:
            errors.extend(rerun_tests(package, recomputed["tests"], repository))
    except (EvidenceError, KeyError, TypeError) as exc:
        errors.append(str(exc))
        recomputed = None
    return recomputed, errors


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument("--repository", type=Path, default=ROOT)
    parser.add_argument("--no-git", action="store_true", help="skip local Git object checks")
    parser.add_argument("--run-tests", action="store_true", help="rerun pytest and compare testcase identities")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result, errors = verify(
        args.package.resolve(),
        repository=args.repository.resolve(),
        check_git=not args.no_git,
        run_tests=args.run_tests,
    )
    print(
        json.dumps(
            {"ok": not errors, "errors": errors, "recomputed": result},
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
