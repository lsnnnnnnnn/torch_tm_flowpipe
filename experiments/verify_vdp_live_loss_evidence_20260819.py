#!/usr/bin/env python3
"""Fail-closed verifier for the packaged VDP live-loss/C1 evidence."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence
import xml.etree.ElementTree as ET

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from tamper_test_vdp_live_loss_ablation_20260819 import run as run_tamper
from verify_vdp_live_loss_ablation_20260819 import verify as verify_gates


SCIENTIFIC_SHA = "dbe03dcdfbf2f36b1d58013373d1d235ace1a48e"
EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


class EvidenceError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EvidenceError(message)


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _summary(path: Path) -> Mapping[str, Any]:
    row = _load(path)
    _require(row["commit"] == SCIENTIFIC_SHA, f"scientific SHA: {path}")
    _require(row["worktree_dirty"] is False, f"dirty scientific run: {path}")
    _require(row["tracked_diff_sha256"] == EMPTY_SHA256, f"tracked diff: {path}")
    _require(row["endpoint_repair_used"] is False, f"endpoint repair: {path}")
    _require(row["endpoint_tightening_used"] is False, f"endpoint tightening: {path}")
    return row


def _widths(summary: Mapping[str, Any]) -> dict[str, float]:
    return {
        "endpoint_x": float(summary["raw_endpoint"]["x_width"]),
        "endpoint_y": float(summary["raw_endpoint"]["y_width"]),
        "segment_x": float(summary["last_segment"]["x_width"]),
        "segment_y": float(summary["last_segment"]["y_width"]),
    }


def verify(root: Path) -> dict[str, Any]:
    root = root.resolve()
    manifest = _load(root / "manifest.json")
    _require(manifest["schema"] == "vdp_live_loss_c1_evidence_manifest_v1", "manifest schema")
    _require(manifest["scientific_sha"] == SCIENTIFIC_SHA, "manifest scientific SHA")
    expected_paths = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and path.name not in {"manifest.json", "SHA256SUMS"}
    }
    rows = manifest["files"]
    _require(len(rows) == manifest["file_count"], "manifest file count field")
    _require({row["path"] for row in rows} == expected_paths, "manifest file coverage")
    _require(len(expected_paths) == manifest["file_count"], "manifest file count")
    total = 0
    for row in rows:
        path = root / row["path"]
        _require(path.stat().st_size == int(row["bytes"]), f"file size: {row['path']}")
        _require(_sha(path) == row["sha256"], f"file hash: {row['path']}")
        total += path.stat().st_size
    _require(total == manifest["total_bytes"], "manifest total bytes")
    sums = {}
    for line in (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        sums[relative] = digest
    _require(sums == {row["path"]: row["sha256"] for row in rows}, "SHA256SUMS")

    provenance = _load(root / "00_provenance/provenance.json")
    _require(provenance["scientific_sha"] == SCIENTIFIC_SHA, "provenance scientific SHA")
    _require(provenance["base_sha"] == "2cdb7a9509d5908baef79a02cfde18ea0682430c", "base SHA")
    _require(provenance["scientific_runs_all_report_clean"] is True, "clean run claim")
    _require(provenance["scientific_tracked_diff_sha256"] == EMPTY_SHA256, "clean diff claim")

    gate_result = verify_gates(root / "01_gates")
    tamper_result = run_tamper(root / "01_gates")
    _require(tamper_result["passed"] is True and len(tamper_result["cases"]) == 5, "tamper suite")

    matrix = _load(root / "02_scientific_matrix/matrix.json")
    _require(matrix["schema"] == "vdp_live_loss_c1_scientific_matrix_v1", "matrix schema")
    _require(matrix["scientific_sha"] == SCIENTIFIC_SHA, "matrix scientific SHA")
    gates = matrix["gates"]
    for name in (
        "gate_a",
        "gate_b",
        "gate_c",
        "legacy_h1_h2_step1_bitwise_unchanged",
        "T6p32_no_channel_regression_vs_current_h1_h2",
        "native_at_least_6p482041958201616",
        "runtime_at_most_2x_legacy",
        "v100_candidate_consistent_at_1e_12",
    ):
        _require(gates[name] is True, f"matrix passing gate: {name}")
    _require(gates["T1_T3_all_four_channels_remove_10pct_legacy_excess"] is False, "early failure")
    _require(gates["reaches_T10_stretch"] is False, "T10 stretch failure")
    _require(
        matrix["decision"]
        == "C1_SOUND_AND_PRODUCTION_USEFUL__OVERALL_T1_T3_SUCCESS_FAILED__T10_STRETCH_FAILED",
        "matrix decision",
    )

    raw = root / "03_raw_runs"
    expected_modes = {
        "legacy": ("normalized_insertion", "flowstar_raw_remainder_compat"),
        "h1": ("normalized_insertion_dependency_preserving", "flowstar_raw_remainder_compat"),
        "h1_h2": (
            "normalized_insertion_dependency_preserving",
            "flowstar_raw_remainder_compat_factorized_joint",
        ),
        "candidate": (
            "normalized_insertion_dependency_preserving",
            "flowstar_raw_remainder_compat_factorized_joint_closure",
        ),
    }
    for lane, (reset, mode) in expected_modes.items():
        summary = _summary(raw / f"step1/{lane}/summary.json")
        _require(summary["reset_mode"] == reset and summary["validation_mode"] == mode, f"step1 lane: {lane}")
        _require(_widths(summary) == matrix["step1"]["widths"][lane], f"step1 widths: {lane}")

    fixed_runs = {
        lane: _summary(raw / f"fixed_T6p32/{lane}/summary.json")
        for lane in ("legacy", "candidate")
    }
    _require(fixed_runs["candidate"]["accepted_steps"] == 632, "candidate fixed accepted steps")
    _require(fixed_runs["candidate"]["rejected_attempts"] == 0, "candidate fixed rejected attempts")
    fixed_ratio = fixed_runs["candidate"]["runtime_s"] / fixed_runs["legacy"]["runtime_s"]
    _require(fixed_ratio == matrix["runtime_ratios"]["fixed_T6p32_candidate_over_legacy"], "fixed runtime ratio")
    _require(fixed_ratio <= 2.0, "fixed runtime gate")
    candidate_rows = {}
    with (raw / "fixed_T6p32/candidate/segments.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            t_hi = float(row["t_hi"])
            if any(abs(t_hi - horizon) <= 1.0e-12 for horizon in (1.0, 3.0, 6.32)):
                candidate_rows[f"T{format(t_hi, 'g').replace('.', 'p')}"] = {
                    "endpoint_x": float(row["endpoint_x_width"]),
                    "endpoint_y": float(row["endpoint_y_width"]),
                    "segment_x": float(row["segment_x_width"]),
                    "segment_y": float(row["segment_y_width"]),
                }
    _require(set(candidate_rows) == {"T1", "T3", "T6p32"}, "fixed checkpoint coverage")
    for horizon, widths in candidate_rows.items():
        for channel, width in widths.items():
            stored = matrix["fixed"][horizon][channel]
            _require(width == stored["candidate_width"], f"candidate checkpoint: {horizon}/{channel}")
            expected_fraction = (stored["legacy_width"] - width) / stored["legacy_excess"]
            _require(expected_fraction == stored["fraction_of_legacy_excess_removed"], f"recovery formula: {horizon}/{channel}")

    native_runs = {
        lane: _summary(raw / f"native_T10/{lane}/summary.json")
        for lane in ("legacy", "candidate")
    }
    native_ratio = native_runs["candidate"]["runtime_s"] / native_runs["legacy"]["runtime_s"]
    _require(native_ratio == matrix["runtime_ratios"]["native_T10_request_candidate_over_legacy"], "native runtime ratio")
    _require(native_runs["candidate"]["completed_horizon"] == 6.589638579126679, "native endpoint")
    _require(native_runs["candidate"]["completed_requested_horizon"] is False, "native T10 failure")
    diagnostic = matrix["native_candidate_terminal_diagnostic"]
    _require(diagnostic["limiting_component"] == "y" and diagnostic["limiting_side"] == "upper", "terminal side")
    _require(diagnostic["largest_additive_ledger_category"] == "composition_overflow", "terminal ledger category")

    consistency = {
        device: _summary(raw / f"consistency_T0p1/{device}/summary.json")
        for device in ("cpu", "cuda")
    }
    _require(_widths(consistency["cpu"]) == _widths(consistency["cuda"]), "CPU/V100 widths")
    _require(matrix["v100_consistency"]["scope"].startswith("implementation consistency only"), "V100 scope")

    xml_counts = {}
    for name in ("targeted.xml", "full.xml", "final_all.xml"):
        xml_root = ET.parse(root / "04_tests" / name).getroot()
        suites = [xml_root] if xml_root.tag == "testsuite" else list(xml_root.findall("testsuite"))
        tests = sum(int(suite.attrib.get("tests", 0)) for suite in suites)
        failures = sum(int(suite.attrib.get("failures", 0)) for suite in suites)
        errors = sum(int(suite.attrib.get("errors", 0)) for suite in suites)
        _require(failures == 0 and errors == 0, f"test failures: {name}")
        xml_counts[name] = tests
    _require(xml_counts["targeted.xml"] >= 30, "targeted test count")
    _require(xml_counts["full.xml"] >= 786, "full test count")
    _require(xml_counts["final_all.xml"] >= 789, "final package-inclusive test count")

    result = {
        "status": "verified",
        "files": manifest["file_count"],
        "bytes": manifest["total_bytes"],
        "scientific_sha": SCIENTIFIC_SHA,
        "gate_events": gate_result["events"],
        "tamper_cases": len(tamper_result["cases"]),
        "matrix_decision": matrix["decision"],
        "tests": xml_counts,
    }
    print(json.dumps(result, sort_keys=True))
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    return parser.parse_args(argv)


if __name__ == "__main__":
    verify(parse_args().root)
