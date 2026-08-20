#!/usr/bin/env python3
"""Fail-closed verifier for the packaged VDP C2 evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence
import xml.etree.ElementTree as ET


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from tamper_test_vdp_c2_refinement_20260820 import run as run_tamper
from tamper_test_vdp_c2_refinement_20260820 import verify_refinement_ledger
from package_vdp_c2_evidence_20260820 import RAW_RUN_EXCLUDED_FILES


EMPTY_DIFF_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
ALLOWED_DECISIONS = {
    "C2_SOUND_AND_T1_T3_TARGET_MET__T10_MET",
    "C2_SOUND_AND_T1_T3_TARGET_MET__T10_NOT_MET",
    "C2_SOUND_AND_PRODUCTION_USEFUL__T1_T3_TARGET_NOT_MET",
    "POST_ACCEPT_REFINEMENT_CAUSAL_GATE_FAILED",
    "C2_SOUNDNESS_OR_PROVENANCE_FAILED",
}


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


def _forbidden_candidate_width(value: Any) -> bool:
    if isinstance(value, Mapping):
        return "candidate_width" in value or any(
            _forbidden_candidate_width(child) for child in value.values()
        )
    if isinstance(value, list):
        return any(_forbidden_candidate_width(child) for child in value)
    return False


def _summary(path: Path, scientific_sha: str) -> Mapping[str, Any]:
    row = _load(path)
    _require(row["commit"] == scientific_sha, f"scientific SHA: {path}")
    _require(row["worktree_dirty"] is False, f"dirty run: {path}")
    _require(row["tracked_diff_sha256"] == EMPTY_DIFF_SHA256, f"tracked diff: {path}")
    _require(row["endpoint_repair_used"] is False, f"endpoint repair: {path}")
    _require(row["endpoint_tightening_used"] is False, f"endpoint tightening: {path}")
    return row


def verify(root: Path) -> dict[str, Any]:
    root = root.resolve()
    manifest = _load(root / "manifest.json")
    _require(manifest["schema"] == "vdp_c2_evidence_manifest_v1", "manifest schema")
    _require(
        manifest["packaging_commit_separate_from_scientific_commit"] is True,
        "scientific/package commit separation",
    )
    _require(
        manifest["raw_run_excluded_files"] == list(RAW_RUN_EXCLUDED_FILES),
        "raw-run exclusion disclosure",
    )
    _require(
        not any(
            path.name in RAW_RUN_EXCLUDED_FILES
            for path in (root / "03_raw_runs").rglob("*")
        ),
        "excluded raw trace present",
    )
    expected_paths = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and path.name not in {"manifest.json", "SHA256SUMS"}
    }
    rows = manifest["files"]
    _require({row["path"] for row in rows} == expected_paths, "manifest file coverage")
    _require(len(rows) == manifest["file_count"] == len(expected_paths), "manifest file count")
    total = 0
    for row in rows:
        path = root / row["path"]
        _require(path.stat().st_size == int(row["bytes"]), f"file bytes: {row['path']}")
        _require(_sha(path) == row["sha256"], f"file hash: {row['path']}")
        total += path.stat().st_size
    _require(total == manifest["total_bytes"], "manifest total bytes")
    sums = {}
    for line in (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        sums[relative] = digest
    _require(sums == {row["path"]: row["sha256"] for row in rows}, "SHA256SUMS")

    scientific_sha = str(manifest["scientific_sha"])
    baseline = _load(root / "00_provenance/baseline_verification.json")
    _require(baseline["h2_package_verified"] is True, "H2 baseline verifier")
    _require(baseline["c1_package_verified"] is True, "C1 baseline verifier")
    gate_dir = root / "01_step1_causal_gate"
    gate = _load(gate_dir / "gate_a.json")
    oracle = _load(gate_dir / "exact_fraction_bernstein_oracle.json")
    refinement = [
        json.loads(line)
        for line in (gate_dir / "refinement_ledger.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    _require(gate["scientific_sha"] == scientific_sha, "Gate A scientific SHA")
    _require(gate["gate_pass"] is True and not gate["failure_code"], "Gate A")
    _require(gate["committed_refinement_count"] >= 1, "Gate A committed iteration")
    _require(gate["c1_vs_flowstar_x_gap_fraction_removed"] >= 0.5, "Gate A x gap")
    _require(gate["y_raw_image_no_regression"] is True, "Gate A y regression")
    _require(gate["all_published_channels_no_wider"] is True, "Gate A published channels")
    _require(gate["first_acceptance_decision_identical"] is True, "first acceptance")
    _require(gate["candidate_polynomial_bitwise_identical"] is True, "candidate polynomial")
    _require(gate["final_remainder_ledger_matches_last_commit"] is True, "final ledger")
    verify_refinement_ledger(refinement, oracle, gate)
    tamper = run_tamper(gate_dir)
    _require(tamper["passed"] is True and len(tamper["cases"]) == 6, "tamper suite")

    source = _load(gate_dir / "flowstar_pinned_contract.json")
    _require(source["commit"] == "b85a3211748cb77b736fe4ad42ee02d8d2b81148", "Flow* SHA")
    _require(source["max_refinement_steps_macro"] == 490, "Flow* max refinement")
    _require(source["inclusive_zero_based_replay_limit"] == 491, "Flow* replay limit")
    _require(source["stop_ratio"] == 0.99, "Flow* stop ratio")
    _require(source["width_ratio_direction"] == "new_width_divided_by_old_width", "widthRatio")
    _require(len(source["files"]) == 5, "Flow* source files")

    matrix = _load(root / "02_scientific_matrix/matrix.json")
    _require(matrix["schema"] == "vdp_c2_scientific_matrix_v1", "matrix schema")
    _require(matrix["scientific_sha"] == scientific_sha, "matrix scientific SHA")
    _require(not _forbidden_candidate_width(matrix), "ambiguous candidate_width field")
    _require(matrix["decision"] in ALLOWED_DECISIONS, "allowed final decision")
    _require(matrix["decision"] not in {"POST_ACCEPT_REFINEMENT_CAUSAL_GATE_FAILED", "C2_SOUNDNESS_OR_PROVENANCE_FAILED"}, "successful soundness/provenance")
    lane_naming = matrix["lane_naming"]
    _require(set(lane_naming) >= {
        "gate_b_h1_h2_candidate",
        "production_c1_candidate",
        "production_c2_candidate",
    }, "lane naming")
    for horizon, channels in matrix["fixed"].items():
        for channel, row in channels.items():
            expected = (row["legacy_width"] - row["production_c2_candidate_width"]) / (
                row["legacy_width"] - row["flowstar_width"]
            )
            _require(
                expected == row["original_target_fraction_legacy_excess_removed"],
                f"target formula: {horizon}/{channel}",
            )
            _require(
                row["c2_incremental_reduction_vs_c1"]
                == row["production_c1_candidate_width"] - row["production_c2_candidate_width"],
                f"incremental formula: {horizon}/{channel}",
            )
    _require(matrix["gates"]["T6p32_all_channels_no_wider_than_c1"] is True, "T6.32 gate")
    _require(matrix["gates"]["native_not_below_6p589638579126679"] is True, "native floor")
    _require(matrix["gates"]["c2_over_legacy_runtime_at_most_2x"] is True, "runtime gate")
    _require(matrix["gates"]["v100_implementation_consistency_at_1e_12"] is True, "V100 consistency")
    _require(matrix["v100_consistency"]["scope"].startswith("implementation consistency only"), "CUDA scope")
    _require("directed-rounding soundness" in matrix["cuda_claim_scope"], "CUDA soundness disclaimer")

    raw = root / "03_raw_runs"
    lane_modes = {
        "legacy": "flowstar_raw_remainder_compat",
        "production_c1_candidate": "flowstar_raw_remainder_compat_factorized_joint_closure",
        "production_c2_candidate": "flowstar_raw_remainder_compat_factorized_joint_closure_refined",
    }
    for scenario in ("step1", "fixed_T1", "fixed_T3", "fixed_T6p32", "native_T10"):
        for lane, mode in lane_modes.items():
            summary = _summary(raw / scenario / lane / "summary.json", scientific_sha)
            _require(summary["validation_mode"] == mode, f"mode: {scenario}/{lane}")
            if scenario != "native_T10":
                _require(summary["completed_requested_horizon"] is True, f"fixed completion: {scenario}/{lane}")
    for device in ("cpu", "cuda"):
        summary = _summary(raw / "consistency_T0p1" / device / "summary.json", scientific_sha)
        _require(summary["device"] == device, f"consistency device: {device}")

    if not matrix["gates"]["reaches_T10"]:
        diagnostic = matrix["terminal_diagnostic"]
        _require(isinstance(diagnostic, Mapping), "missing terminal diagnostic")
        _require(diagnostic["production_first_self_map_subset"] is False, "terminal first subset")
        _require(diagnostic["production_refinement_committed"] is False, "terminal production commit")
        _require(diagnostic["scheduler_at_h_min"] is True, "terminal h_min")
        _require(
            diagnostic["first_image"]["exact_oracle"]["all_contained"] is True,
            "terminal first image oracle",
        )
        _require("ownership only" in diagnostic["ownership_warning"], "terminal ownership warning")

    xml_results = {}
    for path in sorted((root / "04_tests").glob("*.xml")):
        xml_root = ET.parse(path).getroot()
        suites = [xml_root] if xml_root.tag == "testsuite" else list(xml_root.findall("testsuite"))
        tests = sum(int(suite.attrib.get("tests", 0)) for suite in suites)
        failures = sum(int(suite.attrib.get("failures", 0)) for suite in suites)
        errors = sum(int(suite.attrib.get("errors", 0)) for suite in suites)
        _require(failures == 0 and errors == 0, f"test failures: {path.name}")
        xml_results[path.name] = tests
    _require(xml_results and sum(xml_results.values()) > 0, "test evidence")
    report = (root / "05_report/VDP_C2_POST_ACCEPT_REFINEMENT_20260820.md").read_text(encoding="utf-8")
    _require(matrix["decision"] in report, "report final decision")
    _require("首次验证" in report and "post-accept refinement" in report, "phase report")

    result = {
        "status": "verified",
        "scientific_sha": scientific_sha,
        "files": manifest["file_count"],
        "bytes": manifest["total_bytes"],
        "refinement_iterations": len(refinement),
        "tamper_cases": len(tamper["cases"]),
        "decision": matrix["decision"],
        "tests": xml_results,
    }
    print(json.dumps(result, sort_keys=True))
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    return parser.parse_args(argv)


if __name__ == "__main__":
    verify(parse_args().root)
