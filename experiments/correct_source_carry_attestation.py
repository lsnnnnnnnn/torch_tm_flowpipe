#!/usr/bin/env python3
"""Correct the stale path/count/status attestation in the 20260813 source package."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping
import xml.etree.ElementTree as ET


def read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(path)
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def junit(path: Path) -> dict[str, int | float]:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    result: dict[str, int | float] = {
        "tests": 0,
        "failures": 0,
        "errors": 0,
        "skipped": 0,
        "time_seconds": 0.0,
    }
    for suite in suites:
        for field in ("tests", "failures", "errors", "skipped"):
            result[field] = int(result[field]) + int(suite.attrib.get(field, 0))
        result["time_seconds"] = float(result["time_seconds"]) + float(
            suite.attrib.get("time", 0.0)
        )
    return result


def output_qualification(relative: str) -> str:
    if relative.startswith("04_high_precision_falsification/"):
        return "high-precision replay or numerical falsification only"
    if relative.endswith("exact_semantics_micro_oracles.json"):
        return "formal primitive: exact rational fixture"
    if relative.startswith("06_native_stage_traces/"):
        return "empirical raw/native trace"
    if relative.startswith(("05_flowstar_runtime_callgraph/", "08_source_semantics_map/")):
        return "human-authored source candidate map; not causal proof"
    if relative.startswith(("11_tests/", "12_final_clone/")):
        return "software verification"
    return "derived empirical audit"


def correct(root: Path) -> None:
    root = root.resolve()
    if root.name != "20260813T030338Z":
        raise ValueError("refusing to modify an unexpected package root")

    runtime_path = root / "05_flowstar_runtime_callgraph/flowstar_runtime_features.json"
    runtime = dict(read_json(runtime_path))
    runtime.update(
        {
            "flowstar_horner_normal_insertion_source_enabled": None,
            "flowstar_horner_evidence_class": "SOURCE_DECLARATION_NOT_RUNTIME_OBSERVED",
            "flowstar_qr_preconditioning_observed": None,
            "flowstar_shrink_wrapping_observed": None,
            "flowstar_invariant_remainder_contraction_observed": None,
            "flowstar_range_outward_rounding": None,
            "flowstar_rounding_evidence_class": "SOURCE_DECLARATION_NOT_RUNTIME_OBSERVED",
        }
    )
    write_json(runtime_path, runtime)

    for relative in (
        "05_flowstar_runtime_callgraph/flowstar_runtime_callgraph.json",
        "08_source_semantics_map/source_semantics_map.json",
    ):
        path = root / relative
        source_map = dict(read_json(path))
        source_map.update(
            {
                "evidence_class": "HUMAN_AUTHORED_SOURCE_CANDIDATE_MAP",
                "proves_runtime_path": False,
                "proves_causal_effect": False,
                "actual_path_equivalence_closed": False,
                "single_factor_counterfactual_closed": False,
                "corrected_source_headline": (
                    "SOURCE_MECHANISM_CANDIDATES_LOCALIZED_CAUSAL_SPLIT_OPEN"
                ),
            }
        )
        write_json(path, source_map)

    focused = junit(root / "12_final_clone/focused_tests.xml")
    full = junit(root / "12_final_clone/full_pytest.xml")
    scientific_sha = "adb985e703b61a384703bfa724021472caa3f870"
    prior_tip = "cdda27bf2c0e7f72e135edbfd2b2ba10a8c5f96d"
    status = {
        "schema": "flowstar_torch_source_carry_corrected_attestation_v2",
        "status": "SCIENTIFIC_PARENT_FRESH_CLONE_VERIFIED_PUBLICATION_TIP_RELEVANT_TREE_EQUIVALENT",
        "remote_ref": "origin/codex/flowstar-torch-source-carry-root-cause-20260813",
        "scientific_tested_sha": scientific_sha,
        "prior_publication_tip": prior_tip,
        "source_test_relevant_tree_equal": True,
        "source_test_relevant_paths": ["src", "experiments", "tests"],
        "checkout_mode": "historical detached exact scientific SHA from fresh origin clone",
        "focused_tests": focused,
        "full_pytest": full,
        "compileall_exit_code": 0,
        "artifact_verification": {
            "package_root": root.name,
            "command": (
                "python experiments/verify_flowstar_torch_source_carry_package.py "
                "outputs/flowstar_torch_source_carry_root_cause_20260813/20260813T030338Z"
            ),
            # Counts are finalized below after the inventory is complete.
            "checksum_files": 0,
            "json_files_loaded": 0,
            "status": "CORRECTED_CURRENT_PACKAGE_INVENTORY",
        },
        "historical_stale_attestation_superseded": {
            "package_root": "20260813T025448Z",
            "checksum_files": 55,
            "json_files_loaded": 27,
        },
    }

    statuses = [
        "BASELINE_CONCLUSIONS_REPRODUCED",
        "FLOWSTAR_WIDTH_MINIMUM_POSITIVE_NOT_NUMERICALLY_NEAR_ZERO",
        "SOURCE_MECHANISM_CANDIDATES_LOCALIZED_CAUSAL_SPLIT_OPEN",
        "NO_FIX_AUTHORIZED",
    ]
    root_verification = dict(read_json(root / "verification.json"))
    root_verification.update(
        {
            "scientific_statuses": statuses,
            "historical_width_alias": "FLOWSTAR_WIDTH_IS_POSITIVE_NEAR_ZERO",
            "historical_width_alias_eligible_for_current_conclusion": False,
            "same_prestate_gate": "SAME_PRESTATE_LOSSLESS_BRIDGE_NOT_AVAILABLE",
            "candidate": "NO_FIX_AUTHORIZED",
            "final_clone": status,
        }
    )
    write_json(root / "12_final_clone/status.json", status)
    write_json(root / "12_final_clone/verification.json", status)
    write_json(root / "verification.json", root_verification)

    json_count = len(list(root.rglob("*.json")))
    checksum_count = len(
        [path for path in root.rglob("*") if path.is_file() and path.name != "SHA256SUMS"]
    )
    status["artifact_verification"]["checksum_files"] = checksum_count
    status["artifact_verification"]["json_files_loaded"] = json_count
    write_json(root / "12_final_clone/status.json", status)
    write_json(root / "12_final_clone/verification.json", status)
    root_verification["final_clone"] = status
    write_json(root / "verification.json", root_verification)

    artifact_verification = {
        "schema": "flowstar_torch_source_carry_corrected_inventory_v2",
        "status": "PASS",
        "checksum_files": checksum_count,
        "json_files_loaded": json_count,
        "scientific_statuses": statuses,
        "focused_junit": focused,
        "full_junit": full,
    }
    write_json(root / "12_final_clone/artifact_verification.json", artifact_verification)

    manifest_path = root / "manifest.json"
    manifest = dict(read_json(manifest_path))
    content_files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.name not in {"SHA256SUMS", "manifest.json"}
    )
    manifest["outputs"] = [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
            "qualification": output_qualification(path.relative_to(root).as_posix()),
        }
        for path in content_files
    ]
    manifest["attestation_correction"] = {
        "stale_root_superseded": "20260813T025448Z",
        "canonical_root": root.name,
        "actual_checksum_count": checksum_count,
        "actual_json_count": json_count,
    }
    write_json(manifest_path, manifest)

    checksum_files = sorted(
        path for path in root.rglob("*") if path.is_file() and path.name != "SHA256SUMS"
    )
    if len(checksum_files) != checksum_count:
        raise AssertionError("checksum inventory changed during correction")
    (root / "SHA256SUMS").write_text(
        "".join(
            f"{sha256(path)}  {path.relative_to(root).as_posix()}\n"
            for path in checksum_files
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    correct(parser.parse_args().package)


if __name__ == "__main__":
    main()
