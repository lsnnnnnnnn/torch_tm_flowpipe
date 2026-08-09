from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/tora_q3_stage_parity_fused_20260809"
MANIFESTS = (
    ROOT / "outputs/tora_q3_native_matched_20260806/manifest.sha256",
    ROOT / "outputs/tora_q3_perf_closure_20260806/manifest.sha256",
    OUTPUT / "manifest.sha256",
)


@pytest.mark.regression
@pytest.mark.protocol
def test_checkpoint7_scan_is_complete_and_fail_closed() -> None:
    scan = json.loads(
        (OUTPUT / "provenance/checkpoint7_publication_scan.json").read_text(
            encoding="utf-8"
        )
    )
    assert scan["governance_status"] == "PASS_CLEAN_LINEAGE"
    assert scan["unallowlisted_path_or_credential_match_count"] == 0
    assert scan["current_tree_sensitive_suffix_candidate_count"] == 0
    assert scan["working_untracked_file_count"] == 0


@pytest.mark.regression
@pytest.mark.protocol
def test_public_aggregates_contain_no_private_paths_or_raw_asset_suffixes() -> None:
    forbidden = (
        "/srv/",
        "/home/",
        "private_verification_evidence",
        "controllerTora.onnx",
    )
    for path in OUTPUT.rglob("*"):
        if path.is_file() and path.suffix in {".json", ".csv", ".md", ".svg"}:
            text = path.read_text(encoding="utf-8")
            assert not any(token in text for token in forbidden), path
    assert not any(
        path.suffix.lower() in {".onnx", ".pt", ".pth", ".ckpt", ".safetensors"}
        for path in OUTPUT.rglob("*")
        if path.is_file()
    )


@pytest.mark.regression
@pytest.mark.protocol
def test_all_public_manifests_are_identical_and_cover_the_git_index() -> None:
    assert MANIFESTS[0].read_bytes() == MANIFESTS[1].read_bytes()
    assert MANIFESTS[1].read_bytes() == MANIFESTS[2].read_bytes()
    manifest_paths = {
        line.split("  ", 1)[1]
        for line in MANIFESTS[0].read_text(encoding="utf-8").splitlines()
    }
    tracked = set(
        subprocess.run(
            ["git", "ls-files"], cwd=ROOT, check=True, text=True, capture_output=True
        ).stdout.splitlines()
    )
    expected = {
        path for path in tracked if Path(path).name != "manifest.sha256"
    }
    assert manifest_paths == expected
