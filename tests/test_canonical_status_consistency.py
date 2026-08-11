from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

from torch_tm_flowpipe.comparison_contract import vdp_identity_hashes


ROOT = Path(__file__).parents[1]
CURRENT = "S1_REACHES_TERMINAL_BUT_DOES_NOT_CLOSE_IT"
CURRENT_BRIDGE = "FIXED_SUPPORT_BRIDGE_BLOCKED"


def test_canonical_results_status_and_limitations_share_current_s1_outcome():
    for relative in (
        "docs/RESULTS.md",
        "docs/RESULTS_STATUS.md",
        "docs/STATUS.md",
        "docs/LIMITATIONS.md",
    ):
        text = (ROOT / relative).read_text(encoding="utf-8")
        assert CURRENT in "\n".join(text.splitlines()[:30]), relative
    for relative in ("docs/RESULTS.md", "docs/STATUS.md"):
        text = (ROOT / relative).read_text(encoding="utf-8")
        stale_index = text.index("S1_PREFIX_REJECTS_BEFORE_TERMINAL")
        assert "supersed" in text[:stale_index].lower()


def test_canonical_documents_share_final_bridge_outcome():
    for relative in (
        "README.md",
        "docs/RESULTS.md",
        "docs/RESULTS_STATUS.md",
        "docs/STATUS.md",
        "docs/LIMITATIONS.md",
        "handoff.md",
    ):
        text = (ROOT / relative).read_text(encoding="utf-8")
        assert CURRENT_BRIDGE in "\n".join(text.splitlines()[:35]), relative


def test_historical_s1_documents_have_superseded_or_qualification_banners():
    documents = (
        "docs/S1_PREFIX_INTEGRATION_RESULT_20260810.md",
        "docs/S1_BOUNDARY164_CAUSAL_ATTRIBUTION_20260811.md",
        "docs/S1_CORRECTED_CARRY_RESULT_20260811.md",
        "handoff.md",
    )
    for relative in documents:
        first_lines = "\n".join(
            (ROOT / relative).read_text(encoding="utf-8").splitlines()[:12]
        ).lower()
        assert "supersed" in first_lines or "qualif" in first_lines, relative


def test_contract_file_contains_the_computed_identity_hashes():
    contract = yaml.safe_load(
        (ROOT / "benchmarks/vdp_three_lane_contract_20260810.yaml").read_text(
            encoding="utf-8"
        )
    )
    expected = vdp_identity_hashes()
    identity = contract["identity_canonicalization"]
    assert identity["rhs_sha256"] == expected["rhs_sha256"]
    assert identity["initial_set_sha256"] == expected["initial_set_sha256"]
    assert identity["partitions"]["B1"]["list_sha256"] == expected[
        "partition_b1_sha256"
    ]
    assert identity["partitions"]["B64"]["list_sha256"] == expected[
        "partition_b64_sha256"
    ]


def test_old_evidence_package_anchors_are_immutable():
    old = (
        ROOT
        / "outputs/s1_boundary164_causal_guarded_carry_20260811/20260811T033447Z"
    )
    expected = {
        "manifest.json": "485b24d0b63badf0833b264514a45bb58a7447c429c7ec9f3cbc83f4223af9e6",
        "SHA256SUMS": "ee2ff2c0feafa16e7603257e14c3d307c01fca50501ec9db78981c7768a71148",
        "verification.json": "9cb7cbeeb01629a358a2fcd14fa3b54356c2e6f40f99260afa4ff75b8b248b9f",
    }
    for name, digest in expected.items():
        assert hashlib.sha256((old / name).read_bytes()).hexdigest() == digest
