from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

from torch_tm_flowpipe.comparison_contract import vdp_identity_hashes


ROOT = Path(__file__).parents[1]
CURRENT_FLOW = "FLOWSTAR_TORCH_FIXED_SCHEDULE_COMMON_PREFIX_ONLY"
CURRENT_DIFF = "DIFFREACH_TORCH_DR7_FULL_HORIZON_DIVERGED"
CURRENT_CARRY = "CARRY_MISSING_SYMBOLIC_SEMANTICS"
CURRENT_FIX = "NO_FIX_AUTHORIZED"


def test_canonical_documents_share_current_full_horizon_outcomes():
    for relative in (
        "README.md",
        "docs/RESULTS.md",
        "docs/RESULTS_STATUS.md",
        "docs/STATUS.md",
        "docs/LIMITATIONS.md",
        "handoff.md",
    ):
        text = (ROOT / relative).read_text(encoding="utf-8")
        headline = "\n".join(text.splitlines()[:45])
        for outcome in (CURRENT_FLOW, CURRENT_DIFF, CURRENT_CARRY, CURRENT_FIX):
            assert outcome in headline, (relative, outcome)


def test_previous_bridge_and_s1_claims_are_historical_or_superseded():
    for relative in ("README.md", "docs/RESULTS.md", "docs/STATUS.md", "handoff.md"):
        text = (ROOT / relative).read_text(encoding="utf-8")
        assert "histor" in text.lower() or "supersed" in text.lower(), relative


def test_historical_s1_documents_have_superseded_or_qualification_banners():
    documents = (
        "docs/S1_PREFIX_INTEGRATION_RESULT_20260810.md",
        "docs/S1_BOUNDARY164_CAUSAL_ATTRIBUTION_20260811.md",
        "docs/S1_CORRECTED_CARRY_RESULT_20260811.md",
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
