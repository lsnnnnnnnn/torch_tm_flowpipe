from __future__ import annotations

import hashlib
import json
from pathlib import Path

from experiments.finalize_three_tool_evidence_package import (
    validate_checksum_coverage,
)
from torch_tm_flowpipe.evidence_verification import (
    load_verification_document,
)


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = (
    ROOT
    / "outputs/three_tool_matched_divergence_fixed_support_20260811"
    / "20260811T100304Z"
)


def test_compact_historical_recovery_is_hash_complete() -> None:
    validate_checksum_coverage(PACKAGE)
    manifest = json.loads((PACKAGE / "manifest.json").read_text())
    assert manifest["recovery_status"] == (
        "HISTORICAL_PACKAGE_RECOVERED_COMPACT_AUDITED"
    )
    assert manifest["source_commit"] == (
        "2cb647cd37b530aef12e2b627f48b9b1dcf9aa23"
    )
    assert manifest["runner_count"] == 36
    assert manifest["original_package"]["all_original_checksums_verified"]
    original_sums = PACKAGE / manifest["original_package"]["SHA256SUMS"]
    assert hashlib.sha256(original_sums.read_bytes()).hexdigest() == (
        manifest["original_package"]["SHA256SUMS_sha256"]
    )


def test_original_command_verification_remains_replayable() -> None:
    claims = load_verification_document(
        PACKAGE / "verification.json", source_root=PACKAGE
    )
    assert len(claims) == 36
    assert all(claim.status in {"pass", "qualified"} for claim in claims)


def test_recovery_does_not_restore_prohibited_large_products() -> None:
    files = [path for path in PACKAGE.rglob("*") if path.is_file()]
    assert not any(path.suffix in {".npz", ".so"} for path in files)
    assert not any(path.name == "flowstar_probe" for path in files)
    assert not any(path.suffix == ".jsonl" for path in files)
    assert max(path.stat().st_size for path in files) < 1_000_000


def test_historical_overclaims_are_explicitly_quarantined() -> None:
    manifest = json.loads((PACKAGE / "manifest.json").read_text())
    corrections = manifest["historical_claim_corrections"]
    assert corrections["original_outcomes_are_archival_not_current"]
    assert corrections["diffreach_torch_operator_status"] == (
        "DIFFREACH_TORCH_DR7_OPERATOR_EQUIVALENCE_CLOSED"
    )
    assert corrections["diffreach_torch_full_horizon_status"] == (
        "DIFFREACH_TORCH_DR7_FULL_HORIZON_PAIRWISE_PENDING"
    )
    assert corrections["historical_14_fresh_clone_label"] == (
        "CURRENT_WORKTREE_CHECKS_NOT_TRUE_CLONE"
    )
