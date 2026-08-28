from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

from scripts.verify_brusselator_c5_evidence import verify


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/runs/brusselator_live_range_c5_20260828"


def test_brusselator_c5_no_c5_package_verifies() -> None:
    result = verify(ARTIFACT)
    assert result["passed"], result["errors"]
    assert result["status"] == "LIVE_RANGE_DOMINANT_CAUSE_NOT_IDENTIFIED__NO_C5"
    assert result["canonical_object_count"] == 11
    assert result["matrix_row_count"] == 286


def test_brusselator_c5_semantic_tamper_fails_after_checksum_refresh(tmp_path: Path) -> None:
    copied = tmp_path / "evidence"
    shutil.copytree(ARTIFACT, copied)
    result_path = copied / "RESULT.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["status"] = "TORCH_FLOWSTAR_LIVE_RANGE_CAUSE_IDENTIFIED__C5_T20_REACHED"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    digest = hashlib.sha256(result_path.read_bytes()).hexdigest()
    sums_path = copied / "SHA256SUMS"
    lines = sums_path.read_text(encoding="utf-8").splitlines()
    sums_path.write_text(
        "\n".join(
            f"{digest}  RESULT.json" if line.endswith("  RESULT.json") else line
            for line in lines
        )
        + "\n",
        encoding="utf-8",
    )

    verified = verify(copied)
    assert not verified["passed"]
    assert "RESULT/authorization status mismatch" in verified["errors"]
