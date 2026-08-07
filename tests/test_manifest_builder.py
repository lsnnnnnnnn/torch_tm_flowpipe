from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts/build_tora_q3_public_manifest.py"


@pytest.mark.protocol
def test_manifest_builder_can_hash_exact_index_blobs(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    payload = tmp_path / "payload.txt"
    payload.write_text("staged\n", encoding="utf-8")
    subprocess.run(["git", "add", "payload.txt"], cwd=tmp_path, check=True)
    payload.write_text("unstaged\n", encoding="utf-8")
    output = tmp_path / "manifest.sha256"

    subprocess.run(
        [
            sys.executable,
            str(BUILDER),
            "--repository",
            str(tmp_path),
            "--source",
            "index",
            "--output",
            str(output),
        ],
        check=True,
    )

    expected_digest = hashlib.sha256(b"staged\n").hexdigest()
    assert output.read_text(encoding="utf-8") == f"{expected_digest}  payload.txt\n"
