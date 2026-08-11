from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.integration
def test_r35_overflow_has_bounded_independent_mpfr_replay(tmp_path: Path) -> None:
    output = tmp_path / "r35-mpfr"
    completed = subprocess.run(
        [
            sys.executable,
            "experiments/replay_r35_mpfr_remainder.py",
            "--output-dir",
            str(output),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert report["outcome"] == "R35_BOUNDED_MPFR_REPLAY_PASS"
    assert report["support"]["slot_count"] == 35
    assert report["fixture"]["product_is_overflow"]
    assert not report["ordinary_binary64_remainder"]["directly_contains_mpfr"]
    assert report["two_ulp_companion_envelope"]["contains_mpfr"]
    assert report["mpfr_oracle"]["precision_bits"] == 256
    assert report["mpfr_oracle"]["input_semantics"] == "exact_binary64"
    assert report["mpfr_oracle"]["rounding"] == "mpfr_directed"
