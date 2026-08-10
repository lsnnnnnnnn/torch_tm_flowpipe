from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.integration
def test_exact_fraction_replay_qualifies_only_the_bounded_outward_envelope(tmp_path: Path):
    output = tmp_path / "fraction_replay.json"
    completed = subprocess.run(
        [
            sys.executable,
            "experiments/replay_fixed_support_fraction.py",
            "--output",
            str(output),
            "--device",
            "cpu",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["initial_inclusion_mask_equal"]
    assert report["all_round_masks_equal"]
    assert not report["all_directly_contained"]
    assert not report["ordinary_binary64_directly_qualified"]
    assert report["replay_envelope_qualified"]
    assert report["max_outward_ulps_needed"] == 2
    assert report["ordinary_lane_classification"] == "empirically sampled only"
    assert (
        report["replay_envelope_classification"]
        == "independently outward replayed for exact benchmark workload"
    )
