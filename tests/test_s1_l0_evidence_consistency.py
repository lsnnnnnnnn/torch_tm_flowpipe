import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "experiments" / "run_s1_prefix_complete_o4.py"
PACKAGER_PATH = ROOT / "experiments" / "package_s1_prefix_result.py"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = _load("s1_prefix_runner_evidence", RUNNER_PATH)
packager = _load("s1_prefix_packager_evidence", PACKAGER_PATH)


def test_l0_short_replay_writes_explicit_commit_semantics(tmp_path):
    source = (
        ROOT
        / "outputs/mainline_realignment_20260810/20260810T025910Z/01_native_baselines"
        / "torch_complete_o4_authoritative_t6p5/segments.csv"
    )
    schedule = runner.freeze_schedule(source, tmp_path / "schedule.json")
    summary = runner.replay_lane(schedule, tmp_path / "L0", "L0", max_boundaries=2)
    rows = [
        json.loads(line)
        for line in (tmp_path / "L0/prefix_conservation.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["committed_to_frozen_prefix"] for row in rows] == [True, True]
    assert summary["accepted_boundaries"] == sum(
        row["committed_to_frozen_prefix"] for row in rows
    ) == 2


def test_packager_rejects_missing_or_non_boolean_commit_semantics():
    with pytest.raises(ValueError, match="missing committed"):
        packager._committed({"lane": "L0", "attempt_index": 0})
    with pytest.raises(TypeError, match="must be boolean"):
        packager._committed(
            {"lane": "L0", "attempt_index": 0, "committed_to_frozen_prefix": "False"}
        )
