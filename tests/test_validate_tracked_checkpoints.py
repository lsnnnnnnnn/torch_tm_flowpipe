from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.validate_tracked_checkpoints import validate


def test_checkpoint_loader_hashes_and_loads_all_inputs(tmp_path: Path) -> None:
    first = tmp_path / "a_checkpoint.json"
    second = tmp_path / "nested" / "terminal_state.json"
    second.parent.mkdir()
    first.write_text('{"schema":"a","value":1}\n', encoding="utf-8")
    second.write_text('[1,2,3]\n', encoding="utf-8")
    report = validate((second, first), root=tmp_path)
    assert report["outcome"] == "ALL_TRACKED_JSON_CHECKPOINTS_LOADED"
    assert report["checkpoint_count"] == 2
    assert [row["path"] for row in report["checkpoints"]] == [
        "a_checkpoint.json",
        "nested/terminal_state.json",
    ]
    assert all(len(row["sha256"]) == 64 for row in report["checkpoints"])


def test_checkpoint_loader_rejects_nonfinite_json(tmp_path: Path) -> None:
    path = tmp_path / "bad_checkpoint.json"
    path.write_text(json.dumps({"value": float("nan")}), encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite"):
        validate((path,), root=tmp_path)
