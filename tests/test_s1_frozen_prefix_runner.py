from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments" / "run_s1_prefix_complete_o4.py"
SPEC = importlib.util.spec_from_file_location("run_s1_prefix_complete_o4", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)

SOURCE = (
    ROOT
    / "outputs"
    / "mainline_realignment_20260810"
    / "20260810T025910Z"
    / "01_native_baselines"
    / "torch_complete_o4_authoritative_t6p5"
    / "segments.csv"
)


def test_authoritative_schedule_freezes_all_binary64_steps(tmp_path):
    destination = tmp_path / "schedule.json"
    schedule = runner.freeze_schedule(SOURCE, destination)
    assert schedule["accepted_boundary_count"] == 307
    assert len(schedule["rows"]) == 308
    assert schedule["source_artifact_sha256"] == "28a92e6bb84e2d8b81cb31e57453eb46e28ef03c434d284ec92b9a50f6ac5dc4"
    terminal = schedule["rows"][-1]
    assert terminal["expected_status"] == "rejected"
    assert terminal["t_before"]["hex"] == float(6.397083942944808).hex()
    assert terminal["h_attempted"]["hex"] == float(0.003623635847674574).hex()
    assert json.loads(destination.read_text(encoding="utf-8"))["schema"] == runner.SCHEDULE_SCHEMA


def test_materialized_control_and_k16_lane_share_two_step_schedule(tmp_path):
    schedule = runner.freeze_schedule(SOURCE, tmp_path / "schedule.json")
    summaries = {}
    last_rows = {}
    for lane in ("L1", "L2"):
        directory = tmp_path / lane
        summaries[lane] = runner.replay_lane(
            schedule, directory, lane, max_boundaries=2
        )
        rows = [
            json.loads(line)
            for line in (directory / "prefix_conservation.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        assert all(row["schedule_match"] for row in rows)
        assert all(row["conservation_mask"] for row in rows)
        assert all(row["endpoint_publication_mask"] and row["tube_publication_mask"] for row in rows)
        last_rows[lane] = rows[-1]
    assert summaries["L1"]["accepted_boundaries"] == summaries["L2"]["accepted_boundaries"] == 2
    assert last_rows["L1"]["active_columns_after"] == 0
    assert last_rows["L1"]["immediate_control_materialization_count"] == 1
    assert last_rows["L2"]["active_columns_after"] == 2
