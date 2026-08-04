import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _runner_module():
    path = ROOT / "experiments" / "run_vdp_dense_backend.py"
    spec = importlib.util.spec_from_file_location("run_vdp_dense_backend", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_runner_loads_unmodified_authoritative_contract():
    runner = _runner_module()
    contract = runner.load_contract()
    assert contract["initial_box"] == [[1.1, 1.4], [2.35, 2.45]]
    assert contract["requested_order"] == 4
    assert contract["target_remainder_radius"] == 1e-4
    assert contract["cutoff"] == 1e-10
    assert contract["h_min"] == 0.002
    assert contract["h_max"] == 0.1
    assert contract["target_horizon"] == 10.0
    assert contract["canonical_system_spec"]["rhs"][1]["terms"] == [
        {"coefficient": 1.0, "powers": [0, 1]},
        {"coefficient": -1.0, "powers": [1, 0]},
        {"coefficient": -1.0, "powers": [2, 1]},
    ]


def test_dense_runner_writes_fail_closed_parseable_outputs(tmp_path):
    runner = _runner_module()
    output = tmp_path / "run"
    args = runner.parse_args(
        [
            "--output-dir",
            str(output),
            "--tm-backend",
            "dense",
            "--device",
            "cpu",
            "--horizon",
            "0.02",
            "--wall-cap-s",
            "60",
        ]
    )
    summary = runner.run(args)
    assert summary["completed_requested_horizon"] is True
    assert summary["completed_horizon"] == pytest.approx(0.02)
    assert summary["backend_lane"] == "hybrid_dense_core"
    assert summary["fallback_count"] == 0
    assert summary["endpoint_repair_used"] is False
    assert summary["raw_endpoint"]
    assert summary["last_segment"]
    assert summary["full_tube"]
    for name in (
        "config_snapshot.yaml",
        "command.json",
        "attempts.csv",
        "segments.csv",
        "checkpoints.csv",
        "remainder_ledger.jsonl",
        "profile.csv",
        "summary.json",
        "decision.json",
    ):
        assert (output / name).exists()


def test_runner_refuses_nonempty_output_directory(tmp_path):
    runner = _runner_module()
    output = tmp_path / "occupied"
    output.mkdir()
    (output / "sentinel").write_text("preserve", encoding="utf-8")
    args = runner.parse_args(["--output-dir", str(output), "--tm-backend", "dense", "--horizon", "0.02"])
    with pytest.raises(FileExistsError, match="refusing non-empty"):
        runner.run(args)
    assert (output / "sentinel").read_text(encoding="utf-8") == "preserve"


def test_runner_labels_exactly_one_noncanonical_factor(tmp_path):
    runner = _runner_module()
    output = tmp_path / "range_midpoint"
    args = runner.parse_args(
        [
            "--output-dir",
            str(output),
            "--tm-backend",
            "dense",
            "--horizon",
            "0.02",
            "--right-map-center-mode",
            "range_midpoint",
        ]
    )
    summary = runner.run(args)
    assert summary["single_factor_diagnostic"] is True
    assert summary["diagnostic_factors"] == ["right_map_center_mode=range_midpoint"]
