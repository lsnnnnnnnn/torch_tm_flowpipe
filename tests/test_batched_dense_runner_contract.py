import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

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
    assert "initial_box_exact_decimal" not in contract
    assert runner.EXACT_INITIAL_BOX_DECIMAL == [["1.1", "1.4"], ["2.35", "2.45"]]
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
            "--trace-flush-every",
            "0",
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
        "owner_ledger.jsonl",
        "range_trace.jsonl",
        "profile.csv",
        "summary.json",
        "decision.json",
    ):
        assert (output / name).exists()


def test_dense_runner_fixed_schedule_has_exact_binary64_trace(tmp_path):
    runner = _runner_module()
    output = tmp_path / "fixed"
    args = runner.parse_args(
        [
            "--output-dir",
            str(output),
            "--tm-backend",
            "dense",
            "--device",
            "cpu",
            "--horizon",
            "0.01",
            "--fixed-step",
            "0.01",
            "--wall-cap-s",
            "60",
            "--trace-flush-every",
            "0",
        ]
    )

    summary = runner.run(args)

    assert summary["completed_requested_horizon"] is True
    assert summary["schedule"] == {
        "kind": "fixed",
        "h_decimal": "0.01",
        "h_hex": float(0.01).hex(),
        "requested_steps": 1,
        "adaptive_fallback_allowed": False,
    }
    assert summary["support"] == "complete_total_degree_O4"
    assert summary["partition"] == "B1"
    assert summary["fallback_count"] == 0
    assert summary["endpoint_repair_used"] is False
    assert summary["initialization_contract"] == "binary64_literal_matched_contract"
    assert summary["host_to_device_s"] == 0.0
    assert summary["device_to_host_s"] == 0.0
    assert summary["dense_kernel_s"] > 0.0
    assert summary["peak_rss_bytes"] > 0
    assert summary["trace_flush_every"] == 0
    assert summary["trace_write_count"] == 1

    import csv

    with (output / "segments.csv").open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert row["t_lo_hex"] == float(0.0).hex()
    assert row["t_hi_hex"] == float(0.01).hex()
    assert row["h_attempted_hex"] == float(0.01).hex()
    assert row["h_accepted_hex"] == float(0.01).hex()
    assert row["schedule_kind"] == "fixed"
    assert len(row["prestate_sha256"]) == 64
    assert len(row["retained_coefficient_sha256"]) == 64
    assert row["raw_endpoint_published"] == "True"
    assert row["endpoint_tightening_applied"] == "False"


def test_dense_runner_exact_decimal_annotation_is_opt_in(tmp_path):
    runner = _runner_module()
    output = tmp_path / "exact"
    args = runner.parse_args(
        [
            "--output-dir",
            str(output),
            "--tm-backend",
            "dense",
            "--device",
            "cpu",
            "--horizon",
            "0.01",
            "--fixed-step",
            "0.01",
            "--trace-flush-every",
            "0",
            "--initialization-contract",
            "exact_decimal_contract",
            "--reset-mode",
            "normalized_insertion_bounded_shared_source_o4_g2",
        ]
    )

    summary = runner.run(args)
    snapshot = yaml.safe_load((output / "config_snapshot.yaml").read_text(encoding="utf-8"))

    assert summary["completed_requested_horizon"] is True
    assert summary["initialization_contract"] == "exact_decimal_contract"
    assert snapshot["contract"]["initial_box_exact_decimal"] == [["1.1", "1.4"], ["2.35", "2.45"]]
    assert snapshot["contract"]["initialization_contract"] == "exact_decimal_contract"


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


def test_runner_h2_validation_mode_is_explicitly_opt_in(tmp_path):
    runner = _runner_module()
    defaults = runner.parse_args(
        ["--output-dir", str(tmp_path / "default"), "--tm-backend", "dense"]
    )
    assert defaults.validation_mode is None

    output = tmp_path / "h2"
    args = runner.parse_args(
        [
            "--output-dir",
            str(output),
            "--tm-backend",
            "dense",
            "--device",
            "cpu",
            "--horizon",
            "0.01",
            "--fixed-step",
            "0.01",
            "--initialization-contract",
            "exact_decimal_contract",
            "--reset-mode",
            "normalized_insertion_dependency_preserving",
            "--validation-mode",
            "flowstar_raw_remainder_compat_factorized_joint",
            "--trace-flush-every",
            "0",
        ]
    )
    summary = runner.run(args)

    assert summary["completed_requested_horizon"] is True
    assert summary["validation_mode"] == "flowstar_raw_remainder_compat_factorized_joint"
    assert summary["diagnostic_factors"] == [
        "reset_mode=normalized_insertion_dependency_preserving",
        "validation_mode=flowstar_raw_remainder_compat_factorized_joint",
    ]


def test_runner_c1_validation_mode_is_explicitly_opt_in(tmp_path):
    runner = _runner_module()
    output = tmp_path / "c1"
    mode = "flowstar_raw_remainder_compat_factorized_joint_closure"
    args = runner.parse_args(
        [
            "--output-dir",
            str(output),
            "--tm-backend",
            "dense",
            "--device",
            "cpu",
            "--horizon",
            "0.01",
            "--fixed-step",
            "0.01",
            "--initialization-contract",
            "exact_decimal_contract",
            "--reset-mode",
            "normalized_insertion_dependency_preserving",
            "--validation-mode",
            mode,
            "--trace-flush-every",
            "0",
        ]
    )
    summary = runner.run(args)

    assert summary["completed_requested_horizon"] is True
    assert summary["validation_mode"] == mode
    assert summary["diagnostic_factors"] == [
        "reset_mode=normalized_insertion_dependency_preserving",
        f"validation_mode={mode}",
    ]


def test_runner_c2_writes_post_accept_refinement_ledger(tmp_path):
    runner = _runner_module()
    output = tmp_path / "c2"
    mode = "flowstar_raw_remainder_compat_factorized_joint_closure_refined"
    args = runner.parse_args(
        [
            "--output-dir",
            str(output),
            "--tm-backend",
            "dense",
            "--device",
            "cpu",
            "--horizon",
            "0.01",
            "--fixed-step",
            "0.01",
            "--initialization-contract",
            "exact_decimal_contract",
            "--reset-mode",
            "normalized_insertion_dependency_preserving",
            "--validation-mode",
            mode,
            "--trace-flush-every",
            "0",
        ]
    )
    summary = runner.run(args)

    assert summary["completed_requested_horizon"] is True
    assert summary["validation_mode"] == mode
    rows = [
        json.loads(line)
        for line in (output / "refinement_ledger.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert rows
    assert rows[-1]["stop_reason"] == "stop_ratio"
    assert all(row["committed"] for row in rows)
