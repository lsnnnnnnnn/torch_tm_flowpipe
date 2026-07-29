from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from audit_results import audit
from curate_artifacts import curate
from generate_final_delivery import _artifact_mapping


def _write_required(output: Path) -> None:
    for name, payload in {
        "correctness_checks.json": {"primary_gates_passed": True},
        "final_acceptance.json": {
            "passed": True,
            "require_ten_repetitions": False,
        },
        "bern_feasibility.json": {"cases": 0},
    }.items():
        (output / name).write_text(json.dumps(payload), encoding="utf-8")
    (output / "three_tool_deep_study_report.md").write_text(
        "# fixture\n", encoding="utf-8"
    )


def test_audit_rejects_csv_overflow_and_json_nonfinite(tmp_path: Path) -> None:
    _write_required(tmp_path)
    with (tmp_path / "raw_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["requested_horizon", "successful_horizon", "value"])
        writer.writerow(["1", "1", "1e999"])
    (tmp_path / "overflow.json").write_text(
        '{"value": 1e999}', encoding="utf-8"
    )

    with pytest.raises(SystemExit, match="non-finite"):
        audit(tmp_path)


def test_curator_refuses_non_authoritative_repetition_count(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "final_acceptance.json").write_text(
        json.dumps({"passed": True, "require_ten_repetitions": False}),
        encoding="utf-8",
    )
    (source / "artifact_quality_audit.json").write_text(
        json.dumps({"passed": True}), encoding="utf-8"
    )
    (source / "RUN_COMPLETE").touch()

    with pytest.raises(SystemExit, match="ten-repetition"):
        curate(source, tmp_path / "destination")


def test_every_mandatory_plot_has_explicit_protocol_mapping() -> None:
    for index in range(1, 19):
        prefix = f"{index:02d}_"
        matches = [
            name
            for name in {
                "01_one_step_tube_width_vs_h.png",
                "02_one_step_endpoint_raw_width_vs_h.png",
                "03_exact_inflation_ratios.png",
                "04_common_affine_carry_width_vs_time.png",
                "05_common_box_carry_width_vs_time.png",
                "06_affine_vs_box_carry.png",
                "07_native_low_order_width_curves.png",
                "08_native_practical_width_runtime_pareto.png",
                "09_successful_horizon_vs_runtime.png",
                "10_polynomial_remainder_decomposition.png",
                "11_monomial_family_support.png",
                "12_torch_reset_order_ablation.png",
                "13_diffreach_affine_quasi_symbolic_ablation.png",
                "14_flowstar_order_step_qr_symbolic_refinement_ablation.png",
                "15_matched_basis_results.png",
                "16_common_defect_vs_native_remainder.png",
                "17_runtime_decomposition.png",
                "18_failure_categories.png",
            }
            if name.startswith(prefix)
        ]
        assert len(matches) == 1
        source, producer, protocol = _artifact_mapping(
            f"plots/{matches[0]}"
        )
        assert source.endswith(".csv")
        assert producer == "plot_results.py"
        assert protocol != "Run metadata or supporting evidence"
