from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pytest

from experiments.build_full_horizon_pairwise_figures import build
from experiments.finalize_complete_o4_carry_root_cause import derive


def _reproductions() -> list[dict[str, object]]:
    rows = []
    for cell, batch, completed, failed in (
        ("A3", 1, 1000, None),
        ("A3", 64, 1000, None),
        ("A4", 1, 319, 320),
        ("A4", 64, 333, 334),
    ):
        rows.append(
            {
                "cell": cell,
                "batch": batch,
                "completed_steps": completed,
                "validated_horizon": completed * 0.01,
                "first_failure": None if failed is None else {"step": failed},
                "reproduction_status": "reproduced",
                "no_hidden_fallback": True,
            }
        )
    return rows


def _accounting(*, failure: bool) -> dict[str, object]:
    widths = {
        "degree_gt4_dropped_polynomial": 2e-5 if failure else 0.0,
        "polynomial_times_parameterization_remainder": 7.1 if failure else 0.0,
        "endpoint_remainder_times_parameterization_polynomial": 0.0,
        "remainder_times_remainder": 0.27 if failure else 0.0,
        "outer_endpoint_remainder": 0.018 if failure else 0.0018,
    }
    checkpoint = {
        "source_intervals": {name: {"max_width": value} for name, value in widths.items()},
        "dominant_source_by_max_width": (
            "polynomial_times_parameterization_remainder" if failure else "outer_endpoint_remainder"
        ),
        "roundtrip_contains_before": True,
        "pre_renormalization_remainder": {"max_width": 7.4 if failure else 0.0018},
        "post_renormalization_remainder": {"max_width": 1.98 if failure else 0.029},
        "dependency_loss": "fixture dependency loss",
    }
    return {
        "all_native_observer_parity_bit_exact": True,
        "all_coverage_contains_native_remainder": True,
        "any_double_count_detected": False,
        "checkpoints": [checkpoint],
    }


def _derive_fixture() -> dict[str, object]:
    divergence = {
        "reproduction_status": "reproduced",
        "divergence": [
            {
                "first_coefficient_bit_divergence": {"step": 1},
                "first_remainder_divergence": {"step": 2},
                "first_physical_endpoint_divergence": {"step": 1},
                "first_tube_divergence": {"step": 1},
            }
            for _ in range(2)
        ],
    }
    substitutions = {
        "epsilon_decision_relevant_anywhere": False,
        "checkpoints": [
            {
                "all_substitutions_used_identical_prestate": True,
                "canonical_duplicate_checks": {"CDR": True, "CNI": True},
            }
        ],
    }
    rows = [
        {
            "checkpoint": "before_step_0320.npz",
            "label": "CDR_complete_carry",
            "family": "CDR",
            "accepted": "True",
            "minimum_target_margin": "0.00054",
        },
        {
            "checkpoint": "before_step_0320.npz",
            "label": "CNI_complete_carry",
            "family": "CNI",
            "accepted": "False",
            "minimum_target_margin": "-0.00049",
        },
    ]
    dense = {
        "dense_cni_parity_outcome": "DENSE_CNI_PARITY_NOT_EXPRESSIBLE",
        "basis_roundtrip_status": "closed",
        "dense_api_has_native_complete_composition": False,
        "reason": "fixture missing native carry",
        "fixtures": [
            {
                "exponent_sets_equal": True,
                "coefficient_roundtrip_bit_exact": True,
                "remainder_lo_roundtrip_bit_exact": True,
                "remainder_hi_roundtrip_bit_exact": True,
            }
        ],
    }
    return derive(
        reproductions=_reproductions(),
        divergence=divergence,
        substitutions=substitutions,
        substitution_rows=rows,
        dense=dense,
        first_accounting=_accounting(failure=False),
        failure_accounting=_accounting(failure=True),
    )


def test_root_cause_finalizer_derives_c4_and_rejects_double_count() -> None:
    report = _derive_fixture()
    assert report["root_cause_class"] == "C4"
    assert report["single_fix_authorization"] == "NO_FIX_AUTHORIZED"
    bad = _accounting(failure=True)
    bad["any_double_count_detected"] = True
    with pytest.raises(RuntimeError, match="double count"):
        derive(
            reproductions=_reproductions(),
            divergence={
                "reproduction_status": "reproduced",
                "divergence": [
                    {
                        "first_coefficient_bit_divergence": {"step": 1},
                        "first_remainder_divergence": {"step": 2},
                        "first_physical_endpoint_divergence": {"step": 1},
                        "first_tube_divergence": {"step": 1},
                    }
                ],
            },
            substitutions={
                "epsilon_decision_relevant_anywhere": False,
                "checkpoints": [{"all_substitutions_used_identical_prestate": True, "canonical_duplicate_checks": {"CDR": True, "CNI": True}}],
            },
            substitution_rows=[
                {"checkpoint": "before_step_0320.npz", "label": "CDR_complete_carry", "family": "CDR", "accepted": "True", "minimum_target_margin": "0.1"},
                {"checkpoint": "before_step_0320.npz", "label": "CNI_complete_carry", "family": "CNI", "accepted": "False", "minimum_target_margin": "-0.1"},
            ],
            dense={
                "dense_cni_parity_outcome": "DENSE_CNI_PARITY_NOT_EXPRESSIBLE",
                "basis_roundtrip_status": "closed",
                "dense_api_has_native_complete_composition": False,
                "reason": "fixture",
                "fixtures": [{"exponent_sets_equal": True, "coefficient_roundtrip_bit_exact": True, "remainder_lo_roundtrip_bit_exact": True, "remainder_hi_roundtrip_bit_exact": True}],
            },
            first_accounting=_accounting(failure=False),
            failure_accounting=bad,
        )


def _csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_full_horizon_figure_builder_emits_five_source_backed_figures(tmp_path: Path) -> None:
    common = tmp_path / "common.csv"
    common_row = {
        "step": 1, "time": 0.01, "time_hex": float(0.01).hex(), "both_completed": True,
        "qualification": "fixture", "flowstar_endpoint_x_width": 1, "torch_endpoint_x_width": 1.1,
        "flowstar_endpoint_y_width": 2, "torch_endpoint_y_width": 2.1,
        "flowstar_segment_tube_x_width": 1.2, "torch_segment_tube_x_width": 1.3,
        "flowstar_segment_tube_y_width": 2.2, "torch_segment_tube_y_width": 2.3,
        "flowstar_margin_y": 0.1, "torch_margin_y": 0.09,
    }
    _csv(common, [common_row, {**common_row, "step": 2, "time": 0.02, "time_hex": float(0.02).hex()}])
    delta = tmp_path / "delta.csv"
    _csv(delta, [{"step": 1, "time": 0.01, "endpoint_max_abs": 0.0, "endpoint_max_rel": 0.0, "endpoint_max_ulp": 0, "tube_max_abs": 0.0, "tube_max_rel": 0.0, "tube_max_ulp": 0}])
    metric_row = {"step": 1, "time": 0.0, "decision": "accept", "minimum_target_margin": 0.1, "model_remainder_width_max": 0.0, "parameterization_remainder_width_max": 0.1, "scale_max": 0.2, "inverse_scale_max": 5.0, "endpoint_width_max": 0.3, "tube_width_max": 0.4, "composition_ledger_width_max": 0.1}
    a3, a4 = tmp_path / "a3.csv", tmp_path / "a4.csv"
    _csv(a3, [metric_row])
    _csv(a4, [{**metric_row, "decision": "reject"}])
    categories = (
        "degree_gt4_dropped_polynomial", "polynomial_times_parameterization_remainder",
        "endpoint_remainder_times_parameterization_polynomial", "remainder_times_remainder",
        "outer_endpoint_remainder",
    )
    accounting_paths = []
    for index in range(2):
        path = tmp_path / f"accounting{index}.json"
        path.write_text(json.dumps({"checkpoints": [{"source_intervals": {name: {"max_width": float(index + 1)} for name in categories}}]}) + "\n", encoding="utf-8")
        accounting_paths.append(path)
    output = tmp_path / "figures"
    summary = build(argparse.Namespace(flow_common_prefix=common, diff_delta=delta, a3_metrics=a3, a4_metrics=a4, first_accounting=accounting_paths[0], failure_accounting=accounting_paths[1], output_dir=output))
    assert summary["figure_count"] == 5
    for artifact in summary["artifacts"]:
        assert (output / artifact["figure"]).read_text(encoding="utf-8").startswith("<svg")
        assert "eligibility" in (output / artifact["source_csv"]).read_text(encoding="utf-8").splitlines()[0]
