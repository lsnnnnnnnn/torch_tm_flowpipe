from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from experiments.build_three_tool_causal_figures import build


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _fixture(root: Path) -> None:
    raw = (
        root
        / "07_flowstar_torch_raw_remainder/independent_analysis/artifacts/run"
        / "raw_remainder_node_comparison.csv"
    )
    raw.parent.mkdir(parents=True)
    with raw.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "semantic_node",
                "suboperation_order",
                "operation",
                "flowstar_width",
                "torch_width",
                "width_delta_flowstar_minus_torch",
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
                "semantic_node": "x_squared",
                "suboperation_order": 1,
                "operation": "coefficient_interval_uncertainty",
                "flowstar_width": 0.2,
                "torch_width": 0.0,
                "width_delta_flowstar_minus_torch": 0.2,
            }
        )
    _json(
        root
        / "08_schedule_validator_matrix/adaptive_schedule/artifacts/run"
        / "schedule_validator_matrix.json",
        {
            "checkpoints": [
                {
                    "checkpoint": "last_common_prestate_before_first_split",
                    "t_pre": 0.18,
                    "h": 0.02,
                    "rows": [
                        {
                            "candidate_producer": producer,
                            "flowstar_validator": {
                                "decision": decision,
                                "margins": [margin, margin + 0.1],
                            },
                            "torch_validator": {
                                "decision": decision,
                                "margins": [margin, margin + 0.1],
                            },
                        }
                        for producer, decision, margin in (
                            ("torch_complete_o4", "accept", 0.01),
                            ("flowstar_complete_o4", "reject", -0.01),
                        )
                    ],
                }
            ]
        },
    )
    attributions = []
    for batch in (1, 64):
        for index, factor in enumerate(("support", "picard", "validator", "carry")):
            attributions.append(
                {
                    "batch": batch,
                    "changed_factor": factor,
                    "from": f"A{index}",
                    "to": f"A{index + 1}",
                    "margin_delta": 0.001 * (index + 1),
                    "t1_max_endpoint_width_delta": -0.01 * (index + 1),
                    "t1_max_raw_remainder_width_delta": -0.0001,
                    "t1_max_tube_width_delta": -0.01,
                    "comparison_eligibility": "same B/h/time/output/success",
                }
            )
    cells = []
    for batch in (1, 64):
        for index in range(5):
            cells.append(
                {
                    "cell": f"A{index}",
                    "batch": batch,
                    "completed_steps": 1000,
                    "validated_horizon": 10.0,
                    "completed_requested_gate": True,
                    "runtime_by_stage_seconds": {
                        "polynomial_picard": 1.0,
                        "validation": 2.0,
                        "carry": 0.5,
                        "output_object": 0.25,
                    },
                }
            )
    _json(
        root / "10_bridge_ladder/G3/artifacts/run/bridge_ladder.json",
        {"adjacent_factor_attribution": attributions, "cells": cells},
    )
    for relative in (
        "03_native_flowstar/official_vdp",
        "04_native_diffreach/official_vdp",
        "05_native_torch_complete_o4/authoritative",
        "06_native_torch_fixed_dr7/t10_cpu",
    ):
        runner = root / relative
        _json(runner / "config.json", {"eligibility_status": "capability_only"})
        _json(runner / "summary.json", {"status": "pass"})
        _json(
            runner / "artifacts/run/summary.json",
            {
                "result_status": "completed",
                "representation": "fixture",
                "partition_count": 1,
                "horizon_validated": 10.0,
            },
        )


def test_causal_figure_builder_emits_five_figures_with_source_csv(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    output = tmp_path / "figures"
    _fixture(run_root)
    summary = build(argparse.Namespace(run_root=run_root, output_dir=output))
    assert summary["outcome"] == "CAUSAL_FIGURES_BUILT"
    assert summary["figure_count"] == 5
    assert summary["constraints"]["every_figure_has_source_csv"]
    for artifact in summary["artifacts"]:
        csv_path = output / artifact["source_csv"]
        svg_path = output / artifact["figure"]
        assert csv_path.is_file()
        assert svg_path.read_text(encoding="utf-8").startswith("<svg")
        header = csv_path.read_text(encoding="utf-8").splitlines()[0]
        assert "eligibility" in header
        assert "sample_count" in header
