import copy
from pathlib import Path

import pytest

from torch_tm_flowpipe.archcomp26_final_report import (
    _contract_lines,
    attempt_rows,
    comparison_assessment,
    collect_report,
    render_markdown,
    timing_summary,
    width_rows,
)


ROOT = Path(__file__).resolve().parents[1]


def _sample(role, index, process_total):
    return {
        "role": role,
        "index": index,
        "outcome": "completed",
        "validated_extent": {"kind": "time_s", "value": 5.0},
        "timing_s": {
            "process_total": process_total,
            "driver_total": process_total - 1.0,
            "compile": 0.0,
            "controller_nn": 0.25,
            "solver_core": process_total - 2.0,
            "validation": 0.5,
            "observer": 0.125,
            "output": 0.125,
            "plot_report": 0.125,
        },
        "peak_memory_bytes": {"host": 100, "device": 50},
    }


def test_current_draft_is_deterministic_and_final_gate_refuses():
    report = collect_report()
    assert {
        "experiments_paused",
        "unresolved_contracts=16",
        "matrix_not_terminal=not_started",
        "unassessed_support=64",
        "nonterminal_cells=64",
        "missing_plot_artifacts=16",
    } <= set(report["final_readiness_reasons"])
    text = render_markdown(report)
    assert text == render_markdown(copy.deepcopy(report))
    assert text == (ROOT / "docs/ARCHCOMP26_FINAL_REPORT_DRAFT.md").read_text(
        encoding="utf-8"
    )
    assert "不可作为最终成绩" in text
    assert text.count("(`") >= 16
    assert text.count("### 完整性、性质与结果资格") == 16
    assert text.count("- 宽度记录状态：") == 16
    assert text.count("h / work / point / validation") == 16
    assert text.count("cutoff / cap / SR") == 16
    assert text.count("updates / NN | arithmetic") == 16
    assert text.count("checker / early-stop") == 16
    assert text.count("source / binary identity") == 16
    assert '"revision":null' in text
    assert (
        "h=unresolved; work=unresolved; point=unresolved; "
        "validation=unresolved"
    ) in text
    assert "旧 14 项到 2026 manifest 的差异索引" in text
    assert "75.250099" not in text
    assert "1533.752052" not in text
    assert "/Users/" not in text
    assert "/srv/" not in text
    with pytest.raises(ValueError, match="final report is not ready"):
        render_markdown(report, final=True)


def test_steady_summary_excludes_cold_and_preserves_phase_boundaries():
    record = {
        "run": {
            "status": "completed",
            "requested_extent": {"kind": "time_s", "value": 5.0},
        },
        "eligibility": {"performance_measurement_eligible": True},
        "samples": [
            _sample("cold", 0, 100.0),
            _sample("steady", 0, 3.0),
            _sample("steady", 1, 5.0),
        ],
    }
    summary = timing_summary(record)
    assert summary is not None
    assert summary["cold_n"] == 1
    assert summary["cold_process_s"] == 100.0
    assert summary["steady_n"] == 2
    assert summary["steady_process_median_s"] == 4.0
    assert summary["steady_process_min_s"] == 3.0
    assert summary["steady_process_max_s"] == 5.0
    assert summary["steady_phase_medians_s"]["solver_core"] == 2.0


def test_width_rows_are_absolute_and_keep_each_view_domain():
    coordinate = {
        "coordinate": "x",
        "union": {"lo": -1.0, "hi": 2.0, "width": 3.0},
        "per_partition_width": {"mean": 1.5, "max": 2.0},
    }
    record = {
        "run": {"status": "completed"},
        "widths": {
            "status": "complete",
            "coordinate_order": ["x"],
            "coordinate_units": ["m"],
            "endpoint": {
                "status": "complete",
                "domain": {"kind": "time_s", "start": 5.0, "end": 5.0},
                "per_coordinate": [coordinate],
            },
            "last_segment_tube": {
                "status": "complete",
                "domain": {"kind": "time_s", "start": 4.9, "end": 5.0},
                "per_coordinate": [coordinate],
            },
            "full_horizon_tube": {
                "status": "complete",
                "domain": {"kind": "time_s", "start": 0.0, "end": 5.0},
                "per_coordinate": [coordinate],
            },
        },
    }
    rows = width_rows(record)
    assert [row["view"] for row in rows] == [
        "endpoint", "last_segment_tube", "full_horizon_tube"
    ]
    assert [row["domain"] for row in rows] == [
        {"kind": "time_s", "start": 5.0, "end": 5.0},
        {"kind": "time_s", "start": 4.9, "end": 5.0},
        {"kind": "time_s", "start": 0.0, "end": 5.0},
    ]
    assert all(row["width"] == 3.0 for row in rows)
    assert all(row["unit"] == "m" for row in rows)


def test_width_rows_keep_partial_views_and_common_series_point():
    coordinate = {
        "coordinate": "x",
        "union": {"lo": -0.5, "hi": 1.5, "width": 2.0},
        "per_partition_width": {"mean": 1.0, "max": 1.25},
    }
    unavailable = {
        "status": "unavailable",
        "domain": {"kind": "time_s", "start": 0.5, "end": 0.5},
        "per_coordinate": [],
    }
    complete = {
        "status": "complete",
        "domain": {"kind": "time_s", "start": 0.0, "end": 0.5},
        "per_coordinate": [coordinate],
    }
    record = {
        "run": {"status": "timeout"},
        "widths": {
            "status": "partial",
            "coordinate_order": ["x"],
            "coordinate_units": ["m"],
            "endpoint": unavailable,
            "last_segment_tube": copy.deepcopy(complete),
            "full_horizon_tube": copy.deepcopy(complete),
            "series": [{
                "extent": {"kind": "time_s", "value": 0.5},
                "per_coordinate": [coordinate],
            }],
        },
    }
    rows = width_rows(record, {"kind": "time_s", "value": 0.5})
    assert [row["view"] for row in rows] == [
        "last_segment_tube",
        "full_horizon_tube",
        "four_way_common_prefix_point",
    ]
    assert all(row["unit"] == "m" for row in rows)

    series_only = copy.deepcopy(record)
    series_only["widths"]["last_segment_tube"] = copy.deepcopy(unavailable)
    series_only["widths"]["full_horizon_tube"] = copy.deepcopy(unavailable)
    rows = width_rows(series_only, {"kind": "time_s", "value": 0.25})
    assert [row["view"] for row in rows] == ["last_measured_series_point"]
    assert rows[0]["domain"] == {
        "kind": "time_s", "start": 0.5, "end": 0.5,
    }


def _comparison_row():
    methods = ["pytorch_gpu", "huan", "xiangru", "flowstar_native"]
    record = {
        "run": {
            "status": "completed",
            "requested_horizon_completed": True,
            "requested_extent": {"kind": "time_s", "value": 1.0},
            "partition_coverage": {
                "requested_partitions": 2,
                "completed_partitions": 2,
                "failed_partitions": 0,
                "unattempted_partitions": 0,
            },
        },
        "property": {"certificate_status": "passed"},
        "eligibility": {
            "mathematical_contract_known": True,
            "certificate_semantics_passed": True,
            "finite_outputs": True,
            "performance_measurement_eligible": True,
            "numerical_soundness_class": "formally outward by construction",
            "soundness_scope": "fixed workload",
        },
        "widths": {
            "status": "complete",
            "coordinate_order": ["x"],
            "coordinate_units": ["m"],
            "aggregation_semantics": "union_and_per_partition",
            "series": [
                {"extent": {"kind": "time_s", "value": 0.0}},
                {"extent": {"kind": "time_s", "value": 1.0}},
            ],
        },
    }
    cell = {
        "runtime": {
            "hardware": "host-a", "cpu_threads": 1, "gpu": "gpu-0",
            "resource_limits": {
                "exclusive_host": True,
                "exclusive_gpu_device": True,
                "max_host_memory_bytes": 1_000_000_000,
                "max_device_memory_bytes": 1_000_000_000,
            },
            "timeout_s": 10,
        },
        "measurement_plan": {
            "target_steady_runs": 5,
            "steady_runs": 5,
            "timing_boundary_version": "total_configuration_v2",
        },
    }
    return methods, {
        "cells": {
            method: {"cell": copy.deepcopy(cell), "result": copy.deepcopy(record)}
            for method in methods
        }
    }


def test_four_way_comparability_is_derived_and_fail_closed():
    methods, row = _comparison_row()
    assessment = comparison_assessment(row, methods)
    assert assessment["time_comparable"] is True
    assert assessment["width_comparable"] is True
    assert assessment["common_width_prefix"] == {
        "kind": "time_s", "value": 1.0,
    }
    assert assessment["ranking_eligible"] is True

    mismatched = copy.deepcopy(row)
    mismatched["cells"]["huan"]["result"]["widths"][
        "coordinate_units"
    ] = ["cm"]
    assessment = comparison_assessment(mismatched, methods)
    assert assessment["width_comparable"] is False
    assert assessment["ranking_eligible"] is False
    assert "width_order_units_or_aggregation_mismatch" in assessment["reasons"]

    truncated = copy.deepcopy(row)
    truncated["cells"]["xiangru"]["result"]["widths"]["series"].pop()
    assessment = comparison_assessment(truncated, methods)
    assert assessment["common_width_prefix"] == {
        "kind": "time_s", "value": 0.0,
    }
    assert assessment["width_comparable"] is False


def test_attempt_rows_keep_failed_raw_wall_and_receipt():
    sample = _sample("diagnostic", 0, 7.5)
    sample.update({
        "outcome": "resource_guard",
        "validated_extent": {"kind": "time_s", "value": 0.5},
        "failure": {"reason_code": "resource_guard"},
        "process_identity": {"invocation_id": "attempt-1"},
        "artifact": {"path": "attempt-1.json", "sha256": "a" * 64},
    })
    row = attempt_rows({"samples": [sample]})[0]
    assert row["process_total"] == 7.5
    assert row["reason"] == "resource_guard"
    assert row["invocation_id"] == "attempt-1"


def test_mapping_key_order_does_not_change_rendered_bytes():
    report = collect_report()

    def reversed_mappings(value):
        if isinstance(value, dict):
            return {
                key: reversed_mappings(item)
                for key, item in reversed(list(value.items()))
            }
        if isinstance(value, list):
            return [reversed_mappings(item) for item in value]
        return value

    assert render_markdown(report) == render_markdown(reversed_mappings(report))


def test_resolved_contract_section_exposes_complete_shared_configuration():
    sha = "a" * 64
    row = {
        "instance": {
            "visualization": "x over time",
            "contract": {
                "record": {"path": "contracts/synthetic.json", "sha256": sha},
            },
        },
        "contract_record": {
            "profile": "full_execution_contract_v1",
            "fields": {
                "dynamics": {
                    "source": "model.py", "sha256": sha,
                    "equations": ["x'=u"],
                },
                "controller": {
                    "source": "controller.onnx", "sha256": sha,
                    "input_output_order": ["x", "u"],
                },
                "variable_order": ["x"],
                "initial_set": {
                    "source": "initial.json", "sha256": sha,
                    "partitions": [{"x": [0, 1]}], "boxes_sha256": sha,
                },
                "disturbance": {"x": [0, 0]},
                "integration": {"horizon": 5.0},
                "controller_update": {
                    "period": 0.1, "scheduled_updates": 50,
                    "schedule_points": [0.1 * index for index in range(50)],
                    "schedule_semantics": "update at each left endpoint",
                },
                "property": {
                    "formula": "x <= 2", "time_semantics": "all t in [0,5]",
                    "pass_condition": "upper(x) <= 2",
                },
                "width_comparison": {
                    "coordinate_units": ["m"],
                    "sample_points": [0, 5],
                    "aggregation_semantics": "union_and_per_partition",
                },
            },
        },
    }
    text = "\n".join(_contract_lines(row))
    for expected in (
        "contracts/synthetic.json", "model.py", "x'=u", "controller.onnx",
        "I/O 顺序", "initial.json", '"x":[0,1]', "扰动",
        "all t in [0,5]", "upper(x) <= 2", "update at each left endpoint",
    ):
        assert expected in text
