import copy
from pathlib import Path

import pytest

from torch_tm_flowpipe.archcomp26_final_report import (
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
            "solver_core": process_total - 2.0,
            "compile": 0.0,
            "validation": 0.5,
            "observer_output": 0.25,
            "plot_report": 0.125,
        },
    }


def test_current_draft_is_deterministic_and_final_gate_refuses():
    report = collect_report()
    assert {
        "experiments_paused",
        "unresolved_contracts=16",
        "matrix_not_terminal=not_started",
        "unassessed_support=64",
        "nonterminal_cells=64",
    } <= set(report["final_readiness_reasons"])
    text = render_markdown(report)
    assert text == render_markdown(copy.deepcopy(report))
    assert text == (ROOT / "docs/ARCHCOMP26_FINAL_REPORT_DRAFT.md").read_text(
        encoding="utf-8"
    )
    assert "不可作为最终成绩" in text
    assert text.count("(`") >= 16
    assert text.count("### 完整性、性质与结果资格") == 16
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
            "endpoint": {
                "domain": {"kind": "time_s", "start": 5.0, "end": 5.0},
                "per_coordinate": [coordinate],
            },
            "last_segment_tube": {
                "domain": {"kind": "time_s", "start": 4.9, "end": 5.0},
                "per_coordinate": [coordinate],
            },
            "full_horizon_tube": {
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
