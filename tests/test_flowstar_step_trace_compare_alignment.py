from __future__ import annotations

import csv
import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments" / "flowstar_step_trace_compare.py"

spec = importlib.util.spec_from_file_location("flowstar_step_trace_compare", SCRIPT)
assert spec is not None and spec.loader is not None
compare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare)


def _row(**kwargs):
    base = {
        "trace_source": kwargs.get("trace_source", "flowstar"),
        "accepted_step_index": kwargs.get("step_index", 0),
        "step_index": kwargs.get("step_index", 0),
        "attempt_index_within_step": kwargs.get("attempt_index", 1),
        "adaptive_attempt_index": kwargs.get("attempt_index", 1),
        "t_before": kwargs.get("t_before", 0.0),
        "h_try": kwargs.get("h", 0.025),
        "h": kwargs.get("h", 0.025),
        "accepted": kwargs.get("status", "accepted") == "accepted",
        "rejected": kwargs.get("status", "accepted") == "rejected",
        "status": kwargs.get("status", "accepted"),
    }
    base.update(kwargs)
    return base


def test_accepted_ordinal_h_mismatch_marked_noncausal():
    flow = [_row(trace_source="flowstar", h=0.0125, status="accepted")]
    noqueue = [_row(trace_source="torch_noqueue", h=0.025, status="accepted")]
    v2 = [_row(trace_source="torch_v2", h=0.025, status="accepted")]

    rows = compare.align_traces(flow, noqueue, v2)

    assert rows[0]["first_material_channel"] == "adaptive_step_alignment_mismatch"
    assert rows[0]["channel_attribution_valid"] is False
    assert rows[0]["comparison_kind"] == "accepted_ordinal_trace_diff_noncausal"


def test_attempt_aligned_status_divergence_is_adaptive_acceptance_policy():
    flow = [_row(trace_source="flowstar", h=0.025, status="rejected", rejection_reason="target miss")]
    noqueue = [_row(trace_source="torch_noqueue", h=0.025, status="accepted")]
    v2 = [_row(trace_source="torch_v2", h=0.025, status="accepted")]

    rows = compare.compare_attempt_aligned(flow, noqueue, v2)

    assert rows[0]["first_status_divergence"] == "adaptive_acceptance_policy"
    assert rows[0]["channel_attribution_valid"] is True
    assert rows[0]["flowstar_rejection_reason"] == "target miss"


def test_forced_h_ledger_reports_right_map_first_numeric_channel():
    flow = [
        _row(
            trace_source="flowstar",
            h=0.0125,
            status="accepted",
            right_map_range_width_sum=4.0,
            reset_width_sum=2.0,
            target_remainder_width_sum=0.0004,
            picard_ctrunc_normal_residual_width_sum=0.1,
            output_range_width_sum=3.0,
        )
    ]
    noqueue = [
        _row(
            trace_source="torch_noqueue",
            h=0.0125,
            status="accepted",
            right_map_range_width_sum=10.0,
            reset_width_sum=2.0,
            target_remainder_width_sum=0.0004,
            picard_ctrunc_normal_residual_width_sum=0.1,
            output_range_width_sum=3.0,
        )
    ]
    v2 = [
        _row(
            trace_source="torch_v2",
            h=0.0125,
            status="accepted",
            right_map_range_width_sum=10.0,
            reset_width_sum=2.0,
            target_remainder_width_sum=0.0004,
            picard_ctrunc_normal_residual_width_sum=0.1,
            output_range_width_sum=3.0,
        )
    ]

    rows = compare.compare_forced_h(flow, noqueue, v2)

    assert rows[0]["first_numeric_channel_divergence"] == "right_map_range"
    assert rows[0]["channel_attribution_valid"] is True


def test_missing_fields_mark_unknown_not_zero():
    flow = [_row(trace_source="flowstar", h=0.0125, status="accepted")]
    noqueue = [_row(trace_source="torch_noqueue", h=0.0125, status="accepted")]
    v2 = [_row(trace_source="torch_v2", h=0.0125, status="accepted")]

    rows = compare.compare_forced_h(flow, noqueue, v2)

    assert rows[0]["first_numeric_channel_divergence"] == "unknown"
    assert rows[0]["right_map_width_ratio_noqueue_over_flowstar"] is None


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = sorted({field for row in rows for field in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_cli_smoke_writes_attempt_and_forced_outputs(tmp_path):
    flow = [
        _row(trace_source="flowstar", h=0.025, status="rejected", attempt_index=1, rejection_reason="target miss"),
        _row(trace_source="flowstar", h=0.0125, status="accepted", attempt_index=2, right_map_range_width_sum=4.0),
    ]
    noqueue = [
        _row(trace_source="torch_noqueue", h=0.025, status="accepted", attempt_index=1),
        _row(trace_source="torch_noqueue", h=0.0125, status="accepted", attempt_index=2, right_map_range_width_sum=10.0),
    ]
    v2 = [
        _row(trace_source="torch_v2", h=0.025, status="accepted", attempt_index=1),
        _row(trace_source="torch_v2", h=0.0125, status="accepted", attempt_index=2, right_map_range_width_sum=10.0),
    ]
    flow_path = tmp_path / "flow.csv"
    noqueue_path = tmp_path / "noqueue.csv"
    v2_path = tmp_path / "v2.csv"
    out_dir = tmp_path / "out"
    _write_csv(flow_path, flow)
    _write_csv(noqueue_path, noqueue)
    _write_csv(v2_path, v2)

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--compare-mode",
            "all",
            "--flowstar-trace",
            str(flow_path),
            "--torch-noqueue-trace",
            str(noqueue_path),
            "--torch-v2-trace",
            str(v2_path),
            "--out-dir",
            str(out_dir),
        ],
        check=True,
        cwd=ROOT,
    )

    assert (out_dir / "attempt_aligned_trace_diff.csv").exists()
    assert (out_dir / "forced_h_trace_diff.csv").exists()
    assert (out_dir / "trace_divergence_report.md").exists()
