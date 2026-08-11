from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
import sys

import pytest

from torch_tm_flowpipe.comparison_contract import vdp_identity_hashes


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments" / "compare_flowstar_torch_fixed_schedule.py"
SPEC = importlib.util.spec_from_file_location("compare_flowstar_torch_fixed_schedule", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = sorted({str(key) for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _flow_row(step: int = 0, **updates: object) -> dict[str, object]:
    t_lo = step * MODULE.H
    t_hi = (step + 1) * MODULE.H
    row: dict[str, object] = {
        "accepted": "true",
        "status": "accepted",
        "h": MODULE.H,
        "h_hex": MODULE.H.hex(),
        "t_before": t_lo,
        "t_before_hex": t_lo.hex(),
        "t_after": t_hi,
        "t_after_hex": t_hi.hex(),
        "prestate_state_canonical": f"state-{step}",
        "retained_coefficients_canonical": f"coefficients-{step}",
        "extracted_center_x": 1.0,
        "extracted_center_y": 2.0,
        "extracted_scale_x": 0.2,
        "extracted_scale_y": 0.1,
        "flowstar_tau_h_endpoint_x_lo": 1.0,
        "flowstar_tau_h_endpoint_x_hi": 1.3,
        "flowstar_tau_h_endpoint_y_lo": 2.0,
        "flowstar_tau_h_endpoint_y_hi": 2.2,
        "flowstar_full_step_tube_x_lo": 0.9,
        "flowstar_full_step_tube_x_hi": 1.4,
        "flowstar_full_step_tube_y_lo": 1.9,
        "flowstar_full_step_tube_y_hi": 2.3,
        "flowstar_full_step_tube_source_object": "accepted_result_Flowpipe_composition_after_remainder_refinement",
        "flowstar_tau_h_endpoint_source_object": "accepted_result_Flowpipe_composition_at_tau_h_after_remainder_refinement",
        "flowstar_full_step_tube_includes_ordinary_remainder": "true",
        "flowstar_full_step_tube_includes_symbolic_output_width": "true",
        "flowstar_tau_h_endpoint_includes_ordinary_remainder": "true",
        "flowstar_tau_h_endpoint_includes_symbolic_output_width": "true",
        "raw_remainder_after_poly_diff_x_lo": -2e-5,
        "raw_remainder_after_poly_diff_x_hi": 3e-5,
        "raw_remainder_after_poly_diff_y_lo": -4e-5,
        "raw_remainder_after_poly_diff_y_hi": 5e-5,
        "stage_runtime_seconds": 0.1,
    }
    row.update(updates)
    return row


def _torch_row(step: int = 0, **updates: object) -> dict[str, object]:
    t_lo = step * MODULE.H
    t_hi = (step + 1) * MODULE.H
    row: dict[str, object] = {
        "status": "accepted",
        "h_attempted": MODULE.H,
        "h_accepted": MODULE.H,
        "h_attempted_hex": MODULE.H.hex(),
        "h_accepted_hex": MODULE.H.hex(),
        "t_lo": t_lo,
        "t_lo_hex": t_lo.hex(),
        "t_hi": t_hi,
        "t_hi_hex": t_hi.hex(),
        "prestate_sha256": "a" * 64,
        "retained_coefficient_sha256": "b" * 64,
        "endpoint_x_lo": 1.0,
        "endpoint_x_hi": 1.31,
        "endpoint_y_lo": 2.0,
        "endpoint_y_hi": 2.21,
        "segment_x_lo": 0.91,
        "segment_x_hi": 1.41,
        "segment_y_lo": 1.91,
        "segment_y_hi": 2.31,
        "raw_remainder": json.dumps([[-2e-5, -4e-5], [3e-5, 5e-5]]),
        "post_poly_diff_remainder": json.dumps([[-2e-5, -4e-5], [3e-5, 5e-5]]),
        "target_margins": json.dumps([[7e-5, 5e-5]]),
        "prestate_center": json.dumps([1.0, 2.0]),
        "prestate_scale": json.dumps([0.2, 0.1]),
        "raw_endpoint_published": "True",
        "endpoint_tightening_applied": "False",
        "schedule_kind": "fixed",
        "stage_runtime_s": 0.2,
    }
    row.update(updates)
    return row


def _metadata() -> list[dict[str, str]]:
    return [
        {"key": key, "value": value}
        for key, value in MODULE.EXPECTED_FLOWSTAR_METADATA.items()
    ]


def _summary(accepted_steps: int = 1) -> dict[str, object]:
    return {
        "requested_horizon": 10.0,
        "requested_order": 4,
        "support": "complete_total_degree_O4",
        "partition": "B1",
        "partition_count": 1,
        "contract_identity": vdp_identity_hashes(),
        "cutoff": 1e-10,
        "target_remainder_radius": 1e-4,
        "schedule": {
            "kind": "fixed",
            "h_hex": MODULE.H.hex(),
            "requested_steps": 1000,
            "adaptive_fallback_allowed": False,
        },
        "accepted_steps": accepted_steps,
        "completed_requested_horizon": False,
        "status": "failed",
        "failure_type": "validated_remainder_target_miss",
        "fallback_count": 0,
        "endpoint_repair_used": False,
        "peak_rss_bytes": 123456,
    }


def _args(tmp_path: Path, flow_rows: list[dict[str, object]], torch_rows: list[dict[str, object]]) -> argparse.Namespace:
    flow = tmp_path / "flow.csv"
    metadata = tmp_path / "flow_metadata.csv"
    torch = tmp_path / "torch.csv"
    summary = tmp_path / "torch_summary.json"
    _write_csv(flow, flow_rows)
    _write_csv(metadata, _metadata())
    _write_csv(torch, torch_rows)
    summary.write_text(json.dumps(_summary(sum(row.get("status") == "accepted" for row in torch_rows))), encoding="utf-8")
    return argparse.Namespace(
        flowstar_trace=flow,
        flowstar_metadata=metadata,
        torch_segments=torch,
        torch_summary=summary,
        flowstar_resource=None,
        output_dir=tmp_path / "comparison",
    )


def test_common_prefix_is_derived_only_from_jointly_accepted_steps(tmp_path: Path) -> None:
    flow_rejected = {"status": "rejected", "accepted": "false", "message": "target miss"}
    torch_rejected = {"status": "rejected", "raw_endpoint_published": "False", "message": "target miss"}
    args = _args(tmp_path, [_flow_row(), flow_rejected], [_torch_row(), torch_rejected])

    summary = MODULE.compare(args)

    assert summary["outcome"] == "FLOWSTAR_TORCH_FIXED_SCHEDULE_COMMON_PREFIX_ONLY"
    assert summary["shared"]["accepted_steps"] == 1
    assert summary["shared"]["first_failure"] == [
        {"tool": "flowstar", "step": 2, "time": 0.01},
        {"tool": "torch", "step": 2, "time": 0.01},
    ]
    with (args.output_dir / "common_prefix.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert float(rows[0]["endpoint_x_width_delta_torch_minus_flowstar"]) == pytest.approx(0.01)
    assert rows[0]["torch_peak_rss_bytes"] == "123456"


def test_nonfixed_flowstar_hex_schedule_is_rejected(tmp_path: Path) -> None:
    args = _args(tmp_path, [_flow_row(h_hex=(0.005).hex())], [_torch_row()])

    with pytest.raises(ValueError, match="hex mismatch"):
        MODULE.compare(args)


def test_rejected_torch_candidate_cannot_publish_endpoint(tmp_path: Path) -> None:
    rejected = _torch_row(
        status="rejected",
        raw_endpoint_published="True",
    )
    args = _args(
        tmp_path,
        [_flow_row(), {"status": "rejected", "accepted": "false"}],
        [rejected],
    )

    with pytest.raises(ValueError, match="published as an endpoint"):
        MODULE.compare(args)
