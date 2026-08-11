from __future__ import annotations

import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = (
    ROOT
    / "outputs/three_tool_full_horizon_pairwise_carry_closure_20260811"
    / "20260811T191549Z"
)


def _json(relative: str) -> dict[str, object]:
    return json.loads((PACKAGE / relative).read_text(encoding="utf-8"))


def test_current_reports_match_tracked_package_json() -> None:
    if not PACKAGE.is_dir():
        pytest.skip("the tracked package is introduced at H2, after tested-source H1")
    flow = _json("04_flowstar_torch_fixed_schedule/common_prefix/summary.json")
    diff = _json("05_diffreach_torch_full_horizon/cross_tool_comparison/comparison.json")
    carry = _json("10_root_cause/root_cause.json")
    flow_doc = (ROOT / "docs/FLOWSTAR_TORCH_FIXED_SCHEDULE_COMMON_PREFIX_20260811.md").read_text()
    diff_doc = (ROOT / "docs/DIFFREACH_TORCH_DR7_FULL_HORIZON_CLOSURE_20260811.md").read_text()
    carry_doc = (ROOT / "docs/COMPLETE_O4_CARRY_SEMANTICS_ROOT_CAUSE_20260811.md").read_text()

    assert str(flow["outcome"]) in flow_doc
    assert str(flow["flowstar"]["accepted_steps"]) in flow_doc
    assert str(flow["torch"]["accepted_steps"]) in flow_doc
    assert str(flow["torch"]["first_failure_step"]) in flow_doc
    assert str(diff["outcome"]) in diff_doc
    assert str(diff["first_divergence_step"]) in diff_doc
    assert str(diff["max_ulp"]) in diff_doc.replace(",", "")
    assert str(carry["outcome"]) in carry_doc
    assert str(carry["root_cause_class"]) in carry_doc
    assert str(carry["single_fix_authorization"]) in carry_doc
    dominant = carry["composition"]["pre_failure_dominant_source_width"]
    assert str(dominant) in carry_doc
