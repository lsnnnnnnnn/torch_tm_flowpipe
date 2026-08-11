from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LEGACY_REPORTS = (
    "EVIDENCE_INTEGRITY_CORRECTIONS_20260811.md",
    "THREE_TOOL_PAIRWISE_COMPARISON_20260811.md",
    "FLOWSTAR_TORCH_O4_MATCHED_COMPARISON_20260811.md",
    "DIFFREACH_TORCH_DR7_MATCHED_COMPARISON_20260811.md",
    "VDP_RAW_REMAINDER_ROOT_CAUSE_20260811.md",
    "VDP_SCHEDULE_VALIDATOR_CAUSALITY_20260811.md",
    "TORCH_FIXED_SUPPORT_DESCRIPTOR_BRIDGE_20260811.md",
    "TORCH_SINGLE_IMPROVEMENT_RESULT_20260811.md",
)
CURRENT_REPORTS = (
    "EVIDENCE_PACKAGE_TRACKED_CLOSURE_20260811.md",
    "FLOWSTAR_TORCH_FIXED_SCHEDULE_COMMON_PREFIX_20260811.md",
    "DIFFREACH_TORCH_DR7_FULL_HORIZON_CLOSURE_20260811.md",
    "COMPLETE_O4_CARRY_SEMANTICS_ROOT_CAUSE_20260811.md",
    "THREE_TOOL_PAIRWISE_STATUS_20260811.md",
)
LEGACY_REQUIRED_HEADINGS = (
    "## Outcome",
    "## Eligibility",
    "## What is comparable",
    "## What is unavailable",
    "## Negative results",
    "## Exact evidence paths",
)
CURRENT_REQUIRED_HEADINGS = (
    "## Outcome",
    "## Eligibility",
    "## Contract",
    "## What was actually run",
    "## Exact results",
    "## What is comparable",
    "## What remains unavailable",
    "## Negative results",
    "## Limitations",
    "## Evidence paths",
    "## Reproduction commands",
    "## Next authorized action",
)


def test_every_current_report_leads_with_required_contract_sections() -> None:
    for name in LEGACY_REPORTS:
        text = (ROOT / "docs" / name).read_text(encoding="utf-8")
        positions = [text.index(heading) for heading in LEGACY_REQUIRED_HEADINGS]
        assert positions == sorted(positions), name
    for name in CURRENT_REPORTS:
        text = (ROOT / "docs" / name).read_text(encoding="utf-8")
        positions = [text.index(heading) for heading in CURRENT_REQUIRED_HEADINGS]
        assert positions == sorted(positions), name


def test_pairwise_report_forbids_transitive_ranking() -> None:
    text = (ROOT / "docs/THREE_TOOL_PAIRWISE_STATUS_20260811.md").read_text()
    assert "do not imply `Flow* > Torch > DiffReach`" in text
    assert "FLOWSTAR_TORCH_FIXED_SCHEDULE_COMMON_PREFIX_ONLY" in text
    assert "DIFFREACH_TORCH_DR7_OPERATOR_EQUIVALENCE_CLOSED" in text
    assert "DIFFREACH_TORCH_DR7_FULL_HORIZON_DIVERGED" in text
    assert "CARRY_MISSING_SYMBOLIC_SEMANTICS" in text
    assert "NO_FIX_AUTHORIZED" in text
    assert "VALID_PAIRWISE_COMPARISON_CLOSED" not in text


def test_old_pairwise_reports_are_explicitly_superseded() -> None:
    for name in (
        "THREE_TOOL_PAIRWISE_COMPARISON_20260811.md",
        "FLOWSTAR_TORCH_O4_MATCHED_COMPARISON_20260811.md",
        "DIFFREACH_TORCH_DR7_MATCHED_COMPARISON_20260811.md",
    ):
        first = "\n".join((ROOT / "docs" / name).read_text().splitlines()[:8]).lower()
        assert "supersed" in first, name
