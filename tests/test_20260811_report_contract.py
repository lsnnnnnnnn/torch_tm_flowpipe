from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORTS = (
    "EVIDENCE_INTEGRITY_CORRECTIONS_20260811.md",
    "THREE_TOOL_PAIRWISE_COMPARISON_20260811.md",
    "FLOWSTAR_TORCH_O4_MATCHED_COMPARISON_20260811.md",
    "DIFFREACH_TORCH_DR7_MATCHED_COMPARISON_20260811.md",
    "VDP_RAW_REMAINDER_ROOT_CAUSE_20260811.md",
    "VDP_SCHEDULE_VALIDATOR_CAUSALITY_20260811.md",
    "TORCH_FIXED_SUPPORT_DESCRIPTOR_BRIDGE_20260811.md",
    "TORCH_SINGLE_IMPROVEMENT_RESULT_20260811.md",
)
REQUIRED_HEADINGS = (
    "## Outcome",
    "## Eligibility",
    "## What is comparable",
    "## What is unavailable",
    "## Negative results",
    "## Exact evidence paths",
)


def test_every_current_report_leads_with_required_contract_sections() -> None:
    for name in REPORTS:
        text = (ROOT / "docs" / name).read_text(encoding="utf-8")
        positions = [text.index(heading) for heading in REQUIRED_HEADINGS]
        assert positions == sorted(positions), name


def test_pairwise_report_forbids_transitive_ranking() -> None:
    text = (ROOT / "docs/THREE_TOOL_PAIRWISE_COMPARISON_20260811.md").read_text()
    assert "does not imply `Flow* > Torch > DiffReach`" in text
    assert "PAIRWISE_COMPARISON_PARTIAL" in text
    assert "VALID_PAIRWISE_COMPARISON_CLOSED" in text
