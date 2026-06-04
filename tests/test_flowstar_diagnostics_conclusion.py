from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_stage3_report_exists_and_records_negative_symbolic_result():
    report = ROOT / "outputs" / "flowstar_benchmark_diagnostics_stage3" / "symbolic_remainder_report.md"
    assert report.exists()
    text = report.read_text(encoding="utf-8")
    assert len(text.strip()) > 100
    assert "dependency_window_2_symbolic_o4_s4_q4" in text
    assert "range_only_o6_s4_baseline" in text
    assert "did not beat baseline" in text
    assert "reduced the local ordinary interval-remainder interaction" in text
    assert "diagnostic-only" in text
    assert "does not claim Flow* parity or a new full reachability algorithm" in text


def test_stage3_summary_has_symbolic_and_baseline_rows():
    summary = ROOT / "outputs" / "flowstar_benchmark_diagnostics_stage3" / "symbolic_remainder_summary.csv"
    assert summary.exists()
    with summary.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert any(row["symbolic_remainder"] == "true" for row in rows)
    assert any(row["run_id"] == "range_only_o6_s4_baseline" and row["symbolic_remainder"] == "false" for row in rows)


def test_final_decision_doc_exists_and_contains_required_conclusion_phrases():
    doc = ROOT / "docs" / "flowstar_vanderpol_pytorch_diagnostics_conclusion.md"
    assert doc.exists()
    text = doc.read_text(encoding="utf-8")
    assert "Flow* original/generated parity succeeded" in text
    assert "symbolic remainder prototype did not beat baseline" in text
    assert "diagnostic-only" in text


def test_readme_keeps_symbolic_remainder_out_of_supported_default_api():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    lowered = readme.lower()
    normalized = " ".join(lowered.split())

    assert "it intentionally does not integrate" in normalized
    assert "symbolic remainders" in normalized
    assert "not part of the supported default api" in normalized

    forbidden_claims = [
        "supports symbolic remainders by default",
        "symbolic remainders are supported by default",
        "symbolic remainder is a supported default feature",
        "symbolic remainder as a supported default feature",
    ]
    for claim in forbidden_claims:
        assert claim not in normalized
