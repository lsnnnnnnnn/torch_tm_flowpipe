from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

REPORT_PATHS = [
    ROOT / "outputs" / "flowstar_horner_insertion_diagnostics" / "horner_insertion_report.md",
    ROOT / "outputs" / "flowstar_next_mechanism_decision.md",
    ROOT / "outputs" / "flowstar_normal_eval_h10" / "normal_eval_report.md",
    ROOT / "outputs" / "flowstar_right_map_scaling_diagnostics" / "right_map_scaling_report.md",
    ROOT / "docs" / "flowstar_right_map_scaling_source_map.md",
    ROOT / "docs" / "flowstar_horner_insertion_source_map.md",
]

OPTIONAL_REPORT_PATHS = [
    ROOT / "outputs" / "flowstar_queue_state_audit" / "queue_state_report.md",
    ROOT / "outputs" / "flowstar_normalized_insertion_symqueue_v2_h10" / "symqueue_v2_report.md",
    ROOT / "outputs" / "flowstar_symbolic_queue_v2_h10" / "symqueue_v2_report.md",
    ROOT / "docs" / "flowstar_symbolic_queue_v2_notes.md",
    ROOT / "docs" / "flowstar_source_queue_semantics_audit.md",
    ROOT / "docs" / "flowstar_symbolic_queue_v2_audit_conclusion.md",
]

CSV_PATHS = [
    ROOT / "outputs" / "flowstar_normal_eval_h10" / "normal_eval_summary.csv",
    ROOT / "outputs" / "flowstar_right_map_scaling_diagnostics" / "right_map_scaling_trace.csv",
]


def _assert_markdown_rows_are_physical_lines(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    assert text.count("\n") > 10, path
    assert "\r" not in text, path
    table_lines = [line for line in text.splitlines() if line.lstrip().startswith("|")]
    assert not any(line.count("|") > 35 for line in table_lines), path


def test_latest_reports_are_multiline_physical_markdown():
    for path in REPORT_PATHS:
        _assert_markdown_rows_are_physical_lines(path)
    for path in OPTIONAL_REPORT_PATHS:
        if path.exists():
            _assert_markdown_rows_are_physical_lines(path)


def test_latest_csvs_have_one_physical_line_per_record_plus_header():
    for path in CSV_PATHS:
        text = path.read_text(encoding="utf-8")
        frame = pd.read_csv(path)
        assert len(text.splitlines()) == len(frame) + 1, path
        assert "\r" not in text, path
