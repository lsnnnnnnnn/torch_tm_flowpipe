from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

REPORT_PATHS = [
    ROOT / "outputs" / "flowstar_normal_eval_h10" / "normal_eval_report.md",
    ROOT / "outputs" / "flowstar_right_map_scaling_diagnostics" / "right_map_scaling_report.md",
    ROOT / "docs" / "flowstar_right_map_scaling_source_map.md",
    ROOT / "docs" / "flowstar_horner_insertion_source_map.md",
]

CSV_PATHS = [
    ROOT / "outputs" / "flowstar_normal_eval_h10" / "normal_eval_summary.csv",
    ROOT / "outputs" / "flowstar_right_map_scaling_diagnostics" / "right_map_scaling_trace.csv",
]


def test_latest_reports_are_multiline_physical_markdown():
    for path in REPORT_PATHS:
        text = path.read_text(encoding="utf-8")
        assert text.count("\n") > 10, path
        assert "\r" not in text, path


def test_latest_csvs_have_one_physical_line_per_record_plus_header():
    for path in CSV_PATHS:
        text = path.read_text(encoding="utf-8")
        frame = pd.read_csv(path)
        assert len(text.splitlines()) == len(frame) + 1, path
        assert "\r" not in text, path
