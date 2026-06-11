from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments" / "flowstar_raw_remainder_compat_h5_divergence_audit.py"
spec = importlib.util.spec_from_file_location("flowstar_raw_remainder_compat_h5_divergence_audit", SCRIPT)
assert spec is not None and spec.loader is not None
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)

H5_DIR = ROOT / "outputs" / "flowstar_raw_remainder_compat_h5"
H5_CSVS = [
    "h5_summary.csv",
    "h5_width_vs_flowstar.csv",
    "h5_schedule_compare.csv",
    "h5_sample_containment.csv",
    "h5_segments.csv",
]


def test_fixture_detects_first_schedule_divergence_after_prefix_match():
    flowstar_h = [0.1, 0.2, 0.3, 0.4]
    compat_h = [0.1, 0.2, 0.31, 0.4]

    result = audit.first_schedule_divergence(flowstar_h, compat_h)

    assert result["step_index"] == 2
    assert result["prefix_match_count"] == 2
    assert result["reason"] == "h_mismatch"


def test_fixture_detects_width_ratio_crossing():
    rows = [
        {"t": 0.1, "compat_over_flowstar_ratio": 1.0},
        {"t": 0.2, "compat_over_flowstar_ratio": 1.2},
        {"t": 0.3, "compat_over_flowstar_ratio": 1.7},
        {"t": 0.4, "compat_over_flowstar_ratio": 2.1},
    ]

    crossings = audit.first_width_crossings(rows, (1.1, 1.5, 2.0))

    assert crossings[1.1]["t"] == 0.2
    assert crossings[1.5]["t"] == 0.3
    assert crossings[2.0]["t"] == 0.4


def test_fixture_distinguishes_tube_close_vs_last_segment_not_close():
    summary = [
        {
            "mode": audit.COMPAT_MODE,
            "tube_width_ratio_vs_flowstar": "1.011",
            "last_segment_width_ratio_vs_flowstar": "2.608",
        }
    ]

    assert audit.tube_close_but_last_not(summary)


def test_h5_divergence_audit_has_no_h10_outputs():
    assert "h10" not in str(audit.DEFAULT_OUT_DIR)
    assert not (ROOT / "outputs" / "flowstar_raw_remainder_compat_h10_divergence").exists()
    assert "refusing to write h10 outputs" in SCRIPT.read_text(encoding="utf-8")


def test_checked_in_h5_csv_and_md_artifacts_have_physical_lines():
    for name in H5_CSVS:
        path = H5_DIR / name
        assert path.exists(), path
        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) > 5, path
        assert all("\n" not in line for line in lines), path

    report = H5_DIR / "h5_report.md"
    text = report.read_text(encoding="utf-8")
    assert len(text.splitlines()) > 10
    assert "\\n" not in text
