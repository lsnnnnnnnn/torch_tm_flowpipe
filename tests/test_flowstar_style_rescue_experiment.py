import csv
import subprocess
import sys
from pathlib import Path


def test_flowstar_style_rescue_experiment_smoke(tmp_path):
    script = Path(__file__).resolve().parents[1] / "experiments" / "flowstar_style_rescue_vanderpol.py"
    out_dir = tmp_path / "rescue"

    subprocess.run(
        [
            sys.executable,
            str(script),
            "--out-dir",
            str(out_dir),
            "--max-horizon",
            "0.02",
            "--wall-cap-s",
            "60",
        ],
        check=True,
    )

    required = [
        "rescue_summary.csv",
        "rescue_segments.csv",
        "rescue_validation_attempts.csv",
        "rescue_report.md",
        "rescue_t_x.png",
        "rescue_t_y.png",
        "rescue_phase_xy.png",
    ]
    for name in required:
        assert (out_dir / name).exists()

    with (out_dir / "rescue_summary.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert {row["run_id"] for row in rows} >= {
        "baseline_range_only_o6_s4",
        "baseline_dependency_preserving_o4_s1",
        "flowstar_style_o4_target",
        "flowstar_style_o6_target",
        "flowstar_style_o4_target_cutoff",
        "flowstar_style_o6_target_cutoff",
    }
    assert any(row["mode"] == "flowstar_style" for row in rows)
