from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_stage2_module():
    spec = importlib.util.spec_from_file_location(
        "flowstar_benchmark_stage2_diagnostics",
        ROOT / "experiments" / "flowstar_benchmark_stage2_diagnostics.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _small_params():
    return {
        "initial_set": {"x": [1.1, 1.4], "y": [2.35, 2.45]},
        "taylor_order": 4,
        "time_horizon": 0.002,
    }


def test_stage2_dependency_window_short_run_records_rhs_breakdown():
    stage2 = _load_stage2_module()
    refs = [{"segment_index": 0, "t_lo": 0.0, "t_hi": 0.002}]

    summary, segments, attempts, breakdowns = stage2.run_dependency_window_diagnostic(
        _small_params(),
        refs,
        order=4,
        substep_factor=1,
        dependency_window=2,
        max_wall_s_per_run=20,
        max_horizon=0.002,
    )

    assert summary["status"] == "max_horizon_reached"
    assert segments and attempts
    assert {row["expression"] for row in breakdowns} >= {"x_sq", "x_sq_y", "rhs_y"}
    assert all("total_remainder_width" in row for row in breakdowns)
