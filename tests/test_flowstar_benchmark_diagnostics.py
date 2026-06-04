from __future__ import annotations

import csv
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

from torch_tm_flowpipe import Interval

ROOT = Path(__file__).resolve().parents[1]


def _load_diagnostics_module():
    spec = importlib.util.spec_from_file_location(
        "flowstar_benchmark_failure_diagnostics",
        ROOT / "experiments" / "flowstar_benchmark_failure_diagnostics.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _small_params():
    return {
        "initial_set": {"x": [1.1, 1.4], "y": [2.35, 2.45]},
        "taylor_order": 4,
        "time_horizon": 0.002,
    }


def test_diagnostic_script_produces_required_csv_and_report_on_short_run(tmp_path):
    diag = _load_diagnostics_module()
    refs = [{"segment_index": 0, "t_lo": 0.0, "t_hi": 0.002}]

    run_rows, segment_rows, attempt_rows = diag.run_diagnostics(
        tmp_path,
        _small_params(),
        refs,
        orders=[4],
        substep_factors=[1],
        max_wall_s_per_run=20,
        max_horizon=0.002,
    )

    for name in [
        "diagnostic_runs_summary.csv",
        "diagnostic_segments.csv",
        "diagnostic_validation_attempts.csv",
        "diagnostic_report.md",
    ]:
        assert (tmp_path / name).exists(), name
    assert run_rows
    assert segment_rows
    assert attempt_rows
    assert "not a new reachability algorithm" in (tmp_path / "diagnostic_report.md").read_text(encoding="utf-8")


def test_dependency_preserving_diagnostic_path_carries_final_tm_between_segments(monkeypatch, tmp_path):
    diag = _load_diagnostics_module()

    class DummyTM:
        def range_box(self):
            return [Interval(0.0, 1.0), Interval(2.0, 3.0)]

    first_final = DummyTM()
    second_final = DummyTM()
    received = []

    def fake_step(_ode, current_tm, h, order, *, diagnostics=None, diagnostics_context=None, **_kwargs):
        received.append(current_tm)
        if diagnostics is not None:
            context = dict(diagnostics_context or {})
            diagnostics.append(
                {
                    **context,
                    "attempt_index": 1,
                    "h": h,
                    "order": order,
                    "finite_residual": True,
                    "validation_status": "validated",
                    "validation_message": "",
                }
            )
        final_tm = first_final if len(received) == 1 else second_final
        return SimpleNamespace(
            tm=final_tm,
            final_tm=final_tm,
            status="validated",
            validation_attempts=1,
            message="",
        )

    monkeypatch.setattr(diag, "flowpipe_step_from_tm", fake_step)
    refs = [
        {"segment_index": 0, "t_lo": 0.0, "t_hi": 0.001},
        {"segment_index": 1, "t_lo": 0.001, "t_hi": 0.002},
    ]

    summary, segments, attempts = diag.run_torch_diagnostic(
        _small_params(),
        refs,
        mode="dependency_preserving",
        order=4,
        substep_factor=1,
        max_wall_s_per_run=20,
        max_horizon=0.002,
    )

    assert summary["status"] == "completed"
    assert len(segments) == 2
    assert attempts[0]["mode"] == "dependency_preserving"
    assert received[1] is first_final


def test_diagnostic_csv_writer_blanks_nonfinite_values(tmp_path):
    diag = _load_diagnostics_module()
    path = tmp_path / "nonfinite.csv"
    diag._write_csv(path, ["value"], [{"value": math.inf}, {"value": math.nan}, {"value": 1.25}])

    rows = _rows(path)
    assert rows[0]["value"] == ""
    assert rows[1]["value"] == ""
    assert float(rows[2]["value"]) == 1.25
