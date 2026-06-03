from __future__ import annotations

import csv
import json
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "flowstar_benchmark_parity"


def _csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames or [], list(reader)


def _png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as f:
        header = f.read(24)
    assert header.startswith(b"\x89PNG\r\n\x1a\n"), path
    return struct.unpack(">II", header[16:24])


def test_params_json_exists_and_contains_original_flowstar_parameters():
    path = OUT / "original_flowstar_params.json"
    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["ode"]["x"] == "y"
    assert data["ode"]["y"] == "(1 - x^2) * y - x"
    assert data["initial_set"]["x"] == [1.1, 1.4]
    assert data["initial_set"]["y"] == [2.35, 2.45]
    assert data["time_horizon"] == 10.0
    assert data["step_policy"] == "adaptive"
    assert data["step_min"] == 0.002
    assert data["step_max"] == 0.1
    assert data["taylor_order"] == 4


def test_original_flowstar_pngs_exist():
    for name in ["original_vanderpol_t_x.png", "original_vanderpol_t_y.png"]:
        path = OUT / "original_flowstar" / name
        assert path.exists(), path
        width, height = _png_size(path)
        assert width > 100 and height > 100


def test_generated_cpp_and_runner_use_flowstar_toolbox_api():
    cpp = OUT / "generated_flowstar" / "generated_vanderpol_original_defaults.cpp"
    assert cpp.exists()
    text = cpp.read_text(encoding="utf-8")
    assert '#include "Continuous.h"' in text
    assert "ode.reach(" in text
    assert "setAdaptiveStepsize" in text
    assert "setFixedStepsize" in text
    assert "-lflowstar" in text
    runner = (ROOT / "comparisons" / "flowstar" / "run_flowstar.py").read_text(encoding="utf-8")
    assert '"-lflowstar"' in runner


def test_segments_csv_columns_exist_and_widths_are_nonnegative():
    for path in [
        OUT / "original_flowstar" / "original_flowstar_segments.csv",
        OUT / "generated_flowstar" / "generated_flowstar_segments.csv",
        OUT / "torch" / "torch_segments.csv",
    ]:
        header, rows = _csv_rows(path)
        assert set(header) >= {
            "case_id",
            "segment_index",
            "t_lo",
            "t_hi",
            "x_lo",
            "x_hi",
            "y_lo",
            "y_hi",
            "width_x",
            "width_y",
            "width_sum",
            "box_source",
        }
        assert rows
        for row in rows:
            assert float(row["width_x"]) >= 0.0
            assert float(row["width_y"]) >= 0.0
            assert float(row["width_sum"]) >= 0.0


def test_parity_summary_has_runtime_and_bound_fields():
    header, rows = _csv_rows(OUT / "parity_summary.csv")
    assert set(header) >= {
        "tool",
        "status",
        "original_flowstar_wall_run_s",
        "generated_flowstar_internal_reach_s",
        "generated_flowstar_compile_wall_s",
        "generated_flowstar_run_wall_s",
        "torch_runtime_s",
        "last_segment_width_x",
        "last_segment_width_y",
        "last_segment_width_sum",
        "tube_width_x",
        "tube_width_y",
        "tube_width_sum",
        "endpoint_box_available",
        "box_source",
        "notes",
    }
    by_tool = {row["tool"]: row for row in rows}
    assert by_tool["original_flowstar"]["original_flowstar_wall_run_s"]
    assert by_tool["generated_flowstar"]["generated_flowstar_internal_reach_s"]
    assert by_tool["generated_flowstar"]["generated_flowstar_compile_wall_s"]
    assert by_tool["generated_flowstar"]["generated_flowstar_run_wall_s"]
    assert by_tool["torch_tm"]["torch_runtime_s"]


def test_flowstar_endpoint_box_is_false_without_true_endpoint_source():
    _header, rows = _csv_rows(OUT / "parity_summary.csv")
    for row in rows:
        if row["tool"] in {"original_flowstar", "generated_flowstar"}:
            assert row["endpoint_box_available"] == "false"
            assert "endpoint" not in row["box_source"]


def test_overlay_pngs_exist_nonempty_and_have_dimensions():
    required = [
        OUT / "overlay_original_flowstar_vs_torch_t_x.png",
        OUT / "overlay_original_flowstar_vs_torch_t_y.png",
        OUT / "overlay_original_flowstar_vs_torch_phase_xy.png",
        OUT / "overlay_generated_flowstar_vs_torch_t_x.png",
        OUT / "overlay_generated_flowstar_vs_torch_t_y.png",
        OUT / "torch" / "torch_vanderpol_phase_xy.png",
    ]
    for path in required:
        assert path.exists(), path
        assert path.stat().st_size > 0
        width, height = _png_size(path)
        assert width > 100 and height > 100


def test_docs_are_utf8_lf_without_cr():
    for path in [
        ROOT / "docs" / "flowstar_benchmark_parity.md",
        OUT / "README.md",
        OUT / "parity_report.md",
        OUT / "original_flowstar_params.md",
    ]:
        data = path.read_bytes()
        assert b"\r" not in data
        data.decode("utf-8")


def test_scope_guard_terms_only_appear_as_negative_scope_text():
    terms = ["CROWN", "auto_LiRPA", "Jacobian", "sin/cos", "hybrid", "Flow* Python binding", "NN controller"]
    for path in [ROOT / "docs" / "flowstar_benchmark_parity.md", OUT / "README.md", OUT / "parity_report.md"]:
        for line in path.read_text(encoding="utf-8").splitlines():
            if any(term in line for term in terms):
                lowered = line.lower()
                assert "no " in lowered or "not " in lowered
