from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "comparisons" / "flowstar"))

from export_flowstar_model import load_config, render_legacy_model, render_model, render_toolbox_cpp
from parse_flowstar_output import parse_files, parse_text, widths
from run_flowstar import find_flowstar_root


def test_toolbox_export_matches_current_flowstar_cpp_api():
    cfg = load_config(ROOT / "comparisons" / "flowstar" / "configs" / "scalar_quadratic.yaml")
    text = render_toolbox_cpp(cfg, h=0.01, steps=5, order=4, output_prefix="case")
    assert '#include "Continuous.h"' in text
    assert "Variables vars" in text
    assert 'int x_id = vars.declareVar("x");' in text
    assert 'int t_id = vars.declareVar("t");' in text
    assert 'ODE<Real> ode({"1 + x^2", "1"}, vars);' in text
    assert "setting.setFixedStepsize(0.01, 4);" in text
    assert "setting.setCutoffThreshold(1.0000000000000001e-15);" in text
    assert "ode.reach(result, initialSet, 0.050000000000000003, setting, safeSet);" in text
    assert "result.transformToTaylorModels(setting);" in text
    assert "plot_2D_interval_GNUPLOT" in text


def test_render_model_defaults_to_toolbox_cpp_not_legacy_model():
    cfg = load_config(ROOT / "comparisons" / "flowstar" / "configs" / "scalar_quadratic.yaml")
    text = render_model(cfg, h=0.01, steps=5, order=4, output_name="out")
    assert "continuous reachability" not in text
    assert "Computational_Setting setting(vars);" in text


def test_legacy_model_export_still_available():
    cfg = load_config(ROOT / "comparisons" / "flowstar" / "configs" / "scalar_quadratic.yaml")
    text = render_legacy_model(cfg, h=0.01, steps=5, order=4, output_name="out.plt")
    assert "continuous reachability" in text
    assert "fixed steps 0.01" in text
    assert "time 0.050000000000000003" in text or "time 0.05" in text
    assert "fixed orders 4" in text
    assert "x' = 1 + x^2" in text


def test_flowstar_parser_hulls_boxes():
    parsed = parse_text("""
    x in [0, 1]
    y in [-1, 1]
    x in [0.5, 2]
    y in [-2, 0]
    """, variables=["x", "y"])
    assert parsed.status == "parsed"
    assert parsed.final_box == {"x": (0.5, 2.0), "y": (-2.0, 0.0)}
    assert parsed.flowpipe_box == {"x": (0.0, 2.0), "y": (-2.0, 1.0)}
    assert widths(parsed.final_box, ["x", "y"]) == [1.5, 2.0]


def test_flowstar_parser_reads_gnuplot_numeric_pairs(tmp_path):
    plot = tmp_path / "case_t_x.plt"
    plot.write_text("""
    0.00 0.0
    0.01 0.1

    0.01 0.2
    0.02 0.4
    """, encoding="utf-8")
    parsed = parse_files([plot], variables=["x"], numeric_plot_vars=["x"])
    assert parsed.status == "parsed"
    assert parsed.endpoint_box == {}
    assert parsed.last_segment_box == {"x": (0.2, 0.4)}
    assert parsed.tube_box == {"x": (0.0, 0.4)}
    assert parsed.flowpipe_box == {"x": (0.0, 0.4)}
    assert parsed.final_box == {"x": (0.2, 0.4)}


def test_toolbox_export_accepts_remainder_and_cutoff_overrides():
    cfg = load_config(ROOT / "comparisons" / "flowstar" / "configs" / "scalar_quadratic.yaml")
    text = render_toolbox_cpp(cfg, h=0.01, steps=5, order=4, output_prefix="case", remainder_radius=1e-6, cutoff=1e-12)
    assert "setting.setCutoffThreshold(9.9999999999999998e-13);" in text
    assert "Interval(-9.9999999999999995e-07, 9.9999999999999995e-07)" in text


def test_flowstar_runtime_parser_and_new_width_columns(tmp_path):
    from compare_against_torch_tm import _flowstar_internal_runtime, write_csv
    from run_flowstar import FlowstarRunResult

    stdout = tmp_path / "case.stdout.txt"
    stdout.write_text("FLOWSTAR_RUNTIME_S 0.125\n", encoding="utf-8")
    run = FlowstarRunResult(status="completed", runtime_s=1.0, compile_s=0.4, run_s=0.6, stdout_path=stdout)
    assert _flowstar_internal_runtime(run) == 0.125

    csv_path = tmp_path / "rows.csv"
    write_csv(csv_path, [{
        "system": "van_der_pol",
        "tool": "flowstar",
        "mode": "fixed",
        "setting_label": "loose",
        "status": "completed",
        "last_segment_width_sum": 1.0,
        "tube_width_sum": 2.0,
        "box_source": "flowstar_gnuplot_last_segment_and_tube",
    }])
    text = csv_path.read_text(encoding="utf-8")
    assert "last_segment_width_sum" in text
    assert "flowstar_internal_reach_s" in text
    assert "loose" in text


def test_find_flowstar_root_accepts_repo_or_toolbox_path(tmp_path):
    root = tmp_path / "flowstar"
    toolbox = root / "flowstar-toolbox"
    toolbox.mkdir(parents=True)
    (toolbox / "Continuous.h").write_text("", encoding="utf-8")
    assert find_flowstar_root(str(root)) == root.resolve()
    assert find_flowstar_root(str(toolbox)) == root.resolve()


def test_summary_reports_dependency_preserving_usefulness(tmp_path):
    from summarize_comparison import generate_summary

    csv_path = tmp_path / "comparison.csv"
    csv_path.write_text(
        "system,tool,mode,h,steps,order,status,final_width_sum,final_width_max,flowpipe_width_sum,flowpipe_width_max,runtime_s,num_segments,validation_attempts,term_count,remainder_radius,containment_failures\n"
        "toy,torch_tm_flowpipe,range_only,0.1,2,4,validated,2.0,2.0,2.0,2.0,0.1,2,2,3,0.0,0\n"
        "toy,torch_tm_flowpipe,dependency_preserving,0.1,2,4,validated,1.0,1.0,1.0,1.0,0.2,2,2,4,0.0,0\n"
        "toy,flowstar,fixed,0.1,2,4,completed,0.5,0.5,0.5,0.5,0.05,2,,,,0\n",
        encoding="utf-8",
    )
    md = generate_summary(csv_path)
    assert "mean dep/range width" in md
    assert "0.5" in md
    assert "Torch over Flow*" in md


def test_compare_skip_flowstar_exports_cpp(tmp_path):
    from compare_against_torch_tm import flowstar_row

    cfg_path = ROOT / "comparisons" / "flowstar" / "configs" / "scalar_quadratic.yaml"
    cfg = load_config(cfg_path)
    row = flowstar_row(
        cfg,
        cfg_path,
        h=0.01,
        steps=1,
        order=3,
        model_dir=tmp_path,
        flowstar_target="toolbox_cpp",
        flowstar_bin=None,
        flowstar_root=None,
        skip_flowstar=True,
        timeout_s=None,
        build_flowstar_lib=False,
    )
    assert row["status"] == "skipped"
    generated = tmp_path / "scalar_quadratic_h0.01_s1_o3.cpp"
    assert generated.exists()
    assert "ODE<Real> ode" in generated.read_text(encoding="utf-8")


def test_toolbox_export_preserves_decimal_case_names(tmp_path):
    from export_flowstar_model import export_model

    cfg_path = ROOT / "comparisons" / "flowstar" / "configs" / "van_der_pol.yaml"
    out = tmp_path / "van_der_pol_h0.01_s10_o4.cpp"
    export_model(cfg_path, out, h=0.01, steps=10, order=4, plot_output_name="van_der_pol_h0.01_s10_o4", target="toolbox_cpp")
    text = out.read_text(encoding="utf-8")
    assert "van_der_pol_h0_01_s10_o4_t_x" in text
    assert "van_der_pol_h0_t_x" not in text
