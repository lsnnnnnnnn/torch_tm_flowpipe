from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "comparisons" / "flowstar"))


def test_tm_order_audit_actual_degree_and_tau_drop():
    from tm_order_audit import audit_row, load_config

    cfg = load_config(ROOT / "comparisons" / "flowstar" / "configs" / "van_der_pol.yaml")
    row = audit_row(cfg, h=0.0025, steps=1, order=4, mode="dependency_preserving")
    assert row["order_semantics"] == "total_degree_cutoff"
    assert row["dependency_scope"] == "original_initial_variables"
    assert row["actual_degree_reference"] == "degree_wrt_original_initial_vars"
    assert row["max_final_degree"] <= 4
    assert row["term_count_total"] > 0
    assert row["segment_tau_active_after_drop"] is False
    assert "flowpipe_width_sum" in row
    assert "runtime_s" in row


def test_van_der_pol_diagnostic_decomposition_columns_exist():
    from diagnose_van_der_pol import FIELDS, diagnostic_rows

    rows = diagnostic_rows(h=0.0025, steps=1, order=3, grid=3, substeps=2)
    assert {"range_only", "dependency_preserving"} == {r["mode"] for r in rows}
    for name in [
        "system",
        "requested_order",
        "final_width_sum",
        "poly_range_width_sum",
        "remainder_width_sum",
        "max_final_degree",
        "term_count_by_dim",
        "width_over_sampled_ratio",
        "remainder_width_frac",
        "poly_range_width_frac",
        "quality_label",
    ]:
        assert name in FIELDS
        assert name in rows[0]


def test_flowstar_cli_grid_supports_order_h_steps_filtering():
    from compare_against_torch_tm import _case_grid, load_config

    cfg = load_config(ROOT / "comparisons" / "flowstar" / "configs" / "van_der_pol.yaml")
    cases = _case_grid(cfg, all_cases=False, h_values=[0.01], steps_values=[1, 2], orders=[2, 3])
    assert cases == [(0.01, 1, 2), (0.01, 1, 3), (0.01, 2, 2), (0.01, 2, 3)]


def test_flowstar_failed_or_unparsed_rows_are_written(tmp_path):
    from compare_against_torch_tm import write_csv

    path = tmp_path / "rows.csv"
    rows = [
        {
            "system": "van_der_pol",
            "tool": "flowstar",
            "mode": "fixed",
            "h": 0.01,
            "steps": 10,
            "order": 2,
            "status": "compile_failed",
            "failure_reason": "compiler said no",
        },
        {
            "system": "van_der_pol",
            "tool": "flowstar",
            "mode": "fixed",
            "h": 0.01,
            "steps": 10,
            "order": 3,
            "status": "unparsed",
            "failure_reason": "no variable ranges found",
        },
    ]
    write_csv(path, rows)
    text = path.read_text(encoding="utf-8")
    assert "compile_failed" in text
    assert "unparsed" in text
    assert "compiler said no" in text


def test_plot_order_results_creates_semantic_flowstar_ratio_filenames(tmp_path):
    from plot_order_results import main
    import sys

    diag = tmp_path / "diag.csv"
    flow = tmp_path / "flow.csv"
    diag.write_text(
        "system,mode,h,steps,requested_order,status,endpoint_width_sum,last_segment_width_sum,tube_width_sum,runtime_s,remainder_width_frac,poly_range_width_sum,remainder_width_sum\n"
        "van_der_pol,dependency_preserving,0.01,10,4,validated,5,4,9,0.1,0.2,4,1\n"
        "van_der_pol,range_only,0.01,10,4,validated,3,2,7,0.08,0.1,2,1\n",
        encoding="utf-8",
    )
    flow.write_text(
        "system,tool,mode,h,steps,order,setting_label,status,endpoint_width_sum,last_segment_width_sum,tube_width_sum,runtime_s,flowstar_wall_total_s\n"
        "van_der_pol,torch_tm_flowpipe,dependency_preserving,0.01,10,4,,validated,5,4,9,0.1,\n"
        "van_der_pol,flowstar,fixed,0.01,10,4,loose,completed,,2,3,0.04,0.2\n",
        encoding="utf-8",
    )
    old_argv = sys.argv
    try:
        sys.argv = [
            "plot_order_results.py",
            "--torch-diagnostics", str(diag),
            "--flowstar-csv", str(flow),
            "--out-dir", str(tmp_path),
        ]
        main()
    finally:
        sys.argv = old_argv
    assert (tmp_path / "torch_over_flowstar_last_segment_width_ratio_by_order.png").exists()
    assert (tmp_path / "torch_over_flowstar_tube_width_ratio_by_order.png").exists()
    assert not (tmp_path / "torch_over_flowstar_width_ratio_by_order.png").exists()


def test_report_documents_required_flowstar_semantics():
    text = (ROOT / "docs" / "order_and_vdp_flowstar_report.md").read_text(encoding="utf-8")
    for phrase in [
        "endpoint boxes were not available",
        "last-segment",
        "tube",
        "setting-dependent",
        "range_only degree",
    ]:
        assert phrase in text


def test_old_ambiguous_flowstar_ratio_function_is_not_present():
    text = (ROOT / "experiments" / "plot_order_results.py").read_text(encoding="utf-8")
    assert 'r.get("final_width_sum") or r.get("last_segment_width_sum")' not in text
    assert 'r.get("last_segment_width_sum") or r.get("final_width_sum")' not in text

