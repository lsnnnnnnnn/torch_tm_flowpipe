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
