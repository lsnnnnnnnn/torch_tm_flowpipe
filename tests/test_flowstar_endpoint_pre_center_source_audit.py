from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments" / "flowstar_endpoint_pre_center_source_audit.py"
spec = importlib.util.spec_from_file_location("flowstar_endpoint_pre_center_source_audit", SCRIPT)
assert spec is not None and spec.loader is not None
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def _row(source_object: str = "same_object", *, y_hi: str = "2.0", include_components: bool = True, **kwargs: object) -> dict[str, object]:
    row: dict[str, object] = {
        "t_before": "0",
        "h_try": "0.025",
        "status": "accepted",
        "endpoint_before_center_source_object": source_object,
        "endpoint_before_center_domain_semantics": "same_domain",
        "endpoint_before_center_includes_target_remainder": "false",
        "endpoint_before_center_includes_ordinary_remainder": "false",
        "endpoint_before_center_includes_symbolic_output_width": "false",
        "endpoint_before_center_includes_cutoff_poly_diff": "true",
        "endpoint_before_center_range_eval_method": "same_eval",
        "endpoint_box_before_center_x_lo": "1.0",
        "endpoint_box_before_center_x_hi": "2.0",
        "endpoint_box_before_center_y_lo": "1.5",
        "endpoint_box_before_center_y_hi": y_hi,
        "endpoint_before_center_polynomial_order": "4",
    }
    if include_components:
        row.update(
            {
                "endpoint_before_center_dropped_terms_width_sum": "1e-12",
                "endpoint_before_center_remainder_width_sum": "2e-4",
            }
        )
    row.update(kwargs)
    return row


def test_mismatched_source_object_labels_mark_comparison_noncausal():
    rows, summary = audit.build_endpoint_source_ledger(
        [_row("flowstar_tmvTmp")],
        [_row("torch_final_tm")],
        [_row("torch_final_tm")],
        t=0.0,
        h=0.025,
    )

    assert summary["semantic_comparison_valid"] == "false"
    assert summary["first_endpoint_divergence"] == "source_object_semantics"
    assert rows[1]["semantic_comparison_valid"] is False


def test_same_source_object_different_y_hi_reports_y_hi_divergence():
    rows, summary = audit.build_endpoint_source_ledger(
        [_row("same_object", y_hi="2.4")],
        [_row("same_object", y_hi="2.1")],
        [_row("same_object", y_hi="2.1")],
        t=0.0,
        h=0.025,
    )

    assert summary["semantic_comparison_valid"] == "true"
    assert summary["first_endpoint_divergence"] == "y_hi"
    assert rows[1]["y_hi_delta_vs_flowstar"] == 0.2999999999999998


def test_missing_component_fields_are_unknown_not_zero():
    row = audit._main_endpoint_row("flowstar", _row(include_components=False))

    assert row["dropped_terms_width"] == "unknown"
    assert row["remainder_width"] == "unknown"
    assert row["dropped_terms_width"] != 0
    assert row["remainder_width"] != 0


def test_report_says_diagnostic_only_and_no_solver_change(tmp_path):
    rows, summary = audit.build_endpoint_source_ledger(
        [_row("flowstar_tmvTmp")],
        [_row("torch_final_tm")],
        [_row("torch_final_tm")],
        t=0.0,
        h=0.025,
    )

    report = audit._report(tmp_path, rows, summary, t=0.0, h=0.025)

    assert "diagnostic-only" in report
    assert "no solver change" in report
    assert "noncausal" in report
