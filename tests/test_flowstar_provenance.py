from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _tracked_files() -> set[str]:
    return set(subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines())


def test_flowstar_provenance_outputs_exist_and_are_tracked():
    tracked = _tracked_files()
    for rel in [
        "outputs/flowstar_provenance_manifest.md",
        "outputs/flowstar_provenance_manifest.json",
        "outputs/flowstar_parameter_equivalence_table.md",
    ]:
        assert (ROOT / rel).exists()
        assert rel in tracked


def test_flowstar_provenance_manifest_records_backend_and_semantics():
    data = json.loads((ROOT / "outputs" / "flowstar_provenance_manifest.json").read_text(encoding="utf-8"))
    md = (ROOT / "outputs" / "flowstar_provenance_manifest.md").read_text(encoding="utf-8")

    source = data["torch_tm_flowpipe"]
    assert "head_sha" not in source
    assert "source_head_at_generation" not in source
    assert "remote_origin_main" not in source
    assert "status_short_before_manifest_generation" not in source
    assert re.fullmatch(r"[0-9a-f]{40}", source["source_tree_commit_used_for_generation"])
    assert re.fullmatch(r"[0-9a-f]{40}\trefs/heads/main", source["remote_origin_main_at_generation"])
    assert source["generation_worktree_status"] == "clean"
    assert source["artifact_bundle_commit_note"] == (
        "This manifest is generated from the clean source tree recorded in "
        "source_tree_commit_used_for_generation. The artifact bundle commit "
        "containing this refreshed manifest may be later."
    )
    assert "`source_tree_commit_used_for_generation`" in md
    assert "`remote_origin_main_at_generation`" in md
    assert "`artifact_bundle_commit_note`" in md
    assert "`generation_worktree_status`: `clean`" in md
    assert "`head_sha`" not in md.split("## Flow* Backend", 1)[0]

    backend = data["flowstar_backend"]
    assert backend["FLOWSTAR_ROOT"] == "/srv/local/shengenli/flowstar"
    assert backend["backend"] == "toolbox_cpp"
    assert re.fullmatch(r"[0-9a-f]{40}", backend["head_sha"])
    assert backend["libflowstar_a"]["path"].endswith("flowstar-toolbox/libflowstar.a")
    assert re.fullmatch(r"[0-9a-f]{64}", backend["libflowstar_a"]["sha256"])

    audit = data["generated_cpp_audit"]
    assert audit["generated_cpp_case_count_from_csv"] == 252
    assert audit["all_generated_cases_include_Continuous_h"] is True
    assert audit["all_generated_cases_call_ode_reach"] is True
    assert audit["runner_links_libflowstar_a"] is True

    semantics = data["box_and_ratio_semantics"]
    assert semantics["flowstar_endpoint_box_available_for_gnuplot_rows"] is False
    assert semantics["verified_all_completed_gnuplot_rows_endpoint_box_available_false"] is True
    assert semantics["current_torch_vs_flowstar_ratio_types"] == ["last_segment", "tube"]
    assert semantics["endpoint_ratio_allowed"] is False

    assert "FLOWSTAR_ROOT" in md
    assert "libflowstar.a sha256" in md
    assert "toolbox_cpp" in md
    assert "Continuous.h" in md
    assert "ode.reach" in md
    assert "endpoint_box_available=false" in md
    assert "`last_segment` and `tube` only" in md
    assert "FLOWSTAR_RUNTIME_S" in md


def test_flowstar_parameter_equivalence_table_documents_fixed_order_scope():
    text = (ROOT / "outputs" / "flowstar_parameter_equivalence_table.md").read_text(encoding="utf-8")
    assert "plant-only fixed-step/fixed-order baseline" in text
    assert "not `Flow*_adaptive`" in text
    assert "not a full CROWN-Reach NNCS pipeline reproduction" in text
    assert "setting.setFixedStepsize(h, order)" in text
    assert "range_only; dependency_preserving" in text
    assert "last_segment; tube; endpoint unavailable" in text
    assert "FLOWSTAR_RUNTIME_S/internal reach time" in text


def test_reports_keep_endpoint_adaptive_and_crown_claim_boundaries():
    report_paths = [
        ROOT / "docs" / "flowstar_comparison.md",
        ROOT / "docs" / "order_and_vdp_flowstar_report.md",
        ROOT / "outputs" / "order_and_vdp_flowstar_report.md",
        ROOT / "outputs" / "flowstar_provenance_manifest.md",
        ROOT / "outputs" / "flowstar_parameter_equivalence_table.md",
    ]
    for path in report_paths:
        text = path.read_text(encoding="utf-8")
        lowered = text.lower()
        assert "endpoint ratio" not in lowered or any(
            phrase in lowered
            for phrase in [
                "not compute or claim endpoint ratios",
                "no endpoint ratio",
                "endpoint ratios are not allowed",
                "does not report endpoint ratios",
                "endpoint ratios are allowed only when both compared rows have `endpoint_box_available=true`",
            ]
        )
        for line in text.splitlines():
            low = line.lower()
            if "full crown-reach" in low:
                assert "not" in low or "does not" in low or "no " in low
            if "flow* adaptive" in low or "flow*_adaptive" in low:
                assert any(marker in low for marker in ["not", "future", "separate", "disabled", "outside", "does not represent"])
        assert "reproduces the full CROWN-Reach" not in text
        assert "Flow* adaptive result" not in text
