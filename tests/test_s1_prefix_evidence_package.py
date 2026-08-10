from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from torch_tm_flowpipe import load_terminal_checkpoint


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs" / "s1_prefix_integrated_complete_o4_20260810" / "20260810T095423Z"
PACKAGER = ROOT / "experiments" / "package_s1_prefix_result.py"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_s1_prefix_package_paths_claims_and_stop_rows_resolve():
    manifest = json.loads((RUN / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["primary_outcome"] == "S1_PREFIX_REJECTS_BEFORE_TERMINAL"
    for relative in [*manifest["required_files"], *manifest["required_figures"]]:
        assert (RUN / relative).is_file(), relative
    for relative in manifest["required_directories"]:
        assert (RUN / relative).is_dir(), relative
    for figure in manifest["required_figures"]:
        assert (RUN / figure).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")

    claims = list(csv.DictReader((RUN / "claim_registry.csv").open(newline="", encoding="utf-8")))
    assert len(claims) >= 9
    for row in claims:
        assert (RUN / row["evidence_path"]).exists(), row["evidence_path"]
    for name in ("terminal_ab.json", "terminal_gate.json"):
        assert json.loads((RUN / name).read_text(encoding="utf-8"))["status"] == "not_run_after_stop"
    for name in ("horizon_ladder.csv", "common_time_tightness.csv", "second_system.csv"):
        assert "not_run_after_stop" in (RUN / name).read_text(encoding="utf-8")


def test_s1_prefix_package_gates_and_checkpoint_v2_are_exact():
    summary = json.loads(
        (
            RUN
            / "04_frozen_schedule_prefix"
            / "L2_structured_k16_final_checkpointed"
            / "summary.json"
        ).read_text(encoding="utf-8")
    )
    assert summary["accepted_boundaries"] == 164
    assert summary["final_common_prefix_boundary"] == 164
    assert summary["first_full_k16_boundary"] == 16
    assert summary["first_eviction_boundary"] == 17
    assert summary["outcome"] == "S1_PREFIX_REJECTS_BEFORE_TERMINAL"

    rows = list(csv.DictReader((RUN / "prefix_conservation.csv").open(newline="", encoding="utf-8")))
    committed = [row for row in rows if row["lane"] == "L2" and row["committed_to_frozen_prefix"] == "True"]
    assert len(committed) == 164
    for row in committed:
        for field in (
            "conservation_mask",
            "source_decomposition_mask",
            "no_double_count_mask",
            "finite_mask",
            "endpoint_publication_mask",
            "tube_publication_mask",
            "accepted_mask",
        ):
            assert row[field] == "True", (row["attempt_index"], field)

    checkpoint = RUN / "05_prefix_checkpoints" / "boundary_164_v2"
    loaded = load_terminal_checkpoint(checkpoint, expected_order=4, expected_dtype="float64")
    assert loaded.manifest["schema"] == "torch_tm_flowpipe_terminal_checkpoint_v2"
    assert loaded.manifest["full_checkpoint_sha256"] == "9162f267fcdcf44ca7bb9acfa73975eb8f4f4b80c03ca217aac2f07450cd585b"
    assert loaded.normal_state.structured_remainder_state.accepted_boundary_index == 164
    roundtrip = json.loads((RUN / "05_prefix_checkpoints" / "checkpoint_roundtrip.json").read_text(encoding="utf-8"))
    assert roundtrip["byte_stable"] is True


def test_s1_tables_and_figures_rebuild_deterministically(tmp_path):
    copied = tmp_path / "run"
    shutil.copytree(RUN, copied)
    tracked_outputs = [
        "prefix_conservation.csv",
        "prefix_source_events.jsonl",
        "prefix_state_hashes.csv",
        "capacity_attribution.csv",
        "horizon_ladder.csv",
        "common_time_tightness.csv",
        "second_system.csv",
        "claim_registry.csv",
        "failure_attribution.json",
        "terminal_ab.json",
        "terminal_gate.json",
        *[f"figures/{name}" for name in (
            "ordinary_vs_structured_width_over_prefix.png",
            "active_columns_and_evictions_over_prefix.png",
            "nonlinear_residual_over_prefix.png",
            "terminal_same_pre_state_margins.png",
            "common_time_endpoint_widths.png",
            "common_time_tube_widths.png",
            "validated_horizon_ladder.png",
        )],
    ]
    before = {name: _sha(copied / name) for name in tracked_outputs}
    subprocess.run(
        ["python", str(PACKAGER), "--run-root", str(copied)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    after = {name: _sha(copied / name) for name in tracked_outputs}
    assert before == after
