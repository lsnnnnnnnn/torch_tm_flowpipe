from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/huan_repro_artifact_audit.py"
SPEC = importlib.util.spec_from_file_location("huan_repro_artifact_audit", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def test_discovery_is_case_insensitive_and_bounded(tmp_path: Path) -> None:
    candidate = tmp_path / "CROWN-Reach-GPU"
    candidate.mkdir()
    doc = tmp_path / "nested" / "flowstar_gpu" / "docs" / "REPRODUCE.md"
    doc.parent.mkdir(parents=True)
    doc.write_text("test\n", encoding="utf-8")
    too_deep = tmp_path / "a" / "b" / "c" / "d" / "OPTIMIZATION.md"
    too_deep.parent.mkdir(parents=True)
    too_deep.write_text("test\n", encoding="utf-8")

    hits, errors = AUDIT.discover_candidates(tmp_path, max_depth=4)

    assert errors == []
    assert candidate in hits
    assert doc in hits
    assert too_deep not in hits


def test_engine_root_requires_source_package_and_build_file(tmp_path: Path) -> None:
    incomplete = tmp_path / "incomplete"
    (incomplete / "src" / "flowstar_gpu").mkdir(parents=True)
    complete = tmp_path / "complete"
    (complete / "src" / "flowstar_gpu").mkdir(parents=True)
    (complete / "pyproject.toml").write_text("[build-system]\n", encoding="utf-8")

    assert AUDIT.find_engine_roots(tmp_path) == [complete]


def test_record_provenance_retains_dirty_base_revision(tmp_path: Path) -> None:
    records = tmp_path / "comparison" / "run" / "records"
    records.mkdir(parents=True)
    (records / "dirty.json").write_text(
        json.dumps({"git": {"flowstar_gpu": "abc-dirty"}}), encoding="utf-8"
    )
    (records / "clean.json").write_text(
        json.dumps({"git": {"flowstar_gpu": "def"}}), encoding="utf-8"
    )

    rows = AUDIT.collect_record_provenance(tmp_path)

    assert [(row["base_revision"], row["source_state"]) for row in rows] == [
        ("def", "CLEAN"),
        ("abc", "DIRTY"),
    ]


def test_checksum_verifier_rejects_tamper_and_uncovered_file(tmp_path: Path) -> None:
    (tmp_path / "evidence.txt").write_text("original\n", encoding="utf-8")
    AUDIT.write_checksums(tmp_path)
    assert AUDIT.verify_checksums(tmp_path) == []

    (tmp_path / "evidence.txt").write_text("tampered\n", encoding="utf-8")
    assert AUDIT.verify_checksums(tmp_path) == ["checksum mismatch: evidence.txt"]

    (tmp_path / "extra.txt").write_text("uncovered\n", encoding="utf-8")
    assert AUDIT.verify_checksums(tmp_path) == [
        "checksum mismatch: evidence.txt",
        "uncovered file: extra.txt",
    ]


def test_gap_rows_do_not_conflate_current_head_with_dirty_records() -> None:
    rows = AUDIT.artifact_gap_rows({"git": {"head": "clean-head"}})
    by_id = {row["artifact_id"]: row for row in rows}

    assert by_id["CURRENT_ENGINE_SOURCE"]["status"] == "AVAILABLE"
    assert "clean-head" in by_id["CURRENT_ENGINE_SOURCE"]["evidence"]
    assert by_id["HISTORICAL_DIRTY_PATCHES"]["status"] == "MISSING"
    assert "not exact" in by_id["HISTORICAL_DIRTY_PATCHES"]["effect"]
    assert by_id["PAPER_PDF"]["status"] == "MISSING"


def test_committed_phase_a_package_opens_only_current_source_gate() -> None:
    output_root = ROOT / "outputs" / "huan_repro_audit"
    inventory = json.loads((output_root / "artifact_inventory.json").read_text(encoding="utf-8"))
    manifest = json.loads((output_root / "source_manifest.json").read_text(encoding="utf-8"))

    assert inventory["schema"].endswith("/2")
    assert inventory["source_closure"]["current_clean_engine_source_available"] is True
    assert inventory["source_closure"]["historical_dirty_experiment_state_available"] is False
    assert inventory["phase_gates"]["current_source_build_and_kernel_audit"] == "OPEN"
    assert inventory["phase_gates"]["historical_result_exact_reproduction"].startswith("CLOSED")
    assert inventory["phase_gates"]["controller_coupling_scope"] == "PROHIBITED_NOT_RUN"
    assert manifest["git"]["clean"] is True
    assert manifest["historical_dirty_state_exact"] is False
    assert "src/flowstar_gpu/interval.py" in manifest["key_file_sha256"]
    assert AUDIT.verify_checksums(output_root) == []


def test_artifact_gap_table_replaces_obsolete_missing_source_table() -> None:
    output_root = ROOT / "outputs" / "huan_repro_audit"
    gaps = (output_root / "artifact_gaps.tsv").read_text(encoding="utf-8")

    assert "CURRENT_ENGINE_SOURCE\tAVAILABLE" in gaps
    assert "HISTORICAL_DIRTY_PATCHES\tMISSING" in gaps
    assert not (output_root / "missing_artifacts.tsv").exists()
