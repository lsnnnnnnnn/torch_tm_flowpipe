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


def test_record_provenance_distinguishes_dirty_and_clean(tmp_path: Path) -> None:
    records = tmp_path / "comparison" / "run" / "records"
    records.mkdir(parents=True)
    (records / "dirty.json").write_text(
        json.dumps({"git": {"flowstar_gpu": "abc-dirty"}}), encoding="utf-8"
    )
    (records / "clean.json").write_text(
        json.dumps({"git": {"flowstar_gpu": "def"}}), encoding="utf-8"
    )
    (records / "unrelated.json").write_text(json.dumps({"git": {}}), encoding="utf-8")

    rows = AUDIT.collect_record_provenance(tmp_path)

    assert [(row["flowstar_gpu_revision"], row["source_state"]) for row in rows] == [
        ("def", "CLEAN"),
        ("abc-dirty", "DIRTY"),
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


def test_missing_artifact_requests_name_dirty_patch_and_exact_inputs() -> None:
    rows = AUDIT.missing_artifact_rows(["abc-dirty"])
    by_id = {row["artifact_id"]: row for row in rows}

    assert by_id["EXACT_CLEAN_STATE"]["status"] == "MISSING"
    assert "abc-dirty" in by_id["EXACT_CLEAN_STATE"]["evidence"]
    assert "git diff --binary" in by_id["DIRTY_PATCHES"]["request_from_huan"]
    assert "frozen VDP" in by_id["ENGINE_BENCHMARKS"]["request_from_huan"]


def test_committed_phase_a_package_verifies_and_fails_closed() -> None:
    output_root = ROOT / "outputs" / "huan_repro_audit"
    inventory = json.loads((output_root / "artifact_inventory.json").read_text(encoding="utf-8"))
    candidate = inventory["candidate_repositories"][0]

    assert inventory["primary_decision"] == "HUAN_REPRO_BLOCKED_MISSING_CORE_SOURCE"
    assert inventory["source_closure"]["qualifying_engine_roots"] == []
    assert inventory["stop_rule"] == {
        "phases_not_run": ["B", "C", "D", "E", "F"],
        "reason": (
            "flowstar_gpu/src/flowstar_gpu, engine build files, and exact clean "
            "engine source state are unavailable"
        ),
        "triggered": True,
    }
    assert candidate["result_record_provenance"]["clean_record_count"] == 0
    assert candidate["result_record_provenance"]["dirty_record_count"] == 450
    assert candidate["symlinks"]["inaccessible_count"] == 94
    assert AUDIT.verify_checksums(output_root) == []


def test_inapplicable_scientific_tables_are_ledgered_not_fabricated() -> None:
    output_root = ROOT / "outputs" / "huan_repro_audit"
    report = (ROOT / "docs" / "HUAN_ENGINE_REPRODUCTION_AUDIT_20260826.md").read_text(
        encoding="utf-8"
    )
    inapplicable = (
        "source_manifest.json",
        "proof_to_code_map.csv",
        "step1_common_input.csv",
        "fixed_horizon_matrix.csv",
        "native_terminal.json",
        "batch_throughput.csv",
    )
    for name in inapplicable:
        assert not (output_root / name).exists()
        assert f"`outputs/huan_repro_audit/{name}` | `NOT_RUN_SOURCE_MISSING`" in report
