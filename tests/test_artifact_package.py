import ast
import csv
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from torch_tm_flowpipe.artifact_package import (
    ALLOWED_NUMERICAL_SOUNDNESS_CLASSES,
    ALLOWED_NUMERICAL_SOUNDNESS_SCOPES,
    CANONICAL_RUN_RELATIVE,
    REQUIRED_FIGURES,
    REQUIRED_MACHINE_FILES,
    load_json_artifact,
    reject_nonfinite,
    validate_raw_evidence,
    validate_report_artifact_references,
    validate_required_package,
    verify_artifact_manifests,
    verify_recovery_inventory,
    verify_sha256sums,
    write_sha256sums,
)


def _complete_package(root: Path) -> None:
    (root / "figures").mkdir(parents=True)
    for name in REQUIRED_MACHINE_FILES:
        (root / name).write_text(f"machine:{name}\n", encoding="utf-8")
    for name in REQUIRED_FIGURES:
        (root / "figures" / name).write_bytes(f"figure:{name}\n".encode())


def test_result_package_required_schema_and_recursive_checksums(tmp_path):
    _complete_package(tmp_path)
    validate_required_package(tmp_path)
    count = write_sha256sums(tmp_path)
    assert count == len(REQUIRED_MACHINE_FILES) + len(REQUIRED_FIGURES)
    assert verify_sha256sums(tmp_path) == (True, [])

    (tmp_path / "native_baselines.json").write_text("changed\n", encoding="utf-8")
    valid, errors = verify_sha256sums(tmp_path)
    assert not valid
    assert errors == ["native_baselines.json: digest mismatch"]


def test_result_package_validation_fails_closed_on_missing_artifact(tmp_path):
    _complete_package(tmp_path)
    (tmp_path / "timing.csv").unlink()
    with pytest.raises(ValueError, match="timing.csv"):
        validate_required_package(tmp_path)


def test_checksum_paths_can_be_root_prefixed_for_sha256sum_from_repository_root(tmp_path):
    run_root = tmp_path / "outputs" / "run"
    _complete_package(run_root)
    prefix = "outputs/run"
    write_sha256sums(run_root, path_prefix=prefix)
    assert verify_sha256sums(run_root, path_prefix=prefix) == (True, [])
    first = (run_root / "SHA256SUMS").read_text(encoding="utf-8").splitlines()[0]
    assert f"  {prefix}/" in first


def test_json_loader_accepts_deterministic_gzip_and_rejects_nonfinite(tmp_path):
    import gzip

    compressed = tmp_path / "summary.json.gz"
    with compressed.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as handle:
            handle.write(b'{"finite": 1.25}\n')
    assert load_json_artifact(tmp_path / "summary.json") == {"finite": 1.25}
    with pytest.raises(ValueError, match="nonfinite"):
        reject_nonfinite({"bad": [float("nan")]}, label="fixture")


def test_mainline_builder_has_no_literal_duplicate_dictionary_keys():
    repository_root = Path(__file__).resolve().parents[1]
    source = repository_root / "experiments/build_mainline_realignment_package.py"
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    duplicates: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        constants = [
            key.value
            for key in node.keys
            if isinstance(key, ast.Constant) and isinstance(key.value, (str, int, float))
        ]
        for key in set(constants):
            if constants.count(key) > 1:
                duplicates.append((node.lineno, str(key)))
    assert duplicates == []


def test_committed_mainline_package_validates_in_clean_temporary_copy(tmp_path):
    repository_root = Path(__file__).resolve().parents[1]
    run_root = repository_root / CANONICAL_RUN_RELATIVE
    validate_raw_evidence(run_root)
    validate_required_package(run_root)
    assert verify_artifact_manifests(run_root) == (True, [])
    assert verify_recovery_inventory(run_root) == (True, [])
    assert verify_sha256sums(
        run_root, path_prefix=CANONICAL_RUN_RELATIVE.as_posix()
    ) == (True, [])

    markdown_paths = [repository_root / "README.md", repository_root / "handoff.md"]
    markdown_paths.extend(sorted((repository_root / "docs").glob("*.md")))
    validate_report_artifact_references(
        repository_root,
        markdown_paths,
        require_tracked=True,
    )

    clean_copy = tmp_path / CANONICAL_RUN_RELATIVE.name
    shutil.copytree(run_root, clean_copy)
    validate_raw_evidence(clean_copy)
    validate_required_package(clean_copy)
    assert verify_artifact_manifests(clean_copy) == (True, [])
    assert verify_recovery_inventory(clean_copy) == (True, [])
    assert verify_sha256sums(
        clean_copy, path_prefix=CANONICAL_RUN_RELATIVE.as_posix()
    ) == (True, [])

    subprocess.run(
        [
            sys.executable,
            str(repository_root / "experiments/build_mainline_realignment_package.py"),
            "--run-root",
            str(clean_copy),
        ],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )
    derived = [*REQUIRED_MACHINE_FILES, "SHA256SUMS"]
    derived.extend(f"figures/{name}" for name in REQUIRED_FIGURES)
    assert {
        name: (clean_copy / name).read_bytes() for name in derived
    } == {
        name: (run_root / name).read_bytes() for name in derived
    }


def test_canonical_claim_and_soundness_rows_use_split_eligibility_schema():
    repository_root = Path(__file__).resolve().parents[1]
    run_root = repository_root / CANONICAL_RUN_RELATIVE
    required = {
        "mathematical_contract_known",
        "requested_horizon_completed",
        "certificate_semantics_passed",
        "finite_outputs",
        "numerical_soundness_class",
        "numerical_soundness_scope",
        "formal_claim_eligible",
        "performance_measurement_eligible",
        "cross_tool_ranking_eligible",
    }
    for filename in (
        "soundness_matrix.csv",
        "claim_registry.csv",
        "full_horizon.csv",
        "batch_scaling.csv",
    ):
        with (run_root / filename).open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert rows, filename
        assert required <= rows[0].keys(), filename
        assert "eligible" not in rows[0].keys(), filename
        for row in rows:
            assert row["numerical_soundness_class"] in ALLOWED_NUMERICAL_SOUNDNESS_CLASSES
            assert row["numerical_soundness_scope"] in ALLOWED_NUMERICAL_SOUNDNESS_SCOPES

    native = load_json_artifact(run_root / "native_baselines.json")
    for lane in native["lanes"].values():
        assert required <= lane.keys()
        assert lane["numerical_soundness_class"] in ALLOWED_NUMERICAL_SOUNDNESS_CLASSES
        assert lane["numerical_soundness_scope"] in ALLOWED_NUMERICAL_SOUNDNESS_SCOPES
        assert lane["cross_tool_ranking_eligible"] is False
    assert native["lanes"]["flowstar_stock"]["numerical_soundness_class"] == (
        "unsound/ineligible on a demonstrated counterexample"
    )
