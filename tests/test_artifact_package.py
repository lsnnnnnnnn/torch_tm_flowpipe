from pathlib import Path

import pytest

from torch_tm_flowpipe.artifact_package import (
    REQUIRED_FIGURES,
    REQUIRED_MACHINE_FILES,
    validate_required_package,
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
