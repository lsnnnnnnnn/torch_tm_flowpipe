from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil

import pytest


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "artifacts/runs/vdp_generic_refactor_vdp_zero_regression_20260827"
SCRIPT = ROOT / "scripts/verify_vdp_generic_refactor_regression.py"
SPEC = importlib.util.spec_from_file_location("verify_vdp_generic_refactor_regression", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
VERIFY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFY)


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _refresh_checksums(package: Path) -> None:
    checksum = package / "SHA256SUMS"
    files = sorted(path for path in package.rglob("*") if path.is_file() and path != checksum)
    checksum.write_text(
        "".join(
            f"{VERIFY.sha256(path)}  {path.relative_to(package).as_posix()}\n"
            for path in files
        ),
        encoding="ascii",
    )


@pytest.fixture()
def copied_package(tmp_path: Path) -> Path:
    assert PACKAGE.is_dir(), "the committed Phase-2 regression package is required"
    destination = tmp_path / "evidence"
    shutil.copytree(PACKAGE, destination)
    return destination


def test_committed_vdp_refactor_regression_package_recomputes_cleanly() -> None:
    result, errors = VERIFY.verify(PACKAGE)
    assert errors == []
    assert result is not None
    assert result["passed"] is True
    assert result["maximum_c3_numeric_delta"] == 0.0


def test_c2_scientific_hash_tamper_is_rejected(copied_package: Path) -> None:
    path = copied_package / "raw/candidate/fixed/torch_c2/T1/summary.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["raw_endpoint"]["x_width"] += 1e-9
    _write_json(path, payload)
    _refresh_checksums(copied_package)
    _, errors = VERIFY.verify(copied_package)
    assert errors


def test_c3_over_tolerance_tamper_is_rejected(copied_package: Path) -> None:
    path = copied_package / "raw/candidate/fixed/torch_c3/T3/summary.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["last_segment"]["y_width"] += 1e-9
    _write_json(path, payload)
    _refresh_checksums(copied_package)
    _, errors = VERIFY.verify(copied_package)
    assert errors


def test_native_count_tamper_is_rejected(copied_package: Path) -> None:
    path = copied_package / "raw/candidate/native/torch_c3/T10/summary.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["rejected_attempts"] = 36
    _write_json(path, payload)
    _refresh_checksums(copied_package)
    _, errors = VERIFY.verify(copied_package)
    assert errors
