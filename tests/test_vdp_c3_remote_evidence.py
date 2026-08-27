from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import pytest


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "artifacts/runs/vdp_c3_cross_step_causal_closure_20260827"
SCRIPT = ROOT / "scripts/verify_vdp_c3_remote_evidence.py"
SPEC = importlib.util.spec_from_file_location("verify_vdp_c3_remote_evidence", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
VERIFY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFY)


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


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


@pytest.fixture()
def copied_package(tmp_path: Path) -> Path:
    assert PACKAGE.is_dir(), "the committed Phase-0 evidence package is required"
    destination = tmp_path / "evidence"
    shutil.copytree(PACKAGE, destination)
    return destination


def _semantic_errors(package: Path) -> list[str]:
    _, errors = VERIFY.verify(package, repository=ROOT, check_git=True, run_tests=False)
    return errors


def test_committed_package_recomputes_cleanly() -> None:
    result, errors = VERIFY.verify(PACKAGE, repository=ROOT, check_git=True, run_tests=False)
    assert errors == []
    assert result is not None
    assert all(result["gates"].values())
    assert result["highest_status"] == (
        "CROSS_STEP_CAUSE_IDENTIFIED__C3_PRODUCTION_GATE_PASSED__NATIVE_T10_REACHED"
    )


def test_width_tamper_is_rejected_after_checksum_refresh(copied_package: Path) -> None:
    path = copied_package / "raw/fixed/torch_c3/T1/summary.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["raw_endpoint"]["x_width"] += 1e-3
    _write_json(path, payload)
    _refresh_checksums(copied_package)
    assert _semantic_errors(copied_package)


def test_horizon_tamper_is_rejected_after_checksum_refresh(copied_package: Path) -> None:
    path = copied_package / "raw/native/torch_c3/summary.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["completed_horizon"] = 9.999
    _write_json(path, payload)
    _refresh_checksums(copied_package)
    assert _semantic_errors(copied_package)


def test_source_sha_tamper_is_rejected_after_checksum_refresh(copied_package: Path) -> None:
    path = copied_package / "raw/fixed/torch_c3/T3/command.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["commit"] = "0" * 40
    _write_json(path, payload)
    _refresh_checksums(copied_package)
    assert _semantic_errors(copied_package)


def test_junit_count_tamper_is_rejected_after_checksum_refresh(copied_package: Path) -> None:
    path = copied_package / "raw/tests/pytest.xml"
    tree = ET.parse(path)
    root = tree.getroot()
    suite = root if root.tag == "testsuite" else root.find(".//testsuite")
    assert suite is not None and suite.get("tests") is not None
    suite.set("tests", str(int(suite.get("tests", "0")) + 1))
    tree.write(path, encoding="utf-8", xml_declaration=True)
    _refresh_checksums(copied_package)
    assert _semantic_errors(copied_package)


def test_highest_status_tamper_is_rejected_after_checksum_refresh(copied_package: Path) -> None:
    path = copied_package / "RESULT.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["highest_status"] = "C3_GENERIC_POLYNOMIAL_PLANT_CORE_VALIDATED"
    _write_json(path, payload)
    _refresh_checksums(copied_package)
    assert _semantic_errors(copied_package)
