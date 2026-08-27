from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

from scripts.verify_brusselator_second_system_evidence import (
    DEFAULT_PACKAGE,
    sha256,
    verify,
)


def _rehash(package: Path) -> None:
    files = sorted(
        path for path in package.rglob("*") if path.is_file() and path.name != "SHA256SUMS"
    )
    (package / "SHA256SUMS").write_text(
        "".join(f"{sha256(path)}  {path.relative_to(package).as_posix()}\n" for path in files),
        encoding="ascii",
    )


def test_brusselator_second_system_package_recomputes_terminal_status() -> None:
    result, errors = verify(DEFAULT_PACKAGE)
    assert errors == []
    assert result is not None
    assert result["status"] == "C3_GENERICITY_SOUNDNESS_GATE_FAILED_STOP"
    assert result["exact_fraction_2d_test_passed"] is True
    assert result["flowstar_completed_t20"] is True
    assert result["lane_checks"]["torch_generic_sr100"]["owner_accounting"] is True
    assert result["lane_checks"]["torch_generic_sr100"]["rollback"] is False


def test_brusselator_second_system_raw_tamper_fails_closed(tmp_path: Path) -> None:
    package = tmp_path / "package"
    shutil.copytree(DEFAULT_PACKAGE, package)
    summary_path = package / "raw/torch_generic_sr100/summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["certificate_checks_passed"] = True
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _rehash(package)
    _, errors = verify(package)
    assert errors


def test_brusselator_second_system_result_tamper_fails_closed(tmp_path: Path) -> None:
    package = tmp_path / "package"
    shutil.copytree(DEFAULT_PACKAGE, package)
    result_path = package / "RESULT.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["status"] = "C3_GENERIC_CORE_VALIDATED__SECOND_SYSTEM_PRODUCTION_USEFUL"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    canonical = json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False)
    contract_path = package / "EVIDENCE_CONTRACT.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["result_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _rehash(package)
    _, errors = verify(package)
    assert "RESULT.json does not match raw recomputation" in errors
