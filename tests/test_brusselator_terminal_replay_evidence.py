from __future__ import annotations

import json
from pathlib import Path
import shutil

from scripts.verify_brusselator_terminal_replay_evidence import (
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


def test_terminal_replay_package_recomputes_closed_c3_status() -> None:
    result, errors = verify(DEFAULT_PACKAGE)
    assert errors == []
    assert result is not None
    assert result["soundness_gate_passed"] is True
    assert result["status"] == "C3_GENERIC_CORE_VALIDATED__SECOND_SYSTEM_NO_MATERIAL_GAIN"
    assert result["original_sr100_false_checks"] == ["rollback", "summary_certificate"]
    assert all(result["supplemental_replay_checks"].values())


def test_terminal_replay_checkpoint_tamper_fails_closed(tmp_path: Path) -> None:
    package = tmp_path / "package"
    shutil.copytree(DEFAULT_PACKAGE, package)
    path = package / "raw/checkpoint_after/terminal_state.json"
    path.write_bytes(path.read_bytes() + b" ")
    _rehash(package)
    _, errors = verify(package)
    assert errors


def test_terminal_replay_result_tamper_fails_closed(tmp_path: Path) -> None:
    package = tmp_path / "package"
    shutil.copytree(DEFAULT_PACKAGE, package)
    path = package / "raw/RESULT.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["terminal_attempt_count"] = 2
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _rehash(package)
    _, errors = verify(package)
    assert errors
