from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
OUTPUT = ROOT / "outputs" / "huan_repro_audit"


def _load(name: str):
    path = SCRIPTS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_command_capture_records_failure_without_losing_output(tmp_path: Path) -> None:
    output = tmp_path / "capture.log"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "huan_capture_command.py"),
            "--cwd",
            str(tmp_path),
            "--output",
            str(output),
            "--label",
            "focused-test",
            "--",
            sys.executable,
            "-c",
            "print('captured'); raise SystemExit(7)",
        ],
        text=True,
        stdout=subprocess.PIPE,
    )
    assert result.returncode == 7
    header_text, body = output.read_text().split("\n--- combined stdout/stderr ---\n", 1)
    header = json.loads(header_text)
    assert header["returncode"] == 7
    assert header["label"] == "focused-test"
    assert body == "captured\n"


def test_phase_c_capture_hash_and_renderer_are_deterministic(tmp_path: Path) -> None:
    audit = _load("huan_phase_c_capture")
    item = tmp_path / "item"
    item.write_bytes(b"abc")
    assert audit._sha256(item) == hashlib.sha256(b"abc").hexdigest()
    rendered = audit._render([{"command": ["tool", "--flag"], "returncode": 3, "output": "out\n"}])
    assert "$ tool --flag" in rendered
    assert "[returncode=3]" in rendered


def test_config_path_plugin_changes_only_the_known_absolute_path(monkeypatch, tmp_path: Path) -> None:
    plugin = _load("pytest_huan_config_path")
    monkeypatch.setenv("HUAN_CONFIG_ROOT", str(tmp_path))
    known = SimpleNamespace(CONFIGS=Path("/home/huan/projects/CROWN-Reach/src/configs"))
    unrelated = SimpleNamespace(CONFIGS=Path("/somewhere/else"))
    plugin.pytest_collection_modifyitems(
        [SimpleNamespace(module=known), SimpleNamespace(module=unrelated)]
    )
    assert known.CONFIGS == tmp_path
    assert unrelated.CONFIGS == Path("/somewhere/else")


def test_committed_chunk_and_refinement_boundary_audits() -> None:
    for device in ("cpu", "cuda"):
        chunk = json.loads((OUTPUT / "raw_logs" / f"chunk_boundary_{device}.json").read_text())
        refine = json.loads((OUTPUT / "raw_logs" / f"refinement_boundary_{device}.json").read_text())
        assert chunk["device"] == device and chunk["gate_passed"] is True
        assert [row["requested_member_chunk"] for row in chunk["chunk_cases"]] == [1, 2, 3, 5, 7]
        assert all(row["coefficients_bitwise"] and row["remainders_bitwise"] for row in chunk["chunk_cases"])
        assert all(chunk["b1_embedded_in_b2"].values())
        assert refine["behavioral_passed"] is True
        assert refine["cap_boundary"]["calls_490"] == 490
        assert refine["cap_boundary"]["calls_491"] == 491
        assert refine["initial_self_map_failure"] == {"replay_calls": 0, "unchanged": True}
        assert refine["contract_gate_passed"] is False
        assert refine["api_contract"]["proposal_commit_ledger_exposed"] is False


def test_phase_d_gate_is_fail_closed_and_does_not_fabricate_scientific_tables() -> None:
    gate = json.loads((OUTPUT / "phase_d_gate.json").read_text())
    assert gate["primary_decision"] == "HUAN_SOURCE_BUILDS__PROOF_MAPPING_INCOMPLETE"
    assert gate["overall_gate_passed"] is False
    assert gate["gates"]["D5_PICARD_REFINEMENT"]["status"].startswith("FAIL")
    assert gate["gates"]["D6_STRICT_PARITY"]["status"].startswith("FAIL")
    assert set(gate["scientific_deliverables"].values()) == {"NOT_RUN_D_GATE_FAILED"}
    for name in gate["scientific_deliverables"]:
        assert not (OUTPUT / name).exists()
