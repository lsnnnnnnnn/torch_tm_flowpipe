import csv
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments/package_s1_boundary164_result.py"
SPEC = importlib.util.spec_from_file_location("package_s1_boundary164_result_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
packager = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(packager)

RUN_ROOT = (
    ROOT
    / "outputs/s1_boundary164_causal_guarded_carry_20260811/20260811T033447Z"
)


def test_primary_registry_outcome_is_derived_from_terminal_result():
    assert (
        packager._primary_outcome(
            {"outcome": "CORRECTED_S1_REACHES_TERMINAL_BUT_DOES_NOT_CLOSE_IT"}
        )
        == "S1_REACHES_TERMINAL_BUT_DOES_NOT_CLOSE_IT"
    )
    assert (
        packager._primary_outcome({"outcome": "CORRECTED_S1_TERMINAL_GATE_PASS"})
        == "S1_TOTAL_DELTA_PREFIX_RESTORED"
    )
    assert (
        packager._primary_outcome({"outcome": "different_failure"})
        == "S1_TOTAL_DELTA_REJECTS_BEFORE_TERMINAL"
    )


def test_manifest_links_and_repository_relative_checksums_are_complete():
    manifest = json.loads((RUN_ROOT / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["checksum_entry_count"] == len(manifest["artifacts"])
    assert all((RUN_ROOT / path).is_file() for path in manifest["artifacts"])
    lines = (RUN_ROOT / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    assert len(lines) == manifest["checksum_entry_count"]
    assert not any(line.endswith("/SHA256SUMS") for line in lines)
    assert all("  outputs/s1_boundary164_causal_guarded_carry_20260811/" in line for line in lines)


def test_unauthorized_stages_have_explicit_not_run_after_stop_rows():
    for name in ("horizon_ladder.csv", "second_system.csv"):
        with (RUN_ROOT / name).open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert rows
        assert all(row["status"] == "not_run_after_stop" for row in rows)
        assert all(
            row["primary_outcome"]
            == "S1_REACHES_TERMINAL_BUT_DOES_NOT_CLOSE_IT"
            for row in rows
        )
