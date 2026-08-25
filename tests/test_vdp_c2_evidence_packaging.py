from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments/package_vdp_c2_evidence_20260820.py"
SPEC = importlib.util.spec_from_file_location("vdp_c2_evidence_packaging", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
PACKAGING = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PACKAGING)

if str(ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(ROOT / "experiments"))
from tamper_test_vdp_c2_refinement_20260820 import run as run_tamper
from verify_vdp_c2_evidence_20260820 import ALLOWED_DECISIONS


def test_raw_run_packaging_retains_ledgers_and_discloses_trace_exclusions(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    retained = (
        "attempts.csv",
        "segments.csv",
        "remainder_ledger.jsonl",
        "refinement_ledger.jsonl",
        "summary.json",
    )
    for name in retained + PACKAGING.RAW_RUN_EXCLUDED_FILES:
        (source / name).write_text(name + "\n", encoding="utf-8")
    destination = tmp_path / "destination"
    PACKAGING._copy_raw_runs(source, destination)
    assert {path.name for path in destination.iterdir()} == set(retained)


def test_goal_terminal_decisions_are_the_only_verifier_decisions() -> None:
    assert ALLOWED_DECISIONS == {
        "C2_CAUSAL_GATE_FAILED",
        "C2_SOUND_LOCAL_CAUSE_ACCEPTED__PRODUCTION_GATE_FAILED",
        "C2_T1_T3_GATE_PASSED__T10_FAILED",
        "C2_T1_T3_AND_T10_PASSED",
    }


def test_packaged_gate_a_rejects_required_semantic_tampers() -> None:
    gate_dir = ROOT / (
        "evidence/vdp_c2_post_accept_refinement/20260820T090803Z/"
        "01_step1_causal_gate"
    )
    result = run_tamper(gate_dir)
    cases = {str(case["case"]): case for case in result["cases"]}
    for name in ("swap_components", "partial_commit", "stale_cache", "wrong_stop_ratio"):
        assert cases[name]["rejected"] is True
