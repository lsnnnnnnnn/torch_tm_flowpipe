import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "outputs/vdp_residual_gap_g2_shared_column_20260815/20260815T120000Z"
VERIFIER = ROOT / "experiments/verify_vdp_g2_evidence_20260815.py"


def _load_verifier():
    spec = importlib.util.spec_from_file_location("vdp_g2_evidence_verifier_test", VERIFIER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_compact_g2_evidence_recomputes_to_fail_closed_decisions():
    if not PACKAGE.is_dir():
        pytest.skip("compact evidence is assembled after the bootstrap test run")

    result = _load_verifier().verify(PACKAGE)

    assert result["status"] == "PASS"
    assert result["conclusion"] == "G2_MECHANISM_IMPROVED__PRODUCTION_GATE_NOT_MET"
    assert result["total_cause_conclusion"] == "LOSSLESS_CROSS_OPERATOR_CELL_UNAVAILABLE__TOTAL_CAUSE_OPEN"
    assert result["integrity"]["bytes"] < 25 * 1024 * 1024


def test_verifier_classifies_all_preregistered_outcomes_without_observed_label_assumption():
    verifier = _load_verifier()

    assert verifier.classify_g2(
        production_success=True,
        reached_t10=True,
        mechanism_improved=True,
    ) == "G2_VDP_T10_VALIDATED"
    assert verifier.classify_g2(
        production_success=False,
        reached_t10=False,
        mechanism_improved=True,
    ) == "G2_MECHANISM_IMPROVED__PRODUCTION_GATE_NOT_MET"
    assert verifier.classify_g2(
        production_success=False,
        reached_t10=False,
        mechanism_improved=False,
    ) == "G2_SHARED_COLUMN_CARRY_REJECTED"
