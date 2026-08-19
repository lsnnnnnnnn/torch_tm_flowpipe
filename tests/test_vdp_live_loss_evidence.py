from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments/verify_vdp_live_loss_evidence_20260819.py"
EVIDENCE = ROOT / (
    "evidence/vdp_live_loss_ablation_b3_b4_closure/20260819T073038Z"
)


def _verifier_module():
    spec = importlib.util.spec_from_file_location("verify_vdp_live_loss_evidence", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_vdp_live_loss_c1_evidence_package_verifies() -> None:
    result = _verifier_module().verify(EVIDENCE)

    assert result["status"] == "verified"
    assert result["scientific_sha"] == "dbe03dcdfbf2f36b1d58013373d1d235ace1a48e"
    assert result["gate_events"] == 30
    assert result["tamper_cases"] == 5
