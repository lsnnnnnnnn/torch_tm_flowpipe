from __future__ import annotations

import csv
from fractions import Fraction
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "huan_proof_kernel_audit.py"
OUTPUT = ROOT / "outputs" / "huan_repro_audit"


def _module():
    spec = importlib.util.spec_from_file_location("huan_proof_kernel_audit", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_proof_map_has_required_schema_claims_and_explicit_gaps() -> None:
    audit = _module()
    with (OUTPUT / "proof_to_code_map.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert tuple(reader.fieldnames or ()) == audit.CSV_FIELDS
        rows = list(reader)

    assert len(rows) == 14
    assert {row["status"] for row in rows} <= audit.ALLOWED_STATUSES
    assert all(row["gap"] for row in rows)
    by_id = {row["claim_id"]: row for row in rows}
    assert by_id["FP_NO_FTZ_STARTUP"]["status"] == "CONTRADICTED"
    assert by_id["STRICT_VERSUS_PARITY"]["status"] == "CONTRADICTED"
    assert by_id["TRANSCENDENTAL_ASSUMPTIONS"]["status"] == "ASSUMPTION_ONLY"
    assert by_id["POLYNOMIAL_ONLY_UNCONDITIONAL"]["status"] == "CONTRADICTED"


def test_exact_schedule_helpers_preserve_the_intended_rounding_diversity() -> None:
    audit = _module()
    values = [1e16, 1.0, -1e16]
    exact = sum(map(Fraction, values), Fraction())
    assert exact == 1
    assert audit._sequential(values) == 0.0
    assert audit._pairwise(values) == 0.0
    assert audit._chunked(values, 2) == 0.0
    assert audit._exact_dot(values, [1.0, 1.0, 1.0]) == exact


def test_committed_cpu_and_cuda_microkernel_evidence_passes() -> None:
    payloads = {
        device: json.loads((OUTPUT / "raw_logs" / f"proof_kernel_{device}.json").read_text())
        for device in ("cpu", "cuda")
    }
    for device, payload in payloads.items():
        assert payload["device"] == device
        assert payload["proof_map_errors"] == []
        assert payload["d1"]["passed"] == payload["d1"]["case_count"] == 7
        assert payload["d1"]["no_ftz_observed"] is True
        assert payload["d2"]["checked"] >= 900
        assert payload["d2"]["passed"] == payload["d2"]["checked"]
        assert payload["d2"]["failures"] == []
        assert payload["gate_passed"] is True
    assert payloads["cuda"]["cuda_kernel_available"] is True
