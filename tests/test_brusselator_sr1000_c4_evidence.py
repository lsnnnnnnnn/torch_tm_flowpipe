from __future__ import annotations

import pytest

from scripts.verify_brusselator_sr1000_c4_evidence import DEFAULT_PACKAGE, verify


@pytest.mark.integration
def test_brusselator_sr1000_operator_c4_package_recomputes_closed() -> None:
    result, errors = verify(DEFAULT_PACKAGE)
    assert errors == []
    assert result is not None
    assert result["status"] == "BRUSSELATOR_SR1000_OPERATOR_C4_CLOSED"
    assert result["capacity_reset_decision"] == "NOT_SOLELY_QUEUE_RESET_CAPACITY"
    assert result["first_material_operator_divergence"] == "truncation_cutoff_owners"
    assert result["c4_status"] == "C4_FIX_AUTHORIZED"
    assert all(result["checks"].values())
