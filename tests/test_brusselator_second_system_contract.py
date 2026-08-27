from __future__ import annotations

from pathlib import Path

import torch

from torch_tm_flowpipe import Interval, TMVector
from torch_tm_flowpipe.ode_examples import brusselator_ode


ROOT = Path(__file__).resolve().parents[1]


def test_brusselator_expression_tree_has_the_preregistered_polynomial() -> None:
    domain = [Interval(-1.0, 1.0), Interval(-1.0, 1.0)]
    state = TMVector.identity(domain, order=6)
    result = brusselator_ode(state)
    expected = (
        {(0, 0): 1.0, (1, 0): -4.0, (2, 1): 1.0},
        {(1, 0): 3.0, (2, 1): -1.0},
    )
    for model, terms in zip(result, expected):
        assert set(model.polynomial.terms) == set(terms)
        for exponent, coefficient in terms.items():
            assert model.polynomial.terms[exponent].equal(
                torch.tensor(coefficient, dtype=torch.float64)
            )


def test_second_system_contract_freezes_exactly_three_lanes_and_no_sweep() -> None:
    contract = (ROOT / "SECOND_SYSTEM_CONTRACT.md").read_text(encoding="utf-8")
    assert "Exactly three lanes" in contract
    assert "torch_generic_no_queue" in contract
    assert "torch_generic_sr100" in contract
    assert "queue capacity 1000" in contract
    assert "capacity 100" in contract
    assert "No other queue capacity" in contract
    assert "fixed step = exact decimal 0.02" in contract
    assert "Taylor-model order = 6" in contract
    assert "requested horizon = 20" in contract
