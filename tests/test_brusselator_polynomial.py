from __future__ import annotations
import torch
from torch_tm_flowpipe import Interval, TMVector
from torch_tm_flowpipe.ode_examples import brusselator_ode


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
