from dataclasses import dataclass

import pytest

from torch_tm_flowpipe import PolynomialODE


@dataclass(frozen=True)
class _Expression:
    text: str

    @property
    def domain(self):
        return ()

    def __mul__(self, other):
        return _Expression(f"({self.text}*{_text(other)})")

    def __rmul__(self, other):
        return _Expression(f"({_text(other)}*{self.text})")

    def __add__(self, other):
        return _Expression(f"({self.text}+{_text(other)})")

    def __radd__(self, other):
        return _Expression(f"({_text(other)}+{self.text})")

    def __sub__(self, other):
        return _Expression(f"({self.text}-{_text(other)})")

    def __rsub__(self, other):
        return _Expression(f"({_text(other)}-{self.text})")

    def __neg__(self):
        return _Expression(f"(-{self.text})")


def _text(value):
    return value.text if isinstance(value, _Expression) else repr(value)


def _vdp_spec():
    return {
        "state_names": ["position", "velocity"],
        "rhs": [
            {"terms": [{"coefficient": 1.0, "powers": [0, 1]}]},
            {
                "terms": [
                    {"coefficient": 1.0, "powers": [0, 1]},
                    {"coefficient": -1.0, "powers": [1, 0]},
                    {"coefficient": -1.0, "powers": [2, 1]},
                ]
            },
        ],
    }


def test_polynomial_ode_preserves_canonical_vdp_term_and_factor_order():
    ode = PolynomialODE.from_system_spec(_vdp_spec())

    result = ode([_Expression("x"), _Expression("y")])

    assert result[0].text == "y"
    assert result[1].text == "((y-x)-((x*x)*y))"


def test_polynomial_ode_opt_in_factorized_graph_matches_flowstar_expression():
    ode = PolynomialODE.from_system_spec(_vdp_spec())

    result = ode.evaluate_canonical_factorized([_Expression("x"), _Expression("y")])

    assert result[0].text == "y"
    assert result[1].text == "(((1.0-(x*x))*y)-x)"


def test_polynomial_ode_rejects_bad_power_dimension():
    spec = _vdp_spec()
    spec["rhs"][1]["terms"][0]["powers"] = [0]
    with pytest.raises(ValueError, match="power dimension"):
        PolynomialODE.from_system_spec(spec)
