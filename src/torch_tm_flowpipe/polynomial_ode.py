"""Backend-neutral polynomial ODEs constructed from canonical term specs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .tm_vector import TMVector


@dataclass(frozen=True)
class PolynomialODETerm:
    coefficient: float
    powers: tuple[int, ...]

    @staticmethod
    def from_mapping(value: Mapping[str, Any], state_dim: int) -> "PolynomialODETerm":
        powers = tuple(int(power) for power in value["powers"])
        if len(powers) != int(state_dim):
            raise ValueError("polynomial ODE term power dimension mismatch")
        if any(power < 0 for power in powers):
            raise ValueError("polynomial ODE powers must be nonnegative")
        return PolynomialODETerm(float(value["coefficient"]), powers)


@dataclass(frozen=True)
class PolynomialODE:
    """Ordered polynomial expression usable by sparse and dense TM scalars.

    Term and factor order are retained from the authoritative config because
    raw-remainder replay is expression-tree sensitive even when two formulas
    define the same real polynomial.
    """

    components: tuple[tuple[PolynomialODETerm, ...], ...]
    state_dim: int

    @staticmethod
    def from_system_spec(system: Mapping[str, Any]) -> "PolynomialODE":
        state_dim = len(system["state_names"])
        rhs = system["rhs"]
        if len(rhs) != state_dim:
            raise ValueError("polynomial ODE component count must equal state dimension")
        components = tuple(
            tuple(PolynomialODETerm.from_mapping(term, state_dim) for term in component["terms"])
            for component in rhs
        )
        if any(not component for component in components):
            raise ValueError("each polynomial ODE component requires at least one term")
        return PolynomialODE(components, state_dim)

    def _unsigned_term(self, state: Any, term: PolynomialODETerm) -> Any:
        factors = [state[index] for index, power in enumerate(term.powers) for _ in range(power)]
        if not factors:
            return state[0] * 0.0 + abs(term.coefficient)
        value = factors[0]
        for factor in factors[1:]:
            value = value * factor
        magnitude = abs(term.coefficient)
        if magnitude != 1.0:
            value = value * magnitude
        return value

    def __call__(self, state: Any, control: Any | None = None) -> Any:
        del control
        if len(state) != self.state_dim:
            raise ValueError("polynomial ODE state dimension mismatch")
        outputs = []
        for component in self.components:
            first = component[0]
            first_value = self._unsigned_term(state, first)
            value = first_value if first.coefficient >= 0.0 else state[0] * 0.0 - first_value
            for term in component[1:]:
                term_value = self._unsigned_term(state, term)
                value = value + term_value if term.coefficient >= 0.0 else value - term_value
            outputs.append(value)
        concat = getattr(type(outputs[0]), "concat", None)
        if callable(concat):
            return concat(outputs)
        return TMVector(outputs)


__all__ = ["PolynomialODE", "PolynomialODETerm"]
