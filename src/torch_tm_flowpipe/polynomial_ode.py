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

    def _factorized_coefficient(
        self,
        state: Any,
        terms: Sequence[PolynomialODETerm],
        *,
        outer_index: int,
        initial: Any | None = None,
    ) -> Any:
        """Accumulate signed coefficient terms in an outer-variable Horner form."""

        value: Any | None = initial
        for term in terms:
            factors = [
                state[index]
                for index, power in enumerate(term.powers)
                if index != int(outer_index)
                for _ in range(power)
            ]
            if factors:
                term_value = factors[0]
                for factor in factors[1:]:
                    term_value = term_value * factor
                magnitude = abs(term.coefficient)
                if magnitude != 1.0:
                    term_value = term_value * magnitude
            else:
                # Keep constants scalar until they meet a Taylor model.  A
                # manufactured ``0 * state`` would create a false remainder
                # operation in the audited graph.
                term_value = abs(term.coefficient)
            if value is None:
                value = term_value if term.coefficient >= 0.0 else -term_value
            elif term.coefficient >= 0.0:
                value = value + term_value
            else:
                value = value - term_value
        return 0.0 if value is None else value

    def evaluate_canonical_factorized(self, state: Any, control: Any | None = None) -> Any:
        """Evaluate through a deterministic outer-variable Horner graph.

        The last state variable is the outer Horner variable.  For the frozen
        Van der Pol polynomial this gives exactly ``(1 - x*x) * y - x``:
        equal y paths are combined before their single multiplication, while
        coefficient terms retain the authoritative factor order.
        """

        del control
        if len(state) != self.state_dim:
            raise ValueError("polynomial ODE state dimension mismatch")
        outer_index = self.state_dim - 1
        outputs = []
        for component in self.components:
            grouped: dict[int, list[PolynomialODETerm]] = {}
            for term in component:
                grouped.setdefault(int(term.powers[outer_index]), []).append(term)
            highest = max(grouped)
            value = self._factorized_coefficient(
                state,
                grouped.get(highest, ()),
                outer_index=outer_index,
            )
            for power in range(highest - 1, -1, -1):
                if isinstance(value, (int, float)) and value == 1.0:
                    value = state[outer_index]
                elif isinstance(value, (int, float)) and value == -1.0:
                    value = -state[outer_index]
                else:
                    value = value * state[outer_index]
                value = self._factorized_coefficient(
                    state,
                    grouped.get(power, ()),
                    outer_index=outer_index,
                    initial=value,
                )
            if isinstance(value, (int, float)):
                value = state[0] * 0.0 + value
            outputs.append(value)
        concat = getattr(type(outputs[0]), "concat", None)
        if callable(concat):
            return concat(outputs)
        return TMVector(outputs)

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
