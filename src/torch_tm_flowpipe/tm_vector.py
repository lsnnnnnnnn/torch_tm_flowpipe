"""Vector convenience wrapper for Taylor models."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Iterator, List, Sequence

import torch

from .interval import Interval
from .taylor_model import TaylorModel


@dataclass(frozen=True)
class TMVector:
    models: List[TaylorModel]

    def __init__(self, models: Iterable[TaylorModel]):
        models_l = list(models)
        if models_l:
            base_domain = models_l[0].domain
            for m in models_l[1:]:
                if len(m.domain) != len(base_domain):
                    raise ValueError("all Taylor models must use the same number of variables")
        object.__setattr__(self, "models", models_l)

    @staticmethod
    def identity(domain: Sequence[Interval], *, order: int | None = None) -> "TMVector":
        return TMVector(TaylorModel.variable(i, domain, order=order) for i in range(len(domain)))

    @staticmethod
    def constants(values: Sequence[Any], domain: Sequence[Interval], *, order: int | None = None) -> "TMVector":
        return TMVector(TaylorModel.constant(v, domain, order=order) for v in values)

    def __len__(self) -> int:
        return len(self.models)

    def __iter__(self) -> Iterator[TaylorModel]:
        return iter(self.models)

    def __getitem__(self, i: int) -> TaylorModel:
        return self.models[i]

    @property
    def domain(self) -> list[Interval]:
        return list(self.models[0].domain) if self.models else []

    @property
    def n_vars(self) -> int:
        return len(self.domain)

    def range_box(self) -> list[Interval]:
        return [m.range_box() for m in self.models]

    def widths(self) -> torch.Tensor:
        if not self.models:
            return torch.empty(0, dtype=torch.float64)
        return torch.stack([m.range_box().width() for m in self.models])

    def max_width(self) -> torch.Tensor:
        w = self.widths()
        return torch.max(w) if w.numel() else torch.as_tensor(0.0, dtype=torch.float64)

    def extend_domain(self, new_interval: Interval) -> "TMVector":
        return TMVector(m.extend_domain(new_interval) for m in self.models)

    def substitute_const(self, var_index: int, value: Any) -> "TMVector":
        return TMVector(m.substitute_const(var_index, value) for m in self.models)

    def drop_variable(self, var_index: int, *, require_zero_exponent: bool = True) -> "TMVector":
        return TMVector(m.drop_variable(var_index, require_zero_exponent=require_zero_exponent) for m in self.models)

    def active_variables(self) -> set[int]:
        active: set[int] = set()
        for m in self.models:
            active.update(m.active_variables())
        return active

    def with_remainders(self, remainders: Sequence[Interval]) -> "TMVector":
        if len(remainders) != len(self.models):
            raise ValueError("remainder length mismatch")
        return TMVector(m.with_remainder(r) for m, r in zip(self.models, remainders))

    def apply_cutoff(self, threshold: float | None) -> "TMVector":
        if threshold is None:
            return self
        return TMVector(m.apply_cutoff(threshold) for m in self.models)

    def __add__(self, other: "TMVector") -> "TMVector":
        if len(self) != len(other):
            raise ValueError("TMVector length mismatch")
        return TMVector(a + b for a, b in zip(self.models, other.models))

    def __sub__(self, other: "TMVector") -> "TMVector":
        if len(self) != len(other):
            raise ValueError("TMVector length mismatch")
        return TMVector(a - b for a, b in zip(self.models, other.models))
