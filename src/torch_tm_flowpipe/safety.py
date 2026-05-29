"""Small safety helpers used to fail closed on NaN/Inf during prototype runs."""
from __future__ import annotations

from typing import Iterable

import torch

from .interval import Interval


def tensor_is_finite(x: torch.Tensor) -> bool:
    return bool(torch.all(torch.isfinite(x)))


def interval_is_finite(x: Interval) -> bool:
    return x.is_finite()


def intervals_are_finite(xs: Iterable[Interval]) -> bool:
    return all(interval_is_finite(x) for x in xs)


class ValidationError(RuntimeError):
    """Raised when a flowpipe segment cannot be validated."""

