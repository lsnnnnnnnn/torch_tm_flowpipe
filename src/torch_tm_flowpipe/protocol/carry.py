from __future__ import annotations

from typing import Any, Callable


def projected_affine_box_reset(
    endpoint: Any,
    *,
    project_to_basis: Callable[..., tuple[Any, list[Any]]],
    affine_reset: Callable[..., tuple[Any, Any]],
    stage: str,
    iteration: int,
) -> tuple[Any, int]:
    """Project nonlinear terms before invoking the affine-only reset."""
    affine_endpoint, discarded = project_to_basis(
        endpoint,
        "B1",
        tau_index=None,
        stage=stage,
        iteration=iteration,
    )
    current, _ = affine_reset(affine_endpoint, method="box")
    return current, len(discarded)
