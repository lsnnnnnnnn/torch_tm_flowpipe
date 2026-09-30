"""Run settings for flowstar_gpu (mirrors Flow*'s Computational_Setting).

Defaults replicate the Flow* toolbox defaults (Continuous.cpp:59-78) except
where noted. Every field name states its Flow* counterpart so a reader can diff
configurations against a Flow* driver line-by-line.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    """Settings for a reach computation (fixed-step, or adaptive when step_min > 0).

    Attributes:
        step: integration step size delta (Flow*: setFixedStepsize arg 1).
        order: Taylor-model order k >= 2 (Flow*: setFixedStepsize arg 2).
        cutoff: cutoff threshold epsilon; monomials with |coeff| <= epsilon are
            moved into the remainder (Flow*: setCutoffThreshold; default 1e-10).
        remainder_estimation: a-priori remainder guess, as a symmetric radius
            (the interval used is [-v, +v]).  Two accepted forms, matching
            Flow*'s own per-variable API (`Taylor_Model_Setting::
            setRemainderEstimation(const std::vector<Interval> &)`,
            settings.h:153 / settings.cpp:228 — the field is
            `std::vector<Interval> remainder_estimation`, settings.h:126, and
            Continuous.cpp:963 reads it as `remainder_estimation[i]`):
              * float           — ONE radius broadcast to every declared
                                  variable.  This is what every Flow* driver
                                  and CrownSettings.cpp:86-88 does
                                  (`vector<Interval>(numVars, I)`), and it is
                                  the unchanged legacy behaviour.
              * tuple[float,..] — ONE radius PER declared variable, in
                                  declaration order; length must equal the
                                  ODE's variable count n.
            Why per-variable matters: stage (j) of the Picard validation
            requires total_i(r) subset r_i *for every i separately*
            (Continuous.cpp:985-1006).  With a uniform radius eps that
            collapses to eps >= delta*(b_i + eps*L_i) for every i, where
            L_i = sum_j |J_ij| is the Jacobian's i-th absolute ROW SUM — so it
            is FEASIBLE ONLY IF delta*max_i L_i < 1, and a single badly scaled
            row makes *no* scalar work at that step size.  Per-variable radii
            instead need only rho(delta*|J|) < 1.  See
            docs/GOTCHAS.md #6 and benchmarks/_tools/rem_est_recipe.py.
        sr_queue: symbolic-remainder queue size; 0 disables SR (Flow*:
            Symbolic_Remainder(initialSet, size)).
        mode: "parity" mirrors Flow* width-for-width (tier-S off);
            "strict" adds Rump bounds for the coefficient roundoff Flow* leaves
            unbounded (sounder than Flow*; default).
        device: torch device string.
        max_refinement_steps / stop_ratio: Flow* include.h MAX_REFINEMENT_STEPS
            (490) and STOP_RATIO (0.99) — refinement loop controls.
        validate_intervals: run assert_valid at pipeline stage boundaries
            (debugging aid; off for benchmarking).
    """

    step: float
    order: int
    cutoff: float = 1e-10
    # M6 adaptive stepsize (Flow* setAdaptiveStepsize(min, max, order)):
    # adaptive mode iff step_min > 0; `step` then serves as step_max (the
    # initial and largest step). Mirrors tm_setting.step_min > 0 dispatch.
    step_min: float = -1.0
    remainder_estimation: float | tuple[float, ...] = 1e-4
    sr_queue: int = 0
    mode: str = "strict"
    device: str = "cuda"
    max_refinement_steps: int = 490
    stop_ratio: float = 0.99
    validate_intervals: bool = False
    # Optional audit-only observer.  It receives refinement ledger records and
    # is never called when None, so the production numerical path is unchanged.
    refinement_callback: Callable[[dict[str, object]], None] | None = None

    def __post_init__(self) -> None:
        if self.step <= 0:
            raise ValueError(f"step must be positive, got {self.step}")
        if self.order < 2:
            # Flow* enforces order >= 2 (Computational_Setting::setFixedStepsize).
            raise ValueError(f"order must be >= 2, got {self.order}")
        if self.cutoff < 0:
            raise ValueError(f"cutoff must be >= 0, got {self.cutoff}")
        if self.mode not in ("parity", "strict"):
            raise ValueError(f"mode must be 'parity' or 'strict', got {self.mode!r}")
        if self.sr_queue < 0:
            raise ValueError(f"sr_queue must be >= 0, got {self.sr_queue}")
        # remainder_estimation is a symmetric RADIUS, so it must be >= 0 in
        # both forms; a negative value would silently build the inverted
        # interval [+v, -v] and make iv.contains() vacuously true — i.e. it
        # would defeat the contraction check rather than fail loudly.
        # 0 is legal (and useful): a variable with f_i == 0 identically, such
        # as an NNCS control input, needs no Picard slack of its own.
        if isinstance(self.remainder_estimation, tuple):
            if not self.remainder_estimation:
                raise ValueError("remainder_estimation tuple must be non-empty")
            if any(v < 0 for v in self.remainder_estimation):
                raise ValueError(
                    "remainder_estimation radii must all be >= 0, got "
                    f"{self.remainder_estimation}"
                )
        elif self.remainder_estimation < 0:
            raise ValueError(
                f"remainder_estimation must be >= 0, got {self.remainder_estimation}"
            )
        if self.step_min > 0 and self.step_min >= self.step:
            raise ValueError("adaptive mode needs step_min < step (= step_max)")
        if self.step_min > 0 and self.sr_queue > 0:
            raise NotImplementedError(
                "adaptive stepsize with symbolic remainders is not implemented. "
                "KNOWN GAP: coupled_vanderpol is the one shipped benchmark that "
                "combines them (with subdivision); backlog item recorded in "
                "ARCHITECTURE.md scope notes"
            )
        if self.refinement_callback is not None and not callable(self.refinement_callback):
            raise TypeError("refinement_callback must be callable or None")
