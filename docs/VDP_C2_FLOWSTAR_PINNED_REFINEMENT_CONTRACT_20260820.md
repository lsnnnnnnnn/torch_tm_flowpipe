# Flow* pinned post-accept remainder-refinement contract

This contract is extracted from stock Flow* commit
`b85a3211748cb77b736fe4ad42ee02d8d2b81148`.  It is source evidence, not an
inference from identifier names or runtime output.

## Pinned objects

| Path | Git blob |
|---|---|
| `flowstar-toolbox/Continuous.cpp` | `9cba9bb6fe072679c691a866bba7834c44bb6602` |
| `flowstar-toolbox/TaylorModel.h` | `401c759dea43c359523eec808d308a2733f8ed67` |
| `flowstar-toolbox/expression.h` | `f6f049f4c6ce056de2b7d6db5d13620172667a11` |
| `flowstar-toolbox/Interval.cpp` | `ef6dbe4a241e43e6254e8243a2e1c411ddffb9b8` |
| `flowstar-toolbox/include.h` | `c238d58efe1650fd7fcd53eb94bceb4381f19f97` |

## Proven behavior

`Continuous.cpp:961-1006` assigns the configured target remainder, computes
one `Picard_ctrunc_normal` image, adds the fixed polynomial-difference
intervals, rejects the step if any component is not a subset, and otherwise
replaces the complete remainder vector with that first image.  Refinement
therefore cannot rescue a failed first self-map.

After acceptance, `Continuous.cpp:1008-1036` repeatedly calls
`Picard_ctrunc_normal_remainder` with the accepted/current remainder vector,
the same candidate polynomial, the same time-step interval, and the
`intermediate_ranges` list produced by the first full Picard evaluation.  The
same fixed `intDifferences` are added on every replay.

`TaylorModel.h:3728-3744` proves that the remainder replay evaluates each ODE
component at effective RHS order `order - 1` and multiplies its result by the
time-step interval exactly once.  `expression.h:1833-1897` proves that a
multiplication remainder is replayed as
`P_left*R_right + P_right*R_left + R_left*R_right`, followed by the saved
truncation/cutoff interval.  The saved polynomial operand ranges are
remainder-independent; remainder-dependent transforms and certificates must
be recomputed.

`include.h:39,49` defines:

- `MAX_REFINEMENT_STEPS = 490`;
- `STOP_RATIO = 0.99`.

The loop counter starts at zero and tests `rSteps <= MAX_REFINEMENT_STEPS`, so
the macro permits at most 491 post-accept replay evaluations.

`Interval.cpp:2982-2998` proves that `old.widthRatio(new)` is
`width(new) / width(old)`, with MPFR upward rounding for both width
subtractions, the division, and conversion to `double`.  A ratio at or below
`0.99` requests another replay.  When both widths are zero, MPFR computes
`0/0 = NaN`; the `<= STOP_RATIO` comparison is false, so it does not request
another replay.  The ratio is a performance stop only; subset is the
soundness condition.

The stock component loop is sequential.  It immediately assigns each
successful component; if a later component fails subset, already visited
components retain their new values while the failing and later components
retain their old values.  Thus stock Flow* can retain a hybrid vector after a
refinement failure.  It does not hull-repair or expand and continue.

## Torch C2 compatibility rule

Torch C2 deliberately strengthens the vector update to an atomic rule.  It
first executes the unchanged C1 self-map.  Only after that accepts does it
compute a complete proposal vector.  It commits the proposal only if every
component is finite and is a subset of the corresponding retained component.
Otherwise it retains the complete previous vector.  After an atomic commit it
continues if any finite component ratio is at or below `0.99`, and it permits
at most 491 replay evaluations.  An equal vector is committed as a fixed point
and stops.  No partial update, hull repair, endpoint repair, sampling
containment, scheduler change, or cross-time remainder assumption is allowed.

Production constants are
`FLOWSTAR_MAX_REFINEMENT_STEPS`, `FLOWSTAR_STOP_RATIO`, and
`FLOWSTAR_REFINEMENT_REPLAY_LIMIT` in `batched_dense_tm.py`.  The opt-in mode is
`flowstar_raw_remainder_compat_factorized_joint_closure_refined`; all default,
legacy, H1, and H2 mode selection remains unchanged.
