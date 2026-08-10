# Structured remainder S1 result (2026-08-10)

## Implemented primitive

`normalized_insertion_structured_remainder_k16` is dimension- and
batch-generic. It stores ordinary intervals, `J`, outward interval `Phi`, an
active mask, age, source ID, and inverse scale. Capacity is frozen at K=16.
Only `polynomial_truncation` and `integration_overflow` are eligible; all other
additive ledger sources remain ordinary. The oldest `(age, slot)` column is
outward-materialized before overwrite. A non-affine update must provide an
explicit structured nonlinear residual.

The dense raw-compatible validator now returns an additive decomposition of
the unchanged checked image: base ledger, time-scaled raw RHS ledger, Picard
residual, and explicit roundoff padding. Conservation, centering without
double count, K1/K16 eviction, affine, harmonic, quadratic/cross nonlinear,
determinism, batch permutation, and fail-closed tests pass. Exact formulas and
the field map are in
[the semantics document](STRUCTURED_REMAINDER_SEMANTICS_20260810.md).

## Frozen terminal result

The closest baseline replay at `t=6.397083942944808`,
`h=0.003623635847674574` reproduces the coefficient contract and rejects with
margin `[9.963763341523255e-5, -1.99995911680722e-5]`. Its additive
decomposition contains the unchanged image.

An empty-history local split is conservative and useful only for attribution:
it leaves x margin unchanged and changes ordinary y margin to
`+9.090310982602511e-5`; full materialization still contains the baseline
image. This is not a candidate A/B result. The immutable checkpoint has no
307-step S1 state, and the immutable first native-split observation has neither
an additive validated ledger nor an S1 prefix state. Therefore prefix
conservation/no-double-count and same-pre-state A/B are not established.

The exact gate result is `STRUCTURED_REMAINDER_LOCAL_GATE_FAILED`. `go=false`
and `fresh_horizon_ladder_authorized=false`; none of T=.1/.5/1/4/6/6.5/7.5/10
was started. The machine ladder records every request as `not_run_after_stop`.
The local numerical improvement is not described as a fix or promotion.

## Precise blocker and next action

S1 remains a qualified primitive rather than an integrated complete-O4 carry.
The next action is to thread its bounded state and typed-source removal through
every accepted boundary from t=0, materialize all columns into endpoint/tube,
and save that prefix state in the terminal checkpoint format. Only then should
the same immutable terminal schedule be replayed again; no horizon request is
authorized before that gate.
