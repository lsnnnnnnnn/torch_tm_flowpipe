# VDP schedule/validator causality

Date: 2026-08-11

## Outcome

`SCHEDULE_VALIDATOR_INTERACTION`

## Eligibility

Eligible as a frozen same-prestate/schedule diagnostic, not as a formal
ranking row.

## What is comparable

The two candidate producers under both identical componentwise subset
predicates, plus separate T1 schedule replays.

## What is unavailable

Lossless post-split common producer states and terminal cross-schedule
counterfactuals.

## Negative results

The receiving validator identity alone is not causal at the first split.

## Exact evidence paths

`outputs/three_tool_matched_divergence_fixed_support_20260811/20260811T100304Z/08_schedule_validator_matrix/`,
produced by `experiments/run_vdp_schedule_validator_matrix.py`.

At the last-common prestate and the same proposed
`h=0.019615177354506262`, the 2x2 receiving-validator matrix is lossless:
both validators apply the same componentwise closed-interval subset predicate
to `[-1e-4,1e-4]`.

| candidate producer | Flow* receiving predicate | Torch receiving predicate |
|---|---:|---:|
| Torch complete O4 | accept; minimum margin `6.4106135232520195e-06` | accept; same margin |
| Flow* complete O4 | reject; minimum margin `-3.662398821510613e-06` | reject; same margin |

Thus the first split follows candidate construction, not which of the two
subset predicates receives the candidate.  Flow* rejects the full proposal
and accepts a shorter step, while Torch accepts the full proposal.  That
schedule choice then changes the next producer state, so later schedule and
candidate-construction effects cannot be separated losslessly.

Two forward diagnostics corroborate the interaction without replacing the
same-prestate proof:

- Torch replayed all 51 Flow* accepted step sizes through T1 and completed at
  `0.9999999999990001`.
- Under common fixed `h=0.01`, Flow* and Torch both completed 100 steps to T1.

The first post-split common time, a later near-T1 state, and the historical
terminal state are marked `NOT_MATHEMATICALLY_EXPRESSIBLE` for a lossless
cross-producer matrix.  A box hull is not substituted because it would change
the candidate object.

The machine-readable matrix and the two schedule replays are produced by
`experiments/run_vdp_schedule_validator_matrix.py`.  These rows are causal
diagnostics only and are not eligible for a cross-tool speed or tightness
ranking.
