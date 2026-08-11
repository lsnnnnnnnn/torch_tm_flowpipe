# Flow* / Torch complete-O4 matched comparison

Date: 2026-08-11

## Outcome

Pairwise outcome: `PAIRWISE_COMPARISON_PARTIAL`.

## Eligibility

Same-prestate B1 O4 coefficient/raw-remainder/validator facts are eligible as
matched diagnostics; stock Flow* is not a primary formal oracle.

## What is comparable

Common-basis coefficients, frozen raw candidates, subset margins, and the
lossless 2x2 receiving-validator matrix.

## What is unavailable

Native T10 endpoint/tube tightness, later lossless common producer states, and
a matched timing ratio.

## Negative results

Polynomial roundoff, common-basis error, and receiving-predicate identity do
not explain the split.

## Exact evidence paths

New-package directories `07_flowstar_torch_raw_remainder/` and
`08_schedule_validator_matrix/`; frozen common-basis evidence remains under
`outputs/mainline_realignment_20260810/20260810T025910Z/03_flowstar_causal_divergence/`.

## Identity and eligible scope

Flow* is pinned at `b85a3211748cb77b736fe4ad42ee02d8d2b81148`.
The Torch expression-trace implementation is
`a31fcac7e97f4783d919951c7714b69c6eabda1b`.  The common model is
`x'=y`, `y'=y-x-x^2*y`, initial box `[1.1,1.4] x [2.35,2.45]`, B1,
total-degree O4, and target remainder `[-1e-4,1e-4]`.

The read-only semantic probe binary SHA256 is
`006b65d163ab73ff9757b1860bafedb3473bc596085f1dc85157151ab4c5d771`.
The earlier official observer binary SHA256 is
`6e4d4af60154239d7f281c367337f6ff52958ed146fb3ac1956b104b31e7f2ba`;
its logged/unlogged official plots and 290-row schedule are byte-identical.

## Common basis and first schedule split

The affine common-basis map accounts for centers, scales, local time, and
exponent order.  At the causal candidate, maximum coefficient midpoint errors
are `8.88e-16` (x) and `1.421e-14` (y), and maximum interval enclosure errors
are `1.421e-14` and `7.105e-14`.  These errors are below the decision margins
and do not explain the split.

The first split follows the common accepted prestate at
`t=0.18187433604506256`.  For proposed
`h=0.019615177354506262`, Torch accepts while Flow* rejects and later accepts
`h=0.0098075886772531311`.

| producer | raw x | raw y | minimum margin | decision |
|---|---|---|---:|---|
| Flow* | `[-5.3757103669508146e-6,7.073210412034566e-6]` | `[-1.0366239882151062e-4,1.0359846643018429e-4]` | `-3.662398821521699e-6` | reject |
| Torch | `[-1.961520735450628e-6,1.961520735450628e-6]` | `[-9.14291532216261e-5,9.358938647674799e-5]` | `+6.4106135232520195e-6` | accept |

The expression-tree root cause is Picard iteration 4 at `x*x`: Flow* direct
`TaylorModel<Interval>` multiplication adds retained-coefficient interval
uncertainty of
`[-0.00011204861774257546,0.00008935810062010431]`.  Replacing only that
contribution, with every later input frozen, changes the Flow* y margin to
`+2.4888083156873676e-7`.  Independent MPFR and exact-rational frozen-input
replays contain the reported nodes; they do not prove the replacement sound
for arbitrary MPFR coefficients.

## Validator and schedule counterfactuals

The same-prestate 2x2 matrix is lossless.  Both receiving subset predicates
accept the Torch candidate and reject the Flow* candidate with identical
margins.  The decision therefore follows candidate construction, not the
receiving predicate.  Torch also completes all 51 Flow* accepted steps to T1,
and both tools complete the fixed `h=0.01` T1 diagnostic.

After the first different accepted step the producer states have different
histories.  Later common-time candidate comparisons are unavailable unless a
new state object is introduced; no box hull is used as a substitute.

## Available and unavailable geometric comparisons

Same-prestate candidate coefficients, raw remainders, subset margins, and the
receiving-validator matrix are available.  Common-time endpoint or segment
tube widths are eligible only where both producers represent the same object.
The native full-horizon rows do not satisfy that condition: stock Flow*
completes T10 with segment tubes, while authoritative Torch stops at
`6.397083942944808` and separately records endpoint, last-segment tube, and
prefix tube.  Therefore `TIGHTNESS_COMPARISON_UNAVAILABLE` for a native T10
ratio.

The pinned stock Flow* build also reproduces a scalar-affine under-enclosure,
so it is ineligible as a primary formal oracle for tightness.  This qualifies
the build/workload rather than the abstract Flow* algorithm.

No timing ratio is reported.  Existing runs have different logging,
adaptivity, process/core, and soundness scopes and were not a matched
one-cold/ten-warm timing experiment.
