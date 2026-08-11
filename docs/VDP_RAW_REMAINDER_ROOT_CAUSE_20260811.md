# VDP raw-remainder root cause

Date: 2026-08-11

## Outcome

`RAW_REMAINDER_ROOT_CAUSE_CLOSED`

## Eligibility

Eligible for the frozen B1 checkpoint and expression node; not a universal
Flow* coefficient-soundness claim.

## What is comparable

Same-checkpoint Picard-4 `x*x` multiplication sources, raw y image, target,
and counterfactual margin.

## What is unavailable

Soundness of the narrower replacement for arbitrary MPFR coefficients and
end-to-end formal solver equivalence.

## Negative results

Polynomial roundoff and earlier dropped/product-remainder differences do not
cause the decision.

## Exact evidence paths

New-package directory `07_flowstar_torch_raw_remainder/`, including the common
tree, node CSV, first divergence, counterfactuals, and MPFR/Fraction replay.

At the frozen last-common prestate (`t=0.18187433604506256`) and full
proposal (`h=0.019615177354506262`), the first decision-changing expression
divergence is Picard iteration 4, component `y`, semantic node `x*x`.
Flow*'s direct `TaylorModel<Interval>` multiplication adds retained-coefficient
interval uncertainty that is absent from Torch's point-binary64 retained
coefficient path.

The Flow* production `x*x` remainder is
`[-0.0005719067724810592, 0.0005134076675924922]`.  Replaying the four
ordinary multiplication sources (polynomial/remainder in both directions,
remainder/remainder, and dropped polynomial support) gives
`[-0.00045985815473848396, 0.00042404956697238806]`.  Their difference,
`[-0.00011204861774257546, 0.00008935810062010431]`, is the retained
coefficient-interval uncertainty.

## Decision counterfactual

The production Flow* `y` image has margin
`-3.662398821521699e-06` and rejects.  Holding the frozen inputs and all later
outer-multiplication uncertainty fixed, replacing only the direct `x*x`
coefficient-uncertainty contribution produces margin
`+2.4888083156873676e-07` and accepts.  Removing polynomial-roundoff padding
alone does not change the decision.

Earlier unequal sources do not explain the rejection.  In particular, the
dropped polynomial-product interval is wider in Torch by about
`1.507136216059152e-04`; its direction is opposite to the observed Flow*
rejection.  The two polynomial/remainder terms differ by only
`2.3858990635548718e-08` in width each.

## Independent checks and scope

An independent 256-bit MPFR DAG evaluator, which does not call either
production operator, verifies containment for the Flow* and Torch `x*x`
formulas, the cached Flow* raw image, and the one-node counterfactual.  An
exact `fractions.Fraction` replay separately sums the frozen finite binary64
component endpoints.  Observer-enabled and observer-disabled Torch images are
bit-exact; the Flow* official plot files and accepted schedule are unchanged
by the read-only observer.

This result identifies the decision-changing representation/operation.  It
does **not** prove that the narrower cached formula is sound for arbitrary
Flow* MPFR coefficients, nor does it qualify either complete solver as a
universal formal oracle.  The Fraction result is limited to the frozen finite
binary64 component endpoints.

Machine-readable evidence is emitted by
`experiments/trace_vdp_raw_remainder.py` and
`experiments/analyze_vdp_raw_remainder_trace.py` as the common expression
tree, node comparison, first divergence, counterfactuals, and independent
replay artifacts.
