# Torch single-improvement result

> Superseded bridge decision. The current full-horizon carry result is
> `NO_FIX_AUTHORIZED`; no new carry implementation was made.

Date: 2026-08-11

## Outcome

`IMPROVEMENT_NOT_AUTHORIZED_BY_EVIDENCE`

## Eligibility

This is a fail-closed promotion decision; no production semantic change is
eligible.

## What is comparable

Baseline and frozen root-cause counterfactual margins at the first divergent
node.

## What is unavailable

P0--P7 promotion results and a promoted second-system result because no
candidate is authorized.

## Negative results

The evidence identifies extra Flow* uncertainty, not a Torch defect or a
generally sound narrower Flow* operation.

## Exact evidence paths

`outputs/three_tool_matched_divergence_fixed_support_20260811/20260811T100304Z/`
directories `07_flowstar_torch_raw_remainder/`, `10_bridge_ladder/`, and
`12_single_improvement/`.

The raw-root-cause and schedule studies closed, but the descriptor bridge is
`FIXED_SUPPORT_BRIDGE_BLOCKED` and identifies no universal dominant factor.
The evidence therefore does not authorize a production Torch semantic change
at the earliest proven divergence.  That divergence is
extra retained-coefficient interval uncertainty in Flow*'s direct
`TaylorModel<Interval>` `x*x` multiplication.  Torch's point-binary64 path
does not contain that extra contribution, already has the narrower accepted
candidate, and its frozen-input result contains the independent MPFR replay.

Replacing the Flow* direct interval result with its cached component formula
changes the decision, but the evidence explicitly does not prove that
replacement sound for arbitrary Flow* MPFR coefficients.  Porting such a
replacement into Torch would neither fix a demonstrated Torch defect nor
satisfy the requirement that the Torch root-cause node width/margin improve
without lowering soundness eligibility.

The fixed-support A0–A4 bridge is an experimental representation/validator/
carry ladder, not a promoted change to the authoritative complete-O4 lane.
Its preregistered T1 metrics show B-dependent multi-factor effects, and its G3
failures do not isolate one universally dominant Torch factor.

Accordingly no target, cutoff, Picard depth, minimum step, validator, carry,
or complete-O4 production operation was changed for promotion.  P0–P7 and the
second-system extension were not run because there is no evidence-authorized
candidate.  Existing S1 and complete-O4 baseline results remain historical
and unmodified.
