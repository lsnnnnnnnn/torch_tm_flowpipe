# Flow*–Torch carry source map — 2026-08-13

Corrected status: `SOURCE_MECHANISM_CANDIDATES_LOCALIZED_CAUSAL_SPLIT_OPEN`.

The previous `SOURCE_LEVEL_DEPENDENCY_LOSS_LOCALIZED` label is a historical,
superseded overclaim.  The map below was human-authored from source inspection;
it identifies candidates but does not prove runtime-path equivalence or that a
candidate caused the long-horizon gap.  See the 2026-08-13 causal-factor and
lossless-bridge reports for the actual-path counterfactual evidence.

## Stage map

| mathematical stage | Flow* source and representation | Torch source and representation | first unequal? | dependency consequence |
|---|---|---|---|---|
| benchmark/model | `vanderpol.cpp:7-89`; `ODE<Real>`, `Flowpipe`, queue size 100 | `run_vdp_dense_backend.py:488-533`; dense complete-O4 state | no | Same ODE, box, order, target, cutoff, and fixed schedule. |
| fixed-step loop | `Continuous.h:832-895`; accepted chain with persistent queue | `flowpipe.py:5170+`; `FlowstarNormalFlowpipeState`, legacy queue disabled | not at initial set | Feature choice becomes active after the first boundary. |
| carry decomposition | `Continuous.cpp:2151-2177`; linear `Phi_L/J` plus `x0_other` | `flowpipe.py:1470-1511`; remove constant, insert the entire remaining TM | yes | Flow* propagates each linear old source once; legacy Torch has no corresponding source identity. |
| normal composition | `TaylorModel.h:4213-4243`; recursive Horner insertion | `flowpipe.py:698-739`; build every monomial independently by repeated multiplication | yes | The same already-intervalized right-map remainder is materialized in separate monomial paths. |
| TM multiplication | `TaylorModel.h:797-866`; `P1I2+P2I1+I1I2` plus truncation/cutoff | `taylor_model.py:276-285`; the same local interval formula | no in primitive | Both locally intervalize; call grouping and cross-step source bookkeeping determine repeated cost. |
| Picard/validator | `Continuous.cpp:2328-2410`; expression Picard and subset/refinement | `batched_dense_tm.py:2750-3030`; dense Picard plus remainder ledger | downstream | Both validators consume carry states that already differ. |
| endpoint/tube | `Continuous.cpp:415-454`; composed accepted Flowpipe | `flowpipe.py:5040-5147`; published accepted TMs | downstream | Both include polynomial and stored remainder; no zero-fill explains the delta. |
| serialization/join | probe `:400-408,613-624,695-724`; 17 digits | comparator strict row/time join | no | The positive widths and first difference survive output exactly. |

## First differences

The initial box is a genuinely common prestate. The first published bitwise
difference is accepted step 1. It is already larger than plausible operation
ordering noise: endpoint-width differences are `2.14e-4` and `1.30e-3`, while
the two segment-width differences are `1.86e-6` and `2.16e-5`.

Step 1 then changes the state used for step 2. Flow* step-2 prestate scales are
approximately `(0.150449660092…, 0.060913584414…)`; Torch's are
`(0.150450598494…, 0.060924149581…)`. From step 2 onward the Flow* `Phi_L/J`
queue is active, whereas the frozen legacy Torch path remains plain
`normalized_insertion`. This is the first decision-relevant difference because
it alters subsequent normalization, polynomial-times-parameterization-remainder
terms, and eventually validator margin.

The historical source-level hypothesis was:

> In Flow* `Continuous.cpp:2151-2177 Flowpipe::advance`, the linear portion of
> each old remainder source remains associated with its `Phi_L/J` queue entry
> and is propagated once through the accumulated matrices; nonlinear
> `x0_other` is composed in `TaylorModel.h:4213-4243` using Horner grouping.
> Legacy Torch at `flowpipe.py:1470-1511` sends the full constant-removed TM to
> `flowpipe.py:698-739`, where independent monomial construction repeatedly
> consumes the same intervalized right-map remainder. The resulting step-1
> width difference changes step-2 scales and the excess accumulates thereafter.

This quoted hypothesis is not a causal conclusion.  Step 1 precedes old-source
queue propagation and already contains local Picard coefficient differences.
Flow*'s multiply primitive also intervalizes remainder interactions.

## Same-prestate gate

The probe's canonical coefficient exporter ultimately uses Flow*
`Real::toString()` (`Interval.cpp:366-374`), which converts to double and prints
15 scientific digits. It neither exports a lossless binary coefficient image
nor a complete importable `Phi_L/J` queue. Therefore a post-step 2×2
`F(F-state), T(F-state), F(T-state), T(T-state)` attribution cannot be performed
without changing semantics. No lossy adapter is used or reported as
same-prestate.

The fallback consists of exact-rational affine, quadratic shared-error
cancellation, and cubic `x²y` shared-source fixtures. All independently
intervalized results contain the shared-source range; the quadratic and cubic
fixtures also show strictly positive width excess. These fixtures validate the
local dependency phenomenon, but they are not a proof of a complete O4 carry
implementation.

## Why early and late widening differ

At T=1, all four Torch excesses are already positive (`0.00272`–`0.00890`), and
at T=3 they are about `0.0470`–`0.0488`. This is consistent with repeated
monomial-path intervalization and the missing linear source queue accumulating
from the first boundary. By 6.32 the scales and remainders feed a nonlinear
`x²y` Picard map over an already wider normalized state, producing excesses
`0.763`–`1.468` and a failed next fixed-step containment. The late acceleration
is feedback through a widened prestate; it is not caused by the independent
Flow* width minimum or by a parser denominator.
