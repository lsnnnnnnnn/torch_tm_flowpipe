# Preregistration: complete polynomial normalized carry

Date: 2026-08-10

Primary candidate: F1 only

Status at registration: no F1 implementation exists

## Closest baseline

The closest baseline is the authoritative Torch complete-total-degree O4 lane
with `flowstar_raw_remainder_compat`, physical local time, proactive depth-1
subdivision only on named polynomial-truncation contexts, Flow*-compatible
adaptive steps, constant-center normalized insertion, `h_min=0.002`, fixed
candidate remainder `[-1e-4,1e-4]`, float64, and no repair or fallback.  Its
fresh run from `t=0` validates 307 segments through
`t=6.397083942944808` and rejects the next segment.

## Isolated failure and causal hypothesis

At the first Flow*/Torch schedule split (`t=0.18187433604506256`, proposed
`h=0.019615177354506262`), explicit common-basis comparison shows that the
retained O4 polynomials agree to at most `7.11e-14` coefficient-enclosure
error.  Swapping either polynomial preserves the receiving validator's native
decision.  Flow* is already outside the y target in its raw Picard remainder;
its later polynomial-roundoff interval is only about `[-1.11e-17,4.48e-17]`.
Therefore normalization scale and retained polynomial construction are not the
first schedule-split cause.

The selected candidate addresses the separately quantified later dominant
source allowed by the E-phase gate: cross-step dependency loss.  Time-aligned
Torch/Flow* segment-width ratios exceed 2 near `t=4.205867` and exceed 5 near
`t=6.225303`.  At the terminal Torch rejection, the y ledger includes
`1.0839579370510149e-4` integration-overflow width and
`8.131373677071774e-5` polynomial-truncation width; the subset margin is
`-1.99995911680722e-5`.  The hypothesis is that replacing the validated
endpoint polynomial by a fresh independent center/scale identity before the
next Picard solve destroys useful Q2/Q3/Q4 correlations and drives this later
growth.

## Single changed variable

Only the next-step initial Taylor model changes:

- baseline: fresh physical `center + diagonal_scale * u`, with prior nonlinear
  endpoint structure retained only in the output/right-map path;
- candidate: the complete validated fixed-time endpoint Taylor polynomial and
  its certified remainder, still over the same bounded normalized uncertainty
  variables, become the next Picard initial model.

ODE, initial set, order/support, local-time convention, candidate remainder,
validator, range policy, step controller, `h_min`, cutoff, dtype/device, output
semantics, and right-map reporting stay fixed.

## Soundness argument

The accepted endpoint Taylor model encloses every state at the fixed endpoint.
Using that same enclosure as the next initial set is inclusion-preserving; no
coefficient or remainder is discarded by the carry boundary.  The next local
time variable is appended exactly as in the baseline.  All later truncation,
cutoff, interval, and subset checks remain unchanged.  Any endpoint time
substitution loss stays in the named remainder ledger.  The lane fails closed
if dimensions, domains, support order, dtype/device, or finite bounds disagree.

The implementation must expose, per boundary, the complete retained support,
retained coefficient hash, remainder bounds, time-variable substitution,
transformed-term count, and intervalized-term/remainder count.  No VDP state
index or expression may occur in the kernel.

## Minimal paired experiment

1. Analytic constant, affine, quadratic/Riccati, and harmonic-oscillator
   endpoint-to-next-step inclusion tests.
2. A deliberately correlated endpoint where fresh box reset loses a quadratic
   relation; verify candidate containment and that the relation is retained.
3. Frozen VDP causal checkpoint at unchanged pre-state and h.
4. Frozen terminal checkpoint at `t=6.397083942944808` and its unchanged h
   attempt sequence.
5. Independent fresh requests for T=0.1, 0.5, 1, 4, 6, 6.5, 7.5, and 10.

Every result is paired against the closest baseline with identical settings
except the declared carry mode.

## Primary metric and promotion threshold

Primary metric: highest continuously validated VDP time in a fresh request.

Promotion requires all soundness/equivalence gates and an increase of at least
`0.5` over `6.397083942944808`, i.e. validation through at least
`6.897083942944808`, without changing `h_min`, order, candidate remainder,
initial set, validator, or step rule.  Closing the frozen failed step is also
reported but does not replace the preregistered primary metric.

## Regression budget and stop condition

- zero analytic containment failures;
- zero loss of a previously passing short-horizon certificate;
- no fallback or repair;
- at most 2x batch-1 warm core time;
- finite, explicitly bounded memory growth;
- no benchmark-specific math;
- stop and keep the lane experimental if the primary threshold is missed;
- stop immediately on an inclusion failure, domain/coordinate ambiguity, or
  inability to reproduce the baseline control.

No second primary candidate will be implemented unless this full matrix is
complete and a distinct evidence-supported blocker is isolated.

## Expected tensor shape and complexity

For batch B, state dimension S, n carried uncertainty variables, and complete
degree p, the carried endpoint coefficient tensor is
`[B,S,binom(n+p,p)]`, with remainder bounds `[B,S,2]` and domain bounds
`[B,n,2]`.  No new polynomial slots are added.  Boundary storage is
`O(B*S*binom(n+p,p) + B*n)` and the carry assignment is linear in that storage;
the unchanged Picard algebra retains its existing complexity.
