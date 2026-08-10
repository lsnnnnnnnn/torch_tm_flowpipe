# Fixed-support soundness result (2026-08-10)

The ordinary object, functional, and compiled lanes remain `empirically
sampled only` at multi-step scope. Object/functional bit-exactness is a
semantic equivalence result, not a proof of outward rounding. Inductor changes
arithmetic and is performance-only. The earlier exact-workload two-ULP replay
was not extended to a universal claim.

The implemented CPU float64 reference stores retained coefficients as
intervals and expands every basic product, sequential reduction, polynomial
projection residual, integration, Taylor-model product, range, endpoint, and
tube operation with `nextafter`. Nonfinite work fails closed. Its classification
is `safeguarded outward under declared IEEE/backend assumptions`.

An independent `fractions.Fraction` oracle does not call the expected-value
path through the operators under test. Eleven case families pass: constants,
asymmetric intervals, subnormal, large finite values, cancellation, duplicate
routes, three rational VDP boxes with analytic extrema, scalar Riccati, and a
harmonic affine map. One outward VDP step contains the ordinary endpoint and
tube.

The multi-step result is an implemented negative:

| batch | 1 step | 10 steps | 100 steps | 1000 steps | first failure |
|---:|---|---|---|---|---:|
| 1 | complete | complete | fail closed | fail closed | 33 (zero-based) |
| 64 | complete | complete | not all active | fail closed | 90 first; all inactive by attempt 102 |

Consequently the longest all-batch prefixes are 33 steps for B1 and 90 steps
for B64; neither reaches T1. The outcome is
`FIXED_SUPPORT_FORMAL_SOUNDNESS_NOT_CLOSED`. Ordinary completion is recorded
separately and is never promoted merely because the outward reference contains
an ordinary one-step result.
