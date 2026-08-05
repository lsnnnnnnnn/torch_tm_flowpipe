# Van der Pol terminal polynomial-range closure

## Pre-registered change proposal

- **Closest baseline:** the natural-interval `hybrid_dense_core` lane at
  `82c54a244d996ccc08b09cb4ded5f48167415585`, validated with
  `flowstar_raw_remainder_compat` and sparse normalized-insertion carry.
- **Observed failure:** a fresh CPU float64 run accepts 308 segments through
  `t=6.3172908799330765`, then rejects the unchanged
  `h=0.0039859994324420315` attempt with y target-subset margin
  `-5.111670937766742e-6`. This is a certificate/self-map failure, not a
  timeout or nonfinite failure.
- **Causal hypothesis:** dependency loss in natural interval evaluation of
  grouped high-degree polynomial contributions makes the terminal y remainder
  image too wide. Subdivision of the original merged coefficient/exponent
  polynomial before intervalization can tighten that image while preserving a
  conservative cover.
- **Minimal paired experiment:** replay one frozen terminal pre-state at the
  identical attempted h using A0 natural and pre-registered A1--A4 subdivision
  caps of 4, 8, 16, and 64 leaves. ODE, coefficients, exponent support, order,
  cutoff, target remainder, Picard count, validation predicate, endpoint
  semantics, dtype, and device remain fixed.
- **Primary metric:** unchanged-gate terminal y subset margin, followed only
  after local promotion by the highest fresh validated horizon.
- **Acceptance threshold:** the method must pass the complete range-operator
  correctness gate and make every terminal self-map and target margin
  nonnegative at the original h. A fresh run must strictly exceed
  `6.390931109681597` before long-horizon promotion.
- **Regression budget:** the default natural lane and its short schedule may
  not change; no coefficient, contract, cutoff, remainder, endpoint, repair,
  fallback, or finite-status regression is permitted. Subdivision is capped at
  64 leaves.
- **Stop condition:** stop subdivision if the validated 64-leaf frozen replay
  still rejects. Enter deterministic Horner/factorized evaluation only under
  that declared contingency. On later fresh failure, permit at most one
  evidence-driven cap adjustment.
- **Independent checks:** analytic interval cases, complete-cover invariants,
  sparse split parity on the same merged polynomial/domain, deterministic
  randomized sample-containment sanity checks, and CPU/CUDA parity where CUDA
  is available. Sampling is not treated as proof.

The intended arithmetic claim is a safeguarded float64 enclosure, not a fully
machine-checked directed-rounding proof. The backend remains
`hybrid_dense_core`; full-dense cross-step composition is outside this work.

## Validated result

Numerical implementation commit `24bd6524bf751662afdd613f294678a0a364bdaa`
and phase-timing commit `0360842` advance the
previous S3 backend through R4 (`historical_range_midpoint_horizon_crossed`).
It does **not** complete T=7.5 or T=10 and therefore does not claim R5, R6, or
R7.

The immutable numerical contract remained order 4, float64, `h_min=0.002`,
`h_max=0.1`, cutoff `1e-10`, target remainder radius `1e-4`, normalized
insertion, constant right-map centering, standard right-map range,
`flowstar_compat` scheduling, and `flowstar_raw_remainder_compat` validation.
There was no endpoint repair, endpoint tightening, sample hull, hidden sparse
fallback, changed ODE, changed validation predicate, or lowered gate.

### Frozen replay and checkpoint

The natural baseline was reproduced fresh from t=0 and stopped after 308
accepted segments at `6.3172908799330765`. Its terminal attempt used
`h=0.0039859994324420315`, coefficient hash
`da9db6557b41d24496f261854c6c75a58fd67b5ab1a09b1846c6c21d39d9de1f`,
and exponent-support hash
`d0aa354b9057267556d5bb3bc09a36ed4162b36fb44588b0b930dd9e935041e9`.
The x/y subset margins were respectively `9.96013970567558e-5` and
`-5.111670937766742e-6`.

The frozen pre-state uses canonical JSON with hexadecimal float payloads and
no pickle. Its full checkpoint SHA256 is
`a8d73320c8343ad815edd36e04c228f0693c6e78dda40a216c790fda1d45a343`.
Save/load/save is byte-exact, state/domain/coefficient/support hashes are
checked, and corrupt payload, corrupt manifest, contract, order, and dtype
mismatches fail closed. Natural replay reproduces the rejection, margins,
ledger, candidate remainder, Picard count, and hashes exactly.

### Subdivision range operator

The dense evaluator first merges equal exponents, constructs a complete
owner-indexed cover, evaluates all flattened batch/leaf pairs with tensor
operations, and hulls leaves back to their owners. Depth 1 bisects `u0` and
`u1` into four leaves. Later levels choose the active variable by deterministic
`domain width × derivative magnitude`, with a hard cap of 64 leaves. `tau` is
not split in this result.

Natural and subdivision bounds are both retained in every range trace. A
component selects subdivision only when it is no wider; otherwise the audited
natural result is selected. Powers, monomials, coefficient products,
reductions, and the final hull retain the existing outward safeguards and
gamma-n reduction envelope. Any invalid cover or nonfinite leaf fails closed.

The operator gate covers analytic constants/affine/odd/even/cross-zero/mixed
cases, degree 12, shifted/narrow/zero-width boxes, randomized sample
containment, dense/sparse leaf parity, deterministic CPU, batch 1/16/48, and
CUDA batch 1/16/48 parity. The validated claim remains a safeguarded float64
enclosure, not a machine-checked directed-rounding proof.

### Original terminal A/B

| Lane | Leaves per range call | Accepted | y margin | Runtime (s) |
|---|---:|---:|---:|---:|
| A0 natural | 1 | no | `-5.111670937766742e-6` | 0.1043 |
| A1 fixed depth 1 | 4 | yes | `2.888631469865542e-5` | 0.2069 |
| A2 max 8 | 8 | yes | `2.888631469865542e-5` | 0.2283 |
| A3 max 16 | 16 | yes | `2.888631469865542e-5` | 0.2831 |
| A4 max 64 | 64 | yes | `2.888631469865542e-5` | 0.5282 |

All lanes use the same attempted h and coefficient/support hashes. The focused
`polynomial_truncation` lane alone closes the step with y margin
`2.8883253329832075e-5`, using 14 range calls and 56 leaf evaluations. The
`integration_overflow`-only lane does not change the rejection, and the
polynomial/remainder product lane changes the terminal margin only from
`-5.111670937766742e-6` to `-5.10852777951499e-6`. Accordingly, the production
policy names only `polynomial_truncation`.

The fail-trigger production replay first records the exact natural failure,
then accepts the unchanged attempt at depth 1. This is R3.

### Fresh progression and the one policy adjustment

The initial `on_validation_failure` fresh lane completed T=4 and T=6, but its
T=6.5 request stopped at `6.3497830056387565`. Frozen replay of that new state
improved the y margin from `-7.567393490877273e-5` to
`-1.7639286070320263e-5` at depth 1; depths through the 64-leaf cap produced no
further improvement.

The single permitted evidence-driven adjustment was therefore the
pre-registered `proactive_depth1_on_named_contexts` policy: every
`polynomial_truncation` range from t=0 uses at most four leaves so a smaller
remainder can be retained before carry accumulation. No second production
adjustment was made.

| Fresh request | Status | Validated horizon | Accepted steps | Runtime (s) | Range calls | Leaf evaluations |
|---|---|---:|---:|---:|---:|---:|
| T=0.1 | completed | 0.1 | 7 | 2.6696 | 98 | 392 |
| T=0.5 | completed | 0.5 | 31 | 14.1344 | 434 | 1,736 |
| T=1 | completed | 1.0 | 50 | 25.1899 | 700 | 2,800 |
| T=4 | completed | 4.0 | 138 | 95.7752 | 1,932 | 7,728 |
| T=6 | completed | 6.0 | 234 | 203.5956 | 3,276 | 13,104 |
| T=6.5 | failed | `6.397083942944808` | 307 | 316.7679 | 4,312 | 17,248 |
| T=7.5 | failed | `6.397083942944808` | 307 | 323.0360 | 4,312 | 17,248 |
| T=10 | failed | `6.397083942944808` | 307 | 315.2670 | 4,312 | 17,248 |

The tighter schedule first diverges from natural at segment 12 and
`t=0.18187433604506256`: natural halves the attempted
`0.019615177354506262`, whereas the tighter validator accepts it unchanged.
At T=1 the proactive runtime is 1.0224× natural (25.1899 s versus 24.6378 s).

The synchronized eager range diagnostic covers CPU/CUDA, batch 1/16/48, and
4/16/64 leaves. All 18 cases are finite and coverage-valid, with setup, first
call, two warm-ups, and ten steady calls recorded separately. At batch 1 / 4
leaves, steady median wall time is 2.691 ms CPU versus 6.415 ms CUDA; at batch
48 / 64 leaves it is 269.226 ms CPU versus 564.302 ms CUDA. CUDA makes the
leaf evaluation itself faster in the latter case (2.246 ms versus 33.933 ms),
but cover construction and independent coverage validation remain
host-oriented. No end-to-end GPU speedup is claimed. The path is eager, so
compile time is explicitly recorded as not applicable rather than folded into
warm-up.

All fresh runs have zero sample-sanity violations, zero fallback, zero repair,
zero endpoint tightening, zero device transfers, finite values, and
`hybrid_dense_core` identity. The T=6.5, T=7.5, and T=10 requests independently
reproduce the same segments, attempts, full ledger, terminal coefficient hash,
and terminal support hash.

The final terminal is `t=6.397083942944808` with attempted
`h=0.003623635847674574`, coefficient hash
`bc1433d0d3c89339fca6091e41c0a6667d70c92d2dd4e35ae8b14236d131863c`,
and x/y margins `9.963763341523255e-5` and
`-1.99995911680722e-5`. Replays at depth 1/2/3/5 use 56/112/224/896 total leaf
evaluations and produce the same negative y margin. Thus R4 is the highest
result and the remaining blocker is the unchanged raw-remainder self-map,
not the subdivision or runtime cap.

## Tests and evidence

The implementation baseline was 343 passed and 2 skipped. Final validation is
400 passed and 2 skipped; the dedicated CUDA selection is 3 passed. The tracked
evidence bundle is
`evidence/vdp_terminal_range_closure/20260805T055556Z`. It contains complete
segments, attempts, ledger, and range traces (large raw files are deterministic
gzip), checkpoint payloads, A0–A4 and attribution lanes, all fresh horizons,
runtime tables, the R4 decision, a manifest, and SHA256SUMS.

Because the T=10 request failed, the success-only second T=10 reproduction was
not run and R7 is false.
