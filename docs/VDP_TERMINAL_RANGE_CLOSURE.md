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
