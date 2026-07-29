# Native representation semantics

## Torch TM

Torch propagates a sparse multivariate polynomial with a complete total-degree
ceiling plus one independent interval remainder per state. In a local segment,
the dependency variables from the incoming set are followed by local time
`tau`. Multiplication retains every monomial within the selected total degree;
discarded products, cutoff terms, and polynomial/remainder interactions enter
the interval remainder. The growth validator seeks a Picard self-map in
float64 interval arithmetic. `endpoint_raw_tm` is direct `tau=h` substitution.
The legacy fixed-time residual recomputation is exported only as the distinct
supplemental `endpoint_tightened_tm`.

## DiffReach

DiffReach's plant polynomial is

`P(t,z) = c + L z + t (Lt z)`, with `z = [t, xi_1, ..., xi_n]`.

Thus the default restricted quasi-quadratic basis contains constants, affine
terms, `t^2`, and `t*xi_i`, but no general `xi_i*xi_j`. The affine flag changes
the multiplication/truncation path and removes `Lt`; it is not the same basis
as Torch complete degree one during construction. Native carry composes the
local model with a normalized symbolic parameterization and a finite symbolic
remainder window. The study forces JAX x64 and records that this is an
experiment-local dtype correction around an upstream float32 default.

## Flow*

Flow* propagates a composition of two Taylor-model vectors: `tmvPre` contains
the local preconditioned flow map and `tmv` maps normalized generators from the
previous flowpipe. `Flowpipe::compose` is the authoritative physical-state
expansion. Fixed order 2 is the minimum legal setting in this checkout and
retains a complete multivariate basis. Native interval arithmetic uses MPFR/GMP
and directed rounding. Flow* additionally exposes normalized composition, QR
preconditioning, symbolic remainders, and adaptive step/order policies.

Consequently, “order 1” is neither a common basis nor a common carry contract.
The closest controlled replacement is a validated native local construction
followed by the same sound affine endpoint projection.
