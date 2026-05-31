# Algorithm notes

This repository implements a minimal PyTorch-native Taylor-model flowpipe kernel
for polynomial plant ODEs.  It is a research prototype for the plant-side piece
of an NNCS reachability pipeline, not a rewrite of Flow* and not a CROWN-Reach
integration.

## Taylor model representation

A scalar Taylor model is represented as

```text
TM = p(vars) + I
```

where `p` is a sparse bounded-degree multivariate polynomial and `I` is an
interval remainder.  A `TMVector` is a vector of scalar Taylor models sharing a
common domain.  The state variables in the initial set are represented as
identity Taylor models over the initial interval box.

## One-step flowpipe

For a polynomial ODE

```text
dx/dt = F(x, u)
```

the step constructor adds a local time variable `tau in [0, h]` and constructs a
Picard polynomial iterate

```text
P_F(g)(x0, tau) = x0 + int_0^tau F(g(x0, s), u) ds.
```

After the polynomial part is constructed, a validation loop inflates the interval
remainder until the Picard residual is contained by the candidate Taylor model
remainder.  On validation, the returned `FlowpipeSegment.tm` is a segment over
`(original variables, tau)`.

## Endpoint tightening

The full segment remainder must hold for every `tau in [0, h]`.  For multi-step
propagation, only the endpoint at `tau=h` is passed to the next step.  The code
therefore re-evaluates the Picard residual at `tau=h` to obtain a tighter
`final_tm` remainder than the all-times segment remainder.

## Multi-step modes

`flowpipe_multi_step(..., mode="range_only")` reproduces the baseline behavior:
each step compresses the previous final Taylor model into an interval box and
restarts with fresh identity variables.

`flowpipe_multi_step(..., mode="dependency_preserving")` keeps the previous
`final_tm` as the next step's initial condition.  Each step introduces a fresh
local time variable, substitutes `tau=h` at the endpoint, then drops that local
variable.  As a result, each final Taylor model depends only on the original
initial-state variables and not on stale local time variables.

## Controls

Two lightweight control representations are supported:

* `u_box`: constant interval controls.
* `affine_u`: controls of the form `u = A x0 + b + error`, where `error` is an
  interval or radius.
