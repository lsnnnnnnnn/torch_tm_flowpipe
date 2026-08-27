# Generic accepted-boundary symbolic remainder

The C3 mechanism is split into two layers:

1. `src/torch_tm_flowpipe/accepted_boundary_sr.py` is the plant-independent
   primitive. It consumes a constant-free accepted endpoint Taylor map, its
   computed linear/nonlinear split, the previous normalized right map, the
   normalization scales, the explicit owner generation/boundary, and the queue
   state. It imports no ODE or benchmark.
2. `_vdp_frozen_c3_accepted_boundary_prepare` in `flowpipe.py` is the frozen VDP
   wrapper. It retains the published order 4, cutoff `1e-10`, capacity 100, and
   standard/constant normalization contract before invoking the generic core.

The generic public reset mode is
`normalized_insertion_dependency_preserving_generic_sr`. Unlike the VDP wrapper,
it does not fix state dimension, polynomial order, cutoff, or queue capacity.

## State and commit semantics

- CPU float64 interval operations are outward; nonfinite or non-CPU/float64
  inputs fail closed.
- `prepare_accepted_boundary_sr` creates an immutable shadow transition. It
  cannot mutate the accepted queue, so reject/retry is rollback-safe.
- `commit_accepted_boundary_sr` advances generation and boundary exactly once,
  adds one owner, and resets only after that accepted commit reaches capacity.
- On a capacity reset, history has already been materialized in the accepted
  inserted map. The empty queue forces the following boundary through the full
  re-anchor branch, so history is neither lost nor retained as a second owner.
- Coefficient multiplication/addition roundoff, normalization roundoff, cutoff,
  nonlinear composition remainder, and transported history remain separate
  outward charges until the accepted owner commit.
- Checkpoint v5 serializes either the generic owner schema
  `accepted_boundary_sr_v1` or the compatibility schema `c3_cross_step_sr_v1`.

## Verification

`tests/test_accepted_boundary_sr.py` covers exact `Fraction` images in 1D, 2D,
and 3D; atomic capacity reset and no lost/double-paid history; generic
checkpoint/resume; nonfinite/dtype fail-closed behavior; and a non-VDP scalar
polynomial flowpipe with order 3, cutoff `1e-12`, and capacity 2.

The frozen VDP three-step replay is bit-identical to the pre-refactor raw run in
all non-timing segment fields. The Phase-1 full suite result is 877 passed and 2
skipped.
