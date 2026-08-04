# Generic batched Taylor-model backend implementation

The canonical dense implementation is
`src/torch_tm_flowpipe/batched_dense_tm.py`; no parallel dense solver was
introduced.  Commit `6bf0d9a6b02e3dce8ef699369c4d9836685c6fb3` upgrades that
module from an Euler feasibility prototype to a validated tensor backend.

## Representation and contracts

- Complete total-degree basis with deterministic exponent order and SHA-256
  fingerprint.
- Coefficients `[batch, state_output, monomial]`, remainders
  `[batch, state_output]`, domains `[batch, polynomial_variable]`.
- VDP order 4 uses two state outputs, three polynomial variables (two
  uncertainty generators plus physical local time), `tau_index=2`, and 35
  monomial slots.
- Local time is stored on `[0,h]`; integration introduces the time scale once.
- Sparse/dense conversion is exponent-based, rejects overflow, and is permitted
  only at declared segment boundaries in the hybrid lane.

## Operators

`BatchedMonomialBasis` caches multiply routes, degree-specific routes,
integration routes, overflow routes, and device copies. `BatchedPolynomial`
implements add/subtract/scale/affine map, grouped-exponent truncated multiply,
natural interval range, cutoff, local-time integration, evaluation, endpoint
substitution, and variable removal. Equal dropped exponents are aggregated
before intervalization to match the sparse semantic oracle.

`BatchedTaylorModel` propagates every discarded contribution into named ledger
categories: initial remainder, polynomial truncation, cutoff, integration
overflow, polynomial×remainder, remainder×polynomial,
remainder×remainder, Picard residual, roundoff, and reset. The former
`recenter_rescale` no-op now raises; the Euler-named step helpers also raise and
cannot enter production.

## Solver path

`dense_polynomial_picard` performs `order` polynomial Picard iterations in
physical local time. `dense_picard_validate_step` builds the fixed candidate,
evaluates the remainder self-map, supports the authoritative
`flowstar_raw_remainder_compat` expression replay, retains the static
truncation floor, rejects a failed batch if any leaf fails, and publishes no
endpoint on rejection.

`flowpipe_step_from_tm(..., tm_backend="dense")` selects
`hybrid_dense_core`: sparse boundary input → dense Picard/validation → sparse
boundary output. There are exactly two declared conversions per attempted
segment and no dictionary conversion or scalar loop inside Picard/validation.
Adaptive scheduling, normalized insertion, and right-map carry remain the
existing sparse outer path, so this is S3 rather than a full-dense S5 lane.

The ordered `PolynomialODE` reads the canonical YAML term list and preserves
subtraction/factor order (`y - x - (x*x)*y`), which is required because raw
remainder replay is expression-tree sensitive.

## Validation and performance boundary

The final suite is 343 passed and 2 optional external-backend skips. Dense and
sparse schedules are exact through T=1, and the largest shared reported-range
difference is `6.661338147750939e-16`. The CUDA test executes true
Picard/self-map validation, not the removed Euler helper.

The production microbenchmark calls package operators directly. On the
available V100, CUDA loses to CPU for every batch-1 operation; at batch 128 only
truncated multiplication wins. This is internal backend evidence, not a
cross-tool speed claim. See the checksum-addressed
`evidence/generic_batched_tm_backend_vdp_t10/20260804T152536Z` bundle.
