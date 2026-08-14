# Complete-O4 bounded source-ledger contract (G1)

Status: authoritative preregistered boundary contract for the 2026-08-14
T=1/T=3 causal study.  Production mode, if and only if the independent
micro-oracles pass:

`normalized_insertion_bounded_source_ledger_o4_g1`

The single changed variable is the accepted-boundary carry representation.
ODE, initial set, complete-total-degree O4 basis, cutoff, candidate remainder,
validator, range policy, step controller, output definitions, dtype, and device
remain frozen.

## State and represented set

For a `d`-state polynomial system, the boundary domain has exactly `2d`
uncertainty variables:

- `u_0,...,u_(d-1)` are the usual normalized base coordinates;
- `z_0,...,z_(d-1)` are one-generation source slots, each in `[-1,1]`.

For component `i`, the represented accepted boundary set is

`X_i = c_i + Q_i(u) + R_o,i + rho_i z_i`.

Equivalently, before rebase it is

`X = P(old_sources) + R_o + sum_i Phi_i z_i + R_nonlinear`,

where `Phi_i` is a scalar affine source coefficient in this G1 contract.
`R_nonlinear` is not retained as another symbolic object: it is outward-added
to `R_o` at retirement.  The live source shape is always `d`, independent of
the horizon and of the benchmark.

## Accepted-boundary operator

1. Substitute local time `tau=h` in the accepted dense complete-O4 endpoint.
   Canonical polynomial construction merges equal exponent vectors before any
   interval range is taken.
2. Compose every base coordinate with the prior normalized right map.  Append
   identity maps for live `z` coordinates, so the same source variable is used
   by every occurrence in the next Picard polynomial.
3. Partition the composed polynomial into source-free terms and terms with any
   nonzero `z` exponent.  Merge first, then outward-evaluate the complete
   source-bearing polynomial once.  Add that interval to ordinary remainder.
   This is the deterministic one-generation retire/collapse rule.
4. The tensor-native validated remainder ledger owns all accepted Picard-image
   residual contributions.  Its fixed category schema is
   `initial_remainder`, `polynomial_truncation`, `cutoff`,
   `integration_overflow`, `composition_overflow`,
   `poly_times_remainder`, `remainder_times_poly`,
   `remainder_times_remainder`, `picard_residual`,
   `roundoff_safeguard`, and `reset_or_reconditioning`.
   The complete outward sum must contain the unchanged accepted Picard image.
5. For each state component, outward-lift that complete ledger interval
   `[l_i,h_i]` to `m_i + rho_i z_i`.  The machine witness requires
   `m_i-rho_i <= l_i <= h_i <= m_i+rho_i`.  Shift `m_i` into the physical
   center and create the fresh source ID
   `boundary:n:component:i:validated_remainder_aggregate`.
6. Rebase only the source-free polynomial plus ordinary collapse interval into
   the usual normalized base coordinate.  The fresh source radius is not also
   included in that scale or remainder.
7. Publish the affine reset
   `c_new + scale*u + rho*z`.  This is the actual next dense Picard input.

Thus each ledger residual has exactly one owner: the complete accepted ledger
until lift, the fresh affine source during its consumer generation, and the
ordinary nonlinear-collapse ledger after retirement.  It is never present in
both source and ordinary payloads.

## Containment proof obligations

No sampling is used for the boundary proof.

- The accepted dense decomposition certificate proves
  `accepted_image subseteq sum(named ledger intervals)` using outward
  binary64 interval arithmetic.
- The affine-lift witness proves that each summed ledger interval is contained
  by its fresh source.
- Exact `tau=h` substitution is coefficient substitution in the canonical
  polynomial, followed by duplicate-exponent addition.
- The source partition is exhaustive and disjoint by exponent.  Natural
  outward evaluation of all source-bearing terms proves their collapse
  interval contains those terms for the full box.
- Normalization chooses a magnitude enclosing the source-free polynomial plus
  ordinary interval; dividing by that scale must certify its right-map range is
  within `[-1,1]`.
- Minkowski addition of the preceding enclosures proves that the new boundary
  representation contains the old accepted endpoint represented set.

The exact-rational fixtures independently enumerate polynomial terms and check
these inclusions against binary64 endpoints interpreted as exact rationals.
CPU float64 is authoritative. CUDA decision parity is an implementation check,
not a formal directed-rounding claim.

## Lineage, retry, and boundedness

- Create: only after an accepted validator result and a valid complete ledger.
- Propagate/consume: a live source is a genuine polynomial variable in every
  next-step O4 Picard multiply, integrate, truncate, and cutoff operation.
- Merge: equal full exponent vectors are added before source range evaluation.
- Retire: at the next accepted boundary, all terms containing any old source
  are outward-collapsed once, regardless of coefficient size or state index.
- Retry: state objects are immutable. A rejected candidate returns the exact
  previous object and hash; it cannot advance generation, IDs, or counters.
- Finite shape: exactly `d` source slots and `2d` boundary variables. There is
  no K tuning, magnitude ranking, VDP branch, or horizon-dependent growth.

## Distinction from rejected predecessors

This is not complete endpoint-polynomial carry: after one consumer generation,
all old-source terms are collapsed and the reset remains affine in exactly `d`
fresh sources. Historical endpoint polynomials are not cloned indefinitely.

This is not `structured_total_delta_k16`: it does not propagate a K16 interval
linear map or pad a materialized total-delta image. Source identity is an actual
polynomial variable in the dense Picard consumer, nonlinear paths share that
identity, and retirement is an explicit merged-polynomial containment step.
