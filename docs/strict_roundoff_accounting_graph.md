# Huan strict roundoff accounting graph

Engine SHA: `b0ff55745d69205f3afb4dc8077b9ac1310bfff3`

Scope: polynomial plant reachability in `mode="strict"`, binary64, finite
intermediates, gradual underflow, and the checked `m*u` reduction hypothesis.
Parity deliberately retains the Flow* point-coefficient trust model and is not
part of this soundness claim.

## Accounting invariant

For every retained point polynomial `p_hat`, strict mode maintains an ordinary
interval remainder `R` such that the exact real polynomial induced by all
stored binary64 inputs is contained in `p_hat + R`. A coefficient error is
ranged over the actual dense or sparse normalized monomial support before it is
added. Each source enters `R` once; later nonlinear composition propagates that
ordinary remainder through the usual Taylor-model product algebra.

```text
binary64 operands
  |
  +-- dense/sparse monomial product
  |     point tier-P coefficient (unchanged)
  |     + exact-enclosing interval coefficient product
  |     + interval difference from retained point
  |     + range on actual normalized support
  |     `--> ordinary image remainder (once)
  |
  +-- dense/sparse composition contraction GEMM
  |     Rump dot_error_bound over the true inner dimension
  |     + range coefficient radii on actual support
  |     `--> composed ordinary remainder (once)
  |
  +-- symbolic Phi
  |     exact-enclosing interval lin_a * scalar interval
  |     + outward interval matrix products for every queue push
  |     + outward interval Phi * J reconstruction and queue sum
  |     `--> symbolic-history ordinary remainder (once)
  |
  +-- SR linear polynomial image / final coefficient add
  |     Rump GEMM coefficient radius + interval-vs-point add defect
  |     + range on actual spatial support
  |     `--> ordinary remainder (once)
  |
  +-- reciprocal preconditioning
  |     interval reciprocal + interval coefficient scaling defect
  |     + range on actual support
  |     `--> scaled ordinary remainder (once)
  |
  +-- time-end substitution
  |     exact-enclosing reduction error grouped by spatial monomial
  |     + range on spatial support
  |     `--> endpoint-only remainder (once)
  |
  `-- polynomial Picard point path
        validated interval Picard image - point candidate polynomial
        + range of full coefficient defect
        `--> contraction image used by the self-map certificate (once)
```

## Symbolic-`Phi` path

`flowpipe.advance` keeps the original point `phi_i` for parity-compatible point
state, but strict mode also forms `phi_i_iv` from degenerate/previous interval
factors. `symbolic_remainder.propagate` stores it in `phi_iv_buf`, updates every
live queue entry with outward interval matrix multiplication, and reconstructs
the history remainder by outward interval multiply-and-sum. `scalars_iv`
preserves the reciprocal-preconditioning enclosure across steps and queue
reset reinitializes it from the current point scalars. Thus the two point
matrix products and the reconstruction reduction exposed by the minimized D6
witness are no longer treated as exact.

## Retained monomial-image path

The dense `mul_point_spatial_with_roundoff` and sparse
`mul_point_s_with_roundoff` functions return the same point coefficients as
parity. Independently, they evaluate the induced coefficient convolution with
outward interval multiplication/reduction, subtract the retained point value,
and range the resulting interval-coefficient polynomial on the precise support.
`composition.compose` / `sparse_exec.compose_s` add that range once to the
image-node ordinary remainder. Subsequent image levels consume it through
`I1*I2 + P2*I1 + P1*I2`; there is no second insertion of the original source.

## Preconditions and failure boundary

`determinism.assert_gradual_underflow` runs on CPU and on the selected CUDA
path (including the custom interval kernel when active) before public reach.
Rump bounds check their reduction-length hypothesis. Generic NaN/Inf masks are
propagated through validation/refinement and cannot satisfy `interval.contains`;
`interval.assert_valid` rejects infinity at a finite-certificate boundary.
The exact/adversarial suite covers subnormals, signed zero, mixed signs,
cancellation, ties, near-overflow finite cases, overflow failure, dense/sparse
embeddings, queue capacities and reset.

The machine-readable companion is
`outputs/huan_proof_closure/strict_roundoff_sources.csv`.
