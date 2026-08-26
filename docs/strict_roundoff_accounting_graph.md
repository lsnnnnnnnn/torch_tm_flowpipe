# Huan strict roundoff accounting graph

Engine SHA: `743f6205e6408072193ad76e940e7f15030e8d3c`

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

The following locations and shapes are pinned to engine SHA `743f6205`.
`B` is batch size, `n` the state dimension, `Q` the live symbolic-history
length, and the trailing `2` is the lower/upper endpoint axis.

| Operation | Source line and function | Point value retained | Strict enclosure and insertion | Reduction |
|---|---|---|---|---|
| dense `Phi` factor | `flowpipe.py:586-606`, `advance` | `lin_a * scalars`: `[B,n,n]` | outward `iv.mul([B,n,n,2], [B,1,n,2]) -> phi_i_iv [B,n,n,2]`; pushed beside point `Phi` | one product per entry, no reduction |
| sparse `Phi` factor | `sparse_exec.py:1445-1463`, `advance_sparse` | same `[B,n,n]` point factor | same interval factor and queue insertion | one product per entry, no reduction |
| historical `Phi` update | `symbolic_remainder.py:211-218,253-265`, `_matmul_iv` / `propagate` | `einsum bij,qbjk->qbik`: `[Q-1,B,n,n]` | `iv.mul` then `iv.sum(dim=3)` gives `[Q-1,B,n,n,2]`; replaces the matching `phi_iv_buf` slice | inner length `n`; point schedule is the selected Torch einsum, strict interval sum is outward |
| `Phi*J` reconstruction | `symbolic_remainder.py:271-278`, `propagate` | parity uses point `Phi` and interval `J` | `[B,Q,n,n,2] * [B,Q,1,n,2]`, outward inner sum to `[B,Q,n,2]`, then outward queue sum to `[B,n,2]`; returned as `j_lin` and added once at `flowpipe.py:619` / `sparse_exec.py:1475` | inner length `n`, then queue length `Q` |
| queue reset | `symbolic_remainder.py:72-83,181-194`, `reset` / `reset_if_full` | clears factorization and restores point scalars to one | interval scalars are restored to degenerate one; the emitted flowpipe already owns the full ordinary remainder | no reduction; reset at `jlen >= max_size` |

The exact intended quantity in all four `Phi` rows is the real product/sum
induced by the stored binary64 factors. The point buffers remain the normal
binary64 values for parity. Strict owns the parallel interval buffers and the
ordinary `j_lin` result, so each matrix-product error is charged once. Parity
intentionally omits those coefficient-error enclosures.

## Retained monomial-image path

The dense `mul_point_spatial_with_roundoff` and sparse
`mul_point_s_with_roundoff` functions return the same point coefficients as
parity. Independently, they evaluate the induced coefficient convolution with
outward interval multiplication/reduction, subtract the retained point value,
and range the resulting interval-coefficient polynomial on the precise support.
`composition.compose` / `sparse_exec.compose_s` add that range once to the
image-node ordinary remainder. Subsequent image levels consume it through
`I1*I2 + P2*I1 + P1*I2`; there is no second insertion of the original source.

The retained-image operations are pinned as follows. `Ts` is the complete
spatial support size, `Ps` its gathered pair count, `Lc` a level chunk, `M` a
sparse ancestor count, and `Sm` a sparse output-support size.

| Operation | Source line and function | Point value retained | Strict enclosure and insertion | Reduction |
|---|---|---|---|---|
| dense monomial image | `polynomial.py:432-483`, `mul_point_spatial_with_roundoff`; caller `composition.py:208-241` | RN gathered products `[B,Lc,Ps]` segment-summed to `[B,Lc,Ts2]` | interval convolution minus point coefficient, ranged on the actual complete spatial basis to `[B,Lc,2]`; inserted once into `m_rem` at `composition.py:232-237` | per-output `sp_seg_len`, deterministic segment schedule; Rump bound on Torch path |
| sparse monomial image | `support.py:519-579`, `mul_point_s_with_roundoff`; caller `sparse_exec.py:926-950` | point result `[B,L,S_out]`, Torch segment order or fused CUDA path | interval sparse convolution minus point coefficient, ranged on `pb.sup_out` to `[B,L,2]`; inserted once at `sparse_exec.py:945-946` | per-output `pb.seg_len`; fused directed CUDA interval path or Rump-bounded Torch path |
| dense composition contraction | `composition.py:243-263`, `compose` | `einsum bim,bmt->bit`: `[B,n,Ts]` | `dot_error_bound(abs_dot, Ts)` ranged on the actual spatial factors and inserted once into `out_rem` | inner length `Ts`, selected Torch einsum schedule |
| sparse composition contraction | `sparse_exec.py:952-967`, `compose_s` | `einsum bim,bmt->bit`: `[B,n,Sm]` | `dot_error_bound(abs_dot, M)` ranged on `cs.sup_m` and inserted once into `out_rem` | inner length `M`, selected Torch einsum schedule |

For these rows the exact quantity is the real coefficient convolution or
contraction induced by the binary64 input coefficients. The retained point
coefficient is deliberately unchanged. Strict adds only the independently
computed coefficient-error range; parity deliberately charges zero for this
source.

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
