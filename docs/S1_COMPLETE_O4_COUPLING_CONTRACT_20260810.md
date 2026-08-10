# S1 complete-O4 coupling contract

Date: 2026-08-10 (Asia/Seoul)

Candidate: `normalized_insertion_structured_remainder_k16`

This document is the binding set contract for the opt-in S1 lane. It is not a
claim that Flow* or DiffReach implements this exact design.

## 1. Two coupled representations

Let `D = [-1,1]^d`, let `xi` range over the persistent dependency domain, and
let `tau` be physical local time. At accepted boundary `n`, the right-map state
is

```text
U_n(xi) = Q_n(xi) + R^ordinary_U,n + Z_U,n
Z_U,n   = sum_k Phi_U,n,k J_n,k.
```

`U_n` is in the normalized coordinates consumed by the next local Picard flow
map. `Q_n` is a retained total-degree-at-most-four polynomial. `R^ordinary_U,n`
is an interval vector in normalized units. Each `J_n,k` is an interval vector
in the physical units of the source at the boundary where it was created.
Each `Phi_U,n,k` is an interval matrix from those source-physical units to the
current normalized coordinates. An inactive slot denotes exactly `{0}`.

The local complete-O4 solve constructs

```text
T_n(u,tau) = P_n(u,tau) + R^ordinary_P,n(tau),
u in D, tau in [0,h_n].
```

`P_n` is a physical-state polynomial with a complete total-degree-four support.
`R^ordinary_P,n` is a physical interval vector valid for the full time slab.
The unmodified Picard self-map test must first certify `T_n` for every `u in D`.
S1 never removes `polynomial_truncation` from that raw Picard test.

The right-map domain gate is

```text
range(Q_n(xi) + R^ordinary_U,n + Z_U,n) subseteq D.       (G-domain)
```

The represented physical tube is obtained by composition. For

```text
B_n = range(Q_n(xi) + R^ordinary_U,n)
Zbox_n = materialize(Z_U,n),
```

the complete polynomial image primitive must certify

```text
P_n(B_n + Zbox_n,tau) - P_n(B_n,tau)
  subseteq A_n(tau) Zbox_n + N_n(tau).                    (G-image)
```

Here `A_n(tau)` is an interval matrix from old normalized units to physical
state units, and `N_n(tau)` is an interval vector in physical state units. The
production primitive expands every retained monomial and puts all structured
degree-two-through-four terms, mixed base/structured terms, and any outward
decomposition padding in `N_n`. It is run separately for endpoint `tau={h_n}`
and tube `tau=[0,h_n]`.

Consequently the actual boundary/tube set equation is

```text
X_n(tau) subseteq
    P_n(B_n,tau)
  + R^ordinary_P,n(tau)
  + N_n(tau)
  + sum_k A_n(tau) Phi_U,n,k J_n,k.                       (G-X)
```

Equivalently, after all ordinary composition/truncation/cutoff sources are
included in `R^ordinary_X,n`, this has the requested form

```text
X_n = Pbar_n(xi,tau) + R^ordinary_X,n(tau) + Z_X,n(tau)
Z_X,n(tau) = sum_k Phi_X,n,k(tau) J_n,k.
```

Every quantity in this equation has the following type and scope:

| quantity | units | type | domain/scope |
|---|---|---|---|
| `xi` | dimensionless | real vector | persistent dependency domain |
| `tau` | time | real scalar | `{h}` for endpoint or `[0,h]` for tube |
| `Q_n` | normalized | polynomial | boundary right map |
| `P_n`, `Pbar_n` | physical state | complete-O4 polynomial | endpoint and tube |
| `R^ordinary_U` | normalized | interval vector | boundary |
| `J_k` | source physical state | interval vector | source event |
| `Phi_U,k` | normalized/source-physical | interval matrix | boundary |
| `A_n` | physical/old-normalized | interval matrix | endpoint or tube |
| `N_n` | physical state | interval vector | endpoint or tube |
| `R^ordinary_X` | physical state | interval vector | endpoint or tube |
| `Z_X` | physical state | finite Minkowski sum | endpoint or tube |

No endpoint object is reused as a tube certificate. The endpoint result must be
contained in the independently computed tube result.

## 2. Picard target obligation

For the present S1 candidate the raw complete-O4 Picard target predicate is
unchanged:

```text
Picard(P_n + R_target) subseteq P_n + R_target,
R_target = [-1e-4,1e-4]^d.                                (G-Picard)
```

The typed ledger is a decomposition of the already certified raw-compatible
image, not permission to subtract a category before `G-Picard`. Eligible
categories can become separately owned S1 slots only after the accepted raw
decision. Soundness of checking the ordinary target is therefore the
conjunction

```text
G-Picard and G-domain and G-image and G-publication and G-ownership.
```

This design deliberately does not assert that a locally positive ordinary
margin can turn a rejected raw Picard attempt into an accepted one. Any future
target-split acceptance rule would require a separate structured Picard
self-map, including input/output slot containment; it is outside this frozen
candidate.

This point is essential: **Flow* symbolic remainder is not equivalent to
removing the current raw Picard `polynomial_truncation` interval from the target
remainder check.**

## 3. Coordinate maps

Let `S_n` be the diagonal physical scale defining current normalized
coordinates and let `S_n^{-1}` use `1` on an exactly zero scale only for the
zero perturbation coordinate. The required maps are named explicitly:

```text
old physical perturbation:
    delta_x_old
old normalized perturbation:
    delta_u_old = diag(S_n^{-1}) delta_x_old
accepted endpoint perturbation:
    delta_x_end in A_old_normal_to_endpoint_physical delta_u_old + N_endpoint
new normalized perturbation:
    delta_u_new = diag(S_{n+1}^{-1}) delta_x_end
new physical structured publication:
    delta_x_new = diag(S_{n+1}) delta_u_new.
```

The stored `Phi_U` maps source-physical perturbations directly to current
normalized perturbations. Therefore old columns propagate as

```text
A_old_normal_to_new_normal
  = diag(S_{n+1}^{-1}) A_old_normal_to_endpoint_physical
Phi_U,n+1,k
  = A_old_normal_to_new_normal Phi_U,n,k.
```

A newly created physical source uses

```text
Phi_U,n+1,new = diag(S_{n+1}^{-1}),
J_n+1,new = centered physical source interval.
```

`inverse_scale` is therefore live metadata: it is the point diagonal used to
build every newly inserted `Phi_U`. A reconstruction test must show that
`diag(S) Phi_U J` contains the original physical source, including asymmetric
scales. A nonzero source in a zero-scale coordinate fails closed.

## 4. Complete-O4 nonlinear obligation

For every retained monomial `c_alpha q^alpha`, the production image expands

```text
c_alpha ((q + delta)^alpha - q^alpha)
```

by the multivariate binomial theorem. Terms of total structured degree one form
the outward interval affine matrix `A`; every route of structured degree two,
three, or four forms `N`. Thus `N` includes, where present,

```text
xi*delta, xi^2*delta, xi*delta^2,
delta^2, delta^3, delta^4,
and every cross-state mixed monomial.
```

Coefficients, base boxes, coordinate matrices, and all reductions are binary64
CPU outward intervals using `nextafter` at each operation. Nonfinite inputs,
degree above four, coordinate mismatch, or an inverted interval fail closed.
CUDA float64 and compiled arithmetic have no outward claim.

## 5. Ownership, acceptance, and publication

Each source has identity

```text
(accepted_boundary_index, source_category, occurrence_index).
```

At every committed boundary it has exactly one owner: ordinary, one live S1
slot, or one materialization event. Eligible source center enters ordinary once
and its centered interval enters one slot. An evicted column is first mapped to
the current coordinates and added to ordinary exactly once, then the slot is
overwritten. All state and event tensors are constructed from the frozen
prestate and committed only after every gate passes.

Publication must establish

```text
published endpoint contains ordinary endpoint + every endpoint S1 column;
published tube contains ordinary tube + every tube S1 column;
materialized poststate contains the certified pre-split total image.
```

Removing any active column must make the publication certificate fail.

## 6. External implementation audit

The pinned Flow* source revision is
`b85a3211748cb77b736fe4ad42ee02d8d2b81148`. The tree was already dirty at
round start; it is read-only for this work. Exact anchors in
`flowstar-toolbox/Continuous.cpp` are:

- `2724`: evaluate `tmvPre` at the step end;
- `2743-2753`: split linear/nonlinear terms and right-scale `Phi_L_i` by prior
  inverse scales;
- `2756-2769`: propagate the `Phi_L` list and form the old `J_i` contribution;
- `2771-2783`: construct `J_{i+1}` through nonlinear insertion;
- `2884-2914`: append `J_{i+1}`, normalize, update scalar inverses, and apply
  normalized cutoff;
- `2920-2967`: build the polynomial Picard candidate and test the complete
  `Picard_ctrunc_normal` remainder, including polynomial differences, against
  the target remainder;
- `3004-3029`: refine only with complete Picard remainders.

The pinned DiffReach revision is
`dd628eb443b517d6415de93e7035b4baef73963e`. Exact anchors are:

- `src/reachability.py:137-146`: evaluate the old flow at step end and call the
  symbolic transition;
- `src/symbolic_remainder.py:89-98`: form `A`, `Phi_new`, `J_new` inputs, and
  propagate the seed remainder;
- `src/symbolic_remainder.py:105-123`: propagate old columns, align `J`, and
  form total remainder;
- `src/symbolic_remainder.py:129-170`: normalize, update buffers, and apply its
  clear-on-cap behavior;
- `src/reachability.py:155-181` and `src/picard.py:13-58`: polynomial Picard and
  remainder refinement.

DiffReach uses a quadratic representation, DR-RP through its remainder Picard
operator, and clears its symbolic buffers when the configured cap is reached.
S1 instead uses complete O4, deterministic oldest-first materializing eviction,
and unique source ownership. These are independent obligations; “Flow*-like”
is not evidence for S1 soundness.

## 7. Executable gates

The contract is accepted for implementation only if independent tests prove:

1. physical/normalized reconstruction and zero-scale failure;
2. complete retained-monomial expansion against a rational interval oracle;
3. affine and harmonic maps have exactly zero nonlinear residual;
4. endpoint image is contained in the tube image;
5. raw Picard acceptance remains unchanged in this candidate;
6. `G-domain`, endpoint/tube publication, conservation, and unique ownership
   pass before state commit;
7. rejected attempts return the byte-identical prestate.

Failure of any item is `S1_TARGET_SPLIT_NOT_SOUNDLY_COUPLED` before prefix or
fresh-horizon experiments.
