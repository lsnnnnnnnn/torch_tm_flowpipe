# Structured remainder semantics (2026-08-10)

## Scope and source audit

This document freezes candidate S1,
`normalized_insertion_structured_remainder_k16`. It is a clean-room bounded
state, not a claim that the older Python-float symbolic queues reproduce
Flow*. The audited sources were:

| Source | Pinned revision / file SHA-256 | Relevant semantics |
|---|---|---|
| Flow* | `b85a3211748cb77b736fe4ad42ee02d8d2b81148`; `Continuous.cpp` `4d42b818…d0387b8f1`, `Continuous.h` `6c22a8bf…f3eea2`, `TaylorModel.h` `ed127748…657c7` | `J`, `Phi_L`, inverse reset scalars, normal insertion, Picard remainder subset, clear-on-cap |
| DiffReach | `dd628eb443b517d6415de93e7035b4baef73963e`; `symbolic_remainder.py` `e4b2e070…65796`, `reachability.py` `1c155652…ccc9` | `Phi_new`, old-column propagation, `J_new`, normalization, DR-RP, clear-on-cap |
| clean-room Torch | `symbolic_remainder.py` `d6c12439…d1918`, pre-S1 `batched_dense_tm.py` `4f6a3598…122d9a` | prior diagnostic queues and additive typed dense ledger |

Flow* evaluates the prior endpoint at the step end, separates its linear map,
right-scales that map by the previous inverse scales, left-multiplies stored
`Phi_L`, accumulates `Phi_L[i] J[i-1]`, appends the new local remainder, then
normalizes. DiffReach expresses the same alignment as
`Phi_new @ Phi_old`, a shifted contribution list, and `Past + J_new`. Both
implementations clear a full queue. S1 deliberately differs at capacity: it
materializes and evicts the oldest column, preserving its set contribution.

## Frozen state and source policy

For batch `b`, state coordinate `s`, and slot `k < K=16`, S1 stores:

```text
ordinary_rem_lo/hi[b,s]       ordinary interval carry
J_lo/hi[b,k,s]                centered source interval
Phi_lo/hi[b,k,s,s]            propagated normalized linear map
active[b,k]                   live-slot mask
age[b,k]                      insertion boundary index
source_id[b,k]                1=polynomial_truncation, 2=integration_overflow
inverse_scale[b,s]            previous safeguarded normalization inverse
```

Only the additive dense-ledger fields `polynomial_truncation` and
`integration_overflow` are eligible. `cutoff`, mixed polynomial/remainder
products, remainder/remainder products, initial remainder, raw-compatibility
polynomial difference, and roundoff safeguards remain ordinary. Diagnostics
that do not participate in the validated additive ledger are ineligible.

Capacity, insertion order, and eviction are frozen before terminal replay:
`K=16`; sources are inserted in numeric source-ID order; the oldest `(age,
slot)` is outward-materialized and evicted when no inactive slot exists. K32 is
not an S1 option unless the preregistered 20% eviction-attribution condition is
met before any fresh horizon run.

## Boundary formula

Let the additive, outward ledger at boundary `n` be

```text
R_validated ⊆ R_pad + Σ_q R_q,
```

where `R_pad` contains any decomposition-rounding padding and every ineligible
typed source. For eligible source `R_q=[l_q,u_q]`, define, with outward
arithmetic,

```text
c_q = midpoint(l_q,u_q)
r_q = max(c_q-l_q, u_q-c_q)
R_q ⊆ c_q + [-r_q,r_q].
```

The center `c_q` is added once to the ordinary carry; the symmetric interval
`[-r_q,r_q]` is inserted once as a new `J` column with identity `Phi`. Thus an
asymmetric source is neither lost nor copied.

For the safeguarded degree-1 normalized boundary map `A_n`, old columns update
as

```text
Phi'_{n,k} = A_n Phi_{n,k}.
```

Interval matrix multiplication expands every product and sequential sum
outward. The difference between the true nonlinear source image and the
retained degree-1 propagation is supplied as the separately audited interval
`N_n = structured_nonlinear_residual` and is added once to ordinary carry.
S1 rejects a boundary if this residual is missing for a nonlinear map.

If `e` is evicted, its contribution

```text
E_e = Phi'_{n,e} J_{n,e}
```

is added once to the ordinary carry before the slot is overwritten. New
sources are normalized by the safeguarded scale metadata used by the same
boundary. Zero scales use inverse one and nonfinite/nonpositive inconsistent
scales fail closed.

## Materialization and conservation

The sole materialization function is

```text
materialize(state) = ordinary_rem + Σ_{k:active} Phi_k J_k.
```

Each active column has an independent domain `epsilon_k ∈ [-1,1]`; interval
matrix images are Minkowski-summed. Endpoint and tube publication always add
the full materialized interval. Validator fields distinguish:

- `ordinary_target_checked_remainder`: the ordinary component presented to
  the unchanged target-radius check;
- `structured_validation_contribution`: the structured part required by the
  self-map/containment check;
- `endpoint_total_remainder` and `tube_total_remainder`: ordinary plus every
  active structured column.

The executable invariant is

```text
materialize(post_split_state) contains pre_split_validated_remainder
```

componentwise. A source identifier may occur either in ordinary carry (after
materialization/eviction) or in one live `J` slot, never both. Conservative
decomposition padding is allowed only in `R_pad` and is reported separately;
duplicating a source interval to force containment is forbidden.

## Nonlinear rule

For an analytic map `F` and structured set `Z`, S1 retains `A Z` and requires
an interval enclosure

```text
N ⊇ F(x+Z) - F(x) - A Z.
```

Affine and harmonic-linear maps have `N=[0,0]`. For polynomial systems the
implementation bounds every degree-two-and-higher monomial containing a
structured variable by interval multiplication on the declared source box.
Scalar quadratic/Riccati and the two-state cross-polynomial tests are analytic
gates. Sampling is recorded only as a sanity check.

## Failure and publication rules

S1 fails closed on missing/nonadditive source decomposition, dimension or
domain mismatch, nonfinite bounds, invalid intervals, a missing nonlinear
residual, normalization inconsistency, conservation failure, or publication
that omits an active column. Failed state is frozen and cannot increase the
validated prefix. Output-only quantities never influence target acceptance.

The pre-existing reset modes remain unchanged and opt-in S1 has no fallback to
them. Passing the local frozen checkpoint is only a GO decision; it is not a
formal or cross-tool promotion.
