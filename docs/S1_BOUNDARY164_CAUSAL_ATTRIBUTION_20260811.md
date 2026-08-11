# S1 boundary-164 causal attribution

Date: 2026-08-11

Evidence package:
[`outputs/s1_boundary164_causal_guarded_carry_20260811/20260811T033447Z`](../outputs/s1_boundary164_causal_guarded_carry_20260811/20260811T033447Z/)

## Result

The historical numerical negative result is reproduced exactly from three
complete, byte-stable boundary-164 checkpoints. At step 163, the y subset
margins are:

| lane | y margin |
|---|---:|
| L0 | `+2.60697659917348e-5` |
| L1 | `+1.7291650118437743e-5` |
| L2 | `+1.7363995494671766e-5` |

On the first validator call at the historical full
`h=0.03661680691961388`, the margins are:

| lane | x margin | y margin | decision |
|---|---:|---:|---|
| L0 | `+9.633831630803861e-5` | `+8.058292550874906e-6` | accepted |
| L1 | `+9.633831630803861e-5` | `-3.872231318094365e-6` | rejected |
| L2 | `+9.633831630803861e-5` | `-3.773875528686747e-6` | rejected |

The diagnostic never commits the returned state and cannot overwrite the
full-step evidence with an adaptive half-step. The checkpoint and margin
records are in [`checkpoint_triad.csv`](../outputs/s1_boundary164_causal_guarded_carry_20260811/20260811T033447Z/checkpoint_triad.csv).

## Why L1 and L2 stop together

L1 materializes structured columns at every accepted boundary. L2 retains up
to K16 columns and evicts the oldest after capacity is full. Their nearly
identical terminal loss therefore does not identify K16 or eviction as the
cause. The registered controls isolate the cause:

- C1, the full typed-ledger shadow, is bit-exact to C0 for the complete replay.
- C2, exact carrier split/remerge without image decomposition or physical
  roundtrip, is bit-exact to C0 with set relation `equal`.
- C3, the current decomposition on the canonical center/scale/right
  polynomial without extra renormalization, first fails the exact normalized
  domain gate at boundary 11 (the attempted transition to boundary 12).
- C4 adds outward renormalization and reaches the full diagnostic, but with
  accumulated scale drift.

Thus L1 and L2 share the same post-hoc complete-polynomial image decomposition,
padding, and later scale response. The K16 retention policy changes small
details but is not the primary inflation source. The complete control table is
[`causal_ladder.csv`](../outputs/s1_boundary164_causal_guarded_carry_20260811/20260811T033447Z/causal_ladder.csv).

## First exact divergences

The raw comparator preserves binary64 hex before assigning any ULP grade.
For C0 versus C4 it finds:

| quantity | first boundary/attempt |
|---|---:|
| materialized carrier representation | boundary 0 |
| right-polynomial coefficient hash | boundary 5 |
| forward scale hex | boundary 5 |
| physical right-map hull | boundary 8 |
| validator subset margin | attempt 8 |
| outward renormalization count | boundary 12 |
| center | no difference through boundary 164 |

The boundary-0 materialized difference is carrier encoding, not physical-state
drift. C1/C2 prove that encoding alone is non-causal. The first consequential
state divergence is the coefficient/scale change at boundary 5, followed by
the physical hull and validator margin at boundary 8. Exact records and
neighbor snapshots are indexed by
[`first_divergence.csv`](../outputs/s1_boundary164_causal_guarded_carry_20260811/20260811T033447Z/first_divergence.csv).

## A0--B16 transition attribution

Every structured boundary now records tensor-native A0--B16 intervals with
explicit units. The current contract is:

```text
B = range(Q + R_o)
perturbation = Z
known = A_B (R_o + Z) + N_Z + typed sources
```

The important causal stages are B2 (inflated base), B4 (affine map evaluated
over that base), B5 (the ordinary remainder enters the affine image again),
B7 (nonlinear residual only in `Z`), and B12 (proved padding to the canonical
target). An executable scalar quadratic fixture shows that the unpadded
current reconstruction does not, in general, contain the exact
`P(Q+R_o+Z)-P(Q)` image. It is a conservative decomposition only after B12
padding; it is not an algebraically exact partition of the total delta.

The diagnostic total-delta contract is:

```text
Delta = R_o + Z
P(Q + Delta) - P(Q) subseteq A_total Delta + N_total
```

Its independent Fraction fixtures include affine, harmonic, quadratic, cubic
mixed, quartic mixed, asymmetric, cancellation, and duplicate-exponent cases.
`N_total` contains ordinary degree >=2, structured degree >=2, and
ordinary×structured terms once. On the VDP prefix, both padded contracts
contain the canonical target. Total-delta is never wider after padding across
1,312 component comparisons and is strictly narrower in 256. Before padding,
it is wider in 376 comparisons, so the evidence supports post-padding
dominance only. At boundary 164 the C4 total-delta/current width ratios are
`0.9998880217842235` in x and `0.9884275321837205` in y.

The complete machine table is
[`boundary_stage_widths.csv`](../outputs/s1_boundary164_causal_guarded_carry_20260811/20260811T033447Z/boundary_stage_widths.csv).

## Roundtrip, padding, and renormalization contributions

No registered evidence assigns measurable first-validator loss to a
physical-normal roundtrip. C2 explicitly avoids a physical roundtrip and is
bit-exact to C0. At boundary 164, H3 and H4 materialize L1/L2 into an
ordinary-only carrier without reboxing; their validator-input hashes and
margins remain exactly equal to their native prestates. This is a zero
contribution for the registered same-input validator projection, not a general
proof that every coordinate conversion is exact.

Padding is essential to the current set contract: it restores containment
after the B2/B5 over-intervalized partition. It masks the early width
difference—both padded contracts are equal at decisive boundaries 5, 8, 12,
17, and 70—then total-delta becomes strictly narrower later. Padding is
therefore a sound safeguard and also evidence that the current partition is
not intrinsically exact.

Renormalization is secondary. Coefficient/scale drift begins at boundary 5
and margin drift at attempt 8; the first different renormalization event is
only boundary 12. C3 demonstrates why it is needed: without it the exact
`[-1,1]` domain gate fails. It lets C4 continue but does not create the first
inflation.

## Boundary-164 component substitution

T0-style first-validator projection depends on center and scale, not the
stored right polynomial or remainder carrier. H1 uses L0 center/scale/poly
with the L1 total remainder and exactly matches P0. H2 and H3 use L1
center/scale and exactly match P1. H4 exactly matches P2. L0 and L1 centers
are hex-identical. Consequently the L0→L1 y-margin difference
`-1.1930523868969271e-5` is assigned entirely to scale for this fixed
diagnostic:

| component | contribution |
|---|---:|
| center | `0` |
| scale | `-1.1930523868969271e-5` |
| right polynomial | `0` |
| total remainder | `0` |
| validator/reduction residual | `0` |
| interaction remainder | `0` |

Hybrids are diagnostic only and are excluded from certificate claims. See
[`boundary164_substitutions.csv`](../outputs/s1_boundary164_causal_guarded_carry_20260811/20260811T033447Z/boundary164_substitutions.csv).

## Formal scope

The exact/Fraction oracle qualifies only
`complete_polynomial_structured_image` for given binary64 coefficients on the
CPU outward path. Typed ledger addition and checkpoint byte identity have
their own narrower claims. The retained coefficient operations—ordinary
multiplication, `scatter_add_`, integration coefficient updates, cutoff,
affine and Picard coefficient updates, dense→sparse conversion, and sparse
normalized insertion—do not all carry coefficient-rounding error into the
remainder and do not have an independent decisive-workload exact replay.

The prefix classification is therefore exactly:

```text
safeguarded_binary64_interval_shell
conditional_on_retained_coefficient_arithmetic
primitive_formal_eligible = true
prefix_formal_eligible = false
```

It is not an end-to-end formal flowpipe proof.

## Outcome choice

Outcome A is not selected because C1/C2 expose no side effect, coordinate bug,
omission, or rejected-state mutation. Outcome C is not selected because the
Fraction-backed total-delta contract closes all ordinary/structured nonlinear
interactions and is no wider on every post-padding decisive comparison.
Outcome D is unnecessary because the cause is resolved. The sole Phase-5
choice is Outcome B, `S1_POSTHOC_IMAGE_INTRINSIC_INFLATION`, authorizing only
`normalized_insertion_structured_total_delta_k16`.

