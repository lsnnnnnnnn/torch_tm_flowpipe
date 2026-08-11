# Complete-O4 carry semantics root cause

Date: 2026-08-11

## Outcome

Root-cause class `C4`: `CARRY_MISSING_SYMBOLIC_SEMANTICS`.
`NO_FIX_AUTHORIZED`.

## Eligibility

The result is eligible as an ordinary-float64 same-prestate causal diagnosis
of the implemented R35 CNI contract. It does not establish that A3/CDR is
sounder, and A4 is not eligible as an authoritative dense/Flow* complete-O4
mirror.

## Contract

A3 and A4 are frozen at R35, Picard 4, VRAW, target 0.01, no cutoff,
`h=0.01`, B1/B64, through T10 or first failure. Same-prestate substitutions
use byte-identical inputs and never commit state. The dense audit freezes the
variable order `(local time, xi0, xi1)` and requires bit-exact coefficient and
remainder roundtrip.

## What was actually run

A3/B1, A3/B64, A4/B1, and A4/B64 were rerun with full carry-state observers.
Four A4/B1 checkpoints were replayed with CDR/CNI and reciprocal epsilon
on/off substitutions. Native CNI composition was reconstructed with source
accounting and checked bit-exact. R35/dense basis roundtrips covered affine,
quadratic, cubic VDP, first-material-divergence, and pre-failure fixtures.

## Exact results

- A3 B1/B64: 1,000 accepted steps, T10.
- A4 B1: 319 accepted; failure step 320 at time 3.19.
- A4 B64: 333 accepted; failure step 334 at time 3.33.
- First coefficient/carry divergence: step 1; first ordinary remainder
  divergence: step 2; first physical endpoint/tube divergence: step 1.
- Before A4/B1 step 320 on one identical prestate: CDR accepts with margin
  `+0.0005439375476625798`; CNI rejects with margin
  `-0.0004978288426217593`. Epsilon is decision-irrelevant.
- CNI pre-renormalization remainder width is `7.407163850432577`; the dominant
  polynomial-times-parameterization-remainder source width is
  `7.119613927492569`, versus `2.1828042802002517e-05` degree-overflow and
  `0.01786912525433395` outer endpoint remainder.
- No omission, coordinate mismatch, or outer-remainder double count was
  detected. Native observer parity is bit-exact; physical roundtrip preserves
  the hull.
- Basis roundtrip is bit-exact, but authoritative cross-step dense carry is
  absent: `DENSE_CNI_PARITY_NOT_EXPRESSIBLE`.

## What is comparable

Frozen-prestate CDR/CNI one-step decisions, center/scale, normalized
parameterization, composition sources, physical hull, and exact R35/dense
basis transport. The A3/A4 long traces are empirical reproductions.

## What remains unavailable

An authoritative dense complete-O4 nonlinear insert/carry operator and its
extra symbolic state are unavailable. Therefore no valid dense-vs-R35
cross-step parity result or simple implementation repair exists.

## Negative results

C1 is excluded by physical roundtrip; C2 by once-only source accounting; C3
is not selected because degree-overflow is negligible relative to materialized
parameterization remainder; C5 cannot explain order-one widening and horizon
failure; C6 is unnecessary because one mechanism dominates. A3 reaching T10
is not a soundness proof.

## Limitations

C4 identifies a missing semantic contract, not a local code defect. Replacing
CNI with a narrower formula without an independently validated symbolic-state
contract would be a new algorithm and is prohibited in this round.

## Evidence paths

Under `outputs/three_tool_full_horizon_pairwise_carry_closure_20260811/20260811T191549Z/`:

- `06_carry_reproduction/` through `10_root_cause/root_cause.json`;
- `11_single_fix_if_authorized/no_fix_authorized.json`;
- `13_figures/a3_a4_scale_composition_remainder.svg` and source CSV;
- `13_figures/a4_remainder_sources.svg` and source CSV.

## Reproduction commands

```bash
python experiments/trace_a3_a4_carry_state.py --help
python experiments/run_a3_a4_same_prestate_substitutions.py --help
python experiments/audit_r35_dense_cni_parity.py --help
python experiments/audit_cni_composition_accounting.py --help
python experiments/finalize_complete_o4_carry_root_cause.py --help
```

## Next authorized action

Specify and independently validate an authoritative complete-O4 cross-step
symbolic-remainder contract before proposing any new carry implementation.
