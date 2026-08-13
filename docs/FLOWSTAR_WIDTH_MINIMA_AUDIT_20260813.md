# Flow* width minima audit — 2026-08-13

Status: `BASELINE_CONCLUSIONS_REPRODUCED`, `FLOWSTAR_WIDTH_IS_POSITIVE_NEAR_ZERO`

Precise classification: `Z0_POSITIVE_WIDTH_ONLY_LOOKS_ZERO`.

## Frozen contract and replay

The replay uses the pinned complete-total-degree O4 Van der Pol contract:
`x'=y`, `y'=y-x-x^2*y`, initial box `x=[1.1,1.4]`,
`y=[2.35,2.45]`, B1, fixed `h=0.01`, target horizon 10, candidate
remainder `[-1e-4,1e-4]`, and cutoff `[-1e-10,1e-10]`. Flow* accepts
all 1000 fixed steps. Legacy Torch accepts steps 1–632 and rejects candidate
633 at the fixed minimum step; a process exit code is not used as a scientific
label.

All eight Flow*/Torch endpoint and one-segment width strings over the 632-row
common prefix are decimal-text identical to the historical artifact. The 16
published checkpoint ratios reproduce with zero binary64 deviation:

| t | channel | Flow* width | Torch width | Torch−Flow* | ratio |
|---:|---|---:|---:|---:|---:|
| .01 | endpoint x | 0.301112793636261 | 0.30089849698777327 | -0.00021429664848771068 | 0.9992883176901923 |
| .01 | endpoint y | 0.12263376681992 | 0.1213342911693931 | -0.0012994756505269045 | 0.9894036064925323 |
| .01 | segment x | 0.32524536254432035 | 0.32524722193229505 | 1.8593879747008657e-06 | 1.0000057168777448 |
| .01 | segment y | 0.1494059114443438 | 0.1494274734969525 | 2.156205260872568e-05 | 1.000144318604266 |
| 1 | endpoint x | 0.07951782806323093 | 0.08795592352408854 | 0.008438095460857609 | 1.1061157688329692 |
| 1 | endpoint y | 0.1115769155531654 | 0.11429217453027352 | 0.002715258977108115 | 1.0243353113289309 |
| 1 | segment x | 0.08375257163162497 | 0.09222151766116604 | 0.008468946029541069 | 1.1011186386824114 |
| 1 | segment y | 0.11966699301819556 | 0.12856523859517788 | 0.008898245576982322 | 1.0743583953482423 |
| 3 | endpoint x | 0.13850532673418248 | 0.18725958974064205 | 0.048754263006459575 | 1.352002801307621 |
| 3 | endpoint y | 0.10885167146482555 | 0.15586425555328454 | 0.047012584088458986 | 1.431895840052862 |
| 3 | segment x | 0.16392087538948363 | 0.21273503964284135 | 0.04881416425335772 | 1.29779101738795 |
| 3 | segment y | 0.1256837925656762 | 0.17276779494517935 | 0.047084002379503165 | 1.3746227052696502 |
| 6.32 | endpoint x | 0.15307555562376207 | 0.9165121028676759 | 0.7634365472439139 | 5.987318478989109 |
| 6.32 | endpoint y | 0.12229562798699911 | 1.5898587282667287 | 1.4675631002797296 | 13.000127268946539 |
| 6.32 | segment x | 0.1783272999656937 | 0.9420414425127686 | 0.7637141425470748 | 5.282654101161162 |
| 6.32 | segment y | 0.13982130901719003 | 1.6080698024787519 | 1.4682484934615618 | 11.500892201495919 |

## Raw lower/upper audit

Widths were recomputed as `upper-lower` from accepted Flow* bounds, once with
80-digit decimal arithmetic and independently with binary64. No stored width
column was trusted.

| channel | step/time | lower | upper | binary64 width | width hex |
|---|---:|---:|---:|---:|---|
| endpoint x | 397 / 3.97 | -2.012045351187811 | -2.0034332393764056 | 0.00861211181140531 | `0x1.1a33a14a2c700p-7` |
| endpoint y | 474 / 4.74 | 0.6511552479512857 | 0.6774278488867459 | 0.026272600935460244 | `0x1.ae7346731d420p-6` |
| segment x | 397 / 3.97 | -2.0121483697981315 | -2.003259658434527 | 0.008888711363604695 | `0x1.2343ea4e1d300p-7` |
| segment y | 474 / 4.74 | 0.6465414668530809 | 0.677429520722198 | 0.030888053869117083 | `0x1.fa11e34d1dce0p-6` |

For each of the four 1000-row sequences, the counts of exact decimal zero,
binary64 zero, subnormal, `<1e-16`, `<1e-12`, and `<1e-9` are all zero. The
minimum-context artifact contains ±20 accepted steps per channel. There are no
missing or duplicate accepted indices, empty numeric fields, NaN/Inf values,
post-failure zero fills, or mismatched times.

## Data lineage

The values follow this actual path:

`Flowpipe::advance` accepted `result.tmvPre/domain`
→ `Flowpipe::compose_normal`/`intEvalNormal`
→ the probe reads `Interval::inf/sup` with MPFR-directed conversion
→ `format_double` writes 17 significant decimal digits
→ strict accepted-index/time parser
→ matched common-prefix rows.

The probe reads distinct objects for `tau=h` endpoint and `tau∈[0,h]` segment
tube. `TaylorModel::intEvalNormal` adds the stored remainder to the polynomial
range. `Interval::inf/sup` use `MPFR_RNDD`/`MPFR_RNDU`. The decimal strings remain
different at the minima, and the historical comparison CSV contains the same
positive values. No `abs`, nonnegative clip, smoothing, rounding-to-zero, or
missing-value substitution occurs. The first zero/near-zero layer therefore
does not exist.

## Independent falsification replay

At `t=3.97`, `4.74`, and `6.32`, 81 initial grid points were integrated with
SciPy DOP853 (`rtol=1e-12`, `atol=1e-14`) and nine corners/boundary/interior
points were independently integrated with the 70-decimal-digit mpmath Taylor
solver (`tol=1e-45`, degree 40). None of 540 component observations left the
corresponding Flow* endpoint box.

This result is exactly
`NO_NUMERICAL_CONTAINMENT_WITNESS_IN_TESTED_POINTS` and
`NOT_AN_ENCLOSURE_PROOF`. Sampling does not close Flow* soundness.

The center variational matrix remains nonsingular at all three checkpoints.
At 3.97 its linearized x-projection radius is about 0.00205 while the y radius
is 0.03969; at 4.74 the y radius is about 0.00895. The smallest singular values
are respectively 0.01034 and 0.00344. This is evidence that the displayed local
minima are plausible coordinate-projection contraction, not collapsed state
sets.

## Causality and ratio interpretation

The x minima occur at 3.97 and the y minima at 4.74, so the four channels do not
share one causal event. Flow* and Torch execute independently: a Flow* minimum
cannot modify Torch state. At each minimum the guarded denominator threshold
`1e-9` remains satisfied, but the ratio is visually amplified by the contracted
Flow* projection. Absolute excess has already increased throughout the prior
20-step window in all channels. Torch absolute width itself does not increase
monotonically in all those windows, so “the minimum starts Torch explosion” is
not supported.

The strongest common late change in rolling absolute Torch growth occurs near
step 612 for three channels; the segment-y heuristic selects an earlier
oscillatory change near 316. The scientifically stable statement is that the
cross-tool carry difference exists by T=1 and T=3 and accumulates into large
absolute excess by 6.32. The Flow* minima change when this looks dramatic on a
ratio plot, not when Torch's mechanism begins.
