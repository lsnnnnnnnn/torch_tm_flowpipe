# TORA-Q3 common-control comparison

Status: **FORMALLY ALIGNED and VERIFIED to T20.**

This lane replays the exact per-period, per-leaf control intervals exported by
Xiangru.  At each one-second period, both plant kernels restart from the same
observed pre-controller boxes and propagate ten `h=0.1` segments.  It isolates
the plant, but it is not an independent closed-loop verification.

Both tools completed 200 segments, 48 leaves per segment, and the frozen
property.  The comparator aligned 96,000 scalar enclosures by exact lane,
segment, binary64 time value, controller period, leaf ID, state, and endpoint
versus tube kind.  No interpolation was used.  The corrected Torch trace hash
is `2dc6c3c5c689a2691d22b6ffaeb0ff9219dfe5e6058849f2ebda0f917849bbff`.

## T20 tightness

Ratios are **Torch width / Xiangru width**.  Below one means Torch is tighter.

| state | endpoint median | endpoint max | tube median | tube max |
|---|---:|---:|---:|---:|
| x1 | 0.963604 | 0.974832 | 1.025418 | 1.034496 |
| x2 | 0.916275 | 0.940894 | 0.969176 | 0.997193 |
| x3 | 1.00000000009 | 1.00000000012 | 1.012227 | 1.017730 |
| x4 | 1.00000000007 | 1.00000000009 | 1.036472 | 1.037685 |
| u1 | 1.00000000007 | 1.00000000008 | 1.00000000007 | 1.00000000008 |

Across all 2,000 time/state/kind aggregate rows, Torch has the smaller median
in 408 and Xiangru in 1,592.  The latter count includes near-equality rows that
differ only by Torch's explicit `nextafter`/contraction envelope; it should not
be read as a practically meaningful win for `u1` or the `x3/x4` endpoints.

The worst leaf/time ratios over the complete horizon are retained in
`comparison/worst_leaf_cases.csv`.  Notable tube maxima are 1.13124 for `x1`,
1.17001 for `x2`, 1.14761 for `x3`, and 1.12811 for `x4`; the winning method is
therefore not uniform over time or leaves.

## First divergence and cause

The first accepted-status divergence is absent.  The first numeric difference
is segment 1 `x1`: endpoint max lower/upper differences are `2.1739e-7` and
`3.8234e-6`; tube differences are `2.1739e-7` and `1.0082e-3`.  Leaf 0's first
remainder-width difference is `4.0129e-6` (Torch `1.4535266e-4`, Xiangru
`1.4936558e-4`).  Segment-1 coefficients may legally be compared because its
diagonal normalization and slot permutation are identical; max difference is
`5.1096e-10`.  Later coefficient comparison is intentionally unavailable
because a tested per-segment normalization conversion is not exposed.

The remaining behavior is classified as expected method difference:
independent analytic sine enclosure, fixed-support overflow and roundoff,
Picard remainder handling, and range factorization.  No validation condition
was relaxed to obtain T20.

Machine-readable evidence includes separate endpoint/tube tables, polynomial-
range and interval-remainder widths, property margins, selected leaf overlays,
target-horizon ratios, failure horizons, and a three-segment first-divergence
window under `outputs/tora_q3_native_matched_20260806/comparison/`.
