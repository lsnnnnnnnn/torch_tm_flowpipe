# Flow*–Torch complete-O4 causal factor split — 2026-08-13

Status: `CAUSAL_FACTOR_SPLIT_PARTIAL` and
`SOURCE_MECHANISM_CANDIDATES_LOCALIZED_CAUSAL_SPLIT_OPEN`.

The frozen baseline reproduced from fresh processes: Flow* accepts 1000 steps;
legacy Torch accepts 632 and rejects candidate 633 at pre-time 6.32 with y
subset margin `-8.441898798404161e-06`.  Committed and fresh traces are exactly
equal in every scientific decimal field and parsed binary64 value; only the
per-step runtime field changes.

## Flow* queue factor

All four cells use the same actual stock symbolic-remainder overload and differ
only in queue `max_size`.

| cell | first published difference from Q100 | accepted steps | horizon |
|---|---:|---:|---:|
| F-Q1 | 3 | 620 | 6.20 |
| F-Q2 | 3 | 640 | 6.40 |
| F-Q10 | 11 | 685 | 6.85 |
| F-Q100 | none | 1000 | 10.00 |

All cells are bitwise equal at step 1 and step 2.  The first Q1/Q2 published
difference is step 3 because their reset first changes the state used after
the step-2 boundary; Q10 first differs at step 11.  The Q100 boundary finite
differences at 99/100/101, 199/200/201, and later reset boundaries are
reproducible, but they are superposed on the oscillator's changing local
growth.  They do not by themselves identify a unique cross-tool source line.
The no-symbolic-remainder overload was not mixed into this factorial.

## Torch Horner × queue diagnostic

Dispatch fields prove that each cell entered its requested implementation.
All hidden frozen-contract fields and the tracked diff hash are equal.

| cell | composition | diagnostic queue | accepted steps |
|---|---|---|---:|
| T-D0 | direct monomial | off | 632 |
| T-H0 | Horner | off | 636 |
| T-DQ | direct monomial | `flowstar_linear_v2` | 632 |
| T-HQ | Horner | `flowstar_linear_v2` | 636 |

At step 632, the deterministic factorial main effects on published widths are:

| channel | Horner main effect | queue main effect | interaction |
|---|---:|---:|---:|
| endpoint x | -0.12458650964307494 | 0 | 0 |
| endpoint y | -0.31534328132967304 | 0 | 0 |
| segment x | -0.12516813330957471 | +0.10016443010209825 | -0.001088096264500682 |
| segment y | -0.31583275400567 | +0.06520079964426717 | -0.0007068880948035705 |

Horner first changes published widths at step 3 and extends this diagnostic
run by four accepted steps.  The diagnostic queue first changes segment widths
at step 2, but endpoint widths, next scales, and failure horizon remain exactly
unchanged in both composition levels.  It therefore adds segment uncertainty
without acting as a production-equivalent Flow* old-source queue.  Neither a
narrower result nor a longer horizon establishes soundness.

## Step 1 versus later history

At step 1, Torch/Flow* ratios are `0.9992883176901923`,
`0.9894036064925323`, `1.0000057168777448`, and `1.000144318604266` for
endpoint x/y and segment x/y.  Segment polynomial widths agree to about
`4.8e-15` (x) and `1.3e-14` (y).  Torch's final residual is wider by
`1.8594e-6` and `2.1562e-5`, which explains the slightly wider Torch segments.
At `tau=h`, however, Torch's polynomial endpoint is narrower by about
`2.1616e-4` and `1.3210e-3`; this dominates the larger residual and produces
the narrower Torch endpoints.

The returned step-1 Picard support is the same, but 23 of 31 coefficients are
bitwise different.  This happens before any old `J/Phi_L` source can cross an
accepted boundary.  The earliest localized candidate operator is therefore
local Picard polynomial arithmetic/grouping—Flow*
`Continuous.cpp:2328-2343` and Torch
`batched_dense_tm.py:dense_polynomial_picard`—followed by remainder refinement
and endpoint/tube extraction.  It is not causally closed to a unique source
line because the full cross-operator same-prestate cells are unavailable.

After step 1, the scale used by step 2 is already different:
Flow* `(0.15044966009214522, 0.060913584414125518)` versus Torch
`(0.15045059849388548, 0.06092414958140347)`.  From then on, incoming history,
queue, composition, local TM arithmetic, Picard validation, and range
extraction interact.  The experiments quantify queue and Horner effects inside
each tool, but do not justify assigning the complete long-horizon gap to either
factor.
