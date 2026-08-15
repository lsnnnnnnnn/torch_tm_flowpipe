# VDP G2 shared-column result (2026-08-15)

## Scientific decision

`G2_MECHANISM_IMPROVED__PRODUCTION_GATE_NOT_MET`

The total-cause decision remains independently
`LOSSLESS_CROSS_OPERATOR_CELL_UNAVAILABLE__TOTAL_CAUSE_OPEN`.

The only candidate evaluated was
`normalized_insertion_bounded_shared_source_o4_g2`: fixed `3d` variables,
exactly two source generations, complete current-generation source-polynomial
identity retained across x/y and nonlinear paths, and oldest-only canonical
collapse.  No generation count, owner subset, K, order, step, target, cutoff,
range budget, validator, or initial box was swept after observing the result.

## Correctness result

The standard-library-only exact rational oracle passes all 15 checks.  It is
separate from the project-core tests and imports no project polynomial,
Taylor-model, dense Picard, interval, or source-ledger implementation.  The
project integration tests establish complete-ledger containment, six-variable
shape, shared x/y source identity, oldest/current mixed retirement, no double
count, accepted/rejected atomicity, and v4 checkpoint serialization.

A fresh/resume audit saves a G2 checkpoint after 10 fixed steps, reloads it,
continues both paths through 20 steps, and compares every continuation hash and
the final checkpoint bytes.  It also forces a rejected retry and compares the
prestate fingerprint and retained payload.  All comparisons pass.  The legacy
mode remains the default and the exact-decimal experiment lane is opt-in.

These are formal/discrete and directed-numerical obligations at their stated
scope.  They do not turn ordinary CUDA arithmetic into a formal directed-
rounding implementation.

## Fixed h=0.01 result

The four channel order in the following compact tables is endpoint x,
endpoint y, segment-tube x, segment-tube y.  Every width is recomputed from raw
upper minus lower; no projection or remainder-width surrogate is used.

| checkpoint | G1 widths | G2 widths |
|---|---|---|
| T=1 | `0.087932166410, 0.114279551894, 0.092197701013, 0.128552124264` | `0.087929432328, 0.114277198452, 0.092194966294, 0.128549639943` |
| T=3 | `0.187149350766, 0.155763886087, 0.212624450324, 0.172668567055` | `0.187130492095, 0.155732561961, 0.212605586905, 0.172637230147` |
| T=6.32 | `0.914521768614, 1.585357251109, 0.940050086809, 1.603569103850` | `0.913141302963, 1.582088392700, 0.938669202759, 1.600298982613` |

G2 is narrower than G1 in all four T=1 channels, so production criterion 1
passes.  It also remains narrowly better than G1 at T=3 and T=6.32.

The decisive criterion is reduction of the legacy excess over Flow*.  At T=3,
G2 removes only about `0.265%–0.280%` of that excess across the four channels.
At T=6.32 it removes only about `0.442%–0.529%`.  All eight values are far below
the preregistered `10%` requirement.  Thus criterion 2 fails by roughly two
orders of magnitude.  All fixed prefixes complete without a G2 validator or
containment failure, the variable count remains six, and the term count stays
bounded rather than growing with horizon.

The evidence package contains the full 632-step four-channel curve for
Flow*/legacy/G1/G2, including raw bounds, excess, ratios, increments, reductions,
and separate 1.1/1.5/2/5 ratio crossings.  The 36-request matrix also executes
fresh independent requests for each of T=0.1, 0.5, 1, 2, 3, 6.32 and native
T=1, 3, 6, 6.5, 7.5, 10 for all three Torch lanes.

## Native result

| lane | highest continuous validated time in T=10 request | reached T=10 |
|---|---:|---:|
| legacy | `6.397083942944808` | no |
| G1 | `6.382737816137232` | no |
| G2 | `6.384691066788196` | no |

G2 recovers `0.0019532506509643` beyond G1, but remains
`0.012392876156611443` earlier than legacy.  It therefore fails production
criterion 4 and does not validate T=10.  The final G2 y subset margin in the
diagnostic T=10 request is `-1.7971838005011962e-6`; the package compares this
against the fresh legacy terminal margin without conflating native and fixed
schedules.

## Owner interpretation

G2 behaves in the direction predicted by Gate B: retaining one additional
generation removes a small amount of dependency loss.  The effect remains
small because the recoverable retired source polynomial is tiny next to the
cumulative ordinary parameterization mass.  This is a genuine mechanism
improvement, not a production result and not evidence that Flow* carry has been
reproduced.

The result argues against extending this route by tuning three generations or
an owner subset.  Those experiments were explicitly forbidden and were not
performed.  With the total-cause cross cells still unavailable and the fixed
gain far below threshold, the next useful work is on the local Picard
construction/range/validator contract or on a genuinely lossless cross-tool
operator representation—not a broader shared-source sweep.

## CPU/CUDA scope

CPU float64 B1 is authoritative.  The V100 experiment uses the same exact
10-step B1 workload and synchronized phase timers for H2D boundary conversion,
dense Picard/range/validator work, D2H boundary conversion, non-kernel solver
work, full solver runtime, and the instrumented bridge transfer count.  CUDA is
reported only as implementation consistency and measured performance.  No
kernel-only number is extrapolated into a solver speedup.

On the measured 10-step T=0.1 workload, CPU solver time is
`10.770035097026266 s` and V100 solver time is `24.921248865983216 s`, for a
CUDA/CPU full-solver speedup ratio of `0.43216273610296685`.  The synchronized
CUDA phase ledger records `0.2880250640155282 s` H2D,
`6.392799848050345 s` dense Picard/range/validator work,
`0.2692561090225354 s` D2H, and 20 instrumented sparse/dense bridge transfers.
CPU and CUDA raw endpoint and segment fields agree under the implementation-
consistency comparison, but CUDA is slower and no speedup is claimed.

## Plain-language answers

- This round proves that the fixed two-generation source column is real,
  bounded in shape, restartable, and slightly beneficial.
- The widths and horizons are deterministic empirical observations; they do
  not prove total causal closure or universal soundness.
- The T=1/T=3 total residual remains numerically unidentifiable because the two
  lossless cross-operator cells are missing.
- G2 is better than G1 at the selected fixed checkpoints and slightly later in
  native time, but it is not meaningfully better than legacy under the frozen
  success threshold.
- Native G2 does not exceed `6.397083942944808` and does not reach T=10.
- Further shared-source tuning is not justified by this preregistered result;
  local Picard/range operator work is the more informative next direction.
