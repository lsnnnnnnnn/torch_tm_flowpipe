# DiffReach / Torch DR7 matched comparison

Date: 2026-08-11

## Outcome

Pairwise outcome: `VALID_PAIRWISE_COMPARISON_CLOSED`.

## Eligibility

Explicit-f64 operator equivalence is eligible; stock mixed-builder-dtype and
ordinary multi-step float64 soundness retain their qualifications.

## What is comparable

DR7 support, two Picard constructions, ten DR-RP rounds, masks, retained
intervals, endpoint/tube, and J/Phi carry in the explicit-f64 fixture.

## What is unavailable

Stock full-step tubes, stock later masks, bitwise stock/full-driver identity,
and a matched cross-framework timing ratio.

## Negative results

The one-step exact replay needs up to a two-ULP companion envelope; ordinary
binary64 is not universally outward-qualified.

## Exact evidence paths

`outputs/three_tool_matched_divergence_fixed_support_20260811/20260811T100304Z/`
directories `04_native_diffreach/` and `06_native_torch_fixed_dr7/`; fixture
`tests/fixtures/diffreach_dr7_vdp_one_step_float64.json`.

## Three required rows

| field | stock DiffReach | DiffReach explicit-f64 fixture | Torch fixed DR7 explicit-f64 |
|---|---|---|---|
| source | `dd628eb443b517d6415de93e7035b4baef73963e` | same pinned operators | descriptor implementation `0c9512d59ee5b6c2a7fd0d45868cbaed6c84a98a` |
| builder dtype | mixed: model/identity/symbolic defaults include float32 | float64 forced at every builder | float64 explicit throughout |
| support | native c/L/Lt seven slots | exact `[1,t,xi0,xi1,t^2,t*xi0,t*xi1]` | same slots and frozen DR7 SHA |
| partition/schedule | B64, h=.01, T10 | frozen one-step B2 operator fixture | B64, h=.01, T10 plus one-step fixture |
| result | completed | operator fixture bit-exact | completed; matched fixture bit-exact |

The frozen fixture SHA256 is
`4d6901b205eb606847848582727559652082efb61d32352e9e110ba694155390`.
It compares both polynomial Picard constructions, the initial inclusion mask,
all ten DR-RP masks and accepted lo/hi arrays, retained coefficients, endpoint,
and full-step tube.  The package recreates these fields with the pinned JAX
operators while forcing float64 at every builder.  Normalization and symbolic
J/Phi carry are covered by the separate R7 object/functional regression, not
claimed as fields of this JSON fixture.  The current R7 object regression is
bit-exact after adding the generic R35 descriptor.

## Full-driver boundary

Stock DiffReach completed 1,000 steps with all 128,000 returned initial
component flags true.  Its API saves endpoints because
`BOUND_TIME_STEP=True`; a full-step tube and later DR-RP retain masks are not
exported.  Torch DR7 completes the same B64/h/T10 contract and additionally
records all 1,280,000 later component masks, the last full-step tube, prefix
tube, and the symbolic carry object.

The stock T10 endpoint differs from Torch by roughly `2.2e-6` to `3.7e-6`
per bound.  This is not a failure of the bit-exact operator fixture: the stock
full driver is mixed-builder-dtype, while the matched fixture and Torch row
are explicit float64.  A bitwise stock/full-driver equivalence claim is not
made.

## CPU, CUDA, and soundness scope

The fixture and short-horizon decisions pass on CPU and V100 CUDA; CUDA
values agree within the declared float64 tolerance and decisions agree.  The
ordinary Torch lane remains `empirically sampled only`.  An independent exact
rational one-step replay finds at most two ULP of outward expansion is needed;
only that companion envelope is `independently outward replayed for exact
benchmark workload`.  It is not silently substituted into the 1,000-step
ordinary run and is not a universal GPU rounding proof.

## Runtime scope

Stock DiffReach reports JAX warmup/compile and after-JIT timings; Torch's
historical CPU T10 timing included separate verification load.  They do not
form a matched one-cold/ten-warm measurement, so no speed ratio is emitted.
The semantic comparison is closed independently of performance eligibility.
