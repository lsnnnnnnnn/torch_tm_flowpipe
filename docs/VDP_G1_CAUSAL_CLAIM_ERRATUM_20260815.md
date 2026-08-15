# VDP G1 causal-claim erratum — 2026-08-15

Current scientific status:

`BOUNDED_SOURCE_MATERIALIZATION_CONTRIBUTION_CONFIRMED__TOTAL_T1_T3_CAUSE_OPEN__G1_TERMINAL_REGRESSION`

This document corrects the scope of the conclusion published in
`VDP_T1_T3_WIDTH_CAUSAL_REPORT_20260814.md`. It does not delete or rewrite the
underlying G1 measurements. In particular, the former unqualified label
`T1_T3_WIDTH_CAUSE_CLOSED__EARLY_GAP_IMPROVED__TERMINAL_STILL_OPEN` must be read
as a historical overclaim and is not the current project conclusion.

## What the G1 evidence establishes

G1's affine source is part of the real polynomial passed to the next dense
Picard solve. A payload perturbation changes the consumer output, whereas a
lineage-only metadata perturbation does not. This proves a bounded contribution
from accepted-boundary source materialization; it does not prove that this
mechanism accounts for the whole Flow*–Torch width difference.

The fresh fixed-schedule G1 reductions reported on 2026-08-14 are real but
small relative to the legacy excess over Flow*:

| checkpoint | reduction as a fraction of legacy excess, four raw channels |
|---|---:|
| T=1 | about 0.147%–0.465% |
| T=3 | about 0.211%–0.227% |
| T=6.32 | about 0.261%–0.307% |

These percentages use raw lower/upper subtraction for endpoint x/y and segment
tube x/y. They do not use coordinate-projection pixels, remainder width, normal
scale, or an apparent plotted zero.

At step 1 no old `J/Phi_L` source has crossed an accepted boundary, yet the
returned Torch and Flow* coefficients already differ. Accepted-boundary
materialization therefore cannot by itself explain the complete T=1 gap.

On frozen legacy prestates, retaining the G1 source identity is substantially
better than ordinary-materializing the same affine source set. Nevertheless,
the existing legacy rebox is slightly better than G1 in those isolated cells.
The small full-trajectory G1 reduction is therefore not evidence that G1 has
reproduced Flow* carry semantics.

G1's native run accepts through `6.382737816137232`, while legacy accepts
through `6.397083942944808`. G1 is a terminal regression and is not an
acceptable production improvement.

## Evidence-label correction

The thirteen artifacts formerly called "independent micro-oracles" directly
import the project's `Polynomial`, `TaylorModel`, dense Picard, interval, or
source-ledger implementation. Their correct label is
**project-core-backed exact/discrete micro-oracles**. They remain useful unit
and integration evidence, but they are not an implementation-independent
correctness oracle.

The Flow* lossless same-prestate operator cell remains `UNAVAILABLE`. Existing
fixtures prove Flow* native-state serialization round trips and native
one-step continuation, but Torch cannot losslessly consume the Flow* 3-state,
4-variable, nonempty `Phi_L/J` object; conversely the frozen Flow* operator
rejects Torch's 2-state object. No component-box adapter is admissible.
Accordingly, total causal attribution remains open unless and until all four
lossless cross-operator cells are executed and reproduce the observed raw
deltas with no unexplained residual beyond a declared rounding envelope.

## Evidence classes

- Canonical coefficient/byte equality and exact-rational algebra are
  formal/discrete evidence.
- MPFR direction and binary64 `nextafter` containment are directed-numerical
  evidence in their stated lanes.
- Widths, margins, horizons, runtimes, and intervention outcomes are
  deterministic empirical evidence.
- Sampling is sanity evidence only and proves neither containment nor causal
  closure.
- CUDA results are implementation-consistency/performance observations, not a
  formal directed-rounding claim.

