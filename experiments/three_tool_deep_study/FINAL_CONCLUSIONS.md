# Superseded smoke-only executive summary

This file was generated from the earlier integrated smoke run.  It is
**superseded and non-authoritative**.  The authoritative conclusion is created
only after the ten-repetition formal run and is published under
`experiments/three_tool_deep_study/FINAL_CONCLUSIONS.md`; its source tables,
plots, and canonical run report are retained under
`artifacts/authoritative/<run-id>/`.

The completed three-tool study passes its primary gates:
**True**.  It contains
90 analytic containment checks,
513 common-export point
containment checks,
252
native/export round-trip evaluations, and 0 selected
full-configuration runtime rows with at
least ten repetitions.

The literal question “which tool is best at order 1?” has no sound universal
answer because the tools' order labels select different bases.  Common affine
carry is the closest controlled carry protocol, but it is not used for a
relative winner because native local construction remains unmatched.

Box carry is a wrapping control that discards correlation; the exact width
ratios in `box_carry_summary.csv` may nevertheless fall below one when
re-normalization improves later interval conditioning.  Native low-order and
practical Pareto rows remain valid when labelled with their actual bases,
successful horizon, common absolute evaluation time, and numerical guarantee.
Flow*'s variable-leaf cache patch and full-Picard revalidation both eliminate
the stock Riccati under-enclosure; the corrected original Van der Pol
configuration reaches T=10, but its exported adaptive raw endpoints fail the
separate deterministic trajectory sanity check and are excluded from numerical
Pareto claims.

The matched-basis experiment shows what changes from B1/B_DR/B2/B3 inside one
engine.  The reset controls show why Torch's unchecked dependency carry
deteriorates; DiffReach gains from retained local-time cross structure and
symbolic normalization but omits general higher-order families; Flow* gains
from complete higher order, normalized composition, symbolic remainder, and
adaptation at the cost of MPFR/C++ workload.

Recommended Torch work: supported normalized affine/QR reset, a restricted
time-state basis, better polynomial range bounding and overflow attribution,
validator/runtime observability, and a strict directed-rounding backend.
The BERN range-only prototype contains all
5 analytic cases and tightens
2 cancellation cases.  It is
worth continuing only as a sparse, formally enclosed range backend; this
plant-only evidence does not motivate NN abstraction work or a fourth solver.

See `three_tool_deep_study_report.md` for tables, validity limits, and all six
research questions plus the eleven required final answers.


The complete generated report is stored with the timestamped results directory.
