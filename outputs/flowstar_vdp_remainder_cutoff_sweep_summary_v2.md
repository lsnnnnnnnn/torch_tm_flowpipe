# Plant-only Flow* comparison summary

Source CSV: `outputs/flowstar_vdp_remainder_cutoff_sweep.csv`

## Scope

This is a plant-only comparison of polynomial ODE flowpipe boxes. It is not a full CROWN-Reach NNCS run and it does not compare raw Taylor-model coefficients.

## Status counts

| tool | mode | status | count |
| --- | --- | --- | --- |
| flowstar | fixed | completed | 122 |
| flowstar | fixed | failed | 130 |

## Evidence that dependency-preserving propagation is useful

No paired validated torch rows were found.

Interpretation: a dependency/range width ratio below 1 means the dependency-preserving endpoint box is tighter for the same plant, step size, horizon, and Taylor order. A ratio above 1 is a useful regression signal for nonlinear cases where term growth/remainder accumulation dominates.

## Torch over Flow* ratios

Flow* endpoint boxes were not available. Flow* GNUPLOT-derived last-segment and tube boxes were parsed for 122 completed cases. Torch-vs-Flow* ratios below use last-segment/tube widths, not endpoint widths.

Parsed Flow* last-segment/tube boxes were found, but no matching validated torch rows had compatible semantic widths for ratio reporting.

## Validation note

The `containment_failures` column is a sampling-based regression sanity check. It is not a formal proof. Formal soundness claims should be limited to the implemented Taylor-model validation assumptions and the current floating-point prototype limitations.
