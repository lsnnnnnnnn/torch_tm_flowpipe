status: withdrawn
valid_for_commit: unknown
superseded_by: docs/history/WITHDRAWN_RESULTS.md
allowed_use: provenance only

# Historical result invalidation

The report in
`experiments/three_way_common_contract/results/20260724T132534Z` is a preserved
preliminary artifact, not a valid three-way ranking.

Three experiment-changing confounders invalidate its headline comparisons:

1. The generated Flow* harness replaced every successful native returned
   remainder with the configured candidate remainder after `advance`. This
   moved and widened the reported enclosure and concealed the stock refined
   result.
2. Torch's reported `final_tm` used an extra fixed-time residual tightening,
   while the Flow* and DiffReach endpoints were direct substitutions in their
   validated segment enclosures.
3. Flow* was judged through a constrained fixed-step, minimum-order wrapper
   with generic failure reporting, even though the actual upstream Van der Pol
   benchmark uses adaptive steps and order 4 and reaches `T=10`.

Common-box carry also resets all cross-step structure. It is a useful controlled
representation experiment, but it cannot establish the quality of each tool's
native dependency-preserving mode.

The old numeric files are retained byte-for-byte for reproducibility. The new
runner hashes every file before and after a run and regenerates the old report
and figures in a separate output directory. All scientific conclusions must
instead cite a timestamped result under `three_way_comparison_repair/results`.
