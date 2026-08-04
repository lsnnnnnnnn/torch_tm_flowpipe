status: diagnostic
valid_for_commit: unknown
superseded_by: docs/flowstar_order2_vanderpol_failure.md
allowed_use: diagnostic only

# Flow* failure-path audit

The generated harness calls the fixed-step, fixed-order
`Flowpipe::advance(..., vector<Expression<Real>>, ...)` overload. The audit
worktree instruments this overload without changing the stock mathematical
path. Each trace includes dimension, domain, step, order, cutoff, candidate,
first Picard remainder, `intDifferences`, subset result, refinement boxes and
ratios, returned remainder, source line, and return reason.

Failures are mapped to the following exhaustive schema:

- `order_configuration_rejected`
- `first_picard_inclusion_failed`
- `candidate_remainder_too_small`
- `fixed_order_exhausted`
- `adaptive_order_max_reached`
- `fixed_step_validation_failed`
- `adaptive_step_min_reached`
- `refinement_non_subset`
- `nonfinite_polynomial`
- `nonfinite_remainder`
- `composition_failure`
- `extraction_failure`
- `wrapper_failure`
- `unknown_internal_failure`

The last category is permitted only with the raw compiler/process/trace
diagnostics attached. No report uses “Flow* collapsed.”

The original fixed-order early horizons are configuration-specific validation
failures, not a tool-wide limit. Candidate-radius, cutoff, order, and step
sweeps identify the exact first failed step and category in the final
`corrected_failure_horizon_summary.csv`. The known-working adaptive original
benchmark is a separate sanity protocol and reaches `T=10`.

The no-refinement variant is explicitly enabled by
`FLOWSTAR_AUDIT_DISABLE_REFINEMENT=1`; it stops after the already successful
first inclusion and is never ranked. The full-revalidation variant is likewise
diagnostic and restores the initially accepted remainder only when a regenerated
full Picard self-map test rejects the refined box.
