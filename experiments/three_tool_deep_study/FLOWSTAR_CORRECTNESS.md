# Flow* cached-remainder correctness

The fixed-step/fixed-order `Flowpipe::advance` overload builds a complete first
Picard image with `Picard_ctrunc_normal`, stores intermediate polynomial ranges,
computes an interval for the polynomial difference, proves inclusion in the
configured candidate, and then calls `Picard_ctrunc_normal_remainder` during
refinement.

For Riccati `x'=x^2`, `[0,0.1]`, `h=0.01`, order 2, cutoff `1e-15`, and
candidate radius `1e-4`, stock Flow* returns raw endpoint upper
`0.10010008767642772`, below the analytic `0.10010010010010011`.

The source-level trace isolates the discrepancy:

- the initial polynomial-difference interval is approximately
  `[-1.30e-20, 8.68e-21]` and is unchanged when regenerated;
- multiplication polynomial ranges and cutoff contribution are unchanged;
- the cached remainder-only proposal omits about `1e-7` from the complete
  Picard remainder;
- the omitted quantity is the interval generated when a `NODE_VAR` Taylor
  model is degree-truncated by `ctrunc_normal`.

The full evaluator adds this leaf-truncation interval to the variable
remainder, but upstream `evaluate_remainder` returns only the variable's
pre-existing remainder. The intermediate-range list did not record the
leaf-truncation interval, so cached replay was incomplete after the candidate
remainder changed.

The audit branch now exposes three runtime-selectable paths:

- `flowstar_stock`: unchanged upstream replay, retained as audit evidence;
- `flowstar_full_picard_revalidated`: every cached proposal is treated only as a
  proposal; a new full Picard image, intermediate-range list, and
  polynomial-difference interval are generated, and the vector proposal is
  committed atomically only after complete inclusion;
- `flowstar_root_cause_patch`: records the variable-leaf truncation interval and
  consumes it in remainder-only replay.

The regression `tests/run_refinement_revalidation.sh` requires stock to
reproduce the known analytic miss, both corrected paths to contain the analytic
endpoint, and every endpoint to remain in its tube. No wrapper mutates a
returned remainder and no candidate is re-injected after `advance`.

The root-cause variant is the primary Flow* path because it repairs the exact
missing cached term while preserving native refinement. The full-Picard path is
the conservative cross-check. The study still labels both as audit-branch
variants rather than unmodified upstream Flow*.
