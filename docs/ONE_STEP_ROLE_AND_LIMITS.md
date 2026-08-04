# One-step role and limits

One-step traces are diagnostic, never a substitute for the native end-to-end
success condition.  This run first completed Xiangru's CROWN-Reach/Flow* T=20 and
complete-Q3 B48 T=20 paths and stock Flow*'s official VDP T=10 path.  The saved
Xiangru shared-step trace can therefore be considered only after those facts are
closed.

The existing Xiangru entrypoint
`experiments/remainder_ablation/run_m4_flowstar_shared_step_export.py` reads a
Flow* trace produced with the repository's observation-only patch.  The patch is
numerically intended to observe internals, but it still changes source identity.
Any fresh analysis of that trace is consequently recorded under `diagnostics` as
`patched_diagnostic_only`; it cannot enter the native matrix, certify T=20, or
support a runtime/tightness headline.

The existing scalar-affine check has the same restricted role.  It is retained to
test the known Flow* endpoint/collapsed-path correctness blocker.  No epsilon,
post-hoc hull, endpoint replacement, new generated harness, or model rewrite is
allowed to make the failing enclosure pass.

No new one-step model or adapter is added on this branch.  Existing raw fields are
read exactly as named: an absent endpoint is not filled from a tube or segment,
and an absent internal remainder component is reported `not_exposed`.
