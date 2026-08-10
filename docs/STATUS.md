# Status

Date: 2026-08-10
Round: structured remainder and compiled fixed-support closure
Decision: active; prior evidence package closed, new algorithm gates pending

## Gates

| gate | status | evidence |
|---|---|---|
| repository anchors | pass | remote start `05ae30b4`; isolated worktree; pinned Flow*/DiffReach SHAs |
| previous evidence closure | pass | 851-source-file recovery inventory; groups 03–08 tracked; clean-copy rebuild passes |
| claim taxonomy | pass | completion, certificate, soundness, formal/performance/ranking eligibility split |
| functional fixed core | pending | cached plan and tensor-only loop not yet implemented |
| compiled fixed core | pending | no performance result claimed yet |
| outward fixed reference | pending | one-step 2-ULP claim remains frozen |
| structured S1 | pending | exact semantics must precede implementation |
| terminal/horizon/generalization | pending | no new horizon run is eligible yet |
| delivery | pending | final tests, artifacts, commits, push, and handoff remain |

## Scientific status

- Fixed-support RQ1: qualified at explicit float64 operator level; full stock
  driver differs slightly because several upstream builders remain float32.
- Complete-support baseline: validates through
  `6.397083942944808`, then fails y inclusion with margin
  `-1.99995911680722e-5`.
- First native Flow*/Torch split: raw candidate Picard remainder at the first
  divergent h, not transformed polynomial coefficients or roundoff.
- F1 complete carry: sound, exact, generic primitive; fails at
  `0.04345468750000001`; rejected and non-default.
- GPU: no measured end-to-end or kernel speedup after synchronization and
  validation accounting; no speed or Pareto claim.

The next research operation is a bounded structured-symbol carry for the
terminal integration-overflow and polynomial-truncation terms, with
deterministic sound collapse and a frozen-prestate paired replay before any
horizon sweep.

The verified starting baseline in the current worktree is `469 passed, 2
skipped in 72.48 s`; the two skips are the declared optional external
integration tests. The Phase-A package regression adds a clean temporary copy,
checksum/manifest validation, and deterministic figure/table rebuild without
running numerical reachability jobs.
