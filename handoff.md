# Handoff: TORA-Q3 performance and closed-loop closure

This is the current in-progress handoff for
`codex/tora-q3-performance-closed-loop-closure-20260806`. It replaces the
bootstrap file that described only the dirty historical source worktree.

## Current branch state

The active branch descends from clean tip `7dcbe7cd901a941bd7508a107ecb0cc6f877ca1f`
and root `9fc45344c4379422244b75af705dffd17304f824`. It has no merge base with blocked
historical tip `c49d74bbf48d1004f7f3818174e7f40b6200b142`. The dedicated worktree was
created because the user's main worktree was dirty; those unrelated changes
were not modified.

Phase 0 makes the branch self-contained for its stated TORA-Q3 review surface:
README links and command paths are machine-checked, the quick-start example
runs, portable core/TORA regression coverage is present, optional external
assets skip only when unset and fail closed when supplied incorrectly, and the
artifact scanner covers the entire tracked tree, untracked inventory, and
every path/blob reachable from clean-lineage `HEAD`.

Validation records are deliberately separate:

- `source_worktree_historical_validation`: `506 passed, 6 skipped` (reported by
  the earlier dirty source worktree; not rerun or relabeled here);
- `clean_branch_portable_validation`: bootstrap `52 passed, 14 skipped`;
  current Phase 0 `59 passed, 14 skipped` in `py11`.

## Facts carried forward, not yet remeasured

The prior common-control T20 result is 20 matched one-period plant replays.
Every controller period restarts from the Xiangru-observed pre-controller box
and held-control interval. It is not an independent Torch closed loop.

The prior native full loop certified through `T=4.3` and failed closed at
segment 44 (`T=4.4`) when leaf 0's `x3` tube crossed the unchanged `[-2, 2]`
property. The earlier aggregate did not separate the numerical self-map
certificate from the property predicate; this branch will do so before making
a new root-cause statement.

The historical common-control steady medians were Torch `525.862164 s` and
Xiangru `1.033485 s`, a descriptive quotient of about `508.824`. That is not a
formal implementation speedup because the software stacks differed and the
Torch profile contained `128,472` scalar synchronization events. The same-stack
baseline, source-line attribution, optimized timings, and environment/runtime
decomposition are pending the later phases; no value has been invented here.

The earlier T=1 controller input difference of `0.014211...` has not yet been
assigned to endpoint range, affine projection, carry materialization, or
controller-input construction. R1/R2 deterministic lifecycle replay and a
sound reconditioning candidate remain required before the full-loop gates are
rerun.

The independent VDP issue near `t=6.397083942944808` remains unresolved and is
outside this TORA work. Nothing in this branch claims otherwise.

## Public/private boundary

Public Git may contain implementation, tests, sanitized aggregate CSV/JSON,
hashes, reports, and a manifest. Controller/ONNX bytes, raw per-leaf traces,
observer patches, profiler raw traces, server paths, full environment dumps,
and raw logs remain under the separate private evidence root. The final
handoff will replace the pending measurements above with exact optimized T20,
host-sync, width-attribution, hierarchical-gate, test, manifest, remote ref,
and next-bottleneck results.
