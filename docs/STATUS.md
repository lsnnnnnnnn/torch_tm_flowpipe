# Status

Current state: `branch_converged_final_quality_pending`.

Completed:

- all refs/worktrees inventoried and classified;
- selected base `9a684d9106633e067bfac0747244b769fa49aa0b`;
- verified pre-consolidation archive tag pushed;
- base suite passed with 270 tests and three explicit CUDA skips;
- versioned schema, fail-closed eligibility, canonical config identity, total
  runtime boundary, and post-filter Pareto implementation added;
- local Torch, DiffReach, and Flowstar smoke pipeline independently accepted;
- a first formal attempt completed all numerical gates but correctly failed
  repository hygiene when a legacy diagnostic wrote outside its output;
- the side effect is fixed and the corrected smoke passed at source SHA
  `9bef0ac87544aa97a8088c32e2a6e5cc2ab830a5`;
- a second formal attempt passed its then-current auditor, but manual semantic
  review found that Pareto dominance was incorrectly partitioned by tool; the
  bundle is rejected and the cross-tool grouping now has a regression test;
- cross-tool Pareto smoke passed at
  `2d870f6fd12595eed0a23da59f945986a310e245`;
- final code freeze recorded at
  `0dfdf587ee0fb9cff374dbc41ecdf17dfa2bf781`;
- formal run `20260730T153654Z` completed from a fresh directory with 24 exact
  configurations, 264 raw observations, zero failures/exclusions, and all
  checksum and independent-audit gates passing;
- the accepted formal artifact is authoritative for the bounded claims in
  `docs/RESULTS.md`;
- canonical publication commit `269f3599fcc480984ff651c6c1e083a8ceac74e9`
  is pushed to `origin/codex/repository-consolidation-v1`;
- 11 annotated archive tags are pushed and their remote peeled targets match
  every non-main historical head;
- the fully redundant
  `codex/flowstar-ctrunc-rescue-diagnostics` remote branch was deleted after
  all eight safety conditions passed; its verified archive tag is the exact
  recovery point;
- the other 11 original remote heads are retained, one canonical head is
  added, and the final remote still has 12 heads;
- `main` remains unchanged at `b2f34f5b2077e34662a2559d8c09b1d264bd7d98`;
  a conflict-free, reviewed fast-forward plan is documented;
- before/after inventories and raw post-convergence refs are recorded under
  `audits/repository_consolidation/20260730T083258Z/09_branch_convergence/`;
- old run `20260730T015245Z` marked provisional.

Pending at this checkpoint:

- execute the final compile, marker, shell, path, size, secret, and Git hygiene
  checks;
- commit and push the convergence and final-quality evidence.
