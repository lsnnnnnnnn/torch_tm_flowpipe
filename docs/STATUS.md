# Status

Current state: `formal_accepted_branch_convergence_pending`.

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
- old run `20260730T015245Z` marked provisional.

Pending at this checkpoint:

- commit and push the formal artifact and canonical branch;
- create and remotely verify the remaining annotated archive tags;
- delete only the one fully redundant patch-equivalent remote branch;
- refresh after-inventory and main merge plan.
