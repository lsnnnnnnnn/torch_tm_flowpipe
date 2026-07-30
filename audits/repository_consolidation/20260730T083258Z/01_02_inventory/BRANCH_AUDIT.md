# Branch audit

## Scope and evidence

This audit enumerates every local branch, remote branch, tag, and worktree after
a fresh `git fetch origin --prune --tags`. The remote had 12 branch heads and
no tags. The local clone added one local `main` ref, so
`branch_inventory_before.csv` has 13 rows.

The decision does not use branch names, dates, commit messages, or prior status
claims as proof. Evidence is recorded in:

- `branch_inventory_before.csv`
- `branch_pairwise_relationships.csv`
- `branch_diff_evidence.txt`
- `commit_patch_ids.csv`
- `patch_equivalence.csv`
- `branch_feature_matrix.csv`
- the Phase 0 `git_graph.txt` and `git_ls_remote.txt`

Stable patch IDs found no separately-created patch-equivalent commit pairs.
Equivalent history is represented by shared ancestry rather than rewritten
commits.

## Ancestry findings

`origin/codex/torch-flowstar-diffreach-deep-study` is a strict descendant of
`origin/main` and contains 95 non-merge commits beyond main. It contains these
remote tips as ancestors:

- `codex/flowstar-ctrunc-rescue-diagnostics`
- `codex/flowstar-kernel-alignment`
- `codex/flowstar-normalized-insertion`
- `codex/flowstar-raw-remainder-compat`
- `codex/first-order-three-way-benchmark`
- `codex/first-order-followup-correctness-matched-basis`
- `codex/three-way-common-contract-comparison`
- `codex/three-way-comparison-correctness-repair`

`codex/batched-dense-nncs-gpu` diverges after the shared normalized-insertion
lineage and has three unique commits. Those commits add or extend a batched
dense NNCS/GPU prototype, its demos, tests, and result bundles. NNCS and
controller work are outside the authoritative consolidation scope. The
canonical deep-study lineage already contains the earlier batched-dense
prototype, so the three divergent commits are preserved for archive but are
not migrated into active code.

`origin/master` has unrelated history and a single small initialization commit.
It is not assumed redundant and remains `unknown_keep`.

`codex/flowstar-ctrunc-rescue-diagnostics` is three commits behind main with no
commit beyond main. It is fully represented by descendant history.

## Classification

| branch | classification | decision before acceptance |
| --- | --- | --- |
| `main`, `origin/main` | `protected_or_default` | retain; update only by normal merge after all gates |
| `codex/torch-flowstar-diffreach-deep-study` | `canonical_candidate` | use its tip as the integration base, then repair it |
| `codex/batched-dense-nncs-gpu` | `abandoned_prototype` | archive; do not migrate out-of-scope NNCS/GPU work |
| `codex/flowstar-ctrunc-rescue-diagnostics` | `fully_redundant_patch_equivalent` | archive; remote deletion permitted only after final acceptance |
| eight other `codex/*` lineage branches | `superseded_but_scientifically_relevant` | retain through archive tags until final acceptance |
| `origin/master` | `unknown_keep` | retain |

The CSV is authoritative for the exact tip SHA, merge base, ahead/behind
counts, tree hash, largest file, path classes, and evidence for each row.

## Feature findings

The feature matrix shows that main has the single core interval, polynomial,
Taylor-model, one-step, and multi-step implementation, while the deep-study
lineage accumulates the Flow* compatibility work, raw-remainder diagnostics,
normalized insertion, first-order three-tool adapters, matched-basis study,
common contracts, correctness repair, and deep-study Pareto machinery.

No branch is deleted in Phase 1. Unique branch content remains reachable from
the original remote ref until an annotated archive tag is pushed and verified.

## Post-acceptance convergence

Formal run `20260730T153654Z` passed independent acceptance at frozen code SHA
`0dfdf587ee0fb9cff374dbc41ecdf17dfa2bf781`; publication commit
`269f3599fcc480984ff651c6c1e083a8ceac74e9` was then pushed to
`origin/codex/repository-consolidation-v1`.

All 11 non-main historical heads received annotated
`archive/pre-consolidation-20260730/*` tags. A post-push `ls-remote` verified
each tag object and peeled target. Only
`codex/flowstar-ctrunc-rescue-diagnostics` was deleted: it is non-protected,
appears in no worktree, has zero commits beyond canonical, is an ancestor of
canonical, has no unmigrated unique content, has audit evidence, and its tag
peels to exact tip `bf19292a12bee4695bfc4f44a24be4ea0d341848`.

The other 11 original remote heads remain. Adding the canonical continuation
and deleting the one redundant head leaves 12 remote heads. `main` is
unchanged. Exact before/after inventories, retained/deleted lists, archive
tags, and post-convergence refs are under `09_branch_convergence/`.
