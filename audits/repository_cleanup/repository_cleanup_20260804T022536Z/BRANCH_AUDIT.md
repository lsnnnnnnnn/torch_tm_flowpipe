# Branch, worktree, and tag audit

The authoritative row-level evidence is `branch_inventory.csv`, `branch_feature_matrix.csv`, `branch_pairwise_relationships.csv`, `branch_cherry_evidence.txt`, `commit_patch_ids.csv`, and `patch_equivalence.csv`.

## Counts

- 24 local/remote refs excluding symbolic `origin/HEAD`
- 8 worktrees, of which 2 were dirty
- 11 existing annotated archive tags
- classifications: 17 `scientifically_relevant_history`, 3 `abandoned_prototype`, 2 `protected_or_default`, 1 `canonical_candidate`, and 1 `superseded_patch_equivalent`

## Findings

`origin/codex/repository-consolidation-v1` is 116 commits ahead of `origin/main` and contains the tips of the in-scope Flowstar diagnostic, first-order, matched-basis, common-contract, correctness-repair, and three-tool deep-study lineages. Its exported tree passed 213 tests. It is not accepted unchanged: its historical formal claims used a patched audit backend and must be withdrawn, and its backend startup contract must be repaired.

Three branches diverge from the candidate:

- `codex/batched-dense-nncs-gpu` has three unique commits after the normalized-insertion merge base. NNCS/controller/GPU demo work is outside scope.
- `codex/bern-ibf-tm-feasibility` has three unique commits after the common-contract merge base. BERN integration and attached literature work are outside scope; its worktree is dirty and untouched.
- `codex/deep-study-protocol-repair-v2` has five unique commits after the deep-study tip. Its fail-closed eligibility, timing, completion, exact-set, and output-safety behaviors are represented in the candidate's canonical protocol modules and tests; its old parallel collector/report implementation remains scientifically relevant history rather than a second active architecture.

Stable patch IDs found one functional duplicate across unrelated initial-history SHAs: `origin/master` commit `93e623e3ec2c8f71a305dfa5e94f47bfc0c3498d` is patch-equivalent to `ee53187387b98a5246055242062b14dd8e00d1fe`. The ref is retained; no deletion is proposed in this task.

Local/remote refs at the same SHA remain separately inventoried. SHA inequality was not treated as functional inequality, and date/commit subject was not used as correctness evidence.
