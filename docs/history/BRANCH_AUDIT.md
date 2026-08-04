status: historical
valid_for_commit: 08b6f2416122cbf4220ff351e663caa1a0af13a2
superseded_by: none
allowed_use: repository provenance and recovery

# Branch, worktree, and tag audit

The row-level evidence is preserved under
`audits/repository_cleanup/repository_cleanup_20260804T022536Z/`, including
`branch_inventory.csv`, `branch_feature_matrix.csv`, pairwise ancestry,
cherry evidence, patch IDs, and patch equivalence.

## Counts

- 24 local/remote refs, excluding symbolic `origin/HEAD`;
- 8 worktrees, of which 2 were dirty and left untouched;
- 11 existing annotated archive tags;
- 17 scientifically relevant history refs, 3 abandoned prototypes, 2
  protected/default refs, 1 canonical candidate, and 1 superseded
  patch-equivalent ref.

## Decision

`origin/codex/repository-consolidation-v1` at
`08b6f2416122cbf4220ff351e663caa1a0af13a2` was selected as the cleanup base.
Its exported tree passed 213 tests before selection. Selection did not accept
its historical result claims: the formal artifact used a patched Flowstar
audit route, so current documentation withdraws those claims and the cleanup
repairs startup/provenance contracts.

The batched NNCS/GPU and BERN branches contain three unique commits each but
are outside project scope. The protocol-repair fork has five unique commits;
its relevant fail-closed behavior is represented in the selected base and
current protocol tests. No divergent branch was merged merely because it was
newer.

Stable patch IDs identify `origin/master` commit
`93e623e3ec2c8f71a305dfa5e94f47bfc0c3498d` as functionally equivalent to
`ee53187387b98a5246055242062b14dd8e00d1fe`. Both refs are retained. SHA
inequality was not treated as functional inequality.

No branch/tag is deleted, `main` is not changed, and all original worktrees
remain untouched. Recovery points and active replacements are recorded in
`docs/history/MIGRATION_MAP.csv`.
