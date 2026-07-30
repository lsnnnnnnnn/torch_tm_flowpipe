# Main merge plan

The GitHub default branch is `main` at
`b2f34f5b2077e34662a2559d8c09b1d264bd7d98`. The selected integration base is
the deep-study tip `9a684d9106633e067bfac0747244b769fa49aa0b`. The published
formal-artifact anchor on `codex/repository-consolidation-v1` is
`269f3599fcc480984ff651c6c1e083a8ceac74e9`.

Post-convergence verification found:

- `merge-base(origin/main, canonical) = b2f34f5...`;
- publication-anchor counts are `0` main-only and `112` canonical-only;
- `git merge-tree --write-tree origin/main 269f359...` completed without a
  conflict and produced tree `2b599f1db1319039ae00abcf7b777be7796616ee`;
- the canonical branch is therefore a clean fast-forward of the observed main,
  not a history rewrite.

`main` remains unchanged because it is the default/protected branch and the
existing local main worktree contains user changes. Do not use that dirty
worktree and do not force-push. Review and advance main through the normal
protected-branch flow:

```bash
git fetch origin --prune --tags
test "$(git rev-parse origin/main)" = \
  b2f34f5b2077e34662a2559d8c09b1d264bd7d98
git merge-base --is-ancestor \
  origin/main origin/codex/repository-consolidation-v1
git worktree add -b codex/main-merge-review \
  ../torch_tm_flowpipe-main-merge-review origin/main
git -C ../torch_tm_flowpipe-main-merge-review merge --ff-only \
  origin/codex/repository-consolidation-v1
cd ../torch_tm_flowpipe-main-merge-review
python -m pytest -q
git push origin HEAD:main
```

If branch protection requires a pull request, use
`codex/repository-consolidation-v1` as the head and `main` as the base instead
of the final push command. Re-fetch immediately before approval; any changed
main tip invalidates the recorded fast-forward proof and requires a new
merge-tree/test pass.
