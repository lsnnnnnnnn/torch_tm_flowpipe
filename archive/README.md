# Historical recovery index

The current tree carries selected raw scientific evidence and support dependencies. [The index](index.csv) records historical tracked groups intentionally removed from this branch and the superseded handoff text, with exact parent commit, former path, size/count, purpose, and a reversible restore command. These files are still in Git history and in the pushed parent branch. No archive group is copied into a second competing result directory, and no old evidence is rewritten.

Use `git show <recovery_commit>:<relative_file>` to inspect an individual old file or the listed `git restore --source=<recovery_commit> -- <old_path>` in a disposable worktree to recover a whole group. The restore command changes only that disposable tree. The source commit remains the scientific version; restoring it here does not turn its old run into a current review result.

Several large legacy directories stay in the current branch because current tests, exact model reconstruction, or frozen runner imports still use them. See [the disposition inventory](../docs/maintenance/asset_disposition.csv) and [maintenance notes](../docs/maintenance/REORGANIZATION.md).
