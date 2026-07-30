# Phase 0 start state

- Run ID: `20260730T083258Z`
- Authoritative goal SHA256: `a7c5f6de9ef2307681923185a5bcc67ba3a2bf517be3f0e4051f0bd4b75cffa4`
- Goal source at start: `/Users/shengenli/Downloads/codex_repository_consolidation_goal_local.md`
- Repository: `https://github.com/lsnnnnnnnn/torch_tm_flowpipe.git`
- Local repository at start: `/Users/shengenli/Documents/Codex/2026-07-30/c/torch_tm_flowpipe`
- Branch and commit: `main` at `b2f34f5b2077e34662a2559d8c09b1d264bd7d98`
- Remote default: `origin/main`
- Initial tracked/untracked state before this audit directory: clean
- Platform: Apple M1 MacBook Air, macOS 26.5.2 arm64, 16 GB RAM
- Selected existing environment: conda `onmi`, Python 3.11.11, PyTorch 2.5.1
- CUDA: unavailable
- Flow*: unavailable (no environment variable, sibling checkout, or executable)
- DiffReach: unavailable (no environment variable or sibling checkout)

## Baseline

The first sandboxed run collected 100 tests. It reported 94 passed, 6 failed,
and 2 skipped. Every failure was a subprocess abort with
`OMP: Error #179: Function Can't open SHM failed`, so it was classified as a
sandbox environment failure rather than an implementation regression.

The exact same suite was rerun outside the sandbox:

```text
100 passed, 2 skipped in 36.74s
exit_code=0
```

Both skips are CUDA tests and explicitly state `CUDA not available`.

## Evidence files

- `git_ls_remote.txt`: fresh remote heads/tags query followed by fetch/prune/tags
- `git_refs.txt`: local refs, status, worktrees, tags, object state, submodules,
  and full `git fsck`
- `git_graph.txt`: complete decorated all-ref graph at start
- `environment.json`: structured environment and tool inventory
- `environment_raw.txt`: raw OS, hardware, Python, conda, compiler, and tool
  commands
- `external_DiffReach.txt` and `external_flowstar.txt`: external-tool resolution
- `baseline_tests.log`: original sandboxed baseline
- `baseline_tests_unsandboxed.log`: authoritative local baseline

No implementation change preceded this snapshot.
