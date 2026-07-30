# Main merge plan

The GitHub default branch is `main` at
`b2f34f5b2077e34662a2559d8c09b1d264bd7d98`. The selected integration base is
the deep-study tip `9a684d9106633e067bfac0747244b769fa49aa0b`; therefore the
canonical branch is not a clean fast-forward of current `main`.

Do not force-update `main`. After the canonical branch is pushed and formal
acceptance is verified:

1. fetch `origin` and confirm neither tip changed;
2. open a normal merge from `codex/repository-consolidation-v1` into `main`;
3. preserve the merge commit and resolve only evidenced conflicts;
4. rerun the repository suite and independent artifact audit;
5. update the GitHub default only through the normal protected-branch flow.

Until that review completes, `codex/repository-consolidation-v1` is
`recommended_next_main` and `main` remains unchanged.
