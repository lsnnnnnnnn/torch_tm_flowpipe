# Recovery record

Recovered at 2026-07-29 UTC after an explicit `git fetch --all --prune`.

## Authoritative starting point

- Requested correctness-repair base:
  `codex/three-way-comparison-correctness-repair@9024a8a29bdc0ad668a7c0620bd53872f4313cc8`.
- The fetched `origin/codex/three-way-comparison-correctness-repair` still
  resolves to that exact commit.
- The existing study branch is
  `codex/torch-flowstar-diffreach-deep-study`, based on `9024a8a`.
- The recovered study worktree is
  `/srv/local/shengenli/torch_tm_flowpipe_three_tool_study`.
- The pre-recovery study HEAD was
  `2cd48b5` (`soundly project repeated Torch reset endpoints`).
- No remote tracking branch for the study branch existed at recovery time.

No reset, branch deletion, worktree deletion, force push, or modification of
`main` was performed.

## Worktrees and external repositories

The Torch repository worktree inventory at recovery time was:

| worktree | branch | SHA |
|---|---|---|
| `/srv/local/shengenli/torch_tm_flowpipe` | `codex/flowstar-raw-remainder-compat` | `26a254e` |
| `/srv/local/shengenli/torch_tm_flowpipe_bern_ibf_study` | `codex/bern-ibf-tm-feasibility` | `dd82032` |
| `/srv/local/shengenli/torch_tm_flowpipe_first_order_bench` | `codex/first-order-three-way-benchmark` | `13e5eec` |
| `/srv/local/shengenli/torch_tm_flowpipe_first_order_followup` | `codex/first-order-followup-correctness-matched-basis` | `3d0ae54` |
| `/srv/local/shengenli/torch_tm_flowpipe_three_tool_study` | `codex/torch-flowstar-diffreach-deep-study` | `2cd48b5` |
| `/srv/local/shengenli/torch_tm_flowpipe_three_way_comparison` | `codex/three-way-common-contract-comparison` | `7251adf` |
| `/srv/local/shengenli/torch_tm_flowpipe_three_way_repair` | `codex/three-way-comparison-correctness-repair` | `9024a8a` |

External repository state:

- stock Flow*: `/srv/local/shengenli/flowstar`,
  `master@b85a3211748cb77b736fe4ad42ee02d8d2b81148`;
- Flow* audit: `/srv/local/shengenli/flowstar_three_way_audit`,
  `codex/full-picard-revalidated@94cf3cbb8fa38330d9b92eb07477f906b214c2fd`;
- DiffReach: `/srv/local/shengenli/DiffReach`,
  `main@dd628eb443b517d6415de93e7035b4baef73963e`;
- BERN-NN-Implicit: `/srv/local/shengenli/BERN-NN-Implicit`,
  `main@ebcf54a`.

The stock and audit Flow* worktrees contain untracked build products.  They
were not removed or added to a commit.  The Flow* audit branch contains the
committed trace/validation changes through `94cf3cb`.

## Recovered uncommitted work

The study worktree contained implementation changes for:

- device-preserving normalized Torch affine resets;
- native/export round-trip validation;
- deterministic nonlinear trajectory sanity checks explicitly labelled as
  non-proof checks;
- protocol-aware collection/reporting and numerical eligibility;
- primary CPU versus secondary hardware tables;
- report protocol regression tests; and
- post-generation collector verification.

These changes were reviewed with `git diff`, checked with `git diff --check`,
and preserved in the first recovery checkpoint instead of being discarded.

Five local timestamped result directories were also found:

- `20260729T041318Z` and `20260729T041345Z`: empty/early launcher attempts;
- `20260729T041354Z`: stopped after a failed primary Flow* correctness gate;
- `20260729T041924Z`: reached controlled and native data generation but did
  not complete the final Pareto/report pipeline;
- `20260729T053811Z`: interrupted during the Flow* correctness matrix.

These directories remain in place under `results/` as diagnostic recovery
evidence.  They contain copied build binaries and total several GiB, so the
timestamped scratch directory is ignored by Git.  No interrupted directory is
designated authoritative.  Portable final tables, plots, reports, manifests,
and required traces will be copied to a separately committed artifact
directory only after the complete verification pipeline passes.

No tmux server was running at recovery time, so no background experiment was
silently abandoned.
