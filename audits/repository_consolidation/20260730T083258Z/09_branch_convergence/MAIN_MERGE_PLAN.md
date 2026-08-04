# Main merge plan

Status: refreshed for current integration lineage; **plan only, do not execute**.

The observed GitHub default branch is `main` at
`b2f34f5b2077e34662a2559d8c09b1d264bd7d98`.  The current integration lineage is:

1. published consolidation/history anchors;
2. three-tool correctness parity at
   `308b735ac577cfea39172976a4c08716f1e54d2f`;
3. verified native reproduction at
   `438ee68fd71fa6182eb66cac17229e20dd3cb7d3f`;
4. `codex/flowstar-scalar-affine-correctness-closure-20260804`, created directly
   from `438ee68` for the repository-status and Flow* scalar-affine diagnosis.

The older instruction to fast-forward `main` directly to
`codex/repository-consolidation-v1` is superseded.  Later lineage contains the
governance corrections and native primary evidence that must not be bypassed.
No main integration is authorized in the scalar-affine task, and the existing
dirty user worktree must not be touched.

When a separate merge-review task is eventually authorized, it must first:

- fetch and record the then-current `origin/main` and all lineage branch tips;
- require the reviewed closure commit to descend from exact `438ee68`;
- inspect every commit after `438ee68` and select the reviewed integration tip;
- recompute ancestry and merge-tree evidence against the then-current `main`;
- rerun the full suite and registry/evidence validators from a new clean worktree;
- use the protected-branch pull-request flow without force-push or history rewrite.

The Flow* gate and primary comparison eligibility remain independent of Git
mergeability.  A clean merge-tree or green tests cannot close a scientific gate.
