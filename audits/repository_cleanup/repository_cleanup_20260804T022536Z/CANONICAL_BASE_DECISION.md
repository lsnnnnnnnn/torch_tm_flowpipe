# Canonical base decision

## Selected base

Select `08b6f2416122cbf4220ff351e663caa1a0af13a2`, the fetched tip of `origin/codex/repository-consolidation-v1`, as the base for `codex/repository-cleanup-before-external-torch-audit`.

This selects architecture and tested behavior, not the candidate's old scientific status claims.

## Evidence and priority order

1. Correctness: the candidate contains the repaired raw/tightened, horizon/completion, eligibility-first, timing-boundary, and cross-tool grouping contracts. Its clean exported tree passed 213 tests.
2. Tests: every in-scope historical tip is an ancestor; protocol regressions cover strict Boolean gates, requested/successful horizons, order/basis identity, non-empty output rejection, and timing boundaries.
3. Protocol semantics: it has versioned schema/config/eligibility/provenance modules and separates raw endpoint semantics from tightened endpoints in active eligibility.
4. Maintainability: it has one core package, one canonical benchmark manifest plus profiles, one active orchestration CLI, and an independent artifact auditor.
5. Provenance: it already preserves pre-consolidation refs through tags and contains detailed branch/artifact audit history.
6. Recency: considered only after the above; the selected tip is newer than the historical candidates.

## Why the alternatives were not selected

- `main` lacks 116 commits of in-scope adapters, regression tests, schemas, and consolidation work.
- Intermediate Flowstar/first-order/common-contract/correctness-repair/deep-study tips are strict ancestors and incomplete slices.
- The BERN and batched NNCS/GPU divergences are explicitly outside scope.
- The five-commit protocol-repair divergence retains a superseded parallel architecture; the candidate reimplements the relevant fail-closed contracts in canonical modules and tests.
- `origin/master` has unrelated topology and a sole patch-equivalent root commit; it offers no superior active tree.

## Required repairs before acceptance

- Reclassify every cross-tool headline/Pareto result that used `flowstar-audit` with `FLOWSTAR_AUDIT_CACHE_LEAF_TRUNCATION=1` as `withdrawn_do_not_cite` without rewriting frozen artifacts.
- Add reusable backend identity and contamination checks; primary Flowstar must reject audit-named roots and audit behavior variables.
- Label the current Flowstar checkout `stock-plus-gcc15-compat`, not `unmodified-stock`.
- Keep official-program, generated-stock, and patched-audit identities distinct.
- Separate endpoint/segment/tube and raw/tightened contracts with regression tests.
- Retain exactly one supported order-2 step-trace diagnostic and record validation rejection rather than crash/unsupported/completed.
- Publish current artifact/claim registries and the external-audit preconditions; do not run a new formal matrix.

No unique historical branch is merged wholesale. The current dirty checkout is not the base and none of its unknown changes will be committed directly.
