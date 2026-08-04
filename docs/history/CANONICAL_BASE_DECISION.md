status: historical
valid_for_commit: 0dfdf587ee0fb9cff374dbc41ecdf17dfa2bf781
superseded_by: audits/repository_cleanup/repository_cleanup_20260804T022536Z/CANONICAL_BASE_DECISION.md
allowed_use: provenance only

# Canonical base decision

## Selected base

Use `9a684d9106633e067bfac0747244b769fa49aa0b`, the fetched tip of
`origin/codex/torch-flowstar-diffreach-deep-study`, as the base for
`codex/repository-consolidation-v1`.

This is a base selection, not acceptance of its status claims or old results.
All known protocol defects in the authoritative goal must receive failing
regression tests and independent revalidation.

## Why this commit

The selection follows correctness, test coverage, protocol semantics,
maintainability, provenance, and only then date:

1. It contains the repaired three-way comparison commit
   `9024a8a29bdc0ad668a7c0620bd53872f4313cc8` and every earlier relevant
   Flow*/DiffReach/Torch lineage tip as ancestors.
2. It contains the only integrated common-contract, matched-basis, Flow*
   correctness/export, native protocol, repeated timing, Pareto, report, and
   independent artifact-audit implementation.
3. Starting from an earlier commit would require recreating or cherry-picking
   dozens of interdependent commits and would increase the risk of losing a
   regression test or provenance contract.
4. Its serious defects are localized in protocol, collector, schema, plotting,
   reporting, and active-tree governance. Those can be repaired transparently
   on a new branch without rewriting history.

## Why not another candidate

- `main` lacks 95 commits of relevant correctness, adapter, test, and protocol
  work.
- The correctness-repair branch lacks the later integrated CIR, controlled and
  native protocols, component attribution, repeated timing, and delivery
  audit.
- The first-order and common-contract branches are ancestors and incomplete
  slices of the same lineage.
- The newest time alone is not used as evidence; the selected commit wins on
  ancestry and capability coverage.
- `codex/batched-dense-nncs-gpu` has three divergent commits centered on NNCS
  and GPU demos, which are explicitly out of scope. No active feature is
  migrated from them.
- `master` has unrelated initialization history and is not a viable base.

## Migration strategy

- Keep the single package core implementation and extract stable semantics into
  regression tests rather than octopus-merging branches.
- Repair the deep-study runner/collector/analyzer contracts in the selected
  architecture.
- Replace duplicated benchmark constants with one canonical manifest.
- Remove the out-of-scope BERN and literature/PDF paths from the active tree
  without reading or incorporating them.
- Remove bulk historical outputs from the active tree after an annotated
  archive tag makes the selected base recoverable.
- Preserve unique scientific lineage in `BRANCH_AUDIT.md`,
  `MIGRATION_MAP.csv`, and archive tags.
- Reimplement changes that are coupled to old report/schema assumptions
  instead of cherry-picking them blindly.

No remote branch deletion and no update to main is authorized by this decision
alone.
