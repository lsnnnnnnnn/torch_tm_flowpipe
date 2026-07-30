# Consolidation checkpoint

Branch: `codex/repository-consolidation-v1`

Completed through this checkpoint: start-state capture, ref/branch inventory,
repository/duplicate/schema/artifact inventory, canonical-base decision,
verified deep-study recovery tag, base test suite, protocol regression layer,
portable external adapters, canonical configuration/profile files, supported
runner, independent auditor, and removal of legacy output paths.

Accepted non-authoritative smoke evidence is under `06_smoke/`. Version 4
validated the initial freeze. The first formal attempt
`20260730T124958Z` then exposed a legacy diagnostic write outside the run
directory; the independent auditor correctly returned `failed_acceptance`.
Commit `9bef0ac87544aa97a8088c32e2a6e5cc2ab830a5` confines that document to
the requested output and adds a regression test. Version 5 validates the
corrected source with 12 exact identities, 24 raw observations, zero primary
rows, and passing independent acceptance.

The second formal attempt `20260730T141302Z` passed its then-current auditor,
but a manual cross-tool review found that both collector and independent
auditor partitioned Pareto dominance by tool. Its key evidence is retained
under `08_rejected_formal_semantic_review/`; it is not citable. The corrected
contract groups by system and evaluation horizon so eligible tool families
can dominate one another. Smoke version 6 accepts this correction at
`2d870f6fd12595eed0a23da59f945986a310e245`.

Remaining continuation:

```bash
cd "$(git rev-parse --show-toplevel)"
python -m pytest -q
git diff --check
# create the final code-freeze commit
python experiments/consolidated_study/cli.py formal \
  --output-dir artifacts/runs/<NEW_RUN_ID>
```

Do not treat `20260730T015245Z` as authoritative. Do not delete remote branches
until the formal artifact and all archive tags pass independent verification.
