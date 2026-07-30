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
