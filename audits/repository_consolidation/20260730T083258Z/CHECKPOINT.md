# Consolidation checkpoint

Branch: `codex/repository-consolidation-v1`

Completed through this checkpoint: start-state capture, ref/branch inventory,
repository/duplicate/schema/artifact inventory, canonical-base decision,
verified deep-study recovery tag, base test suite, protocol regression layer,
portable external adapters, canonical configuration/profile files, supported
runner, independent auditor, and removal of legacy output paths.

The earlier development smoke was deliberately kept outside the branch
because later core/auditor changes advanced the code SHA. A fresh smoke from
the final pre-freeze source is the next gate and will be stored under
`06_smoke/`.

Remaining continuation:

```bash
cd "$(git rev-parse --show-toplevel)"
python -m pytest -q
git diff --check
# run a fresh smoke and commit its accepted evidence
# create the final code-freeze commit
python experiments/consolidated_study/cli.py formal \
  --output-dir artifacts/runs/<NEW_RUN_ID>
```

Do not treat `20260730T015245Z` as authoritative. Do not delete remote branches
until the formal artifact and all archive tags pass independent verification.
