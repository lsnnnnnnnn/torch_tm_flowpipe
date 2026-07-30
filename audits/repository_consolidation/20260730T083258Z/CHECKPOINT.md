# Consolidation checkpoint

Branch: `codex/repository-consolidation-v1`

Completed through this checkpoint: start-state capture, ref/branch inventory,
repository/duplicate/schema/artifact inventory, canonical-base decision,
verified deep-study recovery tag, base test suite, protocol regression layer,
portable external adapters, canonical configuration/profile files, supported
runner, independent auditor, and removal of legacy output paths.

The accepted non-authoritative pre-freeze smoke is
`06_smoke/canonical-smoke-20260730-v4`. It records source SHA
`73c07c9fa6e23eeb8475bcd482eaa6f21c811238`, 12 exact configuration
identities, 24 raw observations, zero primary rows, and passing independent
acceptance.

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
