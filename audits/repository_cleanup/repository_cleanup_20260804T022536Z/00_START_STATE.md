# Repository cleanup start state

- Run ID: `repository_cleanup_20260804T022536Z`
- Evidence cutoff: 2026-08-04 UTC
- Main checkout: `/srv/local/shengenli/torch_tm_flowpipe`
- Initial branch/SHA: `codex/flowstar-raw-remainder-compat` at `26a254ef585a9dee394b7e41922c06bf8799f501`
- Initial state: dirty; four tracked files modified and `docs/flowstar_order2_vanderpol_failure.md` untracked.
- Initial patch SHA-256: `d4279db38fa8d026b39b5397974bbe999808201bf862eb08db2bafdce3b0ed77`.
- Safety action: status, diff stat, and patch hash were recorded. No stash, restore, checkout, clean, or commit was performed in that worktree.
- Fetch: `git fetch --all --tags --prune` completed. It observed deletion of the already-absent remote `codex/flowstar-ctrunc-rescue-diagnostics`, fetched `origin/codex/repository-consolidation-v1`, and fetched 11 pre-existing archive tags. This task did not delete a branch or create a tag.
- Refs after fetch: 24 local/remote branch refs excluding symbolic `origin/HEAD`; 8 worktrees; 11 tags. Two worktrees were dirty: the main checkout and the BERN study checkout.
- Baseline install: succeeded in `py11`.
- Baseline tests: `274 passed in 186.46s`; no skip or xfail summary was emitted. This is a dirty-checkout baseline, not final-branch evidence.
- Clean candidate snapshot: `origin/codex/repository-consolidation-v1` at `08b6f2416122cbf4220ff351e663caa1a0af13a2`; exported snapshot tests were `213 passed in 33.44s`.

## External repositories

- Flowstar: `/srv/local/shengenli/flowstar`, `b85a3211748cb77b736fe4ad42ee02d8d2b81148`, dirty. Its only tracked change is the GCC 15 compatibility correction `remainder = 0` to `result.remainder = 0` in `flowstar-toolbox/TaylorModel.h`; generated build outputs are untracked. Factual label: `stock-plus-gcc15-compat`, never `unmodified-stock`.
- Flowstar tracked patch SHA-256: `8f2c7128aa842c01869c4fd7a8ee89e3d3c799b9eb6966c2f406268fab5d8f91`.
- Flowstar static library SHA-256 at capture: `3a658f952cf8e4c30e04adb778a51c19047c37b6c8bf79f18ebae8120eefb117`.
- `/srv/local/shengenli/flowstar-audit`: absent. Historical audit backends remain `patched-audit`, `primary_eligible=false`.
- DiffReach: `/srv/local/shengenli/DiffReach`, `dd628eb443b517d6415de93e7035b4baef73963e`, clean.
- No external repository was modified.

Raw command evidence is in `git_state/`, `environment.json`, `external_backends.json`, `relevant_environment.txt`, and the baseline logs.
