# Reproducibility

From the repository root, install and test:

```bash
python -m pip install -e ".[test]"
python -m pytest -q
```

Resolve the pinned read-only dependencies:

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
WORK_PARENT="$(dirname "$REPO_ROOT")"
export DIFFREACH_ROOT="$WORK_PARENT/DiffReach"
export FLOWSTAR_ROOT="$WORK_PARENT/flowstar"
export FLOWSTAR_AUDIT_ROOT="$WORK_PARENT/flowstar-audit"
export DIFFREACH_PYTHON="/path/to/pinned/diffreach/environment/bin/python"
```

Run smoke into a path that does not yet exist:

```bash
python experiments/consolidated_study/cli.py smoke \
  --output-dir /tmp/torch-tm-flowpipe-smoke
```

After committing a clean code-freeze, run formal:

```bash
python experiments/consolidated_study/cli.py formal \
  --output-dir "artifacts/runs/$(date -u +%Y%m%dT%H%M%SZ)"
```

The formal command refuses a dirty worktree and a non-empty destination. It
runs the full tests, all three tools, normalization, figure/report generation,
checksums, and the independent audit. Environment, commands, external SHAs,
dirty states, and patch hashes are captured in the run.

To re-audit a delivered run:

```bash
python analysis/independent_audit.py artifacts/runs/20260730T153654Z
cd artifacts/runs/20260730T153654Z
shasum -a 256 -c SHA256SUMS
```

The delivered authoritative run is `20260730T153654Z` at code freeze
`0dfdf587ee0fb9cff374dbc41ecdf17dfa2bf781`. Reproduction on other hardware
must use a new run ID and must not combine timing with this Apple Silicon CPU
run.
