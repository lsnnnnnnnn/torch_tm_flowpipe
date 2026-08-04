# Reproducibility

From the repository root, install and test:

```bash
python -m pip install -e ".[test]"
python -m pytest -q
```

Resolve the read-only sibling dependencies without hard-coded private paths:

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
WORK_PARENT="$(dirname "$REPO_ROOT")"
export DIFFREACH_ROOT="$WORK_PARENT/DiffReach"
export FLOWSTAR_ROOT="$WORK_PARENT/flowstar"
export DIFFREACH_PYTHON="/path/to/pinned/diffreach/environment/bin/python"
```

Run smoke into a path that does not yet exist:

```bash
python experiments/consolidated_study/cli.py smoke \
  --output-dir /tmp/torch-tm-flowpipe-smoke-new
```

The smoke run is pipeline evidence only. It records backend identity and
rejects a non-empty output directory.

Do not run formal while any gate in `benchmarks/cross_tool_gates.yaml` is
pending. The command is intentionally fail-closed before output creation.
Once a later independent audit verifies every gate, a clean code freeze can
use:

```bash
python experiments/consolidated_study/cli.py formal \
  --output-dir "artifacts/runs/$(date -u +%Y%m%dT%H%M%SZ)"
```

Formal also refuses a dirty worktree, an invalid or contaminated Flowstar
backend, and a non-empty destination. It records command, environment,
external SHAs/status, patch and library hashes, configuration identity,
checksums, and the independent audit.

Frozen historical bundles may be checksum-checked, but rerunning the current
auditor against them is expected to fail the strengthened backend and bound
contracts:

```bash
cd artifacts/runs/20260730T153654Z
shasum -a 256 -c SHA256SUMS
```

Run `20260730T153654Z` is frozen but withdrawn because it used a patched audit
backend. A new run must use a new run ID, and timings from different hardware
must not be combined.
