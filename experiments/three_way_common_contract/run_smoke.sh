#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
SMOKE_DIR="${THREE_WAY_SMOKE_DIR:-$(mktemp -d /tmp/three_way_common_contract_smoke.XXXXXX)}"

cd "$REPO_ROOT"
conda run -n py11 python -m pytest -q "$HERE/tests"
conda run -n py11 python "$HERE/run_torch.py" --output-dir "$SMOKE_DIR" --smoke
conda run -n diffreach312 python "$HERE/run_diffreach.py" --output-dir "$SMOKE_DIR" --smoke
conda run -n py11 python "$HERE/run_flowstar.py" --output-dir "$SMOKE_DIR" --smoke
conda run -n py11 python "$HERE/collect_results.py" --output-dir "$SMOKE_DIR" --smoke --strict

echo "Smoke gates passed: $SMOKE_DIR"
