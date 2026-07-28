#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
SMOKE_ROOT="${THREE_WAY_REPAIR_SMOKE_ROOT:-$(mktemp -d /tmp/three_way_comparison_repair_smoke.XXXXXX)}"
OUTPUT_DIR="$SMOKE_ROOT/result"

cd "$REPO_ROOT"
conda run -n py11 python -m pytest -q "$HERE/tests"
conda run -n py11 python "$HERE/run_corrected_comparison.py" \
  --smoke \
  --skip-historical-reproduction \
  --output-dir "$OUTPUT_DIR"

echo "Smoke gates completed: $OUTPUT_DIR"
