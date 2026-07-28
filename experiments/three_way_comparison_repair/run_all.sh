#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT_DIR="${1:-${THREE_WAY_REPAIR_OUTPUT_DIR:-$HERE/results/$TIMESTAMP}}"

cd "$REPO_ROOT"
conda run -n py11 python "$HERE/run_corrected_comparison.py" \
  --output-dir "$OUTPUT_DIR"
