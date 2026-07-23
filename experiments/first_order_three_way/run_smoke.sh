#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT_DIR="${FIRST_ORDER_SMOKE_OUTPUT_DIR:-$SCRIPT_DIR/results/smoke_$TIMESTAMP}"

mkdir -p "$OUTPUT_DIR"
echo "SMOKE_OUTPUT_DIR=$OUTPUT_DIR"

conda run --no-capture-output -n py11 \
  python "$SCRIPT_DIR/environment_audit.py" --output-dir "$OUTPUT_DIR"
conda run --no-capture-output -n py11 \
  python "$SCRIPT_DIR/compute_references.py" --output-dir "$OUTPUT_DIR" --smoke
conda run --no-capture-output -n py11 \
  python "$SCRIPT_DIR/run_torch.py" --output-dir "$OUTPUT_DIR" --smoke
conda run --no-capture-output -n py11 \
  python "$SCRIPT_DIR/run_flowstar.py" --output-dir "$OUTPUT_DIR" --smoke
conda run --no-capture-output -n diffreach312 \
  python "$SCRIPT_DIR/run_diffreach.py" --output-dir "$OUTPUT_DIR" --smoke
conda run --no-capture-output -n py11 \
  python "$SCRIPT_DIR/collect_results.py" --output-dir "$OUTPUT_DIR" --smoke

BENCHMARK_RESULTS_DIR="$OUTPUT_DIR" conda run --no-capture-output -n py11 \
  python -m pytest -q "$SCRIPT_DIR/tests/test_benchmark.py"
conda run --no-capture-output -n diffreach312 \
  python -m pytest -q "$SCRIPT_DIR/tests/test_diffreach_support.py"

echo "Smoke comparison passed schema, equation, grid, endpoint/tube, exact-hull, and sample checks."
echo "SMOKE_OUTPUT_DIR=$OUTPUT_DIR"
