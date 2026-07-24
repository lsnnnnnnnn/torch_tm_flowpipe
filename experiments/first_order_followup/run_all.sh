#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT_DIR="${FIRST_ORDER_FOLLOWUP_OUTPUT_DIR:-${1:-$SCRIPT_DIR/results/$TIMESTAMP}}"
mkdir -p "$OUTPUT_DIR/logs/flowstar_audit"
echo "OUTPUT_DIR=$OUTPUT_DIR"

conda run --no-capture-output -n py11 \
  python "$SCRIPT_DIR/audit_environment.py" --output-dir "$OUTPUT_DIR" --stage before
conda run --no-capture-output -n py11 \
  python "$SCRIPT_DIR/run_flowstar_audit.py" \
  --output-dir "$OUTPUT_DIR/logs/flowstar_audit"
conda run --no-capture-output -n py11 \
  python "$SCRIPT_DIR/run_torch_followup.py" --output-dir "$OUTPUT_DIR"
conda run --no-capture-output -n diffreach312 \
  python "$SCRIPT_DIR/run_diffreach_followup.py" --output-dir "$OUTPUT_DIR"
conda run --no-capture-output -n py11 \
  python "$SCRIPT_DIR/run_flowstar_followup.py" --output-dir "$OUTPUT_DIR"
conda run --no-capture-output -n py11 \
  python "$SCRIPT_DIR/collect_results.py" --output-dir "$OUTPUT_DIR"

# All package tests plus the experiment-local regression suites.
conda run --no-capture-output -n py11 \
  python -m pytest -q "$REPO_ROOT/tests" "$SCRIPT_DIR/tests/test_torch_basis.py"
conda run --no-capture-output -n diffreach312 \
  python -m pytest -q \
  "$SCRIPT_DIR/tests/test_diffreach_projection.py" \
  "$SCRIPT_DIR/tests/test_diffreach_parity.py"

conda run --no-capture-output -n py11 \
  python "$SCRIPT_DIR/audit_environment.py" --output-dir "$OUTPUT_DIR" --stage after
conda run --no-capture-output -n py11 \
  python "$SCRIPT_DIR/generate_artifacts.py" --output-dir "$OUTPUT_DIR"

echo "FULL_RUN_COMPLETE=$OUTPUT_DIR"
