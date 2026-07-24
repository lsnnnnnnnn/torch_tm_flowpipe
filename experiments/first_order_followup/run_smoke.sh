#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT_DIR="${FIRST_ORDER_FOLLOWUP_SMOKE_OUTPUT_DIR:-$SCRIPT_DIR/results/smoke_$TIMESTAMP}"
mkdir -p "$OUTPUT_DIR/logs/flowstar_audit"
echo "SMOKE_OUTPUT_DIR=$OUTPUT_DIR"

conda run --no-capture-output -n py11 \
  python "$SCRIPT_DIR/audit_environment.py" --output-dir "$OUTPUT_DIR" --stage before

conda run --no-capture-output -n diffreach312 \
  python -m pytest -q \
  "$SCRIPT_DIR/tests/test_diffreach_projection.py" \
  "$SCRIPT_DIR/tests/test_diffreach_parity.py"
conda run --no-capture-output -n py11 \
  python -m pytest -q "$SCRIPT_DIR/tests/test_torch_basis.py"

conda run --no-capture-output -n py11 \
  python "$SCRIPT_DIR/run_flowstar_audit.py" \
  --output-dir "$OUTPUT_DIR/logs/flowstar_audit"
conda run --no-capture-output -n py11 \
  python "$SCRIPT_DIR/run_torch_followup.py" --output-dir "$OUTPUT_DIR" --smoke
conda run --no-capture-output -n diffreach312 \
  python "$SCRIPT_DIR/run_diffreach_followup.py" --output-dir "$OUTPUT_DIR" --smoke
conda run --no-capture-output -n py11 \
  python "$SCRIPT_DIR/run_flowstar_followup.py" --output-dir "$OUTPUT_DIR" --smoke
conda run --no-capture-output -n py11 \
  python "$SCRIPT_DIR/collect_results.py" --output-dir "$OUTPUT_DIR" --smoke

conda run --no-capture-output -n py11 \
  python "$SCRIPT_DIR/audit_environment.py" --output-dir "$OUTPUT_DIR" --stage after
conda run --no-capture-output -n py11 \
  python "$SCRIPT_DIR/generate_artifacts.py" --output-dir "$OUTPUT_DIR"

echo "SMOKE_GATES_PASSED=$OUTPUT_DIR"
