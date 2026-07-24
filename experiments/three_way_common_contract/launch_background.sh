#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION="tm_three_way_common_contract"

"$HERE/run_smoke.sh"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  exit 1
fi
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT_DIR="${THREE_WAY_OUTPUT_DIR:-$HERE/results/$TIMESTAMP}"
mkdir -p "$OUTPUT_DIR/logs"
tmux new-session -d -s "$SESSION" \
  "cd '$HERE/../..' && THREE_WAY_OUTPUT_DIR='$OUTPUT_DIR' '$HERE/run_all.sh' >'$OUTPUT_DIR/logs/full_run.log' 2>&1"
echo "Launched $SESSION; output: $OUTPUT_DIR"
