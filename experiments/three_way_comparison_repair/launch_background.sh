#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION="tm_three_way_comparison_repair"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "A run is already active in tmux session: $SESSION" >&2
  exit 1
fi

"$HERE/run_smoke.sh"

TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT_DIR="${THREE_WAY_REPAIR_OUTPUT_DIR:-$HERE/results/$TIMESTAMP}"
LOG_PATH="${THREE_WAY_REPAIR_LOG_PATH:-$HERE/results/$TIMESTAMP.full_run.log}"
COMMAND="$HERE/run_all.sh $OUTPUT_DIR"
mkdir -p "$HERE/results"

tmux new-session -d -s "$SESSION" \
  "cd '$HERE/../..' && '$HERE/run_all.sh' '$OUTPUT_DIR' >'$LOG_PATH' 2>&1"

echo "tmux session: $SESSION"
echo "command: $COMMAND"
echo "log path: $LOG_PATH"
echo "result directory: $OUTPUT_DIR"
echo "progress command: tail -f '$LOG_PATH'"
echo "safe stop command: tmux send-keys -t '$SESSION' C-c"
