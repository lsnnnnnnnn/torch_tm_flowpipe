#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION_NAME="tm_first_order_followup"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
SMOKE_OUTPUT="${FIRST_ORDER_FOLLOWUP_SMOKE_OUTPUT_DIR:-$SCRIPT_DIR/results/smoke_$TIMESTAMP}"
OUTPUT_DIR="${FIRST_ORDER_FOLLOWUP_OUTPUT_DIR:-$SCRIPT_DIR/results/$TIMESTAMP}"
LOG_PATH="$OUTPUT_DIR/full_run.log"

if [[ "${FIRST_ORDER_FOLLOWUP_SKIP_SMOKE:-0}" != "1" ]]; then
  FIRST_ORDER_FOLLOWUP_SMOKE_OUTPUT_DIR="$SMOKE_OUTPUT" "$SCRIPT_DIR/run_smoke.sh"
fi

mkdir -p "$OUTPUT_DIR"
if command -v tmux >/dev/null 2>&1; then
  TMUX=(tmux)
elif conda run -n tmux-tools tmux -V >/dev/null 2>&1; then
  TMUX=(conda run --no-capture-output -n tmux-tools tmux)
else
  echo "tmux is required for the focused full sweep" >&2
  exit 1
fi

if "${TMUX[@]}" has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "Session $SESSION_NAME already exists; no duplicate was launched." >&2
  exit 1
fi

"${TMUX[@]}" new-session -d -s "$SESSION_NAME" \
  "env FIRST_ORDER_FOLLOWUP_OUTPUT_DIR='$OUTPUT_DIR' '$SCRIPT_DIR/run_all.sh' >'$LOG_PATH' 2>&1"

echo "SESSION_NAME=$SESSION_NAME"
echo "LOG_PATH=$LOG_PATH"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "Inspect: ${TMUX[*]} attach -t $SESSION_NAME"
echo "Progress: tail -f '$LOG_PATH'"
