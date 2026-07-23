#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION_NAME="tm_first_order_three_way"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
SMOKE_OUTPUT="$SCRIPT_DIR/results/smoke_$TIMESTAMP"
OUTPUT_DIR="${FIRST_ORDER_OUTPUT_DIR:-$SCRIPT_DIR/results/$TIMESTAMP}"
LOG_PATH="$OUTPUT_DIR/full_run.log"

FIRST_ORDER_SMOKE_OUTPUT_DIR="$SMOKE_OUTPUT" "$SCRIPT_DIR/run_smoke.sh"

mkdir -p "$OUTPUT_DIR"
if command -v tmux >/dev/null 2>&1; then
  if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "A tmux session named $SESSION_NAME is already active; no duplicate was launched." >&2
    exit 1
  fi
  tmux new-session -d -s "$SESSION_NAME" \
    "env FIRST_ORDER_OUTPUT_DIR='$OUTPUT_DIR' '$SCRIPT_DIR/run_all.sh' >'$LOG_PATH' 2>&1"
  echo "SESSION_NAME=$SESSION_NAME"
  echo "LOG_PATH=$LOG_PATH"
  echo "OUTPUT_DIR=$OUTPUT_DIR"
  echo "Inspect: tmux attach -t $SESSION_NAME"
  echo "Progress without attaching: tail -f '$LOG_PATH'"
  echo "Stop safely: tmux send-keys -t $SESSION_NAME C-c"
  exit 0
fi

CONDA_TMUX=""
if conda run -n tmux-tools tmux -V >/dev/null 2>&1; then
  CONDA_TMUX="conda run --no-capture-output -n tmux-tools tmux"
fi
if [[ -n "$CONDA_TMUX" ]]; then
  if $CONDA_TMUX has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "A conda tmux session named $SESSION_NAME is already active; no duplicate was launched." >&2
    exit 1
  fi
  $CONDA_TMUX new-session -d -s "$SESSION_NAME" \
    "env FIRST_ORDER_OUTPUT_DIR='$OUTPUT_DIR' '$SCRIPT_DIR/run_all.sh' >'$LOG_PATH' 2>&1"
  echo "SESSION_NAME=$SESSION_NAME (tmux from conda environment tmux-tools)"
  echo "LOG_PATH=$LOG_PATH"
  echo "OUTPUT_DIR=$OUTPUT_DIR"
  echo "Inspect: $CONDA_TMUX attach -t $SESSION_NAME"
  echo "Progress without attaching: tail -f '$LOG_PATH'"
  echo "Stop safely: $CONDA_TMUX send-keys -t $SESSION_NAME C-c"
  exit 0
fi

PID_FILE="$OUTPUT_DIR/full_run.pid"
nohup env FIRST_ORDER_OUTPUT_DIR="$OUTPUT_DIR" "$SCRIPT_DIR/run_all.sh" \
  >"$LOG_PATH" 2>&1 &
RUN_PID=$!
echo "$RUN_PID" >"$PID_FILE"
echo "PID=$RUN_PID"
echo "PID_FILE=$PID_FILE"
echo "LOG_PATH=$LOG_PATH"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "Inspect: tail -f '$LOG_PATH'"
echo "Stop safely: kill -INT $RUN_PID"
