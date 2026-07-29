#!/usr/bin/env bash
set -euo pipefail

STUDY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${STUDY_DIR}/../.." && pwd)"
SESSION_NAME="tm_three_tool_deep_study"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT_DIR="${STUDY_DIR}/results/${TIMESTAMP}"
LOG_PATH="${OUTPUT_DIR}/run_all.log"
COMMAND="cd '${REPO_ROOT}' && bash '${STUDY_DIR}/run_all.sh' '${OUTPUT_DIR}' > '${LOG_PATH}' 2>&1"

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  printf 'Session already exists: %s\n' "${SESSION_NAME}"
  printf 'Progress: tmux attach -t %s\n' "${SESSION_NAME}"
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"
tmux new-session -d -s "${SESSION_NAME}" "${COMMAND}"

printf 'Session: %s\n' "${SESSION_NAME}"
printf 'Command: %s\n' "${COMMAND}"
printf 'Log: %s\n' "${LOG_PATH}"
printf 'Output directory: %s\n' "${OUTPUT_DIR}"
printf 'Progress command: tail -f %s\n' "${LOG_PATH}"
printf 'Safe stop command: tmux send-keys -t %s C-c\n' "${SESSION_NAME}"
