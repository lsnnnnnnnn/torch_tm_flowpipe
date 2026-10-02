#!/usr/bin/env bash
set -euo pipefail

: "${TORA_REACH_RUN_DIR:?}"
: "${TORA_REACH_WORKDIR:?}"
: "${TORA_REACH_BINARY:?}"
: "${TORA_REACH_SERVER_PYTHON:?}"
: "${TORA_REACH_OVERLAY:?}"
: "${TORA_REACH_MODEL:?}"
: "${TORA_REACH_CPUSET:?}"
: "${CUDA_VISIBLE_DEVICES:?}"

cd "$TORA_REACH_WORKDIR"
test -d "$TORA_REACH_RUN_DIR"
test -x "$TORA_REACH_BINARY"
test -f crown_paper.py
test -f "$TORA_REACH_MODEL"
if ss -ltn | awk '$4 ~ /:5111$/ {found=1} END {exit !found}'; then
  echo 'RPC port 5111 occupied' >&2
  exit 2
fi

taskset -c "$TORA_REACH_CPUSET" "$TORA_REACH_BINARY" "$TORA_REACH_RUN_DIR/initial_boxes.json"
export PYTHONPATH="$TORA_REACH_OVERLAY${PYTHONPATH:+:$PYTHONPATH}"
server_pid=
native_pid=
cleanup() {
  if [[ -n "$native_pid" ]]; then kill "$native_pid" 2>/dev/null || :; wait "$native_pid" 2>/dev/null || :; fi
  if [[ -n "$server_pid" ]]; then kill "$server_pid" 2>/dev/null || :; wait "$server_pid" 2>/dev/null || :; fi
}
trap cleanup EXIT

ARCH_RPC_LOG="$TORA_REACH_RUN_DIR/controller_rpc.jsonl" taskset -c "$TORA_REACH_CPUSET" \
  "$TORA_REACH_SERVER_PYTHON" -B -u crown_paper.py >"$TORA_REACH_RUN_DIR/server.log" 2>&1 &
server_pid=$!
printf '%s\n' "$server_pid" >"$TORA_REACH_RUN_DIR/server.pid"
ready=0
for _ in $(seq 1 300); do
  if ! kill -0 "$server_pid" 2>/dev/null; then echo 'RPC server exited during startup' >&2; exit 1; fi
  if "$TORA_REACH_SERVER_PYTHON" -B -c 'import socket; socket.create_connection(("127.0.0.1",5111),0.2).close()' >/dev/null 2>&1; then ready=1; break; fi
  sleep 0.2
done
if [[ "$ready" != 1 ]]; then echo 'RPC server did not listen within 60 s' >&2; exit 1; fi
ss -ltnp >"$TORA_REACH_RUN_DIR/listeners_at_start.txt"
if ! awk -v pid="pid=$server_pid," '$4 ~ /:5111$/ && index($0,pid) {found=1} END {exit !found}' "$TORA_REACH_RUN_DIR/listeners_at_start.txt"; then
  echo 'RPC port 5111 is not owned by the started server' >&2
  exit 1
fi

AUTHOR_FIXED_STEPS=1 ARCH_RANGE_LOG="$TORA_REACH_RUN_DIR/ranges.bin" \
taskset -c "$TORA_REACH_CPUSET" "$TORA_REACH_BINARY" >"$TORA_REACH_RUN_DIR/native.log" 2>&1 &
native_pid=$!
printf '%s\n' "$native_pid" >"$TORA_REACH_RUN_DIR/native.pid"
echo "server_pid=$server_pid native_pid=$native_pid port=5111 gpu=$CUDA_VISIBLE_DEVICES cpus=$TORA_REACH_CPUSET"
set +e
wait "$native_pid"
native_rc=$?
set -e
native_pid=
exit "$native_rc"
