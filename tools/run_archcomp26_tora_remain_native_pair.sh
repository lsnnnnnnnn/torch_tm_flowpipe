#!/usr/bin/env bash
set -euo pipefail

: "${TORA_RUN_DIR:?}"
: "${TORA_WORKDIR:?}"
: "${TORA_BINARY:?}"
: "${TORA_SERVER_PYTHON:?}"
: "${TORA_OVERLAY:?}"
: "${TORA_MODEL:?}"
: "${TORA_CPUSET:?}"
: "${CUDA_VISIBLE_DEVICES:?}"

cd "$TORA_WORKDIR"
test -d "$TORA_RUN_DIR"
test -x "$TORA_BINARY"
test -f crown_paper.py
test -f "$TORA_MODEL"
if ss -ltn | awk '$4 ~ /:5100$/ {found=1} END {exit !found}'; then
  echo 'RPC port 5100 occupied' >&2
  exit 2
fi

export PYTHONPATH="$TORA_OVERLAY${PYTHONPATH:+:$PYTHONPATH}"
server_pid=
native_pid=
cleanup() {
  if [[ -n "$native_pid" ]]; then kill "$native_pid" 2>/dev/null || :; wait "$native_pid" 2>/dev/null || :; fi
  if [[ -n "$server_pid" ]]; then kill "$server_pid" 2>/dev/null || :; wait "$server_pid" 2>/dev/null || :; fi
}
trap cleanup EXIT

ARCH_RPC_LOG="$TORA_RUN_DIR/controller_rpc.jsonl" taskset -c "$TORA_CPUSET" \
  "$TORA_SERVER_PYTHON" -B -u crown_paper.py >"$TORA_RUN_DIR/server.log" 2>&1 &
server_pid=$!
printf '%s\n' "$server_pid" >"$TORA_RUN_DIR/server.pid"
ready=0
for _ in $(seq 1 300); do
  if ! kill -0 "$server_pid" 2>/dev/null; then echo 'RPC server exited during startup' >&2; exit 1; fi
  if "$TORA_SERVER_PYTHON" -B -c 'import socket; socket.create_connection(("127.0.0.1",5100),0.2).close()' >/dev/null 2>&1; then ready=1; break; fi
  sleep 0.2
done
if [[ "$ready" != 1 ]]; then echo 'RPC server did not listen within 60 s' >&2; exit 1; fi
ss -ltnp >"$TORA_RUN_DIR/listeners_at_start.txt"
if ! awk -v pid="pid=$server_pid," '$4 ~ /:5100$/ && index($0,pid) {found=1} END {exit !found}' "$TORA_RUN_DIR/listeners_at_start.txt"; then
  echo 'RPC port 5100 is not owned by the started server' >&2
  exit 1
fi

AUTHOR_FIXED_STEPS=1 ARCH_RANGE_LOG="$TORA_RUN_DIR/ranges.bin" taskset -c "$TORA_CPUSET" \
  "$TORA_BINARY" >"$TORA_RUN_DIR/native.log" 2>&1 &
native_pid=$!
printf '%s\n' "$native_pid" >"$TORA_RUN_DIR/native.pid"
echo "server_pid=$server_pid native_pid=$native_pid port=5100 gpu=$CUDA_VISIBLE_DEVICES cpus=$TORA_CPUSET"
set +e
wait "$native_pid"
native_rc=$?
set -e
native_pid=
exit "$native_rc"
