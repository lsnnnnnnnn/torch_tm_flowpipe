#!/usr/bin/env bash
set -euo pipefail

: "${QUAD_RUN_DIR:?}"
: "${QUAD_WORKDIR:?}"
: "${QUAD_BINARY:?}"
: "${QUAD_SERVER_PYTHON:?}"
: "${QUAD_OVERLAY:?}"
: "${QUAD_MODEL:?}"
: "${QUAD_CPUSET:?}"
: "${CUDA_VISIBLE_DEVICES:?}"

cd "$QUAD_WORKDIR"
test -d "$QUAD_RUN_DIR"
test -x "$QUAD_BINARY"
test -f crown_paper.py
test -f "$QUAD_MODEL"
if ss -ltn | awk '$4 ~ /:5000$/ {found=1} END {exit !found}'; then
  echo 'RPC port 5000 is already occupied' >&2
  exit 2
fi

export PYTHONPATH="$QUAD_OVERLAY${PYTHONPATH:+:$PYTHONPATH}"
server_pid=
native_pid=
cleanup() {
  if [[ -n "$native_pid" ]]; then kill "$native_pid" 2>/dev/null || :; wait "$native_pid" 2>/dev/null || :; fi
  if [[ -n "$server_pid" ]]; then kill "$server_pid" 2>/dev/null || :; wait "$server_pid" 2>/dev/null || :; fi
}
trap cleanup EXIT

QUAD_RPC_LOG="$QUAD_RUN_DIR/controller_rpc.jsonl" taskset -c "$QUAD_CPUSET" \
  "$QUAD_SERVER_PYTHON" -B -u crown_paper.py >"$QUAD_RUN_DIR/server.log" 2>&1 &
server_pid=$!
printf '%s\n' "$server_pid" >"$QUAD_RUN_DIR/server.pid"
ready=0
for _ in $(seq 1 300); do
  if ! kill -0 "$server_pid" 2>/dev/null; then echo 'RPC server exited during startup' >&2; exit 1; fi
  if "$QUAD_SERVER_PYTHON" -B -c 'import socket; socket.create_connection(("127.0.0.1",5000),0.2).close()' >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 0.2
done
if [[ "$ready" != 1 ]]; then echo 'RPC server did not listen within 60 seconds' >&2; exit 1; fi
ss -ltnp >"$QUAD_RUN_DIR/listeners_at_start.txt"
if ! awk -v pid="pid=$server_pid," '$4 ~ /:5000$/ && index($0,pid) {found=1} END {exit !found}' "$QUAD_RUN_DIR/listeners_at_start.txt"; then
  echo 'RPC port 5000 is not owned by the started server' >&2
  exit 1
fi

AUTHOR_FIXED_STEPS=1 ARCH_RANGE_LOG="$QUAD_RUN_DIR/ranges.bin" taskset -c "$QUAD_CPUSET" \
  "$QUAD_BINARY" >"$QUAD_RUN_DIR/native.log" 2>&1 &
native_pid=$!
printf '%s\n' "$native_pid" >"$QUAD_RUN_DIR/native.pid"
echo "server_pid=$server_pid native_pid=$native_pid port=5000 gpu=$CUDA_VISIBLE_DEVICES cpus=$QUAD_CPUSET"
set +e
wait "$native_pid"
native_rc=$?
set -e
native_pid=
exit "$native_rc"
