#!/usr/bin/env bash
set -euo pipefail

: "${SP_RUN_DIR:?}"
: "${SP_WORKDIR:?}"
: "${SP_BINARY:?}"
: "${SP_SERVER_PYTHON:?}"
: "${SP_OVERLAY:?}"
: "${SP_MODEL:?}"
: "${SP_CPUSET:?}"
: "${CUDA_VISIBLE_DEVICES:?}"
SP_PORT=5101

cd "$SP_WORKDIR"
test -d "$SP_RUN_DIR"
test -x "$SP_BINARY"
test -f observed_server_paper.py
test -f "$SP_MODEL"
if ss -ltn | awk -v port=":$SP_PORT" '$4 ~ port "$" {found=1} END {exit !found}'; then
  echo "RPC port $SP_PORT is occupied" >&2
  exit 2
fi

export PYTHONPATH="$SP_OVERLAY${PYTHONPATH:+:$PYTHONPATH}"
export SP_MODEL
native_pid=
server_pid=
cleanup() {
  if [[ -n "$native_pid" ]]; then kill "$native_pid" 2>/dev/null || :; wait "$native_pid" 2>/dev/null || :; fi
  if [[ -n "$server_pid" ]]; then kill "$server_pid" 2>/dev/null || :; wait "$server_pid" 2>/dev/null || :; fi
}
trap cleanup EXIT

ARCH_RPC_LOG="$SP_RUN_DIR/controller_rpc.jsonl" taskset -c "$SP_CPUSET" \
  "$SP_SERVER_PYTHON" -B -u observed_server_paper.py >"$SP_RUN_DIR/server.log" 2>&1 &
server_pid=$!
printf '%s\n' "$server_pid" >"$SP_RUN_DIR/server.pid"
ready=0
for _ in $(seq 1 300); do
  if ! kill -0 "$server_pid" 2>/dev/null; then echo 'RPC server exited during startup' >&2; exit 1; fi
  if "$SP_SERVER_PYTHON" -B -c 'import socket; socket.create_connection(("127.0.0.1",5101),0.2).close()' >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 0.2
done
if [[ "$ready" != 1 ]]; then echo 'RPC server did not listen within 60 seconds' >&2; exit 1; fi
ss -ltnp >"$SP_RUN_DIR/listeners_at_start.txt"
if ! awk -v port=":$SP_PORT" -v pid="pid=$server_pid," '$4 ~ port "$" && index($0,pid) {found=1} END {exit !found}' "$SP_RUN_DIR/listeners_at_start.txt"; then
  echo "RPC port $SP_PORT is not owned by server PID $server_pid" >&2
  exit 1
fi

AUTHOR_FIXED_STEPS=1 ARCH_RANGE_LOG="$SP_RUN_DIR/ranges.bin" taskset -c "$SP_CPUSET" \
  "$SP_BINARY" >"$SP_RUN_DIR/native.log" 2>&1 &
native_pid=$!
printf '%s\n' "$native_pid" >"$SP_RUN_DIR/native.pid"
echo "server_pid=$server_pid native_pid=$native_pid port=$SP_PORT gpu=$CUDA_VISIBLE_DEVICES cpus=$SP_CPUSET"
set +e
wait "$native_pid"
native_rc=$?
set -e
native_pid=
if [[ -f "$SP_RUN_DIR/controller_rpc.jsonl" ]]; then
  echo "rpc_requests=$(wc -l < "$SP_RUN_DIR/controller_rpc.jsonl")"
fi
exit "$native_rc"
