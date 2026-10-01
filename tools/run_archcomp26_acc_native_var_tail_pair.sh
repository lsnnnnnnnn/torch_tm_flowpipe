#!/usr/bin/env bash
set -euo pipefail

: "${ACC_RUN_DIR:?}"
: "${ACC_BINARY:?}"
: "${ACC_SERVER:?}"
: "${ACC_SERVER_PYTHON:?}"
: "${ACC_OVERLAY:?}"
: "${ACC_CPUSET:?}"
: "${CUDA_VISIBLE_DEVICES:?}"
ACC_PORT=${ACC_PORT:-5102}

test -d "$ACC_RUN_DIR"
test -x "$ACC_BINARY"
test -f "$ACC_SERVER"
if ss -ltn | awk -v port=":$ACC_PORT" '$4 ~ port "$" {found=1} END {exit !found}'; then
  echo "RPC port $ACC_PORT is occupied" >&2
  exit 2
fi

export PYTHONPATH="$ACC_OVERLAY${PYTHONPATH:+:$PYTHONPATH}"
server_pid=
native_pid=
cleanup() {
  if [[ -n "$native_pid" ]]; then kill "$native_pid" 2>/dev/null || :; wait "$native_pid" 2>/dev/null || :; fi
  if [[ -n "$server_pid" ]]; then kill "$server_pid" 2>/dev/null || :; wait "$server_pid" 2>/dev/null || :; fi
}
trap cleanup EXIT

ARCH_RPC_LOG="$ACC_RUN_DIR/controller_rpc.jsonl" taskset -c "$ACC_CPUSET" \
  "$ACC_SERVER_PYTHON" -B -u "$ACC_SERVER" >"$ACC_RUN_DIR/server.log" 2>&1 &
server_pid=$!
printf '%s\n' "$server_pid" >"$ACC_RUN_DIR/server.pid"
ready=0
for _ in $(seq 1 300); do
  if ! kill -0 "$server_pid" 2>/dev/null; then echo 'RPC server exited during startup' >&2; exit 1; fi
  if "$ACC_SERVER_PYTHON" -B -c 'import socket,sys; socket.create_connection(("127.0.0.1",int(sys.argv[1])),0.2).close()' "$ACC_PORT" >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 0.2
done
if [[ "$ready" != 1 ]]; then echo 'RPC server did not listen within 60 seconds' >&2; exit 1; fi
ss -ltnp >"$ACC_RUN_DIR/listeners_at_start.txt"
if ! awk -v port=":$ACC_PORT" -v pid="pid=$server_pid," '$4 ~ port "$" && index($0,pid) {found=1} END {exit !found}' "$ACC_RUN_DIR/listeners_at_start.txt"; then
  echo "RPC port $ACC_PORT is not owned by server PID $server_pid" >&2
  exit 1
fi

AUTHOR_FIXED_STEPS=1 ARCH_RANGE_LOG="$ACC_RUN_DIR/ranges.bin" taskset -c "$ACC_CPUSET" \
  "$ACC_BINARY" >"$ACC_RUN_DIR/native.log" 2>&1 &
native_pid=$!
printf '%s\n' "$native_pid" >"$ACC_RUN_DIR/native.pid"
echo "server_pid=$server_pid native_pid=$native_pid port=$ACC_PORT gpu=$CUDA_VISIBLE_DEVICES cpus=$ACC_CPUSET"
set +e
wait "$native_pid"
native_rc=$?
set -e
native_pid=
exit "$native_rc"
