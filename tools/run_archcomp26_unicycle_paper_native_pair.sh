#!/usr/bin/env bash
set -euo pipefail

: "${UNICYCLE_RUN_DIR:?}"
: "${UNICYCLE_WORKDIR:?}"
: "${UNICYCLE_BINARY:?}"
: "${UNICYCLE_SERVER_PYTHON:?}"
: "${UNICYCLE_OVERLAY:?}"
: "${UNICYCLE_MODEL:?}"
: "${UNICYCLE_CPUSET:?}"
: "${CUDA_VISIBLE_DEVICES:?}"

cd "$UNICYCLE_WORKDIR"
test -d "$UNICYCLE_RUN_DIR"
test -x "$UNICYCLE_BINARY"
test -f crown_paper.py
test -f "$UNICYCLE_MODEL"
if ss -ltn | awk '$4 ~ /:5107$/ {found=1} END {exit !found}'; then
  echo 'RPC port 5107 occupied' >&2
  exit 2
fi

export PYTHONPATH="$UNICYCLE_OVERLAY${PYTHONPATH:+:$PYTHONPATH}"
server_pid=
native_pid=
cleanup() {
  if [[ -n "$native_pid" ]]; then kill "$native_pid" 2>/dev/null || :; wait "$native_pid" 2>/dev/null || :; fi
  if [[ -n "$server_pid" ]]; then kill "$server_pid" 2>/dev/null || :; wait "$server_pid" 2>/dev/null || :; fi
}
trap cleanup EXIT

ARCH_RPC_LOG="$UNICYCLE_RUN_DIR/controller_rpc.jsonl" taskset -c "$UNICYCLE_CPUSET" \
  "$UNICYCLE_SERVER_PYTHON" -B -u crown_paper.py >"$UNICYCLE_RUN_DIR/server.log" 2>&1 &
server_pid=$!
printf '%s\n' "$server_pid" >"$UNICYCLE_RUN_DIR/server.pid"
ready=0
for _ in $(seq 1 300); do
  if ! kill -0 "$server_pid" 2>/dev/null; then echo 'RPC server exited during startup' >&2; exit 1; fi
  if "$UNICYCLE_SERVER_PYTHON" -B -c 'import socket; socket.create_connection(("127.0.0.1",5107),0.2).close()' >/dev/null 2>&1; then ready=1; break; fi
  sleep 0.2
done
if [[ "$ready" != 1 ]]; then echo 'RPC server did not listen within 60 s' >&2; exit 1; fi
ss -ltnp >"$UNICYCLE_RUN_DIR/listeners_at_start.txt"
if ! awk -v pid="pid=$server_pid," '$4 ~ /:5107$/ && index($0,pid) {found=1} END {exit !found}' "$UNICYCLE_RUN_DIR/listeners_at_start.txt"; then
  echo 'RPC port 5107 is not owned by the started server' >&2
  exit 1
fi

AUTHOR_FIXED_STEPS=1 ARCH_RANGE_LOG="$UNICYCLE_RUN_DIR/ranges.bin" \
UNICYCLE_INITIAL_AFFINE_LOG="$UNICYCLE_RUN_DIR/initial_affine.tsv" \
UNICYCLE_BOX_LOG="$UNICYCLE_RUN_DIR/initial_boxes.json" \
taskset -c "$UNICYCLE_CPUSET" "$UNICYCLE_BINARY" >"$UNICYCLE_RUN_DIR/native.log" 2>&1 &
native_pid=$!
printf '%s\n' "$native_pid" >"$UNICYCLE_RUN_DIR/native.pid"
echo "server_pid=$server_pid native_pid=$native_pid port=5107 gpu=$CUDA_VISIBLE_DEVICES cpus=$UNICYCLE_CPUSET"
set +e
wait "$native_pid"
native_rc=$?
set -e
native_pid=
exit "$native_rc"
