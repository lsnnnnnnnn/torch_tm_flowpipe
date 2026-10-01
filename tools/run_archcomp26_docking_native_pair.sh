#!/usr/bin/env bash
set -euo pipefail

: "${DOCKING_RUN_DIR:?}"
: "${DOCKING_WORKDIR:?}"
: "${DOCKING_BINARY:?}"
: "${DOCKING_SERVER_PYTHON:?}"
: "${DOCKING_OVERLAY:?}"
: "${DOCKING_MODEL:?}"
: "${DOCKING_CPUSET:?}"
: "${CUDA_VISIBLE_DEVICES:?}"

cd "$DOCKING_WORKDIR"
test -d "$DOCKING_RUN_DIR"
test -x "$DOCKING_BINARY"
test -f crown_paper.py
test -f "$DOCKING_MODEL"
if ss -ltn | awk '$4 ~ /:5104$/ {found=1} END {exit !found}'; then
  echo 'RPC port 5104 occupied' >&2
  exit 2
fi

export PYTHONPATH="$DOCKING_OVERLAY${PYTHONPATH:+:$PYTHONPATH}"
server_pid=
native_pid=
cleanup() {
  if [[ -n "$native_pid" ]]; then kill "$native_pid" 2>/dev/null || :; wait "$native_pid" 2>/dev/null || :; fi
  if [[ -n "$server_pid" ]]; then kill "$server_pid" 2>/dev/null || :; wait "$server_pid" 2>/dev/null || :; fi
}
trap cleanup EXIT

ARCH_RPC_LOG="$DOCKING_RUN_DIR/controller_rpc.jsonl" taskset -c "$DOCKING_CPUSET" \
  "$DOCKING_SERVER_PYTHON" -B -u crown_paper.py >"$DOCKING_RUN_DIR/server.log" 2>&1 &
server_pid=$!
printf '%s\n' "$server_pid" >"$DOCKING_RUN_DIR/server.pid"
ready=0
for _ in $(seq 1 300); do
  if ! kill -0 "$server_pid" 2>/dev/null; then echo 'RPC server exited during startup' >&2; exit 1; fi
  if "$DOCKING_SERVER_PYTHON" -B -c 'import socket; socket.create_connection(("127.0.0.1",5104),0.2).close()' >/dev/null 2>&1; then ready=1; break; fi
  sleep 0.2
done
if [[ "$ready" != 1 ]]; then echo 'RPC server did not listen within 60 s' >&2; exit 1; fi
ss -ltnp >"$DOCKING_RUN_DIR/listeners_at_start.txt"
if ! awk -v pid="pid=$server_pid," '$4 ~ /:5104$/ && index($0,pid) {found=1} END {exit !found}' "$DOCKING_RUN_DIR/listeners_at_start.txt"; then
  echo 'RPC port 5104 is not owned by the started server' >&2
  exit 1
fi

AUTHOR_FIXED_STEPS=1 ARCH_RANGE_LOG="$DOCKING_RUN_DIR/ranges.bin" \
DOCKING_SAFETY_LOG="$DOCKING_RUN_DIR/safety.tsv" \
DOCKING_BOX_LEDGER="$DOCKING_RUN_DIR/initial_boxes.json" \
taskset -c "$DOCKING_CPUSET" "$DOCKING_BINARY" >"$DOCKING_RUN_DIR/native.log" 2>&1 &
native_pid=$!
printf '%s\n' "$native_pid" >"$DOCKING_RUN_DIR/native.pid"
echo "server_pid=$server_pid native_pid=$native_pid port=5104 gpu=$CUDA_VISIBLE_DEVICES cpus=$DOCKING_CPUSET"
set +e
wait "$native_pid"
native_rc=$?
set -e
native_pid=
exit "$native_rc"
