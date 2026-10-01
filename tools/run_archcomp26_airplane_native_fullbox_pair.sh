#!/usr/bin/env bash
set -euo pipefail

: "${AIRPLANE_RUN_DIR:?}"
: "${AIRPLANE_WORKDIR:?}"
: "${AIRPLANE_BINARY:?}"
: "${AIRPLANE_SERVER_PYTHON:?}"
: "${AIRPLANE_OVERLAY:?}"
: "${AIRPLANE_MODEL:?}"
: "${AIRPLANE_CPUSET:?}"
: "${AIRPLANE_NATIVE_AS:?}"
: "${CUDA_VISIBLE_DEVICES:?}"

cd "$AIRPLANE_WORKDIR"
test -d "$AIRPLANE_RUN_DIR"
test -x "$AIRPLANE_BINARY"
test -f crown_paper.py
test -f "$AIRPLANE_MODEL"
command -v prlimit >/dev/null
if ss -ltn | awk '$4 ~ /:5105$/ {found=1} END {exit !found}'; then
  echo 'RPC port 5105 occupied' >&2
  exit 2
fi

AIRPLANE_CHECK_LOG="$AIRPLANE_RUN_DIR/ledger_checks.tsv" \
  "$AIRPLANE_BINARY" "$AIRPLANE_RUN_DIR/initial_boxes.json" \
  >"$AIRPLANE_RUN_DIR/ledger.log" 2>&1

export PYTHONPATH="$AIRPLANE_OVERLAY${PYTHONPATH:+:$PYTHONPATH}"
server_pid=
native_pid=
cleanup() {
  if [[ -n "$native_pid" ]]; then kill "$native_pid" 2>/dev/null || :; wait "$native_pid" 2>/dev/null || :; fi
  if [[ -n "$server_pid" ]]; then kill "$server_pid" 2>/dev/null || :; wait "$server_pid" 2>/dev/null || :; fi
}
trap cleanup EXIT

ARCH_RPC_LOG="$AIRPLANE_RUN_DIR/controller_rpc.jsonl" taskset -c "$AIRPLANE_CPUSET" \
  "$AIRPLANE_SERVER_PYTHON" -B -u crown_paper.py >"$AIRPLANE_RUN_DIR/server.log" 2>&1 &
server_pid=$!
printf '%s\n' "$server_pid" >"$AIRPLANE_RUN_DIR/server.pid"
ready=0
for _ in $(seq 1 300); do
  if ! kill -0 "$server_pid" 2>/dev/null; then echo 'RPC server exited during startup' >&2; exit 1; fi
  if "$AIRPLANE_SERVER_PYTHON" -B -c 'import socket; socket.create_connection(("127.0.0.1",5105),0.2).close()' >/dev/null 2>&1; then ready=1; break; fi
  sleep 0.2
done
if [[ "$ready" != 1 ]]; then echo 'RPC server did not listen within 60 s' >&2; exit 1; fi
ss -ltnp >"$AIRPLANE_RUN_DIR/listeners_at_start.txt"
if ! awk -v pid="pid=$server_pid," '$4 ~ /:5105$/ && index($0,pid) {found=1} END {exit !found}' "$AIRPLANE_RUN_DIR/listeners_at_start.txt"; then
  echo 'RPC port 5105 is not owned by the started server' >&2
  exit 1
fi

AUTHOR_FIXED_STEPS=1 ARCH_RANGE_LOG="$AIRPLANE_RUN_DIR/ranges.bin" \
AIRPLANE_CHECK_LOG="$AIRPLANE_RUN_DIR/safety.tsv" \
prlimit --as="$AIRPLANE_NATIVE_AS" -- taskset -c "$AIRPLANE_CPUSET" \
  "$AIRPLANE_BINARY" >"$AIRPLANE_RUN_DIR/native.log" 2>&1 &
native_pid=$!
printf '%s\n' "$native_pid" >"$AIRPLANE_RUN_DIR/native.pid"
echo "server_pid=$server_pid native_pid=$native_pid port=5105 gpu=$CUDA_VISIBLE_DEVICES cpus=$AIRPLANE_CPUSET native_as=$AIRPLANE_NATIVE_AS"
set +e
wait "$native_pid"
native_rc=$?
set -e
native_pid=
exit "$native_rc"
