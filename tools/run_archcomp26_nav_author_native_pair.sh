#!/usr/bin/env bash
set -euo pipefail

: "${NAV_RUN_DIR:?}"
: "${NAV_WORKDIR:?}"
: "${NAV_BINARY:?}"
: "${NAV_SERVER_PYTHON:?}"
: "${NAV_OVERLAY:?}"
: "${NAV_MODEL:?}"
: "${NAV_CPUSET:?}"
: "${AUTHOR_INITIAL_BOXES:?}"
: "${CUDA_VISIBLE_DEVICES:?}"

cd "$NAV_WORKDIR"
test -d "$NAV_RUN_DIR"
test -x "$NAV_BINARY"
test -f crown_author.py
test -f "$NAV_MODEL"
test -f "$AUTHOR_INITIAL_BOXES"
if ss -ltn | awk '$4 ~ /:5110$/ {found=1} END {exit !found}'; then
  echo 'NAV RPC port 5110 occupied' >&2
  exit 2
fi

export PYTHONPATH="$NAV_OVERLAY${PYTHONPATH:+:$PYTHONPATH}"
server_pid=
native_pid=
cleanup() {
  if [[ -n "$native_pid" ]]; then kill "$native_pid" 2>/dev/null || :; wait "$native_pid" 2>/dev/null || :; fi
  if [[ -n "$server_pid" ]]; then kill "$server_pid" 2>/dev/null || :; wait "$server_pid" 2>/dev/null || :; fi
}
trap cleanup EXIT

ARCH_RPC_LOG="$NAV_RUN_DIR/controller_rpc.jsonl" taskset -c "$NAV_CPUSET" \
  "$NAV_SERVER_PYTHON" -B -u crown_author.py >"$NAV_RUN_DIR/server.log" 2>&1 &
server_pid=$!
printf '%s\n' "$server_pid" >"$NAV_RUN_DIR/server.pid"
ready=0
for _ in $(seq 1 300); do
  if ! kill -0 "$server_pid" 2>/dev/null; then echo 'NAV RPC server exited during startup' >&2; exit 1; fi
  if "$NAV_SERVER_PYTHON" -B -c 'import socket; socket.create_connection(("127.0.0.1",5110),0.2).close()' >/dev/null 2>&1; then ready=1; break; fi
  sleep 0.2
done
if [[ "$ready" != 1 ]]; then echo 'NAV RPC server did not listen within 60 s' >&2; exit 1; fi
ss -ltnp >"$NAV_RUN_DIR/listeners_at_start.txt"
if ! awk -v pid="pid=$server_pid," '$4 ~ /:5110$/ && index($0,pid) {found=1} END {exit !found}' "$NAV_RUN_DIR/listeners_at_start.txt"; then
  echo 'NAV RPC port 5110 is not owned by the started server' >&2
  exit 1
fi

AUTHOR_FIXED_STEPS=1 ARCH_RANGE_LOG="$NAV_RUN_DIR/ranges.bin" taskset -c "$NAV_CPUSET" \
  "$NAV_BINARY" >"$NAV_RUN_DIR/native.log" 2>&1 &
native_pid=$!
printf '%s\n' "$native_pid" >"$NAV_RUN_DIR/native.pid"
echo "server_pid=$server_pid native_pid=$native_pid port=5110 gpu=$CUDA_VISIBLE_DEVICES cpus=$NAV_CPUSET"
set +e
wait "$native_pid"
native_rc=$?
set -e
native_pid=
exit "$native_rc"
