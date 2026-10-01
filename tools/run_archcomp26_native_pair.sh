#!/usr/bin/env bash
set -euo pipefail

: "${DP_RUN_DIR:?}"
: "${DP_WORKDIR:?}"
: "${DP_BINARY:?}"
: "${DP_SERVER_PYTHON:?}"
: "${DP_OVERLAY:?}"
: "${DP_CPUSET:?}"
: "${CUDA_VISIBLE_DEVICES:?}"
DP_PORT=${DP_PORT:-5100}

cd "$DP_WORKDIR"
test -d "$DP_RUN_DIR"
test -x "$DP_BINARY"
test -f observed_server.py
test -f ../../ARCH-COMP2024/benchmarks/Double_Pendulum/controller_double_pendulum_less_robust.onnx
if ss -ltn | awk -v port=":$DP_PORT" '$4 ~ port "$" {found=1} END {exit !found}'; then
  echo "RPC port $DP_PORT is occupied" >&2
  exit 2
fi

export PYTHONPATH="$DP_OVERLAY${PYTHONPATH:+:$PYTHONPATH}"
native_pid=
server_pid=
cleanup() {
  if [[ -n "$native_pid" ]]; then kill "$native_pid" 2>/dev/null || :; wait "$native_pid" 2>/dev/null || :; fi
  if [[ -n "$server_pid" ]]; then kill "$server_pid" 2>/dev/null || :; wait "$server_pid" 2>/dev/null || :; fi
}
trap cleanup EXIT

ARCH_RPC_LOG="$DP_RUN_DIR/controller_rpc.jsonl" taskset -c "$DP_CPUSET" \
  "$DP_SERVER_PYTHON" -u observed_server.py >"$DP_RUN_DIR/server.log" 2>&1 &
server_pid=$!
printf '%s\n' "$server_pid" >"$DP_RUN_DIR/server.pid"
ready=0
for _ in $(seq 1 300); do
  if ! kill -0 "$server_pid" 2>/dev/null; then echo 'RPC server exited during startup' >&2; exit 1; fi
  if "$DP_SERVER_PYTHON" -B -c 'import socket,sys; socket.create_connection(("127.0.0.1",int(sys.argv[1])),0.2).close()' "$DP_PORT" >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 0.2
done
if [[ "$ready" != 1 ]]; then echo 'RPC server did not listen within 60 seconds' >&2; exit 1; fi
ss -ltnp >"$DP_RUN_DIR/listeners_at_start.txt"
if ! awk -v port=":$DP_PORT" -v pid="pid=$server_pid," '$4 ~ port "$" && index($0,pid) {found=1} END {exit !found}' "$DP_RUN_DIR/listeners_at_start.txt"; then
  echo "RPC port $DP_PORT is not owned by server PID $server_pid" >&2
  exit 1
fi

AUTHOR_FIXED_STEPS=1 ARCH_RANGE_LOG="$DP_RUN_DIR/ranges.bin" taskset -c "$DP_CPUSET" \
  "$DP_BINARY" >"$DP_RUN_DIR/native.log" 2>&1 &
native_pid=$!
printf '%s\n' "$native_pid" >"$DP_RUN_DIR/native.pid"
echo "server_pid=$server_pid native_pid=$native_pid port=$DP_PORT gpu=$CUDA_VISIBLE_DEVICES cpus=$DP_CPUSET"
set +e
wait "$native_pid"
native_rc=$?
set -e
native_pid=
exit "$native_rc"
