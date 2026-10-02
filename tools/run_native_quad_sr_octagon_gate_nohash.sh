#!/usr/bin/env bash
set -euo pipefail

base=/srv/local/shengenli/flowstar_acceleration_20260921T153643Z
gate="$base/runs/archcomp26_20261001/native_quad_sr_octagon_gate_20261002_005"
frozen="$base/runs/archcomp26_20261001/native_quad_paper_build_001/archcomp/Quadrotor"
binary="$gate/build/quad_gate"
python="$base/nncs_env/bin/python"
overlay="$base/runs/archcomp26_20261001/native_dp_less_rpc_overlay_preflight_001/rpc_overlay"
export QUAD_MODEL=/srv/local/shengenli/CROWN-Reach-GPU/ARCH-COMP2024/benchmarks/QUAD/quad_controller_3_64_torch.onnx
export PYTHONPATH="$overlay${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export QUAD_RPC_LOG="$gate/controller_rpc.jsonl"

test -x "$binary"
test -f "$QUAD_MODEL"
if ss -ltn | awk '$4 ~ /:5000$/ {found=1} END {exit !found}'; then
  echo 'QUAD gate refused: port 5000 is occupied' >&2
  exit 2
fi
cd "$frozen"
server_pid=
cleanup() {
  if [[ -n "$server_pid" ]]; then
    kill "$server_pid" 2>/dev/null || :
    wait "$server_pid" 2>/dev/null || :
  fi
}
trap cleanup EXIT
taskset -c 10 "$python" -B -u crown_paper.py >"$gate/server.log" 2>&1 &
server_pid=$!
printf '%s\n' "$server_pid" >"$gate/server.pid"
ready=0
for _ in $(seq 1 300); do
  if ! kill -0 "$server_pid" 2>/dev/null; then
    echo 'QUAD gate RPC server exited during startup' >&2
    exit 1
  fi
  if "$python" -B -c 'import socket; socket.create_connection(("127.0.0.1",5000),0.2).close()' >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 0.2
done
if [[ "$ready" != 1 ]]; then
  echo 'QUAD gate RPC server did not listen within 60 seconds' >&2
  exit 1
fi
ss -ltnp >"$gate/listeners_at_start.txt"
if ! awk -v pid="pid=$server_pid," '$4 ~ /:5000$/ && index($0,pid) {found=1} END {exit !found}' "$gate/listeners_at_start.txt"; then
  echo 'QUAD gate RPC port owner mismatch' >&2
  exit 1
fi

for mode in off on; do
  export QUAD_GATE_OUTPUT_DIR="$gate/$mode"
  export ARCH_RANGE_LOG="$gate/$mode/ranges.bin"
  if [[ "$mode" == on ]]; then
    export QUAD_GATE_OBSERVER=1
  else
    unset QUAD_GATE_OBSERVER
  fi
  AUTHOR_FIXED_STEPS=1 timeout -s TERM 120 taskset -c 10 "$binary" \
    >"$gate/$mode/stdout.log" 2>"$gate/$mode/stderr.log"
  echo "$mode completed" >&2
done
