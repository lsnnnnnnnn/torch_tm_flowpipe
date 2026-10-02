#!/usr/bin/env bash
set -euo pipefail

base=/srv/local/shengenli/flowstar_acceleration_20260921T153643Z
run="$base/runs/archcomp26_20261001/native_quad_allbox_firststep_recenter_gate_20261003_003"
frozen="$base/runs/archcomp26_20261001/native_quad_paper_build_001/archcomp/Quadrotor"
python="$base/nncs_env/bin/python"
overlay="$base/runs/archcomp26_20261001/native_dp_less_rpc_overlay_preflight_001/rpc_overlay"
export QUAD_MODEL=/srv/local/shengenli/CROWN-Reach-GPU/ARCH-COMP2024/benchmarks/QUAD/quad_controller_3_64_torch.onnx
export PYTHONPATH="$overlay${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export QUAD_RPC_LOG="$run/run/controller_rpc.jsonl"
export QUAD_GATE_OUTPUT_DIR="$run/run"
export QUAD_GATE_OBSERVER=1

test -x "$run/build/quad_allbox"
test -f "$QUAD_MODEL"
if ss -ltn | awk '$4 ~ /:5000$/ {found=1} END {exit !found}'; then
  echo 'all-box gate refused: port 5000 is occupied' >&2
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
taskset -c 10 "$python" -B -u crown_paper.py >"$run/run/server.log" 2>&1 &
server_pid=$!
printf '%s\n' "$server_pid" >"$run/run/server.pid"
ready=0
for _ in $(seq 1 300); do
  if ! kill -0 "$server_pid" 2>/dev/null; then
    echo 'all-box gate RPC server exited during startup' >&2
    exit 1
  fi
  if "$python" -B -c 'import socket; socket.create_connection(("127.0.0.1",5000),0.2).close()' >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 0.2
done
if [[ "$ready" != 1 ]]; then
  echo 'all-box gate RPC server did not listen within 60 seconds' >&2
  exit 1
fi
ss -ltnp >"$run/run/listeners_at_start.txt"
if ! awk -v pid="pid=$server_pid," '$4 ~ /:5000$/ && index($0,pid) {found=1} END {exit !found}' "$run/run/listeners_at_start.txt"; then
  echo 'all-box gate RPC port owner mismatch' >&2
  exit 1
fi

set +e
AUTHOR_FIXED_STEPS=1 timeout -s TERM -k 10 1080 taskset -c 10 "$run/build/quad_allbox" \
  >"$run/run/stdout.log" 2>"$run/run/stderr.log"
rc=$?
set -e
printf '%s\n' "$rc" >"$run/run/native_exit_code.txt"
exit "$rc"
