#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" != 2 || ( "$1" != first && "$1" != two ) || ( "$2" != 0.005 && "$2" != 0.010 ) ]]; then
  echo 'usage: run_one.sh first|two 0.005|0.010' >&2
  exit 2
fi
base=/srv/local/shengenli/flowstar_acceleration_20260921T153643Z/runs/archcomp26_20261001
run="$base/native_quad_lane0_secondstep_replay_20261003_008"
out="$run/$1"
test ! -e "$out"
test -x "$run/quad_lane0"
test -f "$base/native_quad_allbox_firststep_recenter_gate_20261003_003/run/rpc.json"
mkdir "$out"
date -u +%Y-%m-%dT%H:%M:%SZ >"$out/start_utc.txt"
set +e
AUTHOR_FIXED_STEPS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
QUAD_REPLAY_RPC="$base/native_quad_allbox_firststep_recenter_gate_20261003_003/run/rpc.json" \
QUAD_GATE_DURATION="$2" QUAD_GATE_OUTPUT_DIR="$out" QUAD_GATE_OBSERVER=1 \
timeout -s TERM -k 10 120 taskset -c 10 "$run/quad_lane0" \
  >"$out/stdout.log" 2>"$out/stderr.log"
rc=$?
set -e
printf '%s\n' "$rc" >"$out/native_exit_code.txt"
date -u +%Y-%m-%dT%H:%M:%SZ >"$out/end_utc.txt"
exit "$rc"
