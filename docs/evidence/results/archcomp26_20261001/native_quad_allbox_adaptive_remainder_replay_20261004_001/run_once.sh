#!/usr/bin/env bash
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
base=/srv/local/shengenli/flowstar_acceleration_20260921T153643Z/runs/archcomp26_20261001
old="$base/native_quad_allbox_firststep_recenter_gate_20261003_003"
export QUAD_OLD_RPC="$old/run/rpc.json"
export QUAD_SOURCE_BOXES="$base/native_quad_corrected_control_construct_20261003_001/source_boxes.csv"
export QUAD_CORRECTED_BIASES="$base/native_quad_corrected_control_construct_20261003_001/CORRECTED_BIASES.json"
export QUAD_SAME_SLOPE_REPLAY="$base/native_quad_crown_same_slope_batch_20261003_001/RAW_COEFFICIENTS.json"
export QUAD_PRIOR_TRACE="$base/native_quad_allbox_correlated_remainder_gate_20261003_001/ALLBOX_TM_TRACE.jsonl"
export QUAD_REMAINDER_PLAN="$here/PLAN_ENDPOINTS.json"
export QUAD_TRACE_OUT="$here/ALLBOX_TM_TRACE.jsonl"

test ! -e "$here/START.json"
test ! -e "$here/RESULT.json"
test ! -e "$QUAD_TRACE_OUT"
phase=preflight
finish() {
    rc=$?
    trap - EXIT
    printf '%s\n' "$rc" >"$here/wrapper_exit_code.txt"
    python3 -B - "$here" "$rc" "$phase" <<'PY'
import json, sys
from datetime import datetime, timezone
from pathlib import Path
root, rc, phase = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
exits = {}
for stage in ('build', 'run', 'check'):
    path = root / (stage + '_exit_code.txt')
    exits[stage] = int(path.read_text()) if path.exists() else None
with (root / 'EXIT.json').open('x') as stream:
    json.dump({'finished_utc': datetime.now(timezone.utc).isoformat(),
               'wrapper_exit': rc, 'stopped_phase': phase, 'stage_exits': exits}, stream, indent=2)
    stream.write('\n')
if not (root / 'RESULT.json').exists():
    with (root / 'RESULT.json').open('x') as stream:
        json.dump({'status': 'FIRST_REFUSAL_NATIVE_CONSTRUCTION_STAGE', 'phase': phase,
                   'stage_exits': exits, 'wrapper_exit': rc, 'native_production_gate': 'CLOSED',
                   'condition': 'Original real-affine CROWN inequalities hold on saved RPC boxes',
                   'no_crown_calls': True, 'no_ode_calls': True, 'no_digest_verification': True}, stream, indent=2)
        stream.write('\n')
PY
    exit "$rc"
}
trap finish EXIT

python3 -B - "$here/START.json" <<'PY'
import json, sys
from datetime import datetime, timezone
with open(sys.argv[1], 'x') as stream:
    json.dump({'started_utc': datetime.now(timezone.utc).isoformat(),
        'scope': '1024 first-control source boxes, 3072 adaptive native remainder replacements; construction only',
        'build_cap_seconds': 300, 'native_cap_seconds': 180, 'checker_cap_seconds': 300,
        'cpu_affinity': '12', 'no_crown_calls': True, 'no_ode_calls': True,
        'no_digest_verification': True, 'native_production_gate': 'CLOSED'}, stream, indent=2)
    stream.write('\n')
PY
for file in "$here/INPUT.json" "$here/allbox_control_trace.cpp" "$here/check_replay.py" \
            "$here/exact_checker_primitives.py" "$here/PLAN.csv" "$QUAD_REMAINDER_PLAN" \
            "$QUAD_OLD_RPC" "$QUAD_SOURCE_BOXES" "$QUAD_CORRECTED_BIASES" \
            "$QUAD_SAME_SLOPE_REPLAY" "$QUAD_PRIOR_TRACE" "$old/flowstar-toolbox/libflowstar.a"; do
    test -f "$file"
done
python3 -B "$here/check_replay.py" --self-check >"$here/self_check.log"

phase=build
set +e
timeout -s TERM -k 10 300 taskset -c 12 /usr/bin/g++-15 -O2 -std=c++11 \
    -fpermissive -Wno-template-body -fopenmp \
    -I "$old/flowstar-toolbox" -I "$old/build" -I /usr/include/jsoncpp \
    "$here/allbox_control_trace.cpp" "$old/flowstar-toolbox/libflowstar.a" \
    -lmpfr -lgmp -lgsl -lgslcblas -lm -lglpk -lcolamd -lamd -lz -lltdl \
    -ljsoncpp -lcurl -ljsonrpccpp-common -ljsonrpccpp-client \
    -l:libboost_thread.so.1.90.0 -o "$here/allbox_control_trace" >"$here/build.log" 2>&1
build_rc=$?
set -e
printf '%s\n' "$build_rc" >"$here/build_exit_code.txt"
if [[ "$build_rc" != 0 ]]; then exit "$build_rc"; fi

phase=run
set +e
timeout -s TERM -k 10 180 taskset -c 12 "$here/allbox_control_trace" >"$here/stdout.log" 2>"$here/stderr.log"
run_rc=$?
set -e
printf '%s\n' "$run_rc" >"$here/run_exit_code.txt"
if [[ "$run_rc" != 0 ]]; then exit "$run_rc"; fi

phase=check
set +e
timeout -s TERM -k 10 300 taskset -c 12 python3 -B "$here/check_replay.py" >"$here/check_stdout.log" 2>"$here/check_stderr.log"
check_rc=$?
set -e
printf '%s\n' "$check_rc" >"$here/check_exit_code.txt"
exit "$check_rc"
