#!/usr/bin/env bash
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
base=/srv/local/shengenli/flowstar_acceleration_20260921T153643Z/runs/archcomp26_20261001
old="$base/native_quad_allbox_firststep_recenter_gate_20261003_003"
export QUAD_OLD_RPC="$old/run/rpc.json"
export QUAD_SOURCE_BOXES="$base/native_quad_corrected_control_construct_20261003_001/source_boxes.csv"
export QUAD_CORRECTED_BIASES="$base/native_quad_corrected_control_construct_20261003_001/CORRECTED_BIASES.json"
export QUAD_SAME_SLOPE_REPLAY="$base/native_quad_crown_same_slope_batch_20261003_001/RAW_COEFFICIENTS.json"
export QUAD_TRACE_OUT="$here/LANE0_TM_TRACE.json"

for file in "$here/lane0_control_trace.cpp" "$here/check_lane0.py" "$QUAD_OLD_RPC" \
            "$QUAD_SOURCE_BOXES" "$QUAD_CORRECTED_BIASES" "$QUAD_SAME_SLOPE_REPLAY" \
            "$old/flowstar-toolbox/libflowstar.a"; do
    test -f "$file"
done
test ! -e "$here/START.json"
test ! -e "$QUAD_TRACE_OUT"
test ! -e "$here/RESULT.json"
python3 -B - "$here/START.json" <<'PY'
import json, sys
from datetime import datetime, timezone
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({
    "started_utc": datetime.now(timezone.utc).isoformat(),
    "scope": "lane 0, three controls, corrected first-batch biases",
    "native_build_cap_seconds": 300,
    "native_run_cap_seconds": 120,
    "no_crown_calls": True,
    "no_ode_calls": True,
    "no_digest_verification": True,
}, indent=2) + "\n")
PY

set +e
timeout -s TERM -k 10 300 taskset -c 12 /usr/bin/g++-15 -O2 -std=c++11 \
    -fpermissive -Wno-template-body -fopenmp \
    -I "$old/flowstar-toolbox" -I "$old/build" -I /usr/include/jsoncpp \
    "$here/lane0_control_trace.cpp" "$old/flowstar-toolbox/libflowstar.a" \
    -lmpfr -lgmp -lgsl -lgslcblas -lm -lglpk -lcolamd -lamd -lz -lltdl \
    -ljsoncpp -lcurl -ljsonrpccpp-common -ljsonrpccpp-client \
    -l:libboost_thread.so.1.90.0 \
    -o "$here/lane0_control_trace" >"$here/build.log" 2>&1
build_rc=$?
set -e
printf '%s\n' "$build_rc" >"$here/build_exit_code.txt"
if [[ "$build_rc" != 0 ]]; then exit "$build_rc"; fi

set +e
timeout -s TERM -k 10 120 taskset -c 12 "$here/lane0_control_trace" \
    >"$here/stdout.log" 2>"$here/stderr.log"
run_rc=$?
set -e
printf '%s\n' "$run_rc" >"$here/run_exit_code.txt"
if [[ "$run_rc" != 0 ]]; then exit "$run_rc"; fi
set +e
python3 -B "$here/check_lane0.py" --trace "$QUAD_TRACE_OUT" --output "$here/RESULT.json" \
    >"$here/check_stdout.log" 2>"$here/check_stderr.log"
check_rc=$?
set -e
printf '%s\n' "$check_rc" >"$here/check_exit_code.txt"
exit "$check_rc"
