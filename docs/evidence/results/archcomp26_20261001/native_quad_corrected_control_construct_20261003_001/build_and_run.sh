#!/usr/bin/env bash
set -euo pipefail

base=/srv/local/shengenli/flowstar_acceleration_20260921T153643Z
run="$base/runs/archcomp26_20261001/native_quad_corrected_control_construct_20261003_001"
old="$base/runs/archcomp26_20261001/native_quad_allbox_firststep_recenter_gate_20261003_003"
frozen="$base/runs/archcomp26_20261001/native_quad_paper_build_001/archcomp/Quadrotor"
include="$base/runs/author_nncs_reproduction_v1/native_build/deps/extracted/usr/include"

test -f "$run/control_gate.cpp"
test -f "$run/CORRECTED_BIASES.json"
test -f "$old/run/rpc.json"
test -f "$old/flowstar-toolbox/libflowstar.a"
test ! -e "$run/START.json"
export QUAD_GATE_OUTPUT_DIR="$run"
export QUAD_OLD_RPC="$old/run/rpc.json"
export QUAD_CORRECTED_BIASES="$run/CORRECTED_BIASES.json"

python3 -B - <<'PY' >"$run/START.json"
import json, os
from datetime import datetime, timezone
from pathlib import Path
run = Path(os.environ['QUAD_GATE_OUTPUT_DIR'])
old = Path(os.environ['QUAD_OLD_RPC'])
correction = Path(os.environ['QUAD_CORRECTED_BIASES'])
print(json.dumps({
    'run_id': run.name,
    'started_utc': datetime.now(timezone.utc).isoformat(),
    'contract': 'saved first CROWN batch with conditional outward binary32 biases; exact original control construction; no RPC/CROWN/ODE',
    'frozen_cpp_source': str(run.parent / 'native_quad_allbox_firststep_recenter_gate_20261003_003/quad_allbox.cpp'),
    'old_rpc': str(old), 'old_rpc_bytes': old.stat().st_size,
    'corrected_biases': str(correction), 'corrected_biases_bytes': correction.stat().st_size,
    'build_cap_seconds': 300, 'run_cap_seconds': 120,
    'cpu': 'taskset core 12', 'no_digest_verification': True,
}, indent=2))
PY

set +e
timeout -s TERM -k 10 300 taskset -c 12 /usr/bin/g++-15 -O2 -std=c++11 \
  -fpermissive -Wno-template-body -fopenmp \
  -I "$old/flowstar-toolbox" -I "$old/build" -I "$frozen" -I "$include" \
  -I /usr/include/jsoncpp \
  "$run/control_gate.cpp" "$old/flowstar-toolbox/libflowstar.a" \
  -lmpfr -lgmp -lgsl -lgslcblas -lm -lglpk -lcolamd -lamd -lz -lltdl \
  -ljsoncpp -lcurl -ljsonrpccpp-common -ljsonrpccpp-client \
  -l:libboost_thread.so.1.90.0 \
  -o "$run/control_gate" >"$run/build.log" 2>&1
build_rc=$?
set -e
printf '%s\n' "$build_rc" >"$run/build_exit_code.txt"
if [[ "$build_rc" != 0 ]]; then
  python3 -B - "$run" "$build_rc" <<'PY'
import json, sys
from pathlib import Path
run = Path(sys.argv[1])
(run / 'RESULT.json').write_text(json.dumps({
    'status': 'BUILD_FAILED_OR_TIMED_OUT', 'build_exit_code': int(sys.argv[2]),
    'limits': 'No construction result from a failed build.',
}, indent=2) + '\n')
PY
  exit "$build_rc"
fi

set +e
timeout -s TERM -k 10 120 taskset -c 12 "$run/control_gate" \
  >"$run/stdout.log" 2>"$run/stderr.log"
rc=$?
set -e
printf '%s\n' "$rc" >"$run/run_exit_code.txt"
python3 -B - "$run" "$rc" <<'PY'
import csv, json, sys
from pathlib import Path
run = Path(sys.argv[1])
rc = int(sys.argv[2])
path = run / 'construction.csv'
rows = sum(1 for _ in csv.DictReader(path.open())) if path.is_file() else 0
accepted = rc == 0 and rows == 3072 and 'CONTROL_CONSTRUCTION_ACCEPTED_3072_NO_ODE' in (run / 'stdout.log').read_text()
(run / 'RESULT.json').write_text(json.dumps({
    'status': 'NATIVE_CONSTRUCTION_ACCEPTED' if accepted else 'FIRST_REFUSAL_OR_INCOMPLETE',
    'build_exit_code': 0, 'run_exit_code': rc, 'construction_rows': rows,
    'crown_calls': 0, 'ode_calls': 0,
    'limits': 'A finite construction receipt only; CROWN soundness, exact affine identity and plant propagation need separate checks.',
}, indent=2) + '\n')
PY
exit "$rc"
