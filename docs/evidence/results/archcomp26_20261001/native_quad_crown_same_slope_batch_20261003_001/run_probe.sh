#!/usr/bin/env bash
set -euo pipefail

base=/srv/local/shengenli/flowstar_acceleration_20260921T153643Z
run="$base/runs/archcomp26_20261001/native_quad_crown_same_slope_batch_20261003_001"
frozen="$base/runs/archcomp26_20261001/native_quad_paper_build_001/archcomp/Quadrotor"
old="$base/runs/archcomp26_20261001/native_quad_allbox_firststep_recenter_gate_20261003_003/run/rpc.json"
python="$base/nncs_env/bin/python"
model=/srv/local/shengenli/CROWN-Reach-GPU/ARCH-COMP2024/benchmarks/QUAD/quad_controller_3_64_torch.onnx

test -f "$run/probe.py"
test -f "$old"
test -f "$model"
test -x "$python"
test ! -e "$run/START.json"
if nvidia-smi -i 1 --query-compute-apps=pid --format=csv,noheader | grep -Eq '^[0-9]+'; then
    echo 'GPU 1 acquired a compute process; diagnostic refused before start' >&2
    exit 3
fi

export QUAD_PROBE_RUN="$run"
export QUAD_PROBE_OLD_RPC="$old"
export QUAD_MODEL="$model"
export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1

"$python" -B - <<'PY' >"$run/START.json"
import json, os
from datetime import datetime, timezone
from pathlib import Path
run = Path(os.environ['QUAD_PROBE_RUN'])
old = Path(os.environ['QUAD_PROBE_OLD_RPC'])
model = Path(os.environ['QUAD_MODEL'])
print(json.dumps({
    'run_id': run.name,
    'started_utc': datetime.now(timezone.utc).isoformat(),
    'contract': 'one 1024-box first-call QUAD CROWN same-slope diagnostic; controller only',
    'old_rpc': str(old), 'old_rpc_bytes': old.stat().st_size,
    'model': str(model), 'model_bytes': model.stat().st_size,
    'model_mtime_ns': model.stat().st_mtime_ns,
    'frozen_crown_source': str(run.parent / 'native_quad_paper_build_001/archcomp/Quadrotor/crown_paper.py'),
    'gpu': 'physical 1 via CUDA_VISIBLE_DEVICES=1', 'cpu': 'taskset core 12',
    'cap_seconds': 120, 'no_digest_verification': True,
}, indent=2))
PY

cd "$frozen"
set +e
timeout -s TERM -k 10 120 taskset -c 12 "$python" -B "$run/probe.py" \
    >"$run/stdout.log" 2>"$run/stderr.log"
rc=$?
set -e
printf '%s\n' "$rc" >"$run/exit_code.txt"
if [[ ! -f "$run/RESULT.json" ]]; then
    "$python" -B - "$run" "$rc" <<'PY'
import json, sys
from pathlib import Path
run = Path(sys.argv[1])
rc = int(sys.argv[2])
(run / 'RESULT.json').write_text(json.dumps({
    'status': 'TIMEOUT' if rc == 124 else 'FAILED_BEFORE_RESULT',
    'exit_code': rc,
    'limits': 'No coefficient conclusion from an incomplete diagnostic.',
}, indent=2) + '\n')
PY
fi
exit "$rc"
