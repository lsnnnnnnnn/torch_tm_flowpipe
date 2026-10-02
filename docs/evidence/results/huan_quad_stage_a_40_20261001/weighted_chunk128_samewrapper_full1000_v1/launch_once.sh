#!/usr/bin/env bash
set -euo pipefail

root=/srv/local/shengenli/flowstar_acceleration_20260921T153643Z/runs/archcomp26_20261001/p3_quad_old_weighted128_full1000_samewrapper_nohash_20261003_001
py=/srv/local/shengenli/flowstar_acceleration_20260921T153643Z/nncs_env/bin/python
supervisor=/srv/local/shengenli/flowstar_acceleration_20260921T153643Z/runs/archcomp26_20261001/run_archcomp26_nohash.py

"$py" -B - "$root/PREFLIGHT.json" <<'PY'
import json,sys
assert json.load(open(sys.argv[1]))["status"] == "passed"
PY
test ! -e "$root/run_001"
test ! -e "$root/launcher_pid.txt"
used=$(nvidia-smi -i 3 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')
test "$used" = 0

nohup setsid "$py" -B "$supervisor" \
  --run-dir "$root/run_001" \
  --instance quad-old-author-stage-a-1000 \
  --method ours-p3-strict-weighted128-candidate \
  --contract-label quad-old-author-P3-1000-weighted128-nojit-nohash \
  --cwd "$root" --timeout-s 2520 \
  --env CUDA_VISIBLE_DEVICES=3 --env PYTHONDONTWRITEBYTECODE=1 \
  -- "$py" -B "$root/nncs_watchdog_gpu14.py" \
     --output "$root/run_001/watch" --timeout 2400 \
     -- taskset -c 14-17 "$py" -B "$root/runner.py" \
        --mode observer_on --source-config "$root/quad_author_resolved.yaml" \
        --output "$root/run_001/data" \
  > "$root/launcher_stdout.log" 2> "$root/launcher_stderr.log" < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$root/launcher_pid.txt"
sleep 1
kill -0 "$pid"
printf 'launched_pid=%s inner_timeout_s=2400 outer_timeout_s=2520\n' "$pid"
