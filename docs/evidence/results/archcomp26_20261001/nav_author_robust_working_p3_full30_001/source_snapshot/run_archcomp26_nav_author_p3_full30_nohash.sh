#!/usr/bin/env bash
set -euo pipefail
umask 077

root=/srv/local/shengenli/flowstar_acceleration_20260921T153643Z
instance=${NAV_INSTANCE:-nav-standard}
case "$instance" in
  nav-standard) variant=standard ;;
  nav-robust) variant=robust ;;
  *) printf 'unsupported NAV instance: %s\n' "$instance" >&2; exit 2 ;;
esac
run_id="nav_author_${variant}_working_p3_full30_001"
source_dir="$root/runs/archcomp26_20261001/nav_author_${variant}_working_p3_full30_source_001"
run_dir="$root/runs/archcomp26_20261001/$run_id"
python="$root/nncs_env/bin/python"

test ! -e "$run_dir"
printf '%s\n' "$$" > "$source_dir/launcher.pid"
started=$(date -u +%Y-%m-%dT%H:%M:%SZ)
set +e
timeout -s TERM -k 10s 3600s taskset -c 6-9 env \
  CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTHONDONTWRITEBYTECODE=1 NAV_P3_TIMEOUT_S=3600 \
  "$python" -B -u "$source_dir/archcomp26_nav_author_p3_full30_nohash.py" \
  --instance "$instance" --output "$run_dir" > "$source_dir/console.log" 2>&1 &
supervisor=$!
printf '%s\n' "$supervisor" > "$source_dir/supervisor.pid"
wait "$supervisor"
status=$?
set -e
ended=$(date -u +%Y-%m-%dT%H:%M:%SZ)
printf '{"run_id":"%s","started_utc":"%s","ended_utc":"%s","supervisor_exit_code":%d,"timed_out":%s,"launcher_pid":%d,"supervisor_pid":%d}\n' \
  "$run_id" "$started" "$ended" "$status" "$([[ $status -eq 124 ]] && printf true || printf false)" "$$" "$supervisor" \
  > "$source_dir/SUPERVISOR_RESULT.json"
if test -d "$run_dir"; then
  cp -p "$source_dir/SUPERVISOR_RESULT.json" "$run_dir/SUPERVISOR_RESULT.json"
fi
exit "$status"
