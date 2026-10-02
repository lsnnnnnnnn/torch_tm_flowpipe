#!/usr/bin/env bash
set -euo pipefail

base=/srv/local/shengenli/flowstar_acceleration_20260921T153643Z
old="$base/runs/archcomp26_20261001/native_quad_var_tail_repair_gate_20261002_007"
new="$base/runs/archcomp26_20261001/native_quad_initial_recenter_repair_gate_20261003_002"

test -d "$old/flowstar-toolbox"
test -f "$old/quad/build/quad_gate.cpp"
test ! -e "$new"
mkdir -p "$new/quad/build" "$new/quad/on" "$new/quad/off"
cp -a "$old/flowstar-toolbox" "$new/flowstar-toolbox"
cp "$old/quad/build/quad_gate.cpp" "$old/quad/build/quad_gate_observer.h" "$new/quad/build/"
python3 - "$new/flowstar-toolbox/Interval.cpp" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
source = path.read_text()
before = '\tmpfr_sub(radius.value, up, center.value, MPFR_RNDU);\n\n\tmpfr_clear(tmp1);'
after = ('\tmpfr_sub(radius.value, up, center.value, MPFR_RNDU);\n'
         '\tmpfr_sub(tmp1, center.value, lo, MPFR_RNDU);\n'
         '\tif (mpfr_greater_p(tmp1, radius.value))\n'
         '\t\tmpfr_set(radius.value, tmp1, MPFR_RNDU);\n\n'
         '\tmpfr_clear(tmp1);')
if source.count(before) != 1:
    raise SystemExit('expected one toCenterForm radius site')
path.write_text(source.replace(before, after))
PY
diff -u "$old/flowstar-toolbox/Interval.cpp" "$new/flowstar-toolbox/Interval.cpp" > "$new/interval_recenter.patch" || test "$?" -eq 1
test "$(grep -c '^+.*mpfr_sub(tmp1, center.value, lo' "$new/interval_recenter.patch")" -eq 1

taskset -c 10 timeout -s TERM 600 /usr/bin/make -C "$new/flowstar-toolbox" \
    CXX=/usr/bin/g++-15 \
    'CFLAGS=-I . -I /usr/local/include -g -O3 -std=c++11 -fpermissive -Wno-template-body' \
    -j1 lib > "$new/build.log" 2>&1

taskset -c 10 timeout -s TERM 600 /usr/bin/g++-15 -O2 -std=c++11 -fpermissive \
    -Wno-template-body -fopenmp \
    -I "$new/flowstar-toolbox" \
    -I "$base/runs/archcomp26_20261001/native_quad_paper_build_001/archcomp/Quadrotor" \
    -I "$base/runs/author_nncs_reproduction_v1/native_build/deps/extracted/usr/include" \
    -I /usr/include/jsoncpp \
    "$new/quad/build/quad_gate.cpp" "$new/flowstar-toolbox/libflowstar.a" \
    -lmpfr -lgmp -lgsl -lgslcblas -lm -lglpk -lcolamd -lamd -lz -lltdl \
    -ljsoncpp -lcurl -ljsonrpccpp-common -ljsonrpccpp-client \
    -l:libboost_thread.so.1.90.0 \
    -o "$new/quad/build/quad_gate" > "$new/quad/build.log" 2>&1

sed 's/native_quad_var_tail_repair_gate_20261002_007/native_quad_initial_recenter_repair_gate_20261003_002/g' \
    "$old/quad/run_gate.sh" > "$new/quad/run_gate.sh"
chmod +x "$new/quad/run_gate.sh"
test -x "$new/quad/build/quad_gate"
printf 'BUILD_OK\n'
