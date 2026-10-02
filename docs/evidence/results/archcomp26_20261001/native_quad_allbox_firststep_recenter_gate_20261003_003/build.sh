#!/usr/bin/env bash
set -euo pipefail

base=/srv/local/shengenli/flowstar_acceleration_20260921T153643Z
run="$base/runs/archcomp26_20261001/native_quad_allbox_firststep_recenter_gate_20261003_003"
test -f "$run/build/quad_allbox.cpp"
test -f "$run/build/quad_gate_observer.h"
test -f "$run/flowstar-toolbox/libflowstar.a"

taskset -c 10 timeout -s TERM 300 /usr/bin/g++-15 -O2 -std=c++11 -fpermissive \
  -Wno-template-body -fopenmp \
  -I "$run/flowstar-toolbox" \
  -I "$base/runs/archcomp26_20261001/native_quad_paper_build_001/archcomp/Quadrotor" \
  -I "$base/runs/author_nncs_reproduction_v1/native_build/deps/extracted/usr/include" \
  -I /usr/include/jsoncpp \
  "$run/build/quad_allbox.cpp" "$run/flowstar-toolbox/libflowstar.a" \
  -lmpfr -lgmp -lgsl -lgslcblas -lm -lglpk -lcolamd -lamd -lz -lltdl \
  -ljsoncpp -lcurl -ljsonrpccpp-common -ljsonrpccpp-client \
  -l:libboost_thread.so.1.90.0 \
  -o "$run/build/quad_allbox" >"$run/build.log" 2>&1
test -x "$run/build/quad_allbox"
printf 'BUILD_OK\n'
