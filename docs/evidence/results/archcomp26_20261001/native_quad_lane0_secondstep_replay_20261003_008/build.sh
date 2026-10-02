#!/usr/bin/env bash
set -euo pipefail

base=/srv/local/shengenli/flowstar_acceleration_20260921T153643Z/runs/archcomp26_20261001
run="$base/native_quad_lane0_secondstep_replay_20261003_008"
lib="$base/native_quad_initial_recenter_repair_gate_20261003_002/flowstar-toolbox"
test -f "$run/quad_lane0.cpp"
test -f "$run/quad_gate_observer.h"
test -f "$lib/libflowstar.a"

taskset -c 10 timeout -s TERM 300 /usr/bin/g++-15 -O2 -std=c++11 -fpermissive \
  -Wno-template-body -fopenmp \
  -I "$lib" \
  -I "$base/native_quad_paper_build_001/archcomp/Quadrotor" \
  -I /srv/local/shengenli/flowstar_acceleration_20260921T153643Z/runs/author_nncs_reproduction_v1/native_build/deps/extracted/usr/include \
  -I /usr/include/jsoncpp \
  "$run/quad_lane0.cpp" "$lib/libflowstar.a" \
  -lmpfr -lgmp -lgsl -lgslcblas -lm -lglpk -lcolamd -lamd -lz -lltdl \
  -ljsoncpp -lcurl -ljsonrpccpp-common -ljsonrpccpp-client \
  -l:libboost_thread.so.1.90.0 \
  -o "$run/quad_lane0" >"$run/build.log" 2>&1
test -x "$run/quad_lane0"
printf 'BUILD_OK\n'
