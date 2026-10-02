# Native Flow* octagon probe 001 — periodic

Three separate `reach(0.05)` calls on the plant-only harmonic oscillator
completed 3/3 steps. `directional.csv` records four correlated Taylor-model
forms for the whole-step tube and Flow* propagated endpoint; the two bounds
per form encode eight half-plane inequalities. `native_reference.m` is the
original Flow* tube plot for a copy of the same result. The C++ source is
[here](../../../../../tools/native_flowstar_octagon_probe_nohash.cpp), and the
finite exact-solution checker is [here](../../../../../tools/check_native_flowstar_octagon_probe_nohash.py).
The library path, build flags, and both launch commands are in
[REPRO.json](REPRO.json).

**The independent analytic sample gate failed:** [AUDIT.json](AUDIT.json)
records 28/1,512 directional sample violations. Do not use these polygons
as validated reachability evidence. [Gate analysis](../../../../ARCHCOMP26_NATIVE_OCTAGON_PRODUCTION_GATE_20261002.md)
gives the precise first counterexample and production decision. The remote
source directory is
`/srv/local/shengenli/flowstar_acceleration_20260921T153643Z/runs/archcomp26_20261001/native_octagon_harmonic_probe_20261002_001`.
