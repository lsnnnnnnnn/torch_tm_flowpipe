# Native Flow* octagon probe 002 — single reach

One `reach(0.15)` call on the same plant-only harmonic oscillator completed
3/3 steps. `directional.csv`, `native_reference.m`, `stdout.log` and
`stderr.log` preserve the separate result. The finite exact-solution
[AUDIT.json](AUDIT.json) again fails with 28/1,512 directional sample
violations. This control rules out a failure caused only by restarting at
period boundaries; it does not identify the underlying Flow* cause.

See the [source and production gate](../../../../ARCHCOMP26_NATIVE_OCTAGON_PRODUCTION_GATE_20261002.md).
The remote source directory is
`/srv/local/shengenli/flowstar_acceleration_20260921T153643Z/runs/archcomp26_20261001/native_octagon_harmonic_probe_20261002_002`.
