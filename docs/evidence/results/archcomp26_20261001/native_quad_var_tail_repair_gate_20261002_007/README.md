# Isolated native QUAD VAR-tail repair gate — finite diagnostics passed

The frozen 2026 paper-equation QUAD Flow* source was copied to the new server
directory `/srv/local/shengenli/flowstar_acceleration_20260921T153643Z/runs/archcomp26_20261001/native_quad_var_tail_repair_gate_20261002_007`.
The original source, archive, and 50-period run were not edited or restarted.
The [preparation receipt](PREPARE.json) records the copy stage only, so its
`compiled: false` field precedes the later successful full rebuild recorded in
[BUILD.json](BUILD.json) and the [raw build log](build.log). No content digest
was calculated.

The [exact two-site `expression.h` patch](expression.patch) caches the fixed
variable polynomial truncation tail during full evaluation and replays it with
each refined input remainder. The copied `Continuous.cpp` retains both ordinary
and symbolic-remainder refinement loops. The candidate binaries linked the
new archive explicitly; their compiler logs are in [harmonic/build.log](harmonic/build.log)
and [quad/build.log](quad/build.log). Rebuilt binaries and the archive remain
on the server rather than in this evidence mirror.

The ordinary harmonic probe accepted all three `h=0.05`, order-4 steps in both
[periodic](harmonic/periodic/AUDIT.json) and [single reach](harmonic/single/AUDIT.json)
forms. Each produced 24 saved direction intervals and three native plot
segments; observer on/off intervals matched exactly. The independent analytic
checker found **0 failures in 1,512 point-direction checks** for each form.
The saved [periodic directions](harmonic/periodic/directional.csv),
[single directions](harmonic/single/directional.csv), and raw stdout/stderr
remain here.

The same first of 1,024 paper-QUAD boxes then ran one CROWN call and one
`h=0.005`, order-2 symbolic-remainder step in separate observer off/on native
processes. Both exited successfully with one accepted step and status
`2 = COMPLETED_SAFE` for the empty short-gate safety set. The
[independent audit](quad/AUDIT.json) found equal off/on status, terminal axes,
saved `ranges.bin`, and RPC JSON, with exactly two server requests, each
containing one input and one output box. Its 513 SciPy affine-plus-residual
control witnesses produced **0 failures in 20,520 direction and terminal-axis
checks**, including the former x7/x8 failures. The raw
[off](quad/off/terminal_axes.csv) and [on](quad/on/terminal_axes.csv) terminal
axes, [directional observer](quad/on/octagon.csv), [RPC log](quad/controller_rpc.jsonl),
[server log](quad/server.log), saved [off](quad/off/ranges.bin) and
[on](quad/on/ranges.bin) ranges, and stdout/stderr are mirrored here. The
[runner](quad/run_gate.sh) changes only its output gate path from the prior
short diagnostic.

The [saved width comparison](quad/WIDTH_COMPARISON.json) uses the same RPC
output as the original failing gate and the skip-refinement diagnostic. All
32 repaired intervals lie within the skip-refinement intervals; 30 are
strictly narrower and two are equal. The repaired x7 endpoint width is
`8.05799e-6` versus `1.11841e-3` without refinement (138.8 times narrower);
x8 is `7.93939e-6` versus `1.10777e-3` (139.5 times narrower). The
[original gate](../native_quad_sr_octagon_gate_20261002_005/README.md) had
2,048 sampled x7/x8 terminal failures; the
[skip-refinement gate](../native_quad_sr_refinement_fix_20261002_006/README.md)
was wider. These comparisons describe this first box and first step only.

This is a **finite numerical regression diagnostic**, not a proof that the
repaired implementation encloses every trajectory, that the actual neural
controller is soundly relaxed, or that the full `T=5` QUAD property holds.
The old long-run `VERIFIED` receipt is unchanged and is not requalified by
this short gate. A complete independent enclosure and longer-step validation
remain necessary before production use. No full QUAD run was launched here.
