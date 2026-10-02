# Isolated QUAD initial-recentering repair: first-box, first-step gate

This diagnostic addresses the one-binary64-ULP strip omitted by the frozen
Flow* library's first QUAD RPC input. It uses a **new copied library** at
`/srv/local/shengenli/flowstar_acceleration_20260921T153643Z/runs/archcomp26_20261001/native_quad_initial_recenter_repair_gate_20261003_002`.
The previously copied two-site VAR-tail repair is retained. The original
library, original long run, and earlier isolated gate were not changed or
restarted. No content digest or hash check was performed.

The [only new source change](interval_recenter.patch) is in
`Interval::toCenterForm`: it sets the radius to the larger of `up − center`
and `center − lo`, with both subtractions rounded upward by MPFR. The
[preparation and build script](prepare_build.sh), [library build log](build.log),
[QUAD build log](quad/build.log), copied [gate source](quad/build/quad_gate.cpp),
and actual [checked runner](quad/run_gate.sh) are saved. The runner checked
that each RPC input contained **all twelve coordinates** of the source-defined
first initial box, that `status=2` with one accepted step, and that one native
range record was saved before starting the next arm.

The first source split of each `x1`–`x3` has lower bound
`-0x1.999999999999ap-2` and upper bound `-0x1.3333333333333p-2`. The old RPC
lower bound was `-0x1.9999999999999p-2`, one ULP too high. Both new RPCs have
lower bound `-0x1.999999999999ap-2` and upper bound
`-0x1.3333333333332p-2`; the other nine coordinates also enclose the source
box. Exact rational bounds and all comparisons are in [AUDIT.json](AUDIT.json).
The new and old CROWN JSON coefficients differ slightly, but every coefficient
is equal after the binary32 `asFloat()` transport used by the gate source.

With the 2026 paper ODE, one saved CROWN affine-plus-residual control hull, and
one order-2 symbolic-remainder step of `h=0.005`, observer off/on each made one
RPC call and accepted one step. Their [RPC](quad/on/rpc.json),
[terminal axes](quad/on/terminal_axes.csv), and [ranges](quad/on/ranges.bin)
were equal between arms. The [off](quad/off/SCAN.json) and
[on](quad/on/SCAN.json) range scans found one complete finite record each,
with no reversed component interval. The [finite sample audit](quad/AUDIT.json)
found zero exceedances in 20,520 checks from 513 relaxed-control numerical
samples. The short-gate target printout is `UNKNOWN`: no QUAD reach-and-remain
property was proved in this step.

The independent [directed-interval audit](INDEPENDENT_AUDIT.json) recomputed
the **new RPC domain** through 1,000 outward-rounded Picard substeps. Every
substep passed strict self-inclusion. Broad-box propagation placed 27/32
saved numerical comparisons inside; five were inconclusive through dependency
width, not observed reachable-state counterexamples. The new
[exact-rational algebraic audit](AUDIT.json) recomputed all five using the
new box and control bounds. `x12(0)=x12'=0` gives closed forms for `x10` and
`x11`; the `x6` bound retains its initial-state/control correlation and a
proved bootstrap error. All **20/20 physical composed-state comparisons**
(eight `x1/x2` tube or endpoint directions and twelve terminal axes) lie in
the saved outward-expanded binary64 observer bounds. The five algebraic
comparisons also lie in the unexpanded serialized binary64 columns; the
smallest composed-state margin is about `5.12×10⁻¹⁹`. Twelve physical-state
bounds lie numerically in twelve pre-composition columns, but this is not a
proof of inclusion of the pre symbolic set.

These are one-box, one-step plant and implementation diagnostics under the
saved relaxed control. The algebra uses exact-decimal paper ODE constants;
it does not independently validate Flow* parsing or floating arithmetic,
CROWN or the actual network guarantee, the other 1,023 boxes, later control
periods, `T=5`, or the full-time reach-and-remain property. **The native
octagon production gate remains closed.**

To repeat the read-only audits from the repository root, run:

```bash
/opt/anaconda3/bin/python3 tools/scan_archcomp26_native_ranges_nohash.py --input docs/evidence/results/archcomp26_20261001/native_quad_initial_recenter_repair_gate_20261003_002/quad/on/ranges.bin --output /private/tmp/quad-recenter-on-scan.json --boxes 1 --steps 1 --physical 12
/opt/anaconda3/bin/python3 docs/evidence/results/archcomp26_20261001/native_quad_var_tail_repair_gate_20261002_007/quad/check_native_quad_sr_octagon_gate_nohash.py docs/evidence/results/archcomp26_20261001/native_quad_initial_recenter_repair_gate_20261003_002/quad
/opt/anaconda3/bin/python3 tools/check_native_quad_fullset_interval_gate_nohash.py docs/evidence/results/archcomp26_20261001/native_quad_initial_recenter_repair_gate_20261003_002 > /private/tmp/quad-recenter-directed-interval.json
/opt/anaconda3/bin/python3 docs/evidence/results/archcomp26_20261001/native_quad_initial_recenter_repair_gate_20261003_002/audit_gate_nohash.py
```

The directed-interval checker exits `1` after writing its receipt because its
five broad-box comparisons are inconclusive; the separate algebraic audit
resolves those five physical numerical comparisons. The remote build and run
scripts are historical receipts: do not reuse this run ID or restart the old
experiment.
