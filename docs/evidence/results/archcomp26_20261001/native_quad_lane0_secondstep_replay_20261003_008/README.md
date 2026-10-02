# Isolated paper QUAD lane-0 second-small-step gate

This new diagnostic uses the 2026 paper QUAD ODE and the already saved first
CROWN call. It targets **only source lane 0** and the first two `h=0.005 s`
plant steps of the first `0.1 s` control period. The original long run and
previous gate were not restarted. No hash or digest verification was used.
The native octagon production gate remains **CLOSED**.

## Frozen inputs and native replay

The server run directory is
`/srv/local/shengenli/flowstar_acceleration_20260921T153643Z/runs/archcomp26_20261001/native_quad_lane0_secondstep_replay_20261003_008`.
Its [source](quad_lane0.cpp) and [observer](quad_gate_observer.h) derive from
the saved [all-box first-step diagnostic](../native_quad_allbox_firststep_recenter_gate_20261003_003/README.md), linked against the existing, separate
two-site VAR-tail plus two-sided recentering copied library from
`native_quad_initial_recenter_repair_gate_20261003_002`. The paper ODE,
`u'=0`, source partition, order 2, `h=0.005`, cutoff `1e-6`, symbolic-remainder
path, and float32 CROWN coefficient transfer stay the same. The new source
reads the **saved** 1,024-box RPC response and verifies all 12,288 constructed
input interval pairs exactly match the saved request. It injects the saved
coefficients once, then propagates lane 0 only. No CROWN server, GPU, or second
control update was used.

The [build script](build.sh), [build log](build.log), [120-second capped native
runner](run_one.sh), and [START](START.json) are retained. First, a new
one-step replay wrote [raw first-step outputs](first/) and passed the
[direct receipt audit](FIRST_REPLAY_AUDIT.json): its one binary range record,
eight composed direction intervals, and twelve terminal axis rows are
**exactly equal** to lane 0 in the saved `_003` first-step run. Only then did
a second, new process call `author_matched::reach` once through `T=0.010`,
with the same initial state and held control. It returned `status=2` and
**2/2 accepted small steps**. Its first binary range record and eight
direction rows are again exactly equal to the first replay and `_003`.
The [two-step audit](TWO_REPLAY_AUDIT.json) found 2 complete binary records,
16 direction rows and 12 final terminal axes; the independent
[range scan](TWO_RANGE_SCAN.json) found no nonfinite or reversed intervals.
The [raw two-step outputs](two/) preserve stdout, stderr, exit code, saved
RPC, source boxes, status, ranges, directions, axes, and start/end times.

## Independent plant check for the second step

The [checker](check_step2_plant_nohash.py) reuses the corrected exact-rational
float32 controller-hull construction and directed-Decimal primitives from
the [all-box first-step gate](../native_quad_allbox_independent_plant_adaptive_bootstrap_20261003_006/README.md).
Starting from the source box's saved RPC enclosure, it performs **2,000
strict Picard self-including substeps from `t=0`**, passing an outward interval
enclosure between substeps without a control update or initial-box reset at
`t=0.005`. It accumulates the second-step
whole-step tube from substeps **1,001–2,000 only** and the endpoint at
`t=0.010`. It re-derives and passes all **seven strict first-exit
inequalities** for the full `0.010 s` horizon. At that same horizon it
recomputes the initial-state/control-correlated `x6` bound and exact-rational
closed-form `x10/x11` terminal bounds. The checker had a 30-second alarm and
finished in 3.739 seconds.

The [raw interval audit](PLANT_AUDIT.json) gives **20/20** physical composed
comparisons inside the saved, unexpanded binary64 observer bounds: four
`x1/x2` directions on the second-step tube, four at its endpoint, and all
twelve composed final axes. The same 20/20 fit one-ULP-outward CSV bounds.
The smallest strictly positive raw-bound margin is about `1.024×10⁻¹⁸`
(`x11`); `x12=0` is an exact equality. As concrete second-step final-axis
examples, `x7`'s plant bound is approximately
`[-1.377251×10⁻⁶, 1.455565×10⁻⁵]` inside the saved
`[-5.405546×10⁻⁶, 1.857736×10⁻⁵]`, and `x8`'s is
`[-6.335586×10⁻⁶, 9.541149×10⁻⁶]` inside the saved
`[-1.030261×10⁻⁵, 1.350557×10⁻⁵]`. The concise [RESULT](RESULT.json)
retains counts and scope.

This is a **conditional plant inclusion check**: the saved CROWN
affine-plus-residual set is assumed to contain the actual network controls.
It does not independently certify the CROWN/NN bounder, Flow* parser or
internal floating-point implementation, the other 1,023 source boxes' second
steps, the remaining eighteen steps of the first control period, later
control calls, `T=5`, or the reach-and-remain property. The native elapsed
value is a short diagnostic and gives no speed ranking.

From the repository root, the saved evidence can be re-audited without a
solver, CROWN, GPU, or content digest:

```bash
python3 -B docs/evidence/results/archcomp26_20261001/native_quad_lane0_secondstep_replay_20261003_008/audit_replay_nohash.py first
python3 -B docs/evidence/results/archcomp26_20261001/native_quad_lane0_secondstep_replay_20261003_008/audit_replay_nohash.py two
python3 -B docs/evidence/results/archcomp26_20261001/native_quad_lane0_secondstep_replay_20261003_008/check_step2_plant_nohash.py
```
