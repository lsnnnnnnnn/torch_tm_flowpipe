# TORA-Q3 Native Closed-Loop Closure Report

## Decision

The final classification is **Case C**.  The fused common-control plant passes
the internal 10x P4/P5 performance gates, but no Torch-native closed-loop lane
passes the T5 hierarchy.  The best formal horizon is `4.4 s`; T10 and T20 are
therefore `NOT_RUN`.  No Torch T5/T10/T20 width is published.

This is a reproducible negative native-T20 result, not a common-control
substitution.  The independent Xiangru native reference verifies all 200
segments to `20.0 s`.

## Native hierarchy

All lanes use B48, binary64 complete Q3, the unchanged
`abs(x1..x4) <= 2` property, and ten remainder Picard rounds.  Diagnostic
continuation after a property failure is retained privately for attribution;
it never advances a formal gate.

| Formal lane | Changed contract | one leaf | B48 step | T1 | T5 | T10 | T20 | Certified horizon | First failure |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `baseline_native_k2` | frozen K2 | PASS | PASS | PASS | FAIL | NOT_RUN | NOT_RUN | 4.3 s | segment 44, leaf 0, property |
| `k3_picard` | K2 -> K3 only | PASS | PASS | PASS | FAIL | NOT_RUN | NOT_RUN | 4.4 s | segment 45, leaves 0/1/6, property |
| `algorithm_aligned_q3` | aligned sine/remainder route, K2 retained | PASS | PASS | PASS | FAIL | NOT_RUN | NOT_RUN | 4.3 s | segment 44, leaf 0, property |
| `algorithm_aligned_h005_refresh1` | two h=.05 plant substeps per .1 s report step; controller refresh remains 1.0 s | PASS | PASS | PASS | FAIL | NOT_RUN | NOT_RUN | 4.4 s | segment 45, leaves 0/1/6, property |

At every first failure, all failed leaves still pass `finite_ok`,
`initial_subset_ok`, and `all_remainder_rounds_ok`.  The failure is exclusively
the unchanged physical property, so no numerical-certificate failure is being
reported as a safety result.

## Failure anatomy

The baseline segment-44 leaf-0 x3 tube has center `-0.995792`, radius
`1.046220`, and property margin `-0.042012`.  The aligned lane changes those to
center `-0.995728`, radius `1.045678`, and margin `-0.041406`.  The center moves
by only about `6.43e-5`, whereas the radius exceeds one.  Radius growth, not a
center shift, is the behavior-relevant failure mechanism.

The same conclusion holds for the best lanes at segment 45.  K3's worst x4
tube has maximum absolute center `0.593195`, radius `1.521031`, and margin
`-0.111126`; h=.05 has maximum absolute center `0.617722`, radius `1.513543`,
and margin `-0.127530`.  Their period-5 controller-output radii are `1.795704`
and `1.825201`, respectively.  The controller is reacting to a wide native
input enclosure; it is not the first unequal same-input operation.

At aligned segment 44, the maximum pre-projection polynomial-range width is
only `0.0584778`, while the interval-remainder width is `4.021706`; the latter
is `98.5668%` of their sum.  Compose-then-project and project-then-compose
inflation is only about `2e-12` to `4e-12`.  This is why h=.05 was the single
selected fallback: it targets local remainder/wrapping growth while preserving
the 1.0-second controller schedule.  It improves the horizon by `0.1 s`, but
does not pass T5.  Existing Horner range evidence stayed at `4.3 s`, so no
second range or parameter sweep was run.

The common formal comparison point is segment 43 (`4.3 s`), with no
interpolation:

| Lane | x4 endpoint max width | x4 tube max width | x4 tube center max difference from Xiangru |
| --- | ---: | ---: | ---: |
| baseline K2 | 2.447747 | 2.554788 | 0.084447 |
| K3 | 2.213969 | 2.326468 | 0.067291 |
| algorithm aligned | 2.445568 | 2.552679 | 0.084240 |
| aligned h=.05 | 2.243167 | 2.298497 | 0.042101 |
| Xiangru native | 0.101998 | 0.225424 | 0 |

The width gap is one to two orders of magnitude larger than the center gap.
Xiangru's period-5 maximum pre-control width is `0.141151` and its maximum
action width is `0.386897`; it retains a small enough radius to continue to
T20.  Its complete native run reports `15.637993 s` solver wall excluding
validation and `17.568746 s` including validation.  The Torch failure-prefix
runs have different horizons, so no cross-implementation native-T20 runtime
ratio is claimed.

## Where the tightness difference begins

The first numerical difference is A2 point-sine outward rounding, at only
`4.22e-15`.  The first material difference is A3 sine composition remainder
routing: its one-step width difference reaches `0.0145973` at segment 1,
leaf 0.  The direct T1 endpoint difference `0.014211021942602` is `99.924210%`
present before endpoint projection, after ten accumulated plant steps.

At segment 40, the maximum interval-remainder width is
`1.21861858820087`.  The broad carried `composition_overflow` ledger category
has the same maximum, while the current-step `picard_residual` maximum is only
`0.00126978318227`.  Same-input sine substitution removes at least
`98.174362%` of the local-remainder error; A7/A8 integration overflow is the
secondary residual.  The new aligned lane independently implements the
centered quadratic sine polynomial, signed input-remainder propagation, a
line-segment third-derivative bound, and separated outward remainder routing.
It reduces the local stage mismatch, passes common-control T20, but does not
remove the accumulated native representation/controller-cycle width.

## Why the fused Torch path is faster, and why it is not Xiangru parity

The frozen matched-stack common-control medians are Torch `512.024427 s` and
Xiangru `1.206760 s`, a descriptive `424.296975x` ratio.  Changing the Torch
software environment changes the baseline by only `1.002752x`, about
`0.275%`; software versions therefore explain almost none of the gap.

Before full fusion, `90.898540 s` of the `105.480052 s` T20 median was the
local K2-plus-ten-remainder step.  Generic dense range evaluation, Python
object/ledger boundaries, many small interval operations, and repeated dense
complete-Q3 route traffic produced small CUDA kernels rather than useful
large-batch GPU work.

The deployed fixed-shape implementation covers natural range, K2 polynomial
Picard, initial inclusion, ten remainder rounds, endpoint/tube evaluation, and
all local predicates in four zero-graph-break fullgraphs called through 13
fixed invocations.  Per logical B48 step:

| Telemetry | frozen K2 | aligned eager | fused | Reduction vs frozen |
| --- | ---: | ---: | ---: | ---: |
| CUDA launch APIs | 75,440 | 139,417 | 7,941 | 89.47% |
| Kineto item/local | 80/80 | 80/80 | 7/7 | 91.25% |
| program-issued sync | 4 | 4 | 1 | 75% |
| `aten::to` | 81 | 81 | 25 | 69.14% |

The formal B48 logical-step median falls from `0.508397 s` to `0.129653 s`;
common-control T20 falls from `105.480052 s` to `26.185731 s`.  Relative to
the frozen `512.024427 s` baseline, this is a `19.553566x` internal PyTorch
speedup.  P4 and P5 pass, although the `12.067596 s` stretch target is missed.
The remaining descriptive ratio to Xiangru is `21.6992x`, so this is not GPU
solver parity.

Raw fused-local scaling is almost batch-invariant: B1 `0.041027 s`, B48
`0.040162 s`, and B192 `0.043379 s`.  The GPU amortizes the batch well, but the
fixed launch schedule and dense route/memory traffic dominate.  Remaining
cost is primarily kernel count, materialization and memory traffic across 84
basis slots, and the four-stage/13-invocation boundary; Python scheduling is
now secondary, and changing only the software stack cannot close it.

Cold B48 compilation plus signature verification costs `209.218135 s`.
Against the prior `105.480052 s` path it amortizes after about `2.639` T20
runs, or about `552` individual logical steps.  It is worthwhile for repeated
verification workloads, not a one-off run.

The formal fused T20 peak CUDA allocation is `1,031,874,048` bytes.  A full
protocol resource-only rerun measured peak process RSS `6,925,746,176` bytes
and the same peak CUDA allocation.  Its slower timing is not used for the
performance result; the public aggregate is source- and private-evidence-hash
bound.

## Evidence boundary and remaining work

Public aggregate evidence is in
`outputs/tora_q3_stage_parity_fused_20260809/native_full_loop/`, with separate
endpoint, tube, remainder, property-margin, failure, runtime, and common-horizon
records.  Figures are under the sibling `figures/` directory.  Raw stage
tensors, raw per-leaf traces, controller and ONNX bytes, observation patches,
Inductor caches, commands/environment dumps, and server paths remain private.

The smallest remaining tightness problem is a validated representation that
prevents carried interval remainder from dominating the period-5 controller
input while retaining useful correlations; another sine micro-refinement or
generic range-policy sweep is not supported by the evidence.  The smallest
remaining performance problem is reducing the 13-invocation dense route and
memory schedule without weakening outward containment.

The historical Van der Pol failure near `t=6.397083942944808` remains an
explicitly unresolved, independent line of work.  This TORA branch makes no
VDP claim.
