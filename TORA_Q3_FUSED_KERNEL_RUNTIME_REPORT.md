# TORA-Q3 Fixed-Shape Fused Tensor Kernel

## Result

The complete algorithm-aligned local TORA-Q3 step now has a fixed-shape,
pure-Tensor implementation.  It covers natural polynomial ranges, the K2
polynomial Picard image, initial remainder inclusion, ten fixed remainder
rounds, endpoint/tube bounds, and all local acceptance predicates.  Public
Taylor-model objects and the diagnostic ledger remain outside the compiled
math boundary.

All P0--P5 hard gates pass on the matched PyTorch 2.8.0+cu128 V100 stack.  The
formal B48 full logical one-step median is `0.1296528154052794 s`; the
common-control T20 median is `26.18573095370084 s`.  The T20 result is a
`19.553566325453474x` speedup over the frozen `512.0244269836694 s` baseline
and a `4.028149996989209x` speedup over the prior optimized PyTorch
`105.48005206231028 s` result.  It is an internal PyTorch plant-runtime result,
not Xiangru solver parity.

## Tensor and compilation boundary

The kernel input consists only of the five fixed-shape float64 tensors for
coefficients, remainder lower/upper, and domain lower/upper, plus cached Q3
route metadata and static scalar configuration.  The timed Tensor functions
contain no dataclass construction, ledger append, scalar extraction, CPU
transfer, serialization, or dynamic basis construction.

The deployed boundary uses four `fullgraph=True`, `dynamic=False` stages:

1. F2 polynomial RHS and K2 Picard;
2. F3 remainder initialization;
3. one F3 remainder round, invoked by a fixed ten-call Python schedule with no
   per-round host decision;
4. F4 endpoint, tube, and predicates.

That is 13 fixed compiled invocations per local step.  Every deployed stage has
zero internal graph breaks.  The Python stage boundaries are deliberate and
documented, rather than hidden graph breaks.

The isolated attempts were:

| Boundary | Result | Cold wall | Steady median |
| --- | --- | ---: | ---: |
| F1 natural range | one full graph | 5.18152 s | 0.000114413 s |
| F2 RHS + K2 Picard | one full graph | 63.1562 s | 0.00405470 s |
| F3 init + ten rounds | monolithic attempt stopped at 300 s graph scale; segmented fallback deployed | - | - |
| F4 endpoint/tube/predicates | one full graph | 10.5392 s | 0.000261832 s |
| F5 complete local step | monolithic attempt stopped at 600 s graph scale; four-stage fallback deployed | - | - |

The formal B48 four-stage cold compile, first eager reference, and signature
verification took `209.21813482325524 s`.  B1 and B192 new signatures took
`186.4597845096141 s` and `219.58188796788454 s`, respectively.  The fixed
experiment raises Dynamo's per-frame cache limit to 64 and records it; this
prevents multiple static batch/mode variants from silently exhausting the
default cache.

## Outward soundness

Each compiled signature includes shapes, strides, storage offsets,
`requires_grad`, device, dtype, grad mode, step size, and series length.  On its
first use, the implementation computes both eager and compiled outputs.  It
caches a signature only if all outputs are bitwise equal or if:

- polynomial coefficients are exact;
- compiled remainder, endpoint, and tube intervals outwardly contain eager;
- compiled acceptance predicates never accept a leaf rejected by eager; and
- the compiled initial margin is no larger than eager.

Otherwise the signature is permanently disabled and execution falls back to
eager.  B1, B48, and B192 all verified outward containment; their maximum
initial-margin difference was `2.7755575615628914e-17`.  The formal B48 check
also proves exact coefficients and outward remainder/endpoint/tube containment
against the independent eager `algorithm_aligned_q3` object reference.

The compiled path adds a fixed 16-ULP outward expansion at compiled interval
outputs, then recomputes predicates from those expanded intervals.  The
held-control remainder remains exactly zero.

## Formal runtime protocol

The formal run used one excluded full T20 warm-up, five measured T20 repeats,
ten measured B48 logical steps, CUDA synchronization at every timing boundary,
one CPU thread, grad disabled, and zero controller/serialization time inside
the common-control plant timing.

| Scope | Minimum | Median | Maximum | IQR |
| --- | ---: | ---: | ---: | ---: |
| B48 full logical one-step, 10 repeats | 0.129056 s | 0.129653 s | 0.132416 s | 0.000876 s |
| Common-control T20, 5 repeats | 26.121589 s | 26.185731 s | 26.207486 s | 0.044699 s |

All T20 repeats completed 200/200 segments.  Their checksum was exactly
`190979.0503860716` in all five repeats; the recorded maximum delta is zero
under the declared `1e-9` stability tolerance.  The excluded T20 warm-up was
`26.060500827617943 s`.

Raw fused-local scaling is nearly launch-bound:

| Batch | Steady median | Compile/cache-warm wall |
| ---: | ---: | ---: |
| 1 | 0.0410266 s | 186.460 s |
| 48 | 0.0401622 s | 0.0405349 s |
| 192 | 0.0433791 s | 219.582 s |

Relative to the prior `0.508396873716265 s` logical-step median, the new full
logical median is `3.921217384497794x` faster.  Relative to the prior optimized
T20 result, cold compilation amortizes after about `2.639` T20 runs, or about
`552` individual logical steps.  The new T20 remains `21.6992x` slower than the
matched Xiangru `1.20676 s` runtime.

## Telemetry and gates

| Lane | Compiled graphs | CUDA launch API count | Kineto item/local | Program-issued sync | `aten::to` | Positive CUDA allocation bytes | Peak CUDA bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline_native_k2 | 0 (eager) | 75,440 | 80/80 | 4 | 81 | 12,119,956,480 | 1,031,002,112 |
| algorithm_aligned_q3 | 0 (eager) | 139,417 | 80/80 | 4 | 81 | 12,156,239,360 | 1,030,419,968 |
| fused_segmented | 4 | 7,941 | 7/7 | 1 | 25 | 6,161,790,976 | 1,030,036,480 |

Kineto item/local observations and program-issued dispatcher syncs are kept
separate.  The fused program-issued count is exactly the one fail-closed
validation-batch boundary.  The five-repeat T20 peak CUDA memory was
`1,031,874,048` bytes.

| Gate | Limit | Observed | Result |
| --- | ---: | ---: | --- |
| P0 correctness/soundness | PASS | exact coefficients and outward reference containment | PASS |
| P1 graph breaks | zero or documented | zero inside four graphs; fixed stage boundaries documented | PASS |
| P2 program sync | <=2; stretch 1 | 1 | PASS, stretch met |
| P3 `aten::to` | <=80; stretch <=20 | 25 | PASS, stretch missed |
| P4 B48 logical step | <=0.254 s | 0.129653 s | PASS |
| P5 common-control T20 | <=51.202443 s | 26.185731 s | PASS |
| T20 stretch | <=12.067596 s | 26.185731 s | MISS |

## Public and private evidence

Public aggregate evidence is under
`outputs/tora_q3_stage_parity_fused_20260809/fused_kernel/`.  It contains the
summary, compilation record, one-step and T20 repeat data, scaling data,
profiler telemetry, and all-lane dispatcher audit.  It contains no raw
controller bytes, raw traces, server paths, or private source.

Inductor caches, run metadata containing local paths, and failed pre-formal
debug snapshots remain private and are not tracked by Git.
