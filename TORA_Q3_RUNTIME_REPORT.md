# TORA-Q3 runtime report

Status: **repeated measurements PASS; formal GPU speedup claim DISALLOWED.**

All formal measurements used the same Tesla V100-SXM2-16GB, CUDA, float64,
B48, `h=0.1`, and one host thread.  CUDA was synchronized around measured
regions.  Each full-T20 lane ran one complete excluded warm-up followed by
five measured 200-segment repeats; short Q3 and controller scopes used ten
repeats.  Serialization is excluded from the repeated solver scopes.

## Common-control plant replay

| engine | solver median (s) | IQR (s) | min–max (s) | wall median (s) | peak CPU/GPU |
|---|---:|---:|---:|---:|---:|
| native Torch | 525.862164 | 1.085531 | 522.575163–526.762578 | 525.875456 | 1,074,946,048 / 927,533,568 B |
| Xiangru | 1.033485 | 0.001846 | 1.027768–1.034936 | 1.089862 | 1,455,931,392 / 6,471,680 B |

Every repeat completed 200/200 segments.  Within each engine, all five
checksums were identical.  The checksums are not compared across engines
because their outward-enclosure arithmetic differs.

Median Torch per-repeat totals were 482.353745 s for the local plant step
including remainder validation, 37.994293 s for affine composition and
physical range, 5.283854 s for endpoint projection, 0.202988 s for
normalization, 0.025374 s for period setup, and 0.000112 s for scheduling.
The corresponding Xiangru medians were 1.023048 s for plant propagation,
validation, and range together and 0.010567 s for period setup.  Controller
bound/update and serialization were zero in both lanes because this is frozen
control replay.

The excluded full warm-ups were 526.486789 s (Torch) and 49.103378 s
(Xiangru); Xiangru's first lazy-compiled segment took 46.504523 s.  Separate
cold evidence-export runs, including serialization, were 542.965449 s and
138.844197 s.  The repeated-process wall including warm-up and all five runs
was 3,153.046096 s for Torch and 54.552234 s for Xiangru.

The solver-median quotient is **Torch/Xiangru = 508.824144**.  This is only a
descriptive end-to-end runtime ratio: the engines are method-native, not
algorithm-identical, and the profiler gate below fails.  It is not presented
as a pure-kernel or formal GPU speedup.

## Native Torch short scopes and GPU audit

The complete-Q3 B48 step has 84 terms over six variables.  Route construction
took 0.293989 s; cold process through the first step took 2.860206 s, of which
the first step was 2.428953 s.  Ten-repeat medians were:

| scope | median (s) | IQR (s) |
|---|---:|---:|
| full step including remainder validation | 2.349308 | 0.008878 |
| polynomial RHS | 0.155921 | 0.003539 |
| K2 polynomial Picard | 0.350506 | 0.002144 |
| local-time integration | 0.017528 | 0.000134 |
| affine parameterization composition | 0.133408 | 0.000701 |
| endpoint substitution/evaluation | 0.024664 | 0.000255 |
| full-tube evaluation | 0.015179 | 0.000329 |
| explicit 161,280-byte device-to-host transfer | 0.0000796 | 0.0000020 |

The implementation reports zero sparse, range, and CPU formal-math fallback
paths.  Nevertheless, a profiled full step contained 128,472 paired
`aten::item` / `aten::_local_scalar_dense` host scalar synchronizations and
17,745 `aten::to` events.  Therefore the formal GPU comparison gate is
`FAIL_FREQUENT_HOST_SCALAR_SYNCHRONIZATION`.  Peak CUDA allocation was
927,302,656 B.  Profiler retention raised process peak RSS to 8,809,627,648 B;
that value is a profiler-run peak, not a steady solver-memory claim.

Separate polynomial-propagation and remainder-validation time inside the
monolithic full step is unavailable and is not inferred from totals.

## Controller and native full closed loop

The hash-verified native auto_LiRPA controller's ten-repeat synchronized
median was 0.026925 s (IQR 0.001337 s, range 0.026009–0.031918 s).  Its CROWN
bound median was 0.025866 s and outward-composition median 0.000402 s.  Model
build took 0.656121 s, the excluded warm-up bound took 0.333619 s, and cold
process through the benchmark took 1.870212 s.  Peak CPU/GPU memory was
1,260,015,616 / 59,592,704 B.

Native full-closed-loop T20 runtime is **N/A**: the sound run fails closed at
segment 44 after certifying T=4.3.  The failure run's observed wall was
119.838083 s, including 118.004801 s plant time, but it is neither a T20
measurement nor part of the winner comparison.

Machine-readable repeated rows, raw samples, timing scopes, unavailable-field
declarations, memory values, and the failed GPU claim gate are under
`outputs/tora_q3_native_matched_20260806/runtime/` and `q3_backend/`.
