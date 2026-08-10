# Fixed-support compiled core (2026-08-10)

## Outcome

The object API was refactored behind an immutable `FixedSupportKernelPlan` and
a 26-tensor functional state. Object eager and functional eager are bit-exact
for CPU B1/B8/B64 at 1, 2, 10, and 100 steps and CUDA B1/B8/B64 at 10 steps.
Summary and trace modes have bit-exact final tensor states. Per-batch failure
is frozen in tensors and only the final decision is transferred.

Inductor compiles the complete logical step with `fullgraph=True` and zero
graph breaks, but it changes floating-point reduction arithmetic. All compiled
rows are therefore `performance_only_empirical_arithmetic_changed`, with the
explicit stop outcome `FIXED_SUPPORT_COMPILE_SEMANTICS_CHANGED`. The timings
below are observations, not same-semantics speedups.

## Isolated B64 T10 observations

| device | frozen object warm median | functional eager | compiled stable warm median | raw runtime ratio | compile + first execute | first lazy warm | result |
|---|---:|---:|---:|---:|---:|---:|---|
| CPU | 75.592882 s | 46.335458 s | 5.038308 s | 15.00x | 148.672215 s | 153.743966 s | complete, arithmetic changed |
| V100 | 127.233569 s | 118.174358 s | 6.926640 s | 18.37x | 119.707762 s | 120.685468 s | complete, arithmetic changed |

The frozen object source is `4bb10d5`; all 22 matrix rows share that SHA. The
CPU and V100 object runs were sequential and their T10 warm vectors are stored
in full. The first compiled warm call performs additional lazy compilation and
is retained separately rather than folded into the stable warm set. CPU is
about 1.38x faster than V100 for this compiled signature, so the result is not
described as GPU acceleration.

CPU full-run finite differences reached `2.8421709430404007e-13`; V100 also
failed first/later-input and full-run bit-exact gates (maximum finite full-run
difference `6.856737400084967e-13`). Both completed all B64 steps. B1/B8 CUDA signatures likewise completed their 10-step requests with
zero graph breaks and arithmetic differences; the second post-lazy runs were
0.063524 s and 0.066911 s respectively.

The unpartitioned CPU B1 ordinary and compiled T10 signatures both first fail
at zero-based step 536; B64 real partitions complete. This decision agreement
is recorded separately from bit-exact arithmetic.

## Boundary choice

Only the preregistered boundaries were evaluated. One logical step was the
smallest executable useful boundary and was selected. CPU B1 chunk-10 did not
compile within 180 s; chunk-100 and chunk-1000 did not compile within 90 s
each. No other chunk size was swept.

The B64 T10 core performs zero host synchronizations and zero solver device
transfers, followed by one final decision synchronization. The object solver
performs 1000 inclusion-gate synchronizations. Compile-process peak RSS was
about 1.70 GiB on CPU and 2.02 GiB for the V100 process; peak allocated V100
memory was about 282 MiB. A cache replay profiler observes 369 CUDA kernel
events for one complete logical step and zero `item`, local-scalar, `to`,
`stack`, `index`, `nextafter`, `min`, or `max` calls at the compiled boundary.
The prerefactor object kernel-launch count was not captured and is reported as
unavailable. Compilation and execution memory are not conflated.

## Evidence

Machine tables and raw summaries are in the
[current run](../outputs/structured_remainder_compiled_fixed_support_20260810/20260810T070908Z/).
The fixed equivalence artifact contains every field comparison; compiled
summaries contain six same-signature probes, full-run differences, Dynamo
counters, synchronization counts, memory, and the exact numerical class.
