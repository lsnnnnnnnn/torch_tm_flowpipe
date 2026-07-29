# Adaptive Flow* trajectory-containment audit

## Decision

The native adaptive Van der Pol configuration is retained only with a
conservative raw-endpoint export: the endpoint box is

`hull(compose + evaluate_time, compose + intEval(tau=[h,h]))`.

The hull delta is added explicitly to the endpoint Taylor model's independent
remainder.  It is not endpoint tightening.  The second operand is Flow*'s
native composed flowpipe evaluated with local time fixed to the accepted step.
The repaired path contains all 9 deterministic corner/midpoint DOP853
trajectories at all 303 endpoints through `T=10`; therefore
`excluded_from_authoritative=false`.  DOP853 sampling remains a numerical
sanity check, not a proof.  The proof-strength claim comes from retaining the
validated native flowpipe evaluation rather than the narrower collapsed path.

## First failure before the repair

The first failure of the leaf-truncation-patched adaptive configuration is:

| field | value |
|---|---:|
| segment | 3 |
| segment start | 0.026250000000000002 |
| absolute endpoint time | 0.041375000000000009 |
| requested/accepted local step | 0.015125000000000006 |
| local domain | `[0, 0.015125000000000006]` |
| state | 0 (`position`) |
| initial box | `[1.1,1.4] × [2.35,2.45]` |
| failing deterministic initial point | `(1.1,2.35)` |
| collapsed Flow* endpoint | `[1.195701727252073,1.4980826940976244]` |
| DOP853 value | `1.1957008958185056` |
| lower under-enclosure gap | `8.314335673276219e-7` |
| configured sanity tolerance | `2e-8` |
| native fixed-domain endpoint | `[1.1946195451854615,1.4991648761642355]` |

The benchmark equation is `position'=velocity`,
`velocity'=velocity-position-position^2*velocity`, with `mu=1`, state order
`(position,velocity)`, and the initial box above.  The generated harness adds
the upstream clock state `t'=1`; it does not enter either physical derivative.

## Comparison matrix

The machine audit rebuilds one identical generated harness against both the
stock upstream library and the audit library.  It also executes the original
benchmark binary and checks its adaptive schedule against the generated
stock-mode harness.

| implementation | segments | collapsed failures | native fixed-domain failures | repaired failures |
|---|---:|---:|---:|---:|
| stock upstream `b85a321` | 290 | 19 | 0 | 0 |
| identical generated stock-mode harness | 290 | 19 | 0 | 0 |
| variable-leaf truncation replay patch | 303 | 9 | 0 | 0 |
| adaptive full-Picard revalidation fallback | 290 | 19 | 0 | 0 |
| leaf patch + adaptive full-Picard fallback | 303 | 9 | 0 | 0 |

The unchanged original benchmark and identical generated stock-mode harness
both have 290 segments and reach `T=10` with the same accepted schedule.
The leaf patch legitimately changes the accepted schedule to 303 segments.
Porting atomic full-Picard refinement revalidation to the adaptive symbolic
overload does not change either failure count.  Thus this Van der Pol issue is
not an adaptive regression introduced by the leaf patch and is not caused by
accepting a cached refinement proposal without full revalidation.

## First divergence and call path

The collapsed and native fixed-domain endpoint bounds first differ at segment
1, before either path misses a deterministic sample.  The first sample miss is
segment 3.

The relevant call path is:

1. the upstream benchmark calls `ODE::reach` with
   `Symbolic_Remainder(initial_set,100)`;
2. Flow* advances through
   `Continuous.cpp:3129`,
   `Flowpipe::advance_adaptive_stepsize(..., Symbolic_Remainder&)`;
3. the stored flowpipe is composed at `Continuous.cpp:433`;
4. the old export collapses time through
   `TaylorModel.h:3433` and `Polynomial.h:509`;
5. the comparison path evaluates the same composed Taylor model on the fixed
   endpoint domain through `TaylorModel.h:2987`.

Evaluating `tmvPre` at the endpoint before composition behaves like the
collapsed path and also misses samples (19 stock, 10 leaf-patched).  Only the
general native fixed-domain evaluation preserves the conservative enclosure.
This localizes the observed discrepancy to endpoint restriction/evaluation,
not to the ODE, initial condition, absolute/local time mapping, original
schedule, leaf-patch regression, or full-Picard refinement acceptance.

The exporter repair is at
`experiments/three_tool_deep_study/export_flowstar_segment.py`; the adaptive
parity/native runner uses the same rule in
`experiments/three_way_comparison_repair/run_flowstar_audit.py`.

## Reproducible evidence

Every formal run writes:

- `flowstar_adaptive_trajectory_audit/flowstar_adaptive_failure_trace.json`;
- `flowstar_adaptive_trajectory_audit/flowstar_adaptive_trajectory_summary.json`;
- `flowstar_adaptive_trajectory_audit/adaptive_flowstar_repaired_rows.csv`;
- per-variant stdout/stderr and the stock-upstream generated executable.

The trace includes the benchmark/exporter SHA-256 values and Torch, upstream
Flow*, and audit Flow* Git SHAs.  The audit Flow* fallback extension is local
commit `2310c1ac55357d0b48af3b37495a82a3e10ea4ff`; its portable patch is
`flowstar_patches/fa39f7a_series/0005-extend-full-Picard-revalidation-to-adaptive-overloads.patch`.
The upstream Flow* remote rejects writes, so the portable patch is the
authoritative reproduction mechanism for that final local audit commit.

Regression coverage checks the Riccati analytic endpoint, the exact first
adaptive Van der Pol miss, zero failures for the native fixed-domain repair,
original `T=10` schedule parity, and equality/containment between exported raw
endpoint boxes and their recorded native fixed-domain path.
