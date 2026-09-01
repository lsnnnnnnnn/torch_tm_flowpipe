# C3+C4 reference, performance, and CPU batch foundation

This package closes the accepted VDP C3 and Brusselator C4 research lanes into
one named polynomial-plant reference configuration, separates production from
audit overhead, attributes runtime, implements one profile-authorized
semantics-preserving CPU optimization, and establishes independent B1/B2/B8
lane semantics.

## Frozen identity and scope

The source package is
`ed9c305dc39c25eab23a96f4fb3775cc2d13d396` on
`codex/torch-flowstar-brusselator-live-range-c5-20260828`; pinned stock Flow* is
`b85a3211748cb77b736fe4ad42ee02d8d2b81148`. Work was performed in the
independent branch
`codex/c4-reference-performance-batch-foundation-20260829` without modifying
the dirty source checkout.

The reference is named `flowstar_like_polynomial_plant_reference`. It freezes
CPU binary64 outward arithmetic, ordered RHS terms, whole-vector atomic subset
commit, accepted-boundary SR ownership, the 491 replay ceiling,
`STOP_RATIO=0.99`, normal insertion/reset, cutoff and validation epsilon, and
checkpoint/rollback policy. It is an explicit configuration, not a portfolio.
Existing modes and CLIs remain available.

Only VDP and Brusselator are in scope. No Huan work, C5/C6 mode, controller,
CROWN, CUDA kernel, float32 path, endpoint repair, capacity change, range-policy
change, replay increase, or adaptive-order change is part of this package.

## Scientific changes

The observer split makes production timing meaningful while preserving exact
outputs. Profiling then authorized one mechanism only: tensorizing the
independent accepted-boundary SR owner payload while retaining both numerical
loop orders and every outward rounding operation. The original scalar schedule
is a bitwise oracle. This optimization is shared by VDP C3 and Brusselator C4,
uses no cache, and leaves all queue and replay policies unchanged.

The CPU batch layer is deliberately an isolation contract over B1 lanes. It
provides lane-local commit, rejection freeze, queue ownership, chunk invariance,
and checkpoint/resume. Full-solver fusion is deferred; the batch contract is
the oracle required before a future CUDA implementation.

## Reproduction

Profile the frozen reference:

```bash
C4_PROFILE_CODE_ROOT=<clean-reference-root> taskset -c 0 \
conda run -n py11 python experiments/profile_c4_reference_solver.py \
  --output-dir <profile-dir> \
  --observer-repeats 3 \
  --brusselator-prefix-steps 100 \
  --vdp-prefix-steps 20 \
  --tail-checkpoint <bitwise-equivalent-step-900-checkpoint> \
  --tail-start-step 901 --tail-steps 100

# A second invocation with --brusselator-prefix-steps 20 supplies the
# independently measured 1--20 window.
```

Run clean detached scientific roots through the performance gate:

```bash
conda run -n py11 python experiments/run_c4_performance_gate.py \
  --output-dir <gate-dir> \
  --reference-root <clean-reference-root> \
  --optimized-root <clean-optimized-root> \
  --reference-sha <reference-sha> \
  --optimized-sha <optimized-sha> \
  --prefix100-repeats 5 \
  --prefix300-repeats 3 \
  --vdp-prefix-repeats 3 \
  --run-full --run-vdp-regression --cpu 0
```

Run the real B1/B2/B8 contract and verify the package:

```bash
conda run -n py11 python experiments/run_c4_cpu_batch_equivalence.py \
  --output-dir <batch-dir> --runtime-repeats 1 --cpu 0

conda run -n py11 python scripts/package_c4_performance_batch_evidence.py \
  --observer-profile-dir <observer-profile-dir> \
  --prefix-profile-dir <1-20-profile-dir> \
  --formal-profile-dir <formal-profile-dir> \
  --gate-dir <gate-dir> --batch-dir <batch-dir>

conda run -n py11 python scripts/verify_c4_performance_batch_evidence.py --json
```

All runtime CSVs name the exact scientific SHA, CPU affinity, observer mode,
timer scope, repetitions, peak RSS, and checkpoint/snapshot hashes. Evidence
and package commits are recorded separately in `PROVENANCE.json` and do not
replace the scientific identities.

## Acceptance interpretation

Correctness has priority over speed. VDP native T10, its fixed T1/T3/T6.32
snapshots, and the entire scientific segment ledger must match the frozen C3
evidence. Brusselator must accept 1000/1000 steps to T20 with exact reference,
optimized, and historical final endpoint/tube/queue state. Full pytest and a
fresh-clone verifier are separate mandatory checks.

The final status is derived mechanically by the verifier. A semantics-correct
optimization that misses any 2× production gate is reported as
`C4_REFERENCE_FROZEN__CPU_BATCH_FOUNDATION_PASSED__CPU_SPEED_GATE_FAILED`;
no second optimization is added to make the headline pass. CPU batch and both
zero-regression gates, rather than CPU speed, determine whether the next CUDA
batch round is authorized.

The formal measurements reached 1.113911× at Brusselator step 100, 1.280616×
at step 300, and 2.059557× for the full T20 run. Thus correctness, memory,
full-run speed, and CPU-batch gates passed, while the two prefix speed gates
failed. The formal CPU B8/8×B1 ratio was 0.992680×. This is the expected final
classification above and authorizes the next CUDA batch research round without
claiming that CUDA was implemented here.
