# Results status

## Current status (2026-08-11)

The current Flow*/Torch result is
`FLOWSTAR_TORCH_FIXED_SCHEDULE_COMMON_PREFIX_ONLY`: Flow* reaches T10, Torch
accepts 632 fixed steps and rejects candidate 633. The current explicit-f64
DiffReach/Torch result is `DIFFREACH_TORCH_DR7_FULL_HORIZON_DIVERGED`: both
reach T10 with equal masks, but operator state diverges at step 1 and J/Phi
plus endpoint/tube equality fail. Complete-O4 carry accounting selects C4
`CARRY_MISSING_SYMBOLIC_SEMANTICS`; dense CNI parity is not expressible and
the only allowed implementation decision is `NO_FIX_AUTHORIZED`. There is no
universal timing, tightness, or three-tool ranking.

The S1 statement below is the immediately preceding research result.

The latest preserved S1 result is
`S1_REACHES_TERMINAL_BUT_DOES_NOT_CLOSE_IT`: corrected carry passes the 307-step
frozen accepted prefix and the unchanged historical terminal step rejects.
This supersedes `S1_PREFIX_REJECTS_BEFORE_TERMINAL` as the current headline.
Fresh-horizon and second-system results remain `not_run_after_stop`. Native
Flow*, DiffReach and Torch rows remain capability facts and are not a universal
speed or tightness ranking. See
[the corrected-carry result](S1_CORRECTED_CARRY_RESULT_20260811.md) and
[the evidence-integrity corrections](EVIDENCE_INTEGRITY_CORRECTIONS_20260811.md).

The sections below are earlier closure history.

## Historical terminal-range result

The dense backend now has a safe canonical terminal checkpoint/replay path and
a fully tested batched subdivision polynomial-range evaluator. On the exact
original terminal pre-state, natural range rejects with y margin
`-5.111670937766742e-6`; the focused four-leaf
`polynomial_truncation` range accepts the same coefficient payload, validation
target, and `h=0.0039859994324420315` with y margin
`2.8883253329832075e-5`. This establishes R3 without repair or fallback.

The one pre-registered proactive depth-1 adjustment reaches
`6.397083942944808` fresh from t=0 and establishes R4. It does not complete
T=6.5, T=7.5, or T=10. All three requests deterministically stop at the same
later terminal, where the unchanged raw-remainder y self-map margin is
`-1.99995911680722e-5`; subdivision depths through the 64-leaf cap give the
same bound. R5–R7 are false, and no successful second T=10 reproduction is
claimed. See [the terminal-range closure](VDP_TERMINAL_RANGE_CLOSURE.md) and
its complete hashed evidence bundle.

## Current dense-backend result

The following paragraph records the preceding S3 baseline. The generic dense backend had passed operator, one-step, CUDA, and short
multi-step gates and is integrated as `hybrid_dense_core` (S3). It does not
complete authoritative VDP order-4 T=10: the exact unmodified validated horizon
is 6.3172908799330765, classified `minimum_step_reached`. The single-factor
range-midpoint diagnostic reaches 6.390931109681597 and remains incomplete.
Neither result uses endpoint repair or hidden fallback. These internal results
do not open any cross-tool speed/ranking claim.

No cross-tool speedup, Pareto frontier, winner or runtime/tightness ranking is
citable.  Workloads differ in plant/controller, partitions, effective support,
device and timing boundary, and every row is currently
`primary_comparison_eligible=false`.

## Citable reproduction facts

- Xiangru CROWN-Reach/Flow* TORA B12: exact selected-field reproduction, T=20,
  `VERIFIED`, no Flow* termination.
- Xiangru complete-Q3 DR-RP TORA B48: T=20 and full-tube property reproduced across
  2,850 non-timing numbers within the author's `1e-6` tolerance (maximum absolute
  error `1.421e-13`).
- Xiangru DiffReach CPU U0: the failed/conservative NPZ is byte-identical; only
  66.22% of returned initial shrink flags are true.  This is a reproduced failure,
  not verification.
- Xiangru DiffReach GPU U0: `environment_failed` before step one because the V100
  cuDNN backend found no valid float64 convolution configuration.  This is not a
  native algorithm rejection.
- Stock Flow* official VDP order 4: clean source, official program, 290 segments,
  T=10 and native safe verdict.  Upstream supplies no raw reference.
- Upstream DiffReach official VDP: official README command, 64 partitions, 1,000
  steps and T=10.  Upstream supplies no raw reference, and its returned flag does
  not cover every roundoff-inclusive refinement.
- Our Torch order-4 H10 command: no lane reaches T=10; best fresh adaptive horizon
  is T=6.049038 before the declared wall cap.  It is a partial `runtime_timeout`,
  not an observed mathematical self-map rejection.

The external PyTorch Taylor-model code is no longer unidentified: the clean
Xiangru `27d29050...` complete-Q3 implementation has a field-level code audit.
Its NNCS/controller orchestration is not part of this plant-only core.

## Open correctness gates

The clean-stock Flow* scalar-affine diagnosis is complete, but the gate remains
open. A 256-bit directed MPFR oracle gives endpoint
`[0.010100670013377904, 0.1121208040160535]`; generated-stock exports
`[0.010100670333333329, 0.1121208036666667]`, a strict maximum defect of
`3.4938679727147814e-10`. Containment is first lost at
`refinement_2_accepted_tmv` in unmodified `Continuous.cpp:1013-1029`. The
official public-API route also under-encloses at its own accepted right time.
This is Outcome F, not an exporter, serializer, oracle, or configuration repair.
See [the scalar-affine closure](FLOWSTAR_SCALAR_AFFINE_CORRECTNESS_CLOSURE.md).

DiffReach `src/picard.py` returns the initial contraction flag while later
roundoff-inclusive refinement failures are not combined into that returned flag.
All-true initial flags therefore do not establish a formal full self-map.

Xiangru Q3 uses outward host rounding for controller composition, but its dynamics
interval add/mul/sin/cos path uses ordinary float64 operations.  Its fresh result
is recorded as empirical, not formal.

See [native matrix](NATIVE_REPRODUCTION_MATRIX.md), [Xiangru reproduction](XIANGRU_NATIVE_REPRODUCTION.md),
[code audit](XIANGRU_VS_OUR_TORCH_TM_CODE_AUDIT.md), and
[Flow* scalar-affine closure](FLOWSTAR_SCALAR_AFFINE_CORRECTNESS_CLOSURE.md).
