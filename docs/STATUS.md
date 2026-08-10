# Status

## Latest S1 prefix-integration round

Date: 2026-08-10
Decision: `S1_PREFIX_REJECTS_BEFORE_TERMINAL`

The complete-O4 S1 contract, tensor-native additive ledger, independent
degree-four Fraction oracle, K16 accepted-boundary integration, endpoint/tube
publication, event ownership, and exact checkpoint v2 are implemented and
tested. L0 reproduces all 307 historical accepted boundaries and the terminal
rejection exactly. L1 and L2 pass every representation gate on the same first
164 boundaries, through `t=4.738198114669049`. At proposed boundary 164,
historical `h=0.03661680691961388` was accepted, while both new-plumbing lanes
first reject and can only accept an off-schedule half step. That half-step
poststate is discarded and its pre/post recorded hash is identical.

| gate | status | evidence |
|---|---|---|
| complete-O4 coupling and oracle | pass | `02_coupling_contract_oracles/oracle_results.json` |
| typed source schema | pass | `03_typed_ledger_fixtures/focused_results.json` |
| L0 historical replay | pass, 307 boundaries | `04_frozen_schedule_prefix/L0_historical_baseline/summary.json` |
| L2 conservation/publication | pass on common prefix, 164 boundaries | `prefix_conservation.csv` |
| checkpoint v2 | pass, byte exact at boundary 164 | `05_prefix_checkpoints/checkpoint_roundtrip.json` |
| terminal same-pre-state A/B | `not_run_after_stop` | `terminal_gate.json` |
| fresh horizon and second system | `not_run_after_stop` | `horizon_ladder.csv`, `second_system.csv` |

The fresh validated horizon is not defined because that experiment was not
authorized. The measured frozen-schedule S1 common-prefix time is
`4.738198114669049`; it is not a promoted fresh horizon. K32 is not authorized.

The remainder of this file records the preceding closure round.

Date: 2026-08-10
Round: structured remainder and compiled fixed-support closure
Decision: closure delivered with three explicit implemented-negative outcomes

## Gates

| gate | status | evidence |
|---|---|---|
| repository anchors | pass | remote start `05ae30b4`; isolated worktree; pinned Flow*/DiffReach SHAs |
| previous evidence closure | pass | 851-source-file recovery inventory; groups 03–08 tracked; clean-copy rebuild passes |
| claim taxonomy | pass | completion, certificate, soundness, formal/performance/ranking eligibility split |
| functional fixed core | pass | cached immutable plan; 26-tensor state; CPU/CUDA object equality matrix bit-exact |
| compiled fixed core | implemented negative | fullgraph, zero breaks, B64 T10 complete; `FIXED_SUPPORT_COMPILE_SEMANTICS_CHANGED` |
| outward fixed reference | implemented negative | exact oracle passes; multi-step fails before T1; `FIXED_SUPPORT_FORMAL_SOUNDNESS_NOT_CLOSED` |
| structured S1 | primitive pass / integration fail | K16 primitive and additive source split pass; no qualified prefix state |
| terminal/horizon/generalization | STOP / pass | `STRUCTURED_REMAINDER_LOCAL_GATE_FAILED`; no horizons; fallback generality passes |
| delivery | pass | final suite/static/package gates pass; package commit pushed; remote fresh clone verifies 221 checksums and rebuild tests |

## Scientific status

- Fixed-support RQ1: qualified at explicit float64 operator level; full stock
  driver differs slightly because several upstream builders remain float32.
- Complete-support baseline: validates through
  `6.397083942944808`, then fails y inclusion with margin
  `-1.99995911680722e-5`.
- First native Flow*/Torch split: raw candidate Picard remainder at the first
  divergent h, not transformed polynomial coefficients or roundoff.
- F1 complete carry: sound, exact, generic primitive; fails at
  `0.04345468750000001`; rejected and non-default.
- Compiled B64 T10: stable warm CPU 5.038 s and V100 6.927 s, but arithmetic
  changes; raw timing ratios are not identical-semantics speedups and CPU is
  faster than V100.
- Outward fixed reference: safeguarded primitive/reference scope only; B1
  first failure 33 and B64 first failure 90.
- Structured S1: local terminal attribution closes ordinary y but full-prefix
  conservation cannot be established from a pre-S1 checkpoint; no fresh
  horizon was authorized.
- Generality: harmonic and scalar Riccati fixed-support fallback rows complete
  100 steps on CPU/V100 B1/B64 and contain analytic endpoint hulls.

The next research operation is to integrate the delivered bounded S1 state
through every accepted complete-O4 boundary from t=0, save it in the terminal
checkpoint, and repeat the same-pre-state gate. No horizon sweep precedes that.

The verified starting baseline is `469 passed, 2 skipped in 72.48 s`. The
final suite is `515 passed, 2 skipped in 203.54 s`; both skips are the declared
optional external integration tests. `compileall` and the start-to-HEAD diff
check pass. A remote fresh clone at package commit `3f7d77a` verifies all 221
checksum entries and runs the previous/current package link, tracking, clean
copy, and deterministic table/figure/public-evidence rebuild tests (`3 passed
in 16.29 s`) without rerunning numerical reachability jobs.
