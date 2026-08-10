# Handoff: S1 prefix-integrated complete-O4 lane

Date: 2026-08-10

## Outcome

Primary S1 outcome: `S1_PREFIX_REJECTS_BEFORE_TERMINAL`.

The new S1 lane is soundly coupled, published, and checkpointable from `t=0`,
but it matches only the first 164 accepted boundaries of the 307-boundary
historical schedule. Its final common-prefix time is
`4.738198114669049`. The next historical proposed step
`h=0.03661680691961388` is rejected by both L1 and L2; L2's raw-compatible y
margin is `-3.773875528686747e-6`. The returned half-step state is discarded.

## Delivery

- Branch: `codex/s1-prefix-integrated-complete-o4-closure-20260810`
- Start SHA: `3b7b6ef97d9a33dea8498b7595131ffc6095bc1f`
- Verified package commit: `b4675700a78727573ebc941ae2be1b75483eb553`.
- Remote status at package verification: local and origin synchronized; the
  final handoff-only commit is reported in the terminal response.
- Run: `outputs/s1_prefix_integrated_complete_o4_20260810/20260810T095423Z`
- L0: 307 accepted historical boundaries, exact schedule replay.
- L1/L2: 164 accepted common-prefix boundaries.
- First full K16 / eviction: boundaries 16 / 17.
- Boundary-164 checkpoint v2 full SHA:
  `9162f267fcdcf44ca7bb9acfa73975eb8f4f4b80c03ca217aac2f07450cd585b`.
- Terminal same-pre-state gate: `not_run_after_stop`.
- Fresh horizon authorization: false.
- Fresh validated horizon: not run.
- +0.5 promotion / T10: not evaluated.
- Integrated second system: `not_run_after_stop`.
- Formal eligibility: yes for the analytic oracle, typed contract, exact
  checkpoint, and 164-boundary common-prefix enclosure only.
- Performance/ranking eligibility: false.

The implementation commits remain separated into baseline, contract, typed
ledger, complete-O4 oracle, boundary integration, checkpoint v2, frozen
runner, outward coordinate correction, causal replay, tests, packaging, and
reports. The final verification is `545 passed, 2 skipped in 225.46s`;
compileall and start-to-HEAD diff checks pass; all 164 checksum entries pass.
A clean remote fresh clone at package commit `7851abe8adb1893f49278bd96920002dfc257ca3`
verified remote/local SHA equality, tracked manifest paths, no private absolute
paths, checkpoint v2 loading, claim links, and deterministic table/figure
rebuild (`3 passed in 2.57s`). The follow-up package-verification commit changes
only `verification.json` and `SHA256SUMS`.

## One next action

Retain this negative result and return to the representation decision at the
first divergence. Because L1 and L2 reject the same frozen proposed step, do
not authorize K32, target changes, smaller `h_min`, horizon sweeps, or
multi-factor tuning. Isolate why the new typed/complete boundary plumbing
changes the prestate enough to lose that accepted step while preserving the
proved conservation obligations.
