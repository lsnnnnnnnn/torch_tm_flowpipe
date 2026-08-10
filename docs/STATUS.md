# Status

Date: 2026-08-10
Round: Flow*/DiffReach/Torch mainline realignment
Decision: complete, with `CANDIDATE_REJECTED` and explicit soundness blockers

## Gates

| gate | status | evidence |
|---|---|---|
| direction correction | pass | Xiangru audit, TORA frozen reference, three-lane research contract |
| native baselines | pass | stock Flow* T10, stock DiffReach T10, Torch complete-O4 failure boundary |
| fixed-support Torch code | pass | configurable seven-slot descriptor, routes, Picard/DR-RP/carry, CPU/CUDA |
| causal audit | pass | observation-only Flow* hook, common-basis transform, six counterfactuals |
| generic improvement | pass as negative result | complete polynomial carry implemented; formal ladder and fresh T10; rejected |
| performance/soundness | pass | actual-partition B1…512 CPU/V100, 1 cold + 5 warm, explicit classes |
| delivery | pending final audit | final pytest/checksums/push recorded in handoff after execution |

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
- GPU: no measured end-to-end or kernel speedup after synchronization and
  validation accounting; no speed or Pareto claim.

The next research operation is a bounded structured-symbol carry for the
terminal integration-overflow and polynomial-truncation terms, with
deterministic sound collapse and a frozen-prestate paired replay before any
horizon sweep.
