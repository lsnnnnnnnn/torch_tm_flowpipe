# Handoff: VDP cross-step carry-lineage audit

## State

- Branch: `codex/vdp-cross-step-carry-lineage-audit-20260806`.
- Starting Torch HEAD: `455146df23940caa6f168877ffe6ec6f508c43a4`.
- Formal H1 source: `a1fb3527bb7c12ce23aa2fb49d66f6380c463c90`.
- H1 packaging/report source: `2e4507220a631a21dbe5227a7f9a5201948aedde`.
- Stock Flowstar: `b85a3211748cb77b736fe4ad42ee02d8d2b81148`.
- Verdict: **Case C**. No Torch numerical fix and no new representation implemented.
- Original Torch and Flowstar dirty worktrees were left untouched; all work was done in isolated worktrees.

## Reproduced anchors

- 14/14 actual frozen H1 lanes reproduce exactly.
- Frozen checkpoint SHA256: `dcb8f646d45c9742e0cff23fea596c12e53d8ccd00d1544f70564a44a7463420`.
- Frozen point: `t=6.397083942944808`, `h=0.003623635847674574`.
- D3 margin: `-1.5859969428028492e-5`.
- Candidate/support SHA256: `bc1433d...` / `d0aa354b...`.
- Fresh T=6.5 trace: expected exit 1 at the same frozen point after 307 accepted steps and 48 rejected attempts; terminal margins are `9.963763341523255e-5` (x) and `-1.99995911680722e-5` (y).
- The fresh full checkpoint hash is `54185608...` because provenance/config packaging differs, while current-state, `tmv_pre`, and `tmv_right` TMVector hashes exactly match the frozen `dcb8f646...` artifact.

## New findings

- First native schedule divergence: accepted step 12, `t=0.18187433604506256`, candidate `h=0.019615177354506262`; Torch accepts, stock Flowstar halves and accepts `0.009807588677253131`.
- Last common accepted step: 11.
- The authoritative contracts are only partially matched: proactive Torch truncation subdivision, local basis/state shape, interval backend, normalized insertion, tube storage, and the symbolic queue differ.
- Flowstar stock/instrumented equivalence passes: 290 segments, exit 0, identical printed schedules, and byte-identical plot files.
- Call-44 stable hash: `b7613d231be82bea5afc477540342cfa04ee17a6448d7bfb5921dc70ae341a9d`.
- Call 44 is raw-remainder y `-x²y`; 1,141/1,141 discarded routes are classified, 1,510 DAG nodes have no parent gaps, and the selected interval reconstructs exactly.
- First trajectory-wide Torch cross-step interval enclosure is after accepted step 0. The immediate terminal ancestry crosses boundary 307, then call 44 intervalizes its discarded degree-overflow polynomial.

## Why Case C

No local implementation bug is visible, but the audit cannot isolate symbolic carry as the causal cross-tool difference. The first native schedule split is confounded by an intentional range-policy mismatch, coefficient comparison fails closed on basis/center/scale identity, and the observation-only stock hook cannot see the rejected Picard candidate/image, insertion discarded terms, or J/Phi_L parents.

## One next action

Instrument the exact stock symbolic `Flowpipe::advance_adaptive_stepsize` overload around `insert_ctrunc_normal`, `Picard_ctrunc_normal`, and the subset test. Limit the experiment to the last common state/first divergent h, export J/Phi_L lineage and pre/post polynomials, and implement/test a common-basis transform before comparing coefficients. Do not implement delayed carry/noise symbols yet.

## Commands

Focused tests:

```bash
conda run -n py11 pytest -q tests/test_vdp_cross_step_carry_lineage_audit.py
```

Final gate:

```bash
conda run -n py11 pytest -q
git diff --check
git status --short --branch
```

The exact captured commands, raw streams, exit codes, environment, and final hash manifest are under `outputs/vdp_cross_step_carry_lineage_20260806/`.

Final verification: focused audit tests `8 passed in 7.48s`; full suite `441 passed, 2 skipped in 63.34s`. Both captures record source commit `7a40428bfdf018a0995daf5f9777afc9c807fe88` with an empty tracked diff.
