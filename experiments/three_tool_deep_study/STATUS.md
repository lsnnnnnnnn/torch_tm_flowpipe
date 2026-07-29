# Deep-study status

Last updated: 2026-07-29 UTC.

| phase | status | evidence / next gate |
|---|---|---|
| 0. Recovery | complete | `RECOVERY.md`; fetched base is exactly `9024a8a`; existing worktree recovered without reset |
| 1. Correctness delivery repair | complete | collector now derives the stock-original Van der Pol horizon from `flowstar_original_parity.csv`; regression suite: 16 passed; CSV, correctness JSON, plots, and report regenerated |
| 2. Flow* root cause | complete | audit branch `fa39f7a`; standalone and integrated smoke pass stock miss plus both corrected containments, 53/256-bit comparison, original/generated schedule parity, and original/repaired Van der Pol reach to T=10; exact trace/report and format-patch series committed |
| 3. Common intermediate representation | complete | CIR v2 schema and semantic validator cover every required field; Torch/Flow*/DiffReach adapters emit explicit unavailable markers and all three external schema/round-trip tests pass |
| 4. Fair protocols | complete | B1/B_DR/B2/B3 semantics, explicit native capability gaps, required metric annotations, and within-tool-only Pareto flags pass 23 focused tests, matched-basis smoke, and an integrated 18-plot/1,280-row smoke with no verification failures; the authoritative ten-repetition run remains a Phase 6 gate |
| 5. BERN and literature | complete | range-only clean-room feasibility prototype contains all 5 analytic cases and tightens 2 cancellation cases; 18 deep tests plus 15 recovered BERN CPU-reference tests pass; `BERN_FEASIBILITY.md` and `LITERATURE_MAP.md` distinguish direct plant evidence from CROWN/β-CROWN, IBP, relaxations, MILP/BaB, VNN-COMP, attacks, and course material |
| 6. Repository/final delivery | in progress | quality auditor, fail-closed curator, bilingual delivery generator, complete isolated pytest matrix, and README authority warning are implemented; 337 tests pass with 13 expected skips, and the fresh integrated smoke passes 18 plots, 1,280 raw rows, and 4,353 audited CSV rows; the authoritative ten-repetition run, final artifact generation, artifact-bound test pass, clean status, commit, and push remain |

Checkpoint policy: update this file in every phase checkpoint.  Timestamped
scratch runs are never authoritative unless all final verification gates pass.
