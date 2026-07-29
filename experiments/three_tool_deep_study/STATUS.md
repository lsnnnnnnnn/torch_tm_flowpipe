# Deep-study status

Last updated: 2026-07-29 UTC.

| phase | status | evidence / next gate |
|---|---|---|
| 0. Recovery | complete | `RECOVERY.md`; fetched base is exactly `9024a8a`; existing worktree recovered without reset |
| 1. Correctness delivery repair | complete | collector now derives the stock-original Van der Pol horizon from `flowstar_original_parity.csv`; regression suite: 16 passed; CSV, correctness JSON, plots, and report regenerated |
| 2. Flow* root cause | complete | audit branch `fa39f7a`; standalone and integrated smoke pass stock miss plus both corrected containments, 53/256-bit comparison, original/generated schedule parity, and original/repaired Van der Pol reach to T=10; exact trace/report and format-patch series committed |
| 3. Common intermediate representation | complete | CIR v2 schema and semantic validator cover every required field; Torch/Flow*/DiffReach adapters emit explicit unavailable markers and all three external schema/round-trip tests pass |
| 4. Fair protocols | implementation present; complete run pending | interrupted runs are diagnostic only; one clean end-to-end run and portable artifact curation remain |
| 5. BERN and literature | pending final consolidation | local BERN feasibility branch exists at `dd82032`; required literature map is not yet present |
| 6. Repository/final delivery | in progress | required bilingual reports, artifact index, reproducibility guide, results manifest, README update, full tests, clean status, commits, and pushes remain |

Checkpoint policy: update this file in every phase checkpoint.  Timestamped
scratch runs are never authoritative unless all final verification gates pass.
