# Deep-study status

Last updated: 2026-07-29 UTC.

| phase | status | evidence / next gate |
|---|---|---|
| 0. Recovery | complete | `RECOVERY.md`; fetched base is exactly `9024a8a`; existing worktree recovered without reset |
| 1. Correctness delivery repair | complete | collector now derives the stock-original Van der Pol horizon from `flowstar_original_parity.csv`; regression suite: 16 passed; CSV, correctness JSON, plots, and report regenerated |
| 2. Flow* root cause | implementation complete; integrated smoke pending | audit branch `fa39f7a`; standalone regression passes stock miss plus both corrected containments; 53/256-bit evidence generator passes; exact trace/report and authoritative format-patch series committed |
| 3. Common intermediate representation | implementation present; final schema audit pending | exporters and schema tests exist under this directory |
| 4. Fair protocols | implementation present; complete run pending | interrupted runs are diagnostic only; one clean end-to-end run and portable artifact curation remain |
| 5. BERN and literature | pending final consolidation | local BERN feasibility branch exists at `dd82032`; required literature map is not yet present |
| 6. Repository/final delivery | in progress | required bilingual reports, artifact index, reproducibility guide, results manifest, README update, full tests, clean status, commits, and pushes remain |

Checkpoint policy: update this file in every phase checkpoint.  Timestamped
scratch runs are never authoritative unless all final verification gates pass.
