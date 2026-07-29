# Deep-study status

Last updated: 2026-07-29T09:25:20Z.

| phase | status | evidence / next gate |
|---|---|---|
| 0. Recovery | complete | `RECOVERY.md`; fetched base is exactly `9024a8a`; existing worktree recovered without reset |
| 1. Correctness delivery repair | complete | collector now derives the stock-original Van der Pol horizon from `flowstar_original_parity.csv`; regression suite: 16 passed; CSV, correctness JSON, plots, and report regenerated |
| 2. Flow* root cause | complete | audit branch now includes local SHA `2310c1a` after `fa39f7a`; standalone and integrated audits pass the stock Riccati miss plus both corrected containments, 53/256-bit comparison, original/generated T=10 schedule parity, and the adaptive endpoint-path audit; the five-commit portable format-patch series is retained in this repository |
| 3. Common intermediate representation | complete | CIR v2 schema and semantic validator cover every required field; Torch/Flow*/DiffReach adapters emit explicit unavailable markers and all three external schema/round-trip tests pass |
| 4. Fair protocols | complete | B1/B_DR/B2/B3 semantics, explicit native capability gaps, required metric annotations, and within-tool-only Pareto flags pass 23 focused tests, matched-basis smoke, and an integrated 18-plot/1,280-row smoke with no verification failures; the authoritative ten-repetition run remains a Phase 6 gate |
| 5. BERN and literature | complete | range-only clean-room feasibility prototype contains all 5 analytic cases and tightens 2 cancellation cases; the exact 16 requested PDF filenames were absent in a server-wide filename search and are recorded without invented page claims in `MATERIALS_MISSING.md`; `LITERATURE_MAP.md` now separates the two Week-4 files, Lecture 12 / Modeling Physics, Homework 2, attachment names, and public-schedule numbering |
| 6. Repository/final delivery | in progress | recursive quality auditor, native-failure-closed Pareto collector, fail-closed curator, artifact-derived conclusions, bilingual delivery generator, and complete isolated pytest matrix are implemented; 25 focused tests pass with 4 expected argument-bound skips; the authoritative ten-repetition run, final artifact generation, full repository test matrix, clean status, artifact commit, and push remain |

Checkpoint policy: update this file in every phase checkpoint.  Timestamped
scratch runs are never authoritative unless all final verification gates pass.

## Continuation recovery

The continuation recovery at `2026-07-29T08:46:49Z` is recorded in
`RECOVERY_CONTINUATION.md`.  After fetching, local and remote HEAD were both
`3bf1e25ae85b7857fdd3803adcd0c9ac9d5453d0` and the worktree was clean.  No
tmux server or live study process remained.

Run `20260729T075727Z` is preserved as **incomplete and non-authoritative**.
It stopped in `controlled protocols: Flowstar` after the first-step
order-2 Riccati candidate-remainder inclusion rejection.  It has no
`RUN_COMPLETE`, `final_acceptance.json`, `artifact_quality_audit.json`, or
`pareto_checks.json`; its partial rows must not be mixed with a future
authoritative run.

## Adaptive Flow* correctness closure

The adaptive native Van der Pol audit is complete.  The first collapsed
endpoint miss is segment 3, state 0, at `t=0.04137500000000001`: lower endpoint
`1.195701727252073` misses the DOP853 sanity value
`1.1957008958185056` by `8.314335673276219e-7`.  Stock upstream, the identical
generated harness, the variable-leaf patch, the adaptive full-Picard fallback,
and their combination all show that the failure belongs to collapsed
`compose + evaluate_time` endpoint restriction.  Direct evaluation of the same
composed native flowpipe on fixed `tau=[h,h]` has zero deterministic failures
to T=10 for all variants.

The exporter now uses the hull of the collapsed endpoint and native fixed-time
evaluation, with the hull delta explicitly added to the independent remainder.
The repaired path has zero deterministic failures and remains authoritative.
The acceptance gate now fails on any included native failure and physically
removes failed/excluded configurations from the authoritative Pareto table.
The live full audit in
`/tmp/flowstar_adaptive_full_picard_check_20260729T0930Z` passed; its machine
trace, regenerated CSV, and source-located explanation are produced by
`flowstar_adaptive_trajectory_audit.py`.

The Flow* source change extending full-Picard revalidation across overloads is
committed locally as `2310c1a`.  Its external GitHub remote rejected the push
for missing credentials, so the exact portable patch is versioned here as
`flowstar_patches/fa39f7a_series/0005-extend-full-Picard-revalidation-to-adaptive-overloads.patch`.
This does not block the study because the formal runner builds the frozen local
SHA and records it, while the full patch series makes the change reproducible.

The correctness launch gate is therefore satisfied.  The next step is the
single formal `run_all.sh` execution; no second run may be launched while its
tmux session is live.

## Resolved infrastructure incident

Correctness checkpoint `9a60e744418b59a903a9e1c6c3b44f5af05f22c9`
was committed locally, but the environment could not push it at
`2026-07-29T09:29:56Z`.  Sandbox SSH/HTTPS cannot resolve GitHub, and repeated
branch-scoped network-escalation requests failed in the approval control plane
before execution.  Exact commands and errors are retained in
`PUSH_BLOCKER.md`.  A subsequent branch-scoped escalation succeeded at
`2026-07-29T09:32:10Z`; remote HEAD was verified as `fab3141`.  No tmux server
or formal-run process existed at that verification point, so launch is now
permitted.
