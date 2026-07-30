# Deep-study status

Last updated: 2026-07-30T01:31:58Z.

| phase | status | evidence / next gate |
|---|---|---|
| 0. Recovery | complete | `RECOVERY.md`; fetched base is exactly `9024a8a`; existing worktree recovered without reset |
| 1. Correctness delivery repair | complete | collector now derives the stock-original Van der Pol horizon from `flowstar_original_parity.csv`; regression suite: 16 passed; CSV, correctness JSON, plots, and report regenerated |
| 2. Flow* root cause | complete | audit branch now includes local SHA `2310c1a` after `fa39f7a`; standalone and integrated audits pass the stock Riccati miss plus both corrected containments, 53/256-bit comparison, original/generated T=10 schedule parity, and the adaptive endpoint-path audit; the five-commit portable format-patch series is retained in this repository |
| 3. Common intermediate representation | complete | CIR v2 schema and semantic validator cover every required field; Torch/Flow*/DiffReach adapters emit explicit unavailable markers and all three external schema/round-trip tests pass |
| 4. Fair protocols | complete | B1/B_DR/B2/B3 semantics, explicit native capability gaps, required metric annotations, and within-tool-only Pareto flags pass 23 focused tests, matched-basis smoke, and an integrated 18-plot/1,280-row smoke with no verification failures; the authoritative ten-repetition run remains a Phase 6 gate |
| 5. BERN and literature | complete | range-only clean-room feasibility prototype contains all 5 analytic cases and tightens 2 cancellation cases; the exact 16 requested PDF filenames were absent in a server-wide filename search and are recorded without invented page claims in `MATERIALS_MISSING.md`; `LITERATURE_MAP.md` now separates the two Week-4 files, Lecture 12 / Modeling Physics, Homework 2, attachment names, and public-schedule numbering |
| 6. Repository/final delivery | in progress | formal run `20260730T015245Z` from pushed SHA `129b633` recomputed every stage, passed acceptance and recursive quality audit over 195,551 CSV rows, completed 240 repetition observations, generated 18 plots, passed the complete matrix with 354 passed / 5 skipped / 0 failed, wrote `RUN_COMPLETE`, and curated the artifact; independent review found one presentation-only separation defect in the broad native-low-order report table/plot (supplemental tightened Torch rows shared the view with other raw/native rows despite no ranking); the producer now filters plot 07 and separates the tightened rows into a Torch-internal table, with 14 focused passes / 1 environment skip; regenerate/audit/recurate, run the final-code complete matrix, commit artifacts, push, and verify clean remote HEAD remain |

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

## Preserved formal attempt `20260729T093319Z`

The pushed formal attempt completed correctness, all controlled/native tool
runs, ablations, defect diagnostics, and BERN.  It stopped in the Torch
ten-repetition stage because the secondary CUDA timing path passed a nonlinear
order-4 endpoint directly to the affine-only reset helper.  The CPU path
already performed the required B1 projection.

The run is explicitly **incomplete and non-authoritative**, has no completion
or acceptance markers, and is recorded in `RECOVERY_CONTINUATION.md` plus its
ignored `INCOMPLETE` marker.  The repair now makes CPU and CUDA call the same
project-then-reset helper and records discarded terms.  No prior numerical
rows will be mixed into the replacement run.

## Preserved quality-gated attempt `20260729T162851Z`

The replacement attempt fixed the CUDA reset path and passed all numerical,
correctness, Pareto repetition, plot, and final-acceptance gates.  The recursive
quality audit then found 36 repeated `nan` cells originating from three absent
width fields on four explicitly rejected Flow* ablation configurations.
Those rows now use `unavailable`, never zero.  A full standalone reproduction
of all 12 ablation rows contains no non-finite CSV values.

Because the run stopped before `RUN_COMPLETE` and curation, it is preserved as
incomplete/non-authoritative.  Exact cell provenance and the clean-run rule are
recorded in `RECOVERY_CONTINUATION.md`.

The repaired downstream path was rehearsed against a disposable copy of the
complete numerical output.  Repeated table collection is now idempotent:
103 primary Pareto rows and 18 exclusions remain stable, all six
`order1_legacy_tightened` configurations are excluded with reason
`supplemental_tightened_endpoint_not_raw_comparable`, and no tightened row
appears in the primary table.  Missing Flow* RSS is serialized as
`unavailable`, never `0`; measured Torch/DiffReach peaks remain numeric.
Report generation writes its provisional conclusions only inside the run
directory, so an incomplete run cannot overwrite the repository-level final
conclusion.  The final frontier table now includes step size, basis, and
carry/preconditioning columns.

The rehearsal generated all 18 plots, passed the recursive audit over 36 CSV
files / 195,551 rows with no non-finite cells or horizon/step violations,
curated 185 files, and generated the bilingual delivery plus a 55-row manifest.
This validates downstream transformations only; it does not make
`20260729T162851Z` authoritative or permit reuse of its numerical rows.

## Completed formal computation `20260730T015245Z`

The clean formal run used pushed Torch SHA
`129b63322d7ec5e9617f54579a30ebdd6adc4c43`, Flow* audit SHA
`2310c1ac55357d0b48af3b37495a82a3e10ea4ff`, original Flow* SHA
`b85a3211748cb77b736fe4ad42ee02d8d2b81148`, and DiffReach SHA
`dd628eb443b517d6415de93e7035b4baef73963e`.  It completed every producer:
15,969 controlled rows, 47,844 native rows, 828 component-ablation rows,
240 repetition observations, 18 plots, and all correctness/defect/BERN gates.
The complete isolated repository matrix reported 354 passed, 5 skipped, and
0 failed tests.  `RUN_COMPLETE`, final acceptance, artifact quality, frozen
input integrity, and all SHA-256 records passed.

The primary Pareto partition contains 103 rows; 18 rows are excluded.  All six
`order1_legacy_tightened` rows are in the exclusion partition with reason
`supplemental_tightened_endpoint_not_raw_comparable`, and none is eligible.
All 24 selected timing configurations have ten repetitions.  Missing Flow*
memory is explicit `unavailable`; no unavailable memory is encoded as zero.

Independent post-run review found no numerical or acceptance defect, but it
did find that plot 07 and the detailed native-low-order report table still
displayed the six supplemental tightened Torch endpoints in the same broad
view as other tools' raw/native endpoints.  No ranking used those rows, but the
shared presentation was stricter than the stated comparison policy permits.
The report generator now puts them in a separate Torch-internal diagnostic
table, plot 07 filters them, and its protocol mapping says so explicitly.  The
stale background-launch reproduction snippet is also replaced by the exact
`run_all.sh` invocation.  These are report-only changes; no numerical CSV,
acceptance decision, or repetition result is altered.
