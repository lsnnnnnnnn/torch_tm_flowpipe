# Final acceptance

> **SUPERSEDED STATUS REPORT.** `superseded_by`:
> [`docs/NATIVE_REPRODUCTION_MATRIX.md`](../../../docs/NATIVE_REPRODUCTION_MATRIX.md),
> [`docs/RESULTS_STATUS.md`](../../../docs/RESULTS_STATUS.md), and
> [`docs/XIANGRU_VS_OUR_TORCH_TM_CODE_AUDIT.md`](../../../docs/XIANGRU_VS_OUR_TORCH_TM_CODE_AUDIT.md).
> The cleanup-era statement that the external PyTorch implementation was
> unidentified is no longer current. This acceptance narrative remains provenance
> only.

Status: `passed_ready_for_external_pytorch_audit_repository_contract_only`.

## Repository and history

- Cleanup base: `08b6f2416122cbf4220ff351e663caa1a0af13a2`.
- Cleanup branch: `codex/repository-cleanup-before-external-torch-audit`.
- Every local/remote ref, worktree, and existing archive tag was inventoried.
- No branch/tag was deleted, `main` was not changed, and no external
  repository was modified.
- The original dirty worktree remains at SHA
  `26a254ef585a9dee394b7e41922c06bf8799f501`, with unchanged patch SHA-256
  `d4279db38fa8d026b39b5397974bbe999808201bf862eb08db2bafdce3b0ed77`.

## Results and artifacts

- Frozen formal artifact tree hash stayed
  `26a576298fa01a8db493521ff5dc19720135e698`.
- Run `20260730T153654Z` is retained but `withdrawn_do_not_cite` because its
  Flowstar rows used a patched audit backend and behavior environment.
- No current winner, Pareto, or runtime/tightness claim is citable.
- Artifact and claim registries distinguish supported, provisional,
  historical, unknown, and withdrawn evidence without editing source files.

## Backend and protocol

- Primary Flowstar startup canonicalizes and records root/SHA/dirty/library
  state; rejects audit roots, audit behavior variables, and unknown tracked
  changes; and labels the observed backend `stock-plus-gcc15-compat`.
- Official-program, generated-stock, and patched-audit routes remain distinct.
- Endpoint/segment/tube and raw/tightened dimensions are explicit and tested.
- Requested/effective degree, basis, remainder policy, step policy, requested
  horizon, successful horizon, completion, and failure fields are explicit.
- All eight formal cross-tool gates remain pending and the formal runner is
  fail-closed before output creation.

## Order-2 diagnostic

The permitted single-step smoke compiled and returned normally. It attempted
`[0.1, 0.05, 0.025, 0.0125, 0.00625, 0.003125]`, accepted zero segments, and
recorded y as the failing dimension with self-map defect
`0.00017793541151873635`. The outcome is
`validation_rejected / remainder_self_map_failed`; it is not a crash,
unsupported-order claim, completed horizon, or cross-tool comparison. The
temporary 11.5 MB probe executable was removed after the trace and process
records were saved; it is reproducible from the recorded compile command.

## Verification

- Baseline on the initially supplied feature worktree: `274 passed`.
- Clean selected-base preselection: `213 passed`.
- Final cleanup branch: `228 passed`.
- Final marker suites: `59 unit`, `14 integration`, `1 flowstar`, and
  `1 diffreach` passed; no final skip was hidden as a pass.
- The final branch adds 15 passing tests relative to its selected clean base.
  The 274-test initial baseline is a different feature lineage, so its raw
  count is recorded but not treated as an equivalent-suite regression delta.
- Editable install, supported-script compileall, README example, both CLI help
  paths, invalid-backend rejection before output creation, non-empty output
  rejection with sentinel preservation, structured-file parsing, link/path
  scans, frozen-tree comparison, and `git diff --check` passed.
- A preliminary verification command guessed a nonexistent direct `py11`
  virtualenv path and did not run tests; all recorded verification uses the
  actual `conda run -n py11` environment.

## External PyTorch audit

The repository and comparison contract are prepared for a later identity
audit, but the external implementation itself remains unidentified. Paper,
authors/thesis context, URL/DOI, repository, license, exact revision,
environment, claims, and benchmark source must be confirmed before any clone
or code comparison. No external implementation was guessed, cloned, vendored,
or compared in this cleanup.
