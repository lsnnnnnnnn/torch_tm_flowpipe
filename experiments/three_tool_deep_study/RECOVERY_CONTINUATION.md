# Continuation recovery record

Recovered at `2026-07-29T08:46:49Z` with the existing worktree and branch:

- worktree:
  `/srv/local/shengenli/torch_tm_flowpipe_three_tool_study`;
- branch: `codex/torch-flowstar-diffreach-deep-study`;
- local HEAD:
  `3bf1e25ae85b7857fdd3803adcd0c9ac9d5453d0`;
- fetched remote HEAD:
  `3bf1e25ae85b7857fdd3803adcd0c9ac9d5453d0`;
- worktree status before this record: clean.

`git fetch --all --prune` completed successfully.  The existing branch and
worktree were retained; no reset, replacement branch, force push, `main`
change, result deletion, or result overwrite was performed.

## Run recovery decision

The newest run directory is
`experiments/three_tool_deep_study/results/20260729T075727Z`.  Its frozen
configuration is the checked-in `benchmark_spec.yaml` with SHA-256
`6386020d0da79f58239d87787f0c77aa391f25f06ec5b9c04ca56dcce76f6ce4`;
the launcher pipeline has SHA-256
`ebe0ceea6e9ddb3d1eab67072b11b7024bf3738b3e89ac9a040dcd050617e1e6`.

The last completed progress entries were:

1. Phase-0 provenance and initial frozen checksums;
2. acceptance tests;
3. Flow* correctness matrix and original parity;
4. controlled Torch;
5. controlled DiffReach.

The run entered `controlled protocols: Flowstar` at
`2026-07-29T08:20:18Z` and then stopped.  `run_all.log` records
`run_controlled.py:556 -> export_flowstar_segment.py:541`: the primary
`flowstar_root_cause_patch` rejected the first order-2 Riccati step at
`h=0.05`.  The configured candidate remainder was
`[-1.0e-4, 1.0e-4]`; the first Picard image remainder upper bound was
`1.2913942109863294e-4`, leaving upper inclusion margin
`-2.913942109863294e-5`.  This is a configuration rejection, not evidence of
a global Flow* failure.

No tmux server was present at
`/tmp/tm_three_tool_deep_study.sock`, and no live `run_all.sh`,
`run_controlled.py`, or deep-study process existed.  The run lacks all four
completion/quality markers:

- `RUN_COMPLETE`;
- `final_acceptance.json`;
- `artifact_quality_audit.json`;
- `pareto_checks.json`.

Therefore `20260729T075727Z` is classified as **incomplete,
non-authoritative diagnostic evidence**.  It remains untouched and must not
be resumed into, relabelled as final, or mixed with a new authoritative run.

## Required next gate

Before another formal run starts, the adaptive native Flow* Van der Pol
trajectory failure must be traced from its first divergent endpoint through
native flowpipe evaluation and CIR export.  The acceptance gate must reject
any included native configuration with nonzero trajectory failures.  The
configuration may enter authoritative Pareto/ranking/headline artifacts only
after a zero-failure regression, otherwise it must be explicitly excluded.

## Preserved formal-run failure `20260729T093319Z`

The next formal attempt was launched only after correctness checkpoint
`7868274020d5a0f1209622eba6d907303fad7687` was pushed.  It completed:

1. all isolated launch test groups;
2. the 96-row Flow* correctness matrix, root-cause evidence, original schedule
   parity, and adaptive endpoint-path audit;
3. controlled Torch, DiffReach, and Flow*;
4. native Torch, DiffReach, and Flow*;
5. matched-basis and component ablations;
6. common defect diagnostics; and
7. the five-case BERN feasibility gate.

It entered `ten-repetition native practical timing: Torch` at
`2026-07-29T10:34:59Z` and exited at `2026-07-29T11:39:39Z`.
`run_pareto.py:271` called the affine-only `affine_reset` helper directly on
the nonlinear order-4 CUDA endpoint.  The helper correctly rejected the input
with `ValueError: affine reset received a nonlinear polynomial`.

This was a benchmark protocol implementation error, not a solver validation
failure.  The CPU path already projected the endpoint to B1 before reset; the
secondary CUDA path did not.  The fix routes both paths through one
`_projected_affine_box_reset` helper and records the discarded-term count.
A control-flow regression exercises the helper, and a CUDA functional
regression executes an actual coupled-quadratic step when a CUDA device is
visible.

The run has no `RUN_COMPLETE`, `final_acceptance.json`,
`artifact_quality_audit.json`, or `pareto_checks.json`.  It is marked by its
ignored `results/20260729T093319Z/INCOMPLETE` record and remains
**non-authoritative**.  Its partial data must not be reused by the replacement
formal run.

## Preserved quality-gated run `20260729T162851Z`

Replacement run `20260729T162851Z` used pushed SHA
`a781c705404c30a2f895a985c8e49d57ba727ae5`.  It recomputed every experimental
stage, including 240 native-practical repetition observations, passed the
primary correctness and ten-repetition verification gates, and generated all
18 mandatory figures.  `final_acceptance.json` passed.

The recursive artifact-quality audit then rejected 36 CSV cells.  They are
three width fields in four failed `candidate=1e-6` Flow* ablation rows, repeated
through `flowstar_component_ablation.csv`, `component_ablation.csv`, and
`raw_results.csv`.  The configurations correctly reported
`first_picard_inclusion_failed`, but their absent endpoint, polynomial, and
remainder widths were serialized as `nan` rather than the required explicit
`unavailable` marker.

The producer now computes a finite maximum when data exist and otherwise emits
`unavailable`; it never substitutes zero.  A full 12-row Flow* ablation
reproduction in `/tmp/flowstar_ablation_quality_fix.55P9l0` contains no
`nan`/`inf`, and the first four rejected rows carry `unavailable` in all three
width fields.  Unit coverage checks empty, non-finite, mixed, and finite inputs.

The run has no `RUN_COMPLETE` and was never curated.  It is recorded by
`results/20260729T162851Z/INCOMPLETE` and remains **non-authoritative**.  Its
experimental rows will not be mixed with the next clean formal run.

## Disposable downstream rehearsal after the quality fix

To reduce the risk of discovering another report-only defect after a
multi-hour recomputation, the downstream-only transformations were exercised
on a disposable copy at
`/tmp/three_tool_downstream_validation.RIulsy`.  This was a validation fixture,
not a recovery or an authoritative run.

The rehearsal exposed and closed four provenance defects:

1. absent Flow* process RSS was being coerced to `0.0`; it is now the literal
   `unavailable`, while positive measured Torch/DiffReach peaks remain numeric;
2. six Torch `order1_legacy_tightened` rows could enter the primary within-tool
   Pareto table; they are now supplemental-only explicit exclusions and cannot
   support cross-tool raw-endpoint claims;
3. provisional report generation wrote `FINAL_CONCLUSIONS.md` into the source
   tree before the quality gate; it now writes only inside the timestamped run,
   and repository-level conclusions are produced only after curation; and
4. rerunning table collection on a filtered result could erase the exclusion
   partition and leave a headerless CSV; collection now merges and deduplicates
   both partitions and always writes a schema header.

After those repairs, collection was run twice.  Both passes retained 103
eligible and 18 excluded Pareto rows, with zero tightened rows eligible, six
tightened rows explicitly excluded, Flow* memory equal to `unavailable`, and
zero memory placeholders absent.  Plotting produced all 18 figures.  The
recursive audit passed 36 CSV files / 195,551 rows with no non-finite cells and
no horizon/step violations.  Curation selected 185 files (110,274,524 bytes),
and final-delivery generation produced both reports, conclusions, artifact
index, reproducibility record, and a 55-row manifest.  The focused study suite
passes 29 tests with five environment skips.

These checks validate code paths only.  The next authoritative run must still
start from a new empty timestamped directory and recompute every numerical
stage.
