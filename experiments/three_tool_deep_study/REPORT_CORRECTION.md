# Post-run report correction record

- Authoritative numerical run: `20260730T015245Z`
- Numerical producer SHA:
  `129b63322d7ec5e9617f54579a30ebdd6adc4c43`
- Report-only checkpoint:
  `b0d20d421319a0c66bac4ef54a70e0f8fb2b52dc`
- Superseded presentation preserved at:
  `results/20260730T015245Z_report_v1_superseded`

Independent review after the formal run found no numerical, acceptance, or
Pareto-partition defect.  It found a presentation-policy defect: the detailed
native-low-order table and plot 07 displayed six supplemental Torch
`order1_legacy_tightened` endpoints in a broad view containing other tools'
raw/native endpoints.  Those rows were already absent from Pareto eligibility
and rankings, but the shared view violated the stricter rule that tightened
Torch endpoints must never be presented as cross-tool raw comparisons.

The correction:

1. filters supplemental tightened endpoints from plot 07;
2. moves them to a separately labelled Torch-internal diagnostic table;
3. marks plot 07's protocol mapping as raw/primary native with tightened Torch
   excluded; and
4. replaces a stale background-launch snippet with the exact `run_all.sh`
   reproduction command.

No numerical CSV, correctness result, acceptance decision, repetition timing,
or frozen producer SHA changed.  The corrected scratch output passed the
recursive audit over 36 CSV files / 195,551 rows / 127 JSON files, with no
non-finite or horizon/step violations.  The superseded presentation was moved
to the ignored sibling above rather than deleted or overwritten.
