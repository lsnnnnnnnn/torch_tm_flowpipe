status: historical
valid_for_commit: unknown
superseded_by: docs/EXPERIMENT_PROTOCOL.md
allowed_use: provenance only

# Three-way comparison correctness repair

This directory supersedes the preliminary `three_way_common_contract`
comparison. It separates whole-segment tubes, raw endpoint substitution, and
Torch's supplemental endpoint tightening; it also keeps Flow*'s native returned
remainder separate from the historical candidate-reinjection mutation.

The final decision is data-driven:

- Outcome A permits a corrected three-way comparison only if every stock
  solver passes its applicable correctness gates.
- Outcome B permits a corrected Torch-versus-DiffReach comparison while
  retaining Flow* as audit/sanity evidence.
- Outcome C publishes only the audit.

The current evidence selects Outcome B because stock Flow* misses the analytic
Riccati endpoint after its native remainder-only refinement. The original Flow*
Van der Pol benchmark still reaches `T=10`, so this is not a claim that Flow*
is generally nonfunctional.

Run the interactive smoke suite with:

```bash
./experiments/three_way_comparison_repair/run_smoke.sh
```

Launch the complete run in the required tmux session with:

```bash
./experiments/three_way_comparison_repair/launch_background.sh
```

`run_all.sh [OUTPUT_DIRECTORY]` is the foreground equivalent. Each timestamped
result contains the exact benchmark specification, repository/environment
capture, checksums of the frozen historical result before and after execution,
raw rows, summaries, plots, logs, and the generated technical report.

The frozen directory
`experiments/three_way_common_contract/results/20260724T132534Z` is read-only
historical evidence. Do not overwrite it or cite it for winner claims.
