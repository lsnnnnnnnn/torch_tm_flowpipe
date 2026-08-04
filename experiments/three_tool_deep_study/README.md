status: historical
valid_for_commit: 9a684d9106633e067bfac0747244b769fa49aa0b
superseded_by: experiments/consolidated_study/cli.py
allowed_use: supporting implementation and provenance only

# Historical deep-study support

This directory retains implementation support needed by the canonical
three-tool runner: exporters, selected native adapters, correctness probes, and
the repaired repeated-configuration executor.

Its former orchestration, collectors, plots, reports, and committed
`20260730T015245Z` artifact were removed from the active tree. That run is
provisional because of known timer, Pareto, horizon, failure, and Flowstar
order defects. The original directory is recoverable from
`archive/pre-consolidation-20260730/codex-torch-flowstar-diffreach-deep-study`.

Do not run these modules as an alternative study pipeline. Use:

```bash
python experiments/consolidated_study/cli.py smoke
python experiments/consolidated_study/cli.py formal
```
