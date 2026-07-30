# Reproducibility

## Frozen repositories

| repository | SHA |
| --- | --- |
| diffreach | dd628eb443b517d6415de93e7035b4baef73963e |
| flowstar_audit | 2310c1ac55357d0b48af3b37495a82a3e10ea4ff |
| flowstar_original | b85a3211748cb77b736fe4ad42ee02d8d2b81148 |
| torch | 129b63322d7ec5e9617f54579a30ebdd6adc4c43 |
| torch_repaired_base | 9024a8a29bdc0ad668a7c0620bd53872f4313cc8 |

The authoritative branch is `codex/torch-flowstar-diffreach-deep-study`.
Correctness-delivery base: `9024a8a29bdc0ad668a7c0620bd53872f4313cc8`.
The primary study uses CPU float64. CUDA availability was
`True` and the
DiffReach devices were
`['cpu:0']`. Accelerator rows
are secondary implementation-throughput observations only.

## Environment and complete tests

```bash
cd /srv/local/shengenli/torch_tm_flowpipe_three_tool_study
/srv/local/shengenli/miniforge3/condabin/conda run -n py11 \
  python -m pip install -e ".[test]"
DEEP_STUDY_RESULTS_DIR=/srv/local/shengenli/torch_tm_flowpipe_three_tool_study/experiments/three_tool_deep_study/artifacts/authoritative/20260730T015245Z scripts/run_complete_pytest.sh
```

Historical experiment directories contain repeated top-level module names,
so `run_complete_pytest.sh` uses isolated pytest processes while still
collecting every test file.

The frozen numerical-producer host run reported 354 passed, 5 skipped, and 0
failed.  The exact final-code canonical-artifact run in the publication
sandbox reported 350 passed, 10 skipped, and 0 failed because the sandbox does
not expose host CUDA/external interfaces.  Commands, group totals, and log
digests are in `FINAL_DELIVERY_TEST_RECORD.md`.

## Smoke, full run, audit, curation, and reports

```bash
experiments/three_tool_deep_study/run_smoke.sh
experiments/three_tool_deep_study/run_all.sh \
  experiments/three_tool_deep_study/results/20260730T015245Z
conda run -n py11 python \
  experiments/three_tool_deep_study/audit_results.py \
  --output-dir experiments/three_tool_deep_study/results/20260730T015245Z
conda run -n py11 python \
  experiments/three_tool_deep_study/curate_artifacts.py \
  --source experiments/three_tool_deep_study/results/20260730T015245Z
conda run -n py11 python \
  experiments/three_tool_deep_study/generate_final_delivery.py \
  --artifact-dir artifacts/authoritative/20260730T015245Z
```

The full run writes `RUN_COMPLETE` only after correctness, CIR, analytic,
parity, ten-repetition, plot, and artifact-quality gates pass. Curation refuses
an incomplete, failed, non-ten-repetition, or previously populated target.
`SHA256SUMS.csv` authenticates the curated bundle.

## Expected authoritative counts

- CSV files parsed: 36
- CSV rows parsed: 195551
- JSON files parsed: 127
- analytic checks: 17017
- CIR point checks: 4095
- native/CIR round trips:
  2010
- mandatory plots: 18
- BERN analytic cases: 5

## Interpretation requirements

Use raw endpoints for cross-tool protocol tables. Treat all sampling as
non-proof. Compare Pareto points only within a tool/system/absolute-time group.
Never reinterpret a fixed-order configuration rejection as a crash, and never
mutate Flow*'s returned remainder after `advance`.
