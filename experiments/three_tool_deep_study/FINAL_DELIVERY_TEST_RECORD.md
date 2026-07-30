# Final delivery test record

- Authoritative numerical run: `20260730T015245Z`
- Numerical producer SHA: `129b63322d7ec5e9617f54579a30ebdd6adc4c43`
- Final report-only checkpoint: `b0d20d421319a0c66bac4ef54a70e0f8fb2b52dc`
- Execution host: `huan-c4140-server-3`
- Artifact: `artifacts/authoritative/20260730T015245Z`

The numerical run's immutable `RUN_COMPLETE` records its host-visible
pre-curation complete matrix as 354 passed, 5 skipped, and 0 failed.  After the
report-only tightened-endpoint separation correction, the exact documented
artifact-bound command was rerun in the final sandbox:

```bash
cd /srv/local/shengenli/torch_tm_flowpipe_three_tool_study
DEEP_STUDY_RESULTS_DIR=/srv/local/shengenli/torch_tm_flowpipe_three_tool_study/experiments/three_tool_deep_study/artifacts/authoritative/20260730T015245Z \
  scripts/run_complete_pytest.sh
```

Final-code sandbox group totals were:

| group | passed | skipped | failed |
|---|---:|---:|---:|
| repository core | 270 | 3 | 0 |
| Torch basis | 8 | 1 | 0 |
| DiffReach projection/parity | 8 | 0 | 0 |
| historical first-order benchmark | 6 | 5 | 0 |
| historical DiffReach support | 2 | 0 | 0 |
| common-contract repair | 6 | 0 | 0 |
| three-way comparison repair | 16 | 0 | 0 |
| artifact-bound deep study | 34 | 1 | 0 |
| **total** | **350** | **10** | **0** |

- Final-code sandbox log:
  `results/20260730T015245Z/post_report_complete_pytest_canonical_sandbox.log`
- Final-code sandbox log SHA-256:
  `5fc22f7e817bbbbeecbca47721ad980ac1babc78d8792a7673fc238320df5997`

The sandbox does not expose the host's NVIDIA devices, so CUDA-specific tests
become skips.  The final code contains one additional report-protocol
regression, giving 360 total outcomes (350 pass + 10 skip), versus 359
outcomes in the immutable numerical-producer matrix.  A requested post-report
host rerun did not execute because the approval service rejected the
escalation before process creation; it is not reported as a test result.

The curated `SHA256SUMS.csv` contains 185 entries: zero files are missing and
zero hashes mismatch.  `RESULTS_MANIFEST.csv` contains 55 mappings (37 tables,
18 plots): zero mapped files are missing and zero hashes mismatch.
