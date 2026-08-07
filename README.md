# torch-tm-flowpipe: clean TORA-Q3 review surface

This branch is a self-contained review surface for the native Torch TORA-Q3
plant and closed-loop implementation. It descends only from the parentless
clean-review lineage rooted at `9fc45344c4379422244b75af705dffd17304f824`.
It has no merge base with the historical audit lineage that contains
authorization-unknown controller and raw-result objects.

The historical full repository included additional VDP, Flow*, DiffReach,
benchmark-registry, and consolidated-study material. Those runners and
artifacts are not part of this clean branch. In particular, the historical VDP
failure near `t=6.397083942944808` remains unresolved; this branch makes no
claim that it was fixed.

## Supported review surface

The supported surface is:

- float64 interval arithmetic used by the TORA path, including outward
  `torch.nextafter` rounding;
- the fixed six-variable, complete total-degree Q3 dense Taylor-model kernel;
- sound sine Taylor-model composition, endpoint evaluation, affine carry,
  projection, and TORA one-step propagation;
- the native controller adapter and the TORA common-control/full-loop runners;
- portable contract, provenance, artifact-governance, and report tooling.

Other generic source modules remain because the TORA implementation imports
them or uses them as a semantic reference. This branch does not promise
compatibility with every historical generic-flowpipe workflow.

## Portable quick start

```bash
python -m pip install -e ".[test]"
pytest -q
python examples/tora_q3_one_step.py
python scripts/check_readme_surface.py
```

The example performs one CPU, float64, one-leaf TORA-Q3 plant step with a held
control interval. It requires no external controller or private trace.

## External integration contract

Controller and observation bytes are deliberately absent from Git. External
integration is enabled only through explicit environment variables:

```bash
export XIANGRU_ROOT
export TORA_CONTROLLER_PATH
export TORA_CONTROLLER_TRACE_PATH
pytest -q -m external_integration
```

`XIANGRU_ROOT` must identify the frozen Xiangru source at commit
`27d29050a5f214b56f211ca9cb411e734ed80230`.
`TORA_CONTROLLER_PATH` must identify the original controller with SHA-256
`52a50c6bc6b1b45b89319edc809cd4d3baca06c32f9fd2ddceb2e95007414418`.
If an external variable is absent, its optional test skips with an explicit
reason. If it is supplied but missing or hash-mismatched, validation fails
closed. Raw traces, controller bytes, observer patches, full environment dumps,
and server paths stay outside the public tree.

## Evidence and validation terminology

Two test records must not be conflated:

- `source_worktree_historical_validation`: `506 passed, 6 skipped`, reported
  by the earlier dirty source worktree and retained only as historical context;
- `clean_branch_portable_validation`: the test result produced directly from
  this clean branch. The bootstrap result was `52 passed, 14 skipped`; current
  results are regenerated as the review surface evolves.

The common-control T20 workload is 20 matched, period-local plant replays. Each
controller period restarts from the Xiangru-observed pre-controller state box
and held-control interval. It is not an independent Torch closed loop. The
native Torch full loop previously certified through `T=4.3` and failed closed
at segment 44 (`T=4.4`) when leaf 0's `x3` tube crossed the fixed `[-2, 2]`
safety property. No T4.3 endpoint is compared as a T5/T10/T20 final width.

## Review documents

- [Clean lineage publication note](CLEAN_REVIEW_PUBLICATION.md)
- [Native TORA-Q3 implementation report](TORA_Q3_NATIVE_TORCH_IMPLEMENTATION_REPORT.md)
- [Common-control comparison report](TORA_Q3_COMMON_CONTROL_COMPARISON_REPORT.md)
- [Native full-loop comparison report](TORA_Q3_FULL_CLOSED_LOOP_COMPARISON_REPORT.md)
- [Historical runtime report](TORA_Q3_RUNTIME_REPORT.md)
- [Sine Taylor-model soundness report](SINE_TM_SOUNDNESS_REPORT.md)
- [Public artifact governance audit](PUBLIC_ARTIFACT_GOVERNANCE_AUDIT.md)
- [GPU bottleneck and source-stage attribution](TORA_Q3_GPU_BOTTLENECK_REPORT.md)
- [Optimized matched-stack runtime report](TORA_Q3_OPTIMIZED_RUNTIME_REPORT.md)
- [T4.4 lifecycle width attribution](TORA_Q3_T4_4_WIDTH_ATTRIBUTION_REPORT.md)
- [Performance/full-loop closure](TORA_Q3_CLOSED_LOOP_CLOSURE_REPORT.md)
- [Current handoff](handoff.md)

The historical reports describe the reviewed source-worktree experiments. A
result becomes current evidence for this performance/closed-loop closure branch
only when its sanitized aggregate appears under
`outputs/tora_q3_perf_closure_20260806/` and is covered by that directory's
`manifest.sha256`. The older manifest location under
`outputs/tora_q3_native_matched_20260806/` is retained for compatibility and
must contain the identical complete-tree view.
